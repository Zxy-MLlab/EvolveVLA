#!/usr/bin/env python3
"""Optional task-aware launcher; no existing harness default is changed.

Only explicit task IDs place_mouse_pad/hanging_mug may select a verified
candidate. Other/unknown IDs and pending target routes always select the fixed
baseline with the original controller, before any policy import or loading.
One process runs one task and one accepted episode, loading one model only.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
BASELINE = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
BASELINE_ID = '4cfdd56a75f89160c1e913101bb243e4aa817b69cdb41c72ff7d7ee6384ed0e3'
TARGETS = frozenset(('place_mouse_pad', 'hanging_mug'))
ROBOTWIN = ROOT / 'robotwin_repo_ba0fd21'
OPENPI = ROBOTWIN / 'XPolicyLab/policy/Pi_05/openpi'
HARNESS = SCRIPT.with_name('robotwin_public_openpi_harness.py')
CONTROLLER = SCRIPT.with_name('gen1_duration_controller_20261020.py')
INFERENCE_PYTHON = Path('/root/autodl-tmp/conda_envs/scratch_orbax/bin/python')
SIM_PYTHON = Path('/root/autodl-tmp/conda_envs/robotwin_cu128/bin/python')
RECEIPT_ENV = 'EVOLVEVLA_ROUTING_RECEIPT'
RECEIPT_SHA_ENV = 'EVOLVEVLA_ROUTING_RECEIPT_SHA256'
MODES = ('original', 'fixed15', 'opening15')


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def required_sources():
    return [SCRIPT, HARNESS, CONTROLLER,
        SCRIPT.with_name('robotwin_eval_harness.py'),
        SCRIPT.with_name('robotwin_obs_bridge.py'),
        SCRIPT.with_name('pi05_openpi_adapter.py'),
        ROBOTWIN / 'envs/_base_task.py', ROBOTWIN / 'envs/robot/robot.py',
        ROBOTWIN / 'scripts/eval_policy_xpolicylab.py']


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_file_map(files):
    for name, expected in files.items():
        if digest(name) != expected:
            raise ValueError(f'File hash mismatch: {name}')


def verify_checkpoint(spec):
    """Check the selected model against a frozen full-file identity reference."""
    reference = Path(spec['identity_reference']).resolve()
    if digest(reference) != spec['identity_reference_sha256']:
        raise ValueError('Checkpoint identity reference changed')
    identity = read_json(reference)['checkpoint']
    checkpoint = Path(spec['checkpoint']).resolve()
    expected = identity['files_sha256']
    if (Path(identity['path']).resolve() != checkpoint
            or identity['identity_sha256'] != spec['identity_sha256']
            or hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest() != spec['identity_sha256']):
        raise ValueError('Checkpoint path or identity mismatch')
    actual_names = {str(p.relative_to(checkpoint))
        for sub in ('params', 'assets') for p in (checkpoint/sub).rglob('*') if p.is_file()}
    extras = {'model_config.json', 'verification.json'} if (checkpoint/'model_config.json').exists() else {'SHA256SUMS'}
    if actual_names | extras != set(expected) or not (checkpoint/'params/_METADATA').is_file():
        raise ValueError('Checkpoint inventory mismatch')
    for relative, expected_sha in expected.items():
        path = checkpoint/relative
        if path.is_symlink() or checkpoint not in path.resolve().parents or digest(path) != expected_sha:
            raise ValueError(f'Checkpoint data mismatch: {relative}')
    if 'model_config.json' in extras:
        model, verified = read_json(checkpoint/'model_config.json'), read_json(checkpoint/'verification.json')
        if (model.get('action_horizon') != 50 or model.get('action_dim') != 32
                or verified.get('status') != 'passed'
                or verified.get('kind') != 'full_lora_model_unmerged'
                or verified.get('production_restore_exact') is not True
                or verified.get('normalization_exact') is not True):
            raise ValueError('Candidate restore/config verification failed')
    return identity


def select_route(config, task_id):
    """Pure selection. No candidate paths or weights are read for fallback."""
    if not isinstance(task_id, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', task_id):
        raise ValueError('Explicit canonical task ID required')
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported routing schema')
    baseline = config['baseline']
    if (Path(baseline['checkpoint']).resolve() != BASELINE.resolve()
            or baseline['identity_sha256'] != BASELINE_ID
            or baseline['controller_mode'] != 'original'):
        raise ValueError('Fallback must be the pinned baseline and original controller')
    routes = config.get('target_routes', {})
    if set(routes) - TARGETS:
        raise ValueError('Candidate routes outside the two target tasks are forbidden')
    selected, role, reason = baseline, 'baseline', 'non_target_or_unknown_task'
    if task_id in TARGETS:
        route = routes.get(task_id, {'status': 'pending'})
        status = route.get('status')
        if status not in ('pending', 'verified', 'disabled'):
            raise ValueError('Invalid target route status')
        reason = 'target_route_' + status
        if status == 'verified':
            selected, role = route['candidate'], 'candidate'
            if selected.get('controller_mode') not in MODES:
                raise ValueError('Unsupported candidate controller')
    return dict(task_id=task_id, role=role, reason=reason,
                controller_mode=selected['controller_mode'], model=selected)


def verify_deployment_evidence(route):
    if route['role'] != 'candidate':
        return None
    spec = route['model']
    evidence = Path(spec['deployment_evidence']).resolve()
    if digest(evidence) != spec['deployment_evidence_sha256']:
        raise ValueError('Deployment evidence changed')
    decision = read_json(evidence)
    expected = dict(status='passed', task_id=route['task_id'],
        checkpoint_identity_sha256=spec['identity_sha256'], controller_mode=route['controller_mode'],
        controller_source_sha256=digest(CONTROLLER))
    if any(decision.get(k) != v for k, v in expected.items()):
        raise ValueError('Deployment evidence does not authorize this exact task/model/controller')
    if not decision.get('validation_report') or not decision.get('validation_report_sha256'):
        raise ValueError('Deployment evidence requires a frozen validation report')
    if digest(decision['validation_report']) != decision['validation_report_sha256']:
        raise ValueError('Candidate validation report changed')
    return dict(path=str(evidence), sha256=digest(evidence), decision=decision)


def build_plan(config_path, task_id):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    route = select_route(config, task_id)
    sources = config['runtime_sources_sha256']
    if set(sources) != {str(p) for p in required_sources()}:
        raise ValueError('Incomplete or unexpected runtime source inventory')
    verify_file_map(sources)
    evidence = verify_deployment_evidence(route)
    identity = verify_checkpoint(route['model'])
    return dict(schema_version=1, task_id=task_id, selected_role=route['role'],
        reason=route['reason'], checkpoint=identity['path'],
        checkpoint_identity_sha256=identity['identity_sha256'], controller_mode=route['controller_mode'],
        model_spec=route['model'], deployment_evidence=evidence,
        config=str(config_path), config_sha256=digest(config_path),
        runtime_sources_sha256=sources, gpu_loaded=False,
        selection_inputs=['explicit_task_id'], one_model_per_process=True,
        protocol=dict(task_config='demo_clean', prompt_prefix='none', normalization='quantile',
            noise_source='jax', policy_seed=0, action_horizon=50, episodes_per_process=1,
            robotwin_root=str(ROBOTWIN)),
        limitations=['Routing preserves the selected baseline weights/controller for non-target tasks; it does not itself measure unchanged success rates.',
            'Unknown task IDs select baseline, but unsupported simulator task names still fail normally.',
            'A deployment decision must be based on an independent validation report; restore verification alone is insufficient.'])


def runtime_environment():
    env = {k: v for k, v in os.environ.items()
        if not k.startswith(('ROBOTWIN_', 'GEN1_', 'EVOLVEVLA_ROUTING_'))}
    for key in ('PYTHONPATH', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_ALLOCATOR', 'XLA_CLIENT_MEM_FRACTION'):
        env.pop(key, None)
    env.update(ROBOTWIN_OPENPI_SRC=str(OPENPI/'src'),
        ROBOTWIN_RENDER_BACKEND='rt', ROBOTWIN_RT_DENOISER='oidn',
        ROBOTWIN_OIDN_RUNTIME=str(ROOT/'third_party/oidn-2.5.1/lib'),
        OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_PYTHON_CLIENT_MEM_FRACTION='.12', CUDA_VISIBLE_DEVICES='0',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return env


def command_for(plan, output, seed, episode_steps=2000):
    non_original = plan['controller_mode'] != 'original'
    if non_original and (plan['selected_role'] != 'candidate' or plan['task_id'] not in TARGETS):
        raise ValueError('A non-original controller is forbidden for baseline/non-target routes')
    command = [str(INFERENCE_PYTHON), '-I', '-u', '-B', str(SCRIPT if non_original else HARNESS)]
    if non_original:
        command.append('_runtime')
    command += ['--checkpoint', plan['checkpoint'], '--task', plan['task_id'],
        '--episodes', '1', '--episode-steps', str(episode_steps), '--seed', str(seed),
        '--task-config', 'demo_clean', '--prompt-prefix', 'none', '--normalization', 'quantile',
        '--policy-seed', '0', '--noise-source', 'jax', '--sim-python', str(SIM_PYTHON),
        '--robotwin-root', str(ROBOTWIN), '--output', str(Path(output).resolve())]
    return command


def launch(plan, receipt, output, seed, episode_steps=2000):
    env = runtime_environment()
    env[RECEIPT_ENV], env[RECEIPT_SHA_ENV] = str(Path(receipt).resolve()), digest(receipt)
    return subprocess.run(command_for(plan, output, seed, episode_steps), env=env, cwd=ROOT, check=False).returncode


def load_runtime_receipt():
    path = Path(os.environ[RECEIPT_ENV])
    if digest(path) != os.environ[RECEIPT_SHA_ENV]:
        raise ValueError('Routing receipt changed')
    receipt = read_json(path)
    verify_file_map(receipt['runtime_sources_sha256'])
    config_path = Path(receipt['config'])
    if digest(config_path) != receipt['config_sha256']:
        raise ValueError('Routing configuration changed')
    route = select_route(read_json(config_path), receipt['task_id'])
    if (route['role'] != 'candidate' or route['controller_mode'] == 'original'
            or route['controller_mode'] != receipt['controller_mode']
            or route['model'] != receipt['model_spec']):
        raise ValueError('Internal simulator wrapper requires the selected verified target route')
    verify_deployment_evidence(route)
    return receipt


@contextlib.contextmanager
def simulator_controller_hook(receipt):
    """Install only a selected target controller; reset gate on every setup."""
    sys.path.insert(0, str(SCRIPT.parent))
    import gen1_duration_controller_20261020 as duration
    original_import = importlib.import_module
    patched = []
    stream = None
    def hooked(name, package=None):
        nonlocal stream
        module = original_import(name, package)
        if name != 'eval_policy_xpolicylab' or patched:
            return module
        expected = ROBOTWIN/'scripts/eval_policy_xpolicylab.py'
        if Path(module.__file__).resolve() != expected:
            raise ValueError('Unexpected simulator evaluator')
        construct = module.class_decorator
        def make(task_id, *args, **kwargs):
            nonlocal stream
            if task_id != receipt['task_id'] or task_id not in TARGETS:
                raise ValueError('Simulator task differs from the selected route')
            task = construct(task_id, *args, **kwargs)
            original_take, original_setup = task.take_action, task.setup_demo
            if Path(original_take.__func__.__code__.co_filename).resolve() != ROBOTWIN/'envs/_base_task.py':
                raise ValueError('Unexpected simulator action controller')
            current = duration.create(original_take, receipt['controller_mode'])
            episode_setup, action_index = 0, 0
            if stream is None:
                stream = Path(receipt['controller_trace']).open('x', buffering=1)
            def setup(*a, **kw):
                nonlocal current, episode_setup, action_index
                result = original_setup(*a, **kw)
                current = duration.create(original_take, receipt['controller_mode'])
                episode_setup += 1
                action_index = 0
                return result
            def take(action, *a, **kw):
                nonlocal action_index
                try:
                    return current[0](action, *a, **kw)
                finally:
                    stream.write(json.dumps(dict(task_id=task_id, episode_setup=episode_setup,
                        action_index=action_index, controller=current[2].last_record), allow_nan=False)+'\n')
                    action_index += 1
            task.setup_demo, task.take_action = setup, take
            return task
        module.class_decorator = make
        patched.append((module, construct))
        return module
    importlib.import_module = hooked
    try:
        yield
    finally:
        importlib.import_module = original_import
        for module, construct in patched:
            module.class_decorator = construct
        if stream is not None:
            stream.close()


def internal_runtime():
    receipt = load_runtime_receipt()
    sys.path[:0] = [str(SCRIPT.parent), str(OPENPI/'src'), str(OPENPI/'packages/openpi-client/src')]
    import robotwin_public_openpi_harness as harness
    args = harness.parse_args()
    if args.task != receipt['task_id'] or Path(args.robotwin_root).resolve() != ROBOTWIN:
        raise ValueError('Runtime arguments do not match routing receipt')
    if args.role == 'sim':
        with simulator_controller_hook(receipt):
            return harness.run_simulator(args)
    if Path(args.checkpoint).resolve() != Path(receipt['checkpoint']).resolve():
        raise ValueError('Runtime checkpoint differs from routing receipt')
    verify_checkpoint(receipt['model_spec'])
    # The legacy parent derives only its child-script path from __file__.
    # Redirect that path in this process; no on-disk harness source is changed.
    harness.__file__ = str(SCRIPT)
    return harness.run_parent(args)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_runtime':
        del sys.argv[1]
        return internal_runtime()
    if len(sys.argv) > 1 and sys.argv[1] == '--role':
        return internal_runtime()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('operation', choices=('plan', 'run'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=100000)
    parser.add_argument('--episode-steps', type=int, default=2000)
    args = parser.parse_args()
    if args.operation == 'run' and args.output is None:
        parser.error('run requires --output')
    if args.episode_steps < 1 or args.episode_steps > 2000 or args.seed < 0:
        parser.error('Invalid action ceiling or seed')
    if args.output and args.output.exists():
        raise ValueError('Output already exists; use a fresh path')
    plan = build_plan(args.config, args.task)
    args.receipt = args.receipt.resolve()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    plan['controller_trace'] = str(args.receipt.with_suffix('.controller.jsonl'))
    if Path(plan['controller_trace']).exists():
        raise ValueError('Controller trace already exists')
    with args.receipt.open('x') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(task_id=plan['task_id'], selected_role=plan['selected_role'],
        reason=plan['reason'], checkpoint=plan['checkpoint'], controller_mode=plan['controller_mode'],
        receipt=str(args.receipt), gpu_loaded=False)), flush=True)
    if args.operation == 'run':
        return launch(plan, args.receipt, args.output, args.seed, args.episode_steps)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
