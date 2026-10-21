#!/usr/bin/env python3
"""Isolated fixed-panel Clean ablation: predict 50 actions, execute 50/10/5.

The panel is fixed before launch. Every case/prefix has one attempt, the same
source seed/index/literal instruction, and the same simulator action limit.
This deliberately reuses the established passive physical trace and expert
gate, without editing either evaluator used by the ongoing trainset rollout.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen1_recovery_eval as evaluation

ROOT = evaluation.ROOT
SCRIPT = Path(__file__).resolve()
ORIGINAL_TEMPLATE = evaluation.binding.template_spec
ORIGINAL_BIND = evaluation.binding.bind_prompt


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def literal_template(task, seed, description_root):
    prompt = os.environ.get('GEN1_ABLATION_LITERAL_PROMPT')
    if not prompt:
        return ORIGINAL_TEMPLATE(task, seed, description_root)
    return dict(task=task, seed=seed, pool='training_seen_literal', index=-1,
                template=prompt, literal_prompt=True,
                source='local_training_instruction_json')


def literal_bind(spec, episode_info, description_root):
    if not spec.get('literal_prompt'):
        return ORIGINAL_BIND(spec, episode_info, description_root)
    original = ORIGINAL_TEMPLATE(spec['task'], spec['seed'], description_root)
    with evaluation.binding.replaced(evaluation.binding, 'template_spec', ORIGINAL_TEMPLATE):
        bound = ORIGINAL_BIND(original, episode_info, description_root)
    return dict(prompt=spec['template'], fields=bound['fields'],
                objects=bound['objects'], template=spec)


def install_literal_binding():
    evaluation.binding.template_spec = literal_template
    evaluation.binding.bind_prompt = literal_bind


def load_cases(path, requested_ids=None, verify_sources=True):
    payload = json.loads(Path(path).read_text())
    cases = payload['cases'] if isinstance(payload, dict) else payload
    if not cases or len({row['id'] for row in cases}) != len(cases):
        raise ValueError('Panel must contain unique cases')
    if requested_ids:
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError('Duplicate requested case IDs')
        lookup = {row['id']: row for row in cases}
        if not set(requested_ids) <= lookup.keys():
            raise ValueError('Requested case absent from fixed panel')
        cases = [lookup[name] for name in requested_ids]
    for row in cases:
        if row['task'] not in evaluation.TASKS or not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']):
            raise ValueError('Invalid case task or ID')
        if type(row['seed']) is not int or type(row['episode_index']) is not int:
            raise ValueError('Seed and collection index must be integers')
        if row['seed'] < 0 or row['episode_index'] < 0 or not isinstance(row['prompt'], str) or not row['prompt']:
            raise ValueError('Invalid seed, collection index or prompt')
        if verify_sources:
            episode = Path(row['source_episode_directory']).resolve()
            source = Path(row['source_hdf5']).resolve()
            expected = row.get('source_hdf5_sha256', row.get('source_sha256'))
            if not expected or digest(source) != expected or not source.is_relative_to(episode):
                raise ValueError(f'Source HDF5 identity mismatch: {row["id"]}')
            if int(episode.name.split('_')[-1]) != row['episode_index']:
                raise ValueError('Collection episode index differs from source directory')
            prompts = list((episode / 'instruction').glob('*.json'))
            if len(prompts) != 1 or json.loads(prompts[0].read_text())['seen'][0] != row['prompt']:
                raise ValueError('Panel prompt is not the original first seen instruction')
    return cases


def checkpoint_identity(root):
    root = Path(root).resolve()
    verification = json.loads((root / 'verification.json').read_text())
    if (verification.get('status') != 'passed' or
            verification.get('kind') != 'full_lora_model_unmerged' or
            not verification.get('production_restore_exact') or
            not verification.get('normalization_exact')):
        raise ValueError('Verified LoRA export required')
    model = json.loads((root / 'model_config.json').read_text())
    if model['action_horizon'] != 50 or model['action_dim'] != 32:
        raise ValueError('Expected unchanged 50-step, 32-dimensional model')
    files = sorted(p for sub in ('params', 'assets') for p in (root / sub).rglob('*') if p.is_file())
    if not files or not (root / 'params/_METADATA').is_file():
        raise ValueError('Checkpoint parameter shards missing')
    files += [root / 'model_config.json', root / 'verification.json']
    hashes = {str(p.relative_to(root)): digest(p) for p in files}
    return dict(path=str(root), files_sha256=hashes,
                identity_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())


def gpu_status():
    result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free,memory.total',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True, timeout=15)
    free, total = [float(v.strip()) for v in result.stdout.strip().split(',')]
    return dict(free_gib=free / 1024, total_gib=total / 1024)


def execution_loop(policy, first, prefix, send_action, receive_step, decode_observation,
                   record_inference, action_limit=2000, clock=time.perf_counter):
    """Pure control-loop seam: no simulator stepping during policy computation."""
    import numpy as np
    images, state = decode_observation(first['observation'])
    prompt = first['diagnostic_binding']['prompt']
    count = steps = 0
    latency = 0.0
    while steps < action_limit:
        started = clock()
        chunk = np.asarray(policy.infer_chunk(images=images, state=state, prompt=prompt))
        duration = clock() - started
        if chunk.shape != (50, 14) or not np.isfinite(chunk).all():
            raise ValueError('Expected finite 50 x 14 physical action chunk')
        count += 1
        latency += duration
        execute_count = min(prefix, action_limit - steps)
        record_inference(dict(step=steps, inference=count, inference_seconds=duration,
                              execute_prefix=execute_count, state=np.asarray(state).tolist(),
                              actions=chunk.tolist()), images)
        for action in chunk[:execute_count]:
            send_action(action.tolist())
            response = receive_step()
            if response['type'] != 'step':
                raise RuntimeError(str(response))
            steps += 1
            if response.get('terminated') or response.get('truncated'):
                return dict(status='completed', success=bool(response['info']['is_success']),
                            steps=steps, inference_count=count, inference_seconds=latency,
                            termination='terminated' if response.get('terminated') else 'truncated')
            images, state = decode_observation(response['observation'])
    return dict(status='completed', success=False, steps=steps, inference_count=count,
                inference_seconds=latency, termination='fixed_action_ceiling')


@contextlib.contextmanager
def case_timeout(seconds):
    def expired(signum, frame):
        raise TimeoutError(f'Case exceeded {seconds} seconds; no retry')
    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def worker(case, prefix, args, cache, directory):
    from PIL import Image
    from robotwin_eval_harness import receive_message, send_message
    from robotwin_public_openpi_harness import _observation_for_openpi, _configure_sim_renderer_env
    from pi05_openpi_adapter import PublicPi05Adapter

    started = time.monotonic()
    os.environ['GEN1_ABLATION_LITERAL_PROMPT'] = case['prompt']
    description = Path(evaluation.c.CONFIG['robotwin_root']) / 'description'
    spec = literal_template(case['task'], case['seed'], description)
    env = dict(os.environ, GEN1_RAW_TEMPLATE_SPEC=json.dumps(spec),
               ROBOTWIN_CONTROLLED_OUTPUT=str(directory))
    _configure_sim_renderer_env(env)
    command = [evaluation.c.CONFIG['sim_python'], '-I', '-B', str(SCRIPT), 'sim', '--role', 'sim',
               '--controlled-case', '--episodes', '1', '--task', case['task'],
               '--seed', str(case['seed']), '--episode-index', str(case['episode_index']),
               '--episode-steps', str(args.action_limit), '--robotwin-root',
               evaluation.c.CONFIG['robotwin_root'], '--task-config', 'demo_clean',
               '--first-instruction', case['prompt'], '--first-instruction-seed', str(case['seed'])]
    result = dict(status='runner_error')
    child = None
    with (directory / 'simulator.log').open('x') as log:
        try:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                     env=env, cwd=evaluation.c.CONFIG['robotwin_root'])
            with case_timeout(args.case_timeout):
                first = receive_message(child.stdout)
                if first.get('seed') != case['seed']:
                    raise ValueError('Simulator returned a different requested seed')
                if first['type'] == 'case_status':
                    result = {k: v for k, v in first.items() if k != 'type'}
                else:
                    if first['type'] != 'episode_start' or first.get('instruction') != case['prompt']:
                        raise ValueError('Simulator episode type or literal instruction differs')
                    bound = first['diagnostic_binding']
                    expected = literal_bind(spec, {'info': bound['fields']}, description)
                    evaluation.binding.require_same_binding(bound, expected)
                    images, state = _observation_for_openpi(first['observation'])
                    hashes = evaluation.c.capture(directory / 'initial', images, state, bound['prompt'])
                    limits = dict(official_step_limit=first.get('step_limit'),
                                  outer_action_ceiling=args.action_limit)
                    result.update(hashes=hashes, binding=bound, simulator_limits=limits)
                    write_json(directory / 'episode_start.json',
                               dict(info=first.get('info'), step_limit=first.get('step_limit'),
                                    binding=bound, hashes=hashes))
                    if not cache:
                        cache.append(PublicPi05Adapter(str(args.checkpoint), device='cuda:0',
                            prompt_prefix='none', normalization='quantile', policy_seed=0, noise_source='jax'))
                    policy = cache[0]
                    import jax
                    policy.policy._rng = jax.random.key(0)
                    policy._jax_rng = jax.random.key(0)
                    evaluation.c.guard_policy(policy)
                    with (directory / 'actions.jsonl').open('x', buffering=1) as trace:
                        def record(row, current_images):
                            trace.write(json.dumps(row, allow_nan=False) + '\n')
                            Image.fromarray(current_images['cam_high']).save(directory / f'frame_{row["step"]:04d}.jpg')
                        result = execution_loop(policy, first, prefix,
                            lambda action: send_message(child.stdin, dict(cmd='action', action=action)),
                            lambda: receive_message(child.stdout), _observation_for_openpi,
                            record, args.action_limit)
                    result.update(hashes=hashes, binding=bound, simulator_limits=limits)
        except Exception as exc:
            result.update(status='runner_error', error=f'{type(exc).__name__}: {exc}')
        finally:
            if child is not None:
                try:
                    send_message(child.stdin, {'cmd': 'stop'})
                    child.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                result['simulator_returncode'] = child.returncode
                if child.returncode and result['status'] == 'completed':
                    result.update(status='simulator_error', success=None)
    physical = directory / 'physical.jsonl'
    if physical.exists():
        with physical.open() as stream:
            initial = stream.readline().strip()
        if initial:
            result['initial_physical_sha256'] = hashlib.sha256(initial.encode()).hexdigest()
            result['physical_tracking'] = tracking_summary(physical)
    result.update(id=case['id'], task=case['task'], seed=case['seed'],
                  episode_index=case['episode_index'], execute_prefix=prefix,
                  checkpoint=str(args.checkpoint), wall_seconds=time.monotonic() - started,
                  checkpoint_identity_sha256=args.checkpoint_identity_sha256,
                  action_limit=args.action_limit, output=str(directory))
    write_json(directory / 'result.json', result)
    return result


def tracking_summary(path):
    """Measure drive-target versus actual joint tracking without stepping physics."""
    errors = {side: [] for side in ('left', 'right')}
    objects = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            physical = json.loads(line)['physical']
            for side in errors:
                arm = physical[side]
                errors[side].extend(abs(a - b) for a, b in
                    zip(arm['measured_qpos'][:6], arm['drive_targets'][:6], strict=True))
            for name in ('mouse', 'target', 'mug', 'rack'):
                if name in physical:
                    position = physical[name]['position']
                    if name not in objects:
                        objects[name] = dict(initial=position, final=position,
                                             minimum_z=position[2], maximum_z=position[2])
                    row = objects[name]
                    row['final'] = position
                    row['minimum_z'] = min(row['minimum_z'], position[2])
                    row['maximum_z'] = max(row['maximum_z'], position[2])
    return dict(joints={side: dict(cells=len(values),
        mean_abs_radians=sum(values) / len(values) if values else None,
        max_abs_radians=max(values) if values else None) for side, values in errors.items()},
        objects=objects, note='Passive drive-target minus measured-qpos diagnostic; not a task success predicate.')


def summarize(records, prefixes):
    rates = {}
    for prefix in prefixes:
        rates[str(prefix)] = {}
        for task in evaluation.TASKS:
            rows = [r for r in records if r['execute_prefix'] == prefix and r['task'] == task]
            completed = [r for r in rows if r['status'] == 'completed']
            rates[str(prefix)][task] = dict(attempted=len(rows), completed=len(completed),
                successes=sum(bool(r.get('success')) for r in completed),
                inference_seconds=sum(r.get('inference_seconds', 0) for r in completed),
                inference_count=sum(r.get('inference_count', 0) for r in completed))
    matched, mismatched = [], []
    for name in dict.fromkeys(r['id'] for r in records):
        rows = [r for r in records if r['id'] == name]
        if len(rows) != len(prefixes) or any(r['status'] != 'completed' for r in rows):
            continue
        equal = all(r.get('hashes') == rows[0].get('hashes') and
                    r.get('initial_physical_sha256') == rows[0].get('initial_physical_sha256')
                    for r in rows) and bool(rows[0].get('hashes')) and bool(rows[0].get('initial_physical_sha256'))
        (matched if equal else mismatched).append(name)
    pairs = {}
    for prefix in prefixes:
        if prefix == 50 or 50 not in prefixes:
            continue
        comparisons = []
        for name in matched:
            same = {r['execute_prefix']: r for r in records if r['id'] == name}
            comparisons.append((bool(same[50]['success']), bool(same[prefix]['success'])))
        pairs[str(prefix)] = dict(valid_pairs=len(comparisons),
            both_success=sum(a and b for a, b in comparisons),
            prefix_only_success=sum(not a and b for a, b in comparisons),
            baseline50_only_success=sum(a and not b for a, b in comparisons),
            both_failure=sum(not a and not b for a, b in comparisons))
    return dict(attempted=len(records), rates=rates, matched_case_ids=matched,
                initial_mismatch_case_ids=mismatched, paired_vs_50=pairs,
                limitation='Diagnostic panel may be outcome-stratified; initial equality is not proof of complete physics determinism.')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        del sys.argv[1]
        install_literal_binding()
        return evaluation.simulator()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['run'])
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'checkpoints/gen1_expanded_lora_step24000_export_20261015')
    parser.add_argument('--prefixes', type=int, nargs='+', default=[50, 10, 5])
    parser.add_argument('--case-ids', nargs='+')
    parser.add_argument('--action-limit', type=int, default=2000)
    parser.add_argument('--case-timeout', type=int, default=1800)
    parser.add_argument('--device-memory-fraction', type=float, default=0.18)
    parser.add_argument('--min-free-gib', type=float, default=20.0)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if (len(set(args.prefixes)) != len(args.prefixes) or not set(args.prefixes) <= {5, 10, 25, 50}
            or not 0 < args.action_limit <= 2000 or args.case_timeout <= 0
            or not 0 < args.device_memory_fraction <= 0.25):
        parser.error('Invalid prefixes, fixed action limit, timeout or GPU allocation')
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    cases = load_cases(args.panel, args.case_ids)
    identity = checkpoint_identity(args.checkpoint)
    args.checkpoint_identity_sha256 = identity['identity_sha256']
    gpu = gpu_status()
    if gpu['free_gib'] < args.min_free_gib:
        raise RuntimeError(f'Insufficient free GPU memory: {gpu}; no inference launched')
    manifest = dict(panel=str(args.panel.resolve()), panel_sha256=digest(args.panel), cases=cases,
        prefixes=args.prefixes, action_limit=args.action_limit, checkpoint=identity,
        model_output_horizon=50, policy_seed=0, reset_rng_per_episode=True,
        case_timeout_seconds=args.case_timeout, gpu_before=gpu, reuse_policy=True,
        device_memory_fraction=args.device_memory_fraction, preallocate=False,
        source_sha256={str(p): digest(p) for p in [SCRIPT, Path(evaluation.__file__),
            ROOT / 'scripts/pi05_openpi_adapter.py', ROOT / 'scripts/robotwin_eval_harness.py',
            ROOT / 'scripts/gen1_raw_paired_eval.py']},
        protocol='Fixed source seed/index/first instruction; Clean; one attempt per case/prefix; official limits; passive physical traces; no retries',
        noise_note='Each episode resets the same RNG; shorter prefixes make more model calls, so later noise is coupled by call index, not action timestep.')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'manifest.json', manifest)
    if args.preflight_only:
        write_json(args.output / 'preflight.json', dict(status='passed', gpu_used=False))
        print('PREFLIGHT_PASSED', args.output, flush=True)
        return
    env = evaluation.binding.runtime_environment()
    env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.device_memory_fraction),
               CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    os.environ.update(env)
    sys.path[:0] = [str(evaluation.binding.OPENPI / 'src'),
                    str(evaluation.binding.OPENPI / 'packages/openpi-client/src')]
    install_literal_binding()
    cache, records = [], []
    with (args.output / 'results.jsonl').open('x', buffering=1) as stream:
        for case in cases:
            for prefix in args.prefixes:
                directory = args.output / case['id'] / f'prefix_{prefix}'
                directory.mkdir(parents=True)
                result = worker(case, prefix, args, cache, directory)
                result['checkpoint_identity_sha256'] = identity['identity_sha256']
                stream.write(json.dumps(result, allow_nan=False) + '\n')
                records.append(result)
                atomic_json(args.output / 'progress.json', summarize(records, args.prefixes))
                print(json.dumps({k: result.get(k) for k in ('id', 'execute_prefix', 'status',
                    'success', 'steps', 'inference_count', 'wall_seconds', 'error')}), flush=True)
                if any(term in result.get('error', '') for term in ('RESOURCE_EXHAUSTED', 'out of memory', 'OutOfMemory')):
                    raise RuntimeError('Stopped after recorded GPU allocation error; no retry')
    write_json(args.output / 'summary.json', summarize(records, args.prefixes))


if __name__ == '__main__':
    main()
