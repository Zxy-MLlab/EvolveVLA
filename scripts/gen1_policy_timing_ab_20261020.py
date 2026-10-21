#!/usr/bin/env python3
"""Isolated policy A/B: original controller versus four qpos fallback literals 50→15.

Both conditions predict and execute 50-action chunks with unchanged policy,
TOPP paths, velocity flags, gripper interpolation, expert gate and success test.
fallback15 is a narrow fallback-duration intervention, not full time alignment.
No source-index, joint-delta threshold, or expert labels gate the intervention.
"""
import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import gen1_execution_horizon_ablation_20261019 as horizon
import gen1_expert_gripper_timing_replay_20261019 as retime
import gen1_expert_label_replay_20261019 as oracle
from gen1_expert_dense_replay_20261019 import SceneStepCounter

ROOT = horizon.ROOT
BASELINE = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
SOURCE = ROOT / 'robotwin_repo_ba0fd21/envs/_base_task.py'
MODES = ('original', 'fallback15')


def checkpoint_identity(path):
    path = Path(path).resolve()
    if (path / 'model_config.json').exists():
        result = horizon.checkpoint_identity(path)
        result['kind'] = 'verified_lora_candidate'
        return result
    if path != BASELINE.resolve() or not (path / 'params/_METADATA').is_file():
        raise ValueError('Only the fixed public baseline or verified LoRA export is supported')
    expected = {}
    for line in (path / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        expected[name.lstrip('*')] = sha
    files = sorted(p for sub in ('params', 'assets') for p in (path / sub).rglob('*') if p.is_file())
    hashes = {str(p.relative_to(path)): horizon.digest(p) for p in files}
    if not hashes or any(expected.get(name) != sha for name, sha in hashes.items()):
        raise ValueError('Baseline files differ from its recorded SHA256SUMS')
    hashes['SHA256SUMS'] = horizon.digest(path / 'SHA256SUMS')
    return dict(path=str(path), kind='fixed_public_baseline', files_sha256=hashes,
        identity_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())


class ActionTiming:
    """Passive actual-source tracing, identical in both controller conditions."""
    def __init__(self, source=SOURCE):
        self.counter = SceneStepCounter(source)
        _, patch = retime.patch_fallback_function(source, 15)
        rows = patch['replacements']
        self.branches = {r['line']: dict(arm=r['variable'].split('_')[0],
            reason=('exception' if i % 2 == 0 else 'empty_path')) for i, r in enumerate(rows)}
        self.reset()

    def reset(self):
        self.before = self.counter.count
        self.fallbacks = []
        self.final_locals = {}

    def trace(self, frame, event, value):
        if frame.f_code.co_filename != self.counter.source:
            return None
        self.counter.trace(frame, event, value)
        if frame.f_code.co_name == 'take_action':
            if event == 'line' and frame.f_lineno in self.branches:
                self.fallbacks.append(dict(line=frame.f_lineno, **self.branches[frame.f_lineno]))
            if event == 'return':
                for name in ('left_n_step', 'right_n_step', 'topp_left_flag', 'topp_right_flag'):
                    if name in frame.f_locals:
                        v = frame.f_locals[name]
                        self.final_locals[name] = bool(v) if name.startswith('topp_') else int(v)
        return self.trace

    def record(self, index, task, mode, action):
        timestep = float(task.scene.get_timestep())
        count = self.counter.count - self.before
        return dict(action_index=index, timing_mode=mode, physics_steps=count,
            total_physics_steps=self.counter.count, timestep=timestep,
            simulation_seconds=count*timestep, total_simulation_seconds=self.counter.count*timestep,
            fallback_branches=self.fallbacks, fallback_triggered=bool(self.fallbacks),
            modified_fallback_triggered=mode == 'fallback15' and bool(self.fallbacks),
            controller_locals=self.final_locals, target=list(map(float, action)),
            eval_success_latched=bool(task.eval_success))


def simulator():
    mode = os.environ['GEN1_POLICY_TIMING_MODE']
    if mode not in MODES:
        raise ValueError('Unsupported timing mode')
    directory = Path(os.environ['ROBOTWIN_CONTROLLED_OUTPUT'])
    binding = horizon.evaluation.binding
    original_import = importlib.import_module
    with contextlib.ExitStack() as stack:
        stream = stack.enter_context((directory / 'timing.jsonl').open('x', buffering=1))
        attached = False
        def hooked(name, package=None):
            nonlocal attached
            module = original_import(name, package)
            if name != 'eval_policy_xpolicylab' or attached:
                return module
            attached = True
            construct = module.class_decorator
            def make(*args, **kwargs):
                task = construct(*args, **kwargs)
                original_take = task.take_action
                original_play, original_check = task.play_once, task.check_success
                gate = dict(status='not_run', bypass_requested=False, bypassed=False)
                pending_gate = False
                def play(*a, **kw):
                    nonlocal pending_gate
                    result = original_play(*a, **kw)
                    pending_gate = bool(task.plan_success)
                    gate.update(status='pending_predicate' if pending_gate else 'rejected',
                                plan_success=bool(task.plan_success), check_success=None)
                    horizon.atomic_json(directory / 'expert_gate.json', gate)
                    return result
                def check(*a, **kw):
                    nonlocal pending_gate
                    result = original_check(*a, **kw)
                    if pending_gate:
                        pending_gate = False
                        gate.update(status='passed' if result else 'rejected', check_success=bool(result))
                        horizon.atomic_json(directory / 'expert_gate.json', gate)
                    return result
                stack.enter_context(binding.replaced(task, 'play_once', play))
                stack.enter_context(binding.replaced(task, 'check_success', check))
                modified_take, patch = retime.retimed_method(original_take, 15)
                horizon.write_json(directory / 'controller_patch.json', dict(timing_mode=mode,
                    applied=mode == 'fallback15', patch=patch))
                invoked = original_take if mode == 'original' else modified_take
                tracer, action_index = ActionTiming(), 0
                def take(action, *a, **kw):
                    nonlocal action_index
                    tracer.reset()
                    previous = sys.gettrace()
                    sys.settrace(tracer.trace)
                    try:
                        return invoked(action, *a, **kw)
                    finally:
                        sys.settrace(previous)
                        stream.write(json.dumps(tracer.record(action_index, task, mode, action),
                                                allow_nan=False) + '\n')
                        action_index += 1
                stack.enter_context(binding.replaced(task, 'take_action', take))
                return task
            stack.enter_context(binding.replaced(module, 'class_decorator', make))
            return module
        stack.enter_context(binding.replaced(importlib, 'import_module', hooked))
        horizon.install_literal_binding()
        return horizon.evaluation.simulator()


def timing_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if [r['action_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Noncontiguous timing action indices')
    return dict(actions_recorded=len(rows), physics_steps=sum(r['physics_steps'] for r in rows),
        simulation_seconds=sum(r['simulation_seconds'] for r in rows),
        fallback_actions=sum(r['fallback_triggered'] for r in rows),
        modified_fallback_actions=sum(r['modified_fallback_triggered'] for r in rows),
        fallback_branches={f'{arm}_{reason}': sum(
            e['arm'] == arm and e['reason'] == reason for r in rows for e in r['fallback_branches'])
            for arm in ('left', 'right') for reason in ('exception', 'empty_path')})


def source_parity(case, directory):
    import h5py
    import numpy as np
    initial = directory / 'initial'
    if not initial.exists():
        return None
    images = {camera: np.load(initial / f'{camera}.npy', allow_pickle=False) for camera in oracle.CAMERAS}
    states, _ = oracle.source_arrays(case)
    state = np.load(initial / 'state.npy', allow_pickle=False)
    with h5py.File(case['source_hdf5']) as stream:
        metrics = oracle.source_image_metrics(stream, 0, images)
    maximum = float(np.abs(state - states[0]).max())
    return dict(state_max_abs=maximum, state_exact=bool(np.array_equal(state, states[0])),
        images=metrics, all_source_jpeg_bytes_equal=all(x['jpeg_bytes_equal'] for x in metrics.values()),
        observed_source_match=oracle.initial_matches(metrics, maximum))


def summarize(records):
    counts = {}
    for mode in MODES:
        counts[mode] = {}
        for task in horizon.evaluation.TASKS:
            rows = [r for r in records if r['timing_mode'] == mode and r['task'] == task]
            done = [r for r in rows if r['status'] == 'completed']
            counts[mode][task] = dict(attempted=len(rows), completed=len(done),
                successes=sum(bool(r.get('success')) for r in done),
                expert_rejected=sum(r['status'] == 'expert_rejected' for r in rows),
                errors=sum(r['status'] not in ('completed', 'expert_rejected') for r in rows),
                fallback_actions=sum(r['timing_summary']['fallback_actions'] for r in rows),
                modified_fallback_actions=sum(r['timing_summary']['modified_fallback_actions'] for r in rows))
    paired = {}
    for task in horizon.evaluation.TASKS:
        matches, excluded, pending = [], [], []
        for name in dict.fromkeys(r['id'] for r in records if r['task'] == task):
            rows = {r['timing_mode']: r for r in records if r['id'] == name}
            if set(rows) != set(MODES):
                pending.append(name)
                continue
            a, b = (rows[mode] for mode in MODES)
            equal = (all(r['status'] == 'completed' for r in (a, b))
                and bool(a.get('hashes')) and a.get('hashes') == b.get('hashes')
                and bool(a.get('initial_physical_sha256'))
                and a.get('initial_physical_sha256') == b.get('initial_physical_sha256')
                and a.get('binding') == b.get('binding')
                and all((r.get('source_parity') or {}).get('observed_source_match') for r in (a, b)))
            if equal:
                matches.append(dict(id=name, original=bool(a['success']), fallback15=bool(b['success'])))
            else:
                excluded.append(name)
        paired[task] = dict(valid_pairs=len(matches), pairs=matches, excluded_case_ids=excluded,
            pending_case_ids=pending, original_successes=sum(r['original'] for r in matches),
            fallback15_successes=sum(r['fallback15'] for r in matches))
    return dict(attempted=len(records), counts=counts, paired=paired,
        limitation='Outcome-stratified training diagnostics; gate and initial parity must be checked for each comparison; fallback15 may never activate.')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        del sys.argv[1]
        return simulator()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--timing-modes', choices=MODES, nargs='+', default=list(MODES))
    parser.add_argument('--case-ids', nargs='+')
    parser.add_argument('--action-limit', type=int, default=2000)
    parser.add_argument('--case-timeout', type=int, default=1800)
    parser.add_argument('--device-memory-fraction', type=float, default=.12)
    parser.add_argument('--min-free-gib', type=float, default=18.)
    args = parser.parse_args()
    if (len(set(args.timing_modes)) != len(args.timing_modes) or not 0 < args.action_limit <= 2000
        or args.case_timeout <= 0 or not 0 < args.device_memory_fraction <= .25):
        parser.error('Invalid modes, action limit, timeout or memory fraction')
    args.output, args.checkpoint = args.output.resolve(), args.checkpoint.resolve()
    cases = horizon.load_cases(args.panel, args.case_ids)
    identity = checkpoint_identity(args.checkpoint)
    args.checkpoint_identity_sha256 = identity['identity_sha256']
    _, patch = retime.patch_fallback_function(SOURCE, 15)
    sources = [SCRIPT, Path(horizon.__file__), Path(retime.__file__), Path(oracle.__file__),
        SCRIPT.with_name('gen1_expert_dense_replay_20261019.py'), SOURCE,
        Path(horizon.evaluation.__file__), SCRIPT.with_name('gen1_raw_paired_eval.py'),
        SCRIPT.with_name('controlled_paired_eval.py'), SCRIPT.with_name('pi05_openpi_adapter.py'),
        SCRIPT.with_name('robotwin_eval_harness.py'), SCRIPT.with_name('robotwin_public_openpi_harness.py')]
    manifest = dict(protocol=__doc__, cases=cases, panel=str(args.panel.resolve()),
        panel_sha256=horizon.digest(args.panel), checkpoint=identity, timing_modes=args.timing_modes,
        model_output_horizon=50, execute_prefix=50, policy_seed=0, reset_rng_per_episode=True,
        action_limit=args.action_limit, case_timeout_seconds=args.case_timeout,
        device_memory_fraction=args.device_memory_fraction, preallocate=False,
        intervention=dict(kind='qpos_fallback_duration_only', original_ticks=50, modified_ticks=15,
            scope='all qpos fallback branches; no source-index, static-joint or gripper gate',
            complete_time_alignment=False, patch=patch),
        source_sha256={str(p): horizon.digest(p) for p in sources})
    args.output.mkdir(parents=True, exist_ok=False)
    horizon.write_json(args.output / 'manifest.json', manifest)
    if args.mode == 'prepare':
        horizon.write_json(args.output / 'preflight.json', dict(status='passed', gpu_used=False))
        print('PREFLIGHT_PASSED', args.output, flush=True)
        return
    gpu = horizon.gpu_status()
    if gpu['free_gib'] < args.min_free_gib:
        raise RuntimeError(f'Insufficient GPU memory: {gpu}; no inference launched')
    horizon.write_json(args.output / 'gpu_before.json', gpu)
    env = horizon.evaluation.binding.runtime_environment()
    env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.device_memory_fraction),
               CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    os.environ.update(env)
    sys.path[:0] = [str(horizon.evaluation.binding.OPENPI / 'src'),
                   str(horizon.evaluation.binding.OPENPI / 'packages/openpi-client/src')]
    horizon.install_literal_binding()
    cache, records = [], []
    with horizon.evaluation.binding.replaced(horizon, 'SCRIPT', SCRIPT), \
         (args.output / 'results.jsonl').open('x', buffering=1) as stream:
        for case in cases:
            for mode in args.timing_modes:
                for name, sha in manifest['source_sha256'].items():
                    if horizon.digest(name) != sha:
                        raise ValueError(f'Frozen source changed: {name}')
                os.environ['GEN1_POLICY_TIMING_MODE'] = mode
                directory = args.output / case['id'] / mode
                directory.mkdir(parents=True)
                result = horizon.worker(case, 50, args, cache, directory)
                result.update(timing_mode=mode, timing_summary=timing_summary(directory / 'timing.jsonl'),
                              source_parity=source_parity(case, directory))
                gate_path = directory / 'expert_gate.json'
                result['expert_gate'] = json.loads(gate_path.read_text()) if gate_path.exists() else None
                if result['status'] == 'completed' and result['timing_summary']['actions_recorded'] != result['steps']:
                    result.update(status='timing_audit_error', error='Timing rows differ from executed actions')
                horizon.atomic_json(directory / 'result.json', result)
                records.append(result)
                stream.write(json.dumps(result, allow_nan=False) + '\n')
                horizon.atomic_json(args.output / 'progress.json', summarize(records))
                print('TIMING_RESULT', json.dumps({k: result.get(k) for k in
                    ('id', 'timing_mode', 'status', 'success', 'steps', 'timing_summary', 'wall_seconds', 'error')}), flush=True)
                if result['status'] not in ('completed', 'expert_rejected'):
                    raise RuntimeError(f'Recorded infrastructure failure; no retry: {result.get("error")}')
                if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > 2 * 2**30:
                    raise RuntimeError('Stopped at two GiB run output budget after complete case')
    horizon.write_json(args.output / 'summary.json', summarize(records))


if __name__ == '__main__':
    main()
