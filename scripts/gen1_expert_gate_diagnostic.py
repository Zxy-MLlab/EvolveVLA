#!/usr/bin/env python3
"""Separate, review-disabled expert diagnostic. Never produces benchmark results.

Exactly four fresh processes are scheduled, with no retry/replacement. The frozen
harness performs its own expert setup/play/check. A scoped close wrapper exits
before policy setup even on acceptance. Getter taps observe only calls made by
the actual predicate; skipped gripper checks remain skipped. Setup snapshots add
read-only pose getters, not simulation, observation rendering, RNG or planning.
Execution requires a subsequent reviewed source change: no CLI enable switch.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen1_raw_paired_eval as frozen

# The declared four-run diagnostic is complete; future execution stays disabled.
EXECUTION_ENABLED = False
REPETITIONS = (0, 1, 2, 3)
TASK = 'hanging_mug'
SEED = 200002
REPO = frozen.ROOT / 'robotwin_repo_ba0fd21'
SOURCE_MANIFEST = frozen.OUTPUTS / 'gen1_raw_paired_run_v1_20261011/manifest.json'


class ExpertFinished(BaseException):
    """Bypass the harness's ordinary episode-error/retry handler."""


def require_enabled():
    if not EXECUTION_ENABLED:
        raise RuntimeError('Expert diagnostic execution disabled pending main review')


def verify_frozen_sources():
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    # Match the manifest format explicitly rather than silently skipping entries.
    for filename, digest in manifest['code'].items():
        path = Path(filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen code identity mismatch: {path}')
    return manifest


class DiagnosticFailure(BaseException):
    """Abort instrumentation errors without the harness retry handler."""


def scalar_bool(value):
    """Copy only audited boolean scalars without another truth evaluation."""
    if type(value) is bool:
        return value
    if type(value).__module__ == 'numpy' and type(value).__name__ == 'bool_':
        return value.item()
    raise DiagnosticFailure('Unexpected non-boolean predicate scalar')


def vector(value):
    return [float(x) for x in value]


def setup_snapshot(task):
    """Minimal actor physical state, explicitly not a full simulator snapshot."""
    result = {}
    for name in ('mug', 'rack'):
        pose = getattr(task, name).get_pose()
        result[name] = {'position': vector(pose.p), 'quaternion': vector(pose.q)}
    return result


def tapped_check(task, original, record):
    """Return the original value object; never call a getter twice or force bool."""
    values = {}
    def tap(obj, name, key, serialize):
        method = getattr(obj, name)
        def wrapped(*args, **kwargs):
            value = method(*args, **kwargs)
            if key in values:
                raise RuntimeError('Unexpected repeated success getter')
            values[key] = serialize(value)
            return value
        return frozen.replaced(obj, name, wrapped)
    with contextlib.ExitStack() as stack:
        stack.enter_context(tap(task.mug, 'get_functional_point', 'mug_function', vector))
        stack.enter_context(tap(task.rack, 'get_pose', 'rack_position', lambda p: vector(p.p)))
        stack.enter_context(tap(task.rack, 'get_functional_point', 'rack_function', vector))
        stack.enter_context(tap(task, 'is_right_gripper_open', 'right_gripper_open', scalar_bool))
        try:
            result = original()
            record['check_success'] = scalar_bool(result)
            return result
        finally:
            record['predicate_inputs'] = values
            # Derived arithmetic only, after the actual call. No success method,
            # comparison on a simulator object, or extra gripper getter is used.
            if all(k in values for k in ('mug_function', 'rack_position', 'rack_function')):
                m, r, f = (values[k] for k in ('mug_function', 'rack_position', 'rack_function'))
                record['predicate_margins'] = {
                    'xy_strict_positive': [0.02 - abs(m[i] - (r[i] + f[i]) / 2) for i in (0, 1)],
                    'height_strict_positive': m[2] - 0.86,
                    'gripper_check_evaluated': 'right_gripper_open' in values,
                    'height_comparison_evaluated': ('right_gripper_open' in values and values['right_gripper_open'] is True),
                }


@contextlib.contextmanager
def instrument_task(task, record):
    setup, play, check = task.setup_demo, task.play_once, task.check_success
    calls = {'setup': 0, 'play': 0, 'check': 0}
    play_active = [False]
    record['checks_during_play'] = 0
    record['calls'] = calls
    def setup_wrapped(*args, **kwargs):
        calls['setup'] += 1
        if calls['setup'] != 1:
            raise DiagnosticFailure('Repeated setup forbidden')
        result = setup(*args, **kwargs)
        record['setup_actor_poses'] = setup_snapshot(task)
        return result
    def play_wrapped(*args, **kwargs):
        calls['play'] += 1
        if calls['play'] != 1:
            raise RuntimeError('Repeated expert forbidden')
        play_active[0] = True
        try:
            result = play(*args, **kwargs)
        finally:
            play_active[0] = False
        record['plan_success'] = scalar_bool(getattr(task, 'plan_success', True))
        record['play_completed'] = True
        # Do not traverse tensors or planners, or introduce extra device syncs.
        record['planner_trajectory_identifiers'] = None
        record['planner_capture_limitation'] = 'No audited passive identifier accessor; not queried'
        return result
    def check_wrapped():
        if play_active[0]:
            record['checks_during_play'] += 1
            return check()
        calls['check'] += 1
        if calls['check'] != 1:
            raise RuntimeError('Repeated expert check forbidden')
        return tapped_check(task, check, record)
    with frozen.replaced(task, 'setup_demo', setup_wrapped), \
         frozen.replaced(task, 'play_once', play_wrapped), \
         frozen.replaced(task, 'check_success', check_wrapped):
        yield


@contextlib.contextmanager
def harness_instrumentation(harness, evaluator, record):
    """Use the actual gate, stopping at its first close before any policy path."""
    factory, close = evaluator.class_decorator, harness.safe_close_task
    closed = set()
    with contextlib.ExitStack() as stack:
        def create(name):
            if name != TASK:
                raise ValueError('Only declared task permitted')
            task = factory(name)
            stack.enter_context(instrument_task(task, record))
            return task
        def close_and_stop(task):
            if id(task) in closed:
                return
            closed.add(id(task))
            close(task)
            if (sys.exc_info()[0] is None and record.get('play_completed')
                    and (record.get('plan_success') is False or 'check_success' in record)):
                raise ExpertFinished()
        stack.enter_context(frozen.replaced(evaluator, 'class_decorator', create))
        stack.enter_context(frozen.replaced(harness, 'safe_close_task', close_and_stop))
        yield


def run_harness(harness, evaluator, args, record):
    with harness_instrumentation(harness, evaluator, record):
        finish_harness(harness, args, record)


def finish_harness(harness, args, record):
    try:
        harness.run_simulator(args)
    except ExpertFinished:
        if record['calls']['setup'] != 1 or record['calls']['play'] != 1:
            raise RuntimeError('Expert gate did not finish once')
        record['status'] = 'expert_only_finished'
        record['policy_loaded'] = False
        return
    raise RuntimeError('Frozen harness returned without expert-only stop')


def worker(directory, index):
    require_enabled()
    verify_frozen_sources()
    if index not in REPETITIONS:
        raise ValueError('Repetition outside predeclared schedule')
    frozen.restrict_worker_files()
    import importlib
    import robotwin_eval_harness as harness
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / 'scripts'))
    os.chdir(REPO)
    args = SimpleNamespace(task=TASK, seed=SEED, episodes=1, episode_steps=2000,
        robotwin_root=str(REPO), task_config='demo_clean', video_dir=None,
        controlled_case=True, first_instruction='diagnostic-never-used', first_instruction_seed=SEED)
    record = {'kind': 'expert_diagnostic_not_benchmark', 'repetition': index, 'seed': SEED}
    original_import = importlib.import_module
    attached = {}
    def hooked(name, package=None):
        module = original_import(name, package)
        if name == 'eval_policy_xpolicylab' and 'evaluator' not in attached:
            if Path(module.__file__).resolve() != REPO / 'scripts/eval_policy_xpolicylab.py':
                raise ValueError('Unexpected evaluator source')
            attached['evaluator'] = module
            stack = attached['stack']
            stack.enter_context(harness_instrumentation(harness, module, record))
        return module
    with contextlib.ExitStack() as stack:
        attached['stack'] = stack
        with frozen.replaced(importlib, 'import_module', hooked):
            finish_harness(harness, args, record)
    frozen.checked_write(directory / 'diagnostic.json', record, directory.parent)


def run(output, supervisor=None):
    require_enabled()
    manifest = verify_frozen_sources()
    import gen1_raw_diagnostic as resources
    import robotwin_public_openpi_harness as renderer_harness
    output = Path(output)
    if not output.name.startswith('gen1_expert_gate_diagnostic_') or output.exists():
        raise ValueError('A fresh uniquely named diagnostic output is required')
    frozen.guard(output)
    output.mkdir()
    frozen.checked_write(output / 'schedule.json', {
        'kind': 'expert_diagnostic_not_benchmark', 'task': TASK, 'seed': SEED,
        'repetitions': list(REPETITIONS), 'retry': False, 'replacement': False,
        'source_manifest': str(SOURCE_MANIFEST),
        'source_manifest_sha256': hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        'diagnostic_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, output)
    supervise = supervisor or frozen.supervise
    results = []
    for index in REPETITIONS:
        directory = output / f'repetition_{index:02d}'
        directory.mkdir()
        env = frozen.runtime_environment()
        # Helper normally runs in a worker inheriting the sanitized environment.
        # Supply that same metadata rather than coordinator ROBOTWIN_* overrides.
        with frozen.replaced(renderer_harness, '_renderer_metadata', lambda: {
                'render_backend': env['ROBOTWIN_RENDER_BACKEND'],
                'rt_denoiser': env['ROBOTWIN_RT_DENOISER'],
                'oidn_runtime': env['ROBOTWIN_OIDN_RUNTIME']}):
            renderer_harness._configure_sim_renderer_env(env)
        env['ROBOTWIN_CONTROLLED_OUTPUT'] = str(directory)
        with (directory / 'worker.log').open('x', encoding='utf-8') as log:
            evidence = supervise([manifest['config']['sim_python'], '-I', '-B', str(Path(__file__).resolve()),
                '--worker', str(index), '--output', str(directory)], env,
                log, output, resources, timeout=1800)
        if evidence['status'] == 'passed':
            try:
                result = json.loads((directory / 'diagnostic.json').read_text())
                if (result['status'] != 'expert_only_finished' or result['repetition'] != index
                        or result['seed'] != SEED or result['policy_loaded'] is not False):
                    raise ValueError('Invalid diagnostic worker result')
            except (OSError, ValueError, KeyError) as exc:
                evidence = dict(evidence, status='failed', diagnostic_error=str(exc))
        frozen.checked_write(directory / 'supervision.json', evidence, output)
        results.append(evidence)
        if evidence['status'] != 'passed':
            break  # Resource/instrumentation failure aborts, never replaces/retries.
    summary = {
        'kind': 'expert_diagnostic_not_benchmark', 'scheduled': 4,
        'finished': len(results), 'status': 'complete' if len(results) == 4 and all(
            r['status'] == 'passed' for r in results) else 'aborted'}
    frozen.checked_write(output / 'diagnostic_summary.json', summary, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=REPETITIONS)
    args = parser.parse_args()
    require_enabled()
    if args.worker is None:
        if run(args.output)['status'] != 'complete':
            raise SystemExit(1)
    else:
        worker(args.output, args.worker)


if __name__ == '__main__':
    main()
