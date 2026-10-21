#!/usr/bin/env python3
"""A/B replay of sparse labels with an automatically detected terminal release.

This is a CPU-only preparation tool plus an optional replay runner.  Condition
A uses the unchanged sparse controller (including its 50 tick qpos fallback).
Condition B changes only the four qpos fallback literals to ``--retime-ticks``
for the last contiguous source interval where the arm is stationary and a
gripper opens monotonically.  No TOPP flags, velocity behavior, success
predicate, source arrays, or learned policy are changed.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import gen1_expert_label_replay_20261019 as core
import gen1_expert_label_replay_v2_20261019 as raw_replay
import gen1_expert_gripper_timing_replay_20261019 as timing_replay


def release_intervals(state, action, joint_epsilon=1e-6, gripper_epsilon=1e-6):
    """Return contiguous stationary-arm, monotonically opening intervals.

    The source HDF5 stores drive targets.  For a transition ``state[t]`` to
    ``action[t]``, an arm is stationary when all 12 arm dimensions are within
    ``joint_epsilon``.  Both gripper dimensions may stay constant or increase;
    at least one must increase by ``gripper_epsilon``.  A negative gripper
    change breaks the interval, so the rule cannot silently classify closing
    or oscillating actions as release.
    """
    import numpy as np
    state = np.asarray(state)
    action = np.asarray(action)
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
        raise ValueError('Expected matching (N, 14) source arrays')
    arm_delta = np.abs(action[:, list(core.JOINTS)] - state[:, list(core.JOINTS)])
    gripper_delta = action[:, [6, 13]] - state[:, [6, 13]]
    mask = ((arm_delta.max(axis=1) < joint_epsilon)
            & (gripper_delta.min(axis=1) >= -gripper_epsilon)
            & (gripper_delta.max(axis=1) > gripper_epsilon))
    intervals = []
    start = None
    for index, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index - 1
            intervals.append(dict(start=start, end=end, length=end-start+1))
            start = None
    return intervals


def terminal_release(state, action, joint_epsilon=1e-6, gripper_epsilon=1e-6):
    """Select the final release interval and reject short/no candidates."""
    intervals = release_intervals(state, action, joint_epsilon, gripper_epsilon)
    if not intervals:
        raise ValueError('No stationary-arm monotonically opening interval found')
    selected = intervals[-1]
    if selected['length'] < 2:
        raise ValueError(f'Terminal release interval is too short: {selected}')
    return dict(selected, all_intervals=intervals,
                rule='last contiguous interval: stationary 12 arm dims; '
                     'nonnegative gripper deltas; at least one positive')


def _retime_eligible(index, current, target, source_state, allowed, joint_epsilon=1e-6,
                     gripper_epsilon=1e-6):
    """Use source-selected indices plus runtime target matching as safeguards."""
    import numpy as np
    if index not in allowed:
        return False
    return bool(np.max(np.abs(target[list(core.JOINTS)] - source_state[list(core.JOINTS)])) < joint_epsilon
                and np.max(np.abs(target[list(core.JOINTS)] - current[list(core.JOINTS)])) < joint_epsilon
                and np.max(np.abs(target[[6, 13]] - source_state[[6, 13]])) > gripper_epsilon)


def _retime_simulator():
    """Run timing replay with an index allow-list injected process-locally."""
    case = json.loads(os.environ['GEN1_TERMINAL_CASE'])
    config = json.loads(os.environ['GEN1_TERMINAL_CONFIG'])
    allowed = set(config['release_indices'])
    original = timing_replay.eligible_transition

    def eligible(index, start_index, current, target, source_state, joint_epsilon=1e-6,
                 gripper_epsilon=1e-6):
        # Keep the timing module's signature while making the source-derived
        # allow-list authoritative; start_index is set to zero by this script.
        return _retime_eligible(index, current, target, source_state, allowed,
                                joint_epsilon, gripper_epsilon)

    timing_replay.eligible_transition = eligible
    try:
        # The existing timing replay performs the exact same initial parity,
        # expert gate, real physics step counting, and official check_success.
        os.environ['GEN1_RETIME_CASE'] = json.dumps(case)
        os.environ['GEN1_RETIME_CONFIG'] = json.dumps(dict(
            gripper_only_ticks=config['retime_ticks'], retime_from_index=0,
            joint_epsilon=config['joint_epsilon'], gripper_epsilon=config['gripper_epsilon']))
        return timing_replay.simulator()
    finally:
        timing_replay.eligible_transition = original


def simulator():
    mode = os.environ.get('GEN1_TERMINAL_MODE', 'raw50')
    if mode == 'raw50':
        return raw_replay.simulator()
    if mode == 'retime15':
        return _retime_simulator()
    raise ValueError(f'Unknown terminal replay mode: {mode}')


def _prepare_cases(manifest, limit_per_task):
    cases = core.select_cases(manifest, limit_per_task, False)
    counts = {}
    for case in cases:
        counts[case['task']] = counts.get(case['task'], 0) + 1
    if counts.get('hanging_mug', 0) < 10 or counts.get('place_mouse_pad', 0) < 10:
        raise ValueError(f'Panel must contain at least 10 cases per task: {counts}')
    return cases


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        del sys.argv[1]
        return simulator()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-per-task', type=int, default=10)
    parser.add_argument('--retime-ticks', type=int, default=15)
    parser.add_argument('--joint-epsilon', type=float, default=1e-6)
    parser.add_argument('--gripper-epsilon', type=float, default=1e-6)
    parser.add_argument('--initial-mae-tolerance', type=float, default=8.)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if args.limit_per_task < 10 or args.retime_ticks < 1 or args.timeout < 1:
        parser.error('Need at least 10 cases per task, positive retime ticks, and timeout')
    if args.joint_epsilon != 1e-6 or args.gripper_epsilon != 1e-6:
        parser.error('Fixed source detection epsilons are required')
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    cases = _prepare_cases(manifest, args.limit_per_task)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    prepared = []
    for case in cases:
        state, action = core.source_arrays(case)
        roundtrip = float(np.abs(core.roundtrip_actions(
            state, action, json.loads(core.NORM.read_text())['norm_stats']) - action).max())
        if roundtrip > 1e-5:
            raise ValueError(f'Roundtrip mismatch: {case["id"]}: {roundtrip}')
        release = terminal_release(state, action, args.joint_epsilon, args.gripper_epsilon)
        prepared.append(dict(id=case['id'], task=case['task'], frames=len(state),
                             next_state_action_exact=True, roundtrip_max_abs=roundtrip,
                             release=release))
    source_path = core.ROOT/'robotwin_repo_ba0fd21/envs/_base_task.py'
    _, patch_report = timing_replay.patch_fallback_function(source_path, args.retime_ticks)
    config = dict(retime_ticks=args.retime_ticks, joint_epsilon=args.joint_epsilon,
                  gripper_epsilon=args.gripper_epsilon)
    manifest_out = dict(protocol=__doc__, source_manifest=str(manifest_path),
                        source_manifest_sha256=core.digest(manifest_path), cases=cases,
                        preflight=prepared, panel_counts={
                            'hanging_mug': sum(x['task'] == 'hanging_mug' for x in cases),
                            'place_mouse_pad': sum(x['task'] == 'place_mouse_pad' for x in cases)},
                        intervention=config, method_patch=patch_report,
                        gpu_launched=False,
                        code_sha256={str(path): core.digest(path) for path in
                                     (SCRIPT, core.SCRIPT, raw_replay.SCRIPT,
                                      timing_replay.SCRIPT)})
    core.write(args.output/'manifest.json', manifest_out)
    print('CPU_PREFLIGHT_PASSED', json.dumps(manifest_out['preflight']), flush=True)
    if args.mode == 'prepare':
        return

    # run is intentionally explicit: prepare never imports GPU modules and
    # never launches a simulator.  Each condition has independent artifacts.
    records = []
    case_by_id = {case['id']: case for case in cases}
    for item in prepared:
        case = case_by_id[item['id']]
        release_indices = list(range(item['release']['start'], item['release']['end'] + 1))
        for mode in ('raw50', 'retime15'):
            directory = args.output/case['id']/mode
            directory.mkdir(parents=True)
            os.environ['GEN1_TERMINAL_MODE'] = mode
            os.environ['GEN1_TERMINAL_CASE'] = json.dumps(case)
            os.environ['GEN1_TERMINAL_CONFIG'] = json.dumps(dict(
                release_indices=release_indices, retime_ticks=args.retime_ticks,
                joint_epsilon=args.joint_epsilon, gripper_epsilon=args.gripper_epsilon))
            run_args = argparse.Namespace(**vars(args))
            run_args.allow_gate_bypass = False
            core.SCRIPT = SCRIPT
            record = core.run_case(case, directory, run_args)
            record.update(condition=mode, release=item['release'], release_indices=release_indices,
                          source_hash=case.get('source_hdf5_sha256', case.get('source_sha256')))
            core.write(directory/'result.json', record)
            records.append(record)
            core.write(args.output/'progress.json', dict(requested=2*len(cases),
                      attempted=len(records), records=records, gpu_launched=True))
            print('AB_RESULT', json.dumps({k: record.get(k) for k in
                                           ('id', 'condition', 'status', 'success', 'steps',
                                            'elapsed_seconds')}), flush=True)
    core.write(args.output/'summary.json', dict(records=records,
              requested=2*len(cases), successes=sum(bool(r.get('success')) for r in records),
              gpu_launched=True))


if __name__ == '__main__':
    main()
