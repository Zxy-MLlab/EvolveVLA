#!/usr/bin/env python3
"""CPU-only snapshot: locate actual fallback branches relative to observed task stages."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import gen1_horizon_stage_audit_20261019 as stage


def predicates(result, physical):
    if result['task'] == 'hanging_mug':
        return stage.mug_predicates(result, physical)
    p, q = physical['mouse']['position'], physical['mouse']['quaternion']
    target = physical['target']['position']
    error = [p[k] - target[k] for k in (0, 1)]
    return dict(position=abs(error[0]) < .015 and abs(error[1]) < .012,
        orientation=abs(abs(q[2]*q[3])-.49) < .015 or abs(abs(q[0]*q[1])-.49) < .015,
        left_open=physical['left']['drive_targets'][-1] > .8,
        right_open=physical['right']['drive_targets'][-1] > .8), dict(xy_error=error)


def classify(result, physical, timing):
    stages = stage.classify(result, physical)
    side = stages.get('active_arm', 'right')
    events = stages[side + '_commands']
    closed = events['closure_steps']
    release = events['opening_after_closure']
    first_close, first_release = (closed[0] if closed else None), (release[0] if release else None)
    opening_interval = None
    if first_release is not None:
        start = max(i for i in range(first_release)
            if physical[i]['physical'][side]['drive_targets'][-1] <= .2)
        interval = timing[start:first_release]
        start_predicates, start_geometry = predicates(result, physical[start]['physical'])
        end_predicates, end_geometry = predicates(result, physical[first_release]['physical'])
        opening_interval = dict(start_boundary_step=start, end_boundary_step=first_release,
            start_reported_gripper=physical[start]['physical'][side]['drive_targets'][-1],
            end_reported_gripper=physical[first_release]['physical'][side]['drive_targets'][-1],
            first_action_index=start, final_action_index=first_release-1,
            actions=len(interval), physics_steps=sum(r['physics_steps'] for r in interval),
            simulation_seconds=sum(r['simulation_seconds'] for r in interval),
            fallback_actions=sum(r['fallback_triggered'] for r in interval),
            active_arm_fallback_actions=sum(any(e['arm'] == side for e in r['fallback_branches']) for r in interval),
            fallback_action_indices=[r['action_index'] for r in interval if r['fallback_triggered']],
            start_predicates=start_predicates, start_geometry=start_geometry,
            end_predicates=end_predicates, end_geometry=end_geometry,
            meaning='Scene ticks between latest reported command boundary <=.2 and first subsequent >.8 after a closure; not physical gripper/contact duration.')
    actor = 'mouse' if result['task'] == 'place_mouse_pad' else 'mug'
    positions = np.asarray([r['physical'][actor]['position'] for r in physical])
    lower = np.minimum.accumulate(positions[::-1], axis=0)[::-1]
    upper = np.maximum.accumulate(positions[::-1], axis=0)[::-1]
    stable = int(np.flatnonzero(np.max(upper-lower, axis=1) < .002)[0])
    activations = []
    for row in timing:
        if not row['fallback_triggered']:
            continue
        index = row['action_index']
        step = index + 1
        before, after = physical[index]['physical'], physical[step]['physical']
        if first_release is not None and step >= first_release:
            phase = 'at_or_after_terminal_open_command'
        elif first_close is not None and step >= first_close:
            phase = 'after_terminal_close_before_open_command'
        else:
            phase = 'before_terminal_close_command'
        before_predicates, before_geometry = predicates(result, before)
        after_predicates, after_geometry = predicates(result, after)
        suffix = positions[step:]
        prior = positions[max(0, index-20):index+1]
        activations.append(dict(action_index=index, executed_step=step, phase=phase,
            steps_since_first_terminal_open=None if first_release is None else step-first_release,
            fallback_branches=row['fallback_branches'], controller_locals=row['controller_locals'],
            physics_steps=row['physics_steps'], total_physics_steps=row['total_physics_steps'],
            fallback_on_terminal_active_arm=any(e['arm'] == side for e in row['fallback_branches']),
            active_arm=side, target_gripper=row['target'][6 if side == 'left' else 13],
            reported_gripper_before=before[side]['drive_targets'][-1],
            reported_gripper_after=after[side]['drive_targets'][-1],
            actor_displacement_during_action=float(np.linalg.norm(positions[step]-positions[index])),
            actor_coordinate_range_previous20=np.ptp(prior, axis=0).tolist(),
            actor_coordinate_range_remaining=np.ptp(suffix, axis=0).tolist(),
            before_predicates=before_predicates, after_predicates=after_predicates,
            before_geometry=before_geometry, after_geometry=after_geometry,
            remaining_to_action_limit_or_success=result['steps']-step))
    return dict(active_arm=side, first_terminal_close_step=first_close,
        first_terminal_open_step=first_release, terminal_stage=stages,
        first_terminal_opening_command_interval=opening_interval,
        final_actor_suffix_coordinate_range_below2mm_from_step=stable,
        fallback_actions=len(activations), first_fallback=activations[0] if activations else None,
        activations=activations)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    records, sources = [], {}
    for label in ('baseline', 'candidate'):
        run = args.root / (label + '_policy_ab')
        manifest_path = run / 'manifest.json'
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        for case in manifest['cases']:
            for mode in manifest['timing_modes']:
                directory = run / case['id'] / mode
                result_path = directory / 'result.json'
                if not result_path.exists():
                    continue
                result = json.loads(result_path.read_text())
                if 'timing_summary' not in result:
                    continue  # worker result has not been finalized by its parent yet.
                record = dict(model=label, id=case['id'], task=case['task'], timing_mode=mode,
                    status=result['status'], success=result.get('success'), steps=result.get('steps'),
                    source_exact=bool((result.get('source_parity') or {}).get('state_exact') and
                        (result.get('source_parity') or {}).get('all_source_jpeg_bytes_equal')))
                sources[str(result_path)] = stage.horizon.digest(result_path)
                if result['status'] == 'completed':
                    physical_path, timing_path = directory / 'physical.jsonl', directory / 'timing.jsonl'
                    physical, timing = stage.read_lines(physical_path), stage.read_lines(timing_path)
                    if len(physical) != result['steps']+1 or len(timing) != result['steps']:
                        raise ValueError(f'Incomplete completed trace: {directory}')
                    if [r['step'] for r in physical] != list(range(len(physical))):
                        raise ValueError(f'Noncontiguous physical trace: {directory}')
                    record['phase_audit'] = classify(result, physical, timing)
                    for path in (physical_path, timing_path):
                        sources[str(path)] = stage.horizon.digest(path)
                records.append(record)
    report = dict(gpu_used=False, snapshot_finished_conditions=len(records), records=records,
        source_sha256=sources, script_sha256=stage.horizon.digest(SCRIPT), limitations=[
            'All action indices are zero-based; physical and executed steps are one-based after each action.',
            'Close/open are reported gripper command crossings at .2/.8, not contact measurements.',
            'Timing interventions can activate without changing total action duration when the other arm takes longer.',
            'A trigger after release or after the object becomes stationary describes timing, not proof of irreversibility.',
            'Snapshot uses only finished conditions; pending or rejected conditions are not policy failures.'])
    stage.horizon.atomic_json(args.output, report)
    lines = ['# 模型 fallback 触发阶段（CPU 快照）', '',
        '| 模型 | 案例 | 条件 | 结果 | 首次释放后动作 | 首次 fallback 动作 | 触发次数 | 触发阶段 |',
        '|---|---|---|---|---:|---:|---:|---|']
    for row in records:
        audit = row.get('phase_audit', {})
        first = audit.get('first_fallback') or {}
        result = ('成功' if row['success'] else '失败') if row['status'] == 'completed' else row['status']
        lines.append(f'| {row["model"]} | {row["id"]} | {row["timing_mode"]} | {result} | '
            f'{audit.get("first_terminal_open_step")} | {first.get("executed_step")} | '
            f'{audit.get("fallback_actions")} | {first.get("phase", "未触发")} |')
    lines += ['', '动作编号采用执行后的1起始step；详细分支、实际物理步、前后成功条件和物体位移见JSON。'
              '“释放后触发”不证明失败不可逆，也不能代表释放时长已对齐。']
    args.markdown.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(finished=len(records), fallback_conditions=sum(
        (r.get('phase_audit') or {}).get('fallback_actions', 0) > 0 for r in records))))


if __name__ == '__main__':
    main()
