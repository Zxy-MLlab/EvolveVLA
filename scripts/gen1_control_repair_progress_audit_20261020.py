#!/usr/bin/env python3
"""Independent CPU audit of DEVELOPMENT runs only; never opens confirmation results."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path('/root/autodl-tmp/EvolveVLA')
DEFAULT = ROOT / 'outputs/gen1_control_repair_20261020'
OLD = ROOT / 'outputs/gen1_timing_validation_20261020'
MODES = ('original', 'fixed15', 'opening15')
FIELDS = ('left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states')
CAMERAS = {'cam_high': 'cam_head', 'cam_left_wrist': 'cam_left_wrist', 'cam_right_wrist': 'cam_right_wrist'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lines(path):
    if not Path(path).exists():
        return []
    return [json.loads(s) for s in Path(path).read_text().splitlines(keepends=True) if s.strip() and s.endswith('\n')]


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(value)
    temp.replace(path)


def classify(row, actor):
    if row is None or 'timing_summary' not in row or 'timing_mode' not in row:
        return 'pending'
    if row.get('status') == 'expert_rejected':
        return 'expert_rejected'
    allowed = {'completed', 'source_exhausted', 'eval_limit_reached'} if actor == 'expert' else {'completed'}
    if row.get('status') not in allowed:
        return 'infrastructure_or_protocol_error'
    if row.get('simulator_returncode') != 0 or type(row.get('success')) is not bool:
        return 'invalid_terminal_record'
    if row['status'] != 'completed' and row['success']:
        return 'invalid_terminal_record'
    return 'success' if row['success'] else 'failure'


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) if len(a) != len(b) else None)


def source_check(case, directory, actor, result):
    if actor == 'expert':
        with np.load(directory / 'initial_arrays.npz', allow_pickle=False) as a:
            state = a['reset_state'].copy()
            stored_source = a['source_state'].copy()
            images = {k: a[k].copy() for k in CAMERAS}
    else:
        state = np.load(directory / 'initial/state.npy', allow_pickle=False)
        stored_source = None
        images = {k: np.load(directory / 'initial' / f'{k}.npy', allow_pickle=False) for k in CAMERAS}
    with h5py.File(case['source_hdf5']) as f:
        source_state = np.concatenate([f[f'state/{key}'][0] for key in FIELDS]).astype(np.float32)
        source_actions = np.concatenate([f[f'action/{key}'][:] for key in FIELDS], axis=1).astype(np.float32)
        camera_equal = {}
        for name, source in CAMERAS.items():
            ok, data = cv2.imencode('.jpg', images[name])
            camera_equal[name] = bool(ok and data.tobytes() == bytes(f[f'vision/{source}/colors'][0]))
    hashes = {k: dict(shape=list(a.shape), sha256=hashlib.sha256(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).hexdigest())
              for k, a in images.items()}
    hashes.update(state=hashlib.sha256(np.asarray(state, dtype='<f4').reshape(14).tobytes()).hexdigest(),
                  prompt=hashlib.sha256(case['prompt'].encode()).hexdigest())
    checks = dict(source_hash_exact=digest(case['source_hdf5']) == case['source_hdf5_sha256'],
        source_state_exact=bool(np.array_equal(state, source_state)), camera_jpeg_exact=camera_equal,
        source_copy_exact=stored_source is None or bool(np.array_equal(stored_source, source_state)),
        stored_input_hashes_exact=actor == 'expert' or hashes == result.get('hashes'),
        reported_source_parity_exact=(result.get('source_parity') or {}).get('state_exact') is True
            and (result.get('source_parity') or {}).get('all_source_jpeg_bytes_equal') is True)
    exact = all(v if isinstance(v, bool) else all(v.values()) for v in checks.values())
    return dict(checks, all_exact=exact), hashes, source_actions


def opening_windows(physical, timing):
    answer = {}
    for side in ('left', 'right'):
        v = [r[side]['drive_targets'][-1] for r in physical]
        closed = [i for i in range(1, len(v)) if v[i - 1] >= .2 and v[i] < .2]
        opened = [i for i in range(1, len(v)) if v[i - 1] <= .8 and v[i] > .8 and any(c < i for c in closed)]
        intervals = []
        for end in opened:
            start = max(i for i in range(end) if v[i] <= .2)
            rows = timing[start:end]
            intervals.append(dict(start_boundary=start, end_boundary=end, actions=len(rows),
                start_reported_gripper=v[start], end_reported_gripper=v[end],
                actual_physics_ticks=sum(r['physics_steps'] for r in rows),
                simulation_seconds=sum(r['simulation_seconds'] for r in rows),
                active_action_indices=[r['action_index'] for r in rows if r['duration_control'].get('active')],
                normal_path_resampled_actions=sum(any(a.get('arm_path_resampled') for a in r['duration_control'].get('arms', {}).values()) for r in rows),
                fallback_action_indices=[r['action_index'] for r in rows if r['fallback_triggered']]))
        answer[side] = dict(closure_boundaries=closed, opening_after_closure_boundaries=opened, windows=intervals)
    return answer


def timing_checks(timing, steps, success, mode):
    errors = []
    if len(timing) != steps or [r['action_index'] for r in timing] != list(range(steps)):
        errors.append('incomplete_or_noncontiguous_timing')
    total = 0
    for i, row in enumerate(timing):
        n = row['physics_steps']
        total += n
        if type(n) is not int or n < 0 or total != row['total_physics_steps']:
            errors.append('invalid_or_inconsistent_actual_ticks')
        control = row.get('duration_control') or {}
        if not control.get('qpos_hook_reached') or control.get('mode') != mode:
            errors.append('duration_hook_missing_or_wrong_mode')
        active = control.get('active') is True
        if mode == 'original' and active or mode == 'fixed15' and not active:
            errors.append('mode_activation_contradiction')
        if active:
            scheduled = control.get('scheduled_physics_ticks')
            censored = success and i == len(timing) - 1 and row.get('eval_success_latched') is True
            if scheduled != 15 or n > 15 or (n != 15 and not censored):
                errors.append('active_duration_not_15_or_valid_success_censor')
            for arm, data in control.get('arms', {}).items():
                if data.get('arm_path_resampled'):
                    if data.get('all_finite') is not True or data.get('position_endpoint_max_abs') != 0:
                        errors.append('resampled_path_nonfinite_or_endpoint_changed')
                if not data.get('topp_success') and data.get('arm_setter_preserved'):
                    errors.append('fallback_arm_setter_introduced')
    return sorted(set(errors))


def audit_condition(case, mode, directory, actor, manifest, manifest_errors):
    result_path = directory / 'result.json'
    result = read(result_path) if result_path.exists() else None
    outcome = classify(result, actor)
    report = dict(id=case['id'], task=case['task'], actor=actor, mode=mode,
        outcome=outcome, status=(result or {}).get('status', 'pending'),
        success=(result or {}).get('success') if outcome in ('success', 'failure') else None,
        protocol_eligible=False, exclusions=[], result_path=str(result_path))
    if outcome not in ('success', 'failure'):
        report['exclusions'] = [outcome]
        return report
    errors = list(manifest_errors)
    gate = result.get('expert_gate') or {}
    if not (gate.get('status') == 'passed' and gate.get('bypassed') is False):
        errors.append('expert_gate_not_passed_or_bypassed')
    try:
        source, input_hashes, source_actions = source_check(case, directory, actor, result)
        report.update(source_audit=source, input_hashes=input_hashes)
        if not source['all_exact']:
            errors.append('source_or_saved_input_not_exact')
        timing = lines(directory / 'timing.jsonl')
        if actor == 'expert':
            trace = lines(directory / 'trace.jsonl')
            physical = [result['initial_parity']['physical']] + [r['physical'] for r in trace]
            targets = [r['source_target'] for r in trace]
            if [r['source_action_index'] for r in trace] != list(range(result['steps'])):
                errors.append('expert_trace_indices_or_count')
            if not all(np.array_equal(np.asarray(t, dtype=np.float32), source_actions[i]) for i, t in enumerate(targets)):
                errors.append('expert_targets_not_source_exact')
            if not all(t['timing']['total_physics_steps'] == s['total_physics_steps'] for t, s in zip(trace, timing)):
                errors.append('expert_trace_timing_mismatch')
        else:
            trace = lines(directory / 'physical.jsonl')
            predictions = lines(directory / 'actions.jsonl')
            physical = [r['physical'] for r in trace]
            targets = [r['target'] for r in timing]
            if [r['step'] for r in trace] != list(range(result['steps'] + 1)):
                errors.append('policy_physical_trace_indices_or_count')
            checkpoint = manifest.get('checkpoint') or {}
            matches = (result.get('checkpoint_identity_sha256') == checkpoint.get('identity_sha256')
                       and result.get('checkpoint') == checkpoint.get('path') and bool(checkpoint.get('identity_sha256')))
            report.update(checkpoint=result.get('checkpoint'), checkpoint_identity=result.get('checkpoint_identity_sha256'),
                          checkpoint_matches_frozen_manifest=matches)
            if not matches:
                errors.append('checkpoint_not_frozen_manifest')
            if not predictions or predictions[0].get('step') != 0:
                errors.append('first_policy_prediction_missing')
            report['_predictions'] = predictions
        if len(physical) != result['steps'] + 1 or len(targets) != result['steps']:
            errors.append('physical_or_target_count_mismatch')
        if [r['target'] for r in timing] != targets:
            errors.append('timing_targets_differ_from_executed_trace')
        errors += timing_checks(timing, result['steps'], result['success'], mode)
        if (result['timing_summary'].get('actions_recorded') != len(timing)
                or result['timing_summary'].get('physics_steps') != sum(r['physics_steps'] for r in timing)):
            errors.append('reported_timing_summary_mismatch')
        arms = [a for r in timing for a in r['duration_control'].get('arms', {}).values()]
        active = [r['action_index'] for r in timing if r['duration_control'].get('active')]
        report.update(steps=result['steps'], actual_physics_ticks=sum(r['physics_steps'] for r in timing),
            active_actions=len(active), first_active_action=active[0] if active else None,
            active_action_indices=active,
            normal_path_resampled_arm_calls=sum(a.get('arm_path_resampled', False) for a in arms),
            normal_path_resampled_actions=sum(any(a.get('arm_path_resampled') for a in r['duration_control'].get('arms', {}).values()) for r in timing),
            max_commanded_velocity_rad_s=max((a.get('new_max_abs_command_velocity', 0) for a in arms), default=0),
            max_position_secant_velocity_rad_s=max((a.get('new_max_abs_position_difference_velocity', 0) for a in arms), default=0),
            maximum_velocity_scale=max((a.get('velocity_scale', 0) for a in arms), default=0),
            initial_physical=physical[0] if physical else None,
            opening_windows=opening_windows(physical, timing),
            source_artifact_hashes={p.name: digest(p) for p in [result_path, directory / 'timing.jsonl', directory / ('trace.jsonl' if actor == 'expert' else 'physical.jsonl')]},
            _physical=physical, _targets=targets, _ticks=[r['physics_steps'] for r in timing],
            _timing=timing)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        errors.append(f'audit_artifact_error:{type(error).__name__}:{error}')
    report.update(exclusions=sorted(set(errors)), protocol_eligible=not errors)
    return report


def transparency(row, actor):
    if not row['protocol_eligible']:
        return dict(status='not_available')
    directory = OLD / ('expert_ab' if actor == 'expert' else 'candidate_policy_ab') / row['id'] / ('raw50' if actor == 'expert' else 'original')
    path = directory / 'result.json'
    if not path.exists():
        return dict(status='archived_reference_missing')
    previous = read(path)
    if previous.get('status') not in ('completed', 'source_exhausted', 'eval_limit_reached') or previous.get('simulator_returncode') != 0:
        return dict(status='archived_reference_not_completed')
    if actor == 'expert':
        old_trace = lines(directory / 'trace.jsonl')
        physical = [previous['initial_parity']['physical']] + [r['physical'] for r in old_trace]
        targets = [r['source_target'] for r in old_trace]
        ticks = [r['timing']['physics_steps'] for r in old_trace]
        same_checkpoint = True
    else:
        old_trace = lines(directory / 'timing.jsonl')
        physical = [r['physical'] for r in lines(directory / 'physical.jsonl')]
        targets = [r['target'] for r in old_trace]
        ticks = [r['physics_steps'] for r in old_trace]
        same_checkpoint = row['checkpoint_identity'] == previous.get('checkpoint_identity_sha256')
    matches = dict(initial_physical=physical[0] == row['_physical'][0],
        complete_physical_trace=physical == row['_physical'], complete_targets=targets == row['_targets'],
        complete_tick_trace=ticks == row['_ticks'], checkpoint=same_checkpoint,
        success_and_steps=(previous['success'], previous['steps']) == (row['success'], row['steps']))
    prediction = {}
    if actor == 'policy':
        previous_predictions = lines(directory / 'actions.jsonl')
        a, b = np.asarray(previous_predictions[0]['actions']), np.asarray(row['_predictions'][0]['actions'])
        prediction = dict(first_full_prediction_exact=bool(np.array_equal(a, b)),
            first_full_prediction_max_abs_difference=float(np.max(np.abs(a - b))),
            initial_input_hashes_exact=row['input_hashes'] == previous.get('hashes'),
            interpretation='Cross-process repeatability only; archived predictions are not the treatment control. Current-run pairs must independently match initial predictions and pre-intervention traces.')
    return dict(status='exact' if all(matches.values()) else 'cross_run_mismatch', checks=matches,
                reference=str(directory), first_physical_difference=first_difference(physical, row['_physical']),
                first_target_difference=first_difference(targets, row['_targets']),
                first_tick_difference=first_difference(ticks, row['_ticks']), **prediction)


def pair_admission(a, b, actor):
    if a['outcome'] == 'pending' or b['outcome'] == 'pending':
        return ['pending']
    if a['outcome'] == 'expert_rejected' or b['outcome'] == 'expert_rejected':
        return ['expert_rejected']
    errors = []
    if not a['protocol_eligible'] or not b['protocol_eligible']:
        errors.append('condition_protocol_exclusion')
        return errors
    if not a.get('initial_physical') or a['initial_physical'] != b.get('initial_physical'):
        errors.append('initial_physical_mismatch_or_missing')
    if not a.get('input_hashes') or a['input_hashes'] != b.get('input_hashes'):
        errors.append('initial_observation_hash_mismatch_or_missing')
    if actor == 'policy' and (not a.get('checkpoint_identity') or a.get('checkpoint_identity') != b.get('checkpoint_identity')
                              or a.get('checkpoint') != b.get('checkpoint')):
        errors.append('checkpoint_mismatch_or_missing')
    return errors


def pair_report(a, b, actor):
    errors = pair_admission(a, b, actor)
    r = dict(id=a['id'], task=a['task'], mode=b['mode'], original_success=a['success'],
             modified_success=b['success'], valid_pair=False, exclusions=errors)
    if errors:
        return r
    first_active = b['first_active_action']
    count = first_active if first_active is not None else len(b['_targets'])
    # Physical boundary at first activation and that action's model target both precede any intervention.
    inclusive_targets = count + 1 if first_active is not None else count
    prefix = dict(physical_before_activation_exact=a['_physical'][:count + 1] == b['_physical'][:count + 1],
        targets_through_activation_exact=a['_targets'][:inclusive_targets] == b['_targets'][:inclusive_targets],
        ticks_before_activation_exact=a['_ticks'][:count] == b['_ticks'][:count],
        original_prefix_long_enough=len(a['_targets']) >= inclusive_targets,
        boundaries_compared=count + 1, targets_compared=inclusive_targets)
    if not all(prefix[k] for k in ['physical_before_activation_exact', 'targets_through_activation_exact', 'ticks_before_activation_exact', 'original_prefix_long_enough']):
        errors.append('pre_intervention_prefix_mismatch_or_missing')
    if first_active is None and (a['_physical'] != b['_physical'] or a['_targets'] != b['_targets'] or a['_ticks'] != b['_ticks']):
        errors.append('inactive_intervention_changed_trace')
    first_prediction = None
    if actor == 'policy':
        prediction_a, prediction_b = a['_predictions'][0], b['_predictions'][0]
        equal = prediction_a['actions'] == prediction_b['actions'] and prediction_a['state'] == prediction_b['state']
        first_prediction = dict(first_full50_prediction_and_state_exact=equal,
            maximum_action_difference=float(np.max(np.abs(np.asarray(prediction_a['actions']) - np.asarray(prediction_b['actions'])))))
        if not equal:
            errors.append('current_run_first_prediction_mismatch')
    r.update(valid_pair=not errors, exclusions=errors, pre_intervention=prefix,
        first_prediction=first_prediction,
        first_active_action=first_active, first_physical_difference_boundary=first_difference(a['_physical'], b['_physical']),
        first_target_difference_action=first_difference(a['_targets'], b['_targets']),
        first_tick_difference_action=first_difference(a['_ticks'], b['_ticks']),
        original_actual_ticks=a['actual_physics_ticks'], modified_actual_ticks=b['actual_physics_ticks'])
    return r


def paired_counts(pairs):
    valid = [p for p in pairs if p['valid_pair']]
    return dict(scheduled_pairs=len(pairs), valid_pairs=len(valid),
        original_successes=sum(p['original_success'] for p in valid), modified_successes=sum(p['modified_success'] for p in valid),
        original_rate=sum(p['original_success'] for p in valid) / len(valid) if valid else None,
        modified_rate=sum(p['modified_success'] for p in valid) / len(valid) if valid else None,
        improved_ids=[p['id'] for p in valid if not p['original_success'] and p['modified_success']],
        regressed_ids=[p['id'] for p in valid if p['original_success'] and not p['modified_success']],
        exclusions=dict(Counter(e for p in pairs for e in p['exclusions'])))


def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if not k.startswith('_')}
    if isinstance(value, list):
        return [public(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    args = parser.parse_args()
    panel = read(args.root / 'panel_development4.json')
    cases = panel['cases']
    reports = {}
    for actor, name in [('expert', 'expert_development'), ('policy', 'candidate_development')]:
        run = args.root / name
        manifest = read(run / 'manifest.json') if (run / 'manifest.json').exists() else {}
        checks = {}
        errors = []
        if manifest:
            checks = dict(panel_hash=digest(manifest['panel']) == manifest['panel_sha256'],
                cases=[c['id'] for c in manifest['cases']] == [c['id'] for c in cases],
                actor=manifest['actor'] == actor, modes=manifest['timing_modes'] == list(MODES),
                frozen_code=all(digest(p) == h for p, h in manifest['source_sha256'].items()))
            errors = ['manifest_' + k for k, v in checks.items() if not v]
        conditions = [audit_condition(c, mode, run / c['id'] / mode, actor, manifest, errors) for c in cases for mode in MODES]
        lookup = {(c['id'], c['mode']): c for c in conditions}
        for c in cases:
            original = lookup[c['id'], 'original']
            original['original_transparency'] = transparency(original, actor)
        pairs = [pair_report(lookup[c['id'], 'original'], lookup[c['id'], mode], actor) for mode in MODES[1:] for c in cases]
        totals = {mode: {task: paired_counts([p for p in pairs if p['mode'] == mode and (task == 'all' or p['task'] == task)])
                         for task in ['place_mouse_pad', 'hanging_mug', 'all']} for mode in MODES[1:]}
        reports[name] = dict(actor=actor, manifest_present=bool(manifest), manifest_checks=checks,
            complete=(run / 'summary.json').exists(), conditions_finalized=sum(c['outcome'] != 'pending' for c in conditions),
            conditions_planned=len(conditions), outcome_counts=dict(Counter(c['outcome'] for c in conditions)),
            paired_totals=totals, pairs=pairs, conditions=conditions)
    report = public(dict(created_utc=datetime.now(timezone.utc).isoformat(), gpu_used=False,
        development_only=True, confirmation_results_opened=False, reports=reports,
        audit_script_sha256=digest(__file__), limitations=[
            'Development4 deliberately includes known successes/failures and cannot estimate population task success.',
            'Only exact-source, equal-input/initial-physical, complete zero-returncode pairs enter primary rates.',
            'Expert rejection, pending results and errors are excluded, not failures; no seed replacement is performed.',
            'Checkpoint identity is matched to frozen runner manifest; this audit does not reread all model tensor shards.',
            'Reported velocity peaks cover resampled paths. Original/unmodified paths are not instrumented for full peak velocities.',
            'Opening windows use reported-command boundaries after closure, not physical gripper aperture/contact timing.',
            'Archived original runs provide a separate repeatability check, not the treatment control. Cross-process mismatch does not exclude an otherwise strictly matched current-run pair; current-run first full prediction and pre-intervention histories must be exact.',
            'No confirmation results are read or used for selecting modes.']))
    atomic(args.root / 'progress_summary.json', json.dumps(report, indent=2) + '\n')
    md = ['# 控制修复开发集：独立配对审计', '', f'更新时间：{report["created_utc"]}', '',
          '| 执行者 | 模式 | 任务 | 有效配对 | original成功 | 新模式成功 | 改善 | 退化 |',
          '|---|---|---|---:|---:|---:|---|---|']
    for name, run in reports.items():
        for mode, tasks in run['paired_totals'].items():
            for task, row in tasks.items():
                md.append(f'| {name} | {mode} | {task} | {row["valid_pairs"]} | {row["original_successes"]} | '
                          f'{row["modified_successes"]} | {", ".join(row["improved_ids"]) or "—"} | {", ".join(row["regressed_ids"]) or "—"} |')
    md += ['', '| 执行者 | 案例 | 模式 | 状态/成功 | 动作数 | 激活动作数 | resample动作数 | 实际ticks | 发出速度峰值(rad/s)* | 排除项 |',
           '|---|---|---|---|---:|---:|---:|---:|---:|---|']
    for name, run in reports.items():
        for row in run['conditions']:
            velocity = row.get('max_commanded_velocity_rad_s') if row.get('normal_path_resampled_arm_calls') else '未测'
            md.append(f'| {name} | {row["id"]} | {row["mode"]} | {row["outcome"]}/{row["success"]} | {row.get("steps", "—")} | '
                f'{row.get("active_actions", "—")} | {row.get("normal_path_resampled_actions", "—")} | {row.get("actual_physics_ticks", "—")} | '
                f'{velocity} | {", ".join(row["exclusions"]) or "—"} |')
    md += ['', '| 执行者 | 案例 | 模式 | 首激活动作 | 首物理差异边界 | 首目标差异动作 | 干预前一致 | 当前首次完整预测一致 |',
           '|---|---|---|---:|---:|---:|---|---|']
    for name, run in reports.items():
        for row in run['pairs']:
            if 'pre_intervention' not in row:
                continue
            p = row['pre_intervention']
            md.append(f'| {name} | {row["id"]} | {row["mode"]} | {row["first_active_action"]} | '
                f'{row["first_physical_difference_boundary"]} | {row["first_target_difference_action"]} | '
                f'{p["physical_before_activation_exact"] and p["targets_through_activation_exact"] and p["ticks_before_activation_exact"]} | '
                f'{(row.get("first_prediction") or {}).get("first_full50_prediction_and_state_exact", "专家固定源目标")} |')
    md += ['', '| 执行者 | 案例 | 模式 | 终末臂首次打开命令边界 | 窗口实际ticks | 窗口激活动作数 |',
           '|---|---|---|---|---:|---:|']
    for name, run in reports.items():
        for row in run['conditions']:
            case = next(c for c in cases if c['id'] == row['id'])
            side = 'right' if case['task'] == 'hanging_mug' else case['active_arm']
            windows = row.get('opening_windows', {}).get(side, {}).get('windows', [])
            if not windows:
                continue
            w = windows[0]
            md.append(f'| {name} | {row["id"]} | {row["mode"]} | {w["start_boundary"]}→{w["end_boundary"]} | '
                f'{w["actual_physics_ticks"]} | {len(w["active_action_indices"])} |')
    md += ['', '与旧轮original的可重复性核对（不是本轮处理效果）：', '']
    for name, run in reports.items():
        for row in run['conditions']:
            audit = row.get('original_transparency')
            if audit and audit['status'] != 'not_available':
                md.append(f'- {name}/{row["id"]}：{audit["status"]}；首次预测最大差异 '
                          f'{audit.get("first_full_prediction_max_abs_difference", "不适用")}。')
    md += ['', '*速度峰值仅涵盖执行重采样的路径；未测不代表机器人速度为0。', '',
           '详细JSON包含源state/JPEG独立核验、初始物理/输入/权重一致性、干预前前缀、首个物理/目标/时长差异、每臂打开命令窗口ticks和original透明回归。', '',
           '本表只读开发集；拒绝、缺失、非零返回码与审计不通过不进入有效配对分母。确认集未读取。', '']
    atomic(args.root / 'progress_summary.md', '\n'.join(md))
    print(json.dumps({name: dict(finalized=r['conditions_finalized'], planned=r['conditions_planned'], complete=r['complete'],
        pairs={m: t['all'] for m, t in r['paired_totals'].items()}) for name, r in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
