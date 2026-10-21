#!/usr/bin/env python3
"""Independent CPU review of completed original/fallback15 learned-policy pairs."""
import argparse
import datetime
import json
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import gen1_policy_fallback_phase_audit_20261020 as phases
from gen1_execution_horizon_ablation_20261019 import atomic_json, digest


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_difference(a, b):
    return next((i for i in range(max(len(a), len(b)))
        if i >= len(a) or i >= len(b) or a[i] != b[i]), None)


def review_pair(original, modified):
    a, b = original['result'], modified['result']
    gate = lambda r: (r.get('expert_gate') or {}).get('status') == 'passed' and not (r.get('expert_gate') or {}).get('bypassed')
    source = lambda r: (r.get('source_parity') or {}).get('state_exact') and (r.get('source_parity') or {}).get('all_source_jpeg_bytes_equal')
    checks = dict(both_completed=all(r['status'] == 'completed' for r in (a, b)),
        expert_gate_passed=bool(gate(a) and gate(b)), source_state_and_jpeg_exact=bool(source(a) and source(b)),
        initial_inputs_equal=bool(a.get('hashes')) and a.get('hashes') == b.get('hashes'),
        initial_physical_equal=bool(a.get('initial_physical_sha256')) and a.get('initial_physical_sha256') == b.get('initial_physical_sha256'),
        prompt_binding_equal=a.get('binding') == b.get('binding'),
        checkpoint_identity_equal=a.get('checkpoint_identity_sha256') == b.get('checkpoint_identity_sha256'))
    result = dict(id=a['id'], task=a['task'], statuses=[a['status'], b['status']],
        successes=[a.get('success'), b.get('success')], pair_checks=checks,
        valid_pair=all(checks.values()))
    if not result['valid_pair']:
        return result
    physical = [read_lines(x['directory'] / 'physical.jsonl') for x in (original, modified)]
    timing = [read_lines(x['directory'] / 'timing.jsonl') for x in (original, modified)]
    predictions = [read_lines(x['directory'] / 'actions.jsonl') for x in (original, modified)]
    for condition, rows, clocks in zip((a, b), physical, timing, strict=True):
        if [r['step'] for r in rows] != list(range(condition['steps']+1)):
            raise ValueError('Completed physical trace is noncontiguous')
        if [r['action_index'] for r in clocks] != list(range(condition['steps'])):
            raise ValueError('Completed timing trace is noncontiguous')
    audits = [phases.classify(r, p, t) for r, p, t in zip((a, b), physical, timing, strict=True)]
    actions = [[r['target'] for r in rows] for rows in timing]
    plain_predictions = [[{k: v for k, v in r.items() if k != 'inference_seconds'} for r in rows] for rows in predictions]
    plain_timing = [[{k: v for k, v in r.items() if k not in ('timing_mode', 'modified_fallback_triggered')} for r in rows] for rows in timing]
    trigger_indices = [[r['action_index'] for r in rows if r['fallback_triggered']] for rows in timing]
    first_trigger = trigger_indices[1][0] if trigger_indices[1] else None
    first_physics = first_difference(physical[0], physical[1])
    result.update(steps=[a['steps'], b['steps']], fallback_actions=list(map(len, trigger_indices)),
        fallback_action_indices=trigger_indices, first_modified_fallback_action_index=first_trigger,
        first_physical_difference_after_executed_step=first_physics,
        first_executed_target_difference_action_index=first_difference(actions[0], actions[1]),
        first_prediction_difference_chunk_index=first_difference(plain_predictions[0], plain_predictions[1]),
        first_timing_semantics_difference_action_index=first_difference(plain_timing[0], plain_timing[1]),
        first_action_physics_ticks_difference_index=first_difference(
            [r['physics_steps'] for r in timing[0]], [r['physics_steps'] for r in timing[1]]),
        all_action_physics_ticks_equal=[r['physics_steps'] for r in timing[0]] == [r['physics_steps'] for r in timing[1]],
        all_executed_targets_equal=actions[0] == actions[1],
        all_full50_predictions_equal_excluding_latency=plain_predictions[0] == plain_predictions[1],
        all_physical_records_equal=physical[0] == physical[1],
        all_timing_records_equal_excluding_condition_labels=plain_timing[0] == plain_timing[1],
        physical_trace_bytes_equal=(original['directory']/'physical.jsonl').read_bytes() == (modified['directory']/'physical.jsonl').read_bytes(),
        original_stage=audits[0], fallback15_stage=audits[1])
    intervals = [audit['first_terminal_opening_command_interval'] for audit in audits]
    result['opening_command_interval_comparison'] = dict(original=intervals[0], fallback15=intervals[1],
        both_observed=all(interval is not None for interval in intervals))
    if all(interval is not None for interval in intervals):
        x, y = intervals
        result['opening_command_interval_comparison'].update(
            same_boundary_steps=(x['start_boundary_step'], x['end_boundary_step']) == (y['start_boundary_step'], y['end_boundary_step']),
            same_action_count=x['actions'] == y['actions'], same_physics_ticks=x['physics_steps'] == y['physics_steps'],
            physics_ticks_delta=y['physics_steps']-x['physics_steps'],
            fallback_actions=[x['fallback_actions'], y['fallback_actions']])
    if first_trigger is not None:
        k = first_trigger
        result['first_intervention_isolation'] = dict(
            executed_targets_through_intervention_equal=actions[0][:k+1] == actions[1][:k+1],
            physical_before_intervention_equal=physical[0][:k+1] == physical[1][:k+1],
            fallback_indices_before_first_modified_trigger_equal=[i for i in trigger_indices[0] if i < k] == [],
            original_controller_at_trigger=timing[0][k]['controller_locals'] if k < len(timing[0]) else None,
            modified_controller_at_trigger=timing[1][k]['controller_locals'],
            original_action_ticks=timing[0][k]['physics_steps'] if k < len(timing[0]) else None,
            modified_action_ticks=timing[1][k]['physics_steps'],
            first_physical_difference_before_trigger=first_physics is not None and first_physics <= k)
    else:
        result['zero_fallback_transparency_passed'] = bool(not trigger_indices[0]
            and result['all_executed_targets_equal'] and result['all_full50_predictions_equal_excluding_latency']
            and result['all_physical_records_equal'] and result['all_timing_records_equal_excluding_condition_labels']
            and a['success'] == b['success'])
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    pairs, pending, sources = [], [], {}
    all_finished = True
    for label in ('baseline', 'candidate'):
        run = args.root / (label + '_policy_ab')
        manifest_path = run / 'manifest.json'
        if not manifest_path.exists():
            all_finished = False
            pending.append(dict(model=label, id=None, timing_mode=None))
            continue
        manifest = json.loads(manifest_path.read_text())
        sources[str(manifest_path)] = digest(manifest_path)
        all_finished &= (run / 'summary.json').exists()
        for case in manifest['cases']:
            found = {}
            for mode in ('original', 'fallback15'):
                directory = run / case['id'] / mode
                path = directory / 'result.json'
                result = json.loads(path.read_text()) if path.exists() else None
                if result is None or 'timing_summary' not in result:
                    pending.append(dict(model=label, id=case['id'], timing_mode=mode))
                    all_finished = False
                    continue
                found[mode] = dict(result=result, directory=directory)
                sources[str(path)] = digest(path)
            if len(found) == 2:
                pair = review_pair(found['original'], found['fallback15'])
                pair['model'] = label
                pairs.append(pair)
                if pair['valid_pair']:
                    for member in found.values():
                        for name in ('physical.jsonl', 'timing.jsonl', 'actions.jsonl'):
                            path = member['directory'] / name
                            sources[str(path)] = digest(path)
    valid = [p for p in pairs if p['valid_pair']]
    zero = [p for p in valid if not any(p['fallback_actions'])]
    activated = [p for p in valid if p['fallback_actions'][1]]
    summary = dict(finished_pairs=len(pairs), valid_pairs=len(valid), excluded_pairs=len(pairs)-len(valid),
        pending_conditions=len(pending), zero_fallback_pairs=len(zero),
        zero_fallback_transparency_passes=sum(p.get('zero_fallback_transparency_passed', False) for p in zero),
        modified_fallback_active_pairs=len(activated),
        modified_fallback_active_but_all_physical_equal_pairs=[dict(model=p['model'], id=p['id'])
            for p in activated if p['all_physical_records_equal']],
        first_modified_fallback_total_action_ticks_unchanged_pairs=[dict(model=p['model'], id=p['id'])
            for p in activated if p['first_intervention_isolation']['original_action_ticks'] == p['first_intervention_isolation']['modified_action_ticks']],
        pre_intervention_isolation_violations=[dict(model=p['model'], id=p['id']) for p in activated
            if not p['first_intervention_isolation']['physical_before_intervention_equal']
            or not p['first_intervention_isolation']['executed_targets_through_intervention_equal']],
        changed_outcome_pairs=[dict(model=p['model'], id=p['id'], successes=p['successes'])
            for p in valid if p['successes'][0] != p['successes'][1]],
        first_activation_at_or_after_terminal_open_pairs=sum(
            (p['fallback15_stage']['first_fallback'] or {}).get('phase') == 'at_or_after_terminal_open_command' for p in activated),
        opening_command_interval_ticks_changed_pairs=[dict(model=p['model'], id=p['id'],
            original_ticks=p['opening_command_interval_comparison']['original']['physics_steps'],
            modified_ticks=p['opening_command_interval_comparison']['fallback15']['physics_steps'],
            fallback_actions=p['opening_command_interval_comparison']['fallback_actions']) for p in valid
            if p['opening_command_interval_comparison']['both_observed']
            and not p['opening_command_interval_comparison']['same_physics_ticks']])
    report = dict(complete=all_finished, generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        gpu_used=False, summary=summary, pairs=pairs, pending=pending, source_sha256=sources,
        reviewer_script_sha256=digest(SCRIPT), analysis_code_sha256={str(p): digest(p) for p in
            (SCRIPT, Path(phases.__file__), Path(phases.stage.__file__))}, limitations=[
            'A/B comparisons use the same checkpoint; the four-cell cross-model success summary is produced separately.',
            'fallback15 changes four qpos fallback constants only; no activation does not test release-time alignment.',
            'Branch activation, changed action physics duration, changed object trajectory and changed success are separate observations.',
            'Recorded initial physical equality does not prove equality of unrecorded contact/velocity caches.',
            'Source training panel includes outcome-based selection and cannot estimate population success rates.'])
    basename = 'independent_policy_final_review' if all_finished else 'independent_policy_live_review'
    atomic_json(args.root / (basename + '.json'), report)
    lines = ['# 模型 fallback15 独立审查' + ('（最终）' if all_finished else '（进行中快照）'), '',
        f'完成配对 {len(pairs)}，严格有效配对 {len(valid)}，待完成条件 {len(pending)}。'
        f'零触发配对 {len(zero)}，完整轨迹透明性通过 {summary["zero_fallback_transparency_passes"]}。', '',
        f'有效配对中的成功/失败翻转数为 {len(summary["changed_outcome_pairs"])}；'
        f'首次干预前的动作/物理隔离违规数为 {len(summary["pre_intervention_isolation_violations"])}。'
        f'{len(activated)} 对触发修改，其中 {summary["first_activation_at_or_after_terminal_open_pairs"]} 对首次触发位于终末打开命令到达时或之后。', '',
        'fallback 分支触发、整步物理时长改变、物理轨迹改变和任务结果改变分别记录。'
        '本次仅将四处 qpos TOPP 失败回退常数从50改为15；正常 TOPP 时长未直接修改。'
        '因此不能将本实验称为完整时序对齐，也不能用零触发案例否定释放时长假设。', '',
        '| 模型 | 案例 | 原→15结果 | fallback数原/15 | 首次15触发动作 | 首次物理差异后动作 | 首触发阶段 |',
        '|---|---|---|---:|---:|---:|---|']
    for p in pairs:
        if not p['valid_pair']:
            lines.append(f'| {p["model"]} | {p["id"]} | 无效配对：{p["statuses"]} | — | — | — | — |')
            continue
        trigger = p['first_modified_fallback_action_index']
        first = p['fallback15_stage']['first_fallback'] or {}
        outcomes = '→'.join('成功' if x else '失败' for x in p['successes'])
        lines.append(f'| {p["model"]} | {p["id"]} | {outcomes} | {p["fallback_actions"][0]}/{p["fallback_actions"][1]} | '
            f'{trigger+1 if trigger is not None else "—"} | {p["first_physical_difference_after_executed_step"]} | {first.get("phase", "未触发")} |')
    lines += ['', '表中动作按1起始执行step计；JSON中的action_index为0起始。'
        '有效配对要求原始输入、物理记录、指令、权重、源state/JPEG与专家准入全部通过。'
        '释放后触发描述干预位置，不证明任务已经无法恢复；零触发结果不能评价释放时长修复效果。']
    lines += ['', '终末打开命令区间的物理步数发生变化：', '',
        '| 模型 | 案例 | 原控制器ticks | fallback15 ticks | 区间内fallback次数原/15 |',
        '|---|---|---:|---:|---:|']
    for row in summary['opening_command_interval_ticks_changed_pairs']:
        lines.append(f'| {row["model"]} | {row["id"]} | {row["original_ticks"]} | {row["modified_ticks"]} | '
            f'{row["fallback_actions"][0]}/{row["fallback_actions"][1]} |')
    lines += ['', '该区间取首次终末臂闭合后，首次命令>.8之前最近的命令≤.2边界到>.8边界；'
        '累计场景实际物理步。它是命令阈值区间，不是接触持续时间或实际夹爪开合测量。']
    lines += ['', '这是按历史成败挑选的训练场景诊断面板，不能估计总体成功率。'
        '此处按同一权重的原始/修改控制器严格配对；跨权重比较须另取四格共同有效分母。'
        '完整源文件与分析脚本SHA256、排除条件、逐例轨迹比较和窗口起止几何见同名JSON。']
    (args.root / (basename + '.md')).write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=all_finished, **summary), ensure_ascii=False))


if __name__ == '__main__':
    main()
