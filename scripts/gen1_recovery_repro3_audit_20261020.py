#!/usr/bin/env python3
"""Independently admit at most one source after all three fixed repeats finish."""
from collections import Counter
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import gen1_recovery_suffix_independent_audit_20261020 as replay
import gen1_recovery_suffix_dataset_20261020 as dataset

ROOT = replay.ROOT
OUT = ROOT / 'outputs/gen1_control_repair_20261020/recovery_repro3'


def main():
    protocol = replay.read(OUT / 'protocol.json')
    selection = replay.read(OUT / 'selection_rule.json')
    protocol_checks = dict(fixed_three=protocol.get('repeat_count') == 3,
        no_success_stop=protocol.get('stop_on_success') is False,
        frozen_runner=replay.sha(ROOT / 'scripts/gen1_recovery_suffix_replay_20261020.py') == protocol['runner_sha256'],
        frozen_source_manifest=replay.sha(Path(protocol['source_run']) / 'manifest.json') == protocol['source_manifest_sha256'])
    records, group_keys = [], []
    for index in range(1, 4):
        directory = OUT / f'repeat_{index}'
        if not (directory / 'manifest.json').exists():
            records.append(dict(repeat=index, attempted=False, complete=False, eligible=False))
            continue
        replay.OUT = directory
        with contextlib.redirect_stdout(io.StringIO()):
            replay.main()
        audited = replay.read(directory / 'independent_audit.json')
        manifest = replay.read(directory / 'manifest.json')
        case = manifest['cases'][0]
        conditions = audited['cases'][protocol['case']]
        complete = (directory / 'summary.json').exists() and all(c['finalized'] for c in conditions)
        group = {key: case[key] for key in ('source_hdf5_sha256', 'source_checkpoint_identity_sha256',
            'task', 'seed', 'episode_index', 'takeover_boundary_step')}
        group['source_suffix_files_sha256'] = hashlib.sha256(json.dumps(case['recovery_source_sha256'], sort_keys=True).encode()).hexdigest()
        group_keys.append(group)
        command = protocol['commands'][index - 1]
        command_match = (Path(command[command.index('--output') + 1]).resolve() == directory
            and command[command.index('--sparse-modes') + 1] == 'original'
            and command[command.index('--terminal-hold-ticks') + 1] == '250')
        eligible = bool(complete and command_match and not audited['source_errors']
            and all(c.get('passed') is True and c.get('success') is True for c in conditions))
        readiness = directory / protocol['case'] / 'readiness.json'
        builder_gate = None
        if eligible:
            arrays, row = dataset.validate_case(readiness, 'original')
            builder_gate = dict(dataset_ready=row['dataset_ready'], transitions=row['frames'],
                full_windows=row['valid_start_count'], parent_episode_uid=row['parent_episode_uid'])
            del arrays
            eligible &= builder_gate['dataset_ready']
        result_paths = list(directory.glob('*/*/result.json'))
        early_rule = all(datetime.fromisoformat(selection['created_utc']).timestamp() < p.stat().st_mtime for p in result_paths)
        records.append(dict(repeat=index, attempted=True, complete=complete, eligible=eligible,
            command_matches=command_match, source_errors=audited['source_errors'],
            selection_rule_recorded_before_results=early_rule, conditions=conditions,
            builder_gate=builder_gate, readiness=str(readiness) if readiness.exists() else None,
            audit_sha256=replay.sha(directory / 'independent_audit.json')))
    protocol_checks['same_source_group'] = bool(group_keys) and all(g == group_keys[0] for g in group_keys)
    protocol_checks['selection_rule_precedes_results'] = all(r.get('selection_rule_recorded_before_results', True) for r in records)
    complete = all(r['complete'] for r in records)
    passing = [r for r in records if r['eligible']]
    chosen = passing[0] if complete and all(protocol_checks.values()) and passing else None
    old = ROOT / 'outputs/gen1_control_repair_20261020/recovery_suffix_replay_dev3/hanging_mug_87749/original/result.json'
    old_result = replay.read(old)
    outcomes = Counter(c.get('status', 'pending') for r in records for c in r.get('conditions', []))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), gpu_used=False,
        protocol_sha256=replay.sha(OUT / 'protocol.json'), selection_rule_sha256=replay.sha(OUT / 'selection_rule.json'),
        protocol_checks=protocol_checks, planned_repeats=3, attempted_repeats=sum(r['attempted'] for r in records),
        complete=complete, passing_repeats=len(passing), condition_outcomes=dict(outcomes), records=records,
        selected_repeat=chosen['repeat'] if chosen else None, selected_readiness=chosen['readiness'] if chosen else None,
        admitted_source_episodes=1 if chosen else 0,
        historical_original_rejection=dict(path=str(old), sha256=replay.sha(old), status=old_result['status'],
            success=old_result['success'], retained=True, excluded_from_three_repeat_denominator=True),
        limitations=['Three repeats are one original source episode and one recovery suffix, never three independent training episodes.',
            'Lowest numbered complete pass is selected only after all three planned attempts finish; every failed/rejected/unattempted condition remains visible.',
            'This verifies feasibility of an expert recovery demonstration with the original controller; it is not learned-policy improvement or generalization evidence.'])
    (OUT / 'independent_selection.json').write_text(json.dumps(report, indent=2) + '\n')
    md = ['# 固定3重复：恢复示范独立准入审计', '', '更新时间：' + report['created_utc'], '',
        '| 重复 | 全部结束 | 完整通过 | Dense | Original sparse |', '|---|---|---|---|---|']
    for record in records:
        values = record.get('conditions', [])
        md.append(f'| {record["repeat"]} | {record["complete"]} | {record["eligible"]} | '
            + ' | '.join(c.get('status', 'pending') for c in values) + ('待开始 | 待开始' if not values else '') + ' |')
    md += ['', f'协议核验：{protocol_checks}', '',
        f'预定3次全部结束：{complete}；通过重复数：{len(passing)}；按预定编号选择：{report["selected_repeat"]}；只准入{report["admitted_source_episodes"]}条独立source episode。', '',
        f'选用readiness：{report["selected_readiness"]}', '',
        '旧original专家拒绝记录保留，未覆盖且不作为新3次试验的替代。每次均复核原控制prefix、source captures RGB/drive/物理、均匀15tick标签及末partial剔除、官方成功早停与额外250tick连续成功hold。', '',
        '这只证明一份专家纠偏示范可执行；不能报告为新策略成功率、泛化提升或3条新episode。', '']
    (OUT / 'independent_selection.md').write_text('\n'.join(md))
    print(json.dumps({k: report[k] for k in ('protocol_checks', 'attempted_repeats', 'complete', 'passing_repeats',
        'condition_outcomes', 'selected_repeat', 'selected_readiness', 'admitted_source_episodes')}, indent=2))


if __name__ == '__main__':
    main()
