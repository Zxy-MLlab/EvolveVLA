#!/usr/bin/env python3
"""CPU artifact audit; never imports or changes simulator or replay sources."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OUT = ROOT / 'outputs/gen1_control_repair_20261020/recovery_suffix_replay_dev3'
CAMERAS = ('head_camera', 'left_camera', 'right_camera')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def raw(row):
    path = Path(row['path'])
    if sha(path) != row['sha256']:
        raise ValueError('Raw hash mismatch: ' + str(path))
    with path.open('rb') as stream:
        value = pickle.load(stream)
    drive = np.asarray(value['joint_action']['vector'])
    if drive.shape != (14,) or not np.isfinite(drive).all():
        raise ValueError('Invalid native14D drive target')
    for camera in CAMERAS:
        image = np.asarray(value['observation'][camera]['rgb'])
        if image.shape != (240, 320, 3) or image.dtype != np.uint8:
            raise ValueError('Invalid native RGB')
    return value, drive


def check_case(case, mode):
    directory = OUT / case['id'] / mode
    if not (directory / 'result.json').exists():
        return dict(mode=mode, finalized=False)
    result = read(directory / 'result.json')
    if result.get('status') not in ('dense_replay_completed', 'sparse_replay_completed'):
        return dict(mode=mode, finalized=True, status=result.get('status'), success=result.get('success'),
                    eligible=False, passed=None, exclusions=[result.get('status')],
                    expert_gate=result.get('expert_gate'), error=result.get('error'),
                    simulator_returncode=result.get('simulator_returncode'))
    saved_case = read(directory / 'case.json')
    failures = []

    def check(name, value):
        if not value:
            failures.append(name)

    check('success_boolean', type(result.get('success')) is bool)
    check('simulator_clean_exit', result.get('simulator_returncode') == 0)
    check('saved_prefix_definition', all(saved_case[k] == case[k] for k in
        ('prefix_targets', 'prefix_physical', 'prefix_physics_steps', 'takeover_boundary_step',
         'source_input_hashes', 'task', 'seed', 'episode_index', 'prompt')))
    prefix = rows(directory / 'prefix_verification.jsonl')
    boundary = case['takeover_boundary_step']
    check('prefix_boundaries', [r['step'] for r in prefix] == list(range(boundary + 1)) + [boundary])
    check('prefix_physical', all(r['physical'] == case['prefix_physical'][r['step']]
                                 and r['exact'] is True and not r['differences'] for r in prefix))
    check('takeover_state', result.get('before_physical') == case['prefix_physical'][boundary])
    initial = read(directory / 'initial/hashes.json')
    check('initial_hash_record', initial == case['source_input_hashes'])
    state = np.load(directory / 'initial/state.npy', allow_pickle=False)
    check('initial_state_bytes', hashlib.sha256(np.asarray(state, dtype='<f4').reshape(14).tobytes()).hexdigest() == initial['state'])
    for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
        image = np.load(directory / 'initial' / (camera + '.npy'), allow_pickle=False)
        check('initial_camera_' + camera, list(image.shape) == initial[camera]['shape']
              and hashlib.sha256(np.ascontiguousarray(image, dtype=np.uint8).tobytes()).hexdigest() == initial[camera]['sha256'])
    check('initial_prompt', hashlib.sha256(case['prompt'].encode()).hexdigest() == initial['prompt'])
    check('expert_gate', result.get('expert_gate', {}).get('status') == 'passed')
    hold = result.get('terminal_hold', {})
    if hold.get('physics_ticks'):
        hold_path = directory / ('terminal_hold_predicates.jsonl' if mode == 'dense' else 'expert_suffix/terminal_hold_predicates.jsonl')
        checks = rows(hold_path)
        check('hold250_real_tick_records', hold['physics_ticks'] == hold['requested_ticks'] == 250
              and [r['physics_tick'] for r in checks] == list(range(hold['start_tick'] + 1, hold['start_tick'] + 251)))
        check('hold_status', all(r['success'] is True for r in checks) == hold['all_ticks_success'])
        check('hold_positions_preserved', all(hold['before_physical'][arm]['drive_targets'] ==
              hold['after_physical'][arm]['drive_targets'] for arm in ('left', 'right')))
        check('hold_no_measured_reset', hold.get('measured_state_reset') is False)
        check('hold_final_accounting', result['physics_ticks'] == hold['start_tick'] + hold['physics_ticks'])
    report = dict(mode=mode, finalized=True, status=result['status'], success=result['success'],
        physics_ticks=result.get('physics_ticks'), terminal_hold_ticks=hold.get('physics_ticks'),
        stable_success=result.get('stable_success'))
    if mode == 'dense':
        source = Path(case['recovery_source_directory']) / 'expert_suffix'
        expected = read(source / 'result.json')
        native = rows(directory / 'expert_suffix/captures.jsonl')
        original = rows(source / 'captures.jsonl')
        check('native_capture_count', len(native) == len(original))
        for index, (a, b) in enumerate(zip(original, native)):
            check(f'native_capture_meta_{index}', all(a[k] == b[k] for k in ('physics_tick', 'physical', 'success', 'components')))
            x, xd = raw(a)
            y, yd = raw(b)
            check(f'native_capture_drive_{index}', np.array_equal(xd, yd))
            check(f'native_capture_rgb_{index}', all(np.array_equal(x['observation'][c]['rgb'], y['observation'][c]['rgb']) for c in CAMERAS))
        check('dense_source_end', result['source_end_physics_tick'] == expected['physics_ticks']
              and result['source_end_physical'] == expected['after_physical']
              and result['source_end_final_success'] == expected['final_success'])
        check('dense_hold_after_source_end', not hold.get('physics_ticks') or hold['start_tick'] == expected['physics_ticks'])
        uniform = rows(directory / 'uniform_captures.jsonl')
        check('uniform_manifest_hash', sha(directory / 'uniform_captures.jsonl') == result['uniform_captures_sha256'])
        regular = [r for r in uniform if r['full_interval_boundary']]
        check('uniform_global15', [r['physics_tick'] for r in regular] == list(range(0, result['physics_ticks'] + 1, 15)))
        check('uniform_interval_count', result['uniform_full_intervals'] == len(regular) - 1)
        partial = [r for r in uniform if not r['full_interval_boundary']]
        check('partial_tail_only', not partial or len(partial) == 1 and partial[0] is uniform[-1]
              and partial[0]['partial_final_interval'] is True and partial[0]['physics_tick'] == result['physics_ticks']
              and 0 < partial[0]['physics_tick'] - regular[-1]['physics_tick'] < 15)
        for row in uniform:
            raw(row)
        report.update(source_captures=len(native), full_intervals=len(regular) - 1,
                      partial_tail_ticks=partial[0]['physics_tick'] - regular[-1]['physics_tick'] if partial else 0)
    else:
        uniform = rows(OUT / case['id'] / 'dense/uniform_captures.jsonl')
        regular = [r for r in uniform if r['full_interval_boundary']]
        actions = rows(directory / 'expert_suffix/actions.jsonl')
        check('target_count_excludes_partial', result['target_count'] == len(regular) - 1)
        check('sparse_action_count', len(actions) == result['actions_executed'])
        for index, action in enumerate(actions):
            _, drive = raw(regular[index + 1])
            check(f'sparse_label_{index}', action['index'] == index and action['source_tick'] == regular[index + 1]['physics_tick']
                  and np.array_equal(action['target'], drive))
        action_ticks = sum(r['physics_ticks'] for r in actions)
        check('sparse_tick_accounting', action_ticks + hold.get('physics_ticks', 0) == result['physics_ticks'])
        check('sparse_official_early_stop', not result['official_rollout_success'] or actions[-1]['success'] is True
              and not any(a['success'] is True for a in actions[:-1]))
        check('sparse_hold_after_stop', not hold.get('physics_ticks') or hold['start_tick'] == action_ticks)
        check('sparse_action_budget', result['actions_executed'] + result['prefix_action_count'] <= result['official_step_limit'])
        report.update(actions_executed=len(actions), target_count=result['target_count'], official_success=result['official_rollout_success'])
    report.update(eligible=not failures, passed=not failures, failures=failures)
    return report


def main():
    manifest = read(OUT / 'manifest.json')
    source_errors = [p for p, h in manifest['source_sha256'].items() if sha(p) != h]
    modes = ['dense', *manifest['sparse_modes']]
    cases = {case['id']: [check_case(case, mode) for mode in modes] for case in manifest['cases']}
    if (OUT / 'summary.json').exists():
        for records in cases.values():
            if records[0].get('success') is not True or records[0].get('passed') is not True:
                for record in records[1:]:
                    if not record['finalized']:
                        record.update(finalized=True, eligible=False, passed=None,
                            status='not_attempted_dense_prerequisite_failed',
                            exclusions=['not_attempted_dense_prerequisite_failed'])
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), gpu_used=False,
        source_errors=source_errors, cases=cases,
        limitations=['Source review confirms one scene.step per explicit terminal-hold tick; observed records do not prove all hidden physics states.',
                     'Native saved-source capture equality is checked before the separately recorded250tick hold.',
                     'Sparse replay stops at official success and does not certify replay of unexecuted later labels.',
                     'Artifact audit is not a learned-policy evaluation or a claim of recovery generalization.'])
    (OUT / 'independent_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    md = ['# 恢复后缀重放：独立CPU审计', '', '更新时间：' + report['created_utc'], '',
          '| 案例 | 模式 | 完成 | 成功 | 250tick稳定 | 独立检查 |', '|---|---|---|---|---|---|']
    for name, records in cases.items():
        for record in records:
            md.append('| ' + ' | '.join(str(value) for value in (name, record['mode'], record['finalized'],
                record.get('success', '待完成'), record.get('stable_success', '待完成'),
                record.get('failures', record.get('exclusions', '待完成')))) + ' |')
    md += ['', '源文件冻结检查：' + ('通过' if not source_errors else repr(source_errors)), '',
           '已重读初始NPY、每个原控制前缀物理边界、原生source capture的RGB/14D drive/物理/谓词、uniform全局15tick间隔、末partial剔除，以及sparse标签/预算/提前成功和独立250tick hold。', '',
           'hold保持命令位置并置命令速度为零，不重置实测状态；原生source精确匹配先于hold。sparse按官方成功提前终止，未执行的后续标签没有获得重放认证。这是恢复轨迹可行性检查，不是学习后策略的泛化评估。', '']
    (OUT / 'independent_audit.md').write_text('\n'.join(md))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, default=OUT)
    OUT = parser.parse_args().input.resolve()
    main()
