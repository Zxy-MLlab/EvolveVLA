#!/usr/bin/env python3
"""CPU-only snapshot of horizon outcomes, predicates and paired isolation checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen1_execution_horizon_ablation_20261019 as horizon
import gen1_expert_label_replay_20261019 as oracle
from gen1_mouse_rollout_stages import stages as mouse_stages


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def functional_position(pose, config):
    import numpy as np
    q = np.asarray(pose['quaternion'], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    local = np.asarray(config['functional_matrix'][0], dtype=float)[:3, 3] * config['scale']
    return np.asarray(pose['position']) + rotation @ local


def mug_predicates(result, physical):
    import numpy as np
    model = result['binding']['fields']['{A}'].split('/')[-1].removeprefix('base')
    assets = horizon.ROOT / 'robotwin_repo_ba0fd21/assets/objects'
    mug = json.loads((assets / f'039_mug/model_data{model}.json').read_text())
    rack = json.loads((assets / '040_rack/model_data0.json').read_text())
    point = functional_position(physical['mug'], mug)
    rack_point = functional_position(physical['rack'], rack)
    middle = (rack_point + np.asarray(physical['rack']['position'])) / 2
    delta = point - middle
    return dict(functional_xy=bool(np.all(np.abs(delta[:2]) < .02)),
                functional_height=bool(point[2] > .86),
                right_open=physical['right']['drive_targets'][-1] > .8), dict(
        mug_functional_position=point.tolist(), rack_middle_position=middle.tolist(),
        functional_xy_error=delta[:2].tolist(), required_mug_functional_z=.86)


def command_events(rows, side):
    values = [row['physical'][side]['drive_targets'][-1] for row in rows]
    closed = [i for i in range(1, len(values)) if values[i-1] >= .2 and values[i] < .2]
    opened = [i for i in range(1, len(values)) if values[i-1] <= .8 and values[i] > .8]
    return dict(closure_steps=closed, opening_steps=opened,
                opening_after_closure=[i for i in opened if any(c < i for c in closed)])


def classify(result, rows):
    import numpy as np
    task = result['task']
    actor = 'mouse' if task == 'place_mouse_pad' else 'mug'
    positions = np.asarray([row['physical'][actor]['position'] for row in rows])
    trailing = positions[-min(100, len(positions)):]
    record = dict(steps=result['steps'], official_success=result['success'],
        left_commands=command_events(rows, 'left'), right_commands=command_events(rows, 'right'),
        max_height_above_initial=float(positions[:, 2].max()-positions[0, 2]),
        maximum_actor_step_displacement=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).max()) if len(rows)>1 else 0.,
        final_actor_position=positions[-1].tolist(),
        last100_actor_coordinate_range=np.ptp(trailing, axis=0).tolist())
    if task == 'place_mouse_pad':
        stage = mouse_stages(rows, result)
        record.update(active_arm=stage['active_arm'], terminal_predicates=stage['terminal_predicates'],
            terminal_xy_error=stage['terminal_xy_error'], openings=stage['openings'],
            reconstructed_success=stage['reconstructed_terminal_success'])
        active = record[stage['active_arm'] + '_commands']
    else:
        predicates, geometry = mug_predicates(result, rows[-1]['physical'])
        record.update(terminal_predicates=predicates, geometry=geometry,
                      reconstructed_success=all(predicates.values()))
        active = record['right_commands']
        if active['closure_steps']:
            close = active['closure_steps'][0]
            record['right_phase_max_height_gain_from_closure'] = float(positions[close:, 2].max()-positions[close, 2])
    record['failed_predicates'] = [key for key, value in record['terminal_predicates'].items() if not value]
    record['predicate_reconstruction_agrees'] = bool(record['reconstructed_success']) == bool(result['success'])
    release = active['opening_after_closure']
    side = record['active_arm'] if task == 'place_mouse_pad' else 'right'
    tracking = np.asarray([np.abs(np.asarray(row['physical'][side]['measured_qpos']) -
        np.asarray(row['physical'][side]['drive_targets'][:6])).mean() for row in rows])
    sustained = next((i for i in range(len(tracking)-2) if np.all(tracking[i:i+3] > .01)), None)
    record['active_arm_tracking'] = dict(side=side,
        first_three_consecutive_boundaries_mae_above_001rad=sustained,
        mean_abs_radians=float(tracking.mean()))
    if release:
        first_release = release[0]
        record['active_arm_tracking'].update(first_terminal_opening_step=first_release,
            mean_before_opening=float(tracking[:first_release].mean()),
            mean_last50_before_opening=float(tracking[max(0,first_release-50):first_release].mean()),
            mean_after_opening=float(tracking[first_release:].mean()))
    if result['success']:
        label = 'completed_success'
    elif not active['closure_steps']:
        label = 'no_terminal_arm_closure_observed'
    elif record['max_height_above_initial'] < .04:
        label = 'closure_without_observed_object_lift'
    elif not release:
        label = 'lift_or_transfer_observed_without_terminal_opening'
    elif max(record['last100_actor_coordinate_range']) < .002 and result['steps'] - release[0] >= 100:
        label = 'released_outside_success_tolerance_then_no_visible_correction'
    else:
        label = 'terminal_opening_observed_but_success_predicate_unmet'
    record['descriptive_classification'] = label
    return record


def source_initial_parity(case, directory):
    import h5py
    import numpy as np
    images = {key: np.load(directory / 'initial' / f'{key}.npy') for key in oracle.CAMERAS}
    reset = np.load(directory / 'initial/state.npy')
    with h5py.File(case['source_hdf5']) as source:
        state = np.concatenate([source[f'state/{key}'][0] for key in oracle.FIELDS])
        image_metrics = oracle.source_image_metrics(source, 0, images)
    return dict(state_max_abs=float(np.abs(reset-state).max()), images=image_metrics,
                all_source_reencoded_jpeg_equal=all(v['jpeg_bytes_equal'] for v in image_metrics.values()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    records, isolation, sources = [], [], {}
    for case in manifest['cases']:
        available = {}
        for prefix in manifest['prefixes']:
            directory = args.run / case['id'] / f'prefix_{prefix}'
            result_path = directory / 'result.json'
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            sources[str(result_path)] = horizon.digest(result_path)
            row = dict(id=case['id'], task=case['task'], prefix=prefix,
                       status=result['status'], success=result.get('success'),
                       inference_count=result.get('inference_count'),
                       physical_tracking=result.get('physical_tracking'))
            if result['status'] == 'completed':
                physical = read_lines(directory / 'physical.jsonl')
                actions = read_lines(directory / 'actions.jsonl')
                row['source_initial_parity'] = source_initial_parity(case, directory)
                row['stages'] = classify(result, physical)
                available[prefix] = dict(result=result, physical=physical, actions=actions)
                for filename in ('physical.jsonl', 'actions.jsonl'):
                    path = directory / filename
                    sources[str(path)] = horizon.digest(path)
            records.append(row)
        for prefix, current in available.items():
            if prefix == 50 or 50 not in available:
                continue
            baseline = available[50]
            common = min(prefix, current['result']['steps'], baseline['result']['steps'])
            initial_equal = current['result']['hashes'] == baseline['result']['hashes']
            initial_physical_equal = current['physical'][0] == baseline['physical'][0]
            first_prediction_equal = current['actions'][0]['actions'] == baseline['actions'][0]['actions']
            common_physical_equal = current['physical'][:common+1] == baseline['physical'][:common+1]
            isolation.append(dict(id=case['id'], prefix=prefix, initial_inputs_equal=initial_equal,
                initial_physical_equal=initial_physical_equal, first_full50_prediction_equal=first_prediction_equal,
                common_action_count=common, physical_through_first_prefix_equal=common_physical_equal))
    artifact = dict(gpu_used=False, run=str(args.run.resolve()),
        run_manifest_sha256=horizon.digest(args.run / 'manifest.json'),
        audit_script_sha256=horizon.digest(__file__), source_sha256=sources,
        attempted=len(records), planned=len(manifest['cases']) * len(manifest['prefixes']),
        records=records, isolation=isolation,
        limitations=['Classification describes recorded object motion and command events, not grasp contact or causal dynamics.',
            'Mug functional points reconstructed from recorded actor pose and original asset config; simulator result remains authoritative.',
            'Only completed results appear; missing/pending cases are not counted as failures.',
            'An outcome-stratified smoke panel cannot estimate overall success rates.'])
    horizon.atomic_json(args.output, artifact)
    lines = ['# 执行窗口冒烟实验：物理阶段审计', '',
        f'已记录 {len(records)}/{artifact["planned"]} 个条件；只读分析，没有新增模型推理。', '',
        '| 案例 | 执行窗口 | 状态 | 结果/阶段 |', '|---|---:|---|---|']
    for row in records:
        stage = row.get('stages', {})
        detail = stage.get('descriptive_classification', '专家检查未通过或执行异常，不能作为策略失败')
        if stage.get('failed_predicates'):
            detail += '; 未满足=' + ', '.join(stage['failed_predicates'])
        if 'terminal_xy_error' in stage:
            detail += '; xy误差mm=' + ', '.join(f'{v*1000:.2f}' for v in stage['terminal_xy_error'])
        lines.append(f'| {row["id"]} | {row["prefix"]} | {row["status"]} | {detail} |')
    lines += ['', '每个已完成条件均核对源 HDF5 首帧 state、三相机 JPEG，以及条件间首个完整50步预测和首次重规划前物理轨迹。具体布尔检查与阶段数据见同名 JSON。',
              '', '夹爪命令事件和物体抬升不等于已经测量真实接触；分类是行为描述。初始面板按旧结果分层选取，不能据此估计总体成功率。']
    args.markdown.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(attempted=len(records), planned=artifact['planned'],
                          isolation_checks=len(isolation), output=str(args.output))))


if __name__ == '__main__':
    main()
