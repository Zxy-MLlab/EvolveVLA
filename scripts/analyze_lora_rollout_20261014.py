"""Read saved evaluations and produce a diagnostic snapshot; never touch live jobs."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import statistics

from gen1_recovery_compare import summarize

ROOT = Path('/root/autodl-tmp/EvolveVLA')
PANELS = {
    6000: 'gen1_replay_lora6000_clean_pairs_20261013',
    18000: 'gen1_replay_lora18000_clean_pairs_20261014',
}
KEYS = ('joint_mae', 'gripper_mae', 'first10_joint_mae', 'first10_gripper_mae')


def read(path):
    return json.loads(path.read_text())


def physical_summary(directory, record):
    rows = [json.loads(line) for line in (directory/'physical.jsonl').open()]
    obj = 'mouse' if record['task'] == 'place_mouse_pad' else 'mug'
    initial, final = rows[0]['physical'], rows[-1]['physical']
    result = dict(directory=str(directory), seed=record['seed'],
                  success=record['success'], steps=record['steps'],
                  max_lift_cm=100*max(r['physical'][obj]['position'][2]
                      - initial[obj]['position'][2] for r in rows),
                  initial_position=initial[obj]['position'],
                  final_position=final[obj]['position'])
    if obj == 'mouse':
        delta = [final[obj]['position'][i]-final['target']['position'][i] for i in (0, 1)]
        q = list(map(abs, final[obj]['quaternion']))
        result.update(final_xy_error_cm=[100*x for x in delta],
                      final_position_pass=abs(delta[0]) < .015 and abs(delta[1]) < .012,
                      final_orientation_pass=abs(q[2]*q[3]-.49) < .015
                          or abs(q[0]*q[1]-.49) < .015)
    else:
        # Recorded gripper value is the robot controller's normalized command,
        # NOT a measurement of the finger opening or grasp contact.
        commands = [(r['step'], r['physical']['right']['drive_targets'][-1]) for r in rows]
        close = next((s for s, v in commands if v < .2), None)
        reopened = next((s for s, v in commands if close is not None and s > close and v > .5), None)
        result.update(right_controller_close_step=close,
                      right_controller_reopen_above_half_step=reopened,
                      closest_expert_middle_xy_cm=100*min(math.dist(
                          r['physical'][obj]['position'][:2], [0, -.15]) for r in rows),
                      object_position_at_step300=next((r['physical'][obj]['position']
                          for r in rows if r['step'] == 300), None))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    evidence = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    created_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(), evaluations={},
                    validation=[], cross_checkpoint=[], physical={})
    candidate_maps = {}
    for step, name in PANELS.items():
        root = ROOT/'outputs'/name
        panels = {}
        directories = {}
        for label in ('original', 'candidate'):
            records = []
            for path in sorted(root.glob(f'round_*_{label}/*/result.json')):
                record = read(path)
                records.append(record)
                if label == 'candidate':
                    directories[(record['task'], record['seed'])] = path.parent
            assert all(r['status'] in ('completed', 'expert_rejected') for r in records), records
            panels[label] = dict(records=records)
        result = summarize(panels['original'], panels['candidate'])
        result['final'] = (root/'comparison.json').exists()
        result['source'] = str(root)
        evidence['evaluations'][step] = result
        candidate_maps[step] = {(r['task'], r['seed']): r for r in panels['candidate']['records']}
        # Physical counts only include VALID original/candidate pairs. Excluded
        # candidate successes never get silently included in these statistics.
        physical = {}
        for task, data in result['tasks'].items():
            physical[task] = [physical_summary(directories[(task, pair['seed'])],
                candidate_maps[step][(task, pair['seed'])]) for pair in data['pairs']]
        evidence['physical'][step] = physical
    for key in sorted(candidate_maps[6000].keys() & candidate_maps[18000].keys()):
        a, b = candidate_maps[6000][key], candidate_maps[18000][key]
        if a['status'] == b['status'] == 'completed':
            evidence['cross_checkpoint'].append(dict(task=key[0], seed=key[1],
                initial_input_hashes_equal=a['hashes'] == b['hashes'],
                prompt_binding_equal=a['binding'] == b['binding'],
                success6000=a['success'], success18000=b['success']))
    first = ROOT/'outputs/gen1_replay_lora_b32_20261014'
    base = read(first/'validation_0000.json')['metrics']
    for directory in (first, ROOT/'outputs/gen1_replay_lora_b32_extended24000_20261014'):
        for path in sorted(directory.glob('validation_*.json')):
            data = read(path)
            if data['step'] == 0:
                continue
            metrics = data['metrics']
            tasks = sorted(set(metrics)-{'all'})
            evidence['validation'].append(dict(step=data['step'], source=str(path),
                global_task_component_ratio=statistics.mean(
                    metrics[t][k]/max(base[t][k], 1e-8) for t in tasks for k in KEYS),
                task_metrics={t:metrics[t] for t in ('place_mouse_pad', 'hanging_mug')}))
    evidence['validation'].sort(key=lambda r:r['step'])
    evidence['limitations'] = [
        'Two controlled diagnostic tasks, 10 valid paired trials per task; not official 50-task average.',
        'Snapshot may precede final completion of the running 18000-step evaluation.',
        'Initial input hashes do not certify equality of all hidden simulator physics state.',
        'Object origin is not the mug functional point; do not use its height to replace official success.',
        'Earlier opening and position changes are observations, not an isolated causal intervention.',
        'Successful demonstrations and lower MAE do not establish recovery from policy-induced errors.',
    ]
    (args.output/'evidence.json').write_text(json.dumps(evidence, indent=2))
    lines = ['# LoRA rollout 诊断快照', '', f"北京时间: {evidence['created_beijing']}", '',
        '结论：离线专家动作误差下降，闭环末段定位与释放表现却下降。不能据此断言普通验证集过拟合，也没有证据把问题归因于评估次数。', '',
        '| 更新步数 | 18任务离线综合误差比值（初始=1） | 鼠标实际成功 | 挂杯实际成功 | 评估完整 |',
        '|---|---:|---:|---:|---|']
    for step in (6000, 18000):
        panel = evidence['evaluations'][step]
        v = next(r for r in evidence['validation'] if r['step'] == step)
        cells = [f"{panel['tasks'][t]['candidate_successes']}/{panel['tasks'][t]['paired_completed']}" for t in ('place_mouse_pad', 'hanging_mug')]
        lines.append(f"| {step} | {v['global_task_component_ratio']:.4f} | {cells[0]} | {cells[1]} | {panel['final']} |")
    latest = evidence['evaluations'][18000]
    original_success = sum(t['original_successes'] for t in latest['tasks'].values())
    candidate_success = sum(t['candidate_successes'] for t in latest['tasks'].values())
    pairs = sum(t['paired_completed'] for t in latest['tasks'].values())
    lines += ['', f'18000步本轮有效配对：微调前 {original_success}/{pairs}，微调后 {candidate_success}/{pairs}。',
        '', '已核实的具体失败：', '',
        '- 鼠标 seed 1301000：6000步成功；18000步已抓起并放下，但最终y误差1.56cm，超过官方1.2cm容差。',
        '- 鼠标 seed 1301003：6000步成功；18000步最终y误差3.76cm，位置和姿态均不满足。',
        '- 18000步鼠标8次失败中，2次没有明显抬起（最大抬升不足2cm）；其余6次抬起后最终位置不合格。这是描述性阶段分类。',
        '- 挂杯 seed 1301007/1301009：两个checkpoint的初始输入和prompt相同；6000步成功，18000步均失败。都完成了前半段搬运和右手抬起，差异出现在接近杯架、释放阶段。',
        '- 18000步挂杯的右夹爪控制值更早超过半开，且第300步杯子位置与6000步明显不同。控制值不等于实际接触状态，不单独证明提前释放是唯一原因。', '',
        '训练目标和数据的限制：', '',
        '- 当前1:1官方/本地数据均匀按18任务采样；本地训练数据仅包含成功、规划成功、重放成功的专家轨迹。指令变体并未提供失败状态的纠偏标签。',
        '- validation使用720个本地留出专家状态；当前选checkpoint依据18任务平均动作MAE，不使用实际rollout成功率。因此它会偏好离线误差更小而实际失败更多的checkpoint。',
        '- 每次推理完整执行50步动作才重看图像，与既有协议一致。精细操作中可能积累误差；这不是新发现的协议错误，未经独立对照不能宣布它是主因。',
        '- 当前LoRA冻结全部非adapter参数；上游便捷freeze_filter会保留部分视觉/投影层可训练。应作独立消融，不能直接断言解冻必定改善。',
        '- 原模型在这两个任务上的成功率也很低；公开50任务82.7%不能直接与两任务数字对比，原始基线仍需独立核查。', '',
        '需要验证的改进方向：先用训练种子构建任务末段/纠偏监督；保留官方replay和原norm；独立评估开发与确认种子；比较checkpoint以实际成功率为准。保持正在运行的训练和正式配对协议。', '',
        '以上为诊断，没有改动当前训练、权重、归一化或成功判定。详细每种子证据见 evidence.json。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(args.output/'report.md')


if __name__ == '__main__':
    main()
