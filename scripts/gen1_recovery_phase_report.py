#!/usr/bin/env python3
"""Explain existing held-out errors by source gripper transitions, CPU only.

Phase labels are diagnostics, never checkpoint-selection or success predicates.
All within-episode anchors are pooled before reporting across episodes.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path('/root/autodl-tmp/EvolveVLA')
METRICS = ('joint_mae', 'gripper_mae', 'first10_joint_mae', 'first10_gripper_mae')


def mean_rows(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in METRICS}


def main():
    manifest = json.loads((ROOT/'outputs/gen1_diagnostic_prepare_cpu_v3_20261011/manifest.json').read_text())
    anchors = {a['anchor_id']: a for a in manifest['anchors'] if a['split'] == 'val'}
    reports = {label: json.loads((ROOT/f'outputs/gen1_recovery_padding_{label}_20261012/report.json').read_text())
               for label in ('original', 'augmented', 'coverage')}
    if len({r['observation_sha256'] for r in reports.values()}) != 1:
        raise ValueError('Audit observations differ')
    if any({row['anchor_id'] for row in r['records']} != set(anchors) for r in reports.values()):
        raise ValueError('Audit anchor sets differ')
    sources = {}
    labels = {}
    for anchor_id, anchor in anchors.items():
        path = Path(anchor['source'])
        if path not in sources:
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != anchor['source_sha256']:
                    raise ValueError('Source trajectory changed')
            with h5py.File(path) as f:
                sources[path] = np.concatenate([f[f'action/{arm}_ee_joint_states'][:]
                                               for arm in ('left', 'right')], axis=1)
        grippers = sources[path]
        if grippers.shape != (anchor['length'], 2):
            raise ValueError('Unexpected source gripper shape')
        start = anchor['frame']
        stop = min(start+50, anchor['length'])
        future = grippers[start:stop]
        delta = np.diff(future, axis=0)
        closing = bool(np.any(delta < -0.05))
        opening = bool(np.any(delta > 0.05))
        if closing and opening:
            state = 'closing_and_opening'
        elif closing:
            state = 'closing'
        elif opening:
            state = 'opening'
        else:
            state = 'no_transition'
        fraction = start/max(anchor['length']-1, 1)
        phase = 'early' if fraction < 0.2 else 'late' if fraction >= 0.8 else 'middle'
        labels[anchor_id] = dict(task=anchor['task'], episode=anchor['episode'],
            frame=start, phase=phase, gripper_event=state, valid_frames=stop-start,
            first10_all_valid=stop-start >= 10)
    groups = {}
    for task in ('place_mouse_pad', 'hanging_mug'):
        for dimension, names in (('phase', ('early', 'middle', 'late')),
                                  ('gripper_event', ('closing', 'opening', 'closing_and_opening', 'no_transition'))):
            for name in names:
                selected = {key for key, value in labels.items()
                            if value['task'] == task and value[dimension] == name}
                if not selected:
                    continue
                metrics = {}
                episode_counts = {}
                for model, report in reports.items():
                    by_episode = {}
                    for row in report['records']:
                        if row['anchor_id'] in selected:
                            by_episode.setdefault(labels[row['anchor_id']]['episode'], []).append(row)
                    per_episode = [mean_rows(rows) for rows in by_episode.values()]
                    metrics[model] = mean_rows(per_episode)
                    episode_counts[model] = len(per_episode)
                groups[f'{task}/{dimension}/{name}'] = dict(anchors=len(selected),
                    episodes=episode_counts, metrics=metrics,
                    coverage_change_percent={k: 100*(metrics['coverage'][k]/metrics['original'][k]-1)
                                             for k in METRICS})
    report = dict(status='completed', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_audits={k: v['checkpoint'] for k, v in reports.items()}, anchors=labels, groups=groups,
        rule='Event if either source target gripper changes by more than 0.05 between neighboring frames in next 50 valid frames. Early <20%, middle 20-80%, late >=80% of trajectory.',
        aggregation='Mean anchors within each episode, then mean represented episodes per group.',
        limitations=['Post hoc explanation, not causal proof or rollout success.',
            'At most five held-out episodes per task; overlapping windows and source events are dependent.',
            'Gripper error includes both arms, not just the changing arm.',
            'Legacy first10 metrics include repeated terminal targets when fewer than ten frames remain; tagged in anchors.'])
    output = ROOT/'outputs/gen1_recovery_phase_diagnostic_20261012.json'
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(output)
    for group, row in groups.items():
        print(group, 'anchors', row['anchors'], 'episodes', row['episodes']['original'],
              'coverage % change', json.dumps(row['coverage_change_percent']))


if __name__ == '__main__':
    main()
