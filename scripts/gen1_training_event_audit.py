#!/usr/bin/env python3
"""CPU audit of task balance and gripper-event exposure in training starts.

Uses source action targets, not inferred grasp success. No training, evaluation
or dataset mutation. Held-out records are excluded from the training counts.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path('/root/autodl-tmp/EvolveVLA')


def event_windows(grippers, horizon=50):
    """Flag starts whose valid future actions contain a >0.05 transition."""
    changes = np.diff(grippers, axis=0)
    result = {}
    for name, flag in [('closing', changes < -0.05), ('opening', changes > 0.05)]:
        prefix = np.concatenate([np.zeros((1, 2), dtype=int), np.cumsum(flag, axis=0)])
        starts = np.arange(len(grippers))
        stops = np.minimum(starts + horizon - 1, len(grippers) - 1)
        result[name] = prefix[stops] - prefix[starts] > 0
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    split_path = ROOT/'data/gen1_pilot_two_task_20261010/split_manifest.json'
    conversion_path = ROOT/'data/gen1_pilot_train_lossless_20261010/conversion_audit.json'
    sampled_path = ROOT/'outputs/gen1_recovery_native_augmented_20261012/sampling_coverage.json'
    split = json.loads(split_path.read_text())
    conversion = json.loads(conversion_path.read_text())
    sampled = json.loads(sampled_path.read_text())
    train = {str(Path(r['path']).resolve()): r for r in split['episodes'] if r['split'] == 'train'}
    rows = []
    for episode, source in enumerate(conversion['sources']):
        path = Path(source['path']).resolve()
        record = train[str(path)]
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != record['sha256'] or digest != source['sha256']:
            raise ValueError('Changed trajectory')
        with h5py.File(path) as stream:
            grippers = np.concatenate([stream[f'action/{arm}_ee_joint_states'][:]
                                       for arm in ('left', 'right')], axis=1)
        if grippers.shape != (record['frames'], 2) or not np.isfinite(grippers).all():
            raise ValueError('Invalid gripper data')
        windows = event_windows(grippers)
        counts = dict(frames=len(grippers), terminal_padded_starts=min(49, len(grippers)),
                      fetched=sampled['per_episode_samples'].get(str(episode), 0))
        for event, flags in windows.items():
            counts[event+'_either'] = int(flags.any(axis=1).sum())
            for column, arm in enumerate(('left', 'right')):
                counts[event+'_'+arm] = int(flags[:, column].sum())
        counts['any_transition'] = int((windows['closing'] | windows['opening']).any(axis=1).sum())
        rows.append(dict(episode=episode, task=record['task'], seed=record['seed'],
                         source=str(path), source_sha256=digest, counts=counts))
    tasks = {}
    total = sum(r['counts']['frames'] for r in rows)
    for task in sorted({r['task'] for r in rows}):
        selected = [r for r in rows if r['task'] == task]
        counts = Counter()
        for row in selected:
            counts.update(row['counts'])
        tasks[task] = dict(episodes=len(selected), counts=dict(counts),
            available_frame_share=counts['frames']/total,
            fetched_share=counts['fetched']/sampled['samples_fetched'],
            uniform_start_fraction={k: v/counts['frames'] for k, v in counts.items()
                                    if k not in ('frames', 'fetched')})
    if sum(t['counts']['fetched'] for t in tasks.values()) != sampled['samples_fetched']:
        raise ValueError('Sampling coverage total mismatch')
    report = dict(status='completed', tasks=tasks, episodes=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (split_path, conversion_path, sampled_path)},
        horizon=50, transition_threshold=0.05,
        limitations=['Events label expert action targets, not physical grasp success.',
            'Window fractions describe all available starts; fetched per-episode counts are actual, but individual fetched frame indices were not persisted.',
            'Overlapping action windows are not independent examples; no causal claim from counts.'])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(tasks, indent=2))


if __name__ == '__main__':
    main()
