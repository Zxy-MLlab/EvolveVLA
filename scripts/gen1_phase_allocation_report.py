"""Measure phase sampling probability, without training or changing selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

from gen1_multitask_data import ordered_episodes, digest
from gen1_multitask_train import phase_weights
from gen1_validation_event_report import labels
from gen1_dual_event_weights import dual_event_weights
from gen1_training_event_audit import event_windows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing overwrite')
    read = lambda name: json.loads((args.training / name).read_text())
    config = read('run_config.json')
    rows = ordered_episodes(read('split_manifest.json'), read('train_conversion_audit.json'), 'train')
    dataset = Path(config['config']['data']['repo_id'])
    home = Path('/root/autodl-tmp/EvolveVLA/data')
    table = pq.read_table(home / dataset / 'data', columns=['index', 'action'])
    if table['index'].to_pylist() != list(range(rows[-1]['stop'])):
        raise ValueError('Parquet read order differs from training frame order')
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    weights, reports = {}, {}
    for mode in ('post', 'prepost'):
        captured = []
        settings = SimpleNamespace(lerobot_home=home, repo_id=str(dataset),
            provenance_dir=args.training, phase_window=mode)
        with patch('gen1_multitask_train.write_new', side_effect=lambda p, v: captured.append(v)):
            weights[mode] = phase_weights(settings, rows)
        reports[mode] = captured[0]
    historical = read('phase_sampling.json')
    for key in ('event_frames', 'late_frames', 'total_frames'):
        if reports['post'][key] != historical[key]:
            raise ValueError('Production post weights differ from running experiment')
    weights['dual_event_candidate'] = dual_event_weights(actions, rows)
    groups, task_indices = defaultdict(list), defaultdict(list)
    for row in rows:
        local = actions[row['start']:row['stop'], [6, 13]]
        transitions = event_windows(local, horizon=50)
        for frame in range(row['frames']):
            index = row['start'] + frame
            event, phase = labels(local, frame)
            task_indices[row['task']].append(index)
            groups[(row['task'], 'event', event)].append(index)
            groups[(row['task'], 'phase', phase)].append(index)
            for transition, flags in transitions.items():
                for column, arm in enumerate(('left', 'right')):
                    if flags[frame, column]:
                        groups[(row['task'], transition + '_arm', arm)].append(index)
            if row['frames'] - frame < 50:
                groups[(row['task'], 'padding', 'less_than_50_valid_targets')].append(index)
    summary = {}
    for (task, dimension, label), indices in sorted(groups.items()):
        total = task_indices[task]
        uniform = len(indices) / len(total)
        modes = {}
        for mode, values in weights.items():
            mass = float(values[indices].sum() / values[total].sum())
            modes[mode] = dict(probability=mass, ratio_to_uniform=mass/uniform)
        summary[f'{task}/{dimension}/{label}'] = dict(frames=len(indices),
            uniform_probability=uniform, modes=modes)
    result = dict(groups=summary, production_weights=reports,
        training=str(args.training),
        source_sha256={str(args.training / name): digest(args.training / name) for name in
            ('run_config.json', 'split_manifest.json', 'train_conversion_audit.json', 'phase_sampling.json')},
        limitations=['Expected probabilities within each task, not observed runtime counts.',
            'Prepost and dual_event_candidate are hypothetical options, not used by the running training.',
            'Labels describe changes in the next 50 valid targets, not actual grasp/release success.',
            'Not a checkpoint selection rule or evidence of causal rollout benefit.'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    for name, value in summary.items():
        if name.split('/')[0] in ('place_mouse_pad', 'hanging_mug'):
            print(name, json.dumps(value))


if __name__ == '__main__':
    main()
