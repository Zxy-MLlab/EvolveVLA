"""CPU-only source-event stratification of existing paired validation errors."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import h5py
import numpy as np

from gen1_multitask_validation_report import checked_records, KEYS


def labels(grippers, frame):
    n = len(grippers)
    future = grippers[frame:min(frame + 50, n)]
    delta = np.diff(future, axis=0)
    closing = bool(np.any(delta < -0.05))
    opening = bool(np.any(delta > 0.05))
    event = ('both' if closing and opening else 'closing' if closing
             else 'opening' if opening else 'no_transition')
    return event, 'late' if frame >= np.ceil(0.8 * (n - 1)) else 'nonlate'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.step <= 0 or args.output.exists():
        raise ValueError('Positive candidate step and new output required')
    paths = [args.training / name for name in ('validation_panel.json',
             'validation_0000.json', f'validation_{args.step:04d}.json')]
    panel, baseline, candidate = [json.loads(p.read_text()) for p in paths]
    before, after = checked_records(baseline), checked_records(candidate)
    if baseline['step'] != 0 or candidate['step'] != args.step or before.keys() != after.keys():
        raise ValueError('Changed step or anchor panel')
    identities = [(a['row']['task'], a['row']['episode'], a['frame']) for a in panel]
    if len(set(identities)) != len(panel) or set(identities) != set(before):
        raise ValueError('Source panel differs from metric records')
    sources, source_hashes, groups = {}, {}, defaultdict(list)
    for anchor, identity in zip(panel, identities):
        row = anchor['row']
        path = Path(row['path'])
        if row['split'] != 'val':
            raise ValueError('Non-held-out source')
        if path not in sources:
            with path.open('rb') as stream:
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            if sha != row['sha256']:
                raise ValueError('Source hash changed')
            source_hashes[str(path)] = sha
            with h5py.File(path) as stream:
                sources[path] = np.concatenate([stream[f'action/{arm}_ee_joint_states'][:]
                                                for arm in ('left', 'right')], axis=1)
        grips = sources[path]
        if grips.shape != (row['frames'], 2) or not np.isfinite(grips).all():
            raise ValueError('Invalid source grippers')
        event, phase = labels(grips, anchor['frame'])
        for dimension, label in [('event', event), ('phase', phase)]:
            groups[(row['task'], dimension, label)].append(identity)
        future = grips[anchor['frame']:min(anchor['frame'] + 50, len(grips))]
        opening = np.any(np.diff(future, axis=0) > 0.05, axis=0)
        if opening.any():
            arm = 'both' if opening.all() else 'left' if opening[0] else 'right'
            groups[(row['task'], 'opening_arm', arm)].append(identity)
    summaries = {}
    for (task, dimension, label), ids in sorted(groups.items()):
        episodes = sorted({i[1] for i in ids})
        metrics = {}
        for key in KEYS:
            pairs = []
            for episode in episodes:
                selected = [i for i in ids if i[1] == episode]
                pairs.append(tuple(statistics.mean(records[i][key] for i in selected)
                                   for records in (before, after)))
            old, new = (statistics.mean(p[j] for p in pairs) for j in (0, 1))
            metrics[key] = dict(before=old, after=new, ratio=new/max(old, 1e-8),
                                improved_episodes=sum(b < a for a, b in pairs))
        summaries[f'{task}/{dimension}/{label}'] = dict(anchors=len(ids),
            episodes=len(episodes), metrics=metrics)
    report = dict(step=args.step, groups=summaries, source_sha256=source_hashes,
        inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        rule='Source gripper neighboring delta >0.05 opening, <-0.05 closing in next 50 valid frames; late starts at ceil(0.8*(length-1)). Average anchors within episodes, then episodes.',
        limitations=['Post hoc diagnostic only; no checkpoint selection changes.',
            'Groups describe entire anchor errors, not isolated event-frame error.',
            'opening_arm labels the source transition; reported errors still pool both arms.',
            'Both arms pooled; overlapping chunks and at most five episodes per task.',
            'No closed-loop success or causal improvement established.'])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for name, value in summaries.items():
        if name.split('/')[0] in ('place_mouse_pad', 'hanging_mug'):
            print(name, json.dumps(value))


if __name__ == '__main__':
    main()
