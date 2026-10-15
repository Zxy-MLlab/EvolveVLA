#!/usr/bin/env python3
"""Bind the completed two-task collection to the unchanged 18-task split."""
import argparse
import json
from pathlib import Path

from gen1_multitask_data import digest, validate_split

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OLD = ROOT / 'data/gen1_multitask_coverage_20261012/split_manifest.json'
BATCH = ROOT / 'data/gen1_task_batches_20261010'
TASKS = ('place_mouse_pad', 'hanging_mug')


def prepare(output):
    output = output.resolve()
    if output.parent != (ROOT / 'data').resolve() or output.exists():
        raise ValueError('Output must be a fresh direct child of data')
    old = json.loads(OLD.read_text())
    validate_split(old)
    rows = list(old['episodes'])
    existing = {(r['task'], r['seed']) for r in rows}
    paths = {r['path'] for r in rows}
    hashes = {r['sha256'] for r in rows}
    for task in TASKS:
        manifest = json.loads((BATCH / task / 'manifest.json').read_text())
        accepted = manifest['credited_episodes'] + manifest['episodes']
        if manifest['status'] != 'complete' or len(accepted) != 1000:
            raise ValueError(f'{task} collection incomplete')
        preserved = [r for r in rows if r['task'] == task and r['split'] == 'val']
        if len(preserved) != 5:
            raise ValueError(f'{task} validation set changed')
        old_by_seed = {r['seed']: r for r in rows if r['task'] == task}
        for source in accepted:
            seed = source['seed']
            if seed in old_by_seed:
                if old_by_seed[seed]['sha256'] != source['hdf5_sha256']:
                    raise ValueError(f'{task}/{seed} historical hash changed')
                continue
            h5s = list((Path(source['episode_directory']) / 'data').glob('*.hdf5'))
            if len(h5s) != 1:
                raise ValueError(f'{task}/{seed} missing source')
            h5 = h5s[0].resolve()
            sha = digest(h5)
            if sha != source['hdf5_sha256'] or (task, seed) in existing or str(h5) in paths or sha in hashes:
                raise ValueError(f'{task}/{seed} duplicate or changed source')
            if not all(source[k] is True for k in ('success', 'plan_success', 'replay_success')):
                raise ValueError(f'{task}/{seed} unaccepted source')
            row = dict(task=task, seed=seed, split='train', path=str(h5), sha256=sha,
                       frames=source['frames'], episode_directory=source['episode_directory'],
                       source_revision=source['source_revision'], success=True,
                       plan_success=True, replay_success=True)
            rows.append(row)
            existing.add((task, seed))
            paths.add(str(h5))
            hashes.add(sha)
    counts = {split: sum(r['split'] == split for r in rows) for split in ('train', 'val')}
    per_task = {split: {task: sum(r['task'] == task and r['split'] == split for r in rows)
                        for task in old['tasks']} for split in ('train', 'val')}
    for task in TASKS:
        if (per_task['train'][task], per_task['val'][task]) != (995, 5):
            raise ValueError(f'{task} expanded split count mismatch')
    report = dict(old, episodes=rows, counts=counts, per_task_counts=per_task,
                  base_split_sha256=digest(OLD),
                  source_manifests=[dict(task=t, path=str(BATCH / t / 'manifest.json'),
                                         sha256=digest(BATCH / t / 'manifest.json')) for t in TASKS],
                  split_method='Preserve original 18-task split and all 5 validation episodes per task; append only newly audited two-task Clean trajectories')
    output.mkdir()
    for split in ('train', 'val'):
        folder = output / f'{split}_inputs'
        folder.mkdir()
        for row in sorted((r for r in rows if r['split'] == split), key=lambda r: (r['task'], r['seed'])):
            link = folder / f"{row['task']}_seed_{row['seed']:06d}.hdf5"
            link.symlink_to(row['path'])
            row['input_link'] = str(link)
    validate_split(report)
    (output / 'split_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), counts=counts, per_task=per_task), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    prepare(p.parse_args().output)
