#!/usr/bin/env python3
"""Create a hash-checked, trajectory-disjoint Clean split from local task batches.

This is intentionally independent of the historical two-task split. It only
creates symlinks and a manifest; source HDF5 files remain read-only. A source
episode is admitted only when its task manifest records successful planning and
replay under the pinned RoboTwin revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path('/root/autodl-tmp/EvolveVLA')
REVISION = 'ba0fd21ad21c336bedc7bd8eacd3e514860910e2'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Fresh direct child of EvolveVLA/data')
    parser.add_argument('--source-root', type=Path,
                        default=ROOT / 'data/gen1_task_batches_20261010')
    parser.add_argument('--train-per-task', type=int, default=45)
    parser.add_argument('--val-per-task', type=int, default=5)
    parser.add_argument('--split-namespace', default='gen1-multitask-coverage-v1')
    args = parser.parse_args()

    output = args.output.resolve()
    source_root = args.source_root.resolve()
    if output.parent != (ROOT / 'data').resolve():
        fail('Output must be a direct child of EvolveVLA/data')
    if output.exists() or output.is_symlink():
        fail(f'Refusing existing output: {output}')
    if not source_root.is_relative_to(ROOT / 'data') or not source_root.is_dir():
        fail('Source root must be an existing directory under EvolveVLA/data')
    if args.train_per_task <= 0 or args.val_per_task <= 0:
        fail('Train and validation counts must both be positive')
    if shutil.disk_usage(ROOT).free < 60 * 2**30:
        fail('Need at least 60 GiB free before preparing a conversion split')

    rows: list[dict] = []
    manifests: list[dict] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    seen_seeds: set[int] = set()
    task_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if not task_dirs:
        fail('No task directories found')

    for task_dir in task_dirs:
        manifest_path = task_dir / 'manifest.json'
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        task = manifest.get('task')
        if not isinstance(task, str) or task != task_dir.name:
            fail(f'Task/directory mismatch: {manifest_path}')
        if manifest.get('status') != 'complete':
            fail(f'Incomplete source manifest: {manifest_path}')
        episodes = list(manifest.get('credited_episodes', [])) + list(manifest.get('episodes', []))
        required = args.train_per_task + args.val_per_task
        if len(episodes) != required:
            fail(f'{task}: expected exactly {required} source episodes, got {len(episodes)}')
        ordered = sorted(
            episodes,
            key=lambda row: hashlib.sha256(
                f'{args.split_namespace}:{task}:{row.get("seed")}'.encode()
            ).hexdigest(),
        )
        manifests.append({'task': task, 'path': str(manifest_path), 'sha256': digest(manifest_path)})
        for rank, row in enumerate(ordered):
            if row.get('task') != task or row.get('config') != 'demo_clean':
                fail(f'{task}: task/config mismatch')
            if row.get('source_revision') != REVISION:
                fail(f'{task}: wrong RoboTwin revision')
            if not all(row.get(key) is True for key in ('success', 'plan_success', 'replay_success')):
                fail(f'{task}: source episode was not accepted')
            seed = row.get('seed')
            if type(seed) is not int or not 0 <= seed < 100000:
                fail(f'{task}: invalid collection seed')
            episode_dir = Path(row.get('episode_directory', '')).resolve()
            candidates = sorted((episode_dir / 'data').glob('*.hdf5'))
            if len(candidates) != 1:
                fail(f'{task}/{seed}: expected exactly one HDF5 source')
            h5 = candidates[0].resolve()
            if not h5.is_relative_to(ROOT / 'data'):
                fail(f'{task}/{seed}: source outside project data root')
            sha = digest(h5)
            if sha != row.get('hdf5_sha256'):
                fail(f'{task}/{seed}: source hash mismatch')
            if str(h5) in seen_paths or sha in seen_hashes or seed in seen_seeds:
                fail(f'{task}/{seed}: duplicated source path, content, or collection seed')
            seen_paths.add(str(h5))
            seen_hashes.add(sha)
            seen_seeds.add(seed)
            split = 'val' if rank < args.val_per_task else 'train'
            rows.append({
                'task': task, 'seed': seed, 'split': split, 'path': str(h5), 'sha256': sha,
                'frames': row.get('frames'), 'episode_directory': str(episode_dir),
                'source_revision': REVISION, 'success': True, 'plan_success': True,
                'replay_success': True,
            })

    tasks = [row['task'] for row in manifests]
    if len(tasks) < 3:
        fail('Coverage split requires at least three tasks')
    expected_total = len(tasks) * (args.train_per_task + args.val_per_task)
    if len(rows) != expected_total:
        fail('Unexpected total episode count')
    counts = {split: sum(row['split'] == split for row in rows) for split in ('train', 'val')}
    for task in tasks:
        task_rows = [row for row in rows if row['task'] == task]
        if sum(row['split'] == 'train' for row in task_rows) != args.train_per_task:
            fail(f'{task}: incorrect train count')
        if sum(row['split'] == 'val' for row in task_rows) != args.val_per_task:
            fail(f'{task}: incorrect validation count')

    output.mkdir()
    try:
        for split in ('train', 'val'):
            inputs = output / f'{split}_inputs'
            inputs.mkdir()
            for row in sorted((row for row in rows if row['split'] == split),
                              key=lambda row: (row['task'], row['seed'])):
                link = inputs / f"{row['task']}_seed_{row['seed']:06d}.hdf5"
                link.symlink_to(row['path'])
                row['input_link'] = str(link)
        report = {
            'schema': 1,
            'status': 'passed',
            'tasks': tasks,
            'source_manifests': manifests,
            'split_method': (
                f'trajectory-level SHA256 rank of {args.split_namespace}:task:seed; '
                f'first {args.val_per_task} validation per task'
            ),
            'counts': counts,
            'per_task_counts': {'train': args.train_per_task, 'val': args.val_per_task},
            'episodes': rows,
            'source_revision': REVISION,
            'original_checkpoint': str(ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'),
            'selection_note': (
                'Broader Clean training coverage for a recovery experiment. '
                'Evaluation seeds are not used as collection seeds; all train/val trajectories are '
                'strictly disjoint by task, path, SHA256 content, and collection seed.'
            ),
        }
        (output / 'split_manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    except BaseException:
        shutil.rmtree(output)
        raise

    frame_counts = {split: sum(int(row['frames']) for row in rows if row['split'] == split)
                    for split in ('train', 'val')}
    print(json.dumps({
        'status': 'passed', 'output': str(output), 'tasks': len(tasks),
        'counts': counts, 'frames': frame_counts,
        'free_gib': round(shutil.disk_usage(ROOT).free / 2**30, 3),
    }, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
