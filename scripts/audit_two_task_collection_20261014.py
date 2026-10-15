#!/usr/bin/env python3
"""Audit all accepted trajectories; completion requires 1,000 per task.

Run in robotwin_cu128. Safe during collection: reads atomic manifest snapshots
and immutable accepted episodes. Does not change data or resume training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import gen1_collect_task_batch_20261010 as collector

TASKS = ('place_mouse_pad', 'hanging_mug')
TARGET = 1000
RANGES = {'place_mouse_pad': (74000, 87000), 'hanging_mug': (87000, 100000)}


def training_processes():
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        # Match actual script arguments, not text in a shell diagnostic.
        for arg in args[1:]:
            name = Path(arg.decode(errors='replace')).name
            if name.startswith('gen1_') and name.endswith('.py') and (
                    'train' in name or 'lora_extend' in name or 'lora_train' in name):
                found.append({'pid': int(entry.name), 'script': name})
                break
    return found


def audit():
    require = collector.require
    require(collector.vetted.digest(collector.WRAPPER) == collector.WRAPPER_HASH,
            'Trusted alignment/image validator changed')
    require(subprocess.check_output(['git', '-C', str(collector.REPO), 'rev-parse', 'HEAD'],
                                    text=True).strip() == collector.vetted.REV, 'Simulator revision changed')
    require(not subprocess.check_output(['git', '-C', str(collector.REPO), 'status', '--porcelain',
                                        '--untracked-files=no'], text=True), 'Simulator tracked source changed')
    split_path = collector.ROOT / 'data/gen1_multitask_coverage_20261012/split_manifest.json'
    split = json.loads(split_path.read_text())
    report = {'checked_at_unix': time.time(), 'target_per_task': TARGET,
              'training_processes': training_processes(), 'tasks': {},
              'original_split_sha256': collector.vetted.digest(split_path),
              'free_gib': collector.free_gib()}
    global_seeds = set()
    for task in TASKS:
        path = collector.BATCH / task / 'manifest.json'
        manifest = json.loads(path.read_text())
        rows = manifest['credited_episodes'] + manifest['episodes']
        require(manifest['task'] == task, 'Wrong manifest task')
        require(len(rows) <= TARGET, 'More accepted episodes than requested')
        require(manifest['target_total'] == TARGET, 'Collector target changed')
        hashes, paths, total_bytes, frames = set(), set(), 0, 0
        initial_head_hashes = set()
        new_rows = []
        for row in rows:
            require(row['task'] == task and all(row[k] for k in
                    ('success', 'plan_success', 'replay_success')), 'Invalid success predicates')
            require(row['source_revision'] == collector.vetted.REV and
                    row['config'] == 'demo_clean', 'Wrong source/config')
            require(row['seed'] not in global_seeds, 'Duplicate seed across accepted episodes')
            global_seeds.add(row['seed'])
            directory = Path(row['episode_directory'])
            files = list((directory / 'data').glob('*.hdf5'))
            require(len(files) == 1, 'Expected exactly one HDF5 per episode')
            h5 = files[0]
            digest = collector.vetted.digest(h5)
            require(digest == row['hdf5_sha256'], f'HDF5 changed: {h5}')
            require(digest not in hashes and str(h5) not in paths, 'Duplicate episode content/path')
            hashes.add(digest)
            paths.add(str(h5))
            require(h5.stat().st_size == row['hdf5_bytes'], 'HDF5 size mismatch')
            require(row['state_action_dim'] == 14 and row['raw_frames'] == row['frames'] + 1,
                    'Invalid alignment dimensions')
            require(len(row['raw_sha256']) == row['raw_frames'], 'Incomplete original raw audit')
            require(row['alignment'] == 'state=q[:-1], action=q[1:], all JPEG bytes match official encoding of raw frame[:-1]',
                    'Missing audited action/image alignment')
            for filename, field in [('resolved_config.json', 'config_sha256')]:
                require(collector.vetted.digest(directory / filename) == row[field],
                        f'Metadata changed: {directory / filename}')
            instruction = list((directory / 'instruction').glob('*.json'))
            require(len(instruction) == 1 and collector.vetted.digest(instruction[0]) == row['instruction_sha256'],
                    'Instruction metadata changed')
            with h5py.File(h5, 'r') as handle:
                n = row['frames']
                require(int(handle['additional_info/frequency'][()]) == 15, 'Wrong frequency')
                instructions = json.loads(handle['instructions'][()].decode())
                require(bool(instructions) and all(instructions), 'Empty instructions')
                for group in ('state', 'action'):
                    width = 0
                    for field in collector.vetted.FIELDS.values():
                        values = handle[group][field][...]
                        require(values.ndim == 2 and values.shape[0] == n and np.isfinite(values).all(),
                                'Invalid state/action values')
                        width += values.shape[1]
                    require(width == 14, 'Wrong state/action width')
                # Official HDF5 also stores cam_third_view, unused by the policy.
                require(set(collector.vetted.CAMS.values()) <= set(handle['vision']), 'Missing policy cameras')
                for camera in collector.vetted.CAMS.values():
                    require(len(handle['vision'][camera]['colors']) == n, 'Wrong image frame count')
                initial_head_hashes.add(hashlib.sha256(bytes(handle['vision/cam_head/colors'][0])).hexdigest())
            total_bytes += h5.stat().st_size
            frames += row['frames']
            if RANGES[task][0] <= row['seed'] < RANGES[task][1]:
                new_rows.append(row)
        # Preserve the five previously held-out trajectories per task.
        heldout = [r for r in split['episodes'] if r['task'] == task and r['split'] == 'val']
        require(len(heldout) == 5, 'Original validation partition changed')
        by_seed = {r['seed']: r for r in rows}
        for row in heldout:
            require(row['seed'] in by_seed and row['sha256'] == by_seed[row['seed']]['hdf5_sha256'],
                    'Original held-out trajectory missing or changed')
        new_seeds = {r['seed'] for r in new_rows}
        accepted_attempts = {r['seed']: r for r in manifest['attempts'] if r['status'] == 'accepted'}
        require(new_seeds <= accepted_attempts.keys(), 'New accepted episode missing successful attempt')
        for seed in new_seeds:
            require(accepted_attempts[seed]['returncode'] == 0, 'Accepted worker did not succeed')
            claim = json.loads((collector.BATCH / 'seed_claims' / f'seed_{seed}.json').read_text())
            require(claim['task'] == task and claim['seed'] == seed, 'Wrong seed ownership')
        if manifest['status'] == 'complete':
            require(not list((collector.BATCH / task / '.staging').iterdir()), 'Unfinished stage at completion')
            directories = {str(p) for p in (collector.BATCH / task / 'episodes').iterdir()}
            require(directories == {r['episode_directory'] for r in manifest['episodes']},
                    'Unmanifested accepted directory at completion')
        remaining = (TARGET - len(rows)) * total_bytes / max(1, len(rows)) / 2**30
        report['tasks'][task] = {
            'accepted': len(rows), 'new_accepted': len(new_rows),
            'hdf5_gib': total_bytes / 2**30, 'frames': frames,
            'unique_initial_head_images': len(initial_head_hashes),
            'collector_status': manifest['status'], 'estimated_remaining_gib': remaining,
            'preserved_validation_seeds': [r['seed'] for r in heldout],
            'manifest_snapshot_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            'complete': len(rows) == TARGET and manifest['status'] == 'complete',
        }
    report['estimated_remaining_gib'] = sum(r['estimated_remaining_gib'] for r in report['tasks'].values())
    report['complete'] = not report['training_processes'] and all(r['complete'] for r in report['tasks'].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = audit()
    if args.output:
        collector.dump(args.output, report)
    print(json.dumps(report, indent=2))
    if args.require_complete:
        collector.require(report['complete'], 'Both tasks must have 1,000 audited episodes and training stopped')


if __name__ == '__main__':
    main()
