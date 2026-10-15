#!/usr/bin/env python3
"""Freeze accepted two-task sources and trajectory-level split; never copy eval data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/root/autodl-tmp/EvolveVLA')
TASKS = ('place_mouse_pad', 'hanging_mug')
REV = 'ba0fd21ad21c336bedc7bd8eacd3e514860910e2'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    assert out.is_relative_to(ROOT / 'data') and not out.exists()
    assert shutil.disk_usage(ROOT).free >= 115 * 2**30
    rows = []
    manifests = []
    seen_paths, seen_hashes, seen_seeds = set(), set(), set()
    for task in TASKS:
        path = ROOT / 'data/gen1_task_batches_20261010' / task / 'manifest.json'
        m = json.loads(path.read_text())
        assert m['status'] == 'complete' and m['task'] == task
        episodes = m.get('credited_episodes', []) + m['episodes']
        assert len(episodes) == 50
        manifests.append({'task': task, 'path': str(path), 'sha256': digest(path)})
        ordered = sorted(episodes, key=lambda r: hashlib.sha256(
            f"gen1-pilot-split-0:{task}:{r['seed']}".encode()).hexdigest())
        for index, r in enumerate(ordered):
            assert r['task'] == task and r['config'] == 'demo_clean'
            assert r['source_revision'] == REV
            assert all(r[k] for k in ('success', 'plan_success', 'replay_success'))
            assert 0 <= r['seed'] < 100000
            files = list((Path(r['episode_directory']) / 'data').glob('*.hdf5'))
            assert len(files) == 1
            h5 = files[0].resolve()
            assert h5.is_relative_to(ROOT / 'data')
            sha = digest(h5)
            assert sha == r['hdf5_sha256']
            assert str(h5) not in seen_paths and sha not in seen_hashes
            assert r['seed'] not in seen_seeds
            seen_paths.add(str(h5)); seen_hashes.add(sha); seen_seeds.add(r['seed'])
            rows.append({'task': task, 'seed': r['seed'], 'split': 'val' if index < 5 else 'train',
                         'path': str(h5), 'sha256': sha, 'frames': r['frames'],
                         'episode_directory': r['episode_directory'], 'source_revision': REV,
                         'success': True, 'plan_success': True, 'replay_success': True})
    out.mkdir()
    for split in ('train', 'val'):
        inputs = out / (split + '_inputs')
        inputs.mkdir()
        for r in sorted((r for r in rows if r['split'] == split), key=lambda r: (r['task'], r['seed'])):
            link = inputs / f"{r['task']}_seed_{r['seed']:06d}.hdf5"
            link.symlink_to(r['path'])
            r['input_link'] = str(link)
    report = {'status': 'passed', 'tasks': list(TASKS), 'source_manifests': manifests,
              'split_method': 'trajectory-level SHA256 rank of gen1-pilot-split-0:task:seed; first 5 val per task',
              'counts': {'train': 90, 'val': 10}, 'episodes': rows,
              'evaluation_seed_start': 100000, 'original_checkpoint': str(ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'),
              'selection_note': 'Two tasks selected from original Clean baseline weak tasks; exploratory paired evaluation, not unseen-task generalization.'}
    with (out / 'split_manifest.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': 'passed', 'output': str(out), 'counts': report['counts'],
                      'frames': {s: sum(r['frames'] for r in rows if r['split'] == s) for s in ('train', 'val')},
                      'free_gib': shutil.disk_usage(ROOT).free / 2**30}), flush=True)


if __name__ == '__main__':
    main()
