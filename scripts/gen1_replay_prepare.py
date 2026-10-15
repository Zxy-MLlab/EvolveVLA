"""Prepare an immutable official raw replay manifest without extracting videos.

Only downloaded archives are read. Task/seed, source instructions, source image
counts, state/action cells and exact duplicate exclusion are checked here.
This gate does not claim source RGB has passed the later converted-video audit.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
import pandas as pd

from gen1_official_replay_audit import ROOT, NORM, digest, require, trajectory_hash
from gen1_prompt_augmentation import compatible

FIELDS = ('left_arm_joint_states', 'left_ee_joint_states',
          'right_arm_joint_states', 'right_ee_joint_states')
CAMERAS = {'cam_high': 'cam_head', 'cam_left_wrist': 'cam_left_wrist',
           'cam_right_wrist': 'cam_right_wrist'}
RAW = ROOT / 'data/gen1_official_raw18_20261013'
LOCAL_SPLIT = ROOT / 'data/gen1_multitask_coverage_20261012/split_manifest.json'


def arrays(source):
    return tuple(np.concatenate([source[f'{group}/{key}'][:] for key in FIELDS], axis=1)
                 .astype(np.float32) for group in ('state', 'action'))


def signature(state, action):
    h = hashlib.sha256()
    for value in (state, action):
        h.update(np.asarray(value, dtype='<f4').tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh manifest output required')
    complete = json.loads((RAW / 'completed.json').read_text())
    require(complete['status'] == 'downloads_verified', 'Downloads incomplete')
    local = json.loads(LOCAL_SPLIT.read_text())
    tasks = local['tasks']
    require(set(s['task'] for s in complete['sources']) == set(tasks), 'Source task mismatch')
    local_seeds = {(r['task'], r['seed']) for r in local['episodes']}
    hashes = set()
    for split in ('train', 'val'):
        root = ROOT / f'data/gen1_multitask_coverage_{split}_20261012'
        table = pd.read_parquet(root / 'data').sort_values('index')
        for _, frame in table.groupby('episode_index'):
            hashes.add(trajectory_hash(frame))
    export = json.loads((ROOT / 'outputs/gen1_official_replay_structural_audit_20261013.json').read_text())
    export_lookup = {r['trajectory_sha256']: r['episode'] for r in export['records']}
    rows, seen, offset = [], set(), 0
    for source in sorted(complete['sources'], key=lambda s: s['task']):
        archive = Path(source['archive'])
        require(digest(archive) == source['sha256'], 'Archive changed')
        require(archive.stat().st_size == source['bytes'], 'Archive size changed')
        task = source['task']
        with zipfile.ZipFile(archive) as z:
            prefix = f'{task}/aloha_agilex/'
            seeds = [int(s) for s in z.read(prefix+'seed.txt').decode().split()]
            names = sorted(n for n in z.namelist() if n.startswith(prefix+'data/') and n.endswith('.hdf5'))
            require(len(names) == len(seeds) == len(set(seeds)) == 50, 'Expected fifty unique task seeds')
            require(names == [prefix+f'data/episode_{i:07d}.hdf5' for i in range(50)],
                    'Cannot bind ordered seed file to episode names')
            for source_episode, (name, seed) in enumerate(zip(names, seeds, strict=True)):
                require((task,seed) not in local_seeds, 'Official/local task-seed overlap')
                content = z.read(name)
                with h5py.File(io.BytesIO(content), 'r') as f:
                    require(f.attrs['source_format'] == 'RoboTwin', 'Wrong raw source format')
                    require(int(f['additional_info/frequency'][()]) == 15, 'Raw timing mismatch')
                    state, action = arrays(f)
                    n = len(state)
                    require(state.shape == action.shape == (n,14) and n > 1, 'Raw dimension mismatch')
                    require(np.isfinite(state).all() and np.isfinite(action).all(), 'Nonfinite raw values')
                    require(np.array_equal(state[1:],action[:-1]), 'Raw next-state action mismatch')
                    sha = signature(state,action)
                    require(sha not in hashes and sha not in seen, 'Duplicate raw/local trajectory')
                    seen.add(sha)
                    prompts = json.loads(f['instructions'][()])
                    require(isinstance(prompts,list) and all(isinstance(p,str) and p for p in prompts),
                            'Invalid source prompt pool')
                    retained = list(dict.fromkeys(p for p in prompts if compatible(task,p)))
                    require(bool(retained), 'No action-compatible source prompts')
                    for camera in CAMERAS.values():
                        require(len(f[f'vision/{camera}/colors']) == n and
                                list(f[f'vision/{camera}/shape'][:]) == [240,320,3],
                                'Source image/state count or shape mismatch')
                rows.append(dict(source='official',split='train',task=task,seed=seed,
                    episode=len(rows),source_episode=source_episode,archive=str(archive),member=name,
                    archive_sha256=source['sha256'],sha256=hashlib.sha256(content).hexdigest(),
                    trajectory_sha256=sha,export_episode=export_lookup.get(sha),
                    frames=n,start=offset,stop=offset+n,prompts=retained,
                    rejected_prompts=[p for p in prompts if p not in retained]))
                offset += n
        print('RAW_MANIFEST',task,'episodes',len(rows),'frames',offset,flush=True)
    require(Counter(r['task'] for r in rows) == Counter({t:50 for t in tasks}), 'Raw coverage mismatch')
    report = dict(status='raw_manifest_passed',ready_for_training=False,
        source_revision=complete['sources'][0]['revision'],tasks=tasks,episodes=rows,
        counts=dict(official_train=len(rows),local_train=local['counts']['train'],local_val=local['counts']['val']),
        frames=offset,local_split=str(LOCAL_SPLIT),local_split_sha256=digest(LOCAL_SPLIT),
        norm_stats=dict(path=str(NORM),sha256=digest(NORM),recomputed=False),
        sampler='Equal task and official/local strata; frame uniform within each stratum',
        official_fraction=0.5,source_action_semantics='next recorded state; absolute joints and grippers',
        fps=15,action_horizon=50,source_prompt_rule='Original pool filtered by existing compatible() rule',
        script_sha256=digest(Path(__file__)),
        validation='Unchanged local held-out trajectories only; official data may be initializer training data',
        checks=dict(all_raw_state_action_finite=True,source_frame_counts=True,
                    task_seed_disjoint_from_local=True,exact_trajectory_hash_disjoint=True),
        remaining='Frame-preserving conversion, full RGB audit and mixed loader preflight')
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print('RAW_MANIFEST_PASSED',len(rows),offset,flush=True)


if __name__ == '__main__':
    main()
