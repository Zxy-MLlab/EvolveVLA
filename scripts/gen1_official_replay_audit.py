"""CPU audit of the pinned official replay export; never changes source data.

This is a structural gate, not an assertion that images match source RGB. In
particular, an episode with a shorter video is quarantined in its entirety:
the missing frame positions cannot be inferred from container metadata.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path('/root/autodl-tmp/EvolveVLA')
SOURCE = ROOT / 'data/gen1_official_replay_source_v2_20261013'
DATA = ROOT / 'data/gen1_official_replay_v2_20261013/RoboTwin_lerobot_v30'
NORM = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k/assets/aloha_agilex_sim/norm_stats.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def trajectory_hash(frame):
    h = hashlib.sha256()
    for key in ('observation.state', 'action'):
        h.update(np.asarray(np.stack(frame[key]), dtype='<f4').tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Audit output must be new')
    completion = json.loads((SOURCE / 'completed.json').read_text())
    archive = Path(completion['archive'])
    require(archive.stat().st_size == completion['bytes'] and
            digest(archive) == completion['sha256'], 'Pinned archive mismatch')
    inventory = json.loads((SOURCE / 'archive_inventory.json').read_text())
    import zipfile
    with zipfile.ZipFile(archive) as z:
        for item in inventory:
            if item['path'].endswith('/'):
                continue
            relative = Path(item['path']).relative_to('RoboTwin_lerobot_v30')
            path = DATA / relative
            require(path.is_file() and path.stat().st_size == item['bytes'],
                    f'Incomplete extraction: {relative}')
            # Verify extracted bytes, including video bytes, against archive.
            h = hashlib.sha256()
            with z.open(item['path']) as stream:
                while block := stream.read(4 * 1024**2):
                    h.update(block)
            require(digest(path) == h.hexdigest(), f'Extracted bytes differ: {relative}')
    info = json.loads((DATA / 'meta/info.json').read_text())
    require(info['fps'] == 15 and info['total_episodes'] == 2500, 'Unexpected export')
    cameras = [k for k, v in info['features'].items() if v['dtype'] == 'video']
    require(set(cameras) == {f'observation.images.cam_{c}' for c in
                            ('high', 'left_wrist', 'right_wrist')}, 'Camera mapping mismatch')
    for key in ('action', 'observation.state'):
        require(info['features'][key]['shape'] == [14], 'Wrong physical dimension')
    episodes = pd.read_parquet(DATA / 'meta/episodes').set_index('episode_index').sort_index()
    require(list(episodes.index) == list(range(2500)), 'Invalid episode indices')
    tasks = pd.read_parquet(DATA / 'meta/tasks.parquet')
    prompts = {int(row.task_index): str(prompt) for prompt, row in tasks.iterrows()}
    local_hashes = defaultdict(list)
    for split in ('train', 'val'):
        root = ROOT / f'data/gen1_multitask_coverage_{split}_20261012'
        local = pd.read_parquet(root / 'data').sort_values('index')
        for episode, frame in local.groupby('episode_index'):
            local_hashes[trajectory_hash(frame)].append(dict(split=split, episode=int(episode)))
    containers, rows, offset = {}, [], 0
    for path in sorted((DATA / 'data').rglob('*.parquet')):
        data = pd.read_parquet(path).sort_values('index')
        for episode, frame in data.groupby('episode_index', sort=True):
            meta = episodes.loc[episode]
            count = len(frame)
            require(count == int(meta.length), 'Episode split across files or wrong length')
            require(np.array_equal(frame['index'], np.arange(offset, offset + count)), 'Index gap')
            require(np.array_equal(frame.frame_index, np.arange(count)), 'Frame index gap')
            require((int(meta.dataset_from_index), int(meta.dataset_to_index)) ==
                    (offset, offset + count), 'Episode boundary mismatch')
            require(np.allclose(frame.timestamp, np.arange(count)/15, atol=1e-5), 'Timestamp mismatch')
            states, actions = np.stack(frame['observation.state']), np.stack(frame.action)
            require(states.shape == actions.shape == (count, 14), 'Array shape mismatch')
            require(np.isfinite(states).all() and np.isfinite(actions).all(), 'Nonfinite actions')
            require(np.array_equal(states[1:], actions[:-1]), 'Next-state action mismatch')
            require(frame.task_index.nunique() == 1, 'Within-episode prompt changes')
            prompt = prompts[int(frame.iloc[0].task_index)]
            require(list(meta.tasks) == [prompt], 'Episode prompt mismatch')
            videos, reasons = {}, []
            for camera in cameras:
                key = f'videos/{camera}'
                video = DATA / info['video_path'].format(video_key=camera,
                    chunk_index=int(meta[key+'/chunk_index']), file_index=int(meta[key+'/file_index']))
                if str(video) not in containers:
                    with av.open(str(video)) as v:
                        s = v.streams.video[0]
                        require(s.average_rate == 15 and (s.width, s.height) == (320, 240),
                                'Video format mismatch')
                        containers[str(video)] = dict(frames=s.frames, assigned_frames=0,
                                                     assigned_end=0, episodes=0)
                record = containers[str(video)]
                start, end = float(meta[key+'/from_timestamp'])*15, float(meta[key+'/to_timestamp'])*15
                require(abs(start-round(start)) < 1e-5 and abs(end-round(end)) < 1e-5,
                        'Non-integer video frame boundary')
                start, end = round(start), round(end)
                require(start == record['assigned_end'] and start < end <= record['frames'],
                        'Noncontiguous or out-of-bounds video interval')
                record['assigned_end'] = end
                record['assigned_frames'] += end-start
                record['episodes'] += 1
                videos[camera] = dict(path=str(video), start=start, stop=end, frames=end-start,
                                      expected_frames=count, missing_frames=count-(end-start))
                if end-start != count:
                    reasons.append(camera + ':frame_count_mismatch')
            sha = trajectory_hash(frame)
            overlaps = local_hashes.get(sha, [])
            if overlaps:
                reasons.append('exact_state_action_overlap_with_local')
            rows.append(dict(episode=int(episode), start=offset, stop=offset+count, frames=count,
                             prompt=prompt, trajectory_sha256=sha, local_overlaps=overlaps,
                             videos=videos, quarantine_reasons=reasons,
                             structural_candidate=not reasons))
            offset += count
    require(offset == info['total_frames'] and len(rows) == 2500, 'Incomplete table audit')
    for path, item in containers.items():
        require(item['assigned_frames'] == item['frames'], f'Video accounting mismatch: {path}')
    report = dict(status='structural_audit_complete', ready_for_training=False,
        dataset=str(DATA), archive_sha256=completion['sha256'], script_sha256=digest(Path(__file__)),
        norm_stats=dict(path=str(NORM), sha256=digest(NORM), policy='reuse unchanged; do not recompute'),
        episodes=len(rows), frames=offset, fps=15, state_action_dim=14,
        next_state_action_exact=True, prompt_binding_exact=True,
        quarantined=sum(bool(r['quarantine_reasons']) for r in rows),
        structural_candidates=sum(r['structural_candidate'] for r in rows),
        local_exact_overlaps=sum(bool(r['local_overlaps']) for r in rows),
        missing_frames_by_camera={c:sum(r['videos'][c]['missing_frames'] for r in rows) for c in cameras},
        videos=containers, records=rows,
        remaining=['Task identity must be verified against original per-task source trajectories',
                   'Source RGB alignment; container count alone does not prove image-action parity',
                   'Original source seed provenance and split assignment',
                   'Mixed loader preflight and actual Clean paired evaluation'],
        caveat='Exact trajectory hashes rule out exact duplicates only, not shared scene seeds or near duplicates')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','videos')}, indent=2))


if __name__ == '__main__':
    main()
