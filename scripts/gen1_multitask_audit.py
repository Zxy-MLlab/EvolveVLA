#!/usr/bin/env python3
"""Audit finalized conversion by sequentially decoding every RGB frame.

Reads only existing finalized data and immutable source HDF5 files. Checks every
state/action cell, source hash, episode boundary, source prompt, image frame and
video timestamp. The conversion audit is newly created only after all checks;
an interrupted converter's unpersisted encoder report is never fabricated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/root/autodl-tmp/EvolveVLA')
PINNED = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/scripts/transform_lerobot_v30_format.py'
sys.path.insert(0, str(ROOT / 'scripts'))
from gen1_multitask_data import digest, require, validate_split


class SequentialVideo:
    def __init__(self):
        self.container = self.stream = self.iterator = self.path = None
        self.frames = 0
        self.report = []

    def finish(self):
        if self.container is None:
            return
        require(next(self.iterator, None) is None, f'Unaccounted video frames: {self.path}')
        require(self.frames == self.stream.frames, f'Video frame count mismatch: {self.path}')
        duration = float(self.stream.duration * self.stream.time_base)
        require(abs(duration - self.frames / 15) < 1e-5, f'Video duration mismatch: {self.path}')
        self.report.append(dict(path=str(self.path), frames=self.frames,
                                duration=duration, every_timestamp_checked=True))
        self.container.close()
        self.container = None

    def read(self, path, start_timestamp, count):
        import av
        import numpy as np
        if path != self.path:
            self.finish()
            self.path = path
            self.container = av.open(str(path))
            self.stream = self.container.streams.video[0]
            require(self.stream.average_rate == 15, 'Wrong video fps')
            self.stream.thread_count = 2
            self.stream.thread_type = 'SLICE'
            self.iterator = self.container.decode(self.stream)
            self.frames = 0
        require(abs(self.frames / 15 - start_timestamp) < 1e-5,
                f'Video/episode offset mismatch: {path}: {self.frames} != {start_timestamp}')
        pixels = []
        for _ in range(count):
            frame = next(self.iterator, None)
            require(frame is not None and frame.pts is not None, f'Missing video frame: {path}')
            timestamp = float(frame.pts * frame.time_base)
            require(abs(timestamp - self.frames / 15) < 1e-5, f'Video timestamp mismatch: {path}')
            pixels.append(frame.to_ndarray(format='rgb24'))
            self.frames += 1
        return np.stack(pixels)


def audit_finalized(dataset, rows, progress=None):
    """Check finalized files once in source order, without random frame seeks."""
    root = Path(dataset).resolve()
    require(root.is_relative_to(ROOT / 'data'), 'Dataset must be under project data')
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', HF_HUB_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled')
    import numpy as np
    import pandas as pd
    import h5py

    info = json.loads((root / 'meta/info.json').read_text())
    require((info['total_episodes'], info['total_frames'], info['fps']) ==
            (len(rows), sum(r['frames'] for r in rows), 15), 'Final dataset count/fps mismatch')
    for key in ('observation.state', 'action'):
        require(info['features'][key]['shape'] == [14], 'Wrong state/action feature shape')
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / 'data').rglob('*.parquet'))],
                     ignore_index=True).sort_values('index')
    require(np.array_equal(data['index'], np.arange(info['total_frames'])), 'Invalid global index')
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((root / 'meta/episodes').rglob('*.parquet'))])
    episodes = episodes.set_index('episode_index').sort_index()
    require(list(episodes.index) == list(range(len(rows))), 'Episode metadata mismatch')
    tasks = pd.read_parquet(root / 'meta/tasks.parquet')
    prompts = {int(row.task_index): str(prompt) for prompt, row in tasks.iterrows()}
    spec = importlib.util.spec_from_file_location('multitask_audit_pinned_converter', PINNED)
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    cameras = converter.CAMERA_CANDIDATES
    readers = {cam: SequentialVideo() for cam in cameras}
    offset = 0
    sources, maximum = [], 0.0
    started = time.monotonic()

    def update_progress(status, completed):
        elapsed = time.monotonic() - started
        value = dict(status=status, dataset=str(root), episodes_checked=completed,
                     total_episodes=len(rows), frames_checked=offset,
                     total_frames=info['total_frames'], cameras_per_frame=len(cameras),
                     percent=round(100 * offset / info['total_frames'], 2),
                     elapsed_s=round(elapsed, 2),
                     estimated_remaining_s=round(elapsed * (info['total_frames'] - offset) / offset)
                         if offset else None,
                     maximum_video_mae=maximum, updated_at_unix=time.time())
        if progress is not None:
            temporary = Path(progress).with_suffix('.json.tmp')
            temporary.write_text(json.dumps(value, indent=2) + '\n')
            temporary.replace(progress)
        print('SEQUENTIAL_AUDIT', json.dumps(value), flush=True)

    update_progress('running', 0)
    for episode, row in enumerate(rows):
        path = Path(row['path']).resolve()
        require(digest(path) == row['sha256'], 'Immutable source changed')
        with h5py.File(path, 'r') as source:
            require(source.attrs['source_format'] == 'RoboTwin', 'Wrong source format')
        item = converter.load(str(path), data_type='RoboDojo', data_version='v1.0')
        states, actions = converter._extract_qpos(item), converter._extract_action(item)
        count = row['frames']
        require(states.shape == actions.shape == (count, 14), 'Wrong source shape')
        require(np.isfinite(states).all() and np.isfinite(actions).all(), 'Nonfinite source vectors')
        np.testing.assert_array_equal(states[1:], actions[:-1])
        require(int(np.asarray(item['additional_info']['frequency'])) == 15, 'Source fps changed')
        frames = data.iloc[offset:offset + count]
        require((frames.episode_index == episode).all(), 'Source/converter order mismatch')
        np.testing.assert_array_equal(frames.frame_index, np.arange(count))
        np.testing.assert_array_equal(np.stack(frames['observation.state']), states)
        np.testing.assert_array_equal(np.stack(frames.action), actions)
        require(frames.task_index.nunique() == 1
                and prompts[int(frames.iloc[0].task_index)] in converter._find_instructions(item),
                'Source prompt binding mismatch')
        meta = episodes.loc[episode]
        require((int(meta.length), int(meta.dataset_from_index), int(meta.dataset_to_index)) ==
                (count, offset, offset + count), 'Episode metadata boundary mismatch')
        for cam, keys in cameras.items():
            key = f'observation.images.{cam}'
            expected = np.asarray(converter._get_nested(item, *keys[0]))
            require(expected.shape == (count, 240, 320, 3), 'Wrong camera source dimensions')
            video_path = root / info['video_path'].format(
                video_key=key, chunk_index=int(meta[f'videos/{key}/chunk_index']),
                file_index=int(meta[f'videos/{key}/file_index']))
            start_ts = float(meta[f'videos/{key}/from_timestamp'])
            require(abs(float(meta[f'videos/{key}/to_timestamp']) - start_ts - count / 15) < 1e-5,
                    'Episode video duration mismatch')
            for start in range(0, count, 32):
                size = min(32, count - start)
                actual = readers[cam].read(video_path, start_ts + start / 15, size).astype(np.float32)
                target = expected[start:start + size].astype(np.float32)
                require(actual.shape == target.shape, 'Decoded camera dimensions mismatch')
                mae = np.abs(actual - target).mean(axis=(1, 2, 3))
                swapped = np.abs(actual - target[..., ::-1]).mean(axis=(1, 2, 3))
                require(bool((mae < 12).all()), 'Video MAE exceeds original gate')
                require(bool((mae <= swapped + .5).all()), 'Suspected RGB channel swap')
                maximum = max(maximum, float(mae.max()))
        sources.append(dict(path=str(path), sha256=row['sha256'], frames=count))
        offset += count
        if (episode + 1) % 10 == 0 or episode + 1 == len(rows):
            update_progress('running', episode + 1)
    for reader in readers.values():
        reader.finish()
        require(sum(r['frames'] for r in reader.report) == offset, 'Per-camera final count mismatch')
    require(not list(root.glob('tmp*')), 'Temporary encoding files remain')
    report = dict(status='passed', converter=str(PINNED), converter_sha256=digest(PINNED),
                  audit_method='Independent sequential finalized-file audit; every RGB frame and timestamp checked',
                  audit_script=str(Path(__file__).resolve()), audit_script_sha256=digest(__file__),
                  sources=sources, episodes=len(rows),
                  frames=offset, fps=15, state_action_dim=14, maximum_video_mae=maximum,
                  all_frames_decoded=True, state_action_all_cells_exact=True,
                  encoding={k: v.get('info', {}) for k, v in info['features'].items() if v['dtype'] == 'video'},
                  final_video_audits={cam: reader.report for cam, reader in readers.items()},
                  features=info['features'], elapsed_s=time.monotonic() - started,
                  limitation='Original per-episode in-memory encoder report was not persisted after audit interruption; final files independently checked in full')
    update_progress('passed', len(rows))
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh audit JSON, dataset is read-only')
    parser.add_argument('--progress', type=Path, help='Atomic progress JSON updated every ten episodes')
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing existing audit output')
    manifest = json.loads(args.split_manifest.read_text())
    validate_split(manifest)
    # Same source order as the converter; checked against parquet and metadata.
    rows = sorted((r for r in manifest['episodes'] if r['split'] == args.split),
                  key=lambda r: str(r['input_link']))
    try:
        report = audit_finalized(args.dataset, rows, args.progress)
    except Exception as error:
        if args.progress is not None:
            value = json.loads(args.progress.read_text()) if args.progress.exists() else {}
            value.update(status='failed', error=str(error), updated_at_unix=time.time())
            temporary = args.progress.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(value, indent=2) + '\n')
            temporary.replace(args.progress)
        raise
    report.update(source_split_manifest=str(args.split_manifest.resolve()),
                  split_sha256=digest(args.split_manifest))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('SEQUENTIAL_AUDIT_PASSED', args.output, 'frames', report['frames'], flush=True)


if __name__ == '__main__':
    main()
