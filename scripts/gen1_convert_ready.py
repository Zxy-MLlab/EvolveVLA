#!/usr/bin/env python3
"""Fail-closed CPU wrapper around pinned v3 converter; no legacy Pi05 processing."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import queue
import time
import types


def install_lossless_encoder(encoder, stall_seconds=120):
    """Offline backpressure: never trade a missing frame for conversion speed."""
    counts = {}
    audits = []
    original_start = encoder.start_episode

    def start(self, video_keys, temp_dir, depth_video_keys=None):
        if self._episode_active:
            raise RuntimeError('Refusing to discard an active encoding episode')
        counts.clear()
        counts.update(dict.fromkeys(video_keys, 0))
        # LeRobot 0.6 added an optional depth-video argument. Keep the
        # adapter compatible with both that API and the older two-argument
        # streaming encoder used by the pinned converter.
        try:
            original_start(video_keys, temp_dir, depth_video_keys=depth_video_keys)
        except TypeError as exc:
            if 'depth_video_keys' not in str(exc):
                raise
            original_start(video_keys, temp_dir)

    def enqueue(self, key, value):
        deadline = time.monotonic() + stall_seconds
        while True:
            if not self._threads[key].is_alive():
                try:
                    result = self._result_queues[key].get_nowait()
                except queue.Empty:
                    result = 'no worker result'
                raise RuntimeError(f'Encoder stopped before enqueue: {key}: {result}')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f'Encoder queue stalled: {key}')
            try:
                self._frame_queues[key].put(value, timeout=min(0.1, remaining))
                return
            except queue.Full:
                continue

    def feed(self, video_key, image):
        if not self._episode_active:
            raise RuntimeError('No active encoding episode')
        enqueue(self, video_key, image.copy())
        counts[video_key] += 1

    def finish(self):
        import av
        if not self._episode_active:
            raise RuntimeError('No active encoding episode')
        if any(self._dropped_frames.values()):
            raise RuntimeError(f'Dropped frames: {self._dropped_frames}')
        for key in self._frame_queues:
            enqueue(self, key, None)
        results = {}
        episode_audit = {}
        for key, thread in self._threads.items():
            thread.join(timeout=stall_seconds)
            if thread.is_alive():
                raise RuntimeError(f'Encoder finish stalled: {key}')
            try:
                status, data = self._result_queues[key].get(timeout=1)
            except queue.Empty as exc:
                raise RuntimeError(f'Missing encoder result: {key}') from exc
            if status != 'ok':
                raise RuntimeError(f'Encoder failed: {key}: {data}')
            path = self._video_paths[key]
            with av.open(str(path)) as video:
                stream = video.streams.video[0]
                if stream.average_rate != self.fps or stream.duration is None:
                    raise RuntimeError(f'Invalid encoded timing: {key}')
                duration = float(stream.duration * stream.time_base)
                if abs(duration - counts[key] / self.fps) > 1e-5:
                    raise RuntimeError(f'Encoded duration mismatch: {key}')
                decoded = 0
                for frame in video.decode(stream):
                    if frame.pts is None or abs(float(frame.pts * frame.time_base) - decoded / self.fps) > 1e-5:
                        raise RuntimeError(f'Encoded timestamp mismatch: {key}: {decoded}')
                    decoded += 1
                if decoded != counts[key] or decoded < 2:
                    raise RuntimeError(f'Encoded frame count mismatch: {key}: {decoded} != {counts[key]}')
            episode_audit[key] = {'frames': decoded, 'duration': duration}
            results[key] = (path, data)
        audits.append(episode_audit)
        self._cleanup()
        self._episode_active = False
        return results

    def preserve_cancel(self):
        # Stop only these workers; leave failed temporary videos for diagnosis.
        for event in self._stop_events.values():
            event.set()
        for thread in self._threads.values():
            thread.join(timeout=5)
        self._episode_active = False

    encoder.start_episode = types.MethodType(start, encoder)
    encoder.feed_frame = types.MethodType(feed, encoder)
    encoder.finish_episode = types.MethodType(finish, encoder)
    encoder.cancel_episode = types.MethodType(preserve_cancel, encoder)
    return audits


ROOT = Path('/root/autodl-tmp/EvolveVLA')
PINNED = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/scripts/transform_lerobot_v30_format.py'

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--lerobot-home', type=Path, required=True)
    p.add_argument('--repo-id', required=True)
    p.add_argument('--min-free-gib', type=float, default=60,
                   help='Stop before conversion if remaining free space falls below this reserve')
    a = p.parse_args()
    if '/' in a.repo_id or a.repo_id in ('', '.', '..'):
        p.error('repo-id must be a single safe path component')
    home = a.lerobot_home.resolve()
    target = home / a.repo_id
    if target.exists() or target.is_symlink():
        p.error('REFUSING existing target (pinned converter would delete it)')
    if not home.is_relative_to(ROOT / 'data'):
        p.error('output must be on data disk below EvolveVLA/data')
    if a.min_free_gib < 15 or shutil.disk_usage(ROOT).free < a.min_free_gib * 2**30:
        p.error('invalid reserve or insufficient free space')
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', HF_LEROBOT_HOME=str(home),
                      HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled',
                      TMPDIR=str(ROOT / 'tmp/gen1_training'))
    import numpy as np
    import h5py
    spec = importlib.util.spec_from_file_location('gen1_pinned_converter', PINNED)
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    files = converter.find_input_files(a.input_dir)
    if not files:
        p.error('no newly collected HDF5 inputs')
    source = []
    for path in files:
        d = converter.load(str(path), data_type='RoboDojo', data_version='v1.0')
        state, action = converter._extract_qpos(d), converter._extract_action(d)
        assert state.shape == action.shape and state.shape[1] == 14 and len(state) > 1
        assert np.isfinite(state).all() and np.isfinite(action).all()
        np.testing.assert_array_equal(state[1:], action[:-1])
        assert converter._find_instructions(d), path
        assert int(np.asarray(d['additional_info']['frequency'])) == 15
        images = {cam: converter._get_nested(d, *keys[0]) for cam, keys in converter.CAMERA_CANDIDATES.items()}
        shapes = {cam: np.asarray(img).shape for cam, img in images.items()}
        assert all(shape == (len(state), 240, 320, 3) for shape in shapes.values()), shapes
        with h5py.File(path, 'r') as f:
            assert f.attrs['source_format'] == 'RoboTwin'
        source.append({'path': str(path.resolve()), 'sha256': hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest(),
                       'frames': len(state)})
    home.mkdir(parents=True, exist_ok=True)
    # Serialize wrappers and forbid the destructive branch even if target appears after preflight.
    lock = home / (a.repo_id + '.conversion.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    def reject_delete(*args, **kwargs):
        raise RuntimeError('Existing target detected; deletion forbidden')
    converter.shutil = type('NoDelete', (), {'rmtree': staticmethod(reject_delete)})
    try:
        cfg = dataclasses.replace(converter.DEFAULT_DATASET_CONFIG,
            image_writer_processes=0, image_writer_threads=2, video_backend='pyav')
        # Keep pinned features, motor names, robot_type and source 15Hz indexing exactly.
        ds = converter.create_empty_dataset(a.repo_id, 'unified_robot',
            converter._build_motor_names_from_dims([7, 7]), 15,
            image_height=240, image_width=320, dataset_config=cfg)
        # LeRobot 0.6.1 stores the streaming encoder on DatasetWriter, while
        # the pinned RoboTwin converter expects the pre-0.6 dataset alias.
        # Point the legacy alias at the exact same encoder object; no data or
        # converter behavior is changed.
        if not hasattr(ds, '_streaming_encoder'):
            writer = getattr(ds, 'writer', None)
            encoder = getattr(writer, '_streaming_encoder', None)
            if encoder is None:
                raise RuntimeError('LeRobot streaming encoder was not initialized')
            ds._streaming_encoder = encoder
        converter.configure_video_encoding(ds, cfg)
        encoder_audits = install_lossless_encoder(ds._streaming_encoder)
        random.seed(0)
        try:
            for path in files:
                if shutil.disk_usage(ROOT).free < a.min_free_gib * 2**30:
                    raise RuntimeError('conversion disk reserve exhausted')
                converter.convert_one(path, ds, 'RoboDojo', 'v1.0', [7, 7], [7, 7], 240, 320)
        finally:
            ds.finalize()  # v0.4.4 flushes data parquet and final episode metadata.
            converter.finalize_dataset(ds)
        from gen1_multitask_audit import audit_finalized
        report = audit_finalized(target, source,
                                 home / (a.repo_id + '.audit_progress.json'))
        assert len(encoder_audits) == len(source)
        for row, encoded in zip(source, encoder_audits, strict=True):
            assert len(encoded) == 3 and all(v['frames'] == row['frames'] for v in encoded.values())
        report.pop('limitation', None)  # This live conversion retained its encoder records.
        report.update(lossless_encoder_audits=encoder_audits,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (target / 'conversion_audit.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        lock.unlink()

if __name__ == '__main__':
    main()
