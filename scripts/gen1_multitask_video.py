"""Bounded process-local PyAV readers, checked against the production loader.

Keeps a few decoder containers open and caches a bounded number of decoded uint8
requests. RGB pixels, timestamps, tolerance, dimensions and dataset files remain
unchanged. Each registered file must pass exact pixel parity before installation.
"""
from collections import OrderedDict
import importlib
from pathlib import Path
from threading import RLock


class BoundedVideoReader:
    def __init__(self, paths, fallback, max_readers=8, max_bytes=64 * 2**20, seek_margin=0):
        self.paths = {str(Path(path).resolve()) for path in paths}
        self.fallback = fallback
        self.max_readers, self.max_bytes = max_readers, max_bytes
        self.readers, self.frames = OrderedDict(), OrderedDict()
        self.used = self.hits = self.requests = 0
        self.seek_margin = seek_margin
        self.lock = RLock()

    def _reader(self, path):
        import av
        if path in self.readers:
            self.readers.move_to_end(path)
            return self.readers[path]
        while len(self.readers) >= self.max_readers:
            _, (container, _) = self.readers.popitem(last=False)
            container.close()
        container = av.open(path)
        stream = container.streams.video[0]
        stream.thread_count = 2
        stream.thread_type = 'SLICE'
        self.readers[path] = (container, stream)
        return container, stream

    def decode(self, video_path, timestamps, tolerance_s, backend=None, **kwargs):
        with self.lock:
            return self._decode(video_path, timestamps, tolerance_s, backend, **kwargs)

    def _decode(self, video_path, timestamps, tolerance_s, backend=None, **kwargs):
        import numpy as np
        import torch
        path = str(Path(video_path).resolve())
        if (path not in self.paths or backend not in (None, 'pyav')
                or kwargs.get('is_depth', False)
                or set(kwargs) - {'return_uint8', 'is_depth'}):
            return self.fallback(video_path, timestamps, tolerance_s, backend, **kwargs)
        if not timestamps:
            raise ValueError('Empty video request')
        key = (path, tuple(float(t) for t in timestamps), float(tolerance_s))
        self.requests += 1
        if key in self.frames:
            self.frames.move_to_end(key)
            self.hits += 1
            pixels = self.frames[key]
        else:
            container, stream = self._reader(path)
            first, last = min(timestamps), max(timestamps)
            offset = int(round(max(0, first) / stream.time_base)) - self.seek_margin
            container.seek(offset, stream=stream, backward=True, any_frame=False)
            times, decoded = [], []
            for frame in container.decode(stream):
                ts = float(frame.pts * frame.time_base)
                times.append(ts)
                decoded.append(frame.to_ndarray(format='rgb24'))
                if ts >= last:
                    break
            if not decoded:
                raise ValueError(f'No frames decoded: {path}')
            # Same float32 nearest-timestamp calculation and strict tolerance
            # as lerobot.decode_video_frames_torchvision.
            distances = torch.cdist(torch.tensor(timestamps)[:, None], torch.tensor(times)[:, None], p=1)
            errors, indices = distances.min(1)
            if not bool((errors < tolerance_s).all()):
                raise ValueError(f'Video timestamps exceed original tolerance: {path}: {errors}')
            pixels = np.stack([decoded[i] for i in indices.tolist()])
            if pixels.nbytes <= self.max_bytes:
                while self.used + pixels.nbytes > self.max_bytes:
                    _, removed = self.frames.popitem(last=False)
                    self.used -= removed.nbytes
                self.frames[key] = pixels
                self.used += pixels.nbytes
        tensor = torch.from_numpy(pixels.copy()).permute(0, 3, 1, 2)
        return tensor if kwargs.get('return_uint8', False) else tensor.float() / 255

    def report(self):
        return dict(max_readers=self.max_readers, open_readers=len(self.readers),
                    max_cache_bytes=self.max_bytes, cache_bytes=self.used,
                    requests=self.requests, cache_hits=self.hits, decoder_threads=2,
                    seek_margin=self.seek_margin)

    def close(self):
        for container, _ in self.readers.values():
            container.close()
        self.readers.clear()
        self.frames.clear()
        self.used = 0


def install(roots):
    import av
    import torch
    import lerobot.datasets.lerobot_dataset as dataset
    owner = dataset
    if not hasattr(owner, 'decode_video_frames'):
        owner = importlib.import_module('lerobot.datasets.dataset_reader')
    paths = sorted({p.resolve() for root in roots for p in (Path(root) / 'videos').rglob('*.mp4')})
    if not paths:
        raise ValueError('No finalized videos to install reader')
    original = owner.decode_video_frames
    reader = BoundedVideoReader(paths, original, seek_margin=0 if owner is dataset else 1)
    checks = []
    try:
        for path in paths:
            with av.open(str(path)) as video:
                stream = video.streams.video[0]
                count, fps = stream.frames, stream.average_rate
            if not count or not fps:
                raise ValueError('Missing frame count/rate')
            for index in sorted({0, 1, count // 4, count // 2, 3 * count // 4, count - 2, count - 1}):
                timestamp = float(index / fps)
                expected = original(path, [timestamp], .001, 'pyav')
                actual = reader.decode(path, [timestamp], .001, 'pyav')
                if not torch.equal(expected, actual):
                    raise ValueError(f'Original-loader pixel parity failed: {path}:{index}')
            # Also check an unsorted multi-timestamp request and cache hits.
            query = [float((count - 1) / fps), 0.0] if count < 512 else [float(3 / fps), float(1 / fps)]
            expected = original(path, query, .001, 'pyav')
            for _ in range(2):
                if not torch.equal(expected, reader.decode(path, query, .001, 'pyav')):
                    raise ValueError(f'Multi-frame pixel parity failed: {path}')
            checks.append(dict(path=str(path), frames=count, sampled_exact_parity=True))
            print('VIDEO_READER_PARITY', path, flush=True)
    except BaseException:
        reader.close()
        raise
    owner.decode_video_frames = reader.decode
    return reader, dict(videos=checks, **reader.report(),
                        scope='process-local bounded readers; original RGB and timestamp parity')
