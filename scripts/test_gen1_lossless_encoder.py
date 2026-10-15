#!/usr/bin/env python3
"""CPU tests for offline encoder backpressure, failure gates and actual AV1."""
import importlib.util
import inspect
from pathlib import Path
import queue
import tempfile
import threading
import time
import types
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('gen1_conversion', Path(__file__).with_name('gen1_convert_ready.py'))
CONVERSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERSION)


class Worker:
    def __init__(self, alive=True):
        self.alive = alive

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass


def fake(alive=True, stall=.5):
    encoder = types.SimpleNamespace(
        _episode_active=False, _frame_queues={'cam': queue.Queue(maxsize=1)},
        _result_queues={'cam': queue.Queue()}, _threads={'cam': Worker(alive)},
        _stop_events={}, _dropped_frames={})
    encoder.start_episode = lambda keys, temp: setattr(encoder, '_episode_active', True)
    CONVERSION.install_lossless_encoder(encoder, stall)
    encoder.start_episode(['cam'], None)
    return encoder


class LosslessTests(unittest.TestCase):
    def test_backpressure_and_copy(self):
        encoder = fake()
        encoder._frame_queues['cam'].put('prior')
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        consumed = []
        def drain():
            time.sleep(.15)
            consumed.append(encoder._frame_queues['cam'].get())
        thread = threading.Thread(target=drain)
        thread.start()
        encoder.feed_frame('cam', image)
        thread.join()
        image[:] = 255
        self.assertEqual(consumed, ['prior'])
        self.assertEqual(encoder._frame_queues['cam'].get().sum(), 0)
        self.assertFalse(encoder._dropped_frames)

    def test_dead_worker(self):
        encoder = fake(False)
        encoder._result_queues['cam'].put(('error', 'test worker failure'))
        with self.assertRaisesRegex(RuntimeError, 'test worker failure'):
            encoder.feed_frame('cam', np.zeros((4, 4, 3)))

    def test_stall_timeout(self):
        encoder = fake(stall=.03)
        encoder._frame_queues['cam'].put('prior')
        with self.assertRaisesRegex(RuntimeError, 'stalled'):
            encoder.feed_frame('cam', np.zeros((4, 4, 3)))

    def test_finish_sentinel_stall(self):
        encoder = fake(stall=.03)
        encoder._frame_queues['cam'].put('prior')
        with self.assertRaisesRegex(RuntimeError, 'stalled'):
            encoder.finish_episode()

    def test_drop_guard(self):
        encoder = fake()
        encoder._dropped_frames['cam'] = 1
        with self.assertRaisesRegex(RuntimeError, 'Dropped frames'):
            encoder.finish_episode()

    def test_actual_av1(self):
        from lerobot.datasets.video_utils import StreamingVideoEncoder
        from gen1_multitask_audit import SequentialVideo
        if 'rgb_encoder' in inspect.signature(StreamingVideoEncoder).parameters:
            from lerobot.configs.video import RGBEncoderConfig
            options = dict(rgb_encoder=RGBEncoderConfig(vcodec='libsvtav1', crf=18))
        else:
            options = dict(crf=18)
        encoder = StreamingVideoEncoder(fps=15, queue_maxsize=1, encoder_threads=2, **options)
        audits = CONVERSION.install_lossless_encoder(encoder)
        root = Path(tempfile.mkdtemp(prefix='lossless-test-', dir='/root/autodl-tmp/EvolveVLA/tmp/gen1_training'))
        for episode in range(2):
            encoder.start_episode(['cam'], root)
            for index in range(12):
                encoder.feed_frame('cam', np.full((64, 64, 3), index * 10, dtype=np.uint8))
            result = encoder.finish_episode()
            self.assertTrue(result['cam'][0].is_file())
            reader = SequentialVideo()
            try:
                pixels = reader.read(result['cam'][0], 0, 12)
                np.testing.assert_allclose(pixels[:, 0, 0, 0], np.arange(12) * 10, atol=3)
                reader.finish()
            finally:
                if reader.container is not None:
                    reader.container.close()
        self.assertEqual(len(audits), 2)
        self.assertTrue(all(a['cam']['frames'] == 12 for a in audits))
        print('Preserved AV1 test videos:', root, flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
