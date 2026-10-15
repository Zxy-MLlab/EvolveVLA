"""Exercise real video boundaries and fail-closed sequential decoding."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

from gen1_multitask_audit import SequentialVideo


class SequentialTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='sequential-audit-test-')
        self.addCleanup(self.directory.cleanup)
        self.paths = [Path(self.directory.name) / f'{i}.mp4' for i in range(2)]
        for path in self.paths:
            with av.open(str(path), 'w') as output:
                stream = output.add_stream('libx264', rate=15)
                stream.width = stream.height = 32
                stream.pix_fmt = 'yuv420p'
                for value in range(6):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((32, 32, 3), value * 30, dtype=np.uint8), format='rgb24')
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)

    def reader(self):
        reader = SequentialVideo()
        self.addCleanup(lambda: reader.container.close() if reader.container is not None else None)
        return reader

    def test_episode_chunks_and_file_switch_open_each_file_once(self):
        reader = self.reader()
        with patch('av.open', wraps=av.open) as opened:
            for path in self.paths:
                first = reader.read(path, 0, 2)
                last = reader.read(path, 2 / 15, 4)
                np.testing.assert_allclose(first[:, 0, 0, 0], [0, 30], atol=2)
                np.testing.assert_allclose(last[:, 0, 0, 0], [60, 90, 120, 150], atol=2)
            reader.finish()
            self.assertEqual(opened.call_count, 2)
        self.assertEqual([r['frames'] for r in reader.report], [6, 6])

    def test_missing_frame_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing video frame'):
            self.reader().read(self.paths[0], 0, 7)

    def test_extra_frame_rejected(self):
        reader = self.reader()
        reader.read(self.paths[0], 0, 5)
        with self.assertRaisesRegex(ValueError, 'Unaccounted video frames'):
            reader.finish()

    def test_wrong_episode_offset_rejected(self):
        reader = self.reader()
        reader.read(self.paths[0], 0, 2)
        with self.assertRaisesRegex(ValueError, 'offset mismatch'):
            reader.read(self.paths[0], 3 / 15, 2)


if __name__ == '__main__':
    unittest.main()
