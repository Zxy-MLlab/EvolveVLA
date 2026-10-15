"""CPU boundary tests of the production phase_weights function."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa

from gen1_multitask_train import phase_weights


class PhaseWindowTests(unittest.TestCase):
    def weights(self, actions, lengths, mode):
        rows, offset = [], 0
        for length in lengths:
            rows.append(dict(frames=length, start=offset, stop=offset + length))
            offset += length
        table = pa.table({'action': actions.tolist()})
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(lerobot_home=Path(directory), repo_id='fixture',
                                   provenance_dir=Path(directory), phase_window=mode)
            with patch('pyarrow.parquet.read_table', return_value=table):
                result = phase_weights(args, rows)
            report = json.loads((Path(directory) / 'phase_sampling.json').read_text())
        return result, report

    def actions(self, length):
        values = np.zeros((length, 14), dtype=np.float32)
        values[:, [6, 13]] = 1
        return values

    def test_prepost_contains_closure_and_thirty_post_frames(self):
        actions = self.actions(200)
        actions[70:, 6] = 0
        weights, report = self.weights(actions, [200], 'prepost')
        # At c=70, starts 21..70 contain c in their 50-action chunks;
        # post-closure observations are c..c+29. Their union is 21..99.
        expected = np.ones(200)
        expected[21:100] += 1
        expected[160:] += .5
        np.testing.assert_array_equal(weights, expected)
        self.assertEqual(report['event_frames'], 79)

    def test_post_preserves_existing_run_exactly(self):
        actions = self.actions(200)
        actions[70:, 6] = 0
        weights, _ = self.weights(actions, [200], 'post')
        expected = np.ones(200)
        expected[69:99] += 1
        expected[160:] += .5
        np.testing.assert_array_equal(weights, expected)

    def test_no_cross_episode_transition(self):
        actions = self.actions(20)
        actions[10:, 6] = 0
        for mode in ('post', 'prepost'):
            weights, report = self.weights(actions, [10, 10], mode)
            self.assertEqual(report['event_frames'], 0)
            expected = np.ones(20)
            expected[[8, 9, 18, 19]] = 1.5
            np.testing.assert_array_equal(weights, expected)

    def test_opening_only_does_not_mark_closure(self):
        actions = self.actions(20)
        actions[:10, 6] = 0
        _, report = self.weights(actions, [20], 'prepost')
        self.assertEqual(report['event_frames'], 0)

    def test_short_and_terminal_windows_are_clipped(self):
        for length in (2, 10, 49, 50, 51):
            actions = self.actions(length)
            actions[-1, [6, 13]] = 0
            weights, report = self.weights(actions, [length], 'prepost')
            self.assertEqual(report['event_frames'], min(length, 50))
            self.assertTrue(np.all((weights >= 1) & (weights <= 3)))


if __name__ == '__main__':
    unittest.main()
