import unittest
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen1_full_sampler_event_audit import replay_indices, event_flags
from gen1_multitask_data import TaskBalancedSampler


class ReplayTests(unittest.TestCase):
    def test_real_dataloader_epoch_restart_and_drop_last(self):
        for size in (16, 19):
            rows = [dict(split='train', task='a', start=0, stop=7, frames=7),
                    dict(split='train', task='b', start=7, stop=size, frames=size-7)]
            for steps in (1, 4, 5, 13):
                with self.subTest(size=size, steps=steps):
                    sampler = TaskBalancedSampler(rows, size, seed=1)
                    loader = torch.utils.data.DataLoader(list(range(size)),
                        batch_size=4, sampler=sampler, drop_last=True, num_workers=0)
                    expected = []
                    while len(expected) < steps * 4:
                        for batch in loader:
                            expected.extend(batch.tolist())
                            if len(expected) == steps * 4:
                                break
                    self.assertEqual(list(replay_indices(rows, steps, 4)), expected)

    def test_event_flags_against_direct_slicing(self):
        rng = np.random.default_rng(19)
        for size in (1, 2, 49, 50, 51, 100):
            values = rng.random((size, 2))
            for horizon in (1, 10, 50):
                flags = event_flags(values, horizon)
                for start in range(size):
                    changes = np.diff(values[start:start+horizon], axis=0)
                    np.testing.assert_array_equal(flags['opening'][start], (changes > .05).any(axis=0))
                    np.testing.assert_array_equal(flags['closing'][start], (changes < -.05).any(axis=0))


if __name__ == '__main__':
    unittest.main()
