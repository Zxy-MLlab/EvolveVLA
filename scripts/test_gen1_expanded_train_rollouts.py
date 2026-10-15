"""Ensure evaluation waits for committed saves and complete paired results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gen1_expanded_train_rollouts_20261015 as queue


class QueueTests(unittest.TestCase):
    def test_uncommitted_and_foreign_checkpoints_are_not_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'checkpoints' / queue.RUN / '1000'
            target.mkdir(parents=True)
            with patch.object(queue, 'ROOT', root):
                self.assertFalse(queue.checkpoint_ready(1000))
                provenance = dict(step=1000, frozen_parameters_unchanged=True,
                                  adapter_restore_exact=True, run=str(queue.OUTPUTS / queue.RUN))
                (target / 'provenance.json').write_text(json.dumps(provenance))
                self.assertTrue(queue.checkpoint_ready(1000))
                provenance['adapter_restore_exact'] = False
                (target / 'provenance.json').write_text(json.dumps(provenance))
                with self.assertRaisesRegex(ValueError, 'Invalid checkpoint'):
                    queue.checkpoint_ready(1000)
                provenance.update(adapter_restore_exact=True, run='different-run')
                (target / 'provenance.json').write_text(json.dumps(provenance))
                with self.assertRaisesRegex(ValueError, 'Invalid checkpoint'):
                    queue.checkpoint_ready(1000)

    def test_incomplete_rollout_cannot_count_as_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'comparison.json'
            result = dict(tasks={task: dict(paired_completed=10) for task in
                                 ['place_mouse_pad', 'hanging_mug']}, outcome_assessment={'total_pairs': 20})
            path.write_text(json.dumps(result))
            self.assertEqual(queue.checked_result(path), {'total_pairs': 20})
            result['tasks']['hanging_mug']['paired_completed'] = 9
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'ten valid pairs'):
                queue.checked_result(path)


if __name__ == '__main__':
    unittest.main()
