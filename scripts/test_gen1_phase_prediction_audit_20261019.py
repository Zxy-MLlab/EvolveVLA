"""CPU tests ensure terminal padding and absent release crossings are explicit."""
import unittest
import numpy as np
import gen1_phase_prediction_audit_20261019 as audit
from gen1_phase_anchor_metadata_20261019 import crossing_events


class PhaseMetricsTest(unittest.TestCase):
    def test_perfect_actions_and_invalid_tail_are_excluded(self):
        reference = np.zeros((12, 14), dtype=np.float32)
        reference[7:, 13] = 1
        prediction = np.concatenate([reference, np.full((38, 14), 99)])
        anchor = dict(active_joint_axes=list(range(7, 13)), gripper_axis=13,
                      source_release_start_offset=7)
        metrics = audit.chunk_metrics(prediction, reference, anchor)
        self.assertEqual(metrics['windows']['50']['valid_actions'], 12)
        self.assertEqual(metrics['windows']['50']['active_joint_mae_radians'], 0)
        self.assertEqual(metrics['opening_crossings']['02']['signed_sample_shift'], 0)

    def test_no_predicted_release_is_censored_not_false_zero_error(self):
        reference = np.zeros((50, 14), dtype=np.float32)
        reference[10:, 6] = 1
        anchor = dict(active_joint_axes=list(range(6)), gripper_axis=6,
                      source_release_start_offset=10)
        metrics = audit.chunk_metrics(np.zeros((50, 14)), reference, anchor)
        crossing = metrics['opening_crossings']['02']
        self.assertEqual(crossing['source_offset'], 10)
        self.assertIsNone(crossing['signed_sample_shift'])
        self.assertTrue(crossing['prediction_censored'])

    def test_opening_without_prior_closure_is_not_placement_release(self):
        values = [.0, .1, .3, 1., 1., .7, .1, .0, .0, .1, .3, .6, 1.]
        events = crossing_events(values, 'right')
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['closure_crossing_02'], 6)
        self.assertEqual(events[0]['release_start'], 9)
        self.assertEqual(events[0]['release_crossing_02'], 10)
        self.assertEqual(events[0]['release_end'], 12)


if __name__ == '__main__':
    unittest.main()
