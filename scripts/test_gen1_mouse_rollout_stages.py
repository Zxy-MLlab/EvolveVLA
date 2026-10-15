import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen1_mouse_rollout_stages import stages


class StageTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for i, (cmd, z) in enumerate(((1., .74), (.1, .74), (.1, .84), (1., .74))):
            rows.append(dict(step=i, physical=dict(
                left=dict(drive_targets=[0]*6+[cmd]),
                right=dict(drive_targets=[0]*6+[1.]),
                mouse=dict(position=[0., 0., z], quaternion=[.7, .7, 0., 0.]),
                target=dict(position=[0., 0., .74]))))
        result = dict(status='completed', task='place_mouse_pad', steps=3, seed=1,
            checkpoint='fixture', success=True, binding=dict(fields={'{a}':'left'}))
        return rows, result

    def test_closure_release_and_short_window(self):
        rows, result = self.fixture()
        record = stages(rows, result)
        self.assertEqual(record['closures'][0]['observed_followup_steps'], 2)
        self.assertAlmostEqual(record['closures'][0]['max_height_gain_next_50'], .1)
        self.assertTrue(record['openings'][0]['preceded_by_closure'])
        self.assertTrue(record['reconstructed_terminal_success'])

    def test_strict_position_boundary(self):
        rows, result = self.fixture()
        rows[-1]['physical']['mouse']['position'][0] = .015
        self.assertFalse(stages(rows, result)['terminal_predicates']['position'])

    def test_incomplete_trace_rejected(self):
        rows, result = self.fixture()
        with self.assertRaises(ValueError):
            stages(rows[:-1], result)

    def test_no_closure(self):
        rows, result = self.fixture()
        for row in rows:
            row['physical']['left']['drive_targets'][-1] = 1.
        self.assertEqual(stages(rows, result)['closures'], [])


if __name__ == '__main__':
    unittest.main()
