import unittest

import numpy as np

from gen1_training_event_audit import event_windows


class EventWindowsTests(unittest.TestCase):
    def test_closing_and_opening_are_arm_specific(self):
        targets = np.ones((60, 2))
        targets[10:, 0] = 0
        targets[50:, 1] = 0
        targets[55:, 0] = 1
        windows = event_windows(targets, horizon=5)
        self.assertTrue(windows['closing'][6, 0])
        self.assertFalse(windows['closing'][5, 0])
        self.assertFalse(windows['closing'][10, 0])
        self.assertTrue(windows['closing'][46, 1])
        self.assertTrue(windows['opening'][51, 0])
        self.assertFalse(windows['opening'][50, 1])

    def test_terminal_window_has_no_wrapped_events(self):
        targets = np.ones((3, 2))
        targets[1:, 0] = 0
        windows = event_windows(targets, horizon=50)
        self.assertTrue(windows['closing'][0, 0])
        self.assertFalse(windows['closing'][1:, 0].any())


if __name__ == '__main__':
    unittest.main()
