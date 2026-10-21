"""CPU tests for automatic terminal-gripper A/B selection."""
import ast
import unittest
from pathlib import Path
import numpy as np

import gen1_terminal_gripper_ab_20261020 as ab
import gen1_expert_label_replay_20261019 as core
import gen1_expert_gripper_timing_replay_20261019 as timing


class TerminalReleaseTests(unittest.TestCase):
    def arrays(self):
        state = np.zeros((12, 14), dtype=np.float32)
        action = state.copy()
        # Early opening run (must not be selected).
        action[2, 6] = .05
        action[3, 6] = .10
        action[4, 6] = .10
        state[3] = action[2]
        state[4] = action[3]
        # Later terminal release run, with a stationary arm and both grippers
        # monotonic.  The final transition is deliberately index 10, proving
        # selection does not rely on the old hard-coded 319.
        for i, value in enumerate((.15, .20, .25), 8):
            state[i] = action[i-1]
            action[i, 13] = value
        state[11] = action[10]
        return state, action

    def test_selects_last_contiguous_release(self):
        state, action = self.arrays()
        result = ab.terminal_release(state, action)
        self.assertEqual((result['start'], result['end'], result['length']), (8, 10, 3))
        self.assertNotEqual(result['start'], 319)
        self.assertEqual([(x['start'], x['end']) for x in result['all_intervals']], [(2, 3), (8, 10)])

    def test_negative_or_short_release_is_rejected(self):
        state = np.zeros((3, 14), dtype=np.float32)
        action = state.copy()
        action[0, 6] = .1
        action[1, 6] = .05  # closing breaks monotonicity
        action[2, 6] = .2
        state[1] = action[0]
        state[2] = action[1]
        with self.assertRaises(ValueError):
            ab.terminal_release(state, action)
        action = state.copy()
        action[1, 6] = .1
        state[2] = action[1]
        with self.assertRaises(ValueError):
            ab.terminal_release(state, action)

    def test_allowed_index_and_runtime_drive_target_guard(self):
        source = np.zeros(14)
        target = source.copy()
        target[13] = .1
        current = source.copy()
        self.assertTrue(ab._retime_eligible(5, current, target, source, {5}))
        self.assertFalse(ab._retime_eligible(4, current, target, source, {5}))
        current[2] = .01
        self.assertFalse(ab._retime_eligible(5, current, target, source, {5}))

    def test_panel_quota_is_stable(self):
        rows = [dict(id=f'{task}_{i}', task=task) for task in
                ('place_mouse_pad', 'hanging_mug') for i in range(10)]
        selected = ab._prepare_cases({'cases': rows}, 10)
        self.assertEqual(len(selected), 20)
        self.assertEqual(sum(x['task'] == 'hanging_mug' for x in selected), 10)
        self.assertEqual(sum(x['task'] == 'place_mouse_pad' for x in selected), 10)

    def test_patch_changes_only_four_qpos_literals(self):
        source = core.ROOT/'robotwin_repo_ba0fd21/envs/_base_task.py'
        modified, report = timing.patch_fallback_function(source, 15)
        self.assertEqual(len(report['replacements']), 4)
        self.assertEqual(sorted(x['variable'] for x in report['replacements']),
                         ['left_n_step', 'left_n_step', 'right_n_step', 'right_n_step'])
        original_tree = ast.parse(source.read_text(), filename=str(source))
        original = next(n for n in ast.walk(original_tree)
                        if isinstance(n, ast.FunctionDef) and n.name == 'take_action')
        self.assertNotEqual(ast.dump(original), ast.dump(modified))


if __name__ == '__main__':
    unittest.main()
