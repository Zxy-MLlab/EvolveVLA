"""CPU control-path checks on the actual read-only RoboTwin function AST."""
import ast
import contextlib
import hashlib
import io
from pathlib import Path
import types
from typing import Literal
import unittest

import numpy as np
import gen1_duration_controller_20261020 as controller

SOURCE = Path(__file__).resolve().parents[1] / 'robotwin_repo_ba0fd21/envs/_base_task.py'


def original_function():
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'take_action')
    namespace = dict(np=np, Literal=Literal)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['take_action']


class Planner:
    def __init__(self, mode):
        self.mode, self.last = mode, None

    def TOPP(self, path, *args, **kwargs):
        if self.mode == 'exception':
            raise ValueError('Synthetic planner exception')
        n = 0 if self.mode == 'empty' else int(self.mode)
        positions = np.linspace(path[0], path[-1], n)
        velocities = np.linspace(np.arange(1., 7.), np.arange(2., 8.), n)
        self.last = dict(position=positions.copy(), velocity=velocities.copy())
        return np.arange(n)*.004, positions, velocities, [], n*.004


class Robot:
    def __init__(self, left, right, grippers):
        self.left_mplib_planner, self.right_mplib_planner = Planner(left), Planner(right)
        self.joints = {'left': np.zeros(6), 'right': np.zeros(6)}
        self.grippers = dict(zip(('left', 'right'), grippers, strict=True))
        self.arm_calls, self.gripper_calls = [], []
        self.drive_override = None

    def get_left_arm_jointState(self): return list(self.joints['left']) + [self.grippers['left']]
    def get_right_arm_jointState(self): return list(self.joints['right']) + [self.grippers['right']]
    def get_left_gripper_val(self): return self.grippers['left']
    def get_right_gripper_val(self): return self.grippers['right']
    def get_normal_real_gripper_val(self):
        return self.drive_override or [self.grippers['left'], self.grippers['right']]
    def set_arm_joints(self, positions, velocities, arm):
        self.arm_calls.append((positions.tolist(), velocities.tolist(), arm))
        self.joints[arm] = positions.copy()
    def set_gripper(self, value, arm):
        self.grippers[arm] = float(np.clip(value, 0, 1))
        self.gripper_calls.append((self.grippers[arm], arm))


class Scene:
    def __init__(self): self.count = 0
    def step(self): self.count += 1
    def get_timestep(self): return .004


def make_task(left=30, right=45, grippers=(1., 1.), stop_at=None):
    task = types.SimpleNamespace(robot=Robot(left, right, grippers), scene=Scene(),
        take_action_cnt=0, step_lim=1000, eval_success=False, eval_video_path=None,
        render_freq=0, _update_render=lambda: None, get_obs=lambda: None)
    task.check_success = lambda: stop_at is not None and task.scene.count >= stop_at
    task.take_action = types.MethodType(original_function(), task)
    return task


def action(left_gripper=1., right_gripper=1.):
    return np.array([.1, .2, .3, .4, .5, .6, left_gripper,
                     -.1, -.2, -.3, -.4, -.5, -.6, right_gripper])


def invoke(method, target):
    with contextlib.redirect_stdout(io.StringIO()):
        return method(target)


class DurationControllerTests(unittest.TestCase):
    def test_normal_topp_actual_ticks_endpoints_and_shared_velocity_scaling(self):
        task = make_task(30, 45)
        method, metadata, state = controller.create(task.take_action, 'fixed15')
        target = action(.6, .7)
        source_before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        invoke(method, target)
        self.assertEqual(task.scene.count, 15)
        for arm, scale, goal in [('left', 2., target[:6]), ('right', 3., target[7:13])]:
            calls = [r for r in task.robot.arm_calls if r[2] == arm]
            self.assertEqual(len(calls), 15)
            np.testing.assert_array_equal(calls[-1][0], goal)
            np.testing.assert_allclose(calls[-1][1], np.arange(2., 8.) * scale)
            row = state.last_record['arms'][arm]
            self.assertEqual(row['position_endpoint_max_abs'], 0.)
            self.assertTrue(row['all_finite'])
            self.assertEqual(row['velocity_scale'], scale)
        self.assertEqual(task.robot.gripper_calls[-2:], [(.6, 'left'), (.7, 'right')])
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), source_before)
        self.assertTrue(metadata['patch']['original_nodes_preserved'])

    def test_resampling_identity_and_analytic_linear_motion(self):
        current = np.array([0., 2.])
        positions = current + np.arange(1, 31)[:, None] * np.array([.004, -.008])
        velocities = np.tile([1., -2.], (30, 1))
        original = dict(position=positions, velocity=velocities)
        same, _ = controller.resample_path(original, current, 30, 30, .004)
        np.testing.assert_array_equal(same['position'], positions)
        np.testing.assert_array_equal(same['velocity'], velocities)
        changed, _ = controller.resample_path(original, current, 30, 15, .004)
        differences = np.diff(np.vstack((current, changed['position'])), axis=0) / .004
        np.testing.assert_allclose(differences, changed['velocity'], atol=1e-12)
        np.testing.assert_array_equal(changed['position'][-1], positions[-1])
        np.testing.assert_array_equal(original['position'], positions)

    def test_exception_empty_and_one_arm_fallback_never_add_arm_setters(self):
        for left, right in [('exception', 'empty'), ('empty', 45), (30, 'exception')]:
            with self.subTest(left=left, right=right):
                task = make_task(left, right)
                method, _, state = controller.create(task.take_action, 'fixed15')
                invoke(method, action(.5, .75))
                self.assertEqual(task.scene.count, 15)
                for arm, mode in [('left', left), ('right', right)]:
                    rows = [r for r in task.robot.arm_calls if r[2] == arm]
                    fallback = isinstance(mode, str)
                    self.assertEqual(len(rows), 0 if fallback else 15)
                    self.assertEqual(state.last_record['arms'][arm]['topp_success'], not fallback)
                self.assertEqual(task.robot.gripper_calls[-2:], [(.5, 'left'), (.75, 'right')])

    def test_original_and_inactive_opening_are_full_control_trace_identical(self):
        for left, right in [(22, 37), ('empty', 80), ('exception', 'empty')]:
            for mode in ('original', 'opening15'):
                with self.subTest(mode=mode, left=left, right=right):
                    a, b = make_task(left, right), make_task(left, right)
                    method, _, state = controller.create(b.take_action, mode)
                    for target in [action(.95, 1.), action(.6, .9), action(.25, .4)]:
                        invoke(a.take_action, target)
                        invoke(method, target)
                        self.assertFalse(state.last_record['active'])
                        self.assertEqual(a.scene.count, b.scene.count)
                        self.assertEqual(a.robot.arm_calls, b.robot.arm_calls)
                        self.assertEqual(a.robot.gripper_calls, b.robot.gripper_calls)

    def test_opening_starts_from_observed_drive_and_synchronizes_both_arms(self):
        task = make_task(30, 45, grippers=(0., 1.))
        method, _, state = controller.create(task.take_action, 'opening15')
        # Initial low-level command state differs from a previous requested
        # model endpoint. Read the drive target, not a model-target cache.
        task.robot.drive_override = [.02, 1.]
        invoke(method, action(.1, 1.))
        self.assertEqual(task.scene.count, 15)
        self.assertEqual(state.last_record['gate_selected_arms'], ['left'])
        self.assertEqual(state.last_record['gates']['left']['drive_target'], .02)
        self.assertTrue(state.last_record['arms']['right']['arm_path_resampled'])
        self.assertEqual(state.last_record['arms']['right']['velocity_scale'], 3.)

    def test_opening_has_no_preintervention_change_and_includes_terminal_action(self):
        original, task = make_task(), make_task()
        method, _, state = controller.create(task.take_action, 'opening15')
        for left in [.5, 0., .004, .04]:
            invoke(original.take_action, action(left, 1.))
            invoke(method, action(left, 1.))
            self.assertFalse(state.last_record['active'])
            self.assertEqual(original.robot.arm_calls, task.robot.arm_calls)
            self.assertEqual(original.robot.gripper_calls, task.robot.gripper_calls)
            self.assertEqual(original.scene.count, task.scene.count)
        for left in [.1, .4, .8]:
            before = task.scene.count
            invoke(method, action(left, 1.))
            self.assertTrue(state.last_record['active'])
            self.assertEqual(task.scene.count-before, 15)
        for left in [.81, .79, .82]:
            invoke(method, action(left, 1.))
            self.assertFalse(state.last_record['active'])

    def test_gate_reversal_rearm_and_jitter_are_explicit(self):
        gate = controller.OpeningGate()
        self.assertFalse(gate.observe(1., .9)['enabled'])
        self.assertFalse(gate.observe(0., .004)['enabled'])
        self.assertFalse(gate.observe(.004, .04)['enabled'])
        self.assertTrue(gate.observe(.04, .1)['enabled'])
        self.assertTrue(gate.observe(.1, .1-5e-7)['enabled'])
        self.assertFalse(gate.observe(.1, .08)['enabled'])
        self.assertFalse(gate.observe(.08, 0.)['enabled'])
        self.assertTrue(gate.observe(0., .1)['enabled'])
        self.assertTrue(gate.observe(.1, 1.2)['enabled'])
        self.assertFalse(gate.observe(.1, .99)['enabled'])
        # Even if drive remains low after a consumed large target, >.8 jitter
        # cannot rearm until a subsequent closed command is observed.
        self.assertFalse(gate.observe(.1, 1.)['enabled'])

    def test_true_success_stops_at_actual_tick_and_completed_episode_stays_stopped(self):
        for mode in controller.MODES:
            task = make_task('empty', 45, grippers=(0., 1.), stop_at=7)
            method, _, state = controller.create(task.take_action, mode)
            invoke(method, action(.1, 1.))
            self.assertEqual(task.scene.count, 7)
            self.assertTrue(task.eval_success)
            invoke(method, action(.2, 1.))
            self.assertEqual(task.scene.count, 7)
            self.assertFalse(state.last_record['qpos_hook_reached'])

    def test_nonfinite_and_invalid_paths_fail_before_emitting_commands(self):
        for bad in [np.array([[np.nan, 1.]]), np.empty((0, 2))]:
            with self.assertRaises(ValueError):
                controller.resample_path(dict(position=bad, velocity=bad), np.zeros(2), len(bad), 15, .004)


if __name__ == '__main__':
    unittest.main()
