"""CPU checks using the actual controller AST and the real wrapper composition."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
from typing import Literal
import unittest
from unittest.mock import patch
import numpy as np
import gen1_policy_timing_ab_20261020 as runner


def actual_method():
    tree = ast.parse(runner.SOURCE.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'take_action')
    namespace = {'np': np, 'Literal': Literal}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(runner.SOURCE), 'exec'), namespace)
    return namespace['take_action']


class Planner:
    def __init__(self, mode):
        self.mode = mode
    def TOPP(self, *a, **kw):
        if self.mode == 'exception':
            raise ValueError('Synthetic zero-motion rejection')
        count = 0 if self.mode == 'empty' else int(self.mode)
        return [], np.zeros((count, 6)), np.ones((count, 6)), [], 0.


class Robot:
    def __init__(self, left, right):
        self.left_mplib_planner, self.right_mplib_planner = Planner(left), Planner(right)
        self.arm_calls, self.gripper_calls = [], []
    def get_left_arm_jointState(self): return [0.] * 7
    def get_right_arm_jointState(self): return [0.] * 7
    def get_left_gripper_val(self): return 0.
    def get_right_gripper_val(self): return 0.
    def set_arm_joints(self, positions, velocity, arm):
        self.arm_calls.append((positions.tolist(), velocity.tolist(), arm))
    def set_gripper(self, value, arm): self.gripper_calls.append((float(value), arm))


class Scene:
    def __init__(self): self.count = 0
    def step(self): self.count += 1
    def get_timestep(self): return .004


def make_task(left='empty', right='empty', stop_at=None):
    task = types.SimpleNamespace(robot=Robot(left, right), scene=Scene(), take_action_cnt=0,
        step_lim=1000, eval_success=False, eval_video_path=None, render_freq=0,
        _update_render=lambda: None, get_obs=lambda: None, play_once=lambda: None, plan_success=True)
    task.check_success = lambda: stop_at is not None and task.scene.count >= stop_at
    task.take_action = types.MethodType(actual_method(), task)
    return task


class PolicyTimingTests(unittest.TestCase):
    def exercise(self, mode, left, right, stop_at=None):
        task = make_task(left, right, stop_at)
        method = task.take_action
        if mode == 'fallback15':
            method, _ = runner.retime.retimed_method(method, 15)
        tracer = runner.ActionTiming()
        old = runner.sys.gettrace()
        action = np.zeros(14)
        action[6], action[13] = .5, .75
        runner.sys.settrace(tracer.trace)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                method(action)
        finally:
            runner.sys.settrace(old)
        return task, tracer.record(0, task, mode, action)

    def test_fallback_change_is_counted_for_empty_and_exception_paths(self):
        for reason in ('empty', 'exception'):
            for mode, ticks in [('original', 50), ('fallback15', 15)]:
                with self.subTest(reason=reason, mode=mode):
                    task, record = self.exercise(mode, reason, reason)
                    self.assertEqual(task.scene.count, ticks)
                    self.assertEqual(record['physics_steps'], ticks)
                    self.assertEqual(len(record['fallback_branches']), 2)
                    self.assertEqual(record['controller_locals'], dict(left_n_step=ticks,
                        right_n_step=ticks, topp_left_flag=False, topp_right_flag=False))
                    self.assertEqual(task.robot.arm_calls, [])
                    self.assertEqual(task.robot.gripper_calls[-2:], [(.5, 'left'), (.75, 'right')])
                    self.assertEqual(record['modified_fallback_triggered'], mode == 'fallback15')

    def test_no_fallback_keeps_targets_velocities_ticks_and_predicate_unchanged(self):
        original, a = self.exercise('original', 22, 37)
        modified, b = self.exercise('fallback15', 22, 37)
        self.assertFalse(a['fallback_triggered'])
        self.assertFalse(b['fallback_triggered'])
        self.assertEqual(a['physics_steps'], b['physics_steps'])
        self.assertEqual(a['controller_locals'], b['controller_locals'])
        self.assertEqual(original.robot.arm_calls, modified.robot.arm_calls)
        self.assertEqual(original.robot.gripper_calls, modified.robot.gripper_calls)

    def test_one_fallback_does_not_imply_total_ticks_change(self):
        _, a = self.exercise('original', 'empty', 80)
        _, b = self.exercise('fallback15', 'empty', 80)
        self.assertEqual(a['physics_steps'], 80)
        self.assertEqual(b['physics_steps'], 80)
        self.assertEqual(a['controller_locals']['left_n_step'], 50)
        self.assertEqual(b['controller_locals']['left_n_step'], 15)
        self.assertEqual(b['fallback_branches'][0]['arm'], 'left')
        self.assertEqual(len(b['fallback_branches']), 1)

    def test_original_success_still_stops_at_actual_tick(self):
        for mode in runner.MODES:
            task, row = self.exercise(mode, 'empty', 'empty', 7)
            self.assertEqual(row['physics_steps'], 7)
            self.assertTrue(task.eval_success)

    def test_import_wrapper_keeps_bound_method_and_writes_auditable_rows(self):
        for mode, ticks in [('original', 50), ('fallback15', 15)]:
            with tempfile.TemporaryDirectory() as tmp:
                fake = types.SimpleNamespace(class_decorator=lambda *a, **kw: make_task())
                original_import = runner.importlib.import_module
                def load(name, package=None):
                    return fake if name == 'eval_policy_xpolicylab' else original_import(name, package)
                def fake_simulator():
                    module = runner.importlib.import_module('eval_policy_xpolicylab')
                    task = module.class_decorator('hanging_mug')
                    task.take_action(np.array([0.]*6+[.5]+[0.]*6+[.75]), action_type='qpos')
                    return task.scene.count
                with patch.dict(os.environ, GEN1_POLICY_TIMING_MODE=mode, ROBOTWIN_CONTROLLED_OUTPUT=tmp), \
                     patch.object(runner.importlib, 'import_module', load), \
                     patch.object(runner.horizon.evaluation, 'simulator', fake_simulator), \
                     patch.object(runner.horizon, 'install_literal_binding', lambda: None), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.simulator(), ticks)
                rows = [json.loads(x) for x in (Path(tmp)/'timing.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['physics_steps'], ticks)
                self.assertEqual(runner.timing_summary(Path(tmp)/'timing.jsonl')['fallback_actions'], 1)


if __name__ == '__main__':
    unittest.main()
