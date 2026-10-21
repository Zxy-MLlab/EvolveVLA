"""CPU-only correctness checks for the label replay diagnostic."""
import unittest
from pathlib import Path
import sys
import tempfile
import json
import ast
import contextlib
import io
import types
from typing import Literal
import numpy as np
import gen1_expert_label_replay_20261019 as replay
from gen1_expert_dense_replay_20261019 import SceneStepCounter
from gen1_expert_label_replay_v2_20261019 import joint_tracking_summary
from gen1_expert_gripper_timing_replay_20261019 import eligible_transition, retimed_method
from gen1_expert_retime_continue_20261019 import PredicateRecorder


class ReplayTests(unittest.TestCase):
    def test_continuation_records_true_predicate_but_preserves_expert_gate(self):
        calls = []
        def predicate():
            calls.append(True)
            return True
        recorder = PredicateRecorder(predicate)
        self.assertTrue(recorder.check())
        self.assertIsNone(recorder.first_true)
        recorder.in_action = True
        recorder.action_index = 329
        self.assertFalse(recorder.check())
        self.assertEqual(recorder.first_true, dict(source_action_index=329, check_in_action=1))
        recorder.in_action = False
        self.assertTrue(recorder.check())
        self.assertEqual(len(calls), 3)

    def test_continuation_masks_harness_stop_only_after_episode_start(self):
        recorder = PredicateRecorder(lambda: True)
        recorder.detect_action_scope = True
        namespace = {'recorder': recorder}
        code = 'def run_simulator():\n    return recorder.check()\n'
        exec(compile(code, '/project/scripts/robotwin_eval_harness.py', 'exec'), namespace)
        self.assertTrue(namespace['run_simulator']())  # Expert prefilter.
        recorder.continuation_enabled = True
        self.assertFalse(namespace['run_simulator']())  # Same true predicate; continue.
        self.assertTrue(recorder.last_endpoint_value)
        self.assertTrue(recorder.check())  # Ordinary callers remain unchanged.

    def test_continuation_detects_original_controller_stack(self):
        recorder = PredicateRecorder(lambda: True)
        recorder.detect_action_scope = True
        recorder.action_index = 329
        namespace = {'recorder': recorder}
        code = 'def take_action():\n    return recorder.check()\n'
        exec(compile(code, '/project/envs/_base_task.py', 'exec'), namespace)
        self.assertFalse(namespace['take_action']())
        self.assertEqual(recorder.first_true, dict(source_action_index=329, check_in_action=1))

    def test_roundtrip_preserves_grippers_and_50_frame_anchor(self):
        rng = np.random.default_rng(31)
        state = rng.normal(size=(73, 14)).astype(np.float32)
        action = rng.normal(size=(73, 14)).astype(np.float32)
        action[:, [6, 13]] = rng.uniform(0, 1, size=(73, 2))
        norm = {'actions': {'q01': [-2.]*14, 'q99': [3.]*14}}
        restored = replay.roundtrip_actions(state, action, norm)
        np.testing.assert_allclose(restored, action, atol=1e-6, rtol=0)

    def test_selection_honors_manifest_order_and_task_quota(self):
        rows = [dict(id=f'{task}_{i}', task=task, smoke=i == 1)
                for task in ('hanging_mug', 'place_mouse_pad') for i in range(4)]
        selected = replay.select_cases({'cases': rows}, 2)
        self.assertEqual([x['id'] for x in selected],
                         ['hanging_mug_0', 'hanging_mug_1', 'place_mouse_pad_0', 'place_mouse_pad_1'])
        self.assertEqual(len(replay.select_cases({'cases': rows}, 2, True)), 2)

    def test_source_match_requires_all_cameras_and_state(self):
        metrics = {c: dict(mae=2.) for c in replay.CAMERAS}
        self.assertTrue(replay.initial_matches(metrics, 0.))
        self.assertFalse(replay.initial_matches(metrics, .01))
        metrics['cam_high']['mae'] = 9.
        self.assertFalse(replay.initial_matches(metrics, 0.))

    def test_physics_counter_counts_executed_lines_without_extra_steps(self):
        code = ('class Task:\n'
                '    def __init__(self, scene):\n'
                '        self.scene = scene\n'
                '    def take(self):\n'
                '        for _ in range(7):\n'
                '            self.scene.step()\n')
        class Scene:
            calls = 0
            def step(self):
                self.calls += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'task.py'
            path.write_text(code)
            namespace = {}
            exec(compile(code, str(path), 'exec'), namespace)
            counter = SceneStepCounter(path)
            scene = Scene()
            task = namespace['Task'](scene)
            old = sys.gettrace()
            sys.settrace(counter.trace)
            try:
                task.take()
            finally:
                sys.settrace(old)
            self.assertEqual(counter.count, 7)
            self.assertEqual(scene.calls, 7)

    def test_joint_tracking_separates_gripper_reporting_error(self):
        drive = np.zeros(14)
        drive[2], drive[6], drive[13] = .01, .3, .4
        row = dict(source_target=[0.]*14, measured_joint_max_abs_error=.02,
                   physical={'left': {'drive_targets': drive[:7].tolist()},
                             'right': {'drive_targets': drive[7:].tolist()}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.jsonl'
            path.write_text(json.dumps(row)+'\n')
            result = joint_tracking_summary(path)
        self.assertEqual(result['max_joint_drive_target_error'], .01)
        self.assertEqual(result['max_gripper_reported_value_error'], .4)
        self.assertEqual(result['max_measured_joint_error'], .02)

    def test_retime_only_changes_requested_gripper_only_suffix(self):
        source = np.zeros(14)
        target = source.copy()
        target[13] = .075
        self.assertFalse(eligible_transition(318, 319, source, target, source))
        self.assertTrue(eligible_transition(319, 319, source, target, source))
        moving = target.copy()
        moving[2] = .01
        self.assertFalse(eligible_transition(319, 319, source, moving, source))
        off_target = source.copy()
        off_target[7] = .01
        self.assertFalse(eligible_transition(319, 319, off_target, target, source))
        self.assertFalse(eligible_transition(319, 319, source, source, source))

    def test_actual_controller_patch_changes_ticks_without_changing_false_topp_flags(self):
        source = replay.ROOT/'robotwin_repo_ba0fd21/envs/_base_task.py'
        tree = ast.parse(source.read_text(), filename=str(source))
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'take_action')
        namespace = {'np': np, 'Literal': Literal}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        class Planner:
            def __init__(self, fail):
                self.fail = fail
            def TOPP(self, *args, **kwargs):
                if self.fail:
                    raise ValueError('Synthetic zero-motion planner rejection')
                empty = np.empty((0, 6))
                return [], empty, empty, empty, 0.
        class Robot:
            def __init__(self, fail):
                self.left_mplib_planner = self.right_mplib_planner = Planner(fail)
                self.arm_calls, self.gripper_calls = [], []
            def get_left_arm_jointState(self): return [0.]*7
            def get_right_arm_jointState(self): return [0.]*7
            def get_left_gripper_val(self): return 0.
            def get_right_gripper_val(self): return 0.
            def set_arm_joints(self, *args): self.arm_calls.append(args)
            def set_gripper(self, value, side): self.gripper_calls.append((value, side))
        class Scene:
            count = 0
            def step(self): self.count += 1
        for fail in (False, True):
            for ticks in (50, 15):
                task = types.SimpleNamespace(robot=Robot(fail), scene=Scene(), take_action_cnt=0,
                    step_lim=1000, eval_success=False, eval_video_path=None, render_freq=0,
                    _update_render=lambda: None, check_success=lambda: False)
                method = types.MethodType(namespace['take_action'], task)
                if ticks == 15:
                    method, report = retimed_method(method, 15)
                    self.assertEqual(len(report['replacements']), 4)
                action = np.zeros(14)
                action[6], action[13] = .5, .75
                with contextlib.redirect_stdout(io.StringIO()):
                    method(action)
                self.assertEqual(task.scene.count, ticks)
                self.assertEqual(task.robot.arm_calls, [])
                self.assertEqual(len(task.robot.gripper_calls), 2*ticks)
                self.assertEqual(task.robot.gripper_calls[-2:], [(.5, 'left'), (.75, 'right')])


if __name__ == '__main__':
    unittest.main()
