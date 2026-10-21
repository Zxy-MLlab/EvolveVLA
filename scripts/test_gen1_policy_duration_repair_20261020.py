"""Exercise controller/IPC hook composition with the actual take_action AST."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import gen1_policy_duration_repair_20261020 as runner
from test_gen1_policy_timing_ab_20261020 import make_task


class DurationRunnerTests(unittest.TestCase):
    def exercise(self, actor, mode, stop_at=None, reject=False):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(35, 80, stop_at)
            task.robot.get_normal_real_gripper_val = lambda: (0., 0.)
            calls = {'play': 0, 'check': 0}
            def play():
                calls['play'] += 1
                task.plan_success = not reject
            old_check = task.check_success
            def check():
                calls['check'] += 1
                return old_check() if task.take_action_cnt else not reject
            task.play_once, task.check_success = play, check
            fake = types.SimpleNamespace(class_decorator=lambda *a, **kw: task)
            original_import = runner.importlib.import_module
            def load(name, package=None):
                return fake if name == 'eval_policy_xpolicylab' else original_import(name, package)
            def execute():
                module = runner.importlib.import_module('eval_policy_xpolicylab')
                bound = module.class_decorator('hanging_mug')
                bound.play_once()
                if not bound.plan_success:
                    return None
                self.assertTrue(bound.check_success())
                import robotwin_eval_harness as harness
                if actor == 'expert': harness.send_message(None, {'type': 'episode_start'})
                bound.take_action(np.array([0.]*6+[.5]+[0.]*6+[.75]), action_type='qpos')
                if actor == 'expert': harness.send_message(None, {'type': 'step'})
                return bound.scene.count
            import robotwin_eval_harness as harness
            captured = []
            with patch.dict(os.environ, GEN1_DURATION_MODE=mode, GEN1_DURATION_ACTOR=actor,
                            ROBOTWIN_CONTROLLED_OUTPUT=tmp), \
                 patch.object(runner.importlib, 'import_module', load), \
                 patch.object(runner.horizon.evaluation, 'simulator', execute), \
                 patch.object(harness, 'main', execute), \
                 patch.object(harness, 'send_message', lambda channel,payload:captured.append(payload)), \
                 patch.object(runner.oracle, 'physical', lambda task:{'fake_state':task.scene.count}), \
                 patch.object(runner.horizon, 'install_literal_binding', lambda:None), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = runner.simulator()
            rows = [json.loads(s) for s in (Path(tmp)/'timing.jsonl').read_text().splitlines()]
            gate = json.loads((Path(tmp)/'expert_gate.json').read_text())
            return result, rows, gate, captured, calls

    def test_normal_paths_are_retimed_and_actual_ticks_audited(self):
        for mode, ticks in [('original',80),('fixed15',15),('opening15',15)]:
            result, rows, gate, _, calls = self.exercise('policy',mode)
            self.assertEqual(result,ticks)
            self.assertEqual(rows[0]['physics_steps'],ticks)
            self.assertEqual(rows[0]['duration_control']['active'],mode!='original')
            self.assertEqual(gate['status'],'passed')
            self.assertEqual(calls,{'play':1,'check':ticks+1})

    def test_expert_ipc_contains_exact_counter_and_physical_annotations(self):
        result,rows,gate,payloads,calls = self.exercise('expert','fixed15')
        self.assertEqual(result,15)
        self.assertEqual(len(payloads),2)
        self.assertEqual(payloads[0]['oracle_timing']['total_physics_steps'],0)
        self.assertEqual(payloads[1]['oracle_timing']['physics_steps'],15)
        self.assertEqual(payloads[1]['oracle_timing']['total_physics_steps'],15)
        self.assertEqual(payloads[1]['oracle_physical'],{'fake_state':15})
        self.assertEqual(payloads[1]['oracle_gate']['status'],'passed')

    def test_success_still_terminates_mid_action(self):
        for actor in ['expert','policy']:
            result,rows,_,_,calls=self.exercise(actor,'fixed15',stop_at=7)
            self.assertEqual(result,7)
            self.assertTrue(rows[0]['eval_success_latched'])
            self.assertEqual(calls['check'],8)

    def test_expert_rejection_never_executes_policy_or_changes_predicate(self):
        for actor in ['expert','policy']:
            result,rows,gate,_,calls=self.exercise(actor,'fixed15',reject=True)
            self.assertIsNone(result)
            self.assertEqual(rows,[])
            self.assertEqual(gate['status'],'rejected')
            self.assertFalse(gate['bypassed'])
            self.assertEqual(calls,{'play':1,'check':0})


if __name__=='__main__':unittest.main()
