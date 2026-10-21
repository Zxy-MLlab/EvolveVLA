"""Independent CPU review of the frozen v2 legacy argument shim."""
import ast
import contextlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import gen1_task_routing_v2_20261020 as v2
import robotwin_public_openpi_harness as harness

routing=v2.routing


class V2RoutingReview(unittest.TestCase):
    def test_all_direct_simulator_and_parent_args_are_available_after_compat(self):
        with patch.object(sys,'argv',['runner']):args=harness.parse_args()
        args.episode_index=None
        for filename,function in [('robotwin_eval_harness.py','run_simulator'),('robotwin_public_openpi_harness.py','run_parent')]:
            tree=ast.parse((v2.SCRIPT.parent/filename).read_text())
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
            names={n.attr for n in ast.walk(fn) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='args'}
            self.assertEqual(names-set(vars(args)),set())
        self.assertIsNone(args.episode_index)

    def runtime(self,role='baseline',mode='original'):
        receipt=dict(task_id='place_mouse_pad',selected_role=role,controller_mode=mode)
        called=[]
        @contextlib.contextmanager
        def hook(value):
            called.append(value);yield
        with patch.object(sys,'argv',['runner','--role','sim','--task','place_mouse_pad','--robotwin-root',str(routing.ROBOTWIN)]), \
             patch.object(v2,'load_runtime_receipt',return_value=receipt), \
             patch.object(routing,'simulator_controller_hook',side_effect=hook) as controller, \
             patch.object(harness,'run_simulator',side_effect=lambda args:0 if args.episode_index is None else 1) as run:
            self.assertEqual(v2.internal_runtime(),0)
            run.assert_called_once()
            return controller.call_count,called

    def test_baseline_original_never_installs_controller_hook(self):
        self.assertEqual(self.runtime('baseline','original'),(0,[]))

    def test_candidate_original_never_installs_controller_hook(self):
        self.assertEqual(self.runtime('candidate','original'),(0,[]))

    def test_only_nonoriginal_selected_candidate_uses_hook(self):
        count,called=self.runtime('candidate','fixed15')
        self.assertEqual(count,1);self.assertEqual(called[0]['controller_mode'],'fixed15')

    def test_parent_command_routes_once_through_compat_with_original_settings(self):
        plan=dict(task_id='beat_block_hammer',selected_role='baseline',controller_mode='original',checkpoint=str(routing.BASELINE))
        with patch.object(routing,'SCRIPT',v2.SCRIPT):
            cmd=v2.command_for(plan,Path('/tmp/routing_review.json'),123)
        self.assertEqual(cmd[4:6],[str(v2.SCRIPT),'_runtime'])
        self.assertEqual(cmd.count('_runtime'),1)
        self.assertEqual(cmd[cmd.index('--checkpoint')+1],str(routing.BASELINE))
        self.assertEqual(cmd[cmd.index('--noise-source')+1],'jax')
        self.assertEqual(cmd[cmd.index('--policy-seed')+1],'0')
        self.assertEqual(cmd[cmd.index('--task-config')+1],'demo_clean')
        self.assertNotIn('--episode-index',cmd)

    def test_nonoriginal_command_has_no_duplicate_runtime_marker(self):
        plan=dict(task_id='place_mouse_pad',selected_role='candidate',controller_mode='fixed15',checkpoint='/candidate')
        with patch.object(routing,'SCRIPT',v2.SCRIPT):
            cmd=v2.command_for(plan,Path('/tmp/routing_review.json'),123)
        self.assertEqual(cmd[4:6],[str(v2.SCRIPT),'_runtime'])
        self.assertEqual(cmd.count('_runtime'),1)


if __name__=='__main__':unittest.main()
