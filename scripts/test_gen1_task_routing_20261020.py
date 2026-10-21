"""CPU-only routing checks: no simulator or model weights are loaded."""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import gen1_task_routing_20261020 as routing


def config():
    return dict(schema_version=1, baseline=dict(checkpoint=str(routing.BASELINE),
        identity_sha256=routing.BASELINE_ID, controller_mode='original',
        identity_reference='/baseline/reference.json', identity_reference_sha256='baseline_sha'),
        target_routes={task: dict(status='pending', candidate=None) for task in routing.TARGETS},
        runtime_sources_sha256={str(p): routing.digest(p) for p in routing.required_sources()})


def candidate(mode='opening15'):
    return dict(checkpoint='/deliberately_missing_candidate', identity_sha256='candidate_identity',
        controller_mode=mode, identity_reference='/candidate/reference.json',
        identity_reference_sha256='candidate_sha', deployment_evidence='/missing/decision.json',
        deployment_evidence_sha256='missing')


class TaskRoutingTests(unittest.TestCase):
    def test_other_and_unknown_tasks_never_select_or_verify_candidate_weights(self):
        cfg = config()
        cfg['target_routes']['place_mouse_pad'] = dict(status='verified', candidate=candidate())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'routing.json'
            p.write_text(json.dumps(cfg))
            for task in ('beat_block_hammer', 'unknown_future_task', 'hanging_mug'):
                with self.subTest(task=task), patch.object(routing, 'verify_checkpoint',
                        return_value=dict(path=str(routing.BASELINE), identity_sha256=routing.BASELINE_ID)) as verify:
                    plan = routing.build_plan(p, task)
                    self.assertEqual(plan['selected_role'], 'baseline')
                    self.assertEqual(plan['controller_mode'], 'original')
                    verify.assert_called_once_with(cfg['baseline'])
                    command = routing.command_for(plan, Path(tmp)/'out.json', 123)
                    self.assertIn(str(routing.HARNESS), command)
                    self.assertNotIn(str(routing.SCRIPT), command)
                    self.assertIn(str(routing.BASELINE), command)
                    self.assertNotIn(candidate()['checkpoint'], command)

    def test_pending_targets_ignore_unvalidated_candidate_and_use_baseline(self):
        cfg = config()
        for task in routing.TARGETS:
            cfg['target_routes'][task]['candidate'] = candidate()
            selected = routing.select_route(cfg, task)
            self.assertEqual(selected['role'], 'baseline')
            self.assertEqual(selected['model'], cfg['baseline'])
            self.assertEqual(selected['reason'], 'target_route_pending')

    def test_non_target_override_and_non_original_fallback_are_rejected(self):
        cfg = config()
        cfg['target_routes']['beat_block_hammer'] = dict(status='verified', candidate=candidate())
        with self.assertRaises(ValueError):
            routing.select_route(cfg, 'beat_block_hammer')
        cfg = config()
        cfg['baseline']['controller_mode'] = 'fixed15'
        with self.assertRaises(ValueError):
            routing.select_route(cfg, 'place_mouse_pad')
        cfg = config()
        cfg['baseline']['checkpoint'] = '/candidate'
        with self.assertRaises(ValueError):
            routing.select_route(cfg, 'unknown_task')

    def test_verified_target_requires_exact_task_model_controller_evidence(self):
        cfg = config()
        spec = candidate()
        cfg['target_routes']['place_mouse_pad'] = dict(status='verified', candidate=spec)
        with tempfile.TemporaryDirectory() as tmp:
            decision_path = Path(tmp)/'decision.json'
            report = Path(tmp)/'report.md'
            report.write_text('Independent held-out validation report placeholder for CPU unit test only.')
            decision = dict(status='passed', task_id='place_mouse_pad',
                checkpoint_identity_sha256=spec['identity_sha256'], controller_mode='opening15',
                controller_source_sha256=routing.digest(routing.CONTROLLER),
                validation_report=str(report), validation_report_sha256=routing.digest(report))
            decision_path.write_text(json.dumps(decision))
            spec.update(deployment_evidence=str(decision_path), deployment_evidence_sha256=routing.digest(decision_path))
            selected = routing.select_route(cfg, 'place_mouse_pad')
            self.assertEqual(routing.verify_deployment_evidence(selected)['decision'], decision)
            for key, bad in [('task_id', 'hanging_mug'), ('controller_mode', 'fixed15'),
                             ('checkpoint_identity_sha256', 'wrong')]:
                altered = copy.deepcopy(selected)
                if key == 'checkpoint_identity_sha256':
                    altered['model']['identity_sha256'] = bad
                else:
                    altered[key] = bad
                with self.assertRaises(ValueError):
                    routing.verify_deployment_evidence(altered)
            report.write_text('Changed report')
            with self.assertRaises(ValueError):
                routing.verify_deployment_evidence(selected)

    def test_frozen_checkpoint_inventory_detects_mutated_and_added_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'weights'
            (root/'params').mkdir(parents=True)
            (root/'assets').mkdir()
            (root/'params/_METADATA').write_text('metadata')
            (root/'params/chunk').write_bytes(b'weight bytes')
            (root/'assets/stats.json').write_text('{}')
            (root/'SHA256SUMS').write_text('frozen manifest')
            files = {str(p.relative_to(root)): routing.digest(p) for p in root.rglob('*') if p.is_file()}
            identity = dict(path=str(root), files_sha256=files,
                identity_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
            reference = Path(tmp)/'reference.json'
            reference.write_text(json.dumps(dict(checkpoint=identity)))
            spec = dict(checkpoint=str(root), identity_sha256=identity['identity_sha256'],
                identity_reference=str(reference), identity_reference_sha256=routing.digest(reference))
            self.assertEqual(routing.verify_checkpoint(spec), identity)
            (root/'params/chunk').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                routing.verify_checkpoint(spec)
            (root/'params/chunk').write_bytes(b'weight bytes')
            (root/'assets/new.json').write_text('{}')
            with self.assertRaises(ValueError):
                routing.verify_checkpoint(spec)

    def test_launcher_passes_only_selected_model_and_clears_stale_interventions(self):
        plan = dict(task_id='beat_block_hammer', selected_role='baseline',
            controller_mode='original', checkpoint=str(routing.BASELINE))
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp)/'receipt.json'
            receipt.write_text('{}')
            with patch.dict(routing.os.environ, ROBOTWIN_SKIP_EXPERT='1', GEN1_DURATION_MODE='fixed15'), \
                 patch.object(routing.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as run:
                self.assertEqual(routing.launch(plan, receipt, Path(tmp)/'out.json', 7), 0)
            args, kwargs = run.call_args
            self.assertEqual(args[0].count('--checkpoint'), 1)
            self.assertEqual(args[0][args[0].index('--checkpoint')+1], str(routing.BASELINE))
            self.assertNotIn('ROBOTWIN_SKIP_EXPERT', kwargs['env'])
            self.assertNotIn('GEN1_DURATION_MODE', kwargs['env'])

    def test_target_simulator_hook_keeps_task_identity_and_resets_between_setups(self):
        from test_gen1_duration_controller_20261020 import make_task, action
        task = make_task(30, 45)
        task.setup_demo = lambda *a, **kw: None
        module = types.SimpleNamespace(__file__=str(routing.ROBOTWIN/'scripts/eval_policy_xpolicylab.py'),
            class_decorator=lambda task_id: task)
        initial_construct = module.class_decorator
        previous_import = routing.importlib.import_module
        def fake_import(name, package=None):
            return module if name == 'eval_policy_xpolicylab' else previous_import(name, package)
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp)/'controller.jsonl'
            receipt = dict(task_id='place_mouse_pad', controller_mode='fixed15', controller_trace=str(trace))
            with patch.object(routing.importlib, 'import_module', fake_import), \
                 contextlib.redirect_stdout(io.StringIO()):
                with routing.simulator_controller_hook(receipt):
                    loaded = routing.importlib.import_module('eval_policy_xpolicylab')
                    with self.assertRaises(ValueError):
                        loaded.class_decorator('beat_block_hammer')
                    selected = loaded.class_decorator('place_mouse_pad')
                    for _ in range(2):
                        selected.setup_demo()
                        selected.take_action(action(.5, .75))
                self.assertIs(module.class_decorator, initial_construct)
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(task.scene.count, 30)
            self.assertEqual([r['episode_setup'] for r in rows], [1, 2])
            self.assertEqual([r['action_index'] for r in rows], [0, 0])
            self.assertTrue(all(r['controller']['active'] for r in rows))


if __name__ == '__main__':
    unittest.main()
