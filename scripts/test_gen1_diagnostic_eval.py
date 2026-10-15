"""CPU-only tests: fake policies, IPC, simulator and process supervisor."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import gen1_diagnostic_eval as d


def primary():
    return {'cases': [dict(id=f'{task}_{seed}', task=task, seed=seed, panel=panel,
                           prompt=f'fixed {seed}', prompt_source='/fake/log')
                      for task in d.TASKS for seed, panel in ((9, 'primary'), (2, 'primary'),
                                                             (1, 'primary'), (0, 'supplemental'))]}


# This small function supplies the same source-line hook and terminal IPC as the real simulator.
def fake_simulator(args):
    raw_after_action = {'joint_action': {'vector': list(range(14))}}
    _dump_cam_high_png(raw_after_action, 0, 9, 1)
    FAKE_H.send_message(None, dict(type='step', observation=None, terminated=True))
    return 0


def _dump_cam_high_png(*args):
    pass


FAKE_H = types.ModuleType('robotwin_eval_harness')


class DiagnosticTests(unittest.TestCase):
    def test_selection_order_and_outcome_independence(self):
        p = primary()
        cases = d.selected(p)
        self.assertEqual(len(cases), 8)
        self.assertEqual([x['seed'] for x in cases], [9, 9, 2, 2]*2)
        self.assertEqual([x['repetition'] for x in cases], [1, 2]*4)
        for row in p['cases']:
            row['success'] = True
            row['status'] = 'worker_error'
        self.assertEqual(cases, d.selected(p))
        p['cases'][1]['seed'] = 9
        with self.assertRaises(ValueError):
            d.selected(p)

    def test_canonical_finite_hash(self):
        self.assertEqual(d.array_hash(np.arange(14).astype('>f4')), d.array_hash(np.arange(14)))
        with self.assertRaises(ValueError):
            d.array_hash([np.nan])

    def test_fake_episode_full_chunk_terminal_and_repeat(self):
        fake = types.ModuleType('robotwin_public_openpi_harness')
        images = {key: np.zeros((2, 2, 3), dtype=np.uint8) for key in
                  ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}
        fake._observation_for_openpi = lambda obs: (images, np.zeros(14))
        case = d.selected(primary())[0]
        first = dict(type='episode_start', seed=case['seed'], instruction=case['prompt'], observation={})
        results, traces = [], []
        policies = []
        def factory():
            policy = types.SimpleNamespace(checkpoint_format='openpi_orbax', noise_source='jax',
                                           infer_chunk=lambda **kw: np.zeros((50, 14)))
            policies.append(policy)
            return policy
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, robotwin_public_openpi_harness=fake):
            for repeat in (1, 2):
                path = Path(tmp)/str(repeat)
                path.mkdir()
                sent = []
                response = dict(type='step', observation=None, terminated=True, info={'is_success': True},
                                diagnostic_state=d.array_hash(np.ones(14)))
                results.append(d.diagnostic_episode(first, lambda: response, sent.append, factory, path, case))
                traces.append([json.loads(x) for x in (path/'trace.jsonl').read_text().splitlines()])
                self.assertEqual(len(sent), 1)
                self.assertTrue((path/'initial/hashes.json').is_file())
            self.assertEqual(len(policies), 2)
            self.assertEqual(traces[0][1]['actions']['shape'], [50, 14])
            self.assertEqual(traces[0][-1]['state'], d.array_hash(np.ones(14)))
            self.assertIsNone(d.comparison(*results, *traces))
            changed = copy.deepcopy(traces[1])
            changed[1]['actions'] = d.array_hash(np.ones((50, 14)))
            self.assertEqual(d.comparison(*results, traces[0], changed)['event'], 1)
            altered = dict(results[1], hashes={})
            self.assertEqual(d.comparison(results[0], altered, *traces)['stage'], 'initial_observation')

    def test_expert_rejection_never_loads_policy(self):
        fake = types.ModuleType('robotwin_public_openpi_harness')
        fake._observation_for_openpi = Mock()
        factory = Mock(side_effect=AssertionError('policy must not load'))
        case = d.selected(primary())[0]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, robotwin_public_openpi_harness=fake):
            result = d.diagnostic_episode(dict(type='case_status', seed=9, status='expert_rejected'),
                                          None, None, factory, Path(tmp), case)
        self.assertEqual(result['status'], 'expert_rejected')
        factory.assert_not_called()
        self.assertEqual(d.comparison(result, result, [], [])['stage'], 'not_comparable')

    def test_terminal_state_missing_and_short_chunk_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = d.Trace(Path(tmp))
            try:
                with self.assertRaises(ValueError):
                    trace.response(dict(type='step', terminated=False))
                trace.response(dict(type='step', terminated=True))
                policy = types.SimpleNamespace(checkpoint_format='openpi_orbax', noise_source='jax',
                                               infer_chunk=lambda **kw: np.zeros((49, 14)))
                with self.assertRaises(ValueError):
                    trace.policy(policy).infer_chunk()
                policy.noise_source = 'numpy'
                with self.assertRaises(ValueError):
                    trace.policy(policy)
            finally:
                trace.stream.close()

    def test_external_simulator_hook_is_read_only_rng_neutral(self):
        import random
        sent = []
        original = lambda stream, payload: sent.append(payload)
        obs = {'agent_pos': np.arange(14, dtype=np.float32)}
        payload = dict(type='step', observation=obs, terminated=False)
        before = obs['agent_pos'].copy()
        rng, np_rng = random.getstate(), np.random.get_state()
        FAKE_H.main = lambda: (FAKE_H.send_message(None, payload),
                               FAKE_H.send_message(None, dict(type='step', observation=None, terminated=True)), 0)[-1]
        FAKE_H.send_message = original
        with patch.dict(sys.modules, robotwin_eval_harness=FAKE_H), patch.object(d.inspect, 'getsourcefile', return_value=str(d.ROOT/'scripts/robotwin_eval_harness.py')):
            self.assertEqual(d.simulator(), 0)
        self.assertEqual(sent[0]['diagnostic_state'], d.array_hash(np.arange(14)))
        self.assertIsNone(sent[1]['diagnostic_state'])
        self.assertNotIn('diagnostic_state', payload)
        np.testing.assert_array_equal(before, obs['agent_pos'])
        self.assertEqual(rng, random.getstate())
        after = np.random.get_state()
        self.assertEqual(np_rng[0], after[0])
        np.testing.assert_array_equal(np_rng[1], after[1])
        self.assertEqual(np_rng[2:], after[2:])
        self.assertIs(FAKE_H.send_message, original)
        with patch.dict(sys.modules, robotwin_eval_harness=FAKE_H), patch.object(d.inspect, 'getsourcefile', return_value='/wrong/source.py'):
            with self.assertRaises(RuntimeError):
                d.simulator()

    def test_manifest_prepare_immutable_and_identity_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root/'params'
            checkpoint.write_text('checkpoint bytes')
            source = root/'source.py'
            source.write_text('source')
            p = primary()
            p.update(identities={'reference': {str(checkpoint): d.c.sha(checkpoint)}},
                     sources={}, code={str(source): d.c.sha(source)})
            primary_path = root/'primary.json'
            d.c.write(primary_path, p)
            with patch.object(d.c, 'validate'), patch.object(d, 'disk_guard'):
                m = d.prepare(root/'manifest', primary_path)
                with self.assertRaises(FileExistsError):
                    d.prepare(root/'manifest', primary_path)
            manifest = root/'manifest/manifest.json'
            self.assertEqual(manifest.stat().st_mode & 0o222, 0)
            d.validate(m, manifest)
            wrong = dict(m, model='/public29999')
            with self.assertRaises(ValueError):
                d.validate(wrong, manifest)
            missing = dict(m, identities={})
            with self.assertRaises(ValueError):
                d.validate(missing, manifest)
            source.write_text('modified')
            with self.assertRaises(ValueError):
                d.validate(m, manifest)
            source.write_text('source')
            checkpoint.write_text('wrong checkpoint')
            with self.assertRaises(ValueError):
                d.validate(m, manifest)

    def test_disk_budget_and_free_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'x').write_bytes(b'12345')
            with patch.object(d.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=100*d.GIB)):
                with patch.dict(d.LIMITS, output_budget_bytes=5):
                    with self.assertRaises(RuntimeError):
                        d.disk_guard(tmp)
                    d.disk_guard(tmp, check_size=False)
            with patch.object(d.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=69*d.GIB)):
                with self.assertRaises(RuntimeError):
                    d.disk_guard(tmp)

    def test_supervisor_timeout_kills_process_group(self):
        proc = Mock(pid=7654)
        proc.wait.side_effect = [subprocess.TimeoutExpired('fake', 5), 0]
        with patch.object(d.subprocess, 'Popen', return_value=proc), patch.object(d, 'disk_guard'), \
             patch.object(d.time, 'monotonic', side_effect=[0, 1801]), patch.object(d.os, 'killpg') as kill:
            result = d.supervise(['fake'], {}, None, Path('/fake'))
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(kill.call_count, 2)
        self.assertTrue(kill.call_args_list[0].args[0] == 7654)

    def test_abnormal_exit_does_not_become_success_or_retry(self):
        m = {'cases': d.selected(primary())}
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, 'disk_guard'), \
             patch.object(d, 'environment', return_value={}), \
             patch.object(d, 'supervise', return_value=dict(status='process_exit', returncode=7)) as sup:
            output = Path(tmp)/'run'
            self.assertEqual(d.run(m, Path(tmp)/'manifest.json', output), 1)
            self.assertEqual(sup.call_count, 1)
            report = json.loads((output/'summary.json').read_text())
            self.assertEqual(report['executed_cases'], 1)
            self.assertTrue(all(p['status'] == 'incomplete' for p in report['pairs']))
            with self.assertRaises(FileExistsError):
                d.run(m, Path(tmp)/'manifest.json', output)

    def test_expert_rejections_execute_all_eight_without_replacement(self):
        m = {'cases': d.selected(primary())}
        def reject(command, env, log, output):
            index = int(command[-1])
            d.c.write(output/m['cases'][index]['id']/'result.json', dict(status='expert_rejected', gate='expert_rejected'))
            return dict(status='process_exit', returncode=0)
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, 'disk_guard'), \
             patch.object(d, 'environment', return_value={}), patch.object(d, 'supervise', side_effect=reject) as sup:
            output = Path(tmp)/'run'
            self.assertEqual(d.run(m, Path(tmp)/'manifest.json', output), 0)
            self.assertEqual(sup.call_count, 8)
            report = json.loads((output/'summary.json').read_text())
            self.assertEqual(report['executed_cases'], 8)
            self.assertTrue(all(p['earliest_difference']['stage'] == 'not_comparable' for p in report['pairs']))


if __name__ == '__main__':
    unittest.main()
