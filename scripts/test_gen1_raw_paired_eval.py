#!/usr/bin/env python3
"""CPU mocks for scoped raw integration; no checkpoint/model/simulator reads."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('paired', Path(__file__).with_name('gen1_raw_paired_eval.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class GuardTests(unittest.TestCase):
    def test_df_strict_parser(self):
        with patch.object(p.subprocess, 'run', return_value=NS(stdout='Avail\n75161927680\n')) as run:
            self.assertEqual(p.df_available('/tmp'), 70*p.GIB)
            self.assertIn('-B1', run.call_args.args[0])
        for text in ('Avail\n70G\n', 'Avail\n-1\n', '', 'Avail\n1\n2\n'):
            with patch.object(p.subprocess, 'run', return_value=NS(stdout=text)):
                with self.assertRaises(RuntimeError):
                    p.df_available('/tmp')
        with patch.object(p.subprocess, 'run', side_effect=subprocess.TimeoutExpired('df', 10)):
            with self.assertRaises(subprocess.TimeoutExpired):
                p.df_available('/tmp')

    def test_budget_pending_scope_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(p, 'OUTPUTS', Path(tmp)):
            out = Path(tmp)/'run'
            out.mkdir()
            (out/'small').write_bytes(b'1234')
            with patch.object(p, 'df_available', return_value=p.MIN_FREE+10):
                p.guard(out, 10)
                with self.assertRaises(RuntimeError):
                    p.guard(out, 11)
                with patch.object(p, 'LIMIT', 5):
                    with self.assertRaises(RuntimeError):
                        p.guard(out, 2)
                with self.assertRaises(ValueError):
                    p.guard(out/'sub')
                with self.assertRaises(ValueError):
                    p.guard(out, -1)
                (out/'link').symlink_to(out/'small')
                with self.assertRaises(ValueError):
                    p.guard(out)
                (out/'link').unlink()
                with patch.object(p.os, 'scandir', side_effect=PermissionError()):
                    with self.assertRaises(PermissionError):
                        p.guard(out)


class ConfigIdentityTests(unittest.TestCase):
    def test_real_manifest_paths_cover_executed_configuration(self):
        # No manifest_code_paths mock, no simulator import or panel selection.
        paths = set(p.manifest_code_paths())
        repo = p.ROOT/'robotwin_repo_ba0fd21'
        required = {repo/'scripts/eval_policy_xpolicylab.py'}
        required.update(repo/'env_cfg/task_config'/name for name in
                        ('demo_clean.yml', '_embodiment_config.yml',
                         '_camera_config.yml', '_eval_step_limit.yml'))
        required.update(repo/'assets/embodiments/aloha-agilex'/name for name in
                        ('config.yml', 'curobo_left.yml', 'curobo_right.yml'))
        required.update(p.ROOT/'robotwin_repo/assets/embodiments/aloha-agilex'/name
                        for name in ('collision_aloha_left.yml', 'collision_aloha_right.yml'))
        self.assertTrue(required <= paths, required-paths)
        self.assertTrue(all(path.is_file() and path.read_bytes().strip() for path in required))

    def test_fixture_configuration_changes_rejected_by_validate(self):
        import shutil
        import sys
        repo = p.ROOT/'robotwin_repo_ba0fd21'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); target = root/'robotwin_repo_ba0fd21'
            # Only temporary copies are changed; never mutate actual configuration.
            for path in p.manifest_code_paths():
                if path.is_relative_to(p.ROOT) and not path.is_relative_to(p.OPENPI):
                    dest = root/path.relative_to(p.ROOT)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, dest)
            for pool in ('039_mug', '040_rack', '047_mouse'):
                directory = target/'description/objects_description'/pool
                directory.mkdir(parents=True)
                (directory/'fixture.json').write_text('{}')
            source = target/'description/utils/generate_episode_instructions.py'
            source.parent.mkdir(parents=True)
            source.write_text('# fixture\n')
            c = NS(MODELS={'reference': '/unused'}, CONFIG={})
            with patch.object(p, 'ROOT', root), patch.object(p, 'seed_inventories', return_value=(set(), {})), \
                    patch.object(p, 'selected_cases', return_value=[]), \
                    patch.dict(sys.modules, {'controlled_paired_eval': c}):
                baseline = p.expected_manifest(include_weights=False)
                frozen = dict(baseline, identities={})
                p.validate(frozen, check_files=False)
                config = target/'env_cfg/task_config/demo_clean.yml'
                original = config.read_text()
                config.write_text(original+'\n# changed fixture configuration\n')
                changed = p.expected_manifest(include_weights=False)
                self.assertNotEqual(baseline['code'][str(config)], changed['code'][str(config)])
                with self.assertRaisesRegex(ValueError, 'identities changed'):
                    p.validate(frozen, check_files=False)
                config.write_text('')
                with self.assertRaisesRegex(ValueError, 'empty simulator configuration'):
                    p.manifest_code_paths()
                config.unlink()
                with self.assertRaisesRegex(ValueError, 'Missing or empty'):
                    p.manifest_code_paths()


class FactoryTests(unittest.TestCase):
    def setup_factory(self, fail=False):
        group = lambda: NS(inputs=[], outputs=[])
        transforms = NS(Group=group,
                        InjectDefaultPrompt=lambda x: ('prompt', x),
                        Normalize=lambda x, **kw: ('normalize', x, kw),
                        Unnormalize=lambda x, **kw: ('unnormalize', x, kw))
        dc = NS(asset_id='asset', use_quantile_norm=True,
                data_transforms=NS(inputs=['data-in'], outputs=['data-out']),
                model_transforms=NS(inputs=['model-in'], outputs=['model-out']),
                repack_transforms=NS(inputs=['WRONG'], outputs=['WRONG']))
        cfg = NS(model=NS(action_horizon=50, action_dim=32), assets_dirs='assets',
                 data=NS(create=lambda *args: dc), policy_metadata={'key': 'value'})
        pc = NS(transforms=transforms, _checkpoints=NS(load_norm_stats=lambda *args: 'stats'),
                _policy=NS(Policy=lambda model, **kw: (model, kw)), create_trained_policy=object())
        original_guard = object()
        raw = NS(d=NS(guard=original_guard))
        def load(*args):
            self.assertIs(raw.d.guard, p.guard)
            if fail:
                raise RuntimeError('mock load failure')
            return 'raw-model', {'kind': 'raw', 'internal_step': 2000}
        raw.raw_load = load
        return pc, raw, cfg, original_guard

    def test_production_transform_order(self):
        pc, raw, cfg, old_guard = self.setup_factory()
        old_factory = pc.create_trained_policy
        factory = p.raw_factory(pc, raw, '/unused/raw_restore.json')
        with p.replaced(pc, 'create_trained_policy', factory):
            model, kw = pc.create_trained_policy(cfg, p.PILOT,
                repack_transforms=NS(inputs=['repack-in'], outputs=['repack-out']))
        self.assertIs(raw.d.guard, old_guard)
        self.assertIs(pc.create_trained_policy, old_factory)
        self.assertEqual(model, 'raw-model')
        self.assertEqual(kw['transforms'], ['repack-in', ('prompt', None), 'data-in',
                         ('normalize', 'stats', {'use_quantiles': True}), 'model-in'])
        self.assertEqual(kw['output_transforms'], ['model-out',
                         ('unnormalize', 'stats', {'use_quantiles': True}), 'data-out', 'repack-out'])
        self.assertFalse(kw['is_pytorch'])
        self.assertIsNone(kw['pytorch_device'])

    def test_both_patches_restore_on_failure(self):
        pc, raw, cfg, old_guard = self.setup_factory(fail=True)
        old_factory = pc.create_trained_policy
        with self.assertRaisesRegex(RuntimeError, 'mock load failure'):
            with p.replaced(pc, 'create_trained_policy', p.raw_factory(pc, raw, '/unused/evidence')):
                pc.create_trained_policy(cfg, p.PILOT)
        self.assertIs(raw.d.guard, old_guard)
        self.assertIs(pc.create_trained_policy, old_factory)

    def test_strict_factory_options(self):
        pc, raw, cfg, old_guard = self.setup_factory()
        factory = p.raw_factory(pc, raw, '/unused/evidence')
        for kwargs in ({'norm_stats': {}}, {'sample_kwargs': {}}):
            with self.assertRaises(ValueError):
                factory(cfg, p.PILOT, **kwargs)
        with self.assertRaises(ValueError):
            factory(cfg, '/wrong/checkpoint')
        cfg.model.action_horizon = 10
        with self.assertRaises(ValueError):
            factory(cfg, p.PILOT)
        self.assertIs(raw.d.guard, old_guard)

    def test_inherited_attribute_restoration(self):
        class Base:
            value = 'old'
        obj = Base()
        with self.assertRaises(RuntimeError):
            with p.replaced(obj, 'value', 'new'):
                self.assertEqual(obj.value, 'new')
                raise RuntimeError()
        self.assertNotIn('value', vars(obj))
        self.assertEqual(obj.value, 'old')


class BindingTests(unittest.TestCase):
    def test_transparent_calls_rng_and_restore(self):
        import random
        description = p.ROOT / 'robotwin_repo_ba0fd21/description'
        spec = p.template_spec('place_mouse_pad', 123456, description)
        payload = {'info': {'{A}': '047_mouse/base0', '{B}': 'Blue', '{a}': 'right'}}
        calls = []
        class Task:
            def play_once(self):
                calls.append('play')
                return payload
            def set_instruction(self, instruction):
                calls.append(('set', instruction))
                return 'unchanged-return'
        state = random.getstate()
        records = []
        for _ in range(2):
            task, record = Task(), {}
            with p.bound_task(task, spec, description, record):
                self.assertIs(task.play_once(), payload)
                self.assertEqual(task.set_instruction(spec['template']), 'unchanged-return')
            self.assertNotIn('play_once', vars(task))
            self.assertNotIn('set_instruction', vars(task))
            records.append(record)
        self.assertEqual(records[0], records[1])
        self.assertEqual(random.getstate(), state)
        self.assertEqual([x if isinstance(x, str) else x[0] for x in calls], ['play', 'set', 'play', 'set'])
        wrong = {'info': dict(payload['info'], **{'{B}': 'Red'})}
        self.assertNotEqual(p.bind_prompt(spec, wrong, description), records[0])
        with self.assertRaises(ValueError):
            p.bind_prompt(spec, {'info': {'{A}': 'mouse'}}, description)

    def test_simulator_envelope_only_and_finally_restore(self):
        import numpy as np
        description = p.ROOT/'robotwin_repo_ba0fd21/description'
        spec = p.template_spec('place_mouse_pad', 123456, description)
        calls, sent = [], []
        payload = {'info': {'{A}': '047_mouse/base0', '{B}': 'Blue', '{a}': 'right'}}
        task = NS(play_once=lambda: payload,
                  set_instruction=lambda instruction: calls.append(instruction))
        construct = lambda name: task
        send = lambda stream, value: sent.append(value)
        evaluator, harness = NS(class_decorator=construct), NS(send_message=send)
        initial = {'type': 'episode_start', 'instruction': spec['template']}
        obs = {'agent_pos': np.zeros(14, dtype=np.float32)}
        step = {'type': 'step', 'observation': obs}
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with p.simulator_shim(evaluator, harness, spec, description):
                self.assertIs(evaluator.class_decorator('place_mouse_pad'), task)
                self.assertIs(task.play_once(), payload)
                task.set_instruction(spec['template'])
                harness.send_message(None, initial)
                harness.send_message(None, step)
                raise RuntimeError('sentinel')
        self.assertIs(evaluator.class_decorator, construct)
        self.assertIs(harness.send_message, send)
        self.assertNotIn('diagnostic_binding', initial)
        self.assertEqual(sent[0]['instruction'], spec['template'])
        self.assertEqual(sent[0]['diagnostic_binding']['prompt'], calls[0])
        self.assertNotIn('diagnostic_state', step)
        self.assertIs(sent[1]['observation'], obs)
        self.assertEqual(sent[1]['diagnostic_state']['shape'], [14])

    def test_original_exception_identity(self):
        error = RuntimeError('expert failed')
        calls = []
        class Task:
            def play_once(self):
                calls.append('play')
                raise error
            def set_instruction(self, instruction):
                raise AssertionError('must not run')
        task = Task()
        with self.assertRaises(RuntimeError) as caught:
            with p.bound_task(task, {}, '/unused', {}):
                task.play_once()
        self.assertIs(caught.exception, error)
        self.assertEqual(calls, ['play'])
        self.assertNotIn('play_once', vars(task))
        self.assertNotIn('set_instruction', vars(task))


class TraceGateTests(unittest.TestCase):
    def test_full_chunk_and_noise_observation(self):
        import io
        import numpy as np
        calls = []
        class Policy:
            def infer(self, obs, noise):
                calls.append(noise)
                return np.zeros((50, 14), dtype=np.float32)
        class Adapter:
            checkpoint_format = 'openpi_orbax'
            noise_source = 'jax'
            policy = Policy()
            def infer_chunk(self, **kwargs):
                return self.policy.infer({}, noise=np.ones((50, 32), dtype=np.float32))
        adapter = Adapter()
        trace = p.NumericTrace(io.StringIO(), lambda count: None)
        with trace.policy(adapter):
            self.assertEqual(adapter.infer_chunk().shape, (50, 14))
        self.assertEqual(len(calls), 1)
        self.assertEqual(trace.chunks, 1)
        self.assertEqual(len(trace.noises), 1)
        self.assertNotIn('infer_chunk', vars(adapter))
        self.assertNotIn('infer', vars(adapter.policy))
        with self.assertRaises(ValueError):
            p.array_record(np.zeros((10, 14)), [50, 14])
        with self.assertRaises(ValueError):
            trace.response({'type': 'step'})
        value = {'type': 'step', 'terminated': True}
        self.assertIs(trace.response(value), value)

    def test_noise_mismatch_stops_before_inference(self):
        import io
        import numpy as np
        from unittest.mock import Mock
        policy = NS(infer=Mock())
        adapter = NS(checkpoint_format='openpi_orbax', noise_source='jax', policy=policy)
        adapter.infer_chunk = lambda: policy.infer({}, noise=np.zeros((50, 32)))
        original_chunk, original_infer = adapter.infer_chunk, policy.infer
        trace = p.NumericTrace(io.StringIO(), lambda count: None)
        with self.assertRaisesRegex(ValueError, 'before inference'):
            with trace.policy(adapter, reference_noise=['wrong']):
                adapter.infer_chunk()
        original_infer.assert_not_called()
        self.assertIs(policy.infer, original_infer)
        self.assertIs(adapter.infer_chunk, original_chunk)

    def test_paired_binding_checked_before_episode_or_adapter(self):
        import sys
        from unittest.mock import Mock
        description = p.ROOT/'robotwin_repo_ba0fd21/description'
        spec = p.template_spec('place_mouse_pad', 123456, description)
        binding = p.bind_prompt(spec, {'info': {'{A}': '047_mouse/base0',
                               '{B}': 'Blue', '{a}': 'right'}}, description)
        first = dict(type='episode_start', seed=123456, instruction=spec['template'],
                     diagnostic_binding=binding)
        case = dict(seed=123456, template=spec)
        base = NS(episode=Mock())
        reference = dict(status='completed', binding=dict(binding, prompt='different'))
        with patch.dict(sys.modules, {'controlled_paired_eval': base}), \
                patch.object(p, 'build_adapter') as build:
            with self.assertRaisesRegex(ValueError, 'bindings differ'):
                p.paired_episode(first, None, None, Path('/unused'), case, 'raw',
                                 Path('/unused'), reference)
            base.episode.assert_not_called()
            build.assert_not_called()
        self.assertEqual(first['instruction'], spec['template'])

    def test_gate_task_failure_is_valid(self):
        base = dict(status='completed', success=False, hashes={'x': 'same'},
                    binding={'prompt': 'same'}, noise_hashes=['same'])
        self.assertEqual(p.pair_gate(base, dict(base, success=True)), 'valid')
        self.assertEqual(p.pair_gate({'status': 'expert_rejected'}, {'status': 'expert_rejected'}), 'excluded')
        for update in ({'status': 'worker_error'}, {'hashes': {'x': 'different'}},
                       {'binding': {'prompt': 'different'}}, {'noise_hashes': ['different']}):
            with self.assertRaises(ValueError):
                p.pair_gate(base, dict(base, **update))
        p.require_same_binding({'x': 1}, {'x': 1})
        with self.assertRaises(ValueError):
            p.require_same_binding({'x': 1}, {'x': 2})

    def test_manifest_no_unknown_fields_or_identity_changes(self):
        frozen = dict(version=1, cases=[1], code={'a': 'b'})
        manifest = dict(frozen, identities={'raw': {'file': 'hash'}})
        def expected(include_weights=True):
            return manifest if include_weights else frozen
        with patch.object(p, 'expected_manifest', side_effect=expected):
            p.validate(manifest)
            p.validate(manifest, check_files=False)
            for wrong in (dict(manifest, unknown=True), dict(manifest, cases=[2]),
                          dict(manifest, identities={'raw': {}})):
                with self.assertRaises(ValueError):
                    p.validate(wrong)

    def test_base_episode_reuse_full_fifty_and_initial_mismatch(self):
        import io
        import sys
        import numpy as np
        import importlib.util
        source = Path(__file__).with_name('controlled_paired_eval.py')
        spec = importlib.util.spec_from_file_location('base_episode_test', source)
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
        policy_calls = []
        class Policy:
            checkpoint_format = 'openpi_orbax'
            noise_source = 'jax'
            def infer_chunk(self, **kwargs):
                policy_calls.append(kwargs)
                return np.zeros((50, 14), dtype=np.float32)
        def convert(obs):
            return {}, np.zeros(14, dtype=np.float32)
        helper = NS(_observation_for_openpi=convert)
        first = dict(type='episode_start', seed=1, instruction='bound', observation={})
        case = dict(seed=1, prompt='bound')
        sent = []
        responses = iter([dict(type='step', observation={}, terminated=i == 49,
                               info={'is_success': False}) for i in range(50)])
        with patch.dict(sys.modules, {'robotwin_public_openpi_harness': helper}), \
                patch.object(base, 'capture', return_value={'state': 'same'}):
            result = base.episode(first, lambda: next(responses), sent.append,
                                  Policy, Path('/unused'), case)
            self.assertEqual(result['status'], 'completed')
            self.assertFalse(result['success'])
            self.assertEqual(len(sent), 50)
            self.assertEqual(len(policy_calls), 1)
            with patch.object(base, 'guard_policy') as policy_guard:
                result = base.episode(first, lambda: None, lambda value: None,
                                      Policy, Path('/unused'), case,
                                      reference={'state': 'different'}, pilot=True)
                self.assertEqual(result['status'], 'initial_mismatch')
                policy_guard.assert_not_called()

    def test_selected_panel_is_fixed_and_excludes_inventory(self):
        def spec(task, seed, root):
            return {'source': task, 'source_sha256': 'hash', 'seed': seed}
        with patch.object(p, 'template_spec', side_effect=spec):
            a = p.selected_cases(({200000, 200002}, {}), '/unused')
            b = p.selected_cases(({200000, 200002}, {}), '/unused')
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)
        self.assertEqual(len({x['id'] for x in a}), 20)
        self.assertFalse({x['seed'] for x in a} & {200000, 200002})
        for task in ('hanging_mug', 'place_mouse_pad'):
            self.assertEqual(len([x for x in a if x['task'] == task]), 10)

    def test_raw_evidence_rejects_nonparams(self):
        import json
        evidence = dict(kind='raw', internal_step=2000, checkpoint=str(p.PILOT/'train_state'),
                        logical_reads=[['step'], ['params', 'x', 'value']],
                        leaves=[dict(key=['params', 'x', 'value'], native_dtype='float32', inference_dtype='bfloat16')])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'evidence.json'
            path.write_text(json.dumps(evidence))
            self.assertEqual(p.check_raw_evidence(path)['logical_parameter_leaves'], 1)
            evidence['logical_reads'].append(['opt_state', 'x'])
            path.write_text(json.dumps(evidence))
            with self.assertRaises(ValueError):
                p.check_raw_evidence(path)

    def test_worker_ipc_shim_factory_chain(self):
        import io
        import sys
        import json
        import numpy as np
        description = p.ROOT/'robotwin_repo_ba0fd21/description'
        spec = p.template_spec('place_mouse_pad', 123456, description)
        case = dict(id='mock', task='place_mouse_pad', seed=123456, template=spec)
        queue, instructions = [], []
        task = NS(play_once=lambda: {'info': {'{A}': '047_mouse/base0', '{B}': 'Blue', '{a}': 'right'}},
                  set_instruction=lambda instruction: instructions.append(instruction))
        harness = NS(send_message=lambda stream, value: queue.append(value))
        evaluator = NS(class_decorator=lambda name: task)
        with p.simulator_shim(evaluator, harness, spec, description):
            evaluator.class_decorator('place_mouse_pad').play_once()
            task.set_instruction(spec['template'])
            harness.send_message(None, dict(type='episode_start', seed=123456,
                                           instruction=spec['template'], observation={}))
            harness.send_message(None, dict(type='step', terminated=True, info={'is_success': False}))
        pc, raw, cfg, old_guard = FactoryTests().setup_factory()
        class ModelPolicy:
            def infer(self, obs, noise):
                return np.zeros((50, 14), dtype=np.float32)
        pc._policy.Policy = lambda model, **kw: ModelPolicy()
        class Adapter:
            checkpoint_format = 'openpi_orbax'
            noise_source = 'jax'
            def __init__(self, checkpoint, **kwargs):
                self.policy = pc.create_trained_policy(cfg, checkpoint)
            def infer_chunk(self, **kwargs):
                return self.policy.infer({}, noise=np.zeros((50, 32)))
        def episode(first, receive, send, make, directory, case, **kwargs):
            self.assertEqual(first['instruction'], instructions[0])
            adapter = make()
            adapter.infer_chunk()
            send({'cmd': 'action', 'action': [0]*14})
            receive()
            return dict(status='completed', success=False, hashes={'state': 'same'})
        base = NS(CONFIG=dict(sim_python='/unused', episode_steps=1, robotwin_root='/unused', task_config='unused'),
                  MODELS={'reference': '/original'}, episode=episode, capture=lambda *a: None)
        modules = {'controlled_paired_eval': base, 'gen1_raw_diagnostic': raw,
                   'pi05_openpi_adapter': NS(PublicPi05Adapter=Adapter),
                   'openpi': NS(), 'openpi.policies': NS(policy_config=pc),
                   'robotwin_public_openpi_harness': NS(_configure_sim_renderer_env=lambda env: None),
                   'robotwin_eval_harness': NS(receive_message=lambda stream: queue.pop(0), send_message=lambda *a: None)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.object(p, 'guard'), patch.object(p, 'runtime_environment', return_value={}), \
                patch.object(p, 'restrict_worker_files'), \
                patch.object(p, 'check_raw_evidence', return_value={'ok': True}):
            output = Path(tmp); directory = output/'mock_raw'; directory.mkdir()
            original = output/'mock_original'; original.mkdir()
            binding = queue[0]['diagnostic_binding']
            noise = p.array_record(np.zeros((50, 32)), [50, 32])['sha256']
            (original/'result.json').write_text(json.dumps(dict(status='completed', binding=binding,
                                                    hashes={'state': 'same'}, noise_hashes=[noise])))
            child = NS(stdin=io.BytesIO(), stdout=io.BytesIO(), returncode=0, wait=lambda timeout: 0)
            with patch.object(p.subprocess, 'Popen', return_value=child):
                self.assertEqual(p.worker({}, 'raw', case, output), 0)
            self.assertEqual(json.loads((directory/'result.json').read_text())['noise_hashes'], [noise])
        self.assertIs(raw.d.guard, old_guard)

    def test_worker_manifest_digest_version_and_full_identity_check(self):
        import json
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'manifest.json'
            value = dict(runner='gen1_raw_paired_eval_v1', version=1, identities={'raw': {}})
            data = json.dumps(value).encode()
            path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            with patch.object(p, 'validate') as validate:
                self.assertEqual(p.verified_manifest(path, digest), value)
                validate.assert_called_once_with(value, check_files=True)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                p.verified_manifest(path, 'wrong')
            value['version'] = 2
            data = json.dumps(value).encode(); path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'version'):
                p.verified_manifest(path, hashlib.sha256(data).hexdigest())

    def test_historical_seed_schemas_fail_closed(self):
        excluded = set()
        p.extract_seeds(dict(seed=7, accepted_seeds=[9, 10], seed_start=100,
                             explicit_seed=False), excluded)
        self.assertTrue({7, 9, 10, 100, 1099} <= excluded)
        for value in ({'seed': None}, {'seed': '7'}, {'seeds': [1, '2']}):
            with self.assertRaises(ValueError):
                p.extract_seeds(value, set())
        p.extract_seeds(dict(global_step=1, episode='0', step=1, seed=None), set())

    def test_log_cap_stops_before_excess_write(self):
        import io
        state = dict(bytes=8*2**20, truncated=False)
        selector = NS(select=lambda timeout: [(NS(fd=123), None)])
        log = NS(buffer=io.BytesIO(), flush=lambda: None)
        with patch.object(p.os, 'read', return_value=b'x'), patch.object(p, 'guard'):
            with self.assertRaisesRegex(RuntimeError, 'log cap'):
                p.drain_log(None, log, '/unused', state, selector)
        self.assertTrue(state['truncated'])
        self.assertEqual(log.buffer.getvalue(), b'')

    def test_history_membership_ignores_only_current_manifest(self):
        old = {'results/a.json': 'hash'}
        with patch.object(p, 'HISTORICAL_FILES', old), patch.object(p, 'historical_paths',
                return_value={'results/a.json', 'outputs/current.json'}):
            p.audit_inventory_membership(p.ROOT/'outputs/current.json')
        with patch.object(p, 'HISTORICAL_FILES', old), patch.object(p, 'historical_paths',
                return_value={'results/a.json', 'outputs/current.json', 'outputs/new.json'}):
            with self.assertRaisesRegex(ValueError, 'audit required'):
                p.audit_inventory_membership(p.ROOT/'outputs/current.json')

    def test_native_file_caps_are_inherited_and_disable_core(self):
        import resource
        with patch.object(resource, 'getrlimit', return_value=(-1, -1)), \
                patch.object(resource, 'setrlimit') as setlimit:
            p.restrict_worker_files()
        self.assertEqual(setlimit.call_args_list[0].args,
                         (resource.RLIMIT_FSIZE, (16*2**20, 16*2**20)))
        self.assertEqual(setlimit.call_args_list[1].args, (resource.RLIMIT_CORE, (0, 0)))

    def test_runner_first_pair_failure_aborts_without_substitution(self):
        import json
        import sys
        cases = [dict(id=f'case_{i}', seed=i) for i in range(20)]
        frozen = dict(cases=cases, limitation='test', protocol='test')
        base = dict(status='completed', success=False, hashes={'state': 'same'},
                    binding={'prompt': 'same'}, noise_hashes=['same'])
        c = NS(CONFIG={'inference_python': '/unused/python'})
        raw = NS()
        for first_excluded, expected_count in ((True, 2), (False, 40)):
            with tempfile.TemporaryDirectory() as tmp, patch.object(p, 'OUTPUTS', Path(tmp)), \
                    patch.object(p, 'guard'), patch.object(p, 'runtime_environment', return_value={}), \
                    patch.object(p, 'historical_paths', return_value={name for name in p.HISTORICAL_FILES
                        if name.split('/')[0] in ('outputs', 'results', 'artifacts')}), \
                    patch.dict(sys.modules, {'controlled_paired_eval': c, 'gen1_raw_diagnostic': raw}):
                output = Path(tmp)/'run'
                def supervised(command, env, log, out, raw):
                    label = command[command.index('--label')+1]
                    index = int(command[command.index('--case')+1])
                    value = {'status': 'expert_rejected'} if first_excluded and index == 0 else base
                    (out/f'case_{index}_{label}'/'result.json').write_text(json.dumps(value))
                    return {'status': 'passed'}
                manifest = Path(tmp)/'manifest.json'
                manifest.write_text(json.dumps(frozen))
                with patch.object(p, 'supervise', side_effect=supervised) as launch:
                    rc = p.run(frozen, manifest, output)
                self.assertEqual(launch.call_count, expected_count)
                self.assertEqual(rc, 1 if first_excluded else 0)
                saved = json.loads((output/'summary.json').read_text())
                self.assertEqual(saved['integration_gate'], 'excluded' if first_excluded else 'valid')
                self.assertEqual(len(saved['records']), expected_count)
                if not first_excluded:
                    self.assertTrue(all(not x['result']['success'] for x in saved['records']))

    def test_supervisor_cleanup_on_resource_and_timeout(self):
        snapshot = {'groups': {}}
        raw = NS(discover_cgroups=lambda: ['mock'], memory_snapshot=lambda _: snapshot,
                 check_memory=lambda *args, **kwargs: None)
        proc = NS(pid=123, wait=lambda **kwargs: 0)
        with patch.object(p, 'guard'), patch.object(p.subprocess, 'Popen', return_value=proc), \
                patch.object(p, 'kill_group') as kill:
            self.assertEqual(p.supervise([], {}, None, '/unused', raw)['status'], 'passed')
            kill.assert_called_once_with(proc)
        with patch.object(p, 'guard'), patch.object(p.subprocess, 'Popen', return_value=proc), \
                patch.object(p, 'kill_group') as kill, patch.object(p.time, 'monotonic', side_effect=[0, 2]):
            self.assertEqual(p.supervise([], {}, None, '/unused', raw, timeout=1)['status'], 'failed')
            kill.assert_called_once_with(proc)
        raw.check_memory = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('memory'))
        with patch.object(p.subprocess, 'Popen') as launch:
            with self.assertRaises(RuntimeError):
                p.supervise([], {}, None, '/unused', raw)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
