"""CPU-only protocol tests using fake IPC and policy."""
import ast
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import controlled_paired_eval as c
import robotwin_eval_harness as h

class Tests(unittest.TestCase):
    def test_default_parser(self):
        with patch.object(sys, 'argv', ['harness']): a = h.parse_args()
        self.assertFalse(a.controlled_case)
        self.assertIsNone(a.output)
        self.assertEqual((a.seed, a.episodes), (100000, 5))
        self.assertIsNone(a.first_instruction)

    def test_guard(self):
        c.guard_policy(SimpleNamespace(checkpoint_format='openpi_orbax', noise_source='jax'))
        for mode, noise in [('openpi_pytorch', 'jax'), ('openpi_orbax', 'internal')]:
            with self.assertRaises(ValueError):
                c.guard_policy(SimpleNamespace(checkpoint_format=mode, noise_source=noise))

    def test_input_capture(self):
        images = {k: np.zeros((2, 3, 3), dtype=np.uint8) for k in ('cam_high','cam_left_wrist','cam_right_wrist')}
        with tempfile.TemporaryDirectory() as d:
            a = c.capture(Path(d)/'a', images, np.arange(14), 'exact prompt')
            b = c.capture(Path(d)/'b', images, np.arange(14).astype('>f4'), 'exact prompt')
            self.assertEqual(a,b)
            z = c.capture(Path(d)/'z', images, np.arange(14), 'different')
            self.assertNotEqual(a,z)
            self.assertTrue((Path(d)/'a/state.npy').exists())

    def test_rejection_and_seed_gate(self):
        case = dict(seed=100005, prompt='fixed')
        factory = lambda: self.fail('policy must not load')
        r = c.episode(dict(type='case_status', seed=100005, status='expert_rejected'), None, None, factory, Path('/unused'), case)
        self.assertEqual(r['status'],'expert_rejected')
        r = c.episode(dict(type='case_status', seed=100006, status='expert_rejected'), None, None, factory, Path('/unused'), case)
        self.assertEqual(r['status'],'seed_mismatch')

    def test_fake_ipc_and_gates(self):
        case = dict(seed=7,prompt='fixed')
        images = {k: np.zeros((2,3,3),dtype=np.uint8) for k in ('cam_high','cam_left_wrist','cam_right_wrist')}
        loaded = []; sent = []
        def factory():
            loaded.append(0)
            return SimpleNamespace(checkpoint_format='openpi_orbax', noise_source='jax',
                infer_chunk=lambda **kw: np.zeros((1,14)))
        first = dict(type='episode_start', seed=7, instruction='fixed', observation={})
        # Patch conversion only; hashes, ordering, policy guards and IPC remain real.
        import types
        fake = types.ModuleType('robotwin_public_openpi_harness')
        fake._observation_for_openpi = lambda obs: (images,np.zeros(14))
        with patch.dict(sys.modules, robotwin_public_openpi_harness=fake), tempfile.TemporaryDirectory() as d:
            path = Path(d)
            r = c.episode(first,None,None,factory,path,case,pilot=True)
            self.assertEqual(r['status'],'missing_reference'); self.assertEqual(loaded,[])
            hashes = r['hashes']
            (path/'second').mkdir()
            r = c.episode(first,None,None,factory,path/'second',case,{},pilot=True)
            self.assertEqual(r['status'],'initial_mismatch'); self.assertEqual(loaded,[])
            (path/'third').mkdir()
            r = c.episode(first,lambda: dict(type='step',terminated=True,info={'is_success':True}),sent.append,factory,path/'third',case,hashes,pilot=True)
            self.assertEqual(r['status'],'completed'); self.assertTrue(r['success'])
            self.assertEqual(len(loaded),1); self.assertEqual(len(sent),1)
            for name, response, expected in [('failure',dict(type='step',terminated=True),'completed'), ('error',dict(type='error'),'simulator_error'), ('unexpected',dict(type='unexpected'),'simulator_error')]:
                (path/name).mkdir()
                r=c.episode(first,lambda:response,sent.append,factory,path/name,case)
                self.assertEqual(r['status'],expected)
                if expected=='completed': self.assertFalse(r['success'])
            (path/'limit').mkdir()
            with patch.dict(c.CONFIG,episode_steps=2):
                r=c.episode(first,lambda:dict(type='step',observation={}),sent.append,factory,path/'limit',case)
            self.assertEqual(r['steps'],2)
            before=len(loaded)
            r=c.episode(dict(first,instruction='wrong'),None,None,factory,path,case)
            self.assertEqual(r['status'],'prompt_mismatch'); self.assertEqual(len(loaded),before)

    def test_worker_constructor(self):
        import types
        fake = types.ModuleType('robotwin_public_openpi_harness')
        fake._configure_sim_renderer_env=lambda env: None
        child=SimpleNamespace(stdin=SimpleNamespace(close=lambda:None),stdout=object(),wait=lambda **kw:0)
        made=[]
        def run_episode(first, receive, send, factory, *args, **kw):
            factory(); return dict(status='completed', success=False)
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules,robotwin_public_openpi_harness=fake), patch.object(c.subprocess,'Popen',return_value=child), patch.object(h,'receive_message',return_value={}), patch.object(h,'send_message'), patch.object(c,'episode',side_effect=run_episode), patch('pi05_openpi_adapter.PublicPi05Adapter',side_effect=lambda *a,**kw:made.append((a,kw))):
            for _ in range(2):
                sub=Path(d)/str(len(made)); sub.mkdir()
                c.worker(dict(models=c.MODELS),'reference',dict(id='a',panel='primary',seed=7,task='hanging_mug',prompt='fixed'),sub,None)
        self.assertEqual(len(made),2)
        self.assertEqual(made[0],made[1])
        self.assertEqual(made[0][1],dict(device='cuda:0',prompt_prefix='none',normalization='quantile',policy_seed=0,noise_source='jax'))

    def test_fresh_explicit_rng_stream(self):
        import types
        from pi05_openpi_adapter import PublicPi05Adapter
        fake = types.ModuleType('jax')
        fake.random = SimpleNamespace(split=lambda key: (key+1,key+10),
            normal=lambda key, shape: np.full(shape,key,dtype=np.float32))
        def fresh():
            p = PublicPi05Adapter.__new__(PublicPi05Adapter)
            p.checkpoint_format='openpi_orbax'; p.noise_source='jax'; p.prompt_prefix='none'
            p._jax_rng=0; p.action_horizon=1; p.model_action_dim=32
            p.policy=SimpleNamespace(infer=lambda obs, noise: {'actions': noise[:,:14]})
            return p
        kwargs=dict(images={k:np.zeros((2,3,3),dtype=np.uint8) for k in ('cam_high','cam_left_wrist','cam_right_wrist')},state=np.zeros(14),prompt='fixed')
        with patch.dict(sys.modules,jax=fake):
            first=fresh(); a=first.infer_chunk(**kwargs); b=first.infer_chunk(**kwargs)
            reset=fresh().infer_chunk(**kwargs)
        np.testing.assert_array_equal(a,reset)
        self.assertFalse(np.array_equal(a,b))

    def test_status_counts(self):
        totals = c.summarize([('reference',dict(panel='primary',id='a'),dict(status='expert_rejected')),
            ('pilot',dict(panel='supplemental',id='b'),dict(status='completed',success=True))])
        self.assertEqual(totals['reference/primary']['completed_successes'],0)
        self.assertEqual(totals['pilot/supplemental']['completed_successes'],1)

    def test_controlled_exits_before_default_seed_substitution(self):
        source = Path(h.__file__).read_text()
        self.assertIn('if controlled_case:\n                            send_message',source)
        self.assertIn('if controlled_case:\n                    safe_close_task',source)
        # All legacy rejection/exception increments lie after controlled return.
        self.assertIn('return 0\n                        seed += 1',source)
        self.assertIn('return 2\n                # Official RoboTwin generation',source)

if __name__ == '__main__': unittest.main()
