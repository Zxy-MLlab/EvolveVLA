"""CPU-only fake-policy checks; no model, JAX or simulator imports."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('replay', Path(__file__).with_name('first_chunk_replay_diagnostic.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class Policy:
    def __init__(self):
        self.calls = 0
        self._input_transform = lambda obs: {'state': obs['state'], 'tokenized_prompt': np.array([1, 20, 0], dtype=np.int32)}
        self._sample_actions = lambda inputs, **kw: np.ones((1, 50, 32), dtype=np.float32)
        self._output_transform = lambda out: {'actions': out['actions'][:, :14] * 2}

    def infer(self, obs, *, noise):
        self.calls += 1
        inputs = self._input_transform(obs)
        actions = self._sample_actions(inputs, noise=noise)[0]
        return self._output_transform({'actions': actions})


class Adapter:
    def __init__(self):
        self.policy = Policy()

    def infer_chunk(self, **kw):
        return self.policy.infer(kw, noise=np.zeros((50, 32), np.float32))['actions']


class Tests(unittest.TestCase):
    def test_boundaries_once_and_restore(self):
        adapter = Adapter()
        original_transform = adapter.policy._input_transform
        result, captures, counts = r.capture_once(adapter, state=np.zeros(14, np.float32))
        self.assertEqual(adapter.policy.calls, 1)
        self.assertTrue(all(v == 1 for v in counts.values()))
        self.assertIs(adapter.policy._input_transform, original_transform)
        self.assertNotIn('infer', vars(adapter.policy))
        np.testing.assert_array_equal(result, np.full((50, 14), 2))
        np.testing.assert_array_equal(captures['processed_inputs']['tokenized_prompt'], [1, 20, 0])
        np.testing.assert_array_equal(captures['model_space']['actions'], np.ones((50, 32)))

    def test_restore_on_exception(self):
        adapter = Adapter()
        def fail(*args, **kw):
            raise RuntimeError('fake failure')
        adapter.policy._sample_actions = fail
        with self.assertRaisesRegex(RuntimeError, 'fake failure'):
            r.capture_once(adapter, state=np.zeros(14))
        self.assertIs(adapter.policy._sample_actions, fail)
        self.assertNotIn('infer', vars(adapter.policy))

    def test_snapshot_copies(self):
        a = np.ones(2)
        copy = r.snapshot({'a': a})
        a[:] = 0
        np.testing.assert_array_equal(copy['a'], [1, 1])

    def test_signed_zero(self):
        result = r.metrics(np.array([0.0]), np.array([-0.0]))
        self.assertTrue(result['exact'])
        self.assertFalse(result['byte_exact'])
        self.assertEqual(result['signed_zero_mismatches'], 1)
        self.assertEqual(result['max'], 0)

    def test_metrics_axes(self):
        result = r.metrics(np.zeros((2, 2)), np.array([[0., 2.], [0., 0.]]))
        self.assertEqual(result['mae'], .5)
        self.assertEqual(result['rmse'], 1.)
        self.assertEqual(result['per_horizon']['max'], [2., 0.])
        self.assertEqual(result['per_dimension']['mae'], [0., 1.])

    def test_bad_metrics(self):
        with self.assertRaises(ValueError):
            r.metrics(np.array([np.nan]), np.array([0.]))
        with self.assertRaises(ValueError):
            r.metrics(np.zeros(2), np.zeros(3))

    def test_save_compare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rep in ('r1', 'r2'):
                (root / rep).mkdir()
                r.save_leaves(root / rep, {'physical_output': np.zeros((50, 14), np.float32)})
            report = r.compare(root)
            entry = next(iter(report.values()))
            self.assertTrue(entry['byte_exact'])
            self.assertIn('grippers', entry)
            self.assertEqual(len(entry['joints']['per_dimension']['max']), 12)
            self.assertIn('first_action', entry)
            with self.assertRaises(FileExistsError):
                r.compare(root)

    def test_resource_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'data').write_bytes(b'1234')
            with self.assertRaisesRegex(RuntimeError, 'output-budget'):
                r.resource_gate(p, dict(min_free_bytes=0, output_budget_bytes=1))
            with patch.object(r.shutil, 'disk_usage') as usage:
                usage.return_value.free = 0
                with self.assertRaisesRegex(RuntimeError, 'free-space'):
                    r.resource_gate(p)

    def test_identity_change(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'source'
            p.write_text('one')
            identities = {str(p): r.sha(p)}
            r.verify_identities(identities)
            p.write_text('two')
            with self.assertRaises(ValueError):
                r.verify_identities(identities)

    def test_prepare_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                r.prepare(directory)

    def test_supervisor_timeout_kills_group(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(r.subprocess, 'Popen') as popen, patch.object(r.os, 'killpg') as kill, patch.object(r, 'resource_gate'):
            process = popen.return_value
            process.poll.return_value = None
            process.pid = 123
            with self.assertRaises(TimeoutError):
                r.supervise(['fake'], Path(directory) / 'log', directory, dict(timeout_seconds=-1))
            kill.assert_called_once_with(123, r.signal.SIGKILL)
            process.wait.assert_called_once()
            self.assertTrue(popen.call_args.kwargs['start_new_session'])

    def test_supervisor_nonzero(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(r.subprocess, 'Popen') as popen, patch.object(r.os, 'killpg'), patch.object(r, 'resource_gate'):
            process = popen.return_value
            process.poll.return_value = 1
            process.returncode = 1
            with self.assertRaisesRegex(RuntimeError, 'exited 1'):
                r.supervise(['fake'], Path(directory) / 'log', directory, dict(timeout_seconds=1))


if __name__ == '__main__':
    unittest.main()
