#!/usr/bin/env python3
"""CPU-only unit tests: no real checkpoint tensor reads, GPU, or manifest hashing."""
import argparse
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name('gen1_raw_diagnostic.py')
spec = importlib.util.spec_from_file_location('raw_diagnostic_test_target', SCRIPT)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class RawDiagnosticTests(unittest.TestCase):
    def test_import_no_backend(self):
        code = ('import importlib.util,sys; s=importlib.util.spec_from_file_location("r", ' + repr(str(SCRIPT)) +
                '); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                'assert not any(x in sys.modules for x in ("jax","torch","numpy","orbax"))')
        subprocess.run([sys.executable, '-I', '-B', '-c', code], check=True)

    def test_exact_suffix(self):
        self.assertEqual(r.extract_params({('params', 'value', 'value'): 1,
                         ('params', 'value_projection', 'value'): 2, ('optimizer', 'x'): 3}),
                         {('value',): 1, ('value_projection',): 2})
        for key in [('params', 'x', 'myvalue'), ('params', 'value'), 'params/x/value']:
            with self.assertRaises(RuntimeError):
                r.extract_params({key: 1})

    def test_metadata_reference_rejections(self):
        value = SimpleNamespace(shape=(2, 3), dtype='float32')
        reference = {('dense',): ((2, 3), 'float32')}
        r.validate_reference({('dense',): value}, reference)
        for values in [{}, {('extra',): value}, {('dense',): SimpleNamespace(shape=(3, 2), dtype='float32')},
                       {('dense',): SimpleNamespace(shape=(2, 3), dtype='bfloat16')}]:
            with self.assertRaises(RuntimeError):
                r.validate_reference(values, reference)

    def test_native_array_validation(self):
        import numpy as np
        r.validate_array(np.zeros((2,), dtype=np.float32), (2,), 'float32')
        for value in [[1., 2.], np.zeros((2,), dtype=np.float16), np.zeros((3,), dtype=np.float32),
                      np.array([float('nan'), 0], dtype=np.float32), np.array([float('inf'), 0], dtype=np.float32)]:
            with self.assertRaises(RuntimeError):
                r.validate_array(value, (2,), 'float32')

    def test_restore_full_tuple_selection_no_dtype_cast(self):
        import numpy as np
        key = ('params', 'value', 'kernel', 'value')
        calls = []
        ocp = SimpleNamespace(tree=SimpleNamespace(from_flat_dict=lambda x: x, to_flat_dict=lambda x: x),
                              RestoreArgs=lambda **kw: kw, args=SimpleNamespace(PyTreeRestore=lambda **kw: kw))
        reader = SimpleNamespace(restore=lambda path, args: calls.append(args) or {key: np.ones((2,), dtype=np.float32)})
        with mock.patch.object(r, 'trace_leaf', return_value=nullcontext()):
            r.restore_leaf(reader, Path('/fixture'), key, ocp, [])
        self.assertEqual(calls[0]['item'], {key: 0})
        self.assertEqual(calls[0]['transforms'], {})
        self.assertEqual(calls[0]['restore_args'], {key: {'restore_type': np.ndarray}})
        for forbidden in [('optimizer', 'x'), ('params',), ('params', 'x', 'value_extra')]:
            with self.assertRaises(RuntimeError):
                r.restore_leaf(reader, Path('/fixture'), forbidden, ocp, [])
        self.assertEqual(len(calls), 1)

    @staticmethod
    def snapshot(headroom=30 * r.GIB, oom=9, kill=4):
        return dict(groups={'/fixture': dict(headroom=headroom, events={'oom': oom, 'oom_kill': kill})},
                    host_available=100 * r.GIB)

    def test_historical_oom_and_new_events(self):
        baseline = self.snapshot()
        r.check_memory(baseline, baseline=baseline)
        for snap in [self.snapshot(11 * r.GIB), self.snapshot(oom=10), self.snapshot(kill=5), self.snapshot(oom=0)]:
            with self.assertRaises(RuntimeError):
                r.check_memory(snap, baseline=baseline)
        no_limit = self.snapshot(None)
        with self.assertRaisesRegex(RuntimeError, 'host-only'):
            r.check_memory(no_limit)

    def test_headroom_formula(self):
        with mock.patch.object(r, 'memory_snapshot', return_value=self.snapshot()), mock.patch.object(r, 'check_memory') as check:
            r.pre_read([], r.GIB, {})
            self.assertEqual(check.call_args.args[1], 24 * r.GIB)
            r.pre_read([], 8 * r.GIB, {})
            self.assertEqual(check.call_args.args[1], 32 * r.GIB)

    def test_min_high_max_visible_ancestors(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / 'parent', Path(folder) / 'child']
            for path, high, maximum, current in [(paths[0], '100', '200', '70'), (paths[1], 'max', '150', '100')]:
                path.mkdir()
                for name, value in [('memory.high', high), ('memory.max', maximum), ('memory.current', current),
                                    ('memory.events', 'oom 9\noom_kill 4\n')]:
                    (path / name).write_text(value)
            snap = r.memory_snapshot(paths)
            self.assertEqual([g['headroom'] for g in snap['groups'].values()], [30, 50])
            with self.assertRaises(RuntimeError):
                r.check_memory(snap, needed=31)

    def test_supervision_kills_on_new_oom(self):
        baseline, bad = self.snapshot(), self.snapshot(oom=10)
        proc = mock.Mock(pid=12345)
        proc.poll.return_value = None
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(r, 'discover_cgroups', return_value=[]), \
             mock.patch.object(r, 'memory_snapshot', side_effect=[baseline, bad]), \
             mock.patch.object(r.subprocess, 'Popen', return_value=proc), mock.patch.object(r.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'oom'):
                r.supervise(['not-executed'], 10, Path(folder))
            kill.assert_called_once_with(12345, r.signal.SIGKILL)
            result = r.d.read_json(Path(folder) / 'supervision.json')
            self.assertEqual(result['baseline']['groups']['/fixture']['events']['oom'], 9)
            self.assertEqual(result['last']['groups']['/fixture']['events']['oom'], 10)
            self.assertEqual(result['status'], 'failed')

    def test_gpu_opt_in_precedes_work(self):
        a = argparse.Namespace(allow_inference=False)
        with mock.patch.object(r, 'validate_manifest') as validate:
            for fn in (r.run, r.worker):
                with self.assertRaisesRegex(RuntimeError, 'allow-inference'):
                    fn(a)
            validate.assert_not_called()

    def test_full_requires_smoke(self):
        a = argparse.Namespace(allow_inference=True, manifest=Path('/fake'), command='evaluate', smoke_report=None)
        with mock.patch.object(r, 'validate_manifest'):
            with self.assertRaisesRegex(RuntimeError, 'smoke'):
                r.run(a)

    def test_raw_load_validation_before_reads_and_transfer_order(self):
        import numpy as np
        events = []
        abstract = {('value',): SimpleNamespace(shape=(2,), dtype='float32'),
                    ('value_projection',): SimpleNamespace(shape=(1,), dtype='float32')}
        metadata = {('params',) + k + ('value',): v for k, v in abstract.items()}
        metadata[('step',)] = SimpleNamespace(shape=(), dtype='int32')
        metadata[('optimizer', 'momentum')] = SimpleNamespace(shape=(999,), dtype='float32')
        reader = mock.MagicMock()
        reader.__enter__.return_value = reader
        reader.metadata.return_value = metadata
        ocp = SimpleNamespace(tree=SimpleNamespace(to_flat_dict=lambda x: x, from_flat_dict=lambda x: x),
                              Checkpointer=lambda handler: reader, PyTreeCheckpointHandler=lambda: None)
        nnx = SimpleNamespace(eval_shape=lambda *args: None,
                              state=lambda x: SimpleNamespace(to_pure_dict=lambda: abstract))
        jax = SimpleNamespace(devices=lambda: [SimpleNamespace(platform='gpu')], random=SimpleNamespace(key=lambda x: x))

        def convert(value, dtype):
            events.append(('cast', value.shape))
            return SimpleNamespace(dtype='bfloat16', block_until_ready=lambda: events.append(('ready', value.shape)))

        def restore(reader, checkpoint, key, ocp, evidence):
            events.append(('read', key))
            evidence.append(list(key))
            if key == ('step',):
                return np.asarray(2000, dtype=np.int32)
            return np.zeros(abstract[key[1:-1]].shape, dtype=np.float32)

        stat = SimpleNamespace(**{name: np.zeros(2) for name in ('mean', 'std', 'q01', 'q99')})
        normalize = SimpleNamespace(load=lambda x: {'actions': stat})
        jax.numpy = SimpleNamespace(asarray=convert, bfloat16='bfloat16')
        fake_modules = {'flax': SimpleNamespace(nnx=nnx), 'flax.nnx': nnx,
                        'jax': jax, 'jax.numpy': jax.numpy,
                        'orbax': SimpleNamespace(checkpoint=ocp), 'orbax.checkpoint': ocp,
                        'openpi': SimpleNamespace(), 'openpi.shared': SimpleNamespace(normalize=normalize)}
        cfg = SimpleNamespace(model=SimpleNamespace(create=lambda: None, load=lambda params, **kw: params))
        baseline = self.snapshot()
        with mock.patch.dict(sys.modules, fake_modules), \
             mock.patch('importlib.metadata.version', return_value='0.11.13'), \
             mock.patch.object(r, 'discover_cgroups', return_value=[]), \
             mock.patch.object(r, 'memory_snapshot', return_value=baseline), \
             mock.patch.object(r, 'pre_read'), mock.patch.object(r, 'restore_leaf', side_effect=restore), \
             mock.patch.object(r.d, 'guard'), mock.patch.object(r.d, 'write_json') as write:
            loaded, dtype = r.raw_load(cfg, r.d.PILOT / '1999', Path('/fake/raw/raw_restore.json'))
            self.assertEqual(set(loaded), set(abstract))
            self.assertEqual(dtype['kind'], 'raw')
            self.assertEqual(dtype['native'], ['float32'])
            self.assertEqual(events, [('read', ('step',)),
                ('read', ('params', 'value', 'value')), ('cast', (2,)), ('ready', (2,)),
                ('read', ('params', 'value_projection', 'value')), ('cast', (1,)), ('ready', (1,))])
            self.assertEqual(write.call_args.args[1]['internal_step'], 2000)
            events.clear()
            metadata[('params', 'value', 'value')] = SimpleNamespace(shape=(3,), dtype='float32')
            with self.assertRaisesRegex(RuntimeError, 'Reference shape'):
                r.raw_load(cfg, r.d.PILOT / '1999', Path('/fake/raw/raw_restore.json'))
            self.assertEqual(events, [])

    def test_saved_original_identities_and_raw_limitations(self):
        fixed = r.fixed_inputs()  # Small fixed manifest only, not its large tracked files.
        r.previous_inputs(fixed)  # Existing saved JSON results only, never tensors.
        records = r.d.read_json(r.PREVIOUS['smoke'] / 'original/records.json')
        complete = r.d.read_json(r.PREVIOUS['smoke'] / 'original/completed.json')
        records[0]['model'] = 'ema'
        with self.assertRaisesRegex(RuntimeError, 'RNG/model'):
            r.check_records(records, complete, fixed, 'original', True)
        self.assertFalse(any('intentionally omitted' in s for s in r.limitations()))


if __name__ == '__main__':
    unittest.main()
