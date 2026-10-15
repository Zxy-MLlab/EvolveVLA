#!/usr/bin/env python3
"""CPU unit tests for gen1_diagnose_pilot without importing JAX or loading weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import copy
import random

SCRIPT = Path(__file__).with_name('gen1_diagnose_pilot.py')
spec = importlib.util.spec_from_file_location('diag', SCRIPT)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class DiagnosticTests(unittest.TestCase):
    def test_anchor_and_horizon_are_deterministic_and_bounded(self):
        self.assertEqual(diag.anchor_frames(8), list(range(8)))
        self.assertEqual(diag.anchor_frames(153), diag.anchor_frames(153))
        frames, valid = diag.horizon_indices(150, 153)
        self.assertEqual(frames[:3], [150, 151, 152])
        self.assertEqual(valid[:3], [True, True, True])
        self.assertEqual(sum(valid), 3)
        self.assertEqual(frames[-1], 152)
        with self.assertRaises(RuntimeError):
            diag.horizon_indices(153, 153)

    def test_common_rng_and_dimensions_are_declared(self):
        self.assertEqual(diag.SEEDS, (1729, 2718, 31415))
        self.assertEqual(len(diag.JOINTS), 12)
        self.assertEqual(set(diag.JOINTS), set(range(14)) - {6, 13})
        self.assertIn('Native compute_loss returns (1,50)', ' '.join(diag.LIMITATIONS))

    def test_manifest_validation_rejects_identity_or_anchor_mutation(self):
        manifest = {
            'schema': 1, 'status': 'prepared', 'original': str(diag.SOURCE),
            'ema': str(diag.PILOT / '1999'), 'seeds': list(diag.SEEDS), 'num_steps': 10,
            'anchors': [dict(anchor_id=str(i), frame=0, length=8,
                horizon_frames=diag.horizon_indices(0, 8)[0], valid=diag.horizon_indices(0, 8)[1])
                for i in range(160)], 'sha256': {},
        }
        with mock.patch.object(diag, 'identity'):
            diag.validate_manifest(manifest)
            for field, value in [('original', '/wrong/baseline'), ('ema', '/wrong/pilot')]:
                changed = copy.deepcopy(manifest)
                changed[field] = value
                with self.assertRaisesRegex(RuntimeError, 'identity'):
                    diag.validate_manifest(changed)
            changed = copy.deepcopy(manifest)
            changed['anchors'][0]['horizon_frames'][0] = 1
            with self.assertRaisesRegex(RuntimeError, 'boundary'):
                diag.validate_manifest(changed)

    def test_selection_is_order_independent(self):
        rows = [dict(split=s, task=t, seed=i) for s, n in [('train', 45), ('val', 5)]
                for t in ('hanging_mug', 'place_mouse_pad') for i in range(n)]
        expected = diag.select_episodes(rows)
        random.Random(999).shuffle(rows)
        self.assertEqual(diag.select_episodes(rows), expected)
        self.assertEqual(len(expected), 20)

    def test_identical_model_paired_deltas_are_zero(self):
        row = dict(split='val', task='hanging_mug', episode=0, native_loss=[1.] * 50,
            abs_error=[[2.] * 14] * 50, squared_error=[[4.] * 14] * 50,
            valid=[True] * 25 + [False] * 25, outside=[[False] * 14] * 50)
        summary = diag.summarize([row])
        paired = diag.paired_summary(summary, copy.deepcopy(summary))
        self.assertTrue(paired['episodes'][0]['ema_minus_original'])
        self.assertTrue(all(x == 0 for x in paired['episodes'][0]['ema_minus_original'].values()))
        self.assertEqual(summary[0]['metrics']['valid']['target_horizons'], 25)
        self.assertEqual(summary[0]['metrics']['all']['joints_rad']['rmse'], 2.)

    def test_prepare_is_exclusive_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory(dir=diag.ROOT / 'outputs') as d:
            path = Path(d)
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                diag.fresh_output(path)

    def test_output_is_bounded_and_json_disallows_nan(self):
        with tempfile.TemporaryDirectory(dir=diag.ROOT / 'outputs') as d:
            path = Path(d) / 'report.json'
            diag.write_json(path, {'status': 'passed', 'value': 1})
            self.assertEqual(json.loads(path.read_text())['status'], 'passed')
            with self.assertRaises(ValueError):
                diag.write_json(Path(d) / 'bad.json', {'x': float('nan')})

    def test_compute_loss_train_flag_is_static_under_module_jit(self):
        import os
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
        import sys
        sys.path.insert(0, str(diag.OPENPI / 'src'))
        import flax.nnx as nnx
        import jax.numpy as jnp
        from openpi.shared.nnx_utils import module_jit

        class BoolBranch(nnx.Module):
            def compute(self, *, train=False):
                return jnp.array(1 if train else 0)

        fn = module_jit(BoolBranch().compute, static_argnames=('train',))
        self.assertEqual(int(fn(train=False)), 0)
        self.assertEqual(int(fn(train=True)), 1)


if __name__ == '__main__':
    unittest.main()
