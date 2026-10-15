"""Synthetic CPU-only Orbax cap/layout test; keeps all evidence, never deletes."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                  JAX_ENABLE_COMPILATION_CACHE='false', HF_HUB_OFFLINE='1',
                  HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import tempfile
import json
import traceback
import dataclasses
import importlib.metadata

ROOT = Path('/root/autodl-tmp/EvolveVLA')
sys.path.insert(0, str(ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi/src'))


def main():
    evidence = Path(tempfile.mkdtemp(prefix='orbax_cap_synthetic_', dir=ROOT / 'tmp/gen1_training'))
    report = {'synthetic': True, 'status': 'failed', 'evidence_dir': str(evidence),
              'save_concurrent_gb_per_handler': 8,
              'limitations': 'Tiny CPU compatibility test only; separate write budgets do not bound full-model D2H staging or prove 110 GiB fit.'}
    print(str(evidence), flush=True)
    try:
        import jax
        import numpy as np
        import orbax.checkpoint as ocp
        import optax
        from flax import nnx
        from openpi.training import checkpoints as ck
        from openpi.training import utils
        from openpi.shared import normalize

        assert all(d.platform == 'cpu' for d in jax.devices())
        report['devices'] = [str(d) for d in jax.devices()]
        report['orbax_version'] = importlib.metadata.version('orbax-checkpoint')
        assert report['orbax_version'] == '0.11.13'
        model = nnx.Linear(3, 2, rngs=nnx.Rngs(0))
        graph, params = nnx.split(model)
        tx = optax.adamw(1e-3)
        ema = jax.tree.map(lambda x: x + np.float32(0.125), params)
        state = utils.TrainState(step=jax.numpy.asarray(2, dtype='int32'), params=params,
                                 model_def=graph, opt_state=tx.init(params), tx=tx,
                                 ema_decay=0.99, ema_params=ema)
        split_state, inference = ck._split_params(state)
        items = {'train_state': split_state, 'params': {'params': inference}}
        stats = {name: normalize.NormStats(mean=np.zeros(3, dtype='float32'),
                    std=np.ones(3, dtype='float32'), q01=-np.ones(3, dtype='float32'),
                    q99=np.ones(3, dtype='float32')) for name in ('state', 'actions')}
        def save_assets(directory):
            normalize.save(directory / 'synthetic_assets', stats)
        checkpoint = evidence / 'checkpoint'
        checkpoint.mkdir()
        options = ocp.CheckpointManagerOptions(max_to_keep=1, keep_period=5000,
                    create=False, async_options=ocp.AsyncOptions(timeout_secs=7200))
        with ocp.CheckpointManager(checkpoint, item_handlers={
                'assets': ck.CallbackHandler(),
                'train_state': ocp.PyTreeCheckpointHandler(save_concurrent_gb=8),
                'params': ocp.PyTreeCheckpointHandler(save_concurrent_gb=8)},
                options=options) as manager:
            assert manager.save(1, dict(items, assets=save_assets))
            manager.wait_until_finished()
            assert manager.latest_step() == 1
            report['async_save_wait_passed'] = True
        step = checkpoint / '1'
        assert all((step / name).is_dir() for name in ('params', 'train_state', 'assets'))
        assert not any('orbax-checkpoint-tmp' in p.name for p in checkpoint.rglob('*'))
        def restore(name, template):
            with ocp.PyTreeCheckpointer() as reader:
                return reader.restore(step / name, args=ocp.args.PyTreeRestore(
                    item=template, restore_args=jax.tree.map(
                        lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), template)))
        def equal(expected, actual):
            assert jax.tree.structure(expected) == jax.tree.structure(actual)
            for x, y in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
                assert isinstance(y, np.ndarray), type(y)
                assert x.shape == y.shape and x.dtype == y.dtype
                np.testing.assert_array_equal(np.asarray(x), y)
        inference_restored = restore('params', items['params'])
        state_restored = restore('train_state', split_state)
        equal(items['params'], inference_restored)
        equal(split_state, state_restored)
        merged = ck._merge_params(state_restored, inference_restored)
        equal(state, merged)
        assert int(merged.step) == 2
        saved_stats = normalize.load(step / 'assets/synthetic_assets')
        assert saved_stats.keys() == stats.keys()
        for key in stats:
            for field in dataclasses.fields(stats[key]):
                np.testing.assert_array_equal(getattr(stats[key], field.name),
                                              getattr(saved_stats[key], field.name))
        report.update(status='passed', numpy_exact_roundtrip=True,
                      full_state_split_merge=True, normalization_exact=True,
                      synthetic_adam_count=0, internal_step=2,
                      array_leaves=len(jax.tree.leaves(state)),
                      logical_array_bytes=sum(x.size*x.dtype.itemsize for x in jax.tree.leaves(state)),
                      layout=sorted(str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file()))
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
    with (evidence / 'result.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
