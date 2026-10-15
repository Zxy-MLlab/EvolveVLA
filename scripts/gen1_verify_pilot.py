#!/usr/bin/env python3
"""Independent CPU-only reload gate for the fixed 2000-update two-task pilot."""
import argparse
import dataclasses
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OPENPI = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi'
SOURCE = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify(a, audit):
    out = a.checkpoint_dir.resolve()
    require(out.is_relative_to(ROOT / 'checkpoints') and out != SOURCE and out != ROOT / 'checkpoints', 'Invalid output')
    require(a.exit_code_file.read_text().strip() == '0', 'Training did not exit successfully')
    provenance = json.loads(a.provenance_file.read_text())
    require(provenance['mode'] == 'train' and Path(provenance['checkpoint']).resolve() == out, 'Wrong run provenance')
    require(provenance['inputs']['status'] == 'passed', 'Input audits not passed')
    require(provenance['fresh_optimizer'] is True and provenance['full_parameter_training'] is True, 'Wrong initialization')
    for path, sha in provenance['sha256'].items():
        require(digest(path) == sha, f'Launch source changed: {path}')
    completed = json.loads((a.provenance_file.parent / 'completed.json').read_text())
    require(completed['status'] == 'completed' and completed['updates'] == 2000 and
            Path(completed['checkpoint']).resolve() == out / '1999', 'Missing completed run identity')
    audit['provenance_sha256'] = digest(a.provenance_file)
    text = a.training_log.read_text(errors='replace')
    log_hash = digest(a.training_log)
    require('Waiting for checkpoint manager to finish' in text, 'Missing checkpoint completion wait')
    require(str(out) in text, 'Training log does not identify this checkpoint')
    metrics = {}
    for m in re.finditer(r'\bStep\s+(\d+):\s*([^\r\n]*)', text):
        step = int(m.group(1))
        require(0 <= step < 2000, 'Unexpected training step')
        record = {}
        for key in ('loss', 'grad_norm', 'param_norm'):
            vals = re.findall(r'\b' + key + r'=([^,\s]+)', m.group(2))
            require(len(vals) == 1 and math.isfinite(float(vals[0])), f'Invalid {key} at {step}')
            record[key] = float(vals[0])
        require(step not in metrics or metrics[step] == record, 'Conflicting step log')
        metrics[step] = record
    require(set(range(0, 2000, 20)) | {1999} == set(metrics), 'Missing expected logged metrics')
    final = out / '1999'
    require(all((final / k).is_dir() for k in ('params', 'train_state', 'assets')), 'Incomplete checkpoint')
    require(not any('orbax-checkpoint-tmp' in p.name for p in out.iterdir()), 'Unfinished temporary checkpoint')
    require(sorted(p.name for p in out.iterdir() if p.is_dir() and p.name.isdigit()) == ['1999'], 'Unexpected intermediate checkpoints')
    audit.update(training_log_sha256=log_hash, exit_code_sha256=digest(a.exit_code_file), metrics=metrics)
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_ENABLE_COMPILATION_CACHE='false',
                      HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      WANDB_MODE='disabled', OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache')
    sys.path[:0] = [str(OPENPI / 'src'), str(OPENPI / 'packages/openpi-client/src')]
    import flax.nnx as nnx
    from flax import traverse_util
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from openpi.models import model
    from openpi.shared import array_typing as at, normalize
    from openpi.training import checkpoints, config, optimizer, sharding, weight_loaders
    require(all(d.platform == 'cpu' for d in jax.devices()), 'CPU-only verification required')
    cfg = config.get_config('pi05_base_aloha_full_sim_arx-x5_seed_0')
    cfg = dataclasses.replace(cfg, exp_name=out.name,
        model=dataclasses.replace(cfg.model, action_dim=32, action_horizon=50, pi05=True),
        data=dataclasses.replace(cfg.data, repo_id=a.repo_id, adapt_to_pi=False, use_delta_joint_actions=True,
            assets=config.AssetsConfig(assets_dir=str(SOURCE / 'assets'), asset_id='aloha_agilex_sim'),
            base_config=dataclasses.replace(cfg.data.base_config, video_backend='pyav')),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(SOURCE / 'params')),
        checkpoint_dir_override=str(out), batch_size=1, num_workers=0, fsdp_devices=1,
        num_train_steps=2000, log_interval=20, save_interval=2000, seed=0, ema_decay=0.99,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-6, decay_steps=2000, decay_lr=5e-7),
        resume=False, overwrite=False, wandb_enabled=False)
    cfg = dataclasses.replace(cfg, freeze_filter=nnx.Nothing(), keep_period=None)
    expected_config = json.loads(json.dumps(dataclasses.asdict(cfg), default=str))
    require(provenance['config'] == expected_config, 'Launch configuration differs from fixed verifier configuration')
    audit['reconstructed_config'] = repr(cfg)

    def finite(tree):
        leaves = jax.tree.leaves(tree)
        require(bool(leaves), 'Empty tree')
        count = 0
        for leaf in leaves:
            require(not isinstance(leaf, jax.ShapeDtypeStruct), 'Unrestored leaf')
            flat = np.asarray(leaf).reshape(-1)
            for start in range(0, flat.size, 1_000_000):
                require(bool(np.isfinite(flat[start:start+1_000_000]).all()), 'Nonfinite checkpoint tensor')
            count += flat.size
        return {'finite': True, 'leaves': len(leaves), 'elements': int(count)}

    def tensor_hashes(tree):
        return {k: hashlib.sha256(np.asarray(v).view(np.uint8)).hexdigest()
                for k, v in traverse_util.flatten_dict(tree, sep='/').items()}

    reference = nnx.state(nnx.eval_shape(cfg.model.create, jax.random.key(0))).to_pure_dict()
    source = model.restore_params(SOURCE / 'params', restore_type=np.ndarray)
    at.check_pytree_equality(expected=reference, got=source, check_shapes=True, check_dtypes=True)
    source_hashes = tensor_hashes(source)
    audit['source_raw'] = finite(source)
    del source
    gc.collect()
    params = model.restore_params(final / 'params', restore_type=np.ndarray)
    at.check_pytree_equality(expected=reference, got=params, check_shapes=True, check_dtypes=True)
    changed = [k for k, h in tensor_hashes(params).items() if h != source_hashes[k]]
    require(bool(changed), 'EMA checkpoint is byte-identical to original weights')
    audit['inference_params'] = dict(finite(params), changed_leaves=len(changed), kind='ema', exact_key_shape_dtype_match=True)
    loaded = cfg.model.load(params, remove_extra_params=False)
    audit['inference_params']['strict_model_load'] = True
    del loaded, params
    gc.collect()
    stats = normalize.load(final / 'assets/aloha_agilex_sim')
    original = normalize.load(SOURCE / 'assets/aloha_agilex_sim')
    require(set(stats) == set(original), 'Normalization keys changed')
    for k, v in stats.items():
        for field in ('mean', 'std', 'q01', 'q99'):
            x, y = getattr(v, field), getattr(original[k], field)
            require(x is not None and y is not None and np.array_equal(x, y), 'Normalization changed')
            finite(x)
    audit['normalization_exact'] = True
    spec = importlib.util.spec_from_file_location('pilot_verify_pinned_train', OPENPI / 'scripts/train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    _, init_rng = jax.random.split(jax.random.key(0))
    template, _ = train.init_train_state(cfg, init_rng, sharding.make_mesh(1), resume=True)
    with at.disable_typechecking():
        state_template, params_template = checkpoints._split_params(template)

    def restore(path, template):
        with ocp.PyTreeCheckpointer() as c:
            return c.restore(path, args=ocp.args.PyTreeRestore(item=template,
                restore_args=jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), template)))

    state = restore(final / 'train_state', state_template)
    saved = restore(final / 'params', {'params': params_template})
    state = checkpoints._merge_params(state, saved)
    at.check_pytree_equality(expected=template, got=state, check_shapes=True, check_dtypes=True)
    require(int(np.asarray(state.step)) == 2000, 'Wrong TrainState.step')
    audit['full_train_state'] = dict(internal_step=2000, checkpoint_index=1999,
        exact_template_match=True, live_params=finite(state.params), optimizer=finite(state.opt_state), ema_params=finite(state.ema_params))
    counts = []
    def check_adam(value):
        if all(hasattr(value, k) for k in ('count', 'mu', 'nu')):
            count = int(np.asarray(value.count))
            require(count == 2000, 'Wrong Adam count')
            counts.append(count)
        elif isinstance(value, (tuple, list)):
            for v in value:
                check_adam(v)
        elif isinstance(value, dict):
            for v in value.values():
                check_adam(v)
    check_adam(state.opt_state)
    require(bool(counts), 'Missing Adam state')
    audit['adam_counts'] = counts
    require(digest(a.training_log) == log_hash, 'Training log changed during verification')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--training-log', type=Path, required=True)
    p.add_argument('--exit-code-file', type=Path, required=True)
    p.add_argument('--repo-id', required=True)
    p.add_argument('--provenance-file', type=Path, required=True)
    a = p.parse_args()
    require(a.checkpoint_dir.is_dir() and a.checkpoint_dir.resolve().parent == ROOT / 'checkpoints'
            and a.checkpoint_dir.resolve() != SOURCE, 'Invalid checkpoint directory')
    audit = {'status': 'failed', 'checkpoint_dir': str(a.checkpoint_dir.resolve()), 'started_at': datetime.now(timezone.utc).isoformat(),
             'limitations': ['CPU restoration does not itself measure rollout success.', 'Weights changed is not evidence of improvement.']}
    output = a.checkpoint_dir / ('pilot_verification_' + uuid.uuid4().hex + '.json')
    try:
        verify(a, audit)
        audit['status'] = 'passed'
    except BaseException as exc:
        audit['error'] = repr(exc)
        raise
    finally:
        with output.open('x') as f:
            json.dump(audit, f, indent=2, allow_nan=False)
        print(json.dumps({'status': audit['status'], 'audit': str(output)}), flush=True)


if __name__ == '__main__':
    main()
