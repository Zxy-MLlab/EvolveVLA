#!/usr/bin/env python3
"""Verify a finished two-step gen1 smoke on CPU; write one fresh JSON audit.

Run only AFTER the training process exits successfully. This does not run inference,
resume training, initialize a checkpoint manager, or change saved checkpoint items.
Raw source dtype differences are reported, not hidden: the pinned loader casts them.
"""
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
LAUNCHER = ROOT / 'scripts/gen1_train_ready.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_log(text, output):
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    decoder = json.JSONDecoder()
    manifests = []
    for match in re.finditer(r'\{', text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get('mode') == 'train' and 'checkpoint' in obj:
            manifests.append(obj)
    require(len(manifests) == 1, 'Need exactly one launcher train manifest in log')
    manifest = manifests[0]
    require(Path(manifest['checkpoint']).resolve() == output, 'Log checkpoint does not match target')
    require(manifest.get('resume') is False and manifest.get('overwrite') is False,
            'Log must show resume=False and overwrite=False')
    require(manifest.get('batch_size') == 1, 'Log batch size must be 1')
    repo_match = re.search(r"\brepo_id='([^']+)'", manifest.get('data', ''))
    require(repo_match is not None, 'Cannot recover dataset repo_id from launcher manifest')
    steps = {}
    for match in re.finditer(r'\bStep\s+(\d+):\s*([^\r\n]*)', text):
        step = int(match.group(1))
        require(step in (0, 1), f'Unexpected training step {step}')
        metrics = {}
        for name in ('loss', 'grad_norm', 'param_norm'):
            values = re.findall(r'\b' + name + r'=([^,\s]+)', match.group(2))
            require(len(values) == 1, f'Step {step}: missing/duplicate {name}')
            value = float(values[0])
            require(math.isfinite(value), f'Step {step}: nonfinite {name}')
            metrics[name] = value
        steps.setdefault(str(step), []).append(metrics)
    require(set(steps) == {'0', '1'}, 'Need finite metrics for both Step 0 and Step 1')
    # Some progress loggers redraw records. All repetitions must agree and be finite.
    for step, records in steps.items():
        require(all(record == records[0] for record in records), f'Conflicting Step {step} records')
    require('Waiting for checkpoint manager to finish' in text, 'Missing checkpoint-wait log marker')
    return manifest, repo_match.group(1), steps


def build_config(config, weight_loaders, output, repo_id):
    cfg = config.get_config('pi05_base_aloha_full_sim_arx-x5_seed_0')
    return dataclasses.replace(
        cfg, exp_name=output.name,
        model=dataclasses.replace(cfg.model, action_dim=32, action_horizon=50, pi05=True),
        data=dataclasses.replace(
            cfg.data, repo_id=repo_id, adapt_to_pi=False, use_delta_joint_actions=True,
            assets=config.AssetsConfig(assets_dir=str(SOURCE / 'assets'), asset_id='aloha_agilex_sim'),
            base_config=dataclasses.replace(cfg.data.base_config, video_backend='pyav')),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(SOURCE / 'params')),
        checkpoint_dir_override=str(output), batch_size=1, num_workers=0,
        fsdp_devices=1, num_train_steps=2, log_interval=1, save_interval=1,
        resume=False, overwrite=False, wandb_enabled=False)


def verify(output, log_path, audit):
    # These settings affect this fresh verification process only, never environment files.
    require('jax' not in sys.modules, 'Run verifier in a fresh process before importing JAX')
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_ENABLE_COMPILATION_CACHE='false',
                      HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      WANDB_MODE='disabled', OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache')
    sys.dont_write_bytecode = True
    raw_log = log_path.read_bytes()
    audit['training_log_sha256'] = hashlib.sha256(raw_log).hexdigest()
    manifest, repo_id, metrics = parse_log(raw_log.decode('utf-8', errors='replace'), output)
    audit.update(training_manifest=manifest, repo_id=repo_id, metrics=metrics)
    del raw_log
    step_dir = output / '1'
    for item in ('params', 'train_state', 'assets'):
        require((step_dir / item).is_dir(), f'Missing final checkpoint item: {item}')
    require(not any('orbax-checkpoint-tmp' in p.name for p in output.iterdir()),
            'Temporary Orbax checkpoint present; wait for training to finish')
    sys.path[:0] = [str(OPENPI / 'src'), str(OPENPI / 'packages/openpi-client/src')]
    import flax.nnx as nnx
    from flax import traverse_util
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from openpi.models import model
    from openpi.shared import array_typing as at, normalize
    from openpi.training import checkpoints, config, sharding, weight_loaders

    require(all(device.platform == 'cpu' for device in jax.devices()), 'Non-CPU JAX device')
    audit['devices'] = [str(device) for device in jax.devices()]
    cfg = build_config(config, weight_loaders, output, repo_id)
    for key, expected in (('model', str(cfg.model)), ('data', str(cfg.data)),
                          ('weight_loader', str(cfg.weight_loader))):
        require(manifest.get(key) == expected, f'Launcher manifest {key} differs from reconstructed config')
    audit['config'] = {'repr': repr(cfg), 'ema_decay': cfg.ema_decay,
                       'launcher_sha256': hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()}

    def finite(tree):
        leaves = jax.tree.leaves(tree)
        require(bool(leaves), 'Empty array tree')
        elements = 0
        for i, leaf in enumerate(leaves):
            require(not isinstance(leaf, jax.ShapeDtypeStruct), f'Unrestored placeholder leaf {i}')
            arr = np.asarray(leaf)
            # Chunk checks avoid a huge temporary boolean array for embedding tensors.
            flat = arr.reshape(-1)
            for start in range(0, flat.size, 1_000_000):
                require(bool(np.isfinite(flat[start:start + 1_000_000]).all()), f'Nonfinite leaf {i}')
            elements += arr.size
        return {'finite': True, 'leaves': len(leaves), 'elements': int(elements)}

    def compare(reference, actual):
        ref = traverse_util.flatten_dict(reference, sep='/')
        got = traverse_util.flatten_dict(actual, sep='/')
        missing, extra = sorted(ref.keys() - got.keys()), sorted(got.keys() - ref.keys())
        shape_differences, dtype_differences = [], []
        for key in sorted(ref.keys() & got.keys()):
            if tuple(ref[key].shape) != tuple(got[key].shape):
                shape_differences.append({'key': key, 'expected': list(ref[key].shape),
                                          'actual': list(got[key].shape)})
            if np.dtype(ref[key].dtype) != np.dtype(got[key].dtype):
                dtype_differences.append({'key': key, 'expected': str(ref[key].dtype),
                                          'actual': str(got[key].dtype)})
        return {'missing': missing, 'extra': extra, 'shape_differences': shape_differences,
                'dtype_differences': dtype_differences, 'exact_keys': not missing and not extra,
                'exact_shapes': not shape_differences, 'exact_dtypes': not dtype_differences}

    reference = nnx.state(nnx.eval_shape(cfg.model.create, jax.random.key(0))).to_pure_dict()
    source = model.restore_params(SOURCE / 'params', restore_type=np.ndarray)
    source_check = compare(reference, source)
    audit['source_raw'] = source_check
    source_check['finite'] = finite(source)
    source_check['strict_raw_key_shape_dtype_match'] = all(
        source_check[k] for k in ('exact_keys', 'exact_shapes', 'exact_dtypes'))
    source_check['dtype_policy'] = (
        'Differences are explicitly recorded, not treated as raw dtype equality. '
        'Pinned CheckpointWeightLoader casts matching source leaves to reference dtype.')
    require(source_check['exact_keys'] and source_check['exact_shapes'],
            'Raw source keys/shapes incompatible; missing parameters will NOT be tolerated')
    # Also test finiteness after the exact dtype casts performed by the pinned loader.
    ref_flat = traverse_util.flatten_dict(reference, sep='/')
    for key, leaf in traverse_util.flatten_dict(source, sep='/').items():
        if np.dtype(leaf.dtype) != np.dtype(ref_flat[key].dtype):
            finite(leaf.astype(ref_flat[key].dtype))
    source_check['compatible_with_pinned_loader_casts'] = True
    del source
    gc.collect()

    params = model.restore_params(step_dir / 'params', restore_type=np.ndarray)
    audit['inference_params'] = compare(reference, params)
    audit['inference_params']['finite'] = finite(params)
    loaded_model = cfg.model.load(params, remove_extra_params=False)
    audit['inference_params']['strict_model_load'] = True
    audit['inference_params']['kind'] = 'ema' if cfg.ema_decay is not None else 'live'
    del loaded_model, params
    gc.collect()

    stats = normalize.load(step_dir / 'assets/aloha_agilex_sim')
    source_stats = normalize.load(SOURCE / 'assets/aloha_agilex_sim')
    require(set(stats) == set(source_stats) and {'state', 'actions'} <= set(stats),
            'Normalization asset keys differ or lack state/actions')
    audit['normalization'] = {}
    for key, values in stats.items():
        fields = {}
        for field in ('mean', 'std', 'q01', 'q99'):
            value, original = getattr(values, field), getattr(source_stats[key], field)
            require(value is not None and original is not None, f'Missing normalization {key}.{field}')
            fields[field] = finite(value)
            require(np.array_equal(value, original), f'Normalization differs from source: {key}.{field}')
        audit['normalization'][key] = fields

    # Reuse pinned initialization ONLY for its abstract shape branch. No source reload,
    # random materialized params, training updates, or optimizer initialization buffers.
    spec = importlib.util.spec_from_file_location('gen1_verify_pinned_train', OPENPI / 'scripts/train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    _, init_rng = jax.random.split(jax.random.key(cfg.seed))
    template, _ = train.init_train_state(cfg, init_rng, sharding.make_mesh(cfg.fsdp_devices), resume=True)
    # Like pinned restore_state, allow abstract leaves while splitting the template.
    with at.disable_typechecking():
        state_template, param_template = checkpoints._split_params(template)

    def restore_item(path, item):
        # Explicit NumPy restore args prevent reuse of saved GPU sharding metadata.
        with ocp.PyTreeCheckpointer() as checkpointer:
            return checkpointer.restore(path, args=ocp.args.PyTreeRestore(
                item=item, restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item)))

    state = restore_item(step_dir / 'train_state', state_template)
    saved_params = restore_item(step_dir / 'params', {'params': param_template})
    state = checkpoints._merge_params(state, saved_params)
    at.check_pytree_equality(expected=template, got=state, check_shapes=True, check_dtypes=True)
    require(int(np.asarray(state.step)) == 2, 'Restored TrainState.step must equal 2')
    full = {'internal_step': 2, 'checkpoint_index': 1, 'exact_template_match': True,
            'live_params': finite(state.params), 'optimizer': finite(state.opt_state)}
    audit['full_train_state'] = full
    if state.ema_params is not None:
        full['ema_params'] = finite(state.ema_params)
    moments = []

    def find_adam(value):
        if all(hasattr(value, attr) for attr in ('mu', 'nu', 'count')):
            count = int(np.asarray(value.count))
            require(count == 2, f'Adam update count is {count}, expected 2')
            moments.append({'count': count, 'mu': finite(value.mu), 'nu': finite(value.nu)})
        elif isinstance(value, (tuple, list)):
            for child in value:
                find_adam(child)
        elif isinstance(value, dict):
            for child in value.values():
                find_adam(child)
    find_adam(state.opt_state)
    require(bool(moments), 'No restored Adam moment state found')
    full['adam_moments'] = moments
    require(hashlib.sha256(log_path.read_bytes()).hexdigest() == audit['training_log_sha256'],
            'Training log changed during verification; rerun after training exits')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--training-log', type=Path, required=True)
    args = parser.parse_args()
    output = args.checkpoint_dir.resolve()
    require(output.is_dir() and output.is_relative_to(ROOT / 'checkpoints'),
            'Checkpoint directory must exist under EvolveVLA/checkpoints')
    require(output != SOURCE and output != ROOT / 'checkpoints', 'Refusing source/root directory')
    audit_path = output / ('verification_audit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                           + '_' + uuid.uuid4().hex[:12] + '.json')
    audit = {'status': 'failed', 'checkpoint_dir': str(output),
             'training_log': str(args.training_log.resolve()), 'source_checkpoint': str(SOURCE),
             'started_at': datetime.now(timezone.utc).isoformat(),
             'limitations': [
                 'Run after successful training process exit; log wait marker alone does not prove exit.',
                 'Log metrics are rounded to four decimals; this does not execute inference.',
                 'Fresh optimizer is inferred from launcher non-resume path and final counters, not captured before update.',
                 'Raw source dtype differences are reported separately from loader-compatible casts.',
                 'Reconstructed config is checked against launcher model/data/weight-loader manifest; '
                 'the log does not serialize every optimizer hyperparameter.']}
    code = 1
    # Exclusive creation never overwrites an earlier audit; only this new file is written.
    with audit_path.open('x', encoding='utf-8') as handle:
        try:
            verify(output, args.training_log.resolve(), audit)
            audit['status'] = 'passed'
            code = 0
        except Exception as exc:
            audit['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            audit['finished_at'] = datetime.now(timezone.utc).isoformat()
            json.dump(audit, handle, indent=2, allow_nan=False)
            handle.write('\n')
    print(f"Verification {audit['status']}: {audit_path}", file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
