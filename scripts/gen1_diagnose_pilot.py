#!/usr/bin/env python3
"""Frozen, read-only pilot diagnostics. prepare is CPU-only; smoke/evaluate require GPU opt-in.

No training, optimizer restore, raw checkpoint extraction, clipping, or dataset writes.
Native Pi0.compute_loss reduces action dimensions: only [batch,horizon] is reported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OPENPI = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi'
SOURCE = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
PILOT = ROOT / 'checkpoints/gen1_pilot_2000_20261010'
PROVENANCE = ROOT / 'outputs/gen1_pilot_2000_20261010/run_config.json'
VERIFY = PILOT / 'pilot_verification_1d09951d190546c68007551236c135d8.json'
SEEDS = (1729, 2718, 31415)
GIB = 2**30
LIMIT = 2 * GIB
MIN_FREE = 70 * GIB
JOINTS = [i for i in range(14) if i not in (6, 13)]
LIMITATIONS = [
    'Offline expert-action errors are proxies, not rollout success.',
    'Checkpoint parameters are strictly validated in native dtype then cast to bfloat16, matching production policy_config; runtime observation/prompt parity remains unproven.',
    'Native compute_loss returns (1,50), already averaged over all 32 action dimensions; no 14/18-dimensional native loss is available.',
    'Native evaluation uses train=False (no training augmentation); it is not the augmented online training-loss stream.',
    'Sampled physical metrics cover 14 dimensions; sampled normalized residuals for 18 unused dimensions are separate, not a native loss decomposition.',
    'Raw TrainState weights intentionally omitted: no optimizer restore or unverified selective restore.',
    'Within-episode anchors and RNG draws are dependent; summary units are episodes, without significance claims.',
    'Training/policy parity uses the same converted observation; it does not prove raw simulator/video pixel equality or runtime prompt equality.',
    'AV1 CRF18 conversion preserves frames, not pixels. No RGB swap, clipping, angle wrapping, or normalization change is applied.',
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    encoded = json.dumps(data, indent=2, allow_nan=False) + '\n'
    require(len(encoded.encode()) < 64 * 2**20, 'Oversized JSON output')
    with Path(path).open('x') as f:
        f.write(encoded)


def guard(out, pending=0):
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) if out.exists() else 0
    require(size + pending <= LIMIT, 'Diagnostic output exceeds 2 GiB')
    require(shutil.disk_usage(out if out.exists() else out.parent).free - pending >= MIN_FREE,
            'Diagnostic requires at least 70 GiB free after planned write')


def fresh_output(path):
    require(path.parent.resolve() == ROOT / 'outputs', 'Output must be direct child of established outputs root')
    require(not path.exists() and not path.is_symlink(), 'Output already exists')
    guard(path, 64 * 2**20)
    path.mkdir(exist_ok=False)
    return path


def anchor_frames(length):
    require(length >= 8, 'Need eight distinct anchors')
    return [(i * (length - 1)) // 7 for i in range(8)]


def horizon_indices(frame, length):
    require(0 <= frame < length, 'Anchor outside episode')
    return [min(frame + h, length - 1) for h in range(50)], [frame + h < length for h in range(50)]


def select_episodes(rows):
    picked = []
    for split in ('val', 'train'):
        for task in sorted(('hanging_mug', 'place_mouse_pad')):
            group = sorted((r for r in rows if r['split'] == split and r['task'] == task),
                           key=lambda r: hashlib.sha256(f"gen1-diagnostic-panel-0:{task}:{r['seed']}".encode()).hexdigest())
            require(len(group) == (5 if split == 'val' else 45), 'Wrong split size')
            picked.extend(group[:5])
    return picked


def defaults():
    return argparse.Namespace(mode='loader', repo_id='gen1_pilot_train_lossless_20261010',
        val_repo_id='gen1_pilot_val_20261010', lerobot_home=ROOT / 'data',
        split_manifest=ROOT / 'data/gen1_pilot_two_task_20261010/split_manifest.json',
        train_audit=ROOT / 'data/gen1_pilot_train_lossless_20261010/conversion_audit.json',
        val_audit=ROOT / 'data/gen1_pilot_val_20261010/conversion_audit.json', checkpoint_dir=PILOT)


def helpers():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'scripts'))
    import gen1_train_pilot as trainer
    import gen1_verify_pilot as verifier
    require(trainer.SOURCE == SOURCE == verifier.SOURCE, 'Wrong baseline helper identity')
    return trainer, verifier


def identity():
    trainer, verifier = helpers()
    evidence = read_json(VERIFY)
    provenance = read_json(PROVENANCE)
    require(evidence['status'] == 'passed' and Path(evidence['checkpoint_dir']).resolve() == PILOT,
            'Wrong or unsuccessful pilot verification')
    require(evidence['provenance_sha256'] == verifier.digest(PROVENANCE), 'Pilot provenance changed')
    require(provenance['checkpoint'] == str(PILOT) and provenance['mode'] == 'train', 'Wrong pilot identity')
    require(evidence['inference_params']['kind'] == 'ema' and evidence['normalization_exact'] is True
            and evidence['full_train_state']['internal_step'] == 2000, 'EMA/step/normalization not verified')
    for path, sha in provenance['sha256'].items():
        verifier.require(verifier.digest(path) == sha, f'Launch source changed: {path}')
    require(provenance['config']['weight_loader']['params_path'] == str(SOURCE / 'params'), 'Not Myungkyu original')
    require(read_json(PROVENANCE.parent / 'completed.json')['checkpoint'] == str(PILOT / '1999'), 'Wrong final checkpoint')
    return trainer.validate_inputs(defaults())


def files_under(path):
    return sorted(p for p in Path(path).rglob('*') if p.is_file())


def source_vectors(path, kind):
    import h5py
    import numpy as np
    with h5py.File(path) as f:
        return np.concatenate([f[f'{kind}/{k}'][:] for k in (
            'left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states')], axis=1).astype(np.float32)


def prepare(out):
    import numpy as np
    import pandas as pd
    inputs = identity()
    args = defaults()
    split = read_json(args.split_manifest)
    rows = select_episodes(split['episodes'])
    anchors, hashes = [], {}
    tracked = [Path(__file__), ROOT / 'scripts/gen1_train_pilot.py', ROOT / 'scripts/gen1_verify_pilot.py',
               args.split_manifest, VERIFY, PROVENANCE, PROVENANCE.parent / 'completed.json']
    tracked += files_under(OPENPI / 'src/openpi')
    tracked = [p for p in tracked if '__pycache__' not in p.parts]
    for base in (SOURCE / 'params', PILOT / '1999/params', SOURCE / 'assets', PILOT / '1999/assets'):
        tracked += files_under(base)
    for split_name, repo in [('train', args.repo_id), ('val', args.val_repo_id)]:
        base = args.lerobot_home / repo
        tracked += files_under(base)
        audit = read_json(base / 'conversion_audit.json')
        table = pd.concat([pd.read_parquet(p) for p in sorted((base / 'data').rglob('*.parquet'))], ignore_index=True)
        tasks = pd.read_parquet(base / 'meta/tasks.parquet')
        prompts = {int(row['task_index']): str(name) for name, row in tasks.iterrows()}
        mapping = {str(Path(s['path']).resolve()): i for i, s in enumerate(audit['sources'])}
        for row in (r for r in rows if r['split'] == split_name):
            ep = mapping[str(Path(row['path']).resolve())]
            frames = table[table.episode_index == ep].sort_values('frame_index')
            require(len(frames) == row['frames'] and frames.frame_index.tolist() == list(range(row['frames'])), 'Episode boundary mismatch')
            state, action = np.stack(frames['observation.state']), np.stack(frames.action)
            require(digest(row['path']) == row['sha256'], 'Source episode changed')
            require(np.array_equal(state, source_vectors(row['path'], 'state')) and
                    np.array_equal(action, source_vectors(row['path'], 'action')), 'Audit-order mapping/source vectors mismatch')
            tracked.append(Path(row['path']))
            require(frames.task_index.nunique() == 1, 'Prompt changed within episode')
            for frame in anchor_frames(len(frames)):
                indices, valid = horizon_indices(frame, len(frames))
                anchors.append(dict(anchor_id=f'{split_name}:{ep}:{frame}', split=split_name, task=row['task'],
                    seed=row['seed'], episode=ep, frame=frame, length=len(frames),
                    index=int(frames.iloc[frame]['index']), repo_id=repo, prompt=prompts[int(frames.iloc[frame].task_index)],
                    source=row['path'], source_sha256=row['sha256'], horizon_frames=indices, valid=valid))
    require(len(anchors) == 160, 'Panel must have 160 anchors')
    for p in sorted(set(tracked)):
        hashes[str(p)] = digest(p)
    guard(out, 64 * 2**20)
    manifest = dict(schema=1, status='prepared', original=str(SOURCE), ema=str(PILOT / '1999'),
        inputs=inputs, selection='SHA256 gen1-diagnostic-panel-0:task:seed; 5 episodes/task/split; floor-spaced 8 anchors',
        seeds=list(SEEDS), num_steps=10, anchors=anchors, sha256=hashes, limitations=LIMITATIONS)
    write_json(out / 'manifest.json', manifest)
    return manifest


def validate_manifest(manifest):
    require(manifest.get('schema') == 1 and manifest.get('status') == 'prepared', 'Invalid diagnostic manifest')
    require(manifest['original'] == str(SOURCE) and manifest['ema'] == str(PILOT / '1999'), 'Wrong baseline/checkpoint identity')
    require(manifest['seeds'] == list(SEEDS) and manifest['num_steps'] == 10, 'Changed common RNG/sampler')
    require(len(manifest['anchors']) == 160 and len({a['anchor_id'] for a in manifest['anchors']}) == 160, 'Invalid panel')
    for a in manifest['anchors']:
        indices, valid = horizon_indices(a['frame'], a['length'])
        require(indices == a['horizon_frames'] and valid == a['valid'], 'Changed episode boundary')
    for path, sha in manifest['sha256'].items():
        require(digest(path) == sha, f'Frozen input changed: {path}')
    identity()


def environment(gpu):
    require('jax' not in sys.modules, 'Use fresh process before JAX configuration')
    os.environ.pop('XLA_PYTHON_CLIENT_MEM_FRACTION', None)
    os.environ.update(HF_LEROBOT_HOME=str(ROOT / 'data'), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',
        JAX_ENABLE_COMPILATION_CACHE='false', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        JAX_PLATFORMS='cuda' if gpu else 'cpu', CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '0') if gpu else '')
    sys.path[:0] = [str(OPENPI / 'src'), str(OPENPI / 'packages/openpi-client/src')]


def finite(value):
    import numpy as np
    array = np.asarray(value)
    require(bool(np.isfinite(array).all()), 'Nonfinite tensor or metric')
    return array


def dataset_bundle(cfg, repo):
    from openpi.training import data_loader
    dc = dataclasses.replace(cfg.data, repo_id=repo).create(cfg.assets_dirs, cfg.model)
    raw = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    return dc, raw, data_loader.transform_dataset(raw, dc)


def transforms_for(dc):
    from openpi import transforms as t
    inputs = [*dc.repack_transforms.inputs, t.InjectDefaultPrompt(None), *dc.data_transforms.inputs,
              t.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm), *dc.model_transforms.inputs]
    outputs = [*dc.model_transforms.outputs, t.Unnormalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
               *dc.data_transforms.outputs, *dc.repack_transforms.outputs]
    return t.compose(inputs), t.compose(outputs), inputs, outputs


def checked_anchor(bundle, a):
    import jax
    import numpy as np
    from openpi import transforms as t
    dc, raw_dataset, transformed = bundle
    raw = raw_dataset[a['index']]
    require(int(raw['episode_index']) == a['episode'] and int(raw['frame_index']) == a['frame'], 'Actual loader index mismatch')
    require(str(raw['prompt']) == a['prompt'], 'Actual loader prompt differs')
    expected = source_vectors(a['source'], 'action')[a['horizon_frames']]
    require(np.array_equal(np.asarray(raw['action']), expected), 'Future actions cross boundary or do not match source')
    pad = np.asarray(raw['action_is_pad'], dtype=bool)
    require(np.array_equal(pad, ~np.asarray(a['valid'])), 'Actual loader padding differs')
    trained = transformed[a['index']]
    input_transform, output_transform, _, _ = transforms_for(dc)
    inferred = input_transform(copy.deepcopy(raw))
    train_leaves, train_tree = jax.tree.flatten(trained)
    infer_leaves, infer_tree = jax.tree.flatten(inferred)
    require(train_tree == infer_tree, 'Training/policy tree mismatch')
    for x, y in zip(train_leaves, infer_leaves, strict=True):
        require(np.array_equal(finite(x), finite(y)), 'Training/policy input parity mismatch')
    require(trained['actions'].shape == (50, 32), 'Wrong transformed action dimensions')
    recovered = output_transform(dict(state=np.array(trained['state']), actions=np.array(trained['actions'])))['actions']
    require(np.allclose(recovered, expected, atol=2e-5, rtol=2e-5), 'Delta/absolute normalization roundtrip failed')
    delta = t.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs])(copy.deepcopy(raw))['actions']
    q = dc.norm_stats['actions']
    outside = (delta[..., :14] < q.q01[:14]) | (delta[..., :14] > q.q99[:14])
    image_summary = {k: dict(shape=list(v.shape), min=float(np.min(v)), max=float(np.max(v))) for k, v in trained['image'].items()}
    return raw, trained, expected, outside, dict(input_parity=True, roundtrip=True, images=image_summary,
        state_shape=list(trained['state'].shape), prompt=a['prompt'])


def strict_load(cfg, checkpoint):
    import flax.nnx as nnx
    import jax
    import numpy as np
    from openpi.models import model
    from openpi.shared import array_typing as at, normalize
    # Same exact template/restore checks as gen1_verify_pilot, but never TrainState/optimizer.
    reference = nnx.state(nnx.eval_shape(cfg.model.create, jax.random.key(0))).to_pure_dict()
    params = model.restore_params(checkpoint / 'params', restore_type=np.ndarray)
    at.check_pytree_equality(expected=reference, got=params, check_shapes=True, check_dtypes=True)
    for leaf in jax.tree.leaves(params):
        require(not isinstance(leaf, jax.ShapeDtypeStruct), 'Unrestored checkpoint leaf')
        flat = np.asarray(leaf).reshape(-1)
        for start in range(0, flat.size, 1_000_000):
            finite(flat[start:start + 1_000_000])
    stats, original = normalize.load(checkpoint / 'assets/aloha_agilex_sim'), normalize.load(SOURCE / 'assets/aloha_agilex_sim')
    require(set(stats) == set(original), 'Normalization keys changed')
    for key in stats:
        for field in ('mean', 'std', 'q01', 'q99'):
            x, y = getattr(stats[key], field), getattr(original[key], field)
            require(x is not None and y is not None and np.array_equal(finite(x), finite(y)), 'Normalization changed')
    # Match production policy_config after strict checkpoint-native validation.
    import jax.numpy as jnp
    native_dtypes = sorted({str(np.asarray(v).dtype) for v in jax.tree.leaves(params)})
    params = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.bfloat16), params)
    loaded = cfg.model.load(params, remove_extra_params=False)
    dtype_report = dict(native=native_dtypes, inference='bfloat16', explicit_production_cast=True)
    del params
    gc.collect()
    return loaded, dtype_report


def worker(a):
    manifest = read_json(a.manifest)
    validate_manifest(manifest)
    environment(not a.check_inputs_only)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models import model
    from openpi.policies.policy import Policy
    from openpi.shared import nnx_utils
    require(all(d.platform == ('cpu' if a.check_inputs_only else 'gpu') for d in jax.devices()), 'Unexpected device platform')
    cfg = helpers()[0].build_config(defaults())
    expected_config = json.loads(json.dumps(dataclasses.asdict(cfg), default=str))
    require(expected_config == read_json(PROVENANCE)['config'], 'Diagnostic config differs from training')
    anchors = manifest['anchors'][:1] if a.smoke else manifest['anchors']
    bundles = {}
    policy = loss_fn = None
    dtype_report = None
    if not a.check_inputs_only:
        loaded, dtype_report = strict_load(cfg, Path(manifest['original' if a.worker_model == 'original' else 'ema']))
        # `compute_loss` branches on the Python `train` flag; keep it static
        # because the pinned module_jit wrapper forwards kwargs to jax.jit.
        loss_fn = nnx_utils.module_jit(loaded.compute_loss, static_argnames=('train',))
    records = []
    for index, anchor in enumerate(anchors):
        guard(a.output.parent, 64 * 2**20)
        repo = anchor['repo_id']
        if repo not in bundles:
            bundles[repo] = dataset_bundle(cfg, repo)
        raw, trained, expected, outside, parity = checked_anchor(bundles[repo], anchor)
        if a.check_inputs_only:
            records.append(dict(anchor_id=anchor['anchor_id'], parity=parity))
            continue
        if policy is None:
            _, _, inputs, outputs = transforms_for(bundles[repo][0])
            policy = Policy(loaded, transforms=inputs, output_transforms=outputs, sample_kwargs={'num_steps': 10})
        obs = model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], trained))
        actions = jnp.asarray(trained['actions'])[None]
        for draw, seed in enumerate(SEEDS):
            # Every anchor/model uses the same explicit keys. Native API splits loss key into preprocessing/noise/time.
            key = jax.random.key(seed)
            noise = np.asarray(jax.random.normal(jax.random.fold_in(key, 1), (50, 32)))
            loss = finite(loss_fn(key, obs, actions, train=False))
            require(loss.shape == (1, 50), 'Native loss API changed: expected dimension-reduced (1,50)')
            predicted = finite(policy.infer(raw, noise=noise)['actions'])
            require(predicted.shape == (50, 14), 'Wrong physical sampled actions')
            if index == 0 and draw == 0:
                loss2 = finite(loss_fn(key, obs, actions, train=False))
                pred2 = finite(policy.infer(raw, noise=noise)['actions'])
                require(np.allclose(loss, loss2, atol=1e-6, rtol=1e-6) and
                        np.allclose(predicted, pred2, atol=1e-6, rtol=1e-6), 'Repeated input is numerically unstable')
                parity['repeat_loss_max_abs'] = float(np.max(np.abs(loss-loss2)))
                parity['repeat_actions_max_abs'] = float(np.max(np.abs(predicted-pred2)))
            # Direct native sampling also exposes the 18 padded normalized dimensions discarded by AlohaOutputs.
            sampled = finite(policy._sample_actions(key, obs, noise=jnp.asarray(noise)[None], num_steps=10))[0]
            _, inverse, _, _ = transforms_for(bundles[repo][0])
            direct = inverse(dict(state=np.array(trained['state']), actions=sampled.copy()))['actions']
            require(np.allclose(predicted, direct, atol=1e-6, rtol=1e-6), 'Policy/native sampling parity mismatch')
            error = finite(predicted.astype(np.float64) - expected)
            record = dict(anchor_id=anchor['anchor_id'], split=anchor['split'], task=anchor['task'],
                episode=anchor['episode'], frame=anchor['frame'], draw=draw, seed=seed, model=a.worker_model,
                valid=anchor['valid'], outside=outside.tolist(), native_loss=loss[0].tolist(),
                abs_error=np.abs(error).tolist(), squared_error=np.square(error).tolist(),
                normalized_unused_dim_mae=float(np.mean(np.abs(sampled[:, 14:] - trained['actions'][:, 14:]))),
                parity=parity if draw == 0 else None)
            records.append(record)
        print(f'{a.worker_model} {index+1}/{len(anchors)} {anchor["anchor_id"]}', flush=True)
    guard(a.output.parent, 64 * 2**20)
    write_json(a.output / 'records.json', records)
    write_json(a.output / 'completed.json', dict(status='passed', model=a.worker_model,
        anchors=len(anchors), records=len(records), check_inputs_only=a.check_inputs_only,
        diagnostic_parameter_dtypes=dtype_report, production_parameter_dtype='bfloat16',
        model_compute_dtype=str(cfg.model.dtype)))


def summarize(records):
    """Pool cells within episodes; only then average episode statistics. No frame-level CI."""
    import numpy as np
    groups = defaultdict(list)
    for r in records:
        groups[(r['split'], r['task'], r['episode'])].append(r)
    result = []
    for (split, task, episode), rows in sorted(groups.items()):
        loss = np.stack([r['native_loss'] for r in rows])
        absolute = np.stack([r['abs_error'] for r in rows])
        squared = np.stack([r['squared_error'] for r in rows])
        valid = np.asarray([r['valid'] for r in rows])
        outside = np.asarray([r['outside'] for r in rows])
        slices = {}
        horizon = np.broadcast_to(np.arange(50), valid.shape)
        for name, mask in [('all', np.ones_like(valid)), ('valid', valid), ('padded', ~valid),
                           ('h00_09', horizon < 10), ('h10_24', (horizon >= 10) & (horizon < 25)), ('h25_49', horizon >= 25)]:
            metrics = dict(target_horizons=int(mask.sum()), native_loss=float(loss[mask].mean()) if mask.any() else None)
            for dim_name, dims in [('joints_rad', JOINTS), ('grippers_control', [6, 13])]:
                selected_abs, selected_sq = absolute[mask][:, dims], squared[mask][:, dims]
                metrics[dim_name] = dict(mae=float(selected_abs.mean()), rmse=float(np.sqrt(selected_sq.mean()))) if mask.any() else None
            metrics['dimension_mae'] = absolute[mask].mean(axis=0).tolist() if mask.any() else None
            metrics['dimension_rmse'] = np.sqrt(squared[mask].mean(axis=0)).tolist() if mask.any() else None
            for label, qmask in [('outside_quantile', outside), ('inside_quantile', ~outside)]:
                metrics[label] = {}
                for dim_name, dims in [('joints_rad', JOINTS), ('grippers_control', [6, 13])]:
                    cell = (qmask & mask[..., None])[..., dims]
                    metrics[label][dim_name] = dict(cells=int(cell.sum()),
                        mae=float(absolute[..., dims][cell].mean()) if cell.any() else None,
                        rmse=float(np.sqrt(squared[..., dims][cell].mean())) if cell.any() else None)
            slices[name] = metrics
        result.append(dict(split=split, task=task, episode=episode, metrics=slices))
    return result


def flatten_numbers(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_numbers(child, f'{prefix}/{key}')
    elif isinstance(value, list):
        for key, child in enumerate(value):
            yield from flatten_numbers(child, f'{prefix}/{key}')
    elif isinstance(value, (float, int)):
        yield prefix, value


def paired_summary(original, ema):
    require([(r['split'], r['task'], r['episode']) for r in original] ==
            [(r['split'], r['task'], r['episode']) for r in ema], 'Unpaired episode summaries')
    paired, groups = [], defaultdict(lambda: defaultdict(list))
    for first, second in zip(original, ema, strict=True):
        x, y = dict(flatten_numbers(first['metrics'])), dict(flatten_numbers(second['metrics']))
        require(x.keys() == y.keys(), 'Unpaired metric support')
        delta = {k: y[k] - x[k] for k in x if not k.endswith(('/cells', '/target_horizons'))}
        paired.append(dict(split=first['split'], task=first['task'], episode=first['episode'], ema_minus_original=delta))
        for group in (f"{first['split']}/{first['task']}", f"{first['split']}/all_tasks"):
            for key, value in delta.items():
                groups[group][key].append(value)
    macro = {group: {key: dict(episode_mean_delta=sum(values)/len(values), episodes=len(values))
                     for key, values in metrics.items()} for group, metrics in groups.items()}
    return dict(episodes=paired, equal_episode_macro=macro)


def bounded_worker(command, timeout):
    proc = subprocess.Popen(command, start_new_session=True)
    try:
        code = proc.wait(timeout=timeout)
        require(code == 0, f'Worker exited {code}')
    finally:
        # Clean up the full process group on timeout, interrupt, or worker failure.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def run(a):
    manifest = read_json(a.manifest)
    validate_manifest(manifest)
    require(a.check_inputs_only or a.coordinator_approved_gpu, 'GPU evaluation needs explicit --coordinator-approved-gpu')
    models = ['original'] if a.check_inputs_only else ['original', 'ema']
    for name in models:
        sub = a.output / name
        sub.mkdir(exist_ok=False)
        command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '_worker', '--manifest', str(a.manifest),
                   '--output', str(sub), '--worker-model', name]
        if a.command == 'smoke':
            command.append('--smoke')
        command.append('--check-inputs-only' if a.check_inputs_only else '--coordinator-approved-gpu')
        bounded_worker(command, timeout=1800 if a.command == 'smoke' else 21600)
        require(read_json(sub / 'completed.json')['status'] == 'passed', 'Worker incomplete')
    report = dict(status='passed', mode=a.command, manifest_sha256=digest(a.manifest),
                  check_inputs_only=a.check_inputs_only, limitations=LIMITATIONS)
    if not a.check_inputs_only:
        original = summarize(read_json(a.output / 'original/records.json'))
        ema = summarize(read_json(a.output / 'ema/records.json'))
        report.update(original=original, ema=ema, paired=paired_summary(original, ema))
    guard(a.output, 64 * 2**20)
    write_json(a.output / 'report.json', report)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=('prepare', 'smoke', 'evaluate', '_worker'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--coordinator-approved-gpu', action='store_true')
    p.add_argument('--check-inputs-only', action='store_true', help='CPU-only actual-loader/transform parity, no weights')
    p.add_argument('--worker-model', choices=('original', 'ema'))
    p.add_argument('--smoke', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    require(a.command == 'prepare' or a.manifest is not None, '--manifest required')
    if a.command == '_worker':
        require(a.check_inputs_only or a.coordinator_approved_gpu, 'Worker GPU launch needs opt-in')
        require(a.output.parent.parent.resolve() == ROOT / 'outputs' and a.output.is_dir()
                and not any(a.output.iterdir()) and a.worker_model is not None, 'Unsafe worker output')
        worker(a)
        return
    a.output = fresh_output(a.output)
    try:
        if a.command == 'prepare':
            prepare(a.output)
        else:
            run(a)
    except BaseException as exc:
        write_json(a.output / 'failed.json', dict(status='failed', error=repr(exc)))
        raise
    print(str(a.output), flush=True)


if __name__ == '__main__':
    main()
