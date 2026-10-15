#!/usr/bin/env python3
"""Audited full-parameter recovery training on the broad Clean split.

This is deliberately separate from the historical two-task pilots. The only
interventions are broader task coverage, equal task sampling, and the corrected
source-grounded prompt augmentation retained from recovery experiments;
normalization, action semantics, horizon, native loss, and inference sampling
remain those of the established Myungkyu checkpoint pipeline.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OPENPI = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi'
SOURCE = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
CONVERTER = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/scripts/transform_lerobot_v30_format.py'
INTERVENTION = '18-task coverage, equal task sampling, corrected source prompt augmentation; native 32-dimensional loss'
sys.path.insert(0, str(ROOT / 'scripts'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, default=str, allow_nan=False)
        stream.write('\n')


def free_gib(path=ROOT):
    return shutil.disk_usage(path).free / 2**30


def fresh_child(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    require(not path.exists() and not path.is_symlink(), f'Existing target: {path}')
    require(path.parent == parent and parent.is_dir(), f'Target must be a new child of {parent}')
    require(path != SOURCE.resolve(), 'Refusing to overwrite source checkpoint')
    return path


def validate_inputs(args):
    from gen1_multitask_data import audit_dataset, validate_split

    manifest = read_json(args.split_manifest)
    rows = validate_split(manifest)
    require(Path(manifest['original_checkpoint']).resolve() == SOURCE.resolve(),
            'Split references a different initializer')
    require(args.repo_id != args.val_repo_id, 'Train and validation datasets must differ')
    audit_rows = {}
    for split, repo, audit_path in (
        ('train', args.repo_id, args.train_audit), ('val', args.val_repo_id, args.val_audit)):
        dataset = (args.lerobot_home / repo).resolve()
        require(dataset.is_dir(), f'Missing {split} dataset: {dataset}')
        require(Path(audit_path).resolve() == (dataset / 'conversion_audit.json').resolve(),
                f'{split} audit is not attached to its dataset')
        audit = read_json(audit_path)
        require(audit.get('converter_sha256') == digest(CONVERTER), 'Converter hash changed')
        converted, report = audit_dataset(manifest, dataset, split, CONVERTER)
        audit_rows[split] = dict(report, rows=converted)
    train_frames = audit_rows['train']['frames']
    val_frames = audit_rows['val']['frames']
    require(train_frames == sum(r['frames'] for r in rows if r['split'] == 'train'),
            'Training frame total differs from split')
    require(val_frames == sum(r['frames'] for r in rows if r['split'] == 'val'),
            'Validation frame total differs from split')
    return manifest, audit_rows


def build_config(args):
    import flax.nnx as nnx
    from openpi.training import config, optimizer, weight_loaders

    cfg = config.get_config('pi05_base_aloha_full_sim_arx-x5_seed_0')
    require(cfg.optimizer.clip_gradient_norm == 1.0, 'Pinned optimizer clipping changed')
    cfg = dataclasses.replace(
        cfg,
        exp_name=args.checkpoint_dir.name,
        seed=args.seed,
        model=dataclasses.replace(cfg.model, action_dim=32, action_horizon=50, pi05=True),
        data=dataclasses.replace(
            cfg.data,
            repo_id=args.repo_id,
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            assets=config.AssetsConfig(assets_dir=str(SOURCE / 'assets'), asset_id='aloha_agilex_sim'),
            base_config=dataclasses.replace(cfg.data.base_config, video_backend='pyav'),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(SOURCE / 'params')),
        checkpoint_dir_override=str(args.checkpoint_dir),
        freeze_filter=nnx.Nothing(),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.learning_rate,
            decay_steps=args.steps,
            decay_lr=args.learning_rate / 10,
        ),
        ema_decay=None,
        batch_size=args.batch_size,
        num_workers=0,
        fsdp_devices=1,
        num_train_steps=args.steps,
        log_interval=args.log_interval,
        save_interval=args.steps,
        keep_period=None,
        resume=False,
        overwrite=False,
        wandb_enabled=False,
    )
    require((SOURCE / 'params/_METADATA').is_file(), 'Source checkpoint metadata missing')
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    require(bool(data_config.norm_stats) and data_config.use_quantile_norm,
            'Original quantile normalization is required')
    return cfg


def phase_weights(args, rows):
    """Compute bounded expert-phase weights from audited 14-D actions."""
    import numpy as np
    import pyarrow.parquet as pq
    table = pq.read_table(args.lerobot_home / args.repo_id / 'data', columns=['action'])
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    require(actions.shape == (rows[-1]['stop'], 14), 'Phase action shape mismatch')
    weights = np.ones(len(actions), dtype=np.float32)
    event_frames = late_frames = 0
    offset = 0
    for row in rows:
        n = row['frames']
        local = actions[offset:offset+n]
        # Mark command threshold crossings, not every gripper ramp.
        # A crossing index denotes the earlier of the two adjacent frames.
        previous = local[:-1, [6, 13]]
        current = local[1:, [6, 13]]
        # Focus the intervention on grasp closure/lift, while late windows
        # cover terminal placement. Release timing remains native-supervised.
        crossing = np.any((previous >= 0.2) & (current < 0.2), axis=1)
        event = np.zeros(n, dtype=bool)
        for start in np.flatnonzero(crossing):
            if args.phase_window == 'post':
                # Preserve the historical run's exact earlier-index window.
                begin, stop = int(start), min(n, int(start) + 30)
            else:
                # At closure frame c, starts c-49..c contain c in a
                # 50-action chunk. Include post-closure frames c..c+29.
                closure = int(start) + 1
                begin, stop = max(0, closure - 49), min(n, closure + 30)
            event[begin:stop] = True
        late = np.arange(n) >= int(np.ceil(0.8 * max(n - 1, 1)))
        weights[offset:offset+n] = np.minimum(1.0 + event + 0.5 * late, 3.0)
        event_frames += int(event.sum())
        late_frames += int(late.sum())
        offset += n
    write_new(args.provenance_dir / 'phase_sampling.json', dict(
        status='audited', event_frames=event_frames, late_frames=late_frames,
        total_frames=len(actions), event_weight=2.0, late_weight=1.5, cap=3.0,
        source='14-dim physical action targets', phase_window=args.phase_window,
        window_definition=('legacy earlier-index [c-1,c+29)' if args.phase_window == 'post'
                           else 'closure-index [max(0,c-49),min(n,c+30))')))
    return weights


def install_balanced_loader(cfg, args, rows, sharding):
    """Compose a local loader; leave the pinned loader factory unchanged."""
    import jax
    from openpi.training import data_loader
    from gen1_multitask_data import TaskBalancedSampler, PhaseAwareTaskSampler

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    require(len(raw) == rows[-1]['stop'],
            f'Loader length {len(raw)} differs from audited frames {rows[-1]["stop"]}')
    from gen1_multitask_prompts import prepare, PromptDataset
    pools, prompt_report = prepare(rows)
    write_new(args.provenance_dir / 'prompt_augmentation.json', prompt_report)
    raw = PromptDataset(raw, rows, pools, mode=args.prompt_mode)
    transformed = data_loader.transform_dataset(raw, data_config)
    sampler = (PhaseAwareTaskSampler(rows, len(transformed), phase_weights(args, rows), seed=args.data_seed)
               if args.phase_aware else TaskBalancedSampler(rows, len(transformed), seed=args.data_seed))
    local_batch = args.batch_size // jax.process_count()
    require(local_batch > 0 and args.batch_size % jax.process_count() == 0,
            'Batch size must divide process count')
    loader = data_loader.TorchDataLoader(
        transformed,
        local_batch_size=local_batch,
        sharding=sharding,
        sampler=sampler,
        shuffle=False,
        num_batches=None,
        num_workers=0,
        seed=args.data_seed,
        framework='jax',
    )
    wrapped = data_loader.DataLoaderImpl(data_config, loader)
    return wrapped, sampler


def validation_anchors(cfg, args, rows):
    """Eight floor-spaced anchors per held-out trajectory, including its end."""
    import numpy as np
    import pandas as pd
    import gen1_diagnose_pilot as diagnostic
    from openpi.training import data_loader

    require(all(row['split'] == 'val' for row in rows), 'Non-validation rows in held-out panel')
    require(args.repo_id != args.val_repo_id, 'Validation repo must differ from training')
    data_config = dataclasses.replace(cfg.data, repo_id=args.val_repo_id).create(cfg.assets_dirs, cfg.model)
    raw = data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    require(len(raw) == rows[-1]['stop'], 'Validation loader length mismatch')
    transformed = data_loader.transform_dataset(raw, data_config)
    frames = pd.concat([pd.read_parquet(p, columns=['index', 'task_index']) for p in
                        sorted((args.lerobot_home / args.val_repo_id / 'data').rglob('*.parquet'))])
    frames = frames.set_index('index')
    tasks = pd.read_parquet(args.lerobot_home / args.val_repo_id / 'meta/tasks.parquet')
    prompts = {int(row.task_index): str(prompt) for prompt, row in tasks.iterrows()}
    anchors = []
    for row in rows:
        positions = diagnostic.anchor_frames(row['frames'])
        for frame in positions:
            index = row['start'] + frame
            indices, valid = diagnostic.horizon_indices(frame, row['frames'])
            anchor = dict(index=index, episode=row['episode'], frame=frame, length=row['frames'],
                          source=row['path'], horizon_frames=indices, valid=valid,
                          prompt=prompts[int(frames.loc[index, 'task_index'])])
            _, train_item, expected, _, _ = diagnostic.checked_anchor(
                (data_config, raw, transformed), anchor)
            anchors.append(dict(row=row, index=index, frame=frame,
                                transformed=train_item, expected=expected,
                                valid=np.asarray(valid)))
    require(len(anchors) == 8 * len(rows), 'Unexpected held-out anchor count')
    return anchors


def output_transform(data_config):
    from openpi import transforms as t
    return t.compose([
        *data_config.model_transforms.outputs,
        t.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
        *data_config.repack_transforms.outputs,
    ])


def validate_model(state, cfg, anchors, output_fn, step, out_dir, batch_size):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models import model as model_lib

    def sample(params, obs, noise):
        model = nnx.merge(state.model_def, params)
        model.eval()
        return model.sample_actions(jax.random.key(0), obs, noise=noise, num_steps=10)

    sample_fn = jax.jit(sample)
    noise = jax.random.normal(jax.random.key(1729), (batch_size, 50, 32))
    records = []
    production = jax.tree.map(lambda x: x.astype(jnp.bfloat16), state.params)
    jax.block_until_ready(production)
    for start in range(0, len(anchors), batch_size):
        selected = anchors[start:start + batch_size]
        tree = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)),
                            *[x['transformed'] for x in selected])
        obs = model_lib.Observation.from_dict(tree)
        predicted = np.asarray(sample_fn(production, obs, noise[:len(selected)]))
        for anchor, prediction in zip(selected, predicted, strict=True):
            actual = output_fn(dict(state=np.asarray(anchor['transformed']['state']),
                                    actions=prediction.copy()))['actions']
            error = np.abs(np.asarray(actual) - anchor['expected'])
            valid = anchor['valid']
            joints = [i for i in range(14) if i not in (6, 13)]
            records.append(dict(task=anchor['row']['task'], episode=anchor['row']['episode'],
                frame=anchor['frame'], joint_mae=float(error[valid][:, joints].mean()),
                gripper_mae=float(error[valid][:, [6, 13]].mean()),
                first10_joint_mae=float(error[:10][valid[:10]][:, joints].mean()),
                first10_gripper_mae=float(error[:10][valid[:10]][:, [6, 13]].mean())))
    metrics = {}
    for task in sorted({r['task'] for r in records}) + ['all']:
        rows = [r for r in records if task == 'all' or r['task'] == task]
        metrics[task] = {key: float(np.mean([r[key] for r in rows])) for key in
                         ('joint_mae', 'gripper_mae', 'first10_joint_mae', 'first10_gripper_mae')}
    write_new(Path(out_dir) / f'validation_{step:04d}.json',
              dict(step=step, anchors=len(records), metrics=metrics, records=records))
    return metrics


def save_inference(state, cfg, args, step):
    import jax
    import orbax.checkpoint as ocp
    require(free_gib() >= 16, 'Need checkpoint space plus 10 GiB reserve')
    root = args.checkpoint_dir / str(step)
    root.mkdir()
    shutil.copytree(SOURCE / 'assets', root / 'assets')
    params = jax.tree.map(lambda x: x.astype('bfloat16'), state.params.to_pure_dict())
    with ocp.Checkpointer(ocp.PyTreeCheckpointHandler(save_concurrent_gb=8)) as saver:
        saver.save(root / 'params', {'params': params})
    write_new(root / 'provenance.json', dict(step=step, source=str(SOURCE), kind='raw',
        trainable_parameters=args.trainable_parameters, run=str(args.provenance_dir), production_dtype='bfloat16',
        initialization=str(SOURCE), optimizer_restarted=False, intervention=INTERVENTION))


def run_train(cfg, args, train_rows, val_rows):
    import flax.nnx as nnx
    import jax
    import numpy as np
    from openpi.training import sharding

    spec = importlib.util.spec_from_file_location('gen1_multitask_pinned_train', OPENPI / 'scripts/train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    mesh = sharding.make_mesh(1)
    data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader, sampler = install_balanced_loader(cfg, args, train_rows, data_shard)
    iterator = iter(loader)
    batch = next(iterator)
    for value in jax.tree.leaves(batch):
        require(np.isfinite(np.asarray(value)).all(), 'Nonfinite loader batch')
    require(batch[1].shape == (args.batch_size, 50, 32), f'Wrong batch shape: {batch[1].shape}')
    state, state_shard = train.init_train_state(cfg, jax.random.key(args.seed), mesh, resume=False)
    jax.block_until_ready(state)
    # Empty because this intervention trains all parameters; consumed by the
    # existing independent CPU checkpoint verifier after training exits.
    write_new(args.provenance_dir / 'frozen_initial_sha256.json', {})
    step_fn = jax.jit(functools.partial(train.train_step, cfg),
                      in_shardings=(replicated, state_shard, data_shard),
                      out_shardings=(state_shard, replicated), donate_argnums=(1,))
    anchors = validation_anchors(cfg, args, val_rows)
    write_new(args.provenance_dir / 'validation_panel.json', [
        {k: v for k, v in a.items() if k not in ('transformed', 'expected', 'valid')}
        for a in anchors])
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    inverse = output_transform(data_config)
    initial = validate_model(state, cfg, anchors, inverse, 0, args.provenance_dir, args.batch_size)
    write_new(args.provenance_dir / 'sampler_initial.json', sampler.report())
    started = time.monotonic()
    candidates = []
    with (args.provenance_dir / 'metrics.jsonl').open('x', buffering=1) as stream:
        for step in range(1, args.steps + 1):
            with sharding.set_mesh(mesh):
                state, info = step_fn(jax.random.key(args.seed + 1), state, batch)
            values = {key: float(value) for key, value in jax.device_get(info).items()}
            require(all(np.isfinite(value) for value in values.values()), 'Nonfinite training metric')
            stream.write(json.dumps(dict(step=step, **values), allow_nan=False) + '\n')
            if step % args.log_interval == 0:
                print('STEP', step, 'loss', values['loss'], 'elapsed', round(time.monotonic() - started), flush=True)
            if step in args.save_steps:
                metrics = validate_model(state, cfg, anchors, inverse, step, args.provenance_dir, args.batch_size)
                save_inference(state, cfg, args, step)
                score = sum(metrics[t][k] / max(initial[t][k], 1e-8)
                            for t in initial if t != 'all' for k in initial[t]) / (4 * (len(initial) - 1))
                candidates.append(dict(step=step, score=score, checkpoint=str(args.checkpoint_dir / str(step))))
                print('SAVED', step, json.dumps(metrics), flush=True)
            if step < args.steps:
                batch = next(iterator)
    selected = min(candidates, key=lambda row: (row['score'], row['step']))
    write_new(args.provenance_dir / 'selection.json', dict(selected=selected, candidates=candidates,
        rule='Equal mean of four per-task MAE ratios to step zero; earlier step breaks ties; no rollouts used'))
    write_new(args.provenance_dir / 'completed.json', dict(status='completed', steps=args.steps,
        save_steps=args.save_steps, sampler=sampler.report(), initial_validation=initial,
        intervention=INTERVENTION))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('config', 'loader', 'train'), default='config')
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--val-repo-id', required=True)
    parser.add_argument('--lerobot-home', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--train-audit', type=Path, required=True)
    parser.add_argument('--val-audit', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--provenance-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=24000)
    parser.add_argument('--save-steps', type=int, nargs='+', default=[1000, 4000, 12000, 24000])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=3e-6)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-seed', type=int, default=1)
    parser.add_argument('--phase-aware', action='store_true',
                        help='Oversample audited gripper-transition and late-trajectory windows within each task')
    parser.add_argument('--phase-window', choices=('post', 'prepost'), default='post',
                        help='Closure window for phase-aware sampling; post preserves the original intervention, prepost adds the preceding 50-frame context')
    parser.add_argument('--prompt-mode', choices=('frame', 'episode'), default='frame',
                        help='Prompt sampling granularity; frame preserves the prior run, episode fixes one audited source prompt per trajectory')
    parser.add_argument('--coordinator-approved-gpu', action='store_true')
    args = parser.parse_args()
    require(args.mode != 'train' or args.coordinator_approved_gpu,
            'GPU training requires explicit coordinator approval flag')
    require('jax' not in sys.modules, 'Run this script in a fresh Python process')
    require(args.steps > 0 and args.save_steps and all(0 < x <= args.steps for x in args.save_steps),
            'Invalid training/save steps')
    require(sorted(set(args.save_steps)) == args.save_steps, 'Save steps must be sorted and unique')
    require(args.save_steps[-1] == args.steps, 'Final step must be saved')
    require(0 <= args.warmup_steps < args.steps and args.learning_rate > 0
            and args.log_interval > 0, 'Invalid optimizer/logging parameters')
    require(args.batch_size > 0 and args.batch_size % 1 == 0, 'Invalid batch size')
    require(args.repo_id.isidentifier() and args.val_repo_id.isidentifier(), 'Unsafe repo id')
    args.lerobot_home = args.lerobot_home.resolve()
    require(args.lerobot_home.is_relative_to(ROOT / 'data'), 'LeRobot home must be below project data')
    args.checkpoint_dir = fresh_child(args.checkpoint_dir, ROOT / 'checkpoints')
    args.provenance_dir = fresh_child(args.provenance_dir, ROOT / 'outputs')
    if args.mode == 'train':
        require(free_gib() >= max(65, len(args.save_steps) * 6 + 10), 'Insufficient checkpoint budget')
    os.environ.pop('XLA_PYTHON_CLIENT_MEM_FRACTION', None)
    os.environ.update(
        HF_LEROBOT_HOME=str(args.lerobot_home), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_PLATFORMS='cuda' if args.mode == 'train' else 'cpu',
        CUDA_VISIBLE_DEVICES='0' if args.mode == 'train' else '',
        JAX_COMPILATION_CACHE_DIR=str(ROOT / 'tmp/gen1_training/jax'),
    )
    sys.path[:0] = [str(OPENPI / 'src'), str(OPENPI / 'packages/openpi-client/src')]
    manifest, audits = validate_inputs(args)
    cfg = build_config(args)
    import flax.nnx as nnx
    import jax
    import numpy as np
    abstract = nnx.state(nnx.eval_shape(cfg.model.create, jax.random.key(0)))
    args.trainable_parameters = int(sum(np.prod(x.shape) for x in
                                        jax.tree.leaves(abstract.filter(cfg.trainable_filter))))
    require(args.trainable_parameters == 3353433872, 'Unexpected full-model parameter count')
    report = dict(status='validated', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        argv=sys.argv, mode=args.mode, source=str(SOURCE), source_revision=manifest['source_revision'],
        split_manifest=str(args.split_manifest.resolve()), split_sha256=digest(args.split_manifest),
        tasks=manifest['tasks'], counts=manifest['counts'], audits={k: {x: v for x, v in val.items() if x != 'rows'}
                                                                     for k, val in audits.items()},
        intervention=INTERVENTION + (f'; phase-aware bounded sampling ({args.phase_window})' if args.phase_aware else '')
        + f'; prompt sampling ({args.prompt_mode})',
        optimization_change='Batch 8, 24000 updates, peak LR 3e-6; bundled recovery experiment, not an augmentation-only ablation',
        loss='native 32-dimensional flow matching',
        physical_action_dim=14, action_horizon=50, batch_size=args.batch_size,
        full_parameter_training=True, save_steps=args.save_steps,
        trainable_parameters=args.trainable_parameters, sampled_training_starts=args.steps * args.batch_size,
        checkpoint_selection='Equal mean of four per-task MAE ratios to step zero; earlier step breaks ties',
        validation_panel='8 floor-spaced anchors per held-out trajectory; padded time steps excluded',
        versions={name: importlib.metadata.version(name) for name in
                  ('jax', 'jaxlib', 'flax', 'orbax-checkpoint', 'lerobot', 'optax')},
        free_gib=free_gib(), config=dataclasses.asdict(cfg),
        sha256={str(path): digest(path) for path in
                (Path(__file__), ROOT / 'scripts/gen1_multitask_data.py', ROOT / 'scripts/gen1_diagnose_pilot.py',
                 ROOT / 'scripts/gen1_multitask_prompts.py', ROOT / 'scripts/gen1_prompt_augmentation.py',
                 ROOT / 'scripts/gen1_multitask_video.py', ROOT / 'scripts/gen1_multitask_audit.py',
                 CONVERTER, OPENPI / 'scripts/train.py', SOURCE / 'assets/aloha_agilex_sim/norm_stats.json')})
    print(json.dumps(report, indent=2, default=str, allow_nan=False), flush=True)
    if args.mode == 'config':
        print('CONFIG_OK; no GPU loader or training launched', flush=True)
        return
    args.provenance_dir.mkdir()
    write_new(args.provenance_dir / 'run_config.json', report)
    write_new(args.provenance_dir / 'split_manifest.json', manifest)
    write_new(args.provenance_dir / 'train_conversion_audit.json', read_json(args.train_audit))
    write_new(args.provenance_dir / 'val_conversion_audit.json', read_json(args.val_audit))
    from gen1_multitask_video import install as install_video_reader
    video_reader, video_report = install_video_reader([
        args.lerobot_home / args.repo_id, args.lerobot_home / args.val_repo_id])
    write_new(args.provenance_dir / 'video_reader.json', video_report)
    if args.mode == 'loader':
        import jax
        from openpi.training import sharding
        # Loader mode uses CPU JAX and checks a real batch plus sampler counts.
        mesh = sharding.make_mesh(1)
        shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        loader, sampler = install_balanced_loader(cfg, args, audits['train']['rows'], shard)
        batch = next(iter(loader))
        import numpy as np
        for value in jax.tree.leaves(batch):
            require(np.isfinite(np.asarray(value)).all(), 'Nonfinite loader batch')
        require(batch[1].shape == (args.batch_size, 50, 32), 'Loader action shape mismatch')
        anchors = validation_anchors(cfg, args, audits['val']['rows'])
        write_new(args.provenance_dir / 'loader_preflight.json', dict(status='passed',
            batch_shape=list(batch[1].shape), sampler=sampler.report(), validation_anchors=len(anchors),
            validation_repo=args.val_repo_id, source_action_parity=True, transform_roundtrip=True))
        print('LOADER_OK', json.dumps(sampler.report()), flush=True)
        video_reader.close()
        return
    args.checkpoint_dir.mkdir()
    import gen1_recovery_eval as recovery
    recovery.release_checkpoint_cache()
    try:
        run_train(cfg, args, audits['train']['rows'], audits['val']['rows'])
    finally:
        write_new(args.provenance_dir / 'video_reader_final.json', video_reader.report())
        video_reader.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    main()
