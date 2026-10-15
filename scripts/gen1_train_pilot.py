#!/usr/bin/env python3
"""Fixed 2000-step external pilot. Config/loader are CPU-only; train needs approval.

Config mode tolerates absent conversion audits, but never claims data readiness.
Train/loader require both passed conversion audits and the frozen 90/10 split.
No conversion, collection, resume, overwrite, periodic save, or heldout training.
"""
import argparse
from collections import Counter
import dataclasses
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import types
import uuid

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OPENPI = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi'
SOURCE = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
REVISION = 'ba0fd21ad21c336bedc7bd8eacd3e514860910e2'
TASKS = {'place_mouse_pad', 'hanging_mug'}
STEPS = 2000
FINAL_STEP = STEPS - 1
GIB = 2**30


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def disk_guard(path, minimum):
    free = shutil.disk_usage(path).free
    require(free >= minimum * GIB, f'{path}: {free / GIB:.2f}GiB free; need >= {minimum}GiB')
    return free / GIB


def fresh_target(path, parent):
    require(not path.exists() and not path.is_symlink(), f'Refusing existing target: {path}')
    resolved = path.resolve()
    # Direct children of established output roots cannot nest inside source datasets/checkpoints.
    require(resolved.parent == parent.resolve(), f'Target must be a direct child of {parent}')
    require(parent.is_dir() and not parent.is_symlink(), f'Unsafe/missing output root: {parent}')
    require(resolved != SOURCE.resolve(), 'Cannot overwrite original checkpoint')
    return resolved


def validate_inputs(args):
    paths = [args.split_manifest, args.train_audit, args.val_audit]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        require(args.mode == 'config', f'Missing required split/audit files: {missing}')
        return {'status': 'NOT_VALIDATED', 'missing': missing}
    manifest = read_json(args.split_manifest)
    require(manifest.get('status') == 'passed', 'Split manifest must pass')
    require(set(manifest['tasks']) == TASKS and manifest['counts'] == {'train': 90, 'val': 10},
            'Need two-task 90 train / 10 validation trajectory split')
    require(Path(manifest['original_checkpoint']).resolve() == SOURCE.resolve(), 'Wrong original checkpoint')
    rows = manifest['episodes']
    require(len(rows) == 100, 'Split must contain exactly 100 episodes')
    counts = Counter((r['split'], r['task']) for r in rows)
    require(counts == Counter({(s, t): n for s, n in [('train', 45), ('val', 5)] for t in TASKS}),
            'Need 45 train / 5 validation episodes per task')
    for key in ('path', 'sha256', 'seed'):
        values = [str(Path(r[key]).resolve()) if key == 'path' else r[key] for r in rows]
        require(len(set(values)) == 100, f'Duplicate split {key}: possible leakage')
    for r in rows:
        require(r['source_revision'] == REVISION and all(r[k] is True for k in
                ('success', 'plan_success', 'replay_success')), 'Unaccepted source episode')
        require(type(r['seed']) is int and 0 <= r['seed'] < 100000, 'Evaluation seed in training split')
        require(type(r['frames']) is int and r['frames'] > 1, 'Invalid frame count')
        require(Path(r['path']).resolve().is_relative_to(ROOT / 'data'), 'Source outside data root')
    summary = {'status': 'passed', 'split_manifest': str(args.split_manifest.resolve()),
               'split_sha256': digest(args.split_manifest), 'counts': manifest['counts'], 'audits': {}}
    for split, repo, audit_path, count in [('train', args.repo_id, args.train_audit, 90),
                                          ('val', args.val_repo_id, args.val_audit, 10)]:
        dataset = args.lerobot_home / repo
        require(audit_path.resolve() == (dataset / 'conversion_audit.json').resolve(),
                f'{split} audit must belong to its actual dataset')
        audit = read_json(audit_path)
        require(audit.get('status') == 'passed' and audit.get('all_frames_decoded') is True,
                f'{split} conversion audit did not pass all frames')
        require(audit['episodes'] == count and audit['fps'] == 15 and audit['state_action_dim'] == 14,
                f'{split} conversion dimensions/counts do not match')
        converter = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/scripts/transform_lerobot_v30_format.py'
        require(Path(audit['converter']).resolve() == converter and audit['converter_sha256'] == digest(converter),
                f'{split} audit is not from the pinned converter')
        expected = {(str(Path(r['path']).resolve()), r['sha256'], r['frames'])
                    for r in rows if r['split'] == split}
        actual = [(str(Path(r['path']).resolve()), r['sha256'], r['frames']) for r in audit['sources']]
        require(len(actual) == count and set(actual) == expected, f'{split} audit/split mismatch or leakage')
        frames = sum(r[2] for r in expected)
        require(audit['frames'] == frames, f'{split} audit frame count mismatch')
        info = read_json(dataset / 'meta/info.json')
        require(info['total_episodes'] == count and info['total_frames'] == frames and info['fps'] == 15,
                f'{split} dataset metadata differs from audit')
        for key in ('observation.state', 'action'):
            require(info['features'][key]['shape'] == [14], f'{split} physical dimension must be 14')
        summary['audits'][split] = {'path': str(audit_path.resolve()), 'sha256': digest(audit_path),
                                    'repo_id': repo, 'frames': frames}
    return summary


def build_config(args):
    import flax.nnx as nnx
    from openpi.training import config, optimizer, weight_loaders
    cfg = config.get_config('pi05_base_aloha_full_sim_arx-x5_seed_0')
    require(cfg.optimizer.clip_gradient_norm == 1.0, 'Pinned optimizer clip changed')
    cfg = dataclasses.replace(
        cfg, exp_name=args.checkpoint_dir.name, seed=0,
        model=dataclasses.replace(cfg.model, action_dim=32, action_horizon=50, pi05=True),
        data=dataclasses.replace(
            cfg.data, repo_id=args.repo_id, adapt_to_pi=False, use_delta_joint_actions=True,
            assets=config.AssetsConfig(assets_dir=str(SOURCE / 'assets'), asset_id='aloha_agilex_sim'),
            base_config=dataclasses.replace(cfg.data.base_config, video_backend='pyav')),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(SOURCE / 'params')),
        checkpoint_dir_override=str(args.checkpoint_dir), freeze_filter=nnx.Nothing(),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=5e-6,
                                                 decay_steps=STEPS, decay_lr=5e-7),
        ema_decay=0.99, batch_size=1, num_workers=0, fsdp_devices=1,
        num_train_steps=STEPS, log_interval=20, save_interval=STEPS, keep_period=None,
        resume=False, overwrite=False, wandb_enabled=False)
    require((SOURCE / 'params/_METADATA').is_file(), 'Original checkpoint metadata missing')
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    require(bool(dc.norm_stats) and dc.use_quantile_norm, 'Original quantile normalization required')
    return cfg


def shim_source():
    path = OPENPI / 'scripts/train.py'
    text = path.read_text()
    old = 'if step % config.log_interval == 0:'
    require(text.count(old) == 1, 'Pinned logging condition changed; review external shim')
    text = text.replace(old, 'if step % config.log_interval == 0 or step == config.num_train_steps - 1:')
    return compile(text, str(path), 'exec')


def finite_batch(batch):
    import jax
    import numpy as np
    for value in jax.tree.leaves(batch):
        require(np.isfinite(np.asarray(value)).all(), 'Nonfinite actual loader batch')
    require(batch[1].shape == (1, 50, 32), f'Wrong action shape: {batch[1].shape}')


def run_train(cfg, args):
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from etils import epath
    train = types.ModuleType('gen1_pilot_pinned_train')
    train.__file__ = str(OPENPI / 'scripts/train.py')
    sys.modules[train.__name__] = train
    exec(shim_source(), train.__dict__)
    checkpoints = train._checkpoints
    managers = []
    saved = []

    def initialize(checkpoint_dir, *, keep_period, overwrite, resume):
        require(not overwrite and not resume and keep_period is None, 'Only fresh final saving allowed')
        disk_guard(args.checkpoint_dir.parent, 115)
        directory = epath.Path(fresh_target(Path(checkpoint_dir), ROOT / 'checkpoints'))
        directory.mkdir(parents=False, exist_ok=False)
        manager = ocp.CheckpointManager(
            directory, item_handlers={'assets': checkpoints.CallbackHandler(),
                                      'train_state': ocp.PyTreeCheckpointHandler(save_concurrent_gb=8),
                                      'params': ocp.PyTreeCheckpointHandler(save_concurrent_gb=8)},
            options=ocp.CheckpointManagerOptions(max_to_keep=1, keep_period=None, create=False,
                                                async_options=ocp.AsyncOptions(timeout_secs=7200)))
        managers.append(manager)
        return manager, False

    def save(manager, state, loader, step):
        require(step == FINAL_STEP and not saved, 'Refusing non-final or duplicate checkpoint')
        require(int(jax.device_get(state.step)) == STEPS and state.ema_params is not None,
                'Final state must include 2000 updates and EMA')
        disk_guard(args.checkpoint_dir, 112)
        # Pinned save preserves raw weights + optimizer in train_state, EMA in params.
        checkpoints.save_state(manager, state, loader, step)
        saved.append(step)

    train._checkpoints = types.SimpleNamespace(initialize_checkpoint_dir=initialize, save_state=save)
    original_stack = train.common_utils.stack_forest

    def finite_stack(infos):
        stacked = original_stack(infos)
        # Check every per-step scalar in the logging window, not just its mean.
        for key, value in jax.device_get(stacked).items():
            require(np.isfinite(np.asarray(value)).all(), f'Nonfinite training metric: {key}')
        return stacked

    train.common_utils = types.SimpleNamespace(stack_forest=finite_stack)
    original_wandb = train.wandb

    def log_metrics(metrics, *, step):
        if 'loss' in metrics:
            row = {'step': step, **{k: float(v) for k, v in metrics.items()}}
            with (args.provenance_dir / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        original_wandb.log(metrics, step=step)

    train.wandb = types.SimpleNamespace(init=original_wandb.init, Image=original_wandb.Image, log=log_metrics)
    logging.basicConfig(level=logging.INFO)
    try:
        train.main(cfg)
        require(saved == [FINAL_STEP], 'Final checkpoint was not requested')
        require(list(managers[0].all_steps()) == [FINAL_STEP], 'Final checkpoint did not commit')
        with (args.provenance_dir / 'completed.json').open('x') as stream:
            json.dump({'status': 'completed', 'final_step': FINAL_STEP, 'updates': STEPS,
                       'checkpoint': str(args.checkpoint_dir / str(FINAL_STEP))}, stream, indent=2)
    finally:
        for manager in managers:
            manager.close()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mode', choices=['config', 'loader', 'train'], default='config')
    p.add_argument('--repo-id', required=True, help='Training-only converted dataset')
    p.add_argument('--val-repo-id', required=True, help='Heldout dataset: audit only, never trained')
    p.add_argument('--lerobot-home', type=Path, required=True)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--train-audit', type=Path, required=True)
    p.add_argument('--val-audit', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True, help='New direct child of EvolveVLA/checkpoints')
    p.add_argument('--provenance-dir', type=Path, required=True, help='New direct child of EvolveVLA/outputs')
    p.add_argument('--coordinator-approved-gpu', action='store_true')
    args = p.parse_args()
    require(args.mode != 'train' or args.coordinator_approved_gpu,
            'Train requires explicit --coordinator-approved-gpu after collection has stopped')
    require('jax' not in sys.modules, 'Use a fresh isolated Python process')
    for repo in (args.repo_id, args.val_repo_id):
        require(repo not in ('', '.', '..', 'fake') and Path(repo).name == repo and '/' not in repo,
                'repo-id must be one safe path component, not fake')
    require(args.repo_id != args.val_repo_id, 'Train and validation datasets must differ')
    args.lerobot_home = args.lerobot_home.resolve()
    require(args.lerobot_home.is_relative_to(ROOT / 'data'), 'LeRobot home must be below EvolveVLA/data')
    train_path, val_path = [(args.lerobot_home / repo).resolve() for repo in (args.repo_id, args.val_repo_id)]
    require(train_path != val_path and all(x.is_relative_to(ROOT / 'data') for x in (train_path, val_path)),
            'Unsafe or aliased train/validation dataset paths')
    args.checkpoint_dir = fresh_target(args.checkpoint_dir, ROOT / 'checkpoints')
    args.provenance_dir = fresh_target(args.provenance_dir, ROOT / 'outputs')
    inputs = validate_inputs(args)
    free = disk_guard(args.checkpoint_dir.parent, 115) if args.mode == 'train' else shutil.disk_usage(ROOT).free / GIB
    sys.dont_write_bytecode = True
    # jaxlib 0.5.3 rejects simultaneous legacy and current allocator variables.
    os.environ.pop('XLA_PYTHON_CLIENT_MEM_FRACTION', None)
    os.environ.update(HF_LEROBOT_HOME=str(args.lerobot_home), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                      OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',
                      TMPDIR=str(ROOT / 'tmp/gen1_training'),
                      JAX_COMPILATION_CACHE_DIR=str(ROOT / 'tmp/gen1_training/jax'),
                      XLA_CLIENT_MEM_FRACTION='0.95',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false',
                      CUDA_VISIBLE_DEVICES='0' if args.mode == 'train' else '',
                      JAX_PLATFORMS='cuda' if args.mode == 'train' else 'cpu')
    sys.path[:0] = [str(OPENPI / 'src'), str(OPENPI / 'packages/openpi-client/src')]
    cfg = build_config(args)
    shim_source()  # Fail before any GPU work if the pinned loop's logging seam changed.
    files = [Path(__file__), OPENPI / 'scripts/train.py'] + [OPENPI / 'src/openpi/training' / f'{n}.py'
             for n in ('config', 'optimizer', 'checkpoints', 'data_loader', 'weight_loaders')]
    files += [SOURCE / 'params/_METADATA', SOURCE / 'assets/aloha_agilex_sim/norm_stats.json']
    report = {'run_id': str(uuid.uuid4()), 'created_utc': datetime.now(timezone.utc).isoformat(),
              'mode': args.mode, 'inputs': inputs, 'free_gib': free,
              'argv': sys.argv, 'python': sys.executable, 'source_revision': REVISION,
              'checkpoint': str(args.checkpoint_dir), 'provenance': str(args.provenance_dir),
              'fresh_optimizer': True, 'physical_action_dim': 14, 'full_parameter_training': True,
              'final_save_step': FINAL_STEP, 'periodic_saves': False,
              'save_concurrent_gb_per_handler': 8, 'start_free_gib': 115, 'save_free_gib': 112,
              'config': dataclasses.asdict(cfg),
              'versions': {name: importlib.metadata.version(name) for name in
                           ('jax', 'jaxlib', 'flax', 'orbax-checkpoint', 'lerobot', 'optax')},
              'sha256': {str(path): digest(path) for path in files},
              'environment': {key: os.environ[key] for key in ('HF_LEROBOT_HOME', 'JAX_PLATFORMS',
                              'CUDA_VISIBLE_DEVICES', 'XLA_CLIENT_MEM_FRACTION',
                              'XLA_PYTHON_CLIENT_PREALLOCATE')}}
    encoded = json.dumps(report, indent=2, default=str, allow_nan=False)
    print(encoded, flush=True)
    if args.mode == 'config':
        print('CONFIG_OK; no data readiness claim unless inputs.status=passed; no GPU or weights loaded', flush=True)
        return
    if args.mode == 'train':
        # Exclusive creation; provenance survives checkpoint initialization/save failures.
        args.provenance_dir.mkdir(exist_ok=False)
        with (args.provenance_dir / 'run_config.json').open('x') as stream:
            stream.write(encoded + '\n')
        for name, path in [('split_manifest.json', args.split_manifest),
                           ('train_conversion_audit.json', args.train_audit),
                           ('val_conversion_audit.json', args.val_audit)]:
            with (args.provenance_dir / name).open('xb') as stream:
                stream.write(path.read_bytes())
        disk_guard(args.checkpoint_dir.parent, 115)  # Immediately before actual GPU loader initialization.
    from openpi.training.data_loader import create_data_loader
    loader = create_data_loader(cfg, shuffle=False, num_batches=1)
    iterator = iter(loader)
    batch = next(iterator)
    finite_batch(batch)
    print('OPENPI_LOADER_OK finite actual training batch; actions=(1,50,32); heldout excluded', flush=True)
    del batch, iterator, loader
    gc.collect()
    if args.mode == 'train':
        print('FINAL_ONLY_SAVE=1999; ORBAX_SAVE_CONCURRENT_GB=8 per item; host staging remains unbounded', flush=True)
        run_train(cfg, args)


if __name__ == '__main__':
    main()
