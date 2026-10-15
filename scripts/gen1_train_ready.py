#!/usr/bin/env python3
"""External, fail-closed pinned OpenPI smoke launcher. Default is CPU config check."""
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OPENPI = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi'
CKPT = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mode', choices=['config', 'loader', 'train', 'pinned-help'], default='config')
    p.add_argument('--repo-id', required=True)
    p.add_argument('--lerobot-home', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--coordinator-approved-gpu', action='store_true')
    a = p.parse_args()
    if a.mode == 'train' and not a.coordinator_approved_gpu:
        p.error('GPU training requires explicit coordinator approval after collection stops')
    if a.checkpoint_dir.exists():
        p.error('checkpoint target must not exist; resume and overwrite are forbidden')
    if not a.checkpoint_dir.resolve().is_relative_to(ROOT / 'checkpoints'):
        p.error('checkpoint target must be under EvolveVLA/checkpoints')
    if shutil.disk_usage(ROOT).free < 60 * 2**30:
        p.error('less than 60GiB free')
    os.environ.update(HF_LEROBOT_HOME=str(a.lerobot_home.resolve()), HF_HUB_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                      OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',
                      TMPDIR=str(ROOT / 'tmp/gen1_training'),
                      JAX_COMPILATION_CACHE_DIR=str(ROOT / 'tmp/gen1_training/jax'))
    if a.mode != 'train':
        os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
    else:
        os.environ.update(CUDA_VISIBLE_DEVICES='0', JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    sys.path[:0] = [str(OPENPI / 'src'), str(OPENPI / 'packages/openpi-client/src')]
    from openpi.training import config, weight_loaders
    cfg = config.get_config('pi05_base_aloha_full_sim_arx-x5_seed_0')
    cfg = dataclasses.replace(cfg, exp_name=a.checkpoint_dir.name,
        model=dataclasses.replace(cfg.model, action_dim=32, action_horizon=50, pi05=True),
        data=dataclasses.replace(cfg.data, repo_id=a.repo_id, adapt_to_pi=False, use_delta_joint_actions=True,
            assets=config.AssetsConfig(assets_dir=str(CKPT / 'assets'), asset_id='aloha_agilex_sim'),
            base_config=dataclasses.replace(cfg.data.base_config, video_backend='pyav')),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(CKPT / 'params')),
        checkpoint_dir_override=str(a.checkpoint_dir.resolve()), batch_size=1, num_workers=0,
        fsdp_devices=1, num_train_steps=2, log_interval=1, save_interval=1,
        resume=False, overwrite=False, wandb_enabled=False)
    if not (CKPT / 'params/_METADATA').exists():
        raise FileNotFoundError(CKPT / 'params/_METADATA')
    if a.mode == 'pinned-help':
        sys.argv = [str(OPENPI / 'scripts/train.py'), cfg.name, '--help']
        config.cli()
        return
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert dc.norm_stats and dc.use_quantile_norm and not cfg.data.adapt_to_pi
    print(json.dumps({'mode': a.mode, 'checkpoint': str(cfg.checkpoint_dir), 'weight_loader': str(cfg.weight_loader),
                      'model': str(cfg.model), 'data': str(cfg.data), 'fresh_optimizer': True,
                      'resume': cfg.resume, 'overwrite': cfg.overwrite, 'batch_size': cfg.batch_size}, indent=2))
    if a.mode == 'config':
        return
    if not (a.lerobot_home / a.repo_id / 'conversion_audit.json').exists():
        raise RuntimeError('Missing successful conversion audit for this newly collected dataset')
    from openpi.training.data_loader import create_data_loader
    import jax
    import numpy as np
    batch = next(iter(create_data_loader(cfg, shuffle=False, num_batches=1)))
    for x in jax.tree.leaves(batch):
        assert np.isfinite(np.asarray(x)).all()
    assert batch[1].shape == (1, 50, 32), batch[1].shape
    print('OPENPI_LOADER_OK', jax.tree.map(lambda x: list(x.shape), batch))
    if a.mode == 'train':
        spec = importlib.util.spec_from_file_location('gen1_pinned_train', OPENPI / 'scripts/train.py')
        train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train)
        import orbax.checkpoint as ocp
        from etils import epath

        def initialize_bounded_checkpoint_dir(checkpoint_dir, *, keep_period, overwrite, resume):
            if overwrite or resume:
                raise ValueError('Bounded smoke saving requires a fresh checkpoint directory')
            directory = epath.Path(checkpoint_dir).resolve()
            directory.mkdir(parents=True, exist_ok=False)
            manager = ocp.CheckpointManager(
                directory,
                item_handlers={
                    'assets': train._checkpoints.CallbackHandler(),
                    'train_state': ocp.PyTreeCheckpointHandler(save_concurrent_gb=8),
                    'params': ocp.PyTreeCheckpointHandler(save_concurrent_gb=8),
                },
                options=ocp.CheckpointManagerOptions(
                    max_to_keep=1, keep_period=keep_period, create=False,
                    async_options=ocp.AsyncOptions(timeout_secs=7200)),
            )
            return manager, False

        print('ORBAX_SAVE_CONCURRENT_GB=8 per item; whole-tree host staging remains unbounded', flush=True)
        train._checkpoints.initialize_checkpoint_dir = initialize_bounded_checkpoint_dir
        train.main(cfg)

if __name__ == '__main__':
    main()
