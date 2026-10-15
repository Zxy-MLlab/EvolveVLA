#!/usr/bin/env python3
"""Validate expanded local + unchanged official batches before GPU training."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OPENPI = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi'
SPLIT = ROOT / 'data/gen1_multitask_coverage_expanded_20261014/split_manifest.json'
TRAIN = 'gen1_multitask_coverage_expanded_train_20261014'
VAL = 'gen1_multitask_coverage_val_20261012'
OFFICIAL = 'gen1_official_replay_train_20261013'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preflight output already exists')
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', HF_LEROBOT_HOME=str(ROOT/'data'),
                      HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      WANDB_MODE='disabled', OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    sys.path[:0] = [str(OPENPI/'src'), str(OPENPI/'packages/openpi-client/src')]
    import jax
    import numpy as np
    import gen1_multitask_train as base
    import gen1_replay_loader as replay
    from gen1_multitask_data import digest, require
    from gen1_multitask_video import install
    from openpi.training import sharding

    args = argparse.Namespace(repo_id=TRAIN, val_repo_id=VAL, lerobot_home=ROOT/'data',
        split_manifest=SPLIT, train_audit=ROOT/'data'/TRAIN/'conversion_audit.json',
        val_audit=ROOT/'data'/VAL/'conversion_audit.json',
        replay_repo_id=OFFICIAL, replay_manifest=ROOT/'outputs/gen1_official_replay_manifest_20261013.json',
        checkpoint_dir=ROOT/'checkpoints/gen1_replay_expanded_lora_b64_20261015',
        seed=0, data_seed=1, steps=24000, warmup_steps=100,
        learning_rate=1e-4, batch_size=64, log_interval=20)
    _, audits = base.validate_inputs(args)
    cfg = base.build_config(args)
    require(all(d.platform == 'cpu' for d in jax.devices()), 'GPU unexpectedly used')
    reader, video_report = install([ROOT/'data'/r for r in (TRAIN, VAL, OFFICIAL)])
    try:
        mesh = sharding.make_mesh(1)
        shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        loader, sampler, bindings = replay.build(cfg, args, audits['train']['rows'], shard)
        iterator = iter(loader)
        for _ in range(9):
            batch = next(iterator)
            require(batch[1].shape == (64, 50, 32), 'Wrong batch shape')
            require(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(batch)),
                    'Nonfinite mixed batch')
        require(bindings['local_train_tasks'] == sorted(replay.LOCAL_TRAIN_TASKS)
                and bindings['local_train_episodes'] == 1990
                and bindings['official_train_episodes'] == 900
                and bindings['total_train_episodes'] == 2890, 'Wrong training scope')
        require(len(sampler.counts) == 20 and all(n == (144 if source == 'local' else 16)
                for (task, source), n in sampler.counts.items()), 'Incorrect source/task allocation')
        anchors = base.validation_anchors(cfg, args, audits['val']['rows'])
        require(len(anchors) == 720, 'Original validation anchors changed')
        bindings['local_split_sha256'] = digest(SPLIT)
        report = dict(status='passed', batch_size=64, batches_checked=9, gpu_used=False,
                      local_train_episodes=bindings['local_train_episodes'],
                      local_converted_episodes=audits['train']['episodes'],
                      local_val_episodes=audits['val']['episodes'],
                      local_validation_anchors=len(anchors), sampler=sampler.report(),
                      configuration=bindings, video_reader=video_report,
                      norm_sha256=digest(base.SOURCE/'assets/aloha_agilex_sim/norm_stats.json'))
        a.output.write_text(json.dumps(report, indent=2) + '\n')
        print('EXPANDED_PREFLIGHT_PASSED', json.dumps(dict(output=str(a.output),
              batch_size=64, batches_checked=9, total_train_episodes=2890,
              local_validation_anchors=len(anchors), sampler=sampler.report())), flush=True)
    finally:
        reader.close()


if __name__ == '__main__':
    main()
