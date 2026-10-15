#!/usr/bin/env python3
"""Isolated adapter-only RoboTwin replay training; preserves earlier run code."""
import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import gen1_multitask_train as base

ROOT, OPENPI = base.ROOT, base.OPENPI


def flat(params):
    from flax.traverse_util import flatten_dict
    return flatten_dict(params.to_pure_dict(), sep='/')


def hashes(params, adapters=False):
    import numpy as np
    return {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
            for k, v in flat(params).items() if ('lora' in k) == adapters}


def make_config(args):
    import flax.nnx as nnx
    from openpi.shared.nnx_utils import PathRegex
    cfg = base.build_config(args)
    model = dataclasses.replace(cfg.model, paligemma_variant='gemma_2b_lora',
                                action_expert_variant='gemma_300m_lora')
    # The upstream convenience filter leaves vision/projections trainable.
    # This experiment deliberately trains adapters ONLY.
    return dataclasses.replace(cfg, model=model, freeze_filter=nnx.Not(PathRegex('.*lora.*')))


def zero_b(state):
    """Use random A / zero B so adding adapters preserves the initializer."""
    import jax.numpy as jnp
    from flax.traverse_util import unflatten_dict
    values = flat(state.params)
    count = 0
    for name in values:
        if name.endswith('lora_b'):
            values[name] = jnp.zeros_like(values[name])
            count += 1
    base.require(count > 0, 'No LoRA B weights found')
    state.params.replace_by_pure_dict(unflatten_dict(values, sep='/'))
    return count


def save_adapter(state, cfg, args, step, frozen):
    import jax
    import orbax.checkpoint as ocp
    base.require(hashes(state.params) == frozen, 'Frozen base weights changed')
    base.require(base.free_gib() >= 14, 'Checkpoint reserve exhausted')
    target = args.checkpoint_dir / str(step)
    target.mkdir()
    payload = dict(adapters=state.params.filter(cfg.trainable_filter).to_pure_dict(),
                   opt_state=state.opt_state, step=state.step)
    with ocp.Checkpointer(ocp.PyTreeCheckpointHandler()) as saver:
        saver.save(target / 'adapter_state', payload)
        restored = saver.restore(target / 'adapter_state')
    import numpy as np
    from flax.traverse_util import flatten_dict
    original = flatten_dict(payload['adapters'], sep='/')
    recovered = flatten_dict(restored['adapters'], sep='/')
    base.require(original.keys() == recovered.keys(), 'Adapter restore keys changed')
    for name in original:
        np.testing.assert_array_equal(np.asarray(original[name]), np.asarray(recovered[name]))
    base.write_new(target / 'provenance.json', dict(step=step, source=str(base.SOURCE),
        kind='lora_adapter_with_optimizer', model=dataclasses.asdict(cfg.model),
        norm_sha256=base.digest(base.SOURCE/'assets/aloha_agilex_sim/norm_stats.json'),
        frozen_parameters_unchanged=True, adapter_restore_exact=True,
        run=str(args.provenance_dir), batch_size=args.batch_size))
    print('ADAPTER_SAVED', step, flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mode', choices=['smoke', 'train'], required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--save-steps', type=int, nargs='+', default=[250, 1000, 3000, 6000])
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--run-name', required=True)
    p.add_argument('--split-manifest', type=Path)
    p.add_argument('--train-repo-id')
    p.add_argument('--val-repo-id')
    args = p.parse_args()
    base.require(args.run_name.isidentifier(), 'Unsafe run name')
    base.require(args.batch_size > 8 and args.steps > args.warmup_steps > 0, 'Invalid schedule')
    base.require(sorted(set(args.save_steps)) == args.save_steps and
                 args.save_steps[0] > 0 and args.save_steps[-1] == args.steps, 'Invalid saves')
    args.repo_id = args.train_repo_id or 'gen1_multitask_coverage_train_20261012'
    args.val_repo_id = args.val_repo_id or 'gen1_multitask_coverage_val_20261012'
    args.replay_repo_id = 'gen1_official_replay_train_20261013'
    args.lerobot_home = ROOT/'data'
    args.split_manifest = args.split_manifest or ROOT/'data/gen1_multitask_coverage_20261012/split_manifest.json'
    args.train_audit = args.lerobot_home/args.repo_id/'conversion_audit.json'
    args.val_audit = args.lerobot_home/args.val_repo_id/'conversion_audit.json'
    args.replay_manifest = ROOT/'outputs/gen1_official_replay_manifest_20261013.json'
    args.checkpoint_dir = base.fresh_child(ROOT/'checkpoints'/args.run_name, ROOT/'checkpoints')
    args.provenance_dir = base.fresh_child(ROOT/'outputs'/args.run_name, ROOT/'outputs')
    args.seed, args.data_seed, args.log_interval = 0, 1, 20
    expanded = args.split_manifest != ROOT/'data/gen1_multitask_coverage_20261012/split_manifest.json'
    preflight_path = (ROOT/'outputs/gen1_replay_expanded_loader_preflight_20261014.json' if expanded
                      else ROOT/'outputs/gen1_replay_loader_preflight_20261013.json')
    preflight = base.read_json(preflight_path)
    base.require(preflight['status'] == 'passed', 'Mixed loader gate missing')
    if expanded:
        base.require(preflight.get('batch_size') == args.batch_size,
                     'Training batch size differs from the actual mixed preflight')
        base.require(preflight['configuration'].get('local_train_tasks') == ['hanging_mug', 'place_mouse_pad']
                     and preflight['configuration'].get('total_train_episodes') == 2890,
                     'Preflight does not reflect the requested two-local-task scope')
        for key, filename in [('loader_sha256', 'gen1_replay_loader.py'),
                              ('sampler_sha256', 'gen1_replay_sampler.py')]:
            base.require(preflight['configuration'].get(key) == base.digest(ROOT/'scripts'/filename),
                         'Mixed data code changed since preflight')
    base.require(preflight['configuration']['official_manifest_sha256'] == base.digest(args.replay_manifest)
                 and preflight['configuration'].get('local_split_sha256',base.digest(args.split_manifest))
                 == base.digest(args.split_manifest),
                 'Preflight manifest changed')
    os.environ.update(HF_LEROBOT_HOME=str(args.lerobot_home), HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
        OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        JAX_PLATFORMS='cuda', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        JAX_COMPILATION_CACHE_DIR=str(ROOT/'tmp/gen1_training/jax'))
    os.environ.pop('XLA_PYTHON_CLIENT_MEM_FRACTION', None)
    sys.path[:0] = [str(OPENPI/'src'), str(OPENPI/'packages/openpi-client/src')]
    import flax.nnx as nnx
    import jax
    import numpy as np
    from openpi.training import sharding
    import gen1_replay_loader as replay
    from gen1_multitask_video import install
    manifest, audits = base.validate_inputs(args)
    cfg = make_config(args)
    abstract = nnx.state(nnx.eval_shape(cfg.model.create, jax.random.key(args.seed)))
    trainable = flat(abstract.filter(cfg.trainable_filter))
    base.require(trainable and all('lora' in k for k in trainable), 'Non-adapter parameters trainable')
    count = sum(int(np.prod(v.shape)) for v in trainable.values())
    # Float32 adapters + two Adam moments at each save, final full bf16 model,
    # plus room for caches, evaluation outputs, and emergency reserve.
    budget = count*12*len(args.save_steps)/2**30 + 7 + 10
    base.require(args.mode != 'train' or base.free_gib() >= budget,
                 f'Need {budget:.2f} GiB checkpoint/reserve budget')
    args.provenance_dir.mkdir()
    args.checkpoint_dir.mkdir()
    report = dict(mode=args.mode, batch_size=args.batch_size, gradient_accumulation=1,
        trainable_parameters=count, total_parameters=sum(int(np.prod(v.shape)) for v in flat(abstract).values()),
        training='LoRA only; all non-adapter weights frozen', paligemma_rank=16, action_expert_rank=32,
        source=str(base.SOURCE), model=dataclasses.asdict(cfg.model), learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps, steps=args.steps, sampled_starts=args.steps*args.batch_size,
        save_steps=args.save_steps, checkpoint_budget_gib=budget, free_gib=base.free_gib(),
        norm_sha256=base.digest(base.SOURCE/'assets/aloha_agilex_sim/norm_stats.json'),
        split_sha256=base.digest(args.split_manifest), preflight_sha256=base.digest(preflight_path),
        scripts={str(f):base.digest(f) for f in (Path(__file__), Path(base.__file__),
            ROOT/'scripts/gen1_replay_loader.py',ROOT/'scripts/gen1_replay_sampler.py', OPENPI/'scripts/train.py')},
        limitation='Replay, LoRA and batch changes are bundled; this is not an augmentation-only ablation')
    base.write_new(args.provenance_dir/'run_config.json', report)
    print('LORA_CONFIG', json.dumps(report), flush=True)
    reader, video_report = install([args.lerobot_home/r for r in
                                   (args.repo_id, args.val_repo_id, args.replay_repo_id)])
    try:
        base.write_new(args.provenance_dir/'video_reader.json', video_report)
        mesh = sharding.make_mesh(1)
        data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        loader, sampler, bindings = replay.build(cfg, args, audits['train']['rows'], data_shard)
        base.write_new(args.provenance_dir/'data_bindings.json', bindings)
        iterator = iter(loader)
        spec = importlib.util.spec_from_file_location('replay_lora_native_train', OPENPI/'scripts/train.py')
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        state, state_shard = native.init_train_state(cfg, jax.random.key(args.seed), mesh, resume=False)
        jax.block_until_ready(state)
        zero_b(state)
        frozen = hashes(state.params)
        initial_adapters = hashes(state.params, True)
        base.write_new(args.provenance_dir/'frozen_initial_sha256.json', frozen)
        base.write_new(args.provenance_dir/'adapter_initial_sha256.json', initial_adapters)
        step_fn = jax.jit(functools.partial(native.train_step, cfg),
            in_shardings=(replicated, state_shard, data_shard),
            out_shardings=(state_shard, replicated), donate_argnums=(1,))
        if args.mode == 'train':
            anchors = base.validation_anchors(cfg,args,audits['val']['rows'])
            inverse = base.output_transform(cfg.data.create(cfg.assets_dirs,cfg.model))
            initial = base.validate_model(state,cfg,anchors,inverse,0,args.provenance_dir,8)
        started = time.monotonic()
        candidates = []
        with (args.provenance_dir/'metrics.jsonl').open('x', buffering=1) as metrics:
            for step in range(1, (3 if args.mode == 'smoke' else args.steps)+1):
                batch = next(iterator)
                base.require(batch[1].shape == (args.batch_size,50,32), 'Invalid batch')
                if step == 1:
                    base.require(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(batch)),
                                 'Nonfinite input batch')
                with sharding.set_mesh(mesh):
                    state, info = step_fn(jax.random.key(args.seed+1),state,batch)
                values = {k:float(v) for k,v in jax.device_get(info).items()}
                base.require(all(np.isfinite(v) for v in values.values()), 'Nonfinite training update')
                memory = jax.devices()[0].memory_stats()
                values.update(step=step,elapsed_seconds=time.monotonic()-started,
                              peak_device_bytes=memory.get('peak_bytes_in_use'))
                metrics.write(json.dumps(values)+'\n')
                if step <= 3 or step % 20 == 0:
                    print('LORA_STEP',json.dumps(values),flush=True)
                if args.mode == 'train' and step in args.save_steps:
                    save_adapter(state,cfg,args,step,frozen)
                    val = base.validate_model(state,cfg,anchors,inverse,step,args.provenance_dir,8)
                    score = sum(val[t][k]/max(initial[t][k],1e-8)
                        for t in initial if t != 'all' for k in initial[t])/(4*(len(initial)-1))
                    candidates.append(dict(step=step,score=score,checkpoint=str(args.checkpoint_dir/str(step))))
        base.require(hashes(state.params) == frozen, 'Frozen weights changed')
        base.require(hashes(state.params,True) != initial_adapters, 'Adapters did not update')
        if args.mode == 'smoke':
            save_adapter(state,cfg,args,3,frozen)
        else:
            selected = min(candidates,key=lambda x:(x['score'],x['step']))
            base.write_new(args.provenance_dir/'selection.json',dict(selected=selected,candidates=candidates,
                rule='Same held-out MAE ratios as prior run; no rollout selection'))
        base.write_new(args.provenance_dir/'completed.json',dict(status='passed' if args.mode=='smoke' else 'completed',
            steps=step, frozen_weights_unchanged=True, adapters_updated=True,
            sampler=sampler.report(), peak_device_bytes=memory.get('peak_bytes_in_use'),
            actual_success_rate=None, note='No closed-loop success evidence from this training process'))
        print('LORA_FINISHED',args.mode,flush=True)
    finally:
        reader.close()


if __name__ == '__main__':
    main()
