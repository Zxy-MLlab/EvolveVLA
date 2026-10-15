#!/usr/bin/env python3
"""Action-expert fine-tuning with held-out sampled-action validation.

Uses the audited trajectory split, original normalization and official image
augmentation. Never trains on validation episodes. Saves inference weights only
in fresh directories; no optimizer archives or historical checkpoint overwrites.
"""
import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('/root/autodl-tmp/EvolveVLA')
OPENPI=ROOT/'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi'
sys.path.insert(0,str(ROOT/'scripts'))


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,default=str,allow_nan=False)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-root',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--full-parameters',action='store_true')
    p.add_argument('--augment-prompts',action='store_true')
    p.add_argument('--physical-loss',action='store_true')
    p.add_argument('--initial-checkpoint',type=Path,
                   help='Verified inference weights for continued training; optimizer restarts')
    p.add_argument('--data-seed',type=int,default=0)
    p.add_argument('--config-only',action='store_true')
    a=p.parse_args()
    os.environ.update(HF_LEROBOT_HOME=str(ROOT/'data'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORMS='cpu' if a.config_only else 'cuda',
        CUDA_VISIBLE_DEVICES='' if a.config_only else '0',
        JAX_COMPILATION_CACHE_DIR=str(ROOT/'tmp/gen1_training/jax'))
    sys.path[:0]=[str(OPENPI/'src'),str(OPENPI/'packages/openpi-client/src')]
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import orbax.checkpoint as ocp
    from openpi.shared import nnx_utils
    from openpi.models import model as model_lib
    from openpi.training import optimizer, sharding, data_loader
    import gen1_diagnose_pilot as diagnostic
    import gen1_train_pilot as pilot
    import gen1_recovery_eval as recovery
    args=diagnostic.defaults()
    validated=pilot.validate_inputs(args)
    cfg=pilot.build_config(args)
    initialization=pilot.SOURCE
    if a.initial_checkpoint is not None:
        from openpi.training import weight_loaders
        initialization=a.initial_checkpoint.resolve()
        verification=json.loads((initialization/'verification.json').read_text())
        if (verification['status']!='passed' or not verification['all_finite']
                or not verification['normalization_exact']):
            raise ValueError('Continuation checkpoint has not passed independent verification')
        if ((initialization/'assets/aloha_agilex_sim/norm_stats.json').read_bytes()
                != (pilot.SOURCE/'assets/aloha_agilex_sim/norm_stats.json').read_bytes()):
            raise ValueError('Continuation must use the original normalization')
        cfg=dataclasses.replace(cfg,weight_loader=weight_loaders.CheckpointWeightLoader(str(initialization/'params')))
    freeze=nnx.Any(nnx_utils.PathRegex('PaliGemma/img/.*'),
        nnx.All(nnx_utils.PathRegex('PaliGemma/llm/.*'),nnx.Not(nnx_utils.PathRegex('.*_1.*'))))
    if a.full_parameters:freeze=nnx.Nothing()
    if a.physical_loss:
        # Modify the local method only. The pinned source and sampling code stay
        # untouched. The velocity target stays native flow matching; supervise
        # physical joints and grippers rather than the 18 unused padded axes.
        import inspect
        import textwrap
        from openpi.models import pi0
        original=inspect.unwrap(pi0.Pi0.compute_loss)
        source=textwrap.dedent(inspect.getsource(original))
        source=source[source.index('def compute_loss'):]
        old='return jnp.mean(jnp.square(v_t - u_t), axis=-1)'
        if source.count(old)!=1:raise ValueError('Native loss seam changed')
        source=source.replace(old,
            'weights = jnp.array([1,1,1,1,1,1,2,1,1,1,1,1,1,2], dtype=jnp.float32)\n'
            '    return jnp.sum(jnp.square(v_t[..., :14] - u_t[..., :14]) * weights, axis=-1) / weights.sum()')
        scope=dict(vars(pi0));exec(compile(source,str(Path(__file__)), 'exec'),scope)
        pi0.Pi0.compute_loss=scope['compute_loss']
    cfg=dataclasses.replace(cfg,batch_size=a.batch_size,num_train_steps=a.steps,ema_decay=None,seed=a.data_seed,
        freeze_filter=freeze,lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100,
             peak_lr=a.learning_rate,decay_steps=a.steps,decay_lr=a.learning_rate/10),
        exp_name=a.checkpoint_root.name,checkpoint_dir_override=str(a.checkpoint_root))
    abstract=nnx.state(nnx.eval_shape(cfg.model.create,jax.random.key(0)))
    trainable=abstract.filter(cfg.trainable_filter).to_pure_dict()
    flat=ocp.tree.to_flat_dict(trainable)
    num_trainable=sum(np.prod(v.shape) for v in jax.tree.leaves(trainable))
    assert (2_000_000_000 < num_trainable < 4_000_000_000 if a.full_parameters
            else 200_000_000 < num_trainable < 600_000_000), num_trainable
    if not a.full_parameters:assert not any(k[0]=='PaliGemma' and k[1]=='img' for k in flat)
    report=dict(config=dataclasses.asdict(cfg),inputs=validated,trainable_parameters=int(num_trainable),
        trainable_paths=['/'.join(k) for k in flat],source=str(pilot.SOURCE),
        hypothesis=(('Full-parameter fine-tuning' if a.full_parameters else 'Action-expert fine-tuning')
                    + (' with episode-grounded prompt augmentation' if a.augment_prompts else ' with original prompts')
                    + (' and weighted physical-action loss.' if a.physical_loss else ' and native 32-dimensional loss.')),
        validation='80 fixed held-out anchors; explicit noise; production bfloat16 sampling; no evaluation-seed training',
        sha256={str(x):pilot.digest(x) for x in [Path(__file__),ROOT/'scripts/gen1_train_pilot.py',
            ROOT/'scripts/gen1_prompt_augmentation.py',ROOT/'scripts/gen1_video_cache.py',
            OPENPI/'scripts/train.py',args.split_manifest]},
        steps=a.steps,batch_size=a.batch_size,checkpoint_kind='raw_inference_only')
    report.update(full_parameters=a.full_parameters,augment_prompts=a.augment_prompts,
        physical_loss=a.physical_loss,loss_description='14 physical axes, grippers weight 2' if a.physical_loss else 'native 32-dimensional flow matching')
    report.update(initialization=str(initialization),optimizer_restarted=a.initial_checkpoint is not None,
                  data_seed=a.data_seed,initialization_provenance=(
                      json.loads((initialization/'provenance.json').read_text()) if a.initial_checkpoint else None))
    if a.config_only:
        print(json.dumps(report,default=str,indent=2));return
    if a.output.exists() or a.checkpoint_root.exists():raise FileExistsError('Fresh output roots required')
    expected_saves=(a.steps+a.eval_every-1)//a.eval_every
    if shutil.disk_usage(ROOT).free < (expected_saves*6+10)*2**30:
        raise RuntimeError('Insufficient space for planned inference checkpoints plus 10 GiB reserve')
    a.output.mkdir();a.checkpoint_root.mkdir()
    write(a.output/'run_config.json',report)
    recovery.release_checkpoint_cache()
    import gen1_video_cache
    cache_report=gen1_video_cache.install([ROOT/'data'/args.repo_id,ROOT/'data'/args.val_repo_id])
    write(a.output/'video_cache.json',cache_report)
    if a.augment_prompts:
        import gen1_prompt_augmentation
        gen1_prompt_augmentation.install(a.output)
    spec=importlib.util.spec_from_file_location('recovery_pinned_train',OPENPI/'scripts/train.py')
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    mesh=sharding.make_mesh(1)
    data_shard=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    loader=data_loader.create_data_loader(cfg,sharding=data_shard,shuffle=True)
    iterator=iter(loader)
    batch=next(iterator)
    state,state_shard=train.init_train_state(cfg,jax.random.key(0),mesh,resume=False)
    jax.block_until_ready(state)
    if a.initial_checkpoint is not None:
        restored=model_lib.restore_params(initialization/'params',restore_type=np.ndarray)
        loaded=ocp.tree.to_flat_dict(state.params.to_pure_dict())
        expected=ocp.tree.to_flat_dict(restored)
        if set(loaded)!=set(expected):raise ValueError('Continuation parameter paths differ')
        for key,value in expected.items():
            # All stored bfloat16 values are exactly representable in float32.
            if not np.array_equal(np.asarray(loaded[key]),value):
                raise ValueError(f'Continuation weights changed during loading: {key}')
        write(a.output/'initialization_weights.json',dict(status='passed',
             leaves=len(expected),all_loaded_weights_exact=True))
        del restored,loaded,expected
    def frozen_hashes(params):
        return {'/'.join(k):hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
                for k,v in ocp.tree.to_flat_dict(params.filter(cfg.freeze_filter).to_pure_dict()).items()}
    frozen_before=frozen_hashes(state.params)
    write(a.output/'frozen_initial_sha256.json',frozen_before)
    step_fn=jax.jit(functools.partial(train.train_step,cfg),
        in_shardings=(replicated,state_shard,data_shard),out_shardings=(state_shard,replicated),donate_argnums=(1,))

    manifest=json.loads((ROOT/'outputs/gen1_diagnostic_prepare_cpu_v3_20261011/manifest.json').read_text())
    anchors=[r for r in manifest['anchors'] if r['split']=='val']
    assert len(anchors)==80
    bundle=diagnostic.dataset_bundle(cfg,args.val_repo_id)
    _,inverse,_,_=diagnostic.transforms_for(bundle[0])
    validation=[]
    for anchor in anchors:
        raw,transformed,expected,_,_=diagnostic.checked_anchor(bundle,anchor)
        validation.append((anchor,transformed,expected))
    # Fixed batch of eight anchors, independent of training batch size.
    def sample(params,obs,noise):
        model=nnx.merge(state.model_def,params)
        model.eval()
        return model.sample_actions(jax.random.key(0),obs,noise=noise,num_steps=10)
    sample_fn=jax.jit(sample)
    val_noise=jax.random.normal(jax.random.key(1729),(8,50,32))
    def evaluate(step):
        records=[]
        # Materialize production weights outside the compiled sampling graph.
        # This matches loading persisted bfloat16 arrays; a cast within a larger
        # XLA graph can be fused differently from inference checkpoint loading.
        production_params=jax.tree.map(lambda x:x.astype(jnp.bfloat16),state.params)
        jax.block_until_ready(production_params)
        for start in range(0,len(validation),8):
            selected=validation[start:start+8]
            batch_dict=jax.tree.map(lambda *xs:jnp.asarray(np.stack(xs)),*[r[1] for r in selected])
            obs=model_lib.Observation.from_dict(batch_dict)
            predicted=np.asarray(sample_fn(production_params,obs,val_noise))
            for (anchor,transformed,expected),prediction in zip(selected,predicted):
                actual=inverse(dict(state=np.array(transformed['state']),actions=prediction.copy()))['actions']
                err=np.abs(actual-expected)
                valid=np.asarray(anchor['valid'])
                joints=[i for i in range(14) if i not in (6,13)]
                records.append(dict(task=anchor['task'],episode=anchor['episode'],frame=anchor['frame'],
                    joint_mae=float(err[valid][:,joints].mean()),gripper_mae=float(err[valid][:,[6,13]].mean()),
                    first10_joint_mae=float(err[:10,joints].mean()),first10_gripper_mae=float(err[:10,[6,13]].mean())))
        metrics={}
        for task in ('hanging_mug','place_mouse_pad','all'):
            rows=[r for r in records if task=='all' or r['task']==task]
            metrics[task]={key:float(np.mean([r[key] for r in rows])) for key in
                ('joint_mae','gripper_mae','first10_joint_mae','first10_gripper_mae')}
        write(a.output/f'validation_{step:04d}.json',dict(step=step,metrics=metrics,records=records))
        print('VALIDATION',step,json.dumps(metrics),flush=True)
        return metrics

    def save(step):
        root=a.checkpoint_root/str(step)
        root.mkdir()
        shutil.copytree(pilot.SOURCE/'assets',root/'assets')
        # Persist exactly the bfloat16 weights used by validation/production.
        params=jax.tree.map(lambda x:x.astype(jnp.bfloat16),state.params.to_pure_dict())
        # Largest stacked bfloat16 attention tensor is 2.42 GB; Orbax's byte
        # limiter must admit at least one whole tensor, not merely a shard.
        with ocp.Checkpointer(ocp.PyTreeCheckpointHandler(save_concurrent_gb=8)) as saver:
            saver.save(root/'params',{'params':params})
        write(root/'provenance.json',dict(step=step,source=str(pilot.SOURCE),kind='raw',
            initialization=str(initialization),optimizer_restarted=a.initial_checkpoint is not None,
            trainable_parameters=int(num_trainable),run=str(a.output),production_dtype='bfloat16'))
        print('SAVED',root,flush=True)
    initial_metrics=evaluate(0)
    if a.initial_checkpoint is not None:
        provenance=json.loads((initialization/'provenance.json').read_text())
        previous_validation=Path(provenance['run'])/f"validation_{provenance['step']:04d}.json"
        expected_metrics=json.loads(previous_validation.read_text())['metrics']
        max_difference=max(abs(initial_metrics[task][key]-value)
                           for task,metrics in expected_metrics.items() for key,value in metrics.items())
        write(a.output/'initialization_parity.json',dict(status='exact_weights_checked_separately',
             previous_validation=str(previous_validation),max_absolute_metric_difference=max_difference,
             note='Prior validation cast inside JIT; current validation materializes saved bfloat16 weights before sampling. Use current step zero as the continuation ancestor metric.'))
    started=time.monotonic()
    with (a.output/'metrics.jsonl').open('x',buffering=1) as stream:
        for step in range(1,a.steps+1):
            with sharding.set_mesh(mesh):state,info=step_fn(jax.random.key(1),state,batch)
            values={k:float(v) for k,v in jax.device_get(info).items()}
            if not all(np.isfinite(v) for v in values.values()):raise RuntimeError('Nonfinite training metric')
            stream.write(json.dumps(dict(step=step,**values))+'\n')
            if step%50==0:print('STEP',step,'loss',values['loss'],'elapsed',round(time.monotonic()-started),flush=True)
            if step%a.eval_every==0 or step==a.steps:
                evaluate(step);save(step)
            if step<a.steps:batch=next(iterator)
    # Compare hashes without retaining/copying full host trees.
    frozen_after=frozen_hashes(state.params)
    assert frozen_before==frozen_after, 'Frozen backbone changed'
    if a.augment_prompts:
        write(a.output/'sampling_coverage.json',gen1_prompt_augmentation.sampling_report())
    write(a.output/'completed.json',dict(status='completed',steps=a.steps,
         checkpoints=[str(x) for x in a.checkpoint_root.iterdir()],frozen_backbone_unchanged=True))


if __name__=='__main__':
    logging.basicConfig(level=logging.WARNING)
    main()
