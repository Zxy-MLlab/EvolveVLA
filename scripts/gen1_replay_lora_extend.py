#!/usr/bin/env python3
"""Continue the live 6k LoRA run to 24k, retaining adapters and Adam state."""
import argparse
import dataclasses
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import gen1_replay_lora_train as lora

base, ROOT, OPENPI = lora.base, lora.ROOT, lora.OPENPI
PREVIOUS = ROOT/'outputs/gen1_replay_lora_b32_20261014'
NAME = 'gen1_replay_lora_b32_extended24000_20261014'


@dataclasses.dataclass(frozen=True)
class ContinuationSchedule:
    start: int = 6000
    stop: int = 24000
    initial: float = 1e-5
    final: float = 1e-6

    def create(self):
        import jax.numpy as jnp
        return lambda count: self.final + (self.initial-self.final)*0.5*(
            1+jnp.cos(jnp.pi*jnp.clip((count-self.start)/(self.stop-self.start),0,1)))


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if raw[0] == 'Z' else raw[19]
    except FileNotFoundError:
        return None


def restore_payload(checkpoint, template):
    """Preserve the typed Optax/NNX tree; metadata-only restore loses types."""
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    with ocp.PyTreeCheckpointer() as loader:
        value = loader.restore(checkpoint/'adapter_state', ocp.args.PyTreeRestore(
            item=template, restore_args=jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray),template)))
        # Independently compare every saved leaf, including Adam moments/counters.
        metadata = loader.metadata(checkpoint/'adapter_state')
        raw = loader.restore(checkpoint/'adapter_state', ocp.args.PyTreeRestore(
            item=metadata, restore_args=jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray),metadata)))
    def canonical(tree):
        result = {}
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            parts=[]
            for key in path:
                part = getattr(key,'key',getattr(key,'idx',getattr(key,'name',None)))
                parts.append(str(part))
            result['/'.join(parts)] = leaf
        return result
    typed, untyped = canonical(value), canonical(raw)
    base.require(typed.keys()==untyped.keys(),'Optimizer restore structure changed')
    for key in typed:
        np.testing.assert_array_equal(typed[key],untyped[key])
    return value


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--wait-pid',type=int)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    queue=ROOT/'outputs'/f'{NAME}_queue_active'
    if not a.verify_only:
        base.require(a.wait_pid is not None,'Live predecessor PID required')
        previous_id=identity(a.wait_pid)
        base.require(previous_id is not None,'Predecessor not alive')
        command=Path(f'/proc/{a.wait_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        base.require('gen1_replay_lora_train.py' in command and PREVIOUS.name in command,
                     'Wrong predecessor process')
        queue.mkdir()
        dependencies=[Path(__file__),Path(lora.__file__),Path(base.__file__),
            ROOT/'scripts/gen1_replay_loader.py',ROOT/'scripts/gen1_replay_sampler.py',
            ROOT/'scripts/gen1_multitask_video.py',OPENPI/'scripts/train.py']
        digests={str(path):base.digest(path) for path in dependencies}
        base.write_new(queue/'plan.json',dict(status='waiting',training_pid=a.wait_pid,
            identity=previous_id, resume_step=6000,total_steps=24000,additional_steps=18000,
            batch_size=32,optimizer_restart=False,save_steps=[9000,12000,18000,24000],
            learning_rate='Continue from 1e-5 at step 6000, cosine to 1e-6 at 24000; no new warmup',
            sampling='Same task/source balance; fresh deterministic shuffle with data_seed=2',
            source_hashes=digests))
        print('WAITING_TO_CONTINUE',a.wait_pid,'6000 -> 24000',flush=True)
        while identity(a.wait_pid)==previous_id:
            time.sleep(15)
        base.require(all(base.digest(path)==sha for path,sha in digests.items()),'Dependency changed')
        done=base.read_json(PREVIOUS/'completed.json')
        base.require(done['status']=='completed' and done['steps']==6000 and
            done['frozen_weights_unchanged'],'Previous training incomplete')
    os.environ.update(CUDA_VISIBLE_DEVICES='' if a.verify_only else '0',
        JAX_PLATFORMS='cpu' if a.verify_only else 'cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
        HF_LEROBOT_HOME=str(ROOT/'data'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        JAX_COMPILATION_CACHE_DIR=str(ROOT/'tmp/gen1_training/jax'))
    os.environ.pop('XLA_PYTHON_CLIENT_MEM_FRACTION',None)
    sys.path[:0]=[str(OPENPI/'src'),str(OPENPI/'packages/openpi-client/src')]
    import flax.nnx as nnx
    import jax
    import numpy as np
    from openpi.training import sharding,optimizer
    args=argparse.Namespace(repo_id='gen1_multitask_coverage_train_20261012',
        val_repo_id='gen1_multitask_coverage_val_20261012',
        replay_repo_id='gen1_official_replay_train_20261013',lerobot_home=ROOT/'data',
        split_manifest=ROOT/'data/gen1_multitask_coverage_20261012/split_manifest.json',
        train_audit=ROOT/'data/gen1_multitask_coverage_train_20261012/conversion_audit.json',
        val_audit=ROOT/'data/gen1_multitask_coverage_val_20261012/conversion_audit.json',
        replay_manifest=ROOT/'outputs/gen1_official_replay_manifest_20261013.json',
        checkpoint_dir=ROOT/'checkpoints'/NAME,provenance_dir=ROOT/'outputs'/NAME,
        seed=0,data_seed=2,log_interval=20,batch_size=32,steps=24000,warmup_steps=100,
        learning_rate=1e-4,save_steps=[9000,12000,18000,24000])
    cfg=dataclasses.replace(lora.make_config(args),lr_schedule=ContinuationSchedule())
    abstract=nnx.state(nnx.eval_shape(cfg.model.create,jax.random.key(0)))
    trainable=abstract.filter(cfg.trainable_filter)
    base.require(all('lora' in key for key in lora.flat(trainable)),'Non-LoRA trainable weights')
    tx=optimizer.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None)
    template=dict(adapters=trainable.to_pure_dict(),opt_state=jax.eval_shape(tx.init,trainable),
                  step=jax.ShapeDtypeStruct((),np.int32))
    checkpoint=ROOT/'checkpoints'/('gen1_replay_lora_b32_smoke_20261014/3' if a.verify_only
                                  else 'gen1_replay_lora_b32_20261014/6000')
    restored=restore_payload(checkpoint,template)
    expected_step=3 if a.verify_only else 6000
    base.require(int(restored['step'])==expected_step,'Checkpoint step mismatch')
    counters=[int(v) for path,v in jax.tree_util.tree_flatten_with_path(restored['opt_state'])[0]
              if getattr(path[-1],'name',None)=='count']
    base.require(counters and all(v==expected_step for v in counters),'Adam/schedule counters reset')
    if a.verify_only:
        print('RESUME_VERIFIED',json.dumps(dict(step=expected_step,optimizer_counters=counters,
            adapter_and_optimizer_values_exact=True,typed_optimizer_restored=True,
            lr_at_6000=float(cfg.lr_schedule.create()(6000)),lr_at_24000=float(cfg.lr_schedule.create()(24000)))),flush=True)
        return
    old=base.read_json(PREVIOUS/'run_config.json')
    base.require(old['batch_size']==32 and old['steps']==6000 and old['model']==dataclasses.asdict(cfg.model),
                 'Previous configuration differs')
    base.require(old['norm_sha256']==base.digest(base.SOURCE/'assets/aloha_agilex_sim/norm_stats.json') and
                 old['split_sha256']==base.digest(args.split_manifest),'Source data/stats changed')
    _,audits=base.validate_inputs(args)
    base.require(base.free_gib()>20,'Need 20 GiB continuation/checkpoint/export reserve')
    base.fresh_child(args.provenance_dir,ROOT/'outputs').mkdir()
    base.fresh_child(args.checkpoint_dir,ROOT/'checkpoints').mkdir()
    base.write_new(args.provenance_dir/'run_config.json',dict(old,steps=24000,resumed_from=str(checkpoint),
        save_steps=args.save_steps,data_seed=2,optimizer_restart=False,
        learning_rate_schedule=dataclasses.asdict(cfg.lr_schedule),
        sampled_starts=24000*32,script_sha256=base.digest(Path(__file__))))
    mesh=sharding.make_mesh(1)
    data_shard=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    spec=importlib.util.spec_from_file_location('lora_extend_native',OPENPI/'scripts/train.py')
    native=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    state,state_shard=native.init_train_state(cfg,jax.random.key(0),mesh,resume=False)
    frozen=base.read_json(PREVIOUS/'frozen_initial_sha256.json')
    base.require(lora.hashes(state.params)==frozen,'Frozen initializer differs')
    state.params.replace_by_pure_dict(restored['adapters'])
    state=dataclasses.replace(state,opt_state=restored['opt_state'],step=restored['step'])
    state=jax.device_put(state,state_shard)
    jax.block_until_ready(state)
    base.write_new(args.provenance_dir/'frozen_initial_sha256.json',frozen)
    base.write_new(args.provenance_dir/'resume_verification.json',dict(status='passed',step=6000,
        optimizer_counters=counters,frozen_base_exact=True,adapter_and_optimizer_restore_exact=True))
    del restored
    from gen1_multitask_video import install
    import gen1_replay_loader as replay
    reader,video_report=install([args.lerobot_home/r for r in
                                 (args.repo_id,args.val_repo_id,args.replay_repo_id)])
    try:
        loader,sampler,bindings=replay.build(cfg,args,audits['train']['rows'],data_shard)
        base.require(bindings==base.read_json(PREVIOUS/'data_bindings.json'),'Mixed data binding changed')
        base.write_new(args.provenance_dir/'data_bindings.json',bindings)
        base.write_new(args.provenance_dir/'video_reader.json',video_report)
        iterator=iter(loader)
        anchors=base.validation_anchors(cfg,args,audits['val']['rows'])
        inverse=base.output_transform(cfg.data.create(cfg.assets_dirs,cfg.model))
        initial_panel=base.read_json(PREVIOUS/'validation_0000.json')
        base.write_new(args.provenance_dir/'validation_0000.json',initial_panel)
        initial=initial_panel['metrics']
        candidates=base.read_json(PREVIOUS/'selection.json')['candidates']
        step_fn=jax.jit(functools.partial(native.train_step,cfg),
            in_shardings=(replicated,state_shard,data_shard),out_shardings=(state_shard,replicated),donate_argnums=(1,))
        started=time.monotonic()
        with (args.provenance_dir/'metrics.jsonl').open('x',buffering=1) as log:
            for step in range(6001,24001):
                batch=next(iterator)
                base.require(batch[1].shape==(32,50,32),'Batch changed')
                with sharding.set_mesh(mesh):
                    state,info=step_fn(jax.random.key(1),state,batch)
                metrics={key:float(v) for key,v in jax.device_get(info).items()}
                base.require(all(np.isfinite(v) for v in metrics.values()),'Nonfinite update')
                log.write(json.dumps(dict(step=step,**metrics))+'\n')
                if step==6001 or step%20==0:
                    print('LORA_EXTENDED_STEP',step,metrics,'elapsed',time.monotonic()-started,flush=True)
                if step in args.save_steps:
                    lora.save_adapter(state,cfg,args,step,frozen)
                    validation=base.validate_model(state,cfg,anchors,inverse,step,args.provenance_dir,8)
                    score=sum(validation[t][k]/max(initial[t][k],1e-8)
                        for t in initial if t!='all' for k in initial[t])/(4*(len(initial)-1))
                    candidates.append(dict(step=step,score=score,checkpoint=str(args.checkpoint_dir/str(step))))
        base.write_new(args.provenance_dir/'selection.json',dict(candidates=candidates,
            selected=min(candidates,key=lambda row:(row['score'],row['step'])),
            rule='Same held-out MAE ratios; selection includes original and extended checkpoints'))
        base.write_new(args.provenance_dir/'completed.json',dict(status='completed',steps=24000,
            resumed_from_step=6000,optimizer_restart=False,sampler=sampler.report(),actual_success_rate=None))
    finally:
        reader.close()


if __name__=='__main__':
    main()
