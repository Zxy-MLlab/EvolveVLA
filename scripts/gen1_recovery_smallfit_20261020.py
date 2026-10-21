#!/usr/bin/env python3
"""Independent suffix-only LoRA fitting diagnostic; no generalization claim.

prepare is CPU-only and never loads a full model unless --verify-restore is
explicitly requested. train is a separate action requiring a passed prepare
artifact and ready recovery dataset. The first version has no rehearsal.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np

SCRIPT = Path(__file__).resolve()
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0,str(SCRIPT.parent))
import gen1_recovery_suffix_dataset_20261020 as suffix
import gen1_task_routing_20261020 as routing

ROOT, OPENPI = routing.ROOT, routing.OPENPI
INITIAL = ROOT/'checkpoints/gen1_expanded_lora_step24000_export_20261015'
REFERENCE = ROOT/'outputs/gen1_timing_validation_20261020/candidate_policy_ab/manifest.json'
SAVE_STEPS = (100,300,600)
INITIAL_ID = '201fe2a2530f531f0a51a8e4526ac7cae3cc990eff60a7c28ca6712cf4664c8a'


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def environment(train=False):
    os.environ.update(CUDA_VISIBLE_DEVICES='0' if train else '', JAX_PLATFORMS='cuda' if train else 'cpu',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    os.environ.pop('XLA_PYTHON_CLIENT_MEM_FRACTION',None)
    sys.path[:0]=[str(OPENPI/'src'),str(OPENPI/'packages/openpi-client/src')]


def verified_initializer():
    spec=dict(checkpoint=str(INITIAL),identity_sha256=INITIAL_ID,
        identity_reference=str(REFERENCE),identity_reference_sha256=routing.digest(REFERENCE))
    identity=routing.verify_checkpoint(spec)
    if (INITIAL/'assets/aloha_agilex_sim/norm_stats.json').read_bytes()!=suffix.NORM.read_bytes():
        raise ValueError('Initializer normalization differs from fixed original norm')
    return identity


class SuffixDataset:
    """Raw OpenPI input seam with only complete, source-bounded 50-step windows."""
    def __init__(self, manifest_path):
        self.path=Path(manifest_path).resolve()
        self.manifest=suffix.read(self.path)
        m=self.manifest
        if (m.get('format')!='recovery_suffix_npz_v1' or m.get('operation')!='build'
                or m.get('dataset_ready') is not True or m.get('horizon')!=50
                or m.get('normalization',{}).get('sha256')!=suffix.digest(suffix.NORM)):
            raise ValueError('A built, ready, original-norm suffix dataset is required')
        groups=m.get('source_episode_splits')
        if (not isinstance(groups,dict) or len({r['parent_episode_uid'] for r in m['episodes']})!=len(m['episodes'])
                or groups!={r['parent_episode_uid']:r['split'] for r in m['episodes']}):
            raise ValueError('Source episode split mapping must match unique episode entries')
        self.rows, self.arrays=[],[]
        for row in m['episodes']:
            if row['split'] not in ('train','diagnostic_train'):
                continue
            if not row.get('dataset_ready') or row.get('valid_start_count',0)<1:
                raise ValueError('Unready suffix in training entries')
            for path,sha in row['evidence_sha256'].items():
                if suffix.digest(path)!=sha:
                    raise ValueError('Recovery replay evidence changed')
            # Revalidate upstream gates/raw captures before accepting exported
            # arrays; a ready flag alone is not the training authorization.
            source_arrays,verified=suffix.validate_case(row['source_readiness'],m['deployment_mode'],groups)
            if not verified.get('dataset_ready') or any(row.get(key)!=value for key,value in verified.items()):
                raise ValueError('Recovery upstream verification no longer matches')
            path=Path(row['dataset_path'])
            if suffix.digest(path)!=row['dataset_sha256']:
                raise ValueError('Suffix dataset archive changed')
            with np.load(path,allow_pickle=False) as source:
                data={key:source[key] for key in source.files}
            if set(data)!=set(source_arrays):
                raise ValueError('Exported archive fields differ from verified raw captures')
            for key in data:
                np.testing.assert_array_equal(data[key],source_arrays[key])
            n=len(data['action'])
            if data['action'].shape!=(n,14) or data['observation_state'].shape!=(n,14) or n<50:
                raise ValueError('Invalid physical state/action arrays')
            if row['valid_start_count']!=n-49 or row['valid_start_range']!=[0,n-49]:
                raise ValueError('Training window range differs from episode bounds')
            if not np.all(data['action_physics_tick']-data['observation_physics_tick']==15):
                raise ValueError('Training labels include a nonuniform capture interval')
            if not all(np.isfinite(data[k]).all() for k in ('action','observation_state')):
                raise ValueError('Nonfinite training labels')
            self.rows.append(dict(row,episode=len(self.rows)))
            self.arrays.append(data)
        if not self.rows:
            raise ValueError('No eligible training suffixes')
        groups={}
        for episode,row in enumerate(self.rows):
            groups.setdefault(row['parent_episode_uid'],[]).append(episode)
        self.groups=list(groups.values())

    def draw(self,rng):
        group=self.groups[rng.randrange(len(self.groups))]
        episode=group[rng.randrange(len(group))]
        return episode,rng.randrange(self.rows[episode]['valid_start_count'])

    def raw(self, episode, start):
        row,data=self.rows[episode],self.arrays[episode]
        if not 0<=start<row['valid_start_count']:
            raise IndexError('A full real window is required')
        item={'observation.state':data['observation_state'][start].copy(),
              'action':data['action'][start:start+50].copy(),'prompt':row['prompt']}
        # Native AlohaInputs consumes CHW; avoid video reencoding and preserve
        # exact uint8 pixels before the unchanged model resize/normalization.
        for camera in suffix.CAMERAS:
            image=data[camera][start]
            if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3:
                raise ValueError('Camera input must be uint8 RGB HWC')
            item['observation.images.'+camera]=image.transpose(2,0,1).copy()
        return item


class StrictFullLoraLoader:
    """Typed full-tree Orbax restore: never synthesize absent adapter weights."""
    def __init__(self, checkpoint):
        self.checkpoint=Path(checkpoint)
        self.report=None
        self.expected_hashes=None

    def load(self, template):
        import jax
        import orbax.checkpoint as ocp
        from flax.traverse_util import flatten_dict
        flat_template=flatten_dict(template,sep='/')
        with ocp.PyTreeCheckpointer() as loader:
            metadata=loader.metadata(self.checkpoint/'params')['params']
            flat_metadata=flatten_dict(metadata,sep='/')
            if flat_template.keys()!=flat_metadata.keys():
                raise ValueError('Full LoRA restore requires exact keys, including all adapter leaves')
            for key,value in flat_template.items():
                if tuple(value.shape)!=tuple(flat_metadata[key].shape):
                    raise ValueError(f'Full LoRA shape mismatch: {key}')
            if not any(key.endswith('lora_b') for key in flat_template):
                raise ValueError('No LoRA B leaves in model template')
            restored=loader.restore(self.checkpoint/'params',ocp.args.PyTreeRestore(
                item={'params':template},restore_args={'params':jax.tree.map(
                    lambda x:ocp.ArrayRestoreArgs(restore_type=np.ndarray,dtype=x.dtype),template)}))['params']
        actual=flatten_dict(restored,sep='/')
        if any(not np.isfinite(v).all() for v in actual.values()):
            raise ValueError('Nonfinite restored parameters')
        # This records initialized values in their intended training dtypes;
        # no lora_b reset or missing-regex merge is allowed.
        self.expected_hashes={k:hash_array(v) for k,v in actual.items()}
        self.report=dict(status='passed',typed_restore=True,exact_keys_and_shapes=True,
            lora_b_leaves=sum(k.endswith('lora_b') for k in actual),
            nonzero_lora_b_leaves=sum(bool(np.any(v)) for k,v in actual.items() if k.endswith('lora_b')),
            reset_lora_b=False,optimizer_restored=False,optimizer_restart=True)
        if self.report['nonzero_lora_b_leaves']==0:
            raise ValueError('Continued 24k LoRA unexpectedly has no nonzero B leaves')
        return restored


def hash_array(value):
    import hashlib
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def make_config(args, loader):
    import flax.nnx as nnx
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared.nnx_utils import PathRegex
    import gen1_multitask_train as base
    base_args=argparse.Namespace(checkpoint_dir=args.checkpoint_dir,seed=0,repo_id='recovery_suffix_npz_v1',
        learning_rate=1e-5,warmup_steps=20,steps=600,batch_size=args.batch_size,log_interval=20)
    cfg=base.build_config(base_args)
    model=Pi0Config(**suffix.read(INITIAL/'model_config.json'))
    return dataclasses.replace(cfg,model=model,freeze_filter=nnx.Not(PathRegex('.*lora.*')),
                               weight_loader=loader)


def transformations(cfg):
    from openpi import transforms as t
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    if not dc.use_quantile_norm or not dc.norm_stats:
        raise ValueError('Original quantile normalization required')
    forward=t.compose([*dc.repack_transforms.inputs,*dc.data_transforms.inputs,
        t.Normalize(dc.norm_stats,use_quantiles=True),*dc.model_transforms.inputs])
    inverse=t.compose([*dc.model_transforms.outputs,t.Unnormalize(dc.norm_stats,use_quantiles=True),
        *dc.data_transforms.outputs,*dc.repack_transforms.outputs])
    return forward,inverse


def anchors_and_preflight(dataset, forward, inverse):
    anchors=[]
    for episode,row in enumerate(dataset.rows):
        for start in sorted({0,(row['valid_start_count']-1)//2,row['valid_start_count']-1}):
            raw=dataset.raw(episode,start)
            expected=raw['action'].copy()
            transformed=forward(raw)
            if transformed['actions'].shape!=(50,32):
                raise ValueError('Native transforms must produce 50x32 actions')
            recovered=inverse(dict(state=np.array(transformed['state']),actions=np.array(transformed['actions'])))['actions']
            np.testing.assert_allclose(recovered,expected,atol=2e-5,rtol=2e-5)
            anchors.append(dict(row=row,index=start,frame=start,transformed=transformed,
                                expected=expected,valid=np.ones(50,dtype=bool)))
    return anchors


def code_paths():
    orbax_modules=('orbax.checkpoint._src.handlers.pytree_checkpoint_handler',
        'orbax.checkpoint._src.serialization.type_handlers',
        'orbax.checkpoint._src.checkpointers.pytree_checkpointer',
        'orbax.checkpoint._src.checkpointers.checkpointer')
    return [SCRIPT,Path(suffix.__file__),Path(routing.__file__),
        SCRIPT.with_name('gen1_replay_lora_train.py'),SCRIPT.with_name('gen1_multitask_train.py'),
        SCRIPT.with_name('gen1_replay_lora_export.py'),OPENPI/'scripts/train.py',
        OPENPI/'src/openpi/training/config.py',OPENPI/'src/openpi/transforms.py',
        OPENPI/'src/openpi/policies/aloha_policy.py',OPENPI/'src/openpi/models/model.py',
        OPENPI/'src/openpi/models/pi0.py',OPENPI/'src/openpi/models/pi0_config.py',
        OPENPI/'src/openpi/shared/nnx_utils.py',OPENPI/'src/openpi/shared/array_typing.py',
        OPENPI/'src/openpi/training/weight_loaders.py',OPENPI/'src/openpi/training/optimizer.py',
        OPENPI/'src/openpi/training/sharding.py',
        *sorted((OPENPI/'src/openpi/models').glob('*.py')),
        *[Path(importlib.import_module(name).__file__) for name in orbax_modules]]


def runtime_versions():
    return dict(python_executable=str(Path(sys.executable).resolve()),python_version=sys.version,
        packages={name:importlib.metadata.version(name) for name in
                  ('numpy','jax','jaxlib','flax','optax','orbax-checkpoint')})


def prepare(args):
    environment(False)
    dataset=SuffixDataset(args.dataset)
    identity=verified_initializer()
    loader=StrictFullLoraLoader(INITIAL)
    cfg=make_config(args,loader)
    forward,inverse=transformations(cfg)
    anchors=anchors_and_preflight(dataset,forward,inverse)
    import flax.nnx as nnx
    import jax
    from flax.traverse_util import flatten_dict
    abstract=nnx.state(nnx.eval_shape(cfg.model.create,jax.random.key(0)))
    trainable=flatten_dict(abstract.filter(cfg.trainable_filter).to_pure_dict(),sep='/')
    if not trainable or any('lora' not in key for key in trainable):
        raise ValueError('Unexpected trainable parameter set')
    count=sum(int(np.prod(v.shape)) for v in trainable.values())
    if args.verify_restore:
        # Mirror native init's bf16 frozen params, retaining trainable dtypes.
        import jax.numpy as jnp
        from openpi.shared import nnx_utils
        shape=nnx_utils.state_map(abstract,cfg.freeze_filter,
                                 lambda p:p.replace(jax.ShapeDtypeStruct(p.value.shape,jnp.bfloat16)))
        loaded=loader.load(shape.to_pure_dict())
        del loaded
    report=dict(status='passed',purpose='known-state smallfit positive control only',
        dataset=str(Path(args.dataset).resolve()),dataset_sha256=suffix.digest(args.dataset),
        initializer=identity,source=str(routing.BASELINE),norm_sha256=suffix.digest(suffix.NORM),
        source_groups=len({r['parent_episode_uid'] for r in dataset.rows}),suffixes=len(dataset.rows),
        real_valid_starts=sum(r['valid_start_count'] for r in dataset.rows),
        transformed_anchors=len(anchors),native_transform_roundtrip_passed=True,
        model=dataclasses.asdict(cfg.model),trainable_parameters=count,batch_size=args.batch_size,
        steps=600,save_steps=list(SAVE_STEPS),learning_rate=1e-5,warmup_steps=20,
        checkpoint_evaluation=dict(primary_step=600, intermediate_steps=[100,300],
            intermediate_purpose='saved training diagnostics only; never selected using development rollout',
            selection_rule='predeclared final 600; no best-loss or best-development-rollout selection'),
        rehearsal=False,optimizer_restart=True,reset_lora_b=False,gradient_accumulation=1,
        full_restore_checked=loader.report is not None,restore_report=loader.report,
        deployment_mode=dataset.manifest['deployment_mode'],gpu_used=False,training_started=False,
        resources=dict(adapter_and_adam_bytes_per_save=count*12,
            minimum_free_disk_gib=22,minimum_gpu_free_gib=32,
            expected_training_gpu_gib='Estimate only: batch64 ancestor peaked25.0GiB; batch<=16 must be measured.',
            cpu_restore_additional_ram_gib='Allow at least12GiB plus decoded RGB and model-shape tooling.'),
        code_sha256={str(p):suffix.digest(p) for p in code_paths()},
        runtime=runtime_versions(),
        limitation='No held-out recovery validation, no rehearsal and no claim of generalization or retention. Selection requires paired rollout.')
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'prepare.json',report)
    return report


def train(args):
    prepared=suffix.read(args.prepared)
    if (prepared['status']!='passed' or prepared['dataset_sha256']!=suffix.digest(args.dataset)
            or prepared['batch_size']!=args.batch_size or prepared.get('runtime')!=runtime_versions()):
        raise ValueError('Prepare artifact does not match training inputs')
    for path,sha in prepared['code_sha256'].items():
        if suffix.digest(path)!=sha:
            raise ValueError('Training source changed after prepare')
    if shutil.disk_usage(ROOT).free<22*2**30:
        raise ValueError('Need22GiB free disk including export/reserve')
    gpu=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                       capture_output=True,text=True,check=True,timeout=15)
    if float(gpu.stdout.strip())<32*1024:
        raise ValueError('Need32GiB free GPU memory; training was not started')
    environment(True)
    identity=verified_initializer()
    if identity['identity_sha256']!=prepared['initializer']['identity_sha256']:
        raise ValueError('Initializer changed after prepare')
    dataset=SuffixDataset(args.dataset)
    loader=StrictFullLoraLoader(INITIAL)
    cfg=make_config(args,loader)
    forward,inverse=transformations(cfg)
    anchors=anchors_and_preflight(dataset,forward,inverse)
    import jax
    import flax.nnx as nnx
    from openpi.training import sharding
    from openpi.models import model as model_lib
    import gen1_replay_lora_train as lora
    base=lora.base
    spec=importlib.util.spec_from_file_location('suffix_native_train',OPENPI/'scripts/train.py')
    native=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    args.output.mkdir(parents=True,exist_ok=False)
    args.checkpoint_dir.mkdir(parents=True,exist_ok=False)
    args.provenance_dir=args.output
    mesh=sharding.make_mesh(1)
    data_shard=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    state,state_shard=native.init_train_state(cfg,jax.random.key(0),mesh,resume=False)
    jax.block_until_ready(state)
    actual={k:hash_array(v) for k,v in lora.flat(state.params).items()}
    if actual!=loader.expected_hashes:
        raise ValueError('Initialized parameters differ from typed-restored 24k checkpoint')
    frozen=lora.hashes(state.params)
    initial_adapters=lora.hashes(state.params,True)
    write(args.output/'frozen_initial_sha256.json',frozen)
    write(args.output/'adapter_initial_sha256.json',initial_adapters)
    write(args.output/'restore_verification.json',loader.report)
    write(args.output/'run_config.json',dict(prepared,source=str(routing.BASELINE),
        initialization=str(INITIAL),model=dataclasses.asdict(cfg.model),norm_sha256=suffix.digest(suffix.NORM),
        training_started=True,full_restore_checked=True,prepare_sha256=suffix.digest(args.prepared)))
    step_fn=jax.jit(functools.partial(native.train_step,cfg),
        in_shardings=(replicated,state_shard,data_shard),out_shardings=(state_shard,replicated),donate_argnums=(1,))
    # Existing validation code samples explicit fixed noise in production bf16.
    # These anchors are TRAINING suffixes; files are marked accordingly below.
    def fit_metrics(step):
        metrics=base.validate_model(state,cfg,anchors,inverse,step,args.output,min(args.batch_size,8))
        path=args.output/f'validation_{step:04d}.json'
        record=suffix.read(path)
        record.update(scope='training_fit_only',held_out=False,generalization_evidence=False)
        path.write_text(json.dumps(record,indent=2)+'\n')
        return metrics
    fit_metrics(0)
    rng=random.Random(1)
    counts=[0]*len(dataset.rows)
    started=time.monotonic()
    with (args.output/'metrics.jsonl').open('x',buffering=1) as log:
        for step in range(1,601):
            items=[]
            for _ in range(args.batch_size):
                episode,start=dataset.draw(rng)
                counts[episode]+=1
                items.append(forward(dataset.raw(episode,start)))
            tree=jax.tree.map(lambda *xs:np.stack(xs),*items)
            batch=(model_lib.Observation.from_dict(tree),tree['actions'])
            batch=jax.device_put(batch,data_shard)
            with sharding.set_mesh(mesh):
                state,info=step_fn(jax.random.key(1),state,batch)
            values={k:float(v) for k,v in jax.device_get(info).items()}
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError('Nonfinite training update')
            values.update(step=step,elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(values)+'\n')
            if step<=3 or step%20==0:
                print('SMALLFIT_STEP',json.dumps(values),flush=True)
            if step in SAVE_STEPS:
                lora.save_adapter(state,cfg,args,step,frozen)
                fit_metrics(step)
    if lora.hashes(state.params)!=frozen or lora.hashes(state.params,True)==initial_adapters:
        raise ValueError('Frozen-base/adapters-update final verification failed')
    write(args.output/'completed.json',dict(status='completed',steps=600,frozen_parameters_unchanged=True,
        adapters_updated=True,per_suffix_samples=counts,actual_success_rate=None,held_out_validation=False,
        primary_checkpoint_step=600,intermediate_checkpoints_not_selected=True,
        note='Overfit diagnostic only; evaluate paired rollout before any deployment selection.',
        export_command=[str(INFERENCE_PYTHON),str(SCRIPT.with_name('gen1_replay_lora_export.py')),
            '--training',str(args.output),'--step','600','--output',str(ROOT/'checkpoints'/(args.output.name+'_step600_export'))]))


INFERENCE_PYTHON=routing.INFERENCE_PYTHON


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('operation',choices=('prepare','train'))
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--verify-restore',action='store_true')
    args=parser.parse_args()
    if not 1<=args.batch_size<=16:
        parser.error('Batch must be1..16')
    args.output=args.output.resolve()
    if args.output.parent!=ROOT/'outputs' or not args.output.name.isidentifier() or args.output.exists():
        parser.error('Use a fresh identifier-named direct child of outputs')
    args.checkpoint_dir=ROOT/'checkpoints'/args.output.name
    if args.checkpoint_dir.exists():
        parser.error('Checkpoint directory already exists')
    if args.operation=='train':
        if args.prepared is None:
            parser.error('train requires --prepared')
        return train(args)
    result=prepare(args)
    print(json.dumps({k:result[k] for k in ('status','source_groups','suffixes','real_valid_starts',
        'transformed_anchors','trainable_parameters','full_restore_checked','gpu_used')},indent=2))


if __name__=='__main__':
    main()
