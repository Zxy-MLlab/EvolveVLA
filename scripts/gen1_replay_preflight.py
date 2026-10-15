"""CPU-only real mixed batches and official episode boundary roundtrips."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

ROOT = Path('/root/autodl-tmp/EvolveVLA')
OPENPI = ROOT/'robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi'


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh output required')
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
        HF_LEROBOT_HOME=str(ROOT/'data'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',
        OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    sys.path[:0]=[str(OPENPI/'src'),str(OPENPI/'packages/openpi-client/src')]
    import jax
    import numpy as np
    import pandas as pd
    import gen1_multitask_train as trainer
    import gen1_replay_loader as replay
    from gen1_multitask_data import require,digest
    from gen1_multitask_video import install
    from openpi.training import data_loader,sharding
    args=argparse.Namespace(repo_id='gen1_multitask_coverage_train_20261012',
        val_repo_id='gen1_multitask_coverage_val_20261012',lerobot_home=ROOT/'data',
        split_manifest=ROOT/'data/gen1_multitask_coverage_20261012/split_manifest.json',
        train_audit=ROOT/'data/gen1_multitask_coverage_train_20261012/conversion_audit.json',
        val_audit=ROOT/'data/gen1_multitask_coverage_val_20261012/conversion_audit.json',
        checkpoint_dir=ROOT/'checkpoints/gen1_replay_proposed_20261013',seed=0,data_seed=1,
        steps=24000,warmup_steps=100,learning_rate=3e-6,batch_size=8,log_interval=20,
        replay_repo_id='gen1_official_replay_train_20261013',
        replay_manifest=ROOT/'outputs/gen1_official_replay_manifest_20261013.json')
    _,audits=trainer.validate_inputs(args)
    cfg=trainer.build_config(args)
    require(all(d.platform=='cpu' for d in jax.devices()),'CPU preflight required')
    conversion=json.loads((args.lerobot_home/args.replay_repo_id/'conversion_audit.json').read_text())
    require(conversion['status']=='passed','Full conversion audit missing')
    reader,video_report=install([args.lerobot_home/r for r in
        (args.repo_id,args.val_repo_id,args.replay_repo_id)])
    try:
        mesh=sharding.make_mesh(1)
        shard=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        loader,sampler,report=replay.build(cfg,args,audits['train']['rows'],shard)
        iterator=iter(loader)
        for _ in range(9):
            batch=next(iterator)
            require(batch[1].shape==(8,50,32),'Wrong mixed batch shape')
            require(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(batch)),
                    'Nonfinite mixed batch')
        # 72 draws = two complete task/source cycles (18 tasks * 2 sources).
        require(set(sampler.counts.values())=={2},'Incorrect actual mixed sampling allocation')
        manifest=json.loads(args.replay_manifest.read_text())
        dc=dataclasses.replace(cfg.data,repo_id=args.replay_repo_id).create(cfg.assets_dirs,cfg.model)
        raw=data_loader.create_torch_dataset(dc,50,cfg.model)
        transformed=data_loader.transform_dataset(raw,dc)
        inverse=trainer.output_transform(dc)
        frames=pd.read_parquet(args.lerobot_home/args.replay_repo_id/'data').sort_values('index')
        action=np.stack(frames.action)
        checks=0
        for row in manifest['episodes']:
            for frame in sorted({0,row['frames']//2,row['frames']-1}):
                index=row['start']+frame
                item=raw[index]
                indices=np.minimum(index+np.arange(50),row['stop']-1)
                expected=action[indices]
                np.testing.assert_array_equal(np.asarray(item['action']),expected)
                np.testing.assert_array_equal(np.asarray(item['action_is_pad']),
                                              frame+np.arange(50)>=row['frames'])
                require(int(item['episode_index'])==row['episode'] and
                        int(item['frame_index'])==frame,'Raw loader index mismatch')
                require(item['prompt'] in row['prompts'],'Raw loader prompt mismatch')
                value=transformed[index]
                recovered=inverse(dict(state=np.array(value['state']),
                                       actions=np.array(value['actions'])))['actions']
                np.testing.assert_allclose(recovered,expected,atol=2e-5,rtol=2e-5)
                checks+=1
            if (row['episode']+1)%50==0:
                print('REPLAY_BOUNDARY_PREFLIGHT',row['episode']+1,'/ 900',flush=True)
        anchors=trainer.validation_anchors(cfg,args,audits['val']['rows'])
        require(len(anchors)==720,'Local validation panel changed')
        result=dict(status='passed',gpu_used=False,weights_loaded=False,
            batch_shape=[8,50,32],batches_checked=9,official_boundary_anchors=checks,
            local_validation_anchors=len(anchors),sampler=sampler.report(),
            configuration=report,video_reader=video_report,
            script_sha256=digest(Path(__file__)),
            limitation='Loader preflight only; no new model training or success-rate evidence')
        with a.output.open('x') as stream:
            json.dump(result,stream,indent=2)
        print('REPLAY_PREFLIGHT_PASSED',checks,flush=True)
    finally:
        reader.close()


if __name__=='__main__':
    main()
