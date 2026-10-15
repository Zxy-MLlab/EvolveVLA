"""Compose independently audited official/local datasets using fixed transforms.

No modification to the pinned OpenPI loader or any running trainer is needed.
Every task gets equal official/local allocation; validation is never a source.
"""
from bisect import bisect_right
import dataclasses
import json
from pathlib import Path
import random

from gen1_multitask_data import digest, require
from gen1_replay_sampler import ReplaySampler

EXPANDED_REPO = 'gen1_multitask_coverage_expanded_train_20261014'
LOCAL_TRAIN_TASKS = ('hanging_mug', 'place_mouse_pad')


class EpisodeSubset:
    """Map compact indices to complete audited episodes; never cross episode bounds."""
    def __init__(self, base, rows, tasks):
        self.base, self.rows, self.ends, self.starts = base, [], [], []
        total = 0
        for row in rows:
            if row['task'] not in tasks:
                continue
            require(row['split'] == 'train', 'Held-out episode in subset')
            require(0 <= row['start'] < row['stop'] <= len(base)
                    and row['stop']-row['start'] == row['frames'], 'Invalid subset bounds')
            self.starts.append(row['start'])
            self.rows.append(dict(row, start=total, stop=total+row['frames']))
            total += row['frames']
            self.ends.append(total)
        require({r['task'] for r in self.rows} == set(tasks), 'Missing requested local task')

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        before = self.ends[episode-1] if episode else 0
        return self.base[self.starts[episode]+index-before]


class OfficialPrompts:
    def __init__(self, base, rows, seed=42017):
        self.base,self.rows=base,rows
        self.rng=random.Random(seed)
        require(len(base)==rows[-1]['stop'],'Official dataset length mismatch')
        require(all(r['source']=='official' and r['split']=='train' for r in rows),
                'Invalid official replay split')

    def __len__(self):
        return len(self.base)

    def __getitem__(self,index):
        item=dict(self.base[index])
        row=self.rows[int(item['episode_index'])]
        require(row['start']<=index<row['stop'] and
                int(item['frame_index'])==index-row['start'],'Official index binding mismatch')
        require(item['prompt'] in row['prompts'],'Official prompt absent from source')
        item['prompt']=self.rng.choice(row['prompts'])
        return item


class JoinedDataset:
    def __init__(self, datasets):
        self.datasets=list(datasets)
        require(bool(self.datasets) and all(len(d)>0 for d in self.datasets),'Empty replay source')
        total=0
        self.ends=[]
        for dataset in self.datasets:
            total+=len(dataset)
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self,index):
        if index<0 or index>=len(self):
            raise IndexError(index)
        source=bisect_right(self.ends,index)
        before=self.ends[source-1] if source else 0
        return self.datasets[source][index-before]


def build(cfg, args, local_rows, sharding):
    """Return the native batch wrapper, replay sampler and auditable bindings."""
    import jax
    from openpi.training import data_loader
    from gen1_multitask_prompts import prepare, PromptDataset

    manifest_path=Path(args.replay_manifest)
    manifest=json.loads(manifest_path.read_text())
    require(manifest['status']=='raw_manifest_passed','Invalid replay manifest')
    official_root=Path(args.lerobot_home)/args.replay_repo_id
    audit=json.loads((official_root/'conversion_audit.json').read_text())
    require(audit['status']=='passed' and audit['all_frames_decoded'] and
            audit['all_state_action_cells_exact'] and audit['all_timestamps_checked'] and
            audit['prompt_binding_checked'],'Replay conversion gate missing')
    require(audit['manifest_sha256']==digest(manifest_path),'Conversion/manifest mismatch')
    local_split=json.loads(Path(args.split_manifest).read_text())
    local_hash=digest(Path(args.split_manifest))
    require(local_hash==manifest['local_split_sha256'] or
            (local_split.get('base_split_sha256')==manifest['local_split_sha256'] and
             set(local_split['tasks'])==set(manifest['tasks']) and
             all(not (r['task']==o['task'] and r['seed']==o['seed'])
                 for r in local_split['episodes'] for o in manifest['episodes'])),
            'Local split changed or overlaps official replay')
    require(digest(Path(manifest['norm_stats']['path']))==manifest['norm_stats']['sha256'],
            'Initializer normalization changed')
    require(set(r['task'] for r in local_rows)==set(manifest['tasks']),'Mixed task coverage mismatch')
    require(args.replay_repo_id not in (args.repo_id,args.val_repo_id),'Replay repo must be separate')
    local_config=cfg.data.create(cfg.assets_dirs,cfg.model)
    official_config=dataclasses.replace(cfg.data,repo_id=args.replay_repo_id).create(cfg.assets_dirs,cfg.model)
    for key in local_config.norm_stats:
        import numpy as np
        left,right=local_config.norm_stats[key],official_config.norm_stats[key]
        for name in ('mean','std','q01','q99'):
            np.testing.assert_array_equal(getattr(left,name),getattr(right,name))
    local_raw=data_loader.create_torch_dataset(local_config,cfg.model.action_horizon,cfg.model)
    require(len(local_raw)==local_rows[-1]['stop'],'Local frame count mismatch')
    subset = args.repo_id == EXPANDED_REPO
    prompt_rows = [r for r in local_rows if r['task'] in LOCAL_TRAIN_TASKS] if subset else local_rows
    pools,prompt_report=prepare(prompt_rows)
    local_raw=PromptDataset(local_raw,prompt_rows,pools,mode='frame')
    if subset:
        local_raw = EpisodeSubset(local_raw, local_rows, LOCAL_TRAIN_TASKS)
        local_rows = local_raw.rows
        require(len(local_rows) == 1990 and all(sum(r['task']==t for r in local_rows)==995
                for t in LOCAL_TRAIN_TASKS), 'Unexpected two-task training counts')
    official_raw=data_loader.create_torch_dataset(official_config,cfg.model.action_horizon,cfg.model)
    official_raw=OfficialPrompts(official_raw,manifest['episodes'])
    dataset=JoinedDataset([data_loader.transform_dataset(local_raw,local_config),
                           data_loader.transform_dataset(official_raw,official_config)])
    rows=[dict(r,source='local') for r in local_rows]
    offset=len(local_raw)
    rows.extend(dict(r,start=r['start']+offset,stop=r['stop']+offset) for r in manifest['episodes'])
    sampler=ReplaySampler(rows,len(dataset),num_samples=len(dataset),seed=args.data_seed,
                          allow_local_subset=subset)
    require(jax.process_count()==1,'Single-process replay currently required')
    native=data_loader.TorchDataLoader(dataset,local_batch_size=args.batch_size,
        sharding=sharding,sampler=sampler,shuffle=False,num_batches=None,num_workers=0,
        seed=args.data_seed,framework='jax')
    report=dict(local_frames=len(local_raw),official_frames=len(official_raw),
                local_train_tasks=sorted({r['task'] for r in local_rows}),
                local_train_episodes=len(local_rows), official_train_episodes=len(manifest['episodes']),
                total_train_episodes=len(rows),
                total_frames=len(dataset),local_prompt_report=prompt_report,
                official_manifest_sha256=digest(manifest_path),
                conversion_audit_sha256=digest(official_root/'conversion_audit.json'),
                norm_stats=manifest['norm_stats'],
                sampler='50% local / 50% official; equal tasks within each source; uniform frames',
                loader_sha256=digest(Path(__file__)),
                sampler_sha256=digest(Path(__file__).with_name('gen1_replay_sampler.py')))
    return data_loader.DataLoaderImpl(local_config,native),sampler,report
