"""Use existing, episode-grounded training instructions with arm consistency.

Training only: validation/evaluation prompts are never changed. Hanging-mug's
official expert always starts with the left arm; reject variants that explicitly
assign the initial lift to the right arm. All retained strings are exact source
instructions.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path('/root/autodl-tmp/EvolveVLA')
TRAIN_REPO='gen1_pilot_train_lossless_20261010'
_sampling = None


def sampling_report():
    if _sampling is None:
        raise RuntimeError('Prompt augmentation has not been installed')
    return dict(samples_fetched=_sampling['fetched'],
                unique_frame_indices=len(_sampling['indices']),
                available_frames=_sampling['available_frames'],
                per_episode_samples=dict(_sampling['episodes']))


def compatible(task,prompt):
    if task!='hanging_mug':return True
    # The official template "Lift ..., then move ... with the right arm" names
    # only the final arm, whereas "Use the right arm to lift ..." contradicts
    # the left-first expert. The first named arm alone cannot distinguish them.
    return re.search(r'\b(?:use|using|with)\s+(?:the|your)?\s*right\s+(?:arm|hand)\s+to\s+(?:lift|pick|grab|take)\b',
                     prompt.lower()) is None


def prepare():
    import h5py
    split=json.loads((ROOT/'data/gen1_pilot_two_task_20261010/split_manifest.json').read_text())
    by_source={str(Path(r['path']).resolve()):r for r in split['episodes'] if r['split']=='train'}
    audit=json.loads((ROOT/'data'/TRAIN_REPO/'conversion_audit.json').read_text())
    pools={};rows=[]
    for episode,source in enumerate(audit['sources']):
        path=Path(source['path']).resolve()
        row=by_source[str(path)]
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=source['sha256'] or digest!=row['sha256']:raise ValueError('Source changed')
        with h5py.File(path) as f:
            data=f['instructions'][()]
            prompts=json.loads(data.decode() if isinstance(data,bytes) else data)
        if not isinstance(prompts,list) or not all(isinstance(x,str) and x for x in prompts):
            raise ValueError('Invalid source instructions')
        retained=list(dict.fromkeys(x for x in prompts if compatible(row['task'],x)))
        if not retained:raise ValueError('All source instructions rejected')
        pools[episode]=retained
        rows.append(dict(episode=episode,task=row['task'],seed=row['seed'],source=str(path),
            sha256=digest,retained=len(retained),rejected=[x for x in prompts if not compatible(row['task'],x)]))
    if len(pools)!=90:raise ValueError('Training split changed')
    return pools,dict(training_only=True,episodes=rows,total_retained=sum(map(len,pools.values())),
        rule='Reject explicit right-arm initial lifting in hanging_mug; retain right-arm final hanging descriptions.',
        action_evidence=str(ROOT/'robotwin_repo_ba0fd21/envs/hanging_mug.py'))


def install(output):
    global _sampling
    from collections import Counter
    import numpy as np
    from openpi.training import data_loader
    pools,report=prepare()
    with (Path(output)/'prompt_augmentation.json').open('x') as f:json.dump(report,f,indent=2)
    original=data_loader.create_torch_dataset
    _sampling=dict(fetched=0,indices=set(),episodes=Counter(),available_frames=None)
    class Dataset:
        def __init__(self,base):
            self.base=base;self.rng=np.random.default_rng(42017)
            _sampling['available_frames']=len(base)
        def __len__(self):return len(self.base)
        def __getitem__(self,index):
            item=dict(self.base[index]);pool=pools[int(item['episode_index'])]
            _sampling['fetched']+=1
            _sampling['indices'].add(int(index))
            _sampling['episodes'][int(item['episode_index'])]+=1
            item['prompt']=pool[int(self.rng.integers(len(pool)))]
            return item
    def create(config,action_horizon,model_config):
        base=original(config,action_horizon,model_config)
        return Dataset(base) if config.repo_id==TRAIN_REPO else base
    data_loader.create_torch_dataset=create
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();pools,report=prepare()
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(episodes=len(pools),retained=report['total_retained'],
        rejected=sum(len(r['rejected']) for r in report['episodes']))))
