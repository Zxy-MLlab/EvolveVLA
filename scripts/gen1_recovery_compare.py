#!/usr/bin/env python3
"""Select on held-out actions, then run a fresh fixed Clean diagnostic panel."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/EvolveVLA')
SCRIPT=ROOT/'scripts/gen1_recovery_eval.py'
KEYS=('joint_mae','gripper_mae','first10_joint_mae','first10_gripper_mae')


def select(training,checkpoints):
    baseline=json.loads((training/'validation_0000.json').read_text())['metrics']['all']
    candidates=[]
    for path in sorted(training.glob('validation_*.json')):
        row=json.loads(path.read_text())
        if row['step']==0:continue
        if not (checkpoints/str(row['step'])/'params/_METADATA').is_file():continue
        score=sum(row['metrics']['all'][k]/baseline[k] for k in KEYS)/len(KEYS)
        candidates.append(dict(step=row['step'],score=score,metrics=row['metrics'],checkpoint=str(checkpoints/str(row['step']))))
    if not candidates:raise ValueError('No validated saved candidates')
    best=min(candidates,key=lambda x:(x['score'],x['step']))
    return dict(rule='equal mean of four validation MAE ratios to initialization; lowest score, earlier step breaks tie',
                baseline=baseline,candidates=candidates,selected=best)


def summarize(original,candidate):
    aa={(r['task'],r['seed']):r for r in original['records']}
    bb={(r['task'],r['seed']):r for r in candidate['records']}
    tasks={}
    for task in ('place_mouse_pad','hanging_mug'):
        pairs=[];unpaired=[]
        for key in aa.keys()|bb.keys():
            if key[0]!=task:continue
            a,b=aa.get(key,{}),bb.get(key,{})
            if a.get('status')==b.get('status')=='completed' and a['hashes']==b['hashes'] and a['binding']==b['binding']:
                pairs.append(dict(seed=key[1],original=a['success'],candidate=b['success']))
            else:unpaired.append(dict(seed=key[1],original_status=a.get('status'),candidate_status=b.get('status'),
                                     initial_equal=a.get('hashes')==b.get('hashes')))
        tasks[task]=dict(paired_completed=len(pairs),original_successes=sum(p['original'] for p in pairs),
            candidate_successes=sum(p['candidate'] for p in pairs),
            original_only=sum(p['original'] and not p['candidate'] for p in pairs),
            candidate_only=sum(p['candidate'] and not p['original'] for p in pairs),
            pairs=sorted(pairs,key=lambda p:p['seed']),unpaired=sorted(unpaired,key=lambda p:p['seed']))
    return dict(tasks=tasks,limitation='Controlled diagnostic, fixed bound seen prompts and reset policy RNG per episode; exclusions are not failures. Initial hashes are not full physics state.')


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)


def physical_summary(directory):
    """Explanatory progress measures only; never replace task success predicates."""
    summaries={}
    for path in directory.glob('*/physical.jsonl'):
        rows=[json.loads(line) for line in path.open()]
        if not rows:continue
        first,last=rows[0]['physical'],rows[-1]['physical']
        name='mouse' if 'mouse' in first else 'mug'
        initial=first[name]['position']
        max_lift=max(r['physical'][name]['position'][2]-initial[2] for r in rows)
        report=dict(max_object_lift_m=max_lift,final_object_position=last[name]['position'])
        if name=='mouse':
            report['final_xy_error_m']=[last['mouse']['position'][i]-last['target']['position'][i] for i in (0,1)]
        summaries[path.parent.name]=report
    return summaries


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    p.add_argument('--reference',type=Path,help='Existing complete same-seed original panel; development comparison')
    p.add_argument('--seed',type=int,default=300000)
    a=p.parse_args()
    if a.wait:
        while not (a.training/'completed.json').exists():
            print('WAITING_FOR_TRAINING',a.training,flush=True);time.sleep(30)
    elif not (a.training/'completed.json').exists():raise RuntimeError('Training not completed')
    a.output.mkdir(exist_ok=False)
    selection=select(a.training,a.checkpoints)
    write(a.output/'selection.json',selection)
    print('SELECTED',selection['selected']['checkpoint'],selection['selected']['score'],flush=True)
    chosen=Path(selection['selected']['checkpoint'])
    if not (chosen/'verification.json').exists():
        subprocess.run(['/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python','-I','-B',
            str(ROOT/'scripts/gen1_recovery_verify.py'),'--checkpoint',str(chosen),
            '--training',str(a.training)],check=True)
    original=ROOT/'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'
    for label,checkpoint in [('original',original),('candidate',selection['selected']['checkpoint'])]:
        if label=='original' and a.reference is not None:
            reference_manifest=json.loads((a.reference/'manifest.json').read_text())
            if (reference_manifest['checkpoint']!=str(original) or reference_manifest['seed_start']!=a.seed
                    or reference_manifest['episodes_per_task']!=10):raise ValueError('Reference protocol mismatch')
            continue
        output=ROOT/'outputs'/(a.output.name+'_'+label)
        subprocess.run([sys.executable,'-u',str(SCRIPT),'run','--output',str(output),
            '--checkpoint',str(checkpoint),'--seed',str(a.seed),'--episodes','10','--reuse-policy'],check=True)
        print('FINISHED',label,flush=True)
    reference_dir=a.reference if a.reference is not None else ROOT/'outputs'/(a.output.name+'_original')
    original=json.loads((reference_dir/'summary.json').read_text())
    candidate=json.loads((ROOT/'outputs'/(a.output.name+'_candidate')/'summary.json').read_text())
    result=summarize(original,candidate)
    result['physical_diagnostics']={'original':physical_summary(reference_dir),
        'candidate':physical_summary(ROOT/'outputs'/(a.output.name+'_candidate'))}
    result['reference']=str(reference_dir)
    result['development_panel_reused']=a.reference is not None
    write(a.output/'comparison.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
