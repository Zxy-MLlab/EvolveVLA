#!/usr/bin/env python3
"""Summarize per-episode consistency of the fixed validation panel."""
import argparse, json, statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen1_multitask_validation_report import checked_records, KEYS

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[0,4000,12000])
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    reports={s:json.loads((a.training/f'validation_{s:04d}.json').read_text()) for s in a.steps}
    rec={s:checked_records(x) for s,x in reports.items()}
    keys=set(rec[a.steps[0]])
    if any(set(v)!=keys for v in rec.values()): raise ValueError('Anchor identities changed')
    out={}
    tasks=sorted({k[0] for k in keys})
    for task in tasks:
        episodes=sorted({k[1] for k in keys if k[0]==task})
        out[task]={}
        for metric in KEYS:
            rows=[]
            for episode in episodes:
                ids=[k for k in keys if k[:2]==(task,episode)]
                means={str(s):statistics.mean(rec[s][k][metric] for k in ids) for s in a.steps}
                rows.append({'episode':episode,'means':means,
                    'ratio_to_step0':means[str(a.steps[-1])]/max(means[str(a.steps[0])],1e-8)})
            out[task][metric]={'episodes':rows,
                'improved_count':sum(x['ratio_to_step0']<1 for x in rows),
                'episode_count':len(rows),
                'median_ratio':statistics.median(x['ratio_to_step0'] for x in rows)}
    result={'status':'completed','steps':a.steps,'tasks':out,
            'limitation':'Per-episode expert-action error consistency; not closed-loop success or causal evidence.'}
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({t:out[t] for t in ('place_mouse_pad','hanging_mug')},indent=2))
if __name__=='__main__': main()
