#!/usr/bin/env python3
"""One observation-driven approach replan per eligible arm, original controller.

Development-only candidate. A currently closed arm whose predicted50-action
chunk first opens at index i triggers execution of max(1,i-20) actions before
normal continuous-noise inference resumes. No expert phase, object truth, model
change, simulator controller change or inference-key reset is used.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

SCRIPT=Path(__file__).resolve()
sys.path.insert(0,str(SCRIPT.parent))
import gen1_policy_duration_repair_20261020 as duration
import gen1_recovery_takeover_v2_20261020 as recovery

horizon=duration.horizon
PROTOCOL=horizon.ROOT/'outputs/gen1_control_repair_20261020/approach_replan_protocol.json'
ORIGINAL_WORKER=horizon.worker
ACTIVE={}
EXECUTION_MODE='approach_replan'
SHARED_CACHE=[]
SHARE_PAIRED_CACHE=False


class ApproachGate:
    """Use only current14D drive state and current model50x14 predictions."""
    def __init__(self,task):
        if task not in ('place_mouse_pad','hanging_mug'):raise ValueError('Unsupported approach-replan task')
        self.arms={'left':6,'right':13} if task=='place_mouse_pad' else {'right':13}
        self.used=set()

    def select(self,state,chunk):
        candidates=[]
        for arm,column in self.arms.items():
            if arm in self.used or float(state[column])>.2:continue
            opening=next((i for i,row in enumerate(chunk) if float(row[column])>.2),None)
            if opening is not None:
                candidates.append(dict(arm=arm,first_opening_index=opening,
                    execute_prefix=max(1,opening-20),current_drive=float(state[column])))
        count=min([row['execute_prefix'] for row in candidates],default=50)
        selected=[row['arm'] for row in candidates if row['execute_prefix']==count]
        before=sorted(self.used)
        self.used.update(selected)
        return count,dict(triggered=bool(selected),selected_arms=selected,candidates=candidates,
            used_before=before,used_after=sorted(self.used),max_triggers_per_arm=1,
            closed_threshold=.2,opening_threshold=.2,lead_actions=20,
            no_expert_indices_object_truth_or_rng_reset=True)


def execution_loop(policy,first,prefix,send_action,receive_step,decode_observation,
                   record_inference,action_limit=2000,clock=time.perf_counter):
    import numpy as np
    if prefix!=50 or os.environ.get('GEN1_DURATION_MODE')!='original':
        raise ValueError('Approach replan requires50-prediction horizon and original controller')
    images,state=decode_observation(first['observation'])
    prompt=first['diagnostic_binding']['prompt']
    gate=ApproachGate(ACTIVE['case']['task'])
    steps=count=0;latency=0.;events=[]
    while steps<action_limit:
        started=clock()
        chunk=np.asarray(policy.infer_chunk(images=images,state=state,prompt=prompt))
        elapsed=clock()-started
        if chunk.shape!=(50,14) or not np.isfinite(chunk).all():raise ValueError('Expected finite50x14physical action chunk')
        count+=1;latency+=elapsed
        if EXECUTION_MODE=='approach_replan':requested,decision=gate.select(state,chunk)
        else:requested,decision=50,dict(triggered=False,disabled=True,selected_arms=[])
        execute_count=min(requested,action_limit-steps)
        if decision['triggered']:events.append(dict(step=steps,inference=count,**decision))
        record_inference(dict(step=steps,inference=count,inference_seconds=elapsed,
            execute_prefix=execute_count,state=np.asarray(state).tolist(),actions=chunk.tolist(),
            execution_mode=EXECUTION_MODE,approach_replan=decision),images)
        for action in chunk[:execute_count]:
            send_action(action.tolist());response=receive_step()
            if response['type']!='step':raise RuntimeError(str(response))
            steps+=1
            if response.get('terminated') or response.get('truncated'):
                return dict(status='completed',success=bool(response['info']['is_success']),steps=steps,
                    inference_count=count,inference_seconds=latency,
                    termination='terminated' if response.get('terminated') else 'truncated',
                    execution_mode=EXECUTION_MODE,approach_replan_events=events,
                    rng_reset_only_at_episode_start=True)
            images,state=decode_observation(response['observation'])
    return dict(status='completed',success=False,steps=steps,inference_count=count,inference_seconds=latency,
        termination='fixed_action_ceiling',execution_mode=EXECUTION_MODE,approach_replan_events=events,
        rng_reset_only_at_episode_start=True)


def verify_protocol():
    protocol=recovery.read(PROTOCOL)
    for path,expected in protocol['source_sha256'].items():
        if recovery.digest(path)!=expected:raise ValueError(f'Frozen approach-replan input changed: {path}')
    return protocol


def worker(case,prefix,args,cache,directory):
    protocol=verify_protocol()
    if os.environ.get('GEN1_DURATION_MODE')!='original':raise ValueError('Simulator controller must remain original')
    if Path(args.checkpoint).resolve()!=Path(protocol['checkpoint']).resolve():raise ValueError('Approach development checkpoint changed')
    expected=next((c for c in recovery.read(protocol['panel'])['cases'] if c['id']==case['id']),None)
    if expected!=case:raise ValueError('Development case changed')
    horizon.write_json(directory/'execution_protocol.json',dict(protocol=str(PROTOCOL),
        protocol_sha256=recovery.digest(PROTOCOL),execution_mode=EXECUTION_MODE,
        shared_paired_model_cache=SHARE_PAIRED_CACHE,
        interpretation='Online commanded-state/model-prediction intervention; continuous episode inference RNG'))
    ACTIVE.update(case=case)
    return ORIGINAL_WORKER(case,prefix,args,SHARED_CACHE if SHARE_PAIRED_CACHE else cache,directory)


def run_condition(mode,remaining):
    global EXECUTION_MODE
    EXECUTION_MODE=mode
    sys.argv[1:]=remaining
    with horizon.evaluation.binding.replaced(duration,'SCRIPT',SCRIPT), \
         horizon.evaluation.binding.replaced(horizon,'worker',worker), \
         horizon.evaluation.binding.replaced(horizon,'execution_loop',execution_loop):
        return duration.main()


def main():
    global SHARE_PAIRED_CACHE
    if len(sys.argv)>1 and sys.argv[1]=='sim':
        del sys.argv[1]
        return duration.simulator()
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--execution-mode',choices=('original50','approach_replan','both'),default='both')
    options,remaining=parser.parse_known_args(sys.argv[1:])
    verify_protocol()
    SHARE_PAIRED_CACHE=options.execution_mode=='both'
    SHARED_CACHE.clear()
    if options.execution_mode!='both':return run_condition(options.execution_mode,remaining)
    if '--output' not in remaining:raise ValueError('Both conditions require explicit --output base directory')
    output_index=remaining.index('--output')+1
    base=Path(remaining[output_index]).resolve()
    if base.exists():raise ValueError('Paired output root must be fresh')
    for mode in ('original50','approach_replan'):
        arguments=list(remaining);arguments[output_index]=str(base/mode)
        run_condition(mode,arguments)
    horizon.write_json(base/'paired_execution.json',dict(status='completed',conditions=['original50','approach_replan'],
        shared_model_cache=True,loaded_model_count=len(SHARED_CACHE),protocol=str(PROTOCOL),
        protocol_sha256=recovery.digest(PROTOCOL),selection_from_intermediate_outcomes=False))


if __name__=='__main__':sys.exit(main())
