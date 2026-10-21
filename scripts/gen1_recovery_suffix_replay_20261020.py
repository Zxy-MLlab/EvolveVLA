#!/usr/bin/env python3
"""Verify saved recovery suffixes after exact frozen prefixes, then deployment replay.

Dense replay uses saved planner paths without replanning. Capture native observations
at global suffix physics ticks0,15,...; the optional final short interval is flagged
and excluded from deployment labels. No source suffix is edited; no model is loaded.
"""
import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import pickle
import sys
import traceback

SCRIPT=Path(__file__).resolve()
sys.path.insert(0,str(SCRIPT.parent))
import gen1_recovery_takeover_v2_20261020 as collector
import gen1_duration_controller_20261020 as duration

ORIGINAL_COLLECT=collector.collect_suffix
horizon,oracle=collector.horizon,collector.oracle
read,lines,digest=collector.read,collector.lines,collector.digest
CAMERAS=('head_camera','left_camera','right_camera')


def load_raw(row):
    import numpy as np
    path=Path(row['path'])
    if digest(path)!=row['sha256']:
        raise ValueError('Raw observation hash mismatch')
    with path.open('rb') as stream: observation=pickle.load(stream)
    drive=np.asarray(observation['joint_action']['vector'],dtype=float)
    if drive.shape!=(14,) or not np.isfinite(drive).all():
        raise ValueError('Native drive vector must be finite14D')
    for name in CAMERAS:
        image=np.asarray(observation['observation'][name]['rgb'])
        if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8:
            raise ValueError('Native RGB missing or invalid')
    return observation,drive


def compare_captures(source,actual):
    import numpy as np
    mismatches=[]
    if len(source)!=len(actual):
        mismatches.append(dict(field='capture_count',source=len(source),actual=len(actual)))
    for index,(expected,observed) in enumerate(zip(source,actual)):
        for field in ('physics_tick','physical','success','components'):
            difference=collector.differences(expected[field],observed[field])
            if difference:mismatches.append(dict(index=index,field=field,differences=difference))
        a,ad=load_raw(expected);b,bd=load_raw(observed)
        if not np.array_equal(ad,bd):mismatches.append(dict(index=index,field='joint_action.vector'))
        for name in CAMERAS:
            if not np.array_equal(a['observation'][name]['rgb'],b['observation'][name]['rgb']):
                mismatches.append(dict(index=index,field='rgb/'+name))
    return mismatches


def uniform_rows(path):
    rows=lines(path)
    regular=[r for r in rows if r.get('full_interval_boundary') is True]
    if len(regular)<2 or [r['physics_tick'] for r in regular]!=list(range(0,len(regular)*15,15)):
        raise ValueError('Uniform captures must contain consecutive actual physics ticks0,15,...')
    for row in regular:load_raw(row)
    return regular


def terminal_hold(task,ticks,start_tick,after_tick):
    """Keep commanded positions; explicitly zero command velocities, never measured state."""
    import numpy as np
    before=oracle.physical(task)
    arm_positions={side:list(before[side]['drive_targets'][:6]) for side in ('left','right')}
    result=dict(requested_ticks=ticks,physics_ticks=0,start_tick=start_tick,
        positions_preserved=True,command_velocity_intervention='arm and gripper drive velocity targets set to zero',
        measured_state_reset=False,before_physical=before,initial_success=bool(task.check_success()),
        all_ticks_success=bool(task.check_success()))
    for index in range(ticks):
        for side in ('left','right'):
            task.robot.set_arm_joints(np.asarray(arm_positions[side]),np.zeros(6),side)
            for joint in getattr(task.robot,side+'_gripper'):
                joint[0].set_drive_velocity_target(0.)
        task.scene.step()
        tick=start_tick+index+1
        success=bool(task.check_success())
        result['all_ticks_success'] &= success
        result['physics_ticks']+=1
        after_tick(tick,success)
    result.update(final_success=bool(task.check_success()),after_physical=oracle.physical(task))
    result['success']=result['physics_ticks']==ticks and result['all_ticks_success'] and result['final_success']
    return result


def dense_replay(task,case,directory,max_ticks):
    source=Path(case['recovery_source_directory'])/'expert_suffix'
    expected=read(source/'result.json')
    source_rows=lines(source/'captures.jsonl')
    uniform_directory=directory.parent/'uniform_raw'
    uniform_directory.mkdir(exist_ok=False)
    uniform_path=directory.parent/'uniform_captures.jsonl'
    uniform=[]
    latest_tick=0
    original_audit=collector.PhysicsAudit
    with uniform_path.open('x',buffering=1) as stream:
        def capture(tick,full=True):
            observation=task.get_obs()
            target=uniform_directory/f'{len(uniform):07d}.pkl'
            with target.open('xb') as output:pickle.dump(observation,output,pickle.HIGHEST_PROTOCOL)
            row=dict(index=len(uniform),physics_tick=tick,simulation_seconds=tick*float(task.scene.get_timestep()),
                full_interval_boundary=full,partial_final_interval=not full,path=str(target),sha256=digest(target),
                physical=oracle.physical(task),observables=collector.observables.observe(task),
                success=bool(task.check_success()),provenance='saved_expert_suffix_dense_replay',
                phase='expert_suffix' if tick<=expected['physics_ticks'] else 'explicit_terminal_hold',
                observation_semantics='native get_obs; joint_action.vector is drive target')
            uniform.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
        class UniformAudit(original_audit):
            def __init__(self,*args,after_step=None,**kwargs):
                def after(tick):
                    nonlocal latest_tick
                    latest_tick=tick
                    if after_step:after_step(tick)
                    if tick%15==0:capture(tick)
                super().__init__(*args,after_step=after,**kwargs)
        def suffix(current,active):
            with (source/'planner_paths.pkl').open('rb') as file:paths=pickle.load(file)
            current.need_plan=False
            current.left_joint_path=copy.deepcopy(paths['left_joint_path'])
            current.right_joint_path=copy.deepcopy(paths['right_joint_path'])
            current.left_cnt=current.right_cnt=0
            capture(0)
            collector.native_suffix(current,active)
        with horizon.evaluation.binding.replaced(collector,'PhysicsAudit',UniformAudit):
            result=ORIGINAL_COLLECT(task,case,directory,max_ticks,suffix=suffix)
        mismatches=compare_captures(source_rows,lines(directory/'captures.jsonl'))
        for field in ('physics_ticks','after_physical','final_success','final_predicates'):
            drift=collector.differences(expected[field],result[field])
            if drift:mismatches.append(dict(field=field,differences=drift))
        consumed=dict(left=task.left_cnt,right=task.right_cnt)
        available=dict(left=len(task.left_joint_path),right=len(task.right_joint_path))
        if consumed!=available:mismatches.append(dict(field='planner_path_consumption',consumed=consumed,available=available))
        result.update(replay_mode='dense',dense_source_match=not mismatches,source_mismatches=mismatches,
            source_end_physics_tick=latest_tick,source_end_final_success=result['final_success'],
            source_end_physical=result['after_physical'],source_collection_success=expected['success'],
            path_consumption=consumed,available_paths=available)
        result['success']=bool(result['success'] and result['dense_source_match'])
        hold_ticks=case.get('terminal_hold_ticks',0)
        if result['success'] and hold_ticks:
            if latest_tick+hold_ticks>max_ticks:raise ValueError('Insufficient bounded physics budget for terminal hold')
            with (directory.parent/'terminal_hold_predicates.jsonl').open('x',buffering=1) as checks:
                def hold_observe(tick,success):
                    nonlocal latest_tick
                    latest_tick=tick
                    checks.write(json.dumps(dict(physics_tick=tick,success=success,components=collector.predicates(task,case['task'])))+'\n')
                    if tick%15==0:capture(tick)
                result['terminal_hold']=terminal_hold(task,hold_ticks,latest_tick,hold_observe)
            result['stable_success']=result['terminal_hold']['success']
            result['success'] &= result['stable_success']
        else:
            result['terminal_hold']=dict(requested_ticks=hold_ticks,status='not_requested' if not hold_ticks else 'blocked_source_match_or_success')
            result['stable_success']=None
        if uniform and uniform[-1]['physics_tick']!=latest_tick:
            capture(latest_tick,full=False)
    result.update(uniform_captures_path=str(uniform_path),uniform_captures_sha256=digest(uniform_path),
        uniform_full_intervals=max(0,sum(r['full_interval_boundary'] for r in uniform)-1),
        final_short_interval_excluded=bool(uniform and not uniform[-1]['full_interval_boundary']),
        physics_ticks=latest_tick,final_success=bool(task.check_success()),after_physical=oracle.physical(task))
    result['status']='dense_replay_completed' if result['success'] else 'dense_replay_failed'
    result['dataset_ready']=False
    task.eval_success=result['success']
    horizon.atomic_json(directory.parent/'dense_verification.json',result)
    return result


def sparse_replay(task,case,directory,max_ticks):
    import numpy as np
    mode=case['replay_mode']
    rows=uniform_rows(Path(case['uniform_captures_path']))
    # Every label is obs_t -> the drive target captured exactly15 physics ticks later.
    targets=[load_raw(row)[1] for row in rows[1:]]
    directory.mkdir(exist_ok=False)
    result=dict(status='sparse_replay_error',success=False,final_success=False,ever_success=False,
        replay_mode=mode,deployment_mode=mode,dataset_ready=False,timestep=float(task.scene.get_timestep()),
        target_count=len(targets),actions_executed=0,physics_ticks=0,
        prefix_action_count=task.take_action_cnt,official_step_limit=task.step_lim,
        final_short_interval_excluded=True,label_semantics='obs_t -> drive_target_at_t_plus15_physics_ticks',
        before_physical=oracle.physical(task),before_observables=collector.observables.observe(task))
    original=task.__class__.take_action.__get__(task,task.__class__)
    take,metadata,controller=duration.create(original,mode,ticks=15)
    result['controller']=metadata
    with (directory/'actions.jsonl').open('x',buffering=1) as actions, (directory/'predicates.jsonl').open('x',buffering=1) as checks:
        def observe(tick):
            success=bool(task.check_success())
            if success and not result['ever_success']:result['first_success_physics_tick']=tick
            result['ever_success']|=success
            checks.write(json.dumps(dict(physics_tick=tick,success=success,components=collector.predicates(task,case['task'])))+'\n')
        audit=collector.PhysicsAudit(after_step=observe,limit=max_ticks)
        previous=sys.gettrace()
        try:
            sys.settrace(audit.trace)
            for index,target in enumerate(targets):
                if task.eval_success or task.take_action_cnt>=task.step_lim:break
                before=audit.count
                take(np.asarray(target,dtype=np.float32))
                result['actions_executed']+=1
                actions.write(json.dumps(dict(index=index,target=target.tolist(),source_tick=rows[index+1]['physics_tick'],
                    physics_ticks=audit.count-before,total_physics_ticks=audit.count,
                    physical=oracle.physical(task),observables=collector.observables.observe(task),
                    controller=controller.last_record,success=bool(task.check_success())),allow_nan=False)+'\n')
            result['status']='sparse_replay_completed'
        except collector.RecoveryLimit as exc:
            result.update(status='sparse_physics_limit',error=str(exc))
        except Exception as exc:
            result.update(status='sparse_replay_error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        finally:
            sys.settrace(previous)
            result.update(physics_ticks=audit.count,final_success=bool(task.check_success()),
                native_eval_success=bool(task.eval_success),after_physical=oracle.physical(task),
                after_observables=collector.observables.observe(task),remaining_action_budget=task.step_lim-task.take_action_cnt,
                label_exhaustion=result['actions_executed']==len(targets))
            result['success']=result['status']=='sparse_replay_completed' and result['final_success'] and result['native_eval_success']
    result['official_rollout_success']=result['success']
    hold_ticks=case.get('terminal_hold_ticks',0)
    if result['success'] and hold_ticks:
        if result['physics_ticks']+hold_ticks>max_ticks:raise ValueError('Insufficient sparse physics budget for hold')
        with (directory/'terminal_hold_predicates.jsonl').open('x',buffering=1) as held:
            def held_tick(tick,success):
                held.write(json.dumps(dict(physics_tick=tick,success=success,components=collector.predicates(task,case['task'])))+'\n')
            result['terminal_hold']=terminal_hold(task,hold_ticks,result['physics_ticks'],held_tick)
        result.update(stable_success=result['terminal_hold']['success'],
            final_success=result['terminal_hold']['final_success'],after_physical=oracle.physical(task))
        result['physics_ticks']+=result['terminal_hold']['physics_ticks']
        result['success'] &= result['stable_success']
    else:
        result['stable_success']=None
        result['terminal_hold']=dict(requested_ticks=hold_ticks,status='not_requested' if not hold_ticks else 'blocked_sparse_success')
    result['early_success_preserved']=True
    result['all_labels_replayed']=result['actions_executed']==len(targets)
    horizon.atomic_json(directory/'result.json',result)
    return result


def replay_suffix(task,case,directory,max_ticks):
    if case['replay_mode']=='dense':return dense_replay(task,case,directory,max_ticks)
    return sparse_replay(task,case,directory,max_ticks)


def load_source(run,requested):
    manifest=read(run/'manifest.json')
    lookup={c['id']:c for c in manifest['cases']}
    cases=[];excluded=[]
    for name in requested:
        if name not in lookup:raise ValueError(f'Case absent from source manifest: {name}')
        path=run/name
        result=read(path/'result.json')
        if not (result.get('status')=='recovery_completed' and result.get('success') is True and
                result.get('final_success') is True and result.get('exact_prefix') is True and
                result.get('simulator_returncode')==0 and result.get('expert_gate',{}).get('status')=='passed'):
            excluded.append(dict(id=name,status='source_not_successful_verified_recovery',source_status=result.get('status')))
            continue
        case=dict(lookup[name],recovery_source_directory=str(path))
        files=[path/'result.json',path/'case.json',path/'prefix_verification.jsonl',
               path/'expert_suffix/result.json',path/'expert_suffix/captures.jsonl',path/'expert_suffix/planner_paths.pkl']
        for row in lines(path/'expert_suffix/captures.jsonl'):
            load_raw(row);files.append(Path(row['path']))
        case['recovery_source_sha256']={str(p):digest(p) for p in files}
        if digest(path/'expert_suffix/planner_paths.pkl')!=read(path/'expert_suffix/result.json')['planner_paths_sha256']:
            raise ValueError('Source planner path hash changed')
        cases.append(case)
    return cases,excluded


def readiness(directory,case,dense,sparse):
    result=dict(schema_version=1,id=case['id'],task=case['task'],
        case_path=str(directory/'dense/case.json'),case_sha256=digest(directory/'dense/case.json'),
        dense_result_path=str(directory/'dense/result.json'),dense_result_sha256=digest(directory/'dense/result.json'),
        dense_captures_path=str(directory/'dense/uniform_captures.jsonl'),
        dense_captures_sha256=digest(directory/'dense/uniform_captures.jsonl') if (directory/'dense/uniform_captures.jsonl').exists() else None,
        exact_prefix=dense.get('exact_prefix') is True,dense_success=dense.get('success') is True,
        dense_source_match=dense.get('dense_source_match') is True,timestep=dense.get('timestep'),sparse_modes={})
    for mode,value in sparse.items():
        path=directory/mode/'result.json'
        result['sparse_modes'][mode]=dict(result_path=str(path),sha256=digest(path),
            exact_prefix=value.get('exact_prefix') is True,success=value.get('success') is True)
    result['dataset_ready_by_mode']={mode:bool(result['exact_prefix'] and result['dense_success'] and result['dense_source_match'] and
        entry['exact_prefix'] and entry['success']) for mode,entry in result['sparse_modes'].items()}
    result['dataset_ready']=any(result['dataset_ready_by_mode'].values())
    result['training_acceptance_pending']='Builder must check hashes,50-transition windows,RGB/14D alignment and episode split'
    horizon.atomic_json(directory/'readiness.json',result)
    return result


def main():
    if len(sys.argv)>1 and sys.argv[1]=='sim':
        del sys.argv[1]
        with horizon.evaluation.binding.replaced(collector,'collect_suffix',replay_suffix):
            return collector.simulator()
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-ids',nargs='+',choices=collector.IDS,default=list(collector.IDS))
    parser.add_argument('--sparse-modes',nargs='+',choices=('original','fixed15'),default=['original','fixed15'])
    parser.add_argument('--terminal-hold-ticks',type=int,default=250)
    parser.add_argument('--max-suffix-ticks',type=int,default=10000)
    parser.add_argument('--case-timeout',type=int,default=1800)
    args=parser.parse_args()
    args.output=args.output.resolve();args.source_run=args.source_run.resolve()
    if not 0<=args.terminal_hold_ticks<=1000:parser.error('Terminal hold must be0..1000ticks')
    if not 0<args.max_suffix_ticks<=20000 or not 0<args.case_timeout<=3600:parser.error('Invalid bounded limits')
    if len(set(args.case_ids))!=len(args.case_ids) or len(set(args.sparse_modes))!=len(args.sparse_modes):parser.error('Duplicate cases/modes')
    cases,excluded=load_source(args.source_run,args.case_ids)
    files=[SCRIPT,Path(collector.__file__),Path(collector.observables.__file__),Path(duration.__file__),Path(horizon.__file__),Path(oracle.__file__),args.source_run/'manifest.json']
    files+=horizon.evaluation.binding.manifest_code_paths()
    frozen={str(path):digest(path) for path in files}
    for case in cases:frozen.update(case['recovery_source_sha256']);frozen.update(case['source_files_sha256'])
    manifest=dict(schema_version=1,protocol=__doc__,cases=cases,excluded=excluded,source_sha256=frozen,
        terminal_hold_ticks=args.terminal_hold_ticks,sparse_modes=args.sparse_modes,max_suffix_ticks=args.max_suffix_ticks,case_timeout=args.case_timeout,
        source_run=str(args.source_run),dataset_ready=False,model_loaded=False)
    args.output.mkdir(parents=True,exist_ok=False)
    horizon.write_json(args.output/'manifest.json',manifest)
    if args.mode=='prepare':
        horizon.write_json(args.output/'preflight.json',dict(status='passed' if cases else 'no_eligible_source',gpu_used=False,
            eligible=[c['id'] for c in cases],excluded=excluded))
        print('PREPARED',args.output);return
    records=[]
    for source_case in cases:
        directory=args.output/source_case['id'];directory.mkdir()
        sparse={}
        for mode in ['dense']+args.sparse_modes:
            for path,expected in frozen.items():
                if digest(path)!=expected:raise ValueError(f'Frozen input changed: {path}')
            case=dict(source_case,replay_mode=mode,terminal_hold_ticks=args.terminal_hold_ticks)
            if mode!='dense':case['uniform_captures_path']=str(directory/'dense/uniform_captures.jsonl')
            with horizon.evaluation.binding.replaced(collector,'SCRIPT',SCRIPT):
                result=collector.run_case(case,directory/mode,args)
            if mode=='dense':
                dense=result
                if not (result.get('success') is True and result.get('exact_prefix') is True and result.get('dense_source_match') is True):break
            else:sparse[mode]=result
            print('SUFFIX_REPLAY',json.dumps({k:result.get(k) for k in ('id','replay_mode','status','success','exact_prefix','error')}),flush=True)
        records.append(readiness(directory,source_case,dense,sparse))
        horizon.atomic_json(args.output/'progress.json',dict(recorded=len(records),requested=len(cases),records=records))
    horizon.write_json(args.output/'summary.json',dict(records=records,excluded=excluded,
        note='Privileged recovery replay feasibility only; no policy benchmark or training performed'))


if __name__=='__main__':sys.exit(main())
