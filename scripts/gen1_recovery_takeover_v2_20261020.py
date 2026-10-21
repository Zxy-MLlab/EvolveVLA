#!/usr/bin/env python3
"""Replay a frozen failed-policy prefix, then collect an expert-only suffix.

Development takeover boundary = recorded opening interval start minus20 actions.
This is privileged expert recoverability, not a policy success benchmark. No
model loads; simulation is launched only by explicit `run`, never by `prepare`.
"""
import argparse
import contextlib
import importlib
import hashlib
import traceback
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import gen1_execution_horizon_ablation_20261019 as horizon
import gen1_expert_label_replay_20261019 as oracle
import gen1_recovery_observables_20261020 as observables

ROOT = horizon.ROOT
SOURCE = ROOT/'robotwin_repo_ba0fd21/envs/_base_task.py'
PANEL = ROOT/'outputs/gen1_causal_diagnosis_20261019/panel.json'
PRIOR = ROOT/'outputs/gen1_timing_validation_20261020/candidate_policy_ab'
PHASE = ROOT/'outputs/gen1_timing_validation_20261020/policy_fallback_phase_audit.json'
IDS = ('place_mouse_pad_59008', 'hanging_mug_88038', 'hanging_mug_87749')


def digest(path):
    """Streaming SHA256 compatible with the simulator's Python 3.10."""
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


class SimulatorCaseError(RuntimeError):
    def __init__(self, payload):
        self.payload = payload
        super().__init__(payload.get('message', payload.get('status', 'Unknown case_status')))


def require_step(response, directory):
    if response.get('type') == 'case_status':
        horizon.atomic_json(directory/'ipc_case_status.json', response)
        raise SimulatorCaseError(response)
    if response.get('type') != 'step':
        horizon.atomic_json(directory/'ipc_unexpected_response.json', response)
        raise RuntimeError(f'Unexpected IPC response: {response.get("type")}')
    return response


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def differences(expected, actual, prefix=''):
    """Exact recursive comparison, including missing/unexpected fields."""
    if type(expected) is not type(actual):
        return [prefix + ':type']
    if isinstance(expected, dict):
        result = [prefix + '/' + k + ':missing_or_extra' for k in sorted(expected.keys() ^ actual.keys())]
        for key in sorted(expected.keys() & actual.keys()):
            result.extend(differences(expected[key], actual[key], prefix + '/' + key))
        return result
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [prefix + ':length']
        return [p for i, (a, b) in enumerate(zip(expected, actual))
                for p in differences(a, b, prefix + '/' + str(i))]
    return [] if expected == actual else [prefix]


def prepare_cases(panel=PANEL, phase=PHASE, prior=PRIOR, ids=IDS):
    with horizon.evaluation.binding.replaced(horizon, 'digest', digest):
        cases = horizon.load_cases(panel, ids)
    records = read(phase)['records']
    output = []
    for case in cases:
        selected = [r for r in records if r['id'] == case['id'] and
                    r['model'] == 'candidate' and r['timing_mode'] == 'original']
        if len(selected) != 1:
            raise ValueError('Missing or ambiguous original phase record')
        audit = selected[0]['phase_audit']
        opening = audit['first_terminal_opening_command_interval']['start_boundary_step']
        takeover = opening - 20
        directory = prior/case['id']/'original'
        result = read(directory/'result.json')
        timing, physical = lines(directory/'timing.jsonl'), lines(directory/'physical.jsonl')
        if result['status'] != 'completed' or result['success'] is not False or result['simulator_returncode'] != 0:
            raise ValueError('Recovery source must be an actually completed policy failure')
        if not 0 < takeover < result['steps'] or not takeover < result['simulator_limits']['official_step_limit']:
            raise ValueError('Takeover must precede source termination and official action limit')
        if [r['action_index'] for r in timing] != list(range(result['steps'])):
            raise ValueError('Source timing trace incomplete')
        if [r['step'] for r in physical] != list(range(result['steps'] + 1)):
            raise ValueError('Source physical trace incomplete')
        arm = audit['active_arm']
        if arm not in ('left', 'right') or (case['task'] == 'hanging_mug' and arm != 'right'):
            raise ValueError('Invalid terminal arm')
        if physical[takeover]['physical'][arm]['drive_targets'][-1] > .2:
            raise ValueError('Prespecified takeover boundary is not closed')
        paths = [directory/name for name in ('result.json','timing.jsonl','physical.jsonl','expert_gate.json')]
        output.append(dict(case, takeover_boundary_step=takeover, opening_start_boundary_step=opening,
            active_arm=arm, source_policy_result=str(directory/'result.json'),
            source_checkpoint=result['checkpoint'], source_checkpoint_identity_sha256=result['checkpoint_identity_sha256'],
            source_input_hashes=result['hashes'], source_binding=result['binding'],
            prefix_targets=[r['target'] for r in timing[:takeover]],
            prefix_physics_steps=[r['physics_steps'] for r in timing[:takeover]],
            prefix_physical=[r['physical'] for r in physical[:takeover + 1]],
            trigger_target=timing[takeover]['target'],
            source_files_sha256={str(p): digest(p) for p in paths}))
    return output


class PrefixMismatch(ValueError):
    pass


class RecoveryLimit(RuntimeError):
    pass


class PhysicsAudit:
    """Observe each source scene.step after it returns; never add physics calls."""
    def __init__(self, source=SOURCE, after_step=None, limit=None):
        self.source = str(source.resolve())
        self.step_lines = {i for i, line in enumerate(source.read_text().splitlines(), 1)
                           if line.strip() == 'self.scene.step()'}
        self.after_step, self.limit = after_step, limit
        self.count, self.pending = 0, set()

    def trace(self, frame, event, arg):
        if frame.f_code.co_filename != self.source:
            return None
        key = id(frame)
        if key in self.pending:
            self.pending.remove(key)
            if event != 'exception':
                self.count += 1
                if self.after_step:
                    self.after_step(self.count)
        if event == 'line' and frame.f_lineno in self.step_lines:
            if self.limit is not None and self.count >= self.limit:
                raise RecoveryLimit(f'Expert suffix reached {self.limit} physics ticks')
            self.pending.add(key)
        return self.trace


def predicates(task, task_name):
    import numpy as np
    if task_name == 'place_mouse_pad':
        p, q = task.mouse.get_pose().p, np.abs(task.mouse.get_pose().q)
        error = p[:2] - task.target.get_pose().p[:2]
        return dict(position=bool(np.all(np.abs(error) < np.array([.015,.012]))),
            orientation=bool(abs(q[2]*q[3]-.49)<.015 or abs(q[0]*q[1]-.49)<.015),
            left_open=bool(task.robot.is_left_gripper_open()), right_open=bool(task.robot.is_right_gripper_open()))
    point = task.mug.get_functional_point(0)[:3]
    middle = (task.rack.get_pose().p + task.rack.get_functional_point(0)[:3])/2
    return dict(functional_xy=bool(np.all(np.abs((point-middle)[:2])<.02)),
                functional_height=bool(point[2]>.86), right_open=bool(task.is_right_gripper_open()))


def native_suffix(task, case, arm_factory=None):
    if arm_factory is None:
        from envs.utils import ArmTag
        arm_factory = ArmTag
    arm = arm_factory(case['active_arm'])
    if case['task'] == 'place_mouse_pad':
        task.move(task.place_actor(task.mouse, arm_tag=arm, target_pose=task.target_pose,
                                  constrain='align', pre_dis=.07, dis=.005))
    else:
        task.move(task.place_actor(task.mug, arm_tag=arm, target_pose=task.rack.get_functional_point(0),
            functional_point_id=0, constrain='align', pre_dis=.05, dis=-.05, pre_dis_axis='fp'))
        if task.plan_success:
            task.move(task.move_by_displacement(arm_tag=arm, z=.1, move_axis='arm'))


def collect_suffix(task, case, directory, max_ticks, suffix=native_suffix):
    """Called only after exact prefix validation; owns only a new suffix directory."""
    before = oracle.physical(task)
    result = dict(status='recovery_error', success=False, final_success=False, ever_success=False,
                  takeover_boundary_step=case['takeover_boundary_step'], before_physical=before, before_observables=observables.observe(task),
                  active_arm=case['active_arm'], dataset_ready=False,
                  label_scope='expert suffix only; policy prefix excluded; deployment replay not yet verified')
    if before[case['active_arm']]['drive_targets'][-1] > .2:
        result['status'] = 'takeover_not_closed'
        return result
    if task.eval_success or bool(task.check_success()):
        result['status'] = 'takeover_already_successful'
        return result
    if task.plan_success is not True:
        result['status'] = 'takeover_plan_state_invalid'
        return result
    if task.take_action_cnt != case['takeover_boundary_step'] or task.take_action_cnt >= task.step_lim:
        result['status'] = 'takeover_counter_mismatch'
        return result
    directory.mkdir(exist_ok=False)
    raw = directory/'raw'
    raw.mkdir()
    reset_fields = ('need_plan','plan_success','left_cnt','right_cnt','FRAME_IDX','save_freq','save_data')
    result['planning_reset'] = dict(before={name:getattr(task,name) for name in reset_fields},
        old_left_paths=len(task.left_joint_path), old_right_paths=len(task.right_joint_path),
        scope='Clear only task path lists/cursors; preserve articulation, actors, planner objects and physics state')
    task.need_plan, task.plan_success = True, True
    task.left_joint_path, task.right_joint_path = [], []
    task.left_cnt = task.right_cnt = task.FRAME_IDX = 0
    task.save_freq, task.save_data = 15, False
    result['planning_reset']['after'] = {name:getattr(task,name) for name in reset_fields}
    old_capture = task._take_picture
    captures = 0
    capture_attempts = 0
    result['collection_errors'] = []
    last_success = False
    audit = None
    with (directory/'captures.jsonl').open('x', buffering=1) as captured, \
         (directory/'predicates.jsonl').open('x', buffering=1) as checks:
        def observe_tick(tick):
            nonlocal last_success
            last_success = bool(task.check_success())
            components = predicates(task, case['task'])
            if last_success != all(components.values()):
                raise ValueError('Independent task predicate reconstruction disagrees')
            if last_success and not result['ever_success']:
                result['first_success_physics_tick'] = tick
            result['ever_success'] |= last_success
            checks.write(json.dumps(dict(physics_tick=tick, success=last_success, components=components))+'\n')
        audit = PhysicsAudit(after_step=observe_tick, limit=max_ticks)
        def capture():
            nonlocal captures, capture_attempts
            # Reserve an index before I/O so cleanup never overwrites a partial capture.
            capture_index = capture_attempts
            capture_attempts += 1
            observation = task.get_obs()
            target = raw/f'{capture_index:07d}.pkl' 
            with target.open('xb') as stream:
                pickle.dump(observation, stream, protocol=pickle.HIGHEST_PROTOCOL)
            row = dict(index=capture_index, physics_tick=audit.count, simulation_seconds=audit.count*float(task.scene.get_timestep()),
                path=str(target), sha256=digest(target), physical=oracle.physical(task), observables=observables.observe(task),
                success=bool(task.check_success()), components=predicates(task,case['task']),
                provenance='expert_suffix', observation_semantics='native get_obs; joint_action is drive target')
            captured.write(json.dumps(row, allow_nan=False)+'\n')
            captures += 1
            task.FRAME_IDX = captures
        task._take_picture = capture
        previous_trace = sys.gettrace()
        try:
            capture()
            sys.settrace(audit.trace)
            suffix(task, case)
            result['status'] = 'recovery_completed' if task.plan_success else 'recovery_plan_failed'
        except RecoveryLimit as exc:
            result.update(status='recovery_physics_limit', error=str(exc))
        except Exception as exc:
            result.update(status='recovery_error', error=f'{type(exc).__name__}: {exc}',
                          traceback=traceback.format_exc())
        finally:
            sys.settrace(previous_trace)
            def cleanup_error(stage, exc):
                result['collection_errors'].append(dict(stage=stage, error=f'{type(exc).__name__}: {exc}',
                                                       traceback=traceback.format_exc()))
                result['status'] = 'recovery_error'
            try:
                capture()
            except Exception as exc:
                cleanup_error('final_capture', exc)
            task._take_picture = old_capture
            trajectory = directory/'planner_paths.pkl'
            try:
                with trajectory.open('xb') as stream:
                    pickle.dump(dict(left_joint_path=task.left_joint_path, right_joint_path=task.right_joint_path),stream)
                result.update(planner_paths=str(trajectory), planner_paths_sha256=digest(trajectory))
            except Exception as exc:
                cleanup_error('save_planner_paths', exc)
            result.update(final_success=bool(task.check_success()), plan_success=bool(task.plan_success),
                physics_ticks=audit.count, captures=captures, capture_attempts=capture_attempts,
                timestep=float(task.scene.get_timestep()), after_physical=oracle.physical(task), after_observables=observables.observe(task),
                final_predicates=predicates(task,case['task']))
            result['success'] = result['status']=='recovery_completed' and result['plan_success'] and result['final_success']
            task.eval_success = result['success']
    horizon.atomic_json(directory/'result.json',result)
    return result


def simulator():
    import numpy as np
    import robotwin_eval_harness as harness
    case = read(os.environ['GEN1_RECOVERY_CASE'])
    directory = Path(os.environ['ROBOTWIN_CONTROLLED_OUTPUT'])
    max_ticks = int(os.environ['GEN1_RECOVERY_MAX_TICKS'])
    original_import = importlib.import_module
    binding = horizon.evaluation.binding
    with contextlib.ExitStack() as stack:
        prefix = stack.enter_context((directory/'prefix_verification.jsonl').open('x',buffering=1))
        attached = False
        def hooked(name,package=None):
            nonlocal attached
            module = original_import(name,package)
            if name != 'eval_policy_xpolicylab' or attached:
                return module
            attached = True
            construct = module.class_decorator
            def make(*args,**kwargs):
                task = construct(*args,**kwargs)
                original_take, original_play, original_check = task.take_action,task.play_once,task.check_success
                gate = dict(status='not_run',bypassed=False)
                pending = False
                index, recovery = 0, None
                def play(*a,**kw):
                    nonlocal pending
                    value = original_play(*a,**kw)
                    pending = bool(task.plan_success)
                    gate.update(status='pending_predicate' if pending else 'rejected',plan_success=bool(task.plan_success),check_success=None)
                    horizon.atomic_json(directory/'expert_gate.json',gate)
                    return value
                def check(*a,**kw):
                    nonlocal pending
                    value = original_check(*a,**kw)
                    if pending:
                        pending = False
                        gate.update(status='passed' if value else 'rejected',check_success=bool(value))
                        horizon.atomic_json(directory/'expert_gate.json',gate)
                    return value
                def verify_boundary(step):
                    current = oracle.physical(task)
                    drift = differences(case['prefix_physical'][step],current)
                    prefix.write(json.dumps(dict(step=step, exact=not drift, differences=drift, physical=current))+'\n')
                    if drift:
                        horizon.atomic_json(directory/'prefix_error.json',dict(status='prefix_mismatch',step=step,differences=drift))
                        raise PrefixMismatch(f'Physical mismatch at boundary{step}: {drift[:4]}')
                def take(action,*a,**kw):
                    nonlocal index,recovery
                    if gate.get('status')!='passed':
                        raise PrefixMismatch('Expert gate not passed')
                    if index==0:
                        verify_boundary(0)
                    if index<case['takeover_boundary_step']:
                        target=list(map(float,np.asarray(action,dtype=np.float32)))
                        if target!=case['prefix_targets'][index]:
                            raise PrefixMismatch('Requested prefix target differs')
                        timer=PhysicsAudit()
                        previous=sys.gettrace()
                        sys.settrace(timer.trace)
                        try: original_take(action,*a,**kw)
                        finally: sys.settrace(previous)
                        if timer.count!=case['prefix_physics_steps'][index]:
                            raise PrefixMismatch(f'Physics duration mismatch at action{index}')
                        index+=1
                        verify_boundary(index)
                    elif index==case['takeover_boundary_step'] and recovery is None:
                        # IPC sentinel carries the recorded next target for audit,
                        # but that policy action is deliberately not executed.
                        if list(map(float,np.asarray(action,dtype=np.float32)))!=case['trigger_target']:
                            raise PrefixMismatch('Wrong takeover trigger')
                        verify_boundary(index)
                        try:
                            recovery=collect_suffix(task,case,directory/'expert_suffix',max_ticks)
                        except Exception as exc:
                            horizon.atomic_json(directory/'takeover_exception.json',dict(
                                error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                                prefix_actions_executed=index))
                            raise
                        horizon.atomic_json(directory/'takeover_result.json',recovery)
                    else:
                        raise RuntimeError('No policy actions allowed after expert takeover')
                for attr,value in [('play_once',play),('check_success',check),('take_action',take)]:
                    stack.enter_context(binding.replaced(task,attr,value))
                transmit=harness.send_message
                def send(channel,payload):
                    extra=dict(recovery_gate=dict(gate))
                    if payload.get('type') in ('episode_start','step'):
                        extra.update(recovery_physical=oracle.physical(task),takeover=recovery,prefix_actions_executed=index)
                    return transmit(channel,dict(payload,**extra))
                stack.enter_context(binding.replaced(harness,'send_message',send))
                return task
            stack.enter_context(binding.replaced(module,'class_decorator',make))
            return module
        stack.enter_context(binding.replaced(importlib,'import_module',hooked))
        return harness.main()


def run_case(case,directory,args):
    import h5py
    import numpy as np
    from robotwin_eval_harness import send_message,receive_message
    from robotwin_public_openpi_harness import _configure_sim_renderer_env
    directory.mkdir(parents=True,exist_ok=False)
    horizon.write_json(directory/'case.json',case)
    env=horizon.evaluation.binding.runtime_environment()
    env.update(GEN1_RECOVERY_CASE=str(directory/'case.json'),GEN1_RECOVERY_MAX_TICKS=str(args.max_suffix_ticks),
               ROBOTWIN_CONTROLLED_OUTPUT=str(directory))
    _configure_sim_renderer_env(env)
    command=[horizon.evaluation.c.CONFIG['sim_python'],'-I','-u','-B',str(SCRIPT),'sim',
        '--role','sim','--controlled-case','--episodes','1','--task',case['task'],
        '--seed',str(case['seed']),'--episode-index',str(case['episode_index']),
        '--episode-steps','2000','--robotwin-root',horizon.evaluation.c.CONFIG['robotwin_root'],
        '--task-config','demo_clean','--first-instruction',case['prompt'],'--first-instruction-seed',str(case['seed'])]
    horizon.write_json(directory/'command.json',dict(command=command))
    result=dict(id=case['id'],task=case['task'],seed=case['seed'],episode_index=case['episode_index'],
        status='runner_error',success=None,source_hdf5_sha256=case['source_hdf5_sha256'],
        takeover_boundary_step=case['takeover_boundary_step'],dataset_ready=False)
    started=time.monotonic()
    with (directory/'simulator.log').open('x') as log:
        child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=env,
            cwd=horizon.evaluation.c.CONFIG['robotwin_root'],start_new_session=True)
        try:
            with horizon.case_timeout(args.case_timeout):
                first=receive_message(child.stdout)
                result['expert_gate']=first.get('recovery_gate')
                if first.get('seed')!=case['seed']:
                    raise PrefixMismatch('Wrong returned scene seed')
                if first['type']=='case_status':
                    require_step(first, directory)
                else:
                    if first['type']!='episode_start' or first.get('instruction')!=case['prompt']:
                        raise PrefixMismatch('Wrong initial IPC type or instruction')
                    if result['expert_gate'].get('status')!='passed' or result['expert_gate'].get('bypassed') is not False:
                        raise PrefixMismatch('Missing/invalid expert admission')
                    images,state=oracle.observation_arrays(first['observation'])
                    states,_=oracle.source_arrays(case)
                    with h5py.File(case['source_hdf5']) as source:
                        metrics=oracle.source_image_metrics(source,0,images)
                    exact=bool(np.array_equal(state,states[0])) and all(x['jpeg_bytes_equal'] for x in metrics.values())
                    initial=dict(state_exact=bool(np.array_equal(state,states[0])),images=metrics,source_exact=exact,
                        initial_physical_exact=not differences(case['prefix_physical'][0],first['recovery_physical']))
                    horizon.write_json(directory/'initial_parity.json',initial)
                    if not exact or not initial['initial_physical_exact']:
                        raise PrefixMismatch('Initial source or physical mismatch')
                    hashes=horizon.evaluation.c.capture(directory/'initial',images,state,case['prompt'])
                    if hashes!=case['source_input_hashes']:
                        raise PrefixMismatch('Initial original policy input hashes differ')
                    for index,target in enumerate(case['prefix_targets']+[case['trigger_target']]):
                        send_message(child.stdin,dict(cmd='action',action=target))
                        response=receive_message(child.stdout)
                        require_step(response, directory)
                        if index<len(case['prefix_targets']):
                            if response.get('terminated') or response.get('truncated'):
                                raise PrefixMismatch('Policy replay terminated before takeover')
                        else:
                            recovery=response.get('takeover')
                            if not recovery:
                                raise RuntimeError('Takeover did not produce a record')
                            result.update(recovery)
                            result['harness_success']=bool(response['info']['is_success'])
                            result['source_initial_parity']=initial
        except SimulatorCaseError as exc:
            result.update(status=exc.payload.get('status','simulator_error'), success=None,
                          error=str(exc), ipc_case_status=exc.payload)
        except PrefixMismatch as exc:
            result.update(status='prefix_mismatch',success=None,error=str(exc))
        except Exception as exc:
            result.update(status='runner_error',success=None,error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        finally:
            try: send_message(child.stdin,dict(cmd='stop'));child.stdin.close()
            except (BrokenPipeError,OSError): pass
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
            result.update(simulator_returncode=child.returncode,elapsed_seconds=time.monotonic()-started)
            if child.returncode:
                result.update(status='simulator_error',success=None)
            if (directory/'prefix_error.json').exists():
                result.update(status='prefix_mismatch',success=None,prefix_error=read(directory/'prefix_error.json'))
            rows = lines(directory/'prefix_verification.jsonl') if (directory/'prefix_verification.jsonl').exists() else []
            result['exact_prefix'] = bool(rows and all(r['exact'] for r in rows) and
                                          rows[-1]['step']==case['takeover_boundary_step'] and
                                          result.get('source_initial_parity',{}).get('source_exact'))
            horizon.atomic_json(directory/'result.json',result)
    return result


def main():
    if len(sys.argv)>1 and sys.argv[1]=='sim':
        del sys.argv[1]
        return simulator()
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-ids',nargs='+',choices=IDS,default=list(IDS))
    parser.add_argument('--max-suffix-ticks',type=int,default=10000)
    parser.add_argument('--case-timeout',type=int,default=1800)
    args=parser.parse_args()
    if not 0<args.max_suffix_ticks<=20000 or not 0<args.case_timeout<=3600:
        parser.error('Invalid bounded recovery limits')
    cases=prepare_cases(ids=args.case_ids)
    files=[SCRIPT,Path(horizon.__file__),Path(oracle.__file__),Path(observables.__file__),PANEL,PHASE]
    files+=horizon.evaluation.binding.manifest_code_paths()
    for case in cases: files += [Path(p) for p in case['source_files_sha256']]
    manifest=dict(protocol=__doc__,cases=cases,source_sha256={str(p):digest(p) for p in sorted(set(files))},
        takeover_rule='phase first_terminal_opening_command_interval.start_boundary_step minus20 actions',
        max_suffix_ticks=args.max_suffix_ticks,case_timeout=args.case_timeout,model_loaded=False,
        dataset_ready=False,primary='expert plan completes and final unchanged official predicate passes',
        limitations=['Development cases selected from known training failures.',
            'Exact recorded physical fields do not include all hidden solver/contact state.',
            'Dense expert suffix is privileged; it has not passed deployment-controller replay.',
            'Closure is a command condition, not proof of grasp/contact.',
            'Dense suffix physics budget differs from policy action budget; not a benchmark comparison.'])
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    horizon.write_json(args.output/'manifest.json',manifest)
    if args.mode=='prepare':
        horizon.write_json(args.output/'preflight.json',dict(status='passed',gpu_used=False,
            cases=[dict(id=c['id'],takeover=c['takeover_boundary_step'],arm=c['active_arm']) for c in cases]))
        print('PREPARED',args.output);return
    records=[]
    for case in cases:
        for path,expected in manifest['source_sha256'].items():
            if digest(path)!=expected: raise ValueError(f'Frozen input changed: {path}')
        result=run_case(case,args.output/case['id'],args);records.append(result)
        horizon.atomic_json(args.output/'progress.json',dict(requested=len(cases),recorded=len(records),records=records))
        print('RECOVERY_RESULT',json.dumps({k:result.get(k) for k in ('id','status','success','final_success','physics_ticks','error')}),flush=True)
    horizon.write_json(args.output/'summary.json',dict(requested=len(cases),records=records,dataset_ready=False,
        note='No retries; prefix mismatches/admission/errors are excluded, not policy failures.'))


if __name__=='__main__': sys.exit(main())
