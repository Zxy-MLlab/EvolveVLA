"""CPU checks for replay boundaries and expert suffix collection semantics."""
import ast
import contextlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import gen1_recovery_takeover_v2_20261020 as recovery


def native_dense():
    tree=ast.parse(recovery.SOURCE.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='take_dense_action')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(recovery.SOURCE),'exec'),namespace)
    return namespace['take_dense_action']


def fake_task():
    scene=types.SimpleNamespace(count=0,get_timestep=lambda:.004)
    def step():scene.count+=1
    scene.step=step
    robot=types.SimpleNamespace(set_arm_joints=lambda *a:None,set_gripper=lambda *a:None)
    task=types.SimpleNamespace(scene=scene,robot=robot,take_action_cnt=124,step_lim=400,
        eval_success=False,plan_success=True,need_plan=False,left_cnt=2,right_cnt=3,
        left_joint_path=[{'stale':True}],right_joint_path=[{'stale':True}],FRAME_IDX=999,
        save_freq=15,save_data=False,render_freq=0,_take_picture=lambda:None,_update_render=lambda:None)
    task.check_success=lambda:scene.count>=3
    task.get_obs=lambda:dict(observation={'native_rgb':np.zeros((2,2,3),np.uint8)},
                             joint_action={'vector':np.ones(14)*scene.count})
    task.take_dense_action=types.MethodType(native_dense(),task)
    return task


def physical(task,gripper=0.):
    return {'left':{'drive_targets':[0.]*6+[1.]},
            'right':{'drive_targets':[0.]*6+[gripper]},'tick':task.scene.count}


class RecoveryTests(unittest.TestCase):
    def test_streaming_digest_works_without_python311_file_digest(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(b'abc')
            self.assertEqual(recovery.digest(p),hashlib.sha256(b'abc').hexdigest())

    def test_case_status_preserves_full_payload_and_message(self):
        payload=dict(type='case_status',seed=123,status='simulator_error',
                     message='AttributeError: missing file_digest',extra={'context':'raw'})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(recovery.SimulatorCaseError) as caught:
                recovery.require_step(payload,Path(d))
            self.assertEqual(caught.exception.payload,payload)
            self.assertEqual(json.loads((Path(d)/'ipc_case_status.json').read_text()),payload)
            self.assertIn('missing file_digest',str(caught.exception))

    def test_capture_hash_error_is_not_masked_by_duplicate_file(self):
        with patch.object(recovery,'digest',side_effect=RuntimeError('injected hash failure')):
            result,task,path,seen=self.collect()
        self.assertEqual(result['status'],'recovery_error')
        self.assertEqual(result['error'],'RuntimeError: injected hash failure')
        self.assertIn('injected hash failure',result['traceback'])
        self.assertEqual(result['captures'],0)
        self.assertEqual(result['capture_attempts'],2)
        self.assertTrue((path/'raw/0000000.pkl').exists())
        self.assertTrue((path/'raw/0000001.pkl').exists())
        self.assertEqual([e['stage'] for e in result['collection_errors']],['final_capture','save_planner_paths'])
        self.assertTrue((path/'result.json').exists())

    def test_exact_comparison_catches_nested_missing_and_numeric_drift(self):
        a={'left':{'q':[1.,2.]},'object':[0.,1.]}
        self.assertEqual(recovery.differences(a,a),[])
        b={'left':{'q':[1.,2.00000000000001]},'object':[0.,1.]}
        self.assertEqual(recovery.differences(a,b),['/left/q/1'])
        self.assertTrue(recovery.differences(a,{'left':{'q':[1.,2.]}}))
        self.assertTrue(recovery.differences({'x':False},{'x':0}))

    def test_fixed_development_boundaries_and_closed_arms_match_existing_files(self):
        cases=recovery.prepare_cases()
        self.assertEqual([(c['id'],c['takeover_boundary_step'],c['active_arm']) for c in cases],
            [('place_mouse_pad_59008',124,'right'),('hanging_mug_88038',281,'right'),('hanging_mug_87749',291,'right')])
        for c in cases:
            n=c['takeover_boundary_step']
            self.assertEqual(len(c['prefix_targets']),n)
            self.assertEqual(len(c['prefix_physical']),n+1)
            self.assertEqual(n,c['opening_start_boundary_step']-20)
            self.assertLessEqual(c['prefix_physical'][-1][c['active_arm']]['drive_targets'][-1],.2)

    def collect(self,limit=100,fail_plan=False,gripper=0.):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        output=Path(temporary.name)/'suffix'
        task=fake_task();case=dict(task='place_mouse_pad',active_arm='right',takeover_boundary_step=124)
        observed={}
        def suffix(t,c):
            observed.update(need_plan=t.need_plan,left=list(t.left_joint_path),right=list(t.right_joint_path),
                left_cnt=t.left_cnt,right_cnt=t.right_cnt,take_action_cnt=t.take_action_cnt)
            if fail_plan:
                t.plan_success=False
                return
            trajectory=dict(position=np.zeros((5,6)),velocity=np.zeros((5,6)))
            t.right_joint_path.append(trajectory)
            t.take_dense_action(dict(left_arm=None,left_gripper=None,right_arm=trajectory,right_gripper=None))
        with patch.object(recovery.oracle,'physical',lambda t:physical(t,gripper)), \
             patch.object(recovery,'predicates',lambda t,name:{'original':bool(t.check_success())}):
            result=recovery.collect_suffix(task,case,output,limit,suffix=suffix)
        return result,task,output,observed

    def test_expert_suffix_has_clean_queues_and_native_observations_only(self):
        result,task,path,seen=self.collect()
        self.assertEqual(seen,dict(need_plan=True,left=[],right=[],left_cnt=0,right_cnt=0,take_action_cnt=124))
        self.assertEqual(task.take_action_cnt,124)
        self.assertEqual(result['physics_ticks'],5)
        self.assertTrue(result['success']);self.assertTrue(result['ever_success'])
        self.assertEqual(result['first_success_physics_tick'],3)
        self.assertFalse(result['dataset_ready'])
        ticks=[json.loads(x) for x in (path/'predicates.jsonl').read_text().splitlines()]
        self.assertEqual([r['physics_tick'] for r in ticks],[1,2,3,4,5])
        captures=[json.loads(x) for x in (path/'captures.jsonl').read_text().splitlines()]
        self.assertEqual(captures[0]['physics_tick'],0)
        self.assertEqual(captures[-1]['physics_tick'],5)
        self.assertTrue(all(r['provenance']=='expert_suffix' for r in captures))
        with Path(captures[0]['path']).open('rb') as stream:raw=pickle.load(stream)
        np.testing.assert_array_equal(raw['joint_action']['vector'],np.zeros(14))
        with (path/'planner_paths.pkl').open('rb') as stream:paths=pickle.load(stream)
        self.assertEqual(paths['left_joint_path'],[])
        self.assertEqual(len(paths['right_joint_path']),1)

    def test_physics_limit_is_enforced_before_extra_step_and_not_success(self):
        result,task,path,_=self.collect(limit=3)
        self.assertEqual(task.scene.count,3)
        self.assertEqual(result['status'],'recovery_physics_limit')
        self.assertTrue(result['final_success']);self.assertTrue(result['ever_success'])
        self.assertFalse(result['success']);self.assertFalse(task.eval_success)
        self.assertEqual(result['physics_ticks'],3)

    def test_plan_failure_retained_and_never_marked_accepted(self):
        result,task,path,_=self.collect(fail_plan=True)
        self.assertEqual(result['status'],'recovery_plan_failed')
        self.assertFalse(result['success']);self.assertFalse(result['plan_success'])
        self.assertEqual(result['physics_ticks'],0)
        self.assertTrue((path/'planner_paths.pkl').exists())

    def test_open_takeover_never_collects_or_changes_planning_state(self):
        result,task,path,seen=self.collect(gripper=.25)
        self.assertEqual(result['status'],'takeover_not_closed')
        self.assertFalse(path.exists());self.assertEqual(seen,{})
        self.assertFalse(task.need_plan)
        self.assertEqual(task.right_joint_path,[{'stale':True}])

    def test_native_suffix_skips_regrasp_and_uses_current_target(self):
        calls=[]
        t=types.SimpleNamespace(mouse='mouse',target_pose='pad',mug='mug',plan_success=True,
            rack=types.SimpleNamespace(get_functional_point=lambda i:'live_rack_fp'),
            place_actor=lambda *a,**kw:('place',a,kw),move_by_displacement=lambda **kw:('withdraw',kw),
            move=lambda x:calls.append(x))
        recovery.native_suffix(t,{'task':'place_mouse_pad','active_arm':'right'},arm_factory=str)
        self.assertEqual(len(calls),1);self.assertEqual(calls[0][2]['target_pose'],'pad')
        calls.clear()
        recovery.native_suffix(t,{'task':'hanging_mug','active_arm':'right'},arm_factory=str)
        self.assertEqual(len(calls),2);self.assertEqual(calls[0][2]['target_pose'],'live_rack_fp')
        self.assertEqual(calls[0][2]['functional_point_id'],0)
        self.assertEqual(calls[1],('withdraw',dict(arm_tag='right',z=.1,move_axis='arm')))

    def hooks(self,reject=False,drift=False):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        output=Path(temporary.name)
        task=fake_task();task.take_action_cnt=0
        task.plan_success=not reject
        task.play_once=lambda:None
        task.check_success=lambda:task.take_action_cnt==0 and not reject
        performed=[];taken=[]
        def original(action,*a,**kw):
            performed.append(list(map(float,action)))
            task.take_action_cnt+=1;task.scene.count+=1
        task.take_action=original
        case=dict(task='place_mouse_pad',active_arm='right',takeover_boundary_step=2,
                  prefix_targets=[[0.]*14,[1.]*14],prefix_physics_steps=[0,0],trigger_target=[2.]*14,
                  prefix_physical=[dict(physical(task),tick=n) for n in range(3)])
        if drift:case['prefix_physical'][1]['tick']=99
        case_path=output/'case.json';case_path.write_text(json.dumps(case))
        module=types.SimpleNamespace(class_decorator=lambda *a,**kw:task)
        original_import=recovery.importlib.import_module
        def load(name,package=None):
            return module if name=='eval_policy_xpolicylab' else original_import(name,package)
        import robotwin_eval_harness as harness
        def execute():
            obj=recovery.importlib.import_module('eval_policy_xpolicylab').class_decorator('place_mouse_pad')
            obj.play_once()
            if not obj.plan_success:return
            self.assertTrue(obj.check_success())
            harness.send_message(None,{'type':'episode_start'})
            for action in case['prefix_targets']+[case['trigger_target']]:
                obj.take_action(np.asarray(action,np.float32))
                harness.send_message(None,{'type':'step'})
        def suffix(task,case,directory,max_ticks):
            taken.append(dict(count=task.take_action_cnt,path=str(directory)))
            return dict(status='recovery_completed',success=True)
        messages=[]
        with patch.dict(os.environ,GEN1_RECOVERY_CASE=str(case_path),GEN1_RECOVERY_MAX_TICKS='100',
                        ROBOTWIN_CONTROLLED_OUTPUT=str(output)), \
             patch.object(recovery.importlib,'import_module',load), \
             patch.object(harness,'main',execute), \
             patch.object(harness,'send_message',lambda channel,row:messages.append(row)), \
             patch.object(recovery.oracle,'physical',physical), \
             patch.object(recovery,'collect_suffix',suffix):
            if drift:
                with self.assertRaises(recovery.PrefixMismatch):recovery.simulator()
            else:recovery.simulator()
        return performed,taken,messages,output

    def test_simulator_hooks_execute_exact_prefix_and_intercept_next_policy_action(self):
        actions,taken,messages,path=self.hooks()
        self.assertEqual(actions,[[0.]*14,[1.]*14])
        self.assertEqual(taken[0]['count'],2)
        self.assertEqual(messages[-1]['takeover']['status'],'recovery_completed')
        self.assertEqual(messages[-1]['prefix_actions_executed'],2)
        self.assertEqual(json.loads((path/'expert_gate.json').read_text())['status'],'passed')

    def test_simulator_hook_drift_prevents_takeover_and_persists_mismatch(self):
        actions,taken,messages,path=self.hooks(drift=True)
        self.assertEqual(len(actions),1);self.assertEqual(taken,[])
        self.assertEqual(json.loads((path/'prefix_error.json').read_text())['step'],1)

    def test_simulator_hook_gate_rejection_executes_nothing(self):
        actions,taken,messages,path=self.hooks(reject=True)
        self.assertEqual(actions,[]);self.assertEqual(taken,[]);self.assertEqual(messages,[])
        self.assertEqual(json.loads((path/'expert_gate.json').read_text())['status'],'rejected')


if __name__=='__main__':unittest.main()
