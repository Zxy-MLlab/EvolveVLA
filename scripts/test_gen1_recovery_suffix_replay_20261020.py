"""CPU integration tests for saved paths, strict capture intervals and terminal hold."""
import json
from pathlib import Path
import pickle
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import gen1_recovery_suffix_replay_20261020 as replay
import test_gen1_recovery_takeover_v2_20261020 as fixtures


def task():
    value=fixtures.fake_task()
    value.check_success=lambda:value.scene.count>=3
    value.get_obs=lambda:dict(joint_action={'vector':np.full(14,value.scene.count,dtype=float)},
        observation={name:{'rgb':np.full((2,2,3),value.scene.count%255,dtype=np.uint8)} for name in replay.CAMERAS})
    value.robot.left_gripper=[];value.robot.right_gripper=[]
    return value


def native(t,c):
    if t.need_plan:
        path=dict(status='Success',position=np.zeros((45,6)),velocity=np.zeros((45,6)))
        t.right_joint_path.append(path)
    else:
        path=t.right_joint_path[t.right_cnt];t.right_cnt+=1
    t.take_dense_action(dict(left_arm=None,left_gripper=None,right_arm=path,right_gripper=None))


class ReplayTests(unittest.TestCase):
    def dense(self,hold=0,drift=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);source=root/'source';source.mkdir()
        case=dict(task='place_mouse_pad',active_arm='right',takeover_boundary_step=124,
            recovery_source_directory=str(source),terminal_hold_ticks=hold)
        with patch.object(replay.oracle,'physical',fixtures.physical), \
             patch.object(replay.collector,'predicates',lambda t,name:{'original':bool(t.check_success())}), \
             patch.object(replay.collector,'native_suffix',native):
            result=replay.ORIGINAL_COLLECT(task(),case,source/'expert_suffix',100,suffix=native)
            if drift:
                p=source/'expert_suffix/captures.jsonl'
                rows=replay.lines(p);rows[1]['physical']['tick']+=1
                p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            output=root/'dense';output.mkdir()
            final_task=task()
            result=replay.dense_replay(final_task,case,output/'expert_suffix',100)
        return result,output,case,final_task

    def test_dense_reuses_saved_paths_matches_native_captures_and_global15ticks(self):
        result,path,case,final_task=self.dense()
        self.assertTrue(result['success']);self.assertTrue(result['dense_source_match'])
        self.assertEqual(result['path_consumption'],{'left':0,'right':1})
        rows=replay.uniform_rows(path/'uniform_captures.jsonl')
        self.assertEqual([r['physics_tick'] for r in rows],[0,15,30,45])
        self.assertEqual(final_task.take_action_cnt,124)
        self.assertFalse(result['dataset_ready'])

    def test_terminal_hold_is_real_ticks_and_continuous_capture_not_repeat_padding(self):
        result,path,case,final_task=self.dense(hold=31)
        self.assertTrue(result['success']);self.assertTrue(result['stable_success'])
        self.assertEqual(result['source_end_physics_tick'],45)
        self.assertEqual(result['physics_ticks'],76);self.assertEqual(final_task.scene.count,76)
        self.assertEqual(result['terminal_hold']['physics_ticks'],31)
        rows=replay.lines(path/'uniform_captures.jsonl')
        self.assertEqual([r['physics_tick'] for r in rows],[0,15,30,45,60,75,76])
        self.assertFalse(rows[-1]['full_interval_boundary'])
        self.assertEqual(len(replay.uniform_rows(path/'uniform_captures.jsonl')),6)
        self.assertEqual(len(replay.lines(path/'terminal_hold_predicates.jsonl')),31)
        a,_=replay.load_raw(rows[-3]);b,_=replay.load_raw(rows[-2])
        self.assertFalse(np.array_equal(a['observation']['head_camera']['rgb'],b['observation']['head_camera']['rgb']))

    def test_source_drift_blocks_hold_and_data_acceptance(self):
        result,path,case,final_task=self.dense(hold=31,drift=True)
        self.assertFalse(result['success']);self.assertFalse(result['dense_source_match'])
        self.assertEqual(result['terminal_hold']['status'],'blocked_source_match_or_success')
        self.assertEqual(final_task.scene.count,45)

    def test_uniform_capture_tamper_rejected(self):
        result,path,case,_=self.dense()
        p=path/'uniform_captures.jsonl';rows=replay.lines(p);rows[2]['physics_tick']=31
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'consecutive'):replay.uniform_rows(p)

    def test_raw_tamper_rejected(self):
        result,path,case,_=self.dense()
        row=replay.lines(path/'uniform_captures.jsonl')[1]
        Path(row['path']).write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):replay.load_raw(row)

    def test_hold_failure_is_not_latched_success(self):
        t=task();t.scene.count=4;t.check_success=lambda:t.scene.count<7
        seen=[]
        with patch.object(replay.oracle,'physical',fixtures.physical):
            result=replay.terminal_hold(t,5,4,lambda tick,success:seen.append((tick,success)))
        self.assertEqual(t.scene.count,9);self.assertEqual(len(seen),5)
        self.assertFalse(result['success']);self.assertFalse(result['all_ticks_success'])

    def test_sparse_uses_next_capture_targets_preserves_prefix_counter_and_native_stop(self):
        result,path,case,_=self.dense(hold=31)
        class SparseTask:
            def take_action(self,action):
                self.received.append(action.tolist());self.take_action_cnt+=1
                trajectory=dict(position=np.zeros((15,6)),velocity=np.zeros((15,6)))
                self.take_dense_action(dict(left_arm=None,left_gripper=None,right_arm=trajectory,right_gripper=None))
                self.eval_success=bool(self.check_success())
        t=SparseTask();t.__dict__.update(task().__dict__)
        t.take_dense_action=types.MethodType(fixtures.native_dense(),t)
        t.check_success=lambda:t.scene.count>=30
        t.received=[]
        case.update(replay_mode='original',uniform_captures_path=str(path/'uniform_captures.jsonl'),terminal_hold_ticks=5)
        def controller(original,mode,ticks):
            return original,{'mode':mode},types.SimpleNamespace(last_record={'mode':mode})
        with patch.object(replay.oracle,'physical',fixtures.physical), \
             patch.object(replay.collector,'predicates',lambda t,name:{'original':bool(t.check_success())}), \
             patch.object(replay.duration,'create',controller):
            output=path.parent/'sparse';actual=replay.sparse_replay(t,case,output,100)
        self.assertTrue(actual['official_rollout_success']);self.assertTrue(actual['stable_success'])
        self.assertEqual(actual['actions_executed'],2)
        self.assertEqual(t.take_action_cnt,126)
        self.assertEqual(t.received,[np.full(14,15).tolist(),np.full(14,30).tolist()])
        self.assertEqual(actual['physics_ticks'],35)
        self.assertFalse(actual['all_labels_replayed'])


if __name__=='__main__':unittest.main()
