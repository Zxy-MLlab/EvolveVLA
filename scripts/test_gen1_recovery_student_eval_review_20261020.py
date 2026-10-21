"""Independent CPU review checks; do not mutate the frozen student runner."""
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import gen1_recovery_student_eval_20261020 as student


class ContinuationTests(unittest.TestCase):
    def run_loop(self,drift=False,prefix_stop=False,mode='original'):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        directory=Path(temporary.name)
        initial={'pose':[0.]};middle={'pose':[1.]};last={'pose':[2.]}
        case=dict(id='test',takeover_boundary_step=2,prefix_targets=[[0.]*14,[1.]*14],
            prefix_physics_steps=[10,20],prefix_physical=[initial,middle,last])
        observed=[initial,middle,{'pose':[2.001]} if drift else last]
        (directory/'physical.jsonl').write_text(''.join(json.dumps(dict(step=i,physical=p))+'\n' for i,p in enumerate(observed)))
        (directory/'timing.jsonl').write_text(''.join(json.dumps(dict(action_index=i,physics_steps=t,target=a))+'\n' for i,(t,a) in enumerate(zip(case['prefix_physics_steps'],case['prefix_targets']))))
        student.ACTIVE.update(case=case,directory=directory)
        sent=[];inferences=[];record=[]
        images={camera:np.zeros((2,2,3),np.uint8) for camera in ('cam_high','cam_left_wrist','cam_right_wrist')}
        def infer(**kwargs):
            self.assertEqual(len(sent),2)
            inferences.append(kwargs)
            return np.zeros((50,14))
        def receive():
            return dict(type='step',observation=len(sent),terminated=(len(sent)==4 or prefix_stop),
                truncated=False,info={'is_success':len(sent)==4})
        with patch.dict(os.environ,GEN1_DURATION_MODE=mode):
            result=student.continuation_loop(types.SimpleNamespace(infer_chunk=infer),
                dict(observation=0,diagnostic_binding={'prompt':'original instruction'},step_limit=400),50,
                lambda action:sent.append(action),receive,
                lambda observation:(images,np.full(14,observation)),lambda row,images:record.append(row),action_limit=5)
        return result,sent,inferences,record,directory

    def test_continuation_uses_takeover_observation_prompt_and_correct_absolute_counts(self):
        result,sent,inferences,records,path=self.run_loop()
        self.assertTrue(result['success']);self.assertTrue(result['exact_prefix'])
        self.assertEqual(result['steps'],4);self.assertEqual(result['suffix_steps'],2)
        np.testing.assert_array_equal(inferences[0]['state'],np.full(14,2))
        self.assertEqual(inferences[0]['prompt'],'original instruction')
        self.assertEqual(records[0]['step'],2);self.assertEqual(records[0]['suffix_step'],0)
        self.assertEqual(records[0]['execute_prefix'],3)
        self.assertEqual(sent[:2],[[0.]*14,[1.]*14])
        self.assertTrue(json.loads((path/'prefix_admission.json').read_text())['no_policy_prediction_before_takeover'])

    def test_physical_drift_rejects_before_student_inference(self):
        with self.assertRaisesRegex(ValueError,'did not reproduce exactly'):self.run_loop(drift=True)

    def test_early_prefix_stop_rejected(self):
        with self.assertRaisesRegex(ValueError,'before the frozen takeover'):self.run_loop(prefix_stop=True)

    def test_nonoriginal_controller_rejected(self):
        with self.assertRaisesRegex(ValueError,'original deployment'):self.run_loop(mode='fixed15')


if __name__=='__main__':unittest.main()
