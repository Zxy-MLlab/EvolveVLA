"""CPU regression checks for invalid/missing evidence in timing summaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import gen1_timing_validation_summary_20261020 as summary


def row(success=False):
    return dict(id='x', status='completed', success=success, simulator_returncode=0,
                checkpoint='model', checkpoint_identity_sha256='sha',
                hashes={'images':'image_sha','state':'state_sha','prompt':'prompt_sha'},
                initial_parity=dict(state_max_abs=0., physical={'joint':0.},
                    images={name:dict(jpeg_bytes_equal=True) for name in ('cam_high','cam_left_wrist','cam_right_wrist')}),
                expert_gate=dict(status='passed',bypassed=False))


class OutcomeTests(unittest.TestCase):
    def test_source_exhaustion_is_valid_failure(self):
        value=row(False);value['status']='source_exhausted'
        self.assertEqual(summary.outcome(value,'expert'),'failure')
        self.assertTrue(summary.row_report(value,'expert')['protocol_eligible'])
        self.assertEqual(summary.outcome(value,'policy'),'error_or_excluded')

    def test_false_error_defaults_are_excluded(self):
        for status in ('worker_error','setup_error','timeout','initial_source_mismatch','timing_audit_error'):
            value=row(False);value['status']=status
            result=summary.row_report(value,'expert')
            self.assertFalse(result['protocol_eligible'])
            self.assertIsNone(result['success'])
        value=row(False);value['status']='expert_rejected'
        self.assertEqual(summary.row_report(value,'expert')['outcome'],'expert_rejected')

    def test_malformed_terminal_records_are_excluded(self):
        for field,value in [('success',None),('success',0),('simulator_returncode',2),('simulator_returncode',None)]:
            bad=row();bad[field]=value
            self.assertFalse(summary.row_report(bad,'expert')['protocol_eligible'])
        bad=row(True);bad['status']='source_exhausted'
        self.assertEqual(summary.outcome(bad,'expert'),'invalid_terminal_record')

    def test_approximate_match_does_not_count_as_exact(self):
        bad=row();bad['initial_parity']['images']['cam_high']['jpeg_bytes_equal']=False
        bad['initial_parity']['observed_source_match']=True
        self.assertFalse(summary.row_report(bad,'expert')['protocol_eligible'])

    def test_gate_bypass_never_enters_pair(self):
        bad=row();bad['expert_gate']['bypassed']=True
        self.assertEqual(summary.basic_pair(row(),bad,'expert')['status'],'error_or_excluded')
        bad=row();bad.pop('expert_gate')
        self.assertEqual(summary.basic_pair(row(),bad,'policy')['status'],'error_or_excluded')

    def test_missing_and_physical_mismatch_cannot_pair(self):
        self.assertEqual(summary.basic_pair(row(),None,'expert')['status'],'pending')
        bad=row();bad['initial_parity']['physical']={}
        self.assertEqual(summary.basic_pair(row(),bad,'expert')['status'],'initial_physical_mismatch_or_missing')
        bad=row();bad['initial_parity']['physical']['joint']=1.
        self.assertEqual(summary.basic_pair(row(),bad,'expert')['status'],'initial_physical_mismatch_or_missing')

    def test_policy_input_or_checkpoint_mismatch_cannot_pair(self):
        bad=row();bad['hashes']['state']='changed'
        self.assertEqual(summary.basic_pair(row(),bad,'policy')['status'],'input_mismatch_or_missing')
        bad=row();bad['checkpoint_identity_sha256']='changed'
        self.assertEqual(summary.basic_pair(row(),bad,'policy',True)['status'],'checkpoint_mismatch_or_missing')

    def test_four_way_missing_cell_stays_pending(self):
        case=dict(id='x',task='place_mouse_pad')
        indexed={label+'_policy_ab':{('x',mode):row() for mode in ('original','fallback15')} for label in ('baseline','candidate')}
        indexed['candidate_policy_ab'].pop(('x','fallback15'))
        report=summary.common_four_cases([case],indexed)['place_mouse_pad']
        self.assertEqual(report['common_eligible'],0)
        self.assertEqual(report['statuses'],{'pending':1})


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.a,self.b=row(False),row(True)
        self.at=[dict(source_action_index=i,source_target=[i],physical={'q':i},timing={'physics_steps':50,'total_physics_steps':50*(i+1)}) for i in range(3)]
        self.bt=copy.deepcopy(self.at)
        self.bt[2]['timing']={'physics_steps':15,'total_physics_steps':115}
        for mode,value in [('a',self.a),('b',self.b)]:
            path=root/mode;path.mkdir()
            value.update(_result_path=str(path/'result.json'),source_hash='source',release=dict(start=2,end=2),release_indices=[2])
        (root/'b/retiming.jsonl').write_text(json.dumps(dict(source_action_index=2))+'\n')
        self.manifest=dict(method_patch={'only_four_qpos_fallback_literals_changed':True})
        self.save()

    def save(self):
        for value,trace in [(self.a,self.at),(self.b,self.bt)]:
            Path(value['_result_path']).with_name('trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))

    def test_correct_prefix_controls_duration(self):
        self.assertTrue(summary.expert_comparison(self.a,self.b,self.manifest)['controlled_duration_intervention'])

    def test_missing_or_changed_pretreatment_trace_rejects_causality(self):
        for field in ('physical','source_target','timing'):
            bad=copy.deepcopy(self.bt);bad[1][field]={'changed':1}
            self.bt=bad;self.save()
            self.assertFalse(summary.expert_comparison(self.a,self.b,self.manifest)['controlled_duration_intervention'])
            self.bt=copy.deepcopy(self.at);self.bt[2]['timing']={'physics_steps':15,'total_physics_steps':115}
        self.bt.pop(1);self.save()
        self.assertFalse(summary.expert_comparison(self.a,self.b,self.manifest)['controlled_duration_intervention'])

    def test_no_actual_duration_change_is_not_mechanistic_confirmation(self):
        self.bt=copy.deepcopy(self.at);self.save()
        report=summary.expert_comparison(self.a,self.b,self.manifest)
        self.assertEqual(report['status'],'paired')
        self.assertFalse(report['controlled_duration_intervention'])


if __name__=='__main__':
    unittest.main()
