import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import gen1_approach_replan_audit_20261020 as audit


class PairedAuditTests(unittest.TestCase):
    def fixture(self,mutate=None):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        for name,mode in [('baseline','original50'),('candidate','approach_replan')]:
            p=root/name/'test'/'original';p.mkdir(parents=True)
            result=dict(status='completed',simulator_returncode=0,success=True,steps=6,
                expert_gate=dict(status='passed',bypassed=False),source_parity=dict(state_exact=True,all_source_jpeg_bytes_equal=True),
                hashes={'state':'x'},binding={'prompt':'text'},checkpoint_identity_sha256='model',
                simulator_limits={'official_step_limit':400},initial_physical_sha256='same',
                execution_mode=mode,timing_summary=dict(duration_active_actions=0),
                approach_replan_events=[] if name=='baseline' else [dict(step=0,selected_arms=['right'])])
            action=dict(step=0,state=[0.]*14,actions=[[0.]*14 for _ in range(50)],execute_prefix=50 if name=='baseline' else 20)
            rows=dict(result=result,actions=[action],physical=[dict(step=i,physical={'q':[float(i)]}) for i in range(7)],
                      timing=[dict(target=[float(i)]*14,physics_steps=15) for i in range(6)])
            if name=='candidate' and mutate:mutate(rows)
            (p/'result.json').write_text(json.dumps(rows['result']))
            (p/'execution_protocol.json').write_text(json.dumps({'protocol_sha256':'frozen'}))
            for key in ['actions','physical','timing']:(p/(key+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows[key]))
        with patch.object(audit.r,'digest',return_value='frozen'):
            return audit.audit_pair({'id':'test','task':'place_mouse_pad'},root/'baseline',root/'candidate')

    def test_success_before_planned_replan_is_valid_and_marked_no_intervention(self):
        row=self.fixture()
        self.assertTrue(row['valid']);self.assertEqual(row['shared_action_prefix_checked'],6)
        self.assertFalse(row['first_intervention']['extra_observation_inference_executed'])
        self.assertEqual(row['first_intervention']['scheduled_replan_boundary'],20)

    def test_full_first_prediction_mismatch_rejects_even_unused_action(self):
        def change(rows):rows['actions'][0]['actions'][49][0]=.001
        row=self.fixture(change)
        self.assertFalse(row['valid']);self.assertIn('first50_actions:mismatch',row['errors'])

    def test_actual_tick_drift_rejected(self):
        def change(rows):rows['timing'][2]['physics_steps']=16
        row=self.fixture(change)
        self.assertFalse(row['valid']);self.assertIn('preintervention_physics_steps:mismatch',row['errors'])


if __name__=='__main__':unittest.main()
