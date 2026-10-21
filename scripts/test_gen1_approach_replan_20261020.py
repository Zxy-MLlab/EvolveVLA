"""CPU-only gating and control-loop causal invariants."""
import os
import sys
import tempfile
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import numpy as np
import gen1_approach_replan_20261020 as runner


def chunk(opening=40,arm=13):
    value=np.zeros((50,14));value[:,6]=1.
    if opening is not None:value[opening:,arm]=1.
    return value


class ApproachTests(unittest.TestCase):
    def test_closed_arm_replans20actions_before_first_opening(self):
        state=np.zeros(14);state[6]=1.
        gate=runner.ApproachGate('place_mouse_pad')
        count,row=gate.select(state,chunk(40))
        self.assertEqual(count,20);self.assertEqual(row['selected_arms'],['right'])
        self.assertEqual(row['candidates'][0]['first_opening_index'],40)
        self.assertEqual(gate.select(state,chunk(40))[0],50)

    def test_open_state_and_absent_opening_never_trigger(self):
        state=np.ones(14)
        self.assertEqual(runner.ApproachGate('place_mouse_pad').select(state,chunk())[0],50)
        self.assertEqual(runner.ApproachGate('hanging_mug').select(np.zeros(14),chunk(None))[0],50)

    def test_threshold_and_index0_have_at_least_one_action(self):
        for i in (0,1,19,20,21):
            gate=runner.ApproachGate('hanging_mug')
            state=np.full(14,.2)
            value=np.full((50,14),.2);value[i:,13]=.2000001
            self.assertEqual(gate.select(state,value)[0],max(1,i-20))

    def test_mouse_each_arm_once_mug_only_right(self):
        state=np.zeros(14);value=np.zeros((50,14))
        value[45:,6]=1.;value[30:,13]=1.
        gate=runner.ApproachGate('place_mouse_pad')
        self.assertEqual(gate.select(state,value)[0],10)
        self.assertEqual(gate.used,{'right'})
        self.assertEqual(gate.select(state,value)[0],25)
        self.assertEqual(gate.used,{'right','left'})
        self.assertEqual(gate.select(state,value)[0],50)
        value[:,13]=0.
        self.assertEqual(runner.ApproachGate('hanging_mug').select(state,value)[0],50)

    def loop(self,mode,original=False,opening=40,limit=70):
        sent=[];rows=[];calls=[]
        state=np.zeros(14);state[6]=1.
        def infer(**kwargs):
            calls.append(dict(state=kwargs['state'].copy(),prompt=kwargs['prompt']))
            value=chunk(opening);value[:,:6]=len(calls)
            return value
        def receive():
            return dict(type='step',observation=np.asarray(sent[-1]),terminated=False,truncated=False)
        fn=runner.horizon.execution_loop if original else runner.execution_loop
        runner.ACTIVE['case']={'task':'place_mouse_pad'}
        with patch.object(runner,'EXECUTION_MODE',mode),patch.dict(os.environ,GEN1_DURATION_MODE='original'):
            result=fn(types.SimpleNamespace(infer_chunk=infer),
                dict(observation=state,diagnostic_binding={'prompt':'original text'}),50,
                lambda action:sent.append(action),receive,lambda observation:({},observation),
                lambda row,images:rows.append(row),action_limit=limit,clock=lambda:0.)
        return result,sent,rows,calls

    def test_original50_matches_existing_loop_all_actions_counts_states(self):
        a,aa,ar,ac=self.loop('original50');b,ba,br,bc=self.loop('original50',original=True)
        self.assertEqual(aa,ba)
        for key in ('steps','inference_count','termination','success'):self.assertEqual(a[key],b[key])
        self.assertEqual([r['execute_prefix'] for r in ar],[r['execute_prefix'] for r in br])
        for av,bv in zip(ac,bc):np.testing.assert_array_equal(av['state'],bv['state'])

    def test_first50prediction_and_executed_prefix_match_before_new_observation(self):
        a,aa,ar,ac=self.loop('original50');b,ba,br,bc=self.loop('approach_replan')
        self.assertEqual(ar[0]['actions'],br[0]['actions'])
        self.assertEqual(aa[:20],ba[:20])
        self.assertEqual([r['step'] for r in br],[0,20])
        self.assertEqual([r['execute_prefix'] for r in br],[20,50])
        self.assertEqual([r['inference'] for r in br],[1,2])
        np.testing.assert_array_equal(bc[1]['state'],ba[19])
        self.assertEqual(len(b['approach_replan_events']),1)
        self.assertEqual(b['steps'],70)

    def test_no_trigger_candidate_matches_original_entire_episode(self):
        a,aa,ar,ac=self.loop('original50',opening=None)
        b,ba,br,bc=self.loop('approach_replan',opening=None)
        self.assertEqual(aa,ba);self.assertEqual(b['approach_replan_events'],[])
        self.assertEqual([r['step'] for r in ar],[r['step'] for r in br])

    def test_both_conditions_share_one_cache_in_fixed_order(self):
        calls=[]
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'paired'
            def condition(mode,args):
                calls.append((mode,list(args),id(runner.SHARED_CACHE)))
                self.assertTrue(runner.SHARE_PAIRED_CACHE)
                if not runner.SHARED_CACHE:runner.SHARED_CACHE.append(object())
                Path(args[args.index('--output')+1]).mkdir(parents=True)
            with patch.object(sys,'argv',['runner','run','--execution-mode','both','--output',str(output)]), \
                 patch.object(runner,'verify_protocol',return_value={}), \
                 patch.object(runner,'run_condition',side_effect=condition), \
                 patch.object(runner.recovery,'digest',return_value='frozen'):
                runner.main()
            self.assertEqual([c[0] for c in calls],['original50','approach_replan'])
            self.assertEqual(calls[0][2],calls[1][2])
            self.assertEqual(len(runner.SHARED_CACHE),1)
            self.assertTrue((output/'paired_execution.json').exists())

    def test_worker_uses_shared_cache_across_distinct_condition_local_caches(self):
        panel=runner.horizon.ROOT/'outputs/gen1_control_repair_20261020/panel_development4.json'
        case=runner.recovery.read(panel)['cases'][0]
        protocol=dict(checkpoint='/tmp/fixed_checkpoint',panel=str(panel))
        passed=[]
        def worker(case,prefix,args,cache,directory):
            passed.append(cache)
            if not cache:cache.append(object())
            return {}
        with patch.object(runner,'verify_protocol',return_value=protocol), \
             patch.object(runner,'SHARE_PAIRED_CACHE',True),patch.object(runner,'SHARED_CACHE',[]), \
             patch.object(runner,'ORIGINAL_WORKER',side_effect=worker), \
             patch.object(runner.horizon,'write_json'),patch.object(runner.recovery,'digest',return_value='frozen'), \
             patch.dict(os.environ,GEN1_DURATION_MODE='original'):
            for _ in range(2):runner.worker(case,50,types.SimpleNamespace(checkpoint='/tmp/fixed_checkpoint'),[],Path('/tmp/unused'))
            self.assertIs(passed[0],passed[1]);self.assertEqual(len(passed[0]),1)

    def test_short_outer_budget_not_exceeded(self):
        result,actions,rows,calls=self.loop('approach_replan',limit=7)
        self.assertEqual(len(actions),7);self.assertEqual(result['steps'],7)
        self.assertEqual(rows[0]['execute_prefix'],7)


if __name__=='__main__':unittest.main()
