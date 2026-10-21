#!/usr/bin/env python3
"""Read-only paired audit: identical input,first chunk and pre-intervention physics."""
import argparse
from pathlib import Path
import sys
SCRIPT=Path(__file__).resolve()
sys.path.insert(0,str(SCRIPT.parent))
import gen1_approach_replan_20261020 as runner
r=runner.recovery


def audit_pair(case,baseline,candidate):
    paths={name:root/case['id']/'original' for name,root in [('baseline',baseline),('candidate',candidate)]}
    result={name:r.read(path/'result.json') for name,path in paths.items()}
    row=dict(id=case['id'],task=case['task'],valid=False,errors=[])
    for name,value in result.items():
        gate=value.get('expert_gate') or {};parity=value.get('source_parity') or {}
        if not (value.get('status')=='completed' and value.get('simulator_returncode')==0 and
                type(value.get('success')) is bool and gate.get('status')=='passed' and
                gate.get('bypassed') is False and parity.get('state_exact') is True and
                parity.get('all_source_jpeg_bytes_equal') is True):
            row['errors'].append(name+':not_valid_completed_exact_source')
    if row['errors']:return row
    a,b=result['baseline'],result['candidate']
    for field in ('hashes','binding','checkpoint_identity_sha256','simulator_limits','initial_physical_sha256'):
        if not a.get(field) or a[field]!=b.get(field):row['errors'].append(field+':mismatch')
    if a.get('execution_mode')!='original50' or b.get('execution_mode')!='approach_replan':row['errors'].append('execution_mode:invalid')
    for value in result.values():
        if value['timing_summary']['duration_active_actions']!=0:row['errors'].append('simulator_controller_modified')
    protocols={name:r.read(path/'execution_protocol.json') for name,path in paths.items()}
    if protocols['baseline']['protocol_sha256']!=protocols['candidate']['protocol_sha256']:row['errors'].append('frozen_protocol:mismatch')
    if protocols['baseline']['protocol_sha256']!=r.digest(runner.PROTOCOL):row['errors'].append('frozen_protocol:changed')
    actions={name:r.lines(path/'actions.jsonl') for name,path in paths.items()}
    for field in ('state','actions'):
        if actions['baseline'][0][field]!=actions['candidate'][0][field]:row['errors'].append('first50_'+field+':mismatch')
    events=b.get('approach_replan_events',[])
    used=[arm for event in events for arm in event['selected_arms']]
    if len(used)!=len(set(used)):row['errors'].append('arm_triggered_more_than_once')
    if events:
        event=events[0]
        capture=next(x for x in actions['candidate'] if x['step']==event['step'])
        scheduled_boundary=event['step']+capture['execute_prefix']
        boundary=min(scheduled_boundary,b['steps'])
        row['first_intervention']=dict(inference_step=event['step'],planned_execute_prefix=capture['execute_prefix'],
            boundary_step=boundary,scheduled_replan_boundary=scheduled_boundary,
            extra_observation_inference_executed=any(x['step']==scheduled_boundary for x in actions['candidate'][1:]),
            selected_arms=event['selected_arms'])
    else:boundary=min(a['steps'],b['steps'])
    physical={name:r.lines(path/'physical.jsonl') for name,path in paths.items()}
    timing={name:r.lines(path/'timing.jsonl') for name,path in paths.items()}
    if any(len(rows)<boundary+1 for rows in physical.values()) or any(len(rows)<boundary for rows in timing.values()):
        row['errors'].append('prefix_trace:incomplete')
    else:
        for field in ('target','physics_steps'):
            if [x[field] for x in timing['baseline'][:boundary]]!=[x[field] for x in timing['candidate'][:boundary]]:
                row['errors'].append('preintervention_'+field+':mismatch')
        if [x['physical'] for x in physical['baseline'][:boundary+1]]!=[x['physical'] for x in physical['candidate'][:boundary+1]]:
            row['errors'].append('preintervention_physical:mismatch')
    row.update(valid=not row['errors'],baseline_success=a['success'],candidate_success=b['success'],
        shared_action_prefix_checked=boundary,triggered_arms=used,events=events,
        baseline_steps=a['steps'],candidate_steps=b['steps'])
    return row


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    protocol=runner.verify_protocol();cases=r.read(protocol['panel'])['cases']
    rows=[audit_pair(case,args.baseline,args.candidate) for case in cases]
    valid=[row for row in rows if row['valid']]
    rescued=sum(not row['baseline_success'] and row['candidate_success'] for row in valid)
    lost=sum(row['baseline_success'] and not row['candidate_success'] for row in valid)
    report=dict(protocol=str(runner.PROTOCOL),protocol_sha256=r.digest(runner.PROTOCOL),records=rows,
        requested=len(cases),valid_pairs=len(valid),rescued=rescued,lost=lost,
        baseline_successes=sum(row['baseline_success'] for row in valid),candidate_successes=sum(row['candidate_success'] for row in valid),
        promotion_eligible=len(valid)==len(cases) and rescued>=1 and lost==0,
        limitation='Selected training development scenes; promotion permits independent confirmation only, not a claim of repair')
    r.horizon.write_json(args.output,report)
    print({k:report[k] for k in ('valid_pairs','rescued','lost','promotion_eligible')})


if __name__=='__main__':main()
