#!/usr/bin/env python3
"""Bounded CPU correction preserving prior numeric audit and inspecting flagged commands."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
import json, hashlib, re, sys, uuid
from pathlib import Path
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import numpy as np
import h5py
R=Path('/root/autodl-tmp/EvolveVLA')
old=R/'outputs/gen1_pilot_target_audit_20261010_152526_6fcaa952.json'
def sha(p):
 with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
old_hash=sha(old)
r=json.loads(old.read_text()); original_numeric_hash=hashlib.sha256(json.dumps(r['groups'],sort_keys=True).encode()).hexdigest()
run=R/'outputs/gen1_pilot_2000_20261010/run_config.json'
init=Path(json.loads(run.read_text())['config']['weight_loader']['params_path']).parent
bases={'baseline':R/'results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10','pilot':R/'results/gen1_pilot_2000_clean_two_task_20261010'}
evidence={}; hashes={str(old):old_hash,str(run):sha(run),str(Path(__file__)):sha(__file__)}
for task,seeds in {'place_mouse_pad':[100005,100007],'hanging_mug':[100005,100008,100010]}.items():
 evidence[task]={}
 for label,base in bases.items():
  result=base/task/'result.json'; log=base/task/'run.log'; data=json.loads(result.read_text())
  hashes[str(result)]=sha(result); hashes[str(log)]=sha(log)
  if label=='baseline': assert Path(data['checkpoint']).resolve()==init.resolve()
  else: assert Path(data['checkpoint']).resolve()==(R/'checkpoints/gen1_pilot_2000_20261010/1999').resolve()
  selected={s:{'result':next((x for x in data['reports'] if x['seed']==s),None),'evidence':[]} for s in seeds}
  accepted=[]; seed=None
  with log.open(errors='replace') as f:
   for line_no,line in enumerate(f,1):
    m=re.search(r'^EXPERT_SEED=(\d+) EXPERT_OK=(True|False)',line)
    if m:
     seed=int(m[1])
     if m[2]=='True': accepted.append(seed)
    if seed in selected and (m or line.startswith(('INSTRUCTION=','SEED_ARM ','RESET_SEED=','RESET_SEED '))): selected[seed]['evidence'].append({'line':line_no,'text':line.strip()})
  evidence[task][label]={'checkpoint':data['checkpoint'],'checkpoint_identity_verified':True,'result_path':str(result),'log_path':str(log),'successes':data['successes'],'completed':data['episodes_completed'],'accepted_seeds':accepted,'selected':selected}
 comparisons={}
 for seed in seeds:
  a,b=(evidence[task][k]['selected'][seed] for k in ('baseline','pilot'))
  ins=lambda x:[v['text'] for v in x['evidence'] if v['text'].startswith('INSTRUCTION=')]
  arms=lambda x:[v['text'] for v in x['evidence'] if v['text'].startswith('SEED_ARM ')]
  comparisons[seed]={'same_logged_instruction':bool(ins(a)) and ins(a)==ins(b),'same_logged_arm_line':bool(arms(a)) and arms(a)==arms(b),'baseline_success':a['result']['success'] if a['result'] else None,'pilot_success':b['result']['success'] if b['result'] else None,'exact_reset_pairing_established':False,'limitation':'Seed and arm labels do not provide complete reset RNG or scene fingerprints.'}
 evidence[task]['comparison']=comparisons
assert sum(evidence[t]['baseline']['successes'] for t in evidence)==3
assert sum(evidence[t]['pilot']['successes'] for t in evidence)==4
emb=R/'robotwin_repo_ba0fd21/assets/embodiments/aloha-agilex'
urdf=emb/'urdf/arx5_description_isaac.urdf'
names=[f'fl_joint{i}' for i in range(1,7)]+['left_gripper_normalized']+[f'fr_joint{i}' for i in range(1,7)]+['right_gripper_normalized']
joints={j.get('name'):{'type':j.get('type'),'limits':{k:float(v) for k,v in j.find('limit').attrib.items()}} for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
parts=('left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states')
mask=np.array([True]*6+[False]+[True]*6+[False]); dims=np.flatnonzero(mask)
flagged=[]
for group,g in r['groups'].items():
 for x in g['target_all']['representative_maxima']:
  if x['dimension'] in dims and abs(x['normalized'])>2: flagged.append(dict(x,group=group))
by_path={}
for x in flagged: by_path.setdefault(x['path'],[]).append(x)
source_hashes={x['path']:x['sha256'] for x in r['sources']}
norm=R/'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k/assets/aloha_agilex_sim/norm_stats.json'
ns=json.loads(norm.read_text())['norm_stats']['actions']; low=np.array(ns['q01'][:14]); high=np.array(ns['q99'][:14])
profiles=[]; continuity=[]
for path,records in by_path.items():
 assert sha(path)==source_hashes[path]
 with h5py.File(path,'r') as f:
  arrays=[]
  for group in ('state','action'):
   arrays.append(np.concatenate([np.asarray(f[group][p][()],dtype=np.float32).reshape(len(f[group][p]),-1) for p in parts],axis=1))
 state,action=arrays; n=len(state); np.testing.assert_array_equal(state[1:],action[:-1])
 trajectory=np.concatenate([state,action[-1:]],axis=0).astype(np.float64)
 jump=np.diff(trajectory,axis=0); wrapped=(jump+np.pi)%(2*np.pi)-np.pi
 per_dim=[]
 for d in dims:
  limit=joints[names[d]]['limits']; i=int(np.abs(jump[:,d]).argmax())
  per_dim.append({'dimension':int(d),'joint':names[d],'minimum':float(trajectory[:,d].min()),'maximum':float(trajectory[:,d].max()),'outside_urdf_limits_count':int(((trajectory[:,d]<limit['lower'])|(trajectory[:,d]>limit['upper'])).sum()),'maximum_adjacent_jump':float(abs(jump[i,d])),'signed_maximum_jump':float(jump[i,d]),'max_jump_from_frame':i,'max_jump_to_frame':i+1,'max_jump_implied_rate_radians_per_second':float(abs(jump[i,d])*15),'jumps_above_pi_count':int((abs(jump[:,d])>np.pi).sum()),'near_2pi_jump_count_tolerance_0_1_rad':int((abs(abs(jump[:,d])-2*np.pi)<0.1).sum()),'max_modulo_reduced_adjacent_jump':float(abs(wrapped[:,d]).max())})
 continuity.append({'path':path,'sha256_verified':True,'source_trajectory_frames':len(trajectory),'frequency_hz':15,'dimensions':per_dim})
 for x in records:
  anchor=x['anchor']; indices=np.minimum(anchor+np.arange(50),n-1); absolute=action[indices]; targets=absolute-np.where(mask,state[anchor],0)[None,:]; normalized=(targets-low)/(high-low+1e-6)*2-1
  np.testing.assert_allclose(targets[x['horizon_offset'],x['dimension']],x['value'],rtol=1e-6,atol=1e-6)
  profiles.append({'representative':x,'anchor_state_14':state[anchor].tolist(),'action_indices':indices.tolist(),'padded':(anchor+np.arange(50)>=n).tolist(),'absolute_actions_50_by_14':absolute.tolist(),'targets_50_by_14':targets.tolist(),'normalized_targets_50_by_14':normalized.tolist(),'flagged_joint_adjacent_jumps_from_anchor_then_actions':np.diff(np.concatenate([[state[anchor,x['dimension']]],absolute[:,x['dimension']].astype(np.float64)])).tolist()})
 print('inspected',len(continuity),'/',len(by_path),flush=True)
files=[urdf,emb/'config.yml',R/'robotwin_repo_ba0fd21/env_cfg/task_config/demo_clean.yml',R/'robotwin_repo_ba0fd21/envs/robot/robot.py',R/'robotwin_repo_ba0fd21/envs/_base_task.py',R/'robotwin_repo_ba0fd21/envs/utils/pkl2hdf5.py',norm,init/'params/_METADATA']
hashes.update({str(p):sha(p) for p in files})
r['evaluation_comparison']=evidence
r['correction']={'created_utc':datetime.now(timezone.utc).isoformat(),'historical_report':str(old),'historical_report_sha256':old_hash,'historical_report_unchanged':True,'superseded_section':'evaluation_comparison only','reason':'Historical comparison used protocol_v11_ba0fd21_full_clean_10_oidn_jax0 with pi05_robotwin/29999, not the Myungkyu source checkpoint.','numeric_sections_retained_verbatim':True,'numeric_groups_canonical_sha256':original_numeric_hash,'baseline_initialization_checkpoint':str(init),'baseline_total_successes':3,'pilot_total_successes':4,'episodes_each':20,'new_evidence_sha256':hashes,'numeric_created_utc':r['created_utc']}
r['trajectory_inspection']={'selection':'All target_all per-dimension representative maxima on arm joints with abs(normalized)>2; excludes grippers.','flagged_representatives':len(flagged),'unique_source_episodes':len(by_path),'dimension_names':names,'urdf_path':str(urdf),'urdf_resolved_path':str(urdf.resolve()),'joint_definitions':joints,'recorded_semantics':'Source joints are saved robot drive targets: robot.py get_*_arm_jointState uses get_drive_target; _base_task.py saves joint_action; pkl2hdf5.py shifts successive commands into state/action. These arrays do not establish measured motion. Grippers are normalized controls, not URDF prismatic meters.','continuity':continuity,'full_horizon_profiles':profiles,'interpretation':'Raw adjacent jumps and 50-offset profiles distinguish abrupt coordinate changes from accumulated commanded travel. Modulo reduction is a diagnostic only, never applied to targets. URDF revolute bounds are [-10,10], allowing angles beyond +/-pi. Neither large cumulative displacements nor near-2pi totals alone establish angular wrapping, actual speed, or a rollout failure cause.'}
r['limitations'].append('Historical evaluation comparison is superseded; old baseline hashes remain only as historical audit provenance. New correct baseline and inspection hashes are in correction.new_evidence_sha256. Flagged source arrays contain commanded drive targets, not measured joints; no physical cause is proven.')
assert hashlib.sha256(json.dumps(r['groups'],sort_keys=True).encode()).hexdigest()==original_numeric_hash
assert sha(old)==old_hash
out=R/'outputs'/('gen1_pilot_target_audit_corrected_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.json')
with out.open('x') as f: json.dump(r,f,indent=2,allow_nan=False)
print(json.dumps({'report':str(out),'report_sha256':sha(out),'flagged_representatives':len(flagged),'episodes':len(by_path),'comparisons':{t:evidence[t]['comparison'] for t in evidence},'largest_adjacent_jump':max(z['maximum_adjacent_jump'] for c in continuity for z in c['dimensions']),'jumps_above_pi':sum(z['jumps_above_pi_count'] for c in continuity for z in c['dimensions'])}),flush=True)
