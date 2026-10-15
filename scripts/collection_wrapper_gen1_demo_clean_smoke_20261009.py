#!/usr/bin/env python3
"""Bounded transactional expert smoke. Pinned source remains read-only."""
import argparse, hashlib, importlib, json, os, pickle, shutil, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/EvolveVLA')
REPO=ROOT/'robotwin_repo_ba0fd21'
REV='ba0fd21ad21c336bedc7bd8eacd3e514860910e2'
FIELDS={'left_arm':'left_arm_joint_states','left_gripper':'left_ee_joint_states','right_arm':'right_arm_joint_states','right_gripper':'right_ee_joint_states'}
CAMS={'head_camera':'cam_head','left_camera':'cam_left_wrist','right_camera':'cam_right_wrist'}
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,obj):
 p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); os.replace(tmp,p)
def reserve(p):
 if shutil.disk_usage(p).free<60*1024**3:raise RuntimeError('less than 60 GiB free')
def validate(h5,raw,instructions):
 import cv2,h5py,numpy as np
 files=sorted(raw.glob('*.pkl'),key=lambda p:int(p.stem))
 assert [int(p.stem) for p in files]==list(range(len(files))) and len(files)>1
 hashes={p.name:digest(p) for p in files}
 rows=[]
 for p in files:
  with p.open('rb') as f:rows.append(pickle.load(f))
 n=len(rows)-1
 with h5py.File(h5) as f:
  assert int(f['additional_info/frequency'][()])==15
  assert json.loads(f['instructions'][()].decode())==instructions and all(instructions)
  for group,offset in [('state',0),('action',1)]:
   width=0
   for source,target in FIELDS.items():
    expected=np.asarray([r['joint_action'][source] for r in rows],dtype=np.float32).reshape(len(rows),-1)
    actual=f[group][target][...]; assert actual.shape==(n,expected.shape[1]); assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual,expected[offset:offset+n]); width+=actual.shape[1]
   assert width==14
  for src,dst in CAMS.items():
   encoded=f['vision'][dst]['colors']; assert len(encoded)==n
   for i,row in enumerate(rows[:-1]):
    image=row['observation'][src]['rgb']; assert image.shape[-1]==3
    np.testing.assert_array_equal(f['vision'][dst]['shape'][...],image.shape)
    ok,expected=cv2.imencode('.jpg',image); assert ok
    actual=bytes(encoded[i]); assert actual==expected.tobytes()
    decoded=cv2.imdecode(np.frombuffer(actual,np.uint8),cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape==image.shape
 return {'frames':n,'raw_frames':len(rows),'state_action_dim':14,'cameras':list(CAMS.values()),'raw_sha256':hashes,'alignment':'state=q[:-1], action=q[1:], all JPEG bytes match official encoding of raw frame[:-1]','hdf5_bytes':h5.stat().st_size,'hdf5_sha256':digest(h5)}
def worker(a):
 os.chdir(REPO);sys.path.insert(0,str(REPO))
 import scripts.collect_data as official
 from envs.utils import pkl2hdf5
 # Use the exact official main config resolver, intercept only collection loop.
 def run(env,cfg):
  stage=Path(a.stage);cfg.update(save_path=str(stage),need_plan=True,save_data=False,render_freq=0)
  dump(stage/'resolved_config.json',cfg)
  env.setup_demo(now_ep_num=a.index,seed=a.seed,**cfg)
  env.play_once(); plan=bool(env.plan_success and env.check_success())
  dump(stage/'predicates.json',{'plan_success':plan})
  if not plan:raise RuntimeError('expert planning predicate failed')
  env.save_traj_data(a.index);env.close_env()
  cfg.update(need_plan=False,save_data=True)
  env.setup_demo(now_ep_num=a.index,seed=a.seed,**cfg)
  traj=env.load_tran_data(a.index);cfg.update(traj);env.set_path_lst(cfg)
  info=env.play_once(); replay=bool(env.plan_success and env.check_success())
  dump(stage/'predicates.json',{'plan_success':plan,'replay_success':replay,'checked_before_close_and_merge':True})
  if not replay:raise RuntimeError('expert replay predicate failed')
  descriptions=official.build_episode_instructions(cfg,info,a.index)
  dump(stage/'scene_info.json',info)
  raw=stage/'.cache'/f'episode{a.index}'; paths=sorted(raw.glob('*.pkl'),key=lambda p:int(p.stem))
  data=pkl2hdf5.parse_dict_structure(pkl2hdf5.load_pkl_file(paths[0]))
  for p in paths:pkl2hdf5.append_data_to_structure(data,pkl2hdf5.load_pkl_file(p))
  (stage/'data').mkdir(); h5=stage/'data'/f'episode_{a.index:07d}.hdf5'
  # Official data writer directly; no optional ffmpeg preview dependency.
  pkl2hdf5.create_xpolicylab_hdf5(data,str(h5),descriptions['seen'],15)
  report=validate(h5,raw,descriptions['seen']);env.close_env()
  report.update(task='place_mouse_pad',seed=a.seed,success=True,plan_success=plan,replay_success=replay,source_revision=REV,config='demo_clean',timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  dump(stage/'validation.json',report)
 official.run=run;official.main(task_name='place_mouse_pad',task_config='demo_clean')
def inventory(out):
 baseline=ROOT/'results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10/manifest.json'
 b=json.loads(baseline.read_text());weak=[{'task':x['task'],'baseline_successes':x['successes'],'target_successful_demos':50} for x in b['tasks'] if x['successes']<10]
 assert len(weak)==44 and len(b['tasks'])==50
 dump(out/'tasks44.json',{'baseline_path':str(baseline),'baseline_sha256':digest(baseline),'tasks':weak})
 used=set(); sources=[]
 # Inventory training seed files and recursively read seed metadata in historical outputs.
 def extract(o):
  if isinstance(o,dict):
   for k,v in o.items():
    if k in ('seed','episode_seed') and isinstance(v,int):used.add(v)
    elif k in ('seeds','seed_list') and isinstance(v,list):used.update(x for x in v if isinstance(x,int))
    else:extract(v)
  elif isinstance(o,list):
   for v in o:extract(v)
 for base in [ROOT/'data',ROOT/'results',ROOT/'robotwin_repo/data',ROOT/'robotwin_repo_ba0fd21/data',ROOT/'robotwin_repo_e154702/data']:
  if not base.exists():continue
  for p in base.rglob('*'):
   if not p.is_file() or out in p.parents:continue
   if p.name=='seed.txt':
    vals=[int(x) for x in p.read_text().split()];used.update(vals);sources.append({'path':str(p),'sha256':digest(p),'seeds':vals})
   elif p.suffix=='.json' and p.stat().st_size<10000000:
    try:extract(json.loads(p.read_text()))
    except (ValueError,UnicodeError):pass
 dump(out/'historical_seed_inventory.json',{'sources':sources,'known_seeds':sorted(used),'scope':'data/results seed.txt and JSON seed metadata; evaluation seed_start=100000 excluded by bound'})
 return used

def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);reserve(out)
 assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==REV
 assert (ROOT/'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k/params/manifest.ocdbt').is_file()
 used=inventory(out);(out/'.staging').mkdir();(out/'episodes').mkdir()
 env=os.environ.copy();lib=ROOT/'third_party/oidn-2.5.1/lib'
 env.update(CUDA_VISIBLE_DEVICES='0',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',ROBOTWIN_RENDER_BACKEND='rt',ROBOTWIN_RT_DENOISER='oidn',PYTHONUNBUFFERED='1',LD_PRELOAD=str(lib/'libOpenImageDenoise.so.2'),LD_LIBRARY_PATH=str(lib)+':'+env.get('LD_LIBRARY_PATH',''))
 manifest={'source_revision':REV,'started_at':time.time(),'episodes':[],'attempts':[],'wrapper_sha256':digest(__file__),'baseline_sha256':json.loads((out/'tasks44.json').read_text())['baseline_sha256']}
 for seed in range(a.start_seed,a.start_seed+a.max_attempts):
  assert 20000<=seed<100000
  if seed in used:continue
  reserve(out);idx=len(manifest['episodes']);stage=out/'.staging'/f'seed_{seed}';stage.mkdir()
  log=out/f'seed_{seed}.log';start=time.time()
  with log.open('w') as f:
   p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--stage',str(stage),'--index',str(idx),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   print(f'child_pid={p.pid} seed={seed} log={log}',flush=True)
   try:rc=p.wait(timeout=a.timeout)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();rc=124
  attempt={'seed':seed,'pid':p.pid,'returncode':rc,'elapsed_seconds':time.time()-start,'log':str(log)};manifest['attempts'].append(attempt)
  if rc==0:
   report=json.loads((stage/'validation.json').read_text());h5=next((stage/'data').glob('*.hdf5'));assert digest(h5)==report['hdf5_sha256'];reserve(out)
   report['config_sha256']=digest(stage/'resolved_config.json');report['instruction_sha256']=digest(next((stage/'instruction').glob('*.json')))
   dump(stage/'validation.json',report)
   # Validation is completed against all raw frames; drop only new transient cache.
   shutil.rmtree(stage/'.cache');final=out/'episodes'/f'episode_{idx:07d}';os.rename(stage,final)
   report['episode_directory']=str(final);manifest['episodes'].append(report)
  else:
   attempt['error']='see per-attempt log; rejected and never published'
   shutil.rmtree(stage)
  manifest['elapsed_seconds']=time.time()-manifest['started_at'];dump(out/'manifest.json',manifest)
  if len(manifest['episodes'])==2:break
 assert len(manifest['episodes'])==2,'bounded attempt limit exhausted'
 print(json.dumps({'episodes':2,'manifest':str(out/'manifest.json'),'elapsed_seconds':manifest['elapsed_seconds']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--stage');p.add_argument('--index',type=int);p.add_argument('--seed',type=int);p.add_argument('--out');p.add_argument('--start-seed',type=int,default=20010);p.add_argument('--max-attempts',type=int,default=8);p.add_argument('--timeout',type=int,default=900);a=p.parse_args()
 if a.worker:worker(a)
 else:main(a)
