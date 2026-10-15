#!/usr/bin/env python3
"""CPU-only, bounded horizon-target audit; never imports training frameworks.
Run with Python -I -B. Existing inputs are read-only; report creation is exclusive.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import types
import uuid
import numpy as np
import h5py

ROOT = Path('/root/autodl-tmp/EvolveVLA')
PIN = ROOT / 'robotwin_repo_ba0fd21/XPolicyLab'
OP = PIN / 'policy/Pi_05/openpi'
PARTS = ('left_arm_joint_states', 'left_ee_joint_states', 'right_arm_joint_states', 'right_ee_joint_states')
MASK = np.array([True]*6+[False]+[True]*6+[False])

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def extracted(path, name, namespace, cls=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    node.returns = None
    for a in node.args.args + node.args.kwonlyargs:
        a.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[name]

class Stats:
    def __init__(self, norm=None):
        self.n = 0
        self.lo = np.full(14, np.inf)
        self.hi = -self.lo.copy()
        self.sumabs = np.zeros(14)
        self.sumsq = np.zeros(14)
        self.norm = norm
        self.below = np.zeros(14, dtype=np.int64)
        self.above = self.below.copy()
        self.zmax = np.zeros(14)
        self.zsq = np.zeros(14)
        self.zgt2 = self.below.copy()
        self.outliers = [None]*14
    def add(self, values, meta=None, anchors=None, offsets=None, padded=None):
        x = values.reshape(-1, 14)
        if not len(x): return
        assert np.isfinite(x).all()
        self.n += len(x)
        self.lo = np.minimum(self.lo, x.min(0)); self.hi = np.maximum(self.hi, x.max(0))
        self.sumabs += np.abs(x).sum(0, dtype=np.float64)
        self.sumsq += np.square(x.astype(np.float64)).sum(0)
        if self.norm is not None:
            q01, q99 = self.norm
            z = (x-q01)/(q99-q01+1e-6)*2.-1.
            self.below += (x < q01).sum(0); self.above += (x > q99).sum(0)
            self.zsq += np.square(z).sum(0)
            self.zgt2 += (np.abs(z)>2).sum(0)
            idx = np.abs(z).argmax(0)
            for d, i in enumerate(idx):
                v = abs(z[i,d])
                if v > self.zmax[d]:
                    self.zmax[d] = v
                    if meta is not None:
                        self.outliers[d] = dict(meta, dimension=d, value=float(x[i,d]), normalized=float(z[i,d]), anchor=int(anchors[i]), horizon_offset=int(offsets[i]), padded=bool(padded[i]))
    def report(self):
        if not self.n: return {'count_vectors':0}
        r = dict(count_vectors=self.n, minimum=self.lo.tolist(), maximum=self.hi.tolist(), maximum_absolute=np.maximum(abs(self.lo),abs(self.hi)).tolist(), mean_absolute=(self.sumabs/self.n).tolist(), rms=np.sqrt(self.sumsq/self.n).tolist())
        if self.norm is not None:
            r.update(below_q01_fraction=(self.below/self.n).tolist(), above_q99_fraction=(self.above/self.n).tolist(), outside_quantiles_fraction=((self.below+self.above)/self.n).tolist(), outside_quantiles_fraction_all_physical_dimensions=float((self.below+self.above).sum()/(14*self.n)), normalized_maximum_absolute=self.zmax.tolist(), normalized_rms=np.sqrt(self.zsq/self.n).tolist(), normalized_abs_gt2_fraction=(self.zgt2/self.n).tolist(), representative_maxima=self.outliers)
        return r

def logs():
    bases = {'pilot':ROOT/'results/gen1_pilot_2000_clean_two_task_20261010', 'baseline':ROOT/'results/pi05_official_orbax_baseline/protocol_v11_ba0fd21_full_clean_10_oidn_jax0'}
    out = {}; hashes = {}
    for task, seeds in {'place_mouse_pad':[100005,100007], 'hanging_mug':[100005,100008,100010]}.items():
        out[task] = {}
        for label, base in bases.items():
            path = base/task/'run.log'; result = base/task/'result.json'
            hashes[str(path)] = sha(path); hashes[str(result)] = sha(result)
            data = json.loads(result.read_text())
            selected = {s:{'result':next((r for r in data['reports'] if r['seed']==s), None), 'evidence':[]} for s in seeds}
            seed = None
            with path.open(errors='replace') as f:
                for number, line in enumerate(f, 1):
                    m = re.search(r'^EXPERT_SEED=(\d+) EXPERT_OK=(True|False)', line)
                    if m: seed = int(m[1])
                    if seed in selected and (m or line.startswith(('INSTRUCTION=', 'SEED_ARM ', 'RESET_SEED=', 'RESET_SEED '))):
                        selected[seed]['evidence'].append({'line':number, 'text':line.strip()})
            out[task][label] = {'successes':data['successes'], 'completed':data['episodes_completed'], 'selected':selected}
        for s in seeds:
            entries = [out[task][k]['selected'][s] for k in ('baseline','pilot')]
            instructions = [[e['text'] for e in v['evidence'] if e['text'].startswith('INSTRUCTION=')] for v in entries]
            out[task].setdefault('comparison', {})[s] = {'same_logged_instruction':bool(instructions[0]) and instructions[0]==instructions[1], 'exact_reset_pairing_established':False, 'limitation':'Logged expert/report seed and arm are available, not complete reset RNG/scene fingerprints; instructions and accepted seed sets can differ.'}
    return out, hashes

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--split', type=Path, default=ROOT/'data/gen1_pilot_two_task_20261010/split_manifest.json')
    p.add_argument('--norm', type=Path, default=ROOT/'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k/assets/aloha_agilex_sim/norm_stats.json')
    p.add_argument('--loader-code', type=Path, default=Path('/root/autodl-tmp/conda_envs/gen1_train_ready/lib/python3.11/site-packages/lerobot/datasets/lerobot_dataset.py'))
    p.add_argument('--chunk-anchors', type=int, default=64)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    assert a.root == ROOT, 'Pinned root only'
    assert 1 <= a.chunk_anchors <= 1024
    output = a.output or ROOT/'outputs'/('gen1_pilot_target_audit_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.json')
    assert output.parent.resolve() == (ROOT/'outputs').resolve() and not output.exists()
    manifest = json.loads(a.split.read_text()); rows = manifest['episodes']
    assert manifest['status']=='passed' and len(rows)==100
    assert Counter((r['task'],r['split']) for r in rows)==Counter({(t,s):n for t in ('place_mouse_pad','hanging_mug') for s,n in [('train',45),('val',5)]})
    assert all(len({r[k] for r in rows})==100 for k in ('path','sha256','seed'))
    ns = json.loads(a.norm.read_text())['norm_stats']; norms = {}
    for name in ('state','actions'):
        low, high = (np.asarray(ns[name][k],dtype=np.float64) for k in ('q01','q99'))
        assert low.shape == high.shape and low.shape in ((14,), (32,))
        assert np.isfinite(low).all() and np.isfinite(high).all() and (high[:14]>low[:14]).all()
        norms[name] = (low[:14], high[:14])
    converter = PIN/'scripts/transform_lerobot_v30_format.py'
    namespace = {'np':np}
    ensure = extracted(converter, '_ensure_2d_float32', namespace)
    # Execute only the verified index method, with a minimal tensor stub; no torch import.
    query = extracted(a.loader_code, '_get_query_indices', {'torch':types.SimpleNamespace(BoolTensor=lambda v:np.array(v,dtype=bool))}, 'LeRobotDataset')
    for length in (2,17,50,153):
        obj = types.SimpleNamespace(meta=types.SimpleNamespace(episodes=[{'dataset_from_index':11,'dataset_to_index':11+length}]),delta_indices={'action':list(range(50))})
        for anchor in (0,length-1):
            idx,pad = query(obj,11+anchor,0)
            np.testing.assert_array_equal(np.array(idx['action'])-11,np.minimum(anchor+np.arange(50),length-1))
            np.testing.assert_array_equal(pad['action_is_pad'],anchor+np.arange(50)>=length)
    groups = {}; sources = []; total_pad = 0; total_frames = 0
    for row in rows:
        path = Path(row['path']); digest = sha(path)
        assert digest == row['sha256'], path
        assert all(row[k] is True for k in ('success','plan_success','replay_success'))
        with h5py.File(path,'r') as f:
            assert f.attrs['source_format']=='RoboTwin'
            assert int(np.asarray(f['additional_info/frequency']))==15
            arrays = [np.concatenate([ensure(f[group][part][()],part) for part in PARTS],axis=1) for group in ('state','action')]
        state, action = arrays; n = len(state)
        assert state.shape==action.shape==(row['frames'],14) and n>1
        np.testing.assert_array_equal(state[1:],action[:-1])
        key = row['task']+'/'+row['split']
        if key not in groups:
            groups[key] = {'episodes':0, 'padded_target_vectors':0, 'anchors_with_padding':0, 'state':Stats(norms['state']), 'physical_action':Stats(), 'target_all':Stats(norms['actions']), 'target_real':Stats(norms['actions']), 'target_padded':Stats(norms['actions']), 'target_offset0':Stats(norms['actions'])}
        g = groups[key]; g['episodes'] += 1; g['anchors_with_padding'] += min(49,n)
        meta = {k:row[k] for k in ('task','split','seed','path')}
        g['state'].add(state,meta,np.arange(n),np.zeros(n,int),np.zeros(n,bool)); g['physical_action'].add(action)
        for begin in range(0,n,a.chunk_anchors):
            anchors = np.arange(begin,min(begin+a.chunk_anchors,n))
            indices = anchors[:,None]+np.arange(50)[None,:]
            pad = indices>=n
            target = action[np.minimum(indices,n-1)].copy()
            target -= np.where(MASK,state[anchors],0)[:,None,:]
            av = np.broadcast_to(anchors[:,None],pad.shape).ravel(); hv = np.broadcast_to(np.arange(50),pad.shape).ravel(); pv=pad.ravel()
            flat=target.reshape(-1,14)
            for label, sel in [('target_all',np.ones(len(flat),bool)),('target_real',~pv),('target_padded',pv),('target_offset0',hv==0)]:
                g[label].add(flat[sel],meta,av[sel],hv[sel],pv[sel])
            g['padded_target_vectors'] += int(pad.sum()); total_pad += int(pad.sum())
        sources.append(dict(meta,sha256=digest,hash_verified=True,frames=n)); total_frames += n
        if len(sources)%10==0: print(f'verified {len(sources)}/100 episodes',flush=True)
    evidence, log_hashes = logs()
    files = [Path(__file__),a.split,a.norm,a.loader_code,converter,PIN/'utils/data_loader.py',ROOT/'scripts/gen1_convert_ready.py',ROOT/'scripts/gen1_train_pilot.py',ROOT/'outputs/gen1_pilot_2000_20261010/run_config.json']
    files += [OP/'src/openpi'/s for s in ('transforms.py','policies/aloha_policy.py','training/config.py','training/data_loader.py','models/pi0.py')]
    report = {'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,'argv':sys.argv,'frames':total_frames,'episodes':len(sources),'source_hashes_verified':True,'horizon':50,'physical_dimensions':14,'model_dimensions':32,'absolute_gripper_indices':[6,13],'chunk_anchors':a.chunk_anchors,'normalization_dimensions':{k:len(ns[k]['q01']) for k in norms},'normalization':'(x-q01)/(q99-q01+1e-6)*2-1; no clipping; physical 14 dimensions sliced before model zero-padding','tail_convention':'action[min(anchor+offset, episode_length-1)]; offsets 0..49; subtract the same anchor state for 12 joints only. Verified by AST-executing installed training loader method with nonzero episode base. action_is_pad is dropped by repack/AlohaInputs and does not mask loss.','padded_target_vectors':total_pad,'total_target_vectors':total_frames*50,'padded_fraction':total_pad/(total_frames*50),'groups':{k:{name:(v.report() if isinstance(v,Stats) else v) for name,v in g.items()} for k,g in groups.items()},'sources':sources,'sha256':{str(f):sha(f) for f in files}|log_hashes,'evaluation_comparison':evidence,'limitations':['Source numeric arrays only; images and converted parquet not reread; rely on existing all-frame conversion audits for source-to-dataset equivalence.','One episode numeric state/action in memory; horizon tensors bounded by chunk_anchors; SHA256 streams entire files including images.','Statistics weight every anchor and horizon offset equally, not the realized stochastic 2000-step sampling history.','Quantile exceedance is expected for some samples; no inference that target tails cause rollout failure.','No model loads, GPU use, new train/eval/collection, or normalization changes.','Matching logged seeds does not establish exact reset pairing; scene/RNG fingerprints unavailable.','32D norm assets, if supplied, are sliced to 14 physical dimensions before zero padding; artificial padded dimensions excluded from physical exceedance denominators.']}
    with output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    summary = {'report':str(output),'frames':total_frames,'padded_fraction':report['padded_fraction'],
               'groups':{k:{'state_outside':v['state'].report()['outside_quantiles_fraction_all_physical_dimensions'],
                            'target_outside':v['target_all'].report()['outside_quantiles_fraction_all_physical_dimensions'],
                            'max_abs_normalized_target':float(v['target_all'].zmax.max())}
                         for k,v in groups.items()}}
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
