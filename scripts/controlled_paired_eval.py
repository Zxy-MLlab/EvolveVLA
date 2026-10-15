#!/usr/bin/env python3
"""Frozen paired panel. prepare is CPU-only; run explicitly launches GPU processes.

Each case uses a fresh policy and simulator process (42 model loads). No resume,
retry or seed substitution. Input equality proves initial observed inputs only.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

# -I removes script-directory imports; add only our owned external scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path('/root/autodl-tmp/EvolveVLA')
BASE = ROOT/'results/pi05_official_orbax_baseline/myungkyu_59999_full_clean_10'
PILOT = ROOT/'results/gen1_pilot_2000_clean_two_task_20261010'
MODELS = {'reference': str(ROOT/'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k'),
          'pilot': str(ROOT/'checkpoints/gen1_pilot_2000_20261010/1999')}
CONFIG = dict(episode_steps=2000, policy_seed=0, noise_source='jax', normalization='quantile',
              prompt_prefix='none', workers=1, task_config='demo_clean',
              robotwin_root=str(ROOT/'robotwin_repo_ba0fd21'),
              sim_python='/root/autodl-tmp/conda_envs/robotwin_cu128/bin/python',
              inference_python='/root/autodl-tmp/conda_envs/scratch_orbax/bin/python')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def prompts(path):
    result = {}; seed = None
    for line in Path(path).read_text().splitlines():
        m = re.match(r'EXPERT_SEED=(\d+) EXPERT_OK=True', line)
        if m: seed = int(m[1])
        if line.startswith('INSTRUCTION=') and seed is not None:
            value = ast.literal_eval(line.split('=', 1)[1])
            if seed in result and result[seed] != value:
                raise ValueError('ambiguous historical prompt')
            result[seed] = value
    return result

def prepare(path):
    cases = []; sources = {}
    for task in ('hanging_mug', 'place_mouse_pad'):
        result = BASE/task/'result.json'; log = BASE/task/'run.log'
        data = json.loads(result.read_text())
        assert Path(data['checkpoint']).resolve() == Path(MODELS['reference']).resolve()
        seeds = [r['seed'] for r in data['reports']]
        assert len(seeds) == len(set(seeds)) == 10
        pp = prompts(log)
        for seed in seeds:
            cases.append(dict(id=f'{task}_{seed}', task=task, seed=seed, panel='primary',
                              prompt=pp[seed], prompt_source=str(log)))
        sources.update({str(p): sha(p) for p in (result, log)})
    log = PILOT/'place_mouse_pad/run.log'
    cases.append(dict(id='place_mouse_pad_100010', task='place_mouse_pad', seed=100010,
                      panel='supplemental', prompt=prompts(log)[100010], prompt_source=str(log)))
    sources[str(log)] = sha(log)
    # Pin every parameter shard and normalization file, not merely the directory name.
    identities = {}
    for label, model in MODELS.items():
        files = sorted(p for sub in ('params', 'assets') for p in (Path(model)/sub).rglob('*') if p.is_file())
        assert files and (Path(model)/'params/_METADATA').is_file()
        identities[label] = {str(p): sha(p) for p in files}
    code = {str(ROOT/'scripts'/name): sha(ROOT/'scripts'/name) for name in
            ('controlled_paired_eval.py', 'robotwin_eval_harness.py', 'robotwin_public_openpi_harness.py', 'pi05_openpi_adapter.py')}
    manifest = dict(version=1, models=MODELS, config=CONFIG, cases=cases, sources=sources,
                    identities=identities, code=code,
                    limitation='Equality proves initial observed inputs only, not complete simulator state or later trajectories.',
                    prompt_provenance='Primary prompts from historical baseline; supplemental mouse100010 from pilot. Selection uses task/seed, never outcome.')
    validate(manifest, check_files=False)
    write(path, manifest)

def validate(m, check_files=True):
    if m['version'] != 1 or m['models'] != MODELS or m['config'] != CONFIG:
        raise ValueError('unsupported manifest configuration/model')
    cases = m['cases']
    if len(cases) != 21 or len({c['id'] for c in cases}) != 21:
        raise ValueError('panel must contain 21 unique cases')
    for task in ('hanging_mug', 'place_mouse_pad'):
        expected = [r['seed'] for r in json.loads((BASE/task/'result.json').read_text())['reports']]
        actual = [c['seed'] for c in cases if c['task'] == task and c['panel'] == 'primary']
        if actual != expected: raise ValueError('primary panel changed')
    supplemental = [c for c in cases if c['panel'] == 'supplemental']
    if len(supplemental) != 1 or (supplemental[0]['task'], supplemental[0]['seed']) != ('place_mouse_pad', 100010):
        raise ValueError('supplemental panel changed')
    for c in cases:
        log = (BASE if c['panel'] == 'primary' else PILOT)/c['task']/'run.log'
        if c['id'] != f"{c['task']}_{c['seed']}" or c['prompt_source'] != str(log) or c['prompt'] != prompts(log)[c['seed']]:
            raise ValueError('prompt/case identity changed')
    if check_files:
        for files in (m['sources'], m['code'], *m['identities'].values()):
            for path, digest in files.items():
                if sha(path) != digest: raise ValueError(f'identity changed: {path}')

def capture(directory, images, state, prompt):
    import numpy as np
    directory.mkdir()
    hashes = {}
    for key in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
        a = np.ascontiguousarray(images[key], dtype=np.uint8)
        if a.ndim != 3 or a.shape[-1] != 3: raise ValueError('invalid camera')
        np.save(directory/f'{key}.npy', a, allow_pickle=False)
        hashes[key] = dict(shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest())
    a = np.asarray(state, dtype='<f4').reshape(14)
    if not np.isfinite(a).all(): raise ValueError('nonfinite state')
    np.save(directory/'state.npy', a, allow_pickle=False)
    hashes['state'] = hashlib.sha256(a.tobytes()).hexdigest()
    hashes['prompt'] = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    write(directory/'hashes.json', hashes)
    return hashes

def guard_policy(policy):
    if policy.checkpoint_format != 'openpi_orbax' or policy.noise_source != 'jax':
        raise ValueError('controlled evaluation requires Orbax explicit JAX noise')

def episode(first, receive, send, make_policy, directory, case, reference=None, pilot=False):
    import numpy as np
    from robotwin_public_openpi_harness import _observation_for_openpi
    if first.get('seed') != case['seed']: return dict(status='seed_mismatch')
    if first['type'] == 'case_status':
        return {k: v for k, v in first.items() if k not in ('type', 'seed')}
    if first['type'] != 'episode_start': return dict(status='protocol_error')
    if first['instruction'] != case['prompt']: return dict(status='prompt_mismatch')
    images, state = _observation_for_openpi(first['observation'])
    hashes = capture(directory/'initial', images, state, case['prompt'])
    if pilot and reference is None: return dict(status='missing_reference', hashes=hashes)
    if pilot and hashes != reference: return dict(status='initial_mismatch', hashes=hashes)
    # Fresh adapter per accepted case initializes both policy and explicit noise keys
    # to policy_seed. No resume/advance_rng and no edits to pinned Policy code.
    policy = make_policy(); guard_policy(policy)
    steps = 0; success = False
    while steps < CONFIG['episode_steps']:
        chunk = policy.infer_chunk(images=images, state=state, prompt=case['prompt'])
        for action in chunk:
            action = np.asarray(action, dtype=np.float32).reshape(14)
            if not np.isfinite(action).all(): raise ValueError('nonfinite action')
            send({'cmd': 'action', 'action': action.tolist()})
            response = receive()
            if response['type'] != 'step': return dict(status='simulator_error', detail=response, hashes=hashes)
            steps += 1
            success |= bool(response.get('info', {}).get('is_success'))
            if response.get('terminated') or response.get('truncated'):
                return dict(status='completed', success=success, steps=steps, hashes=hashes)
            images, state = _observation_for_openpi(response['observation'])
            if steps >= CONFIG['episode_steps']: break
    return dict(status='completed', success=success, steps=steps, hashes=hashes)

def worker(m, label, case, directory, reference):
    from robotwin_public_openpi_harness import _configure_sim_renderer_env
    from robotwin_eval_harness import receive_message, send_message
    from pi05_openpi_adapter import PublicPi05Adapter
    env = os.environ.copy(); _configure_sim_renderer_env(env)
    env['ROBOTWIN_CONTROLLED_OUTPUT'] = str(directory.resolve())
    command = [CONFIG['sim_python'], '-I', '-B', str(ROOT/'scripts/robotwin_eval_harness.py'), '--role', 'sim',
               '--controlled-case', '--episodes', '1', '--task', case['task'], '--seed', str(case['seed']),
               '--episode-steps', str(CONFIG['episode_steps']), '--robotwin-root', CONFIG['robotwin_root'],
               '--task-config', CONFIG['task_config'], '--first-instruction', case['prompt'],
               '--first-instruction-seed', str(case['seed'])]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
                             cwd=CONFIG['robotwin_root'])
    try:
        result = episode(receive_message(child.stdout), lambda: receive_message(child.stdout),
                         lambda x: send_message(child.stdin, x),
                         lambda: PublicPi05Adapter(m['models'][label], device='cuda:0', prompt_prefix='none',
                                                   normalization='quantile', policy_seed=0, noise_source='jax'),
                         directory, case, reference, pilot=label == 'pilot')
    except Exception as exc:
        result = dict(status='worker_error', message=f'{type(exc).__name__}: {exc}')
    finally:
        try: send_message(child.stdin, {'cmd': 'stop'}); child.stdin.close()
        except (BrokenPipeError, OSError): pass
        try: child.wait(timeout=30)
        except subprocess.TimeoutExpired: child.kill(); child.wait()
    write(directory/'result.json', dict(case=case['id'], panel=case['panel'], seed=case['seed'], **result))

def summarize(records):
    totals = {}
    for label, c, r in records:
        key = f"{label}/{c['panel']}"
        t = totals.setdefault(key, {'statuses': {}, 'completed_successes': 0})
        t['statuses'][r['status']] = t['statuses'].get(r['status'], 0) + 1
        if r['status'] == 'completed': t['completed_successes'] += int(r['success'])
    paired = {}
    by_case = {}
    for label, c, r in records:
        by_case.setdefault(c['id'], (c, {}))[1][label] = r
    for panel in ('primary', 'supplemental'):
        t = dict(requested_pairs=20 if panel == 'primary' else 1, executed_cases=[],
                 completed_pairs=0, reference_only_wins=0, pilot_only_wins=0,
                 both_success=0, both_failure=0, pair_statuses={})
        for cid, (c, rr) in by_case.items():
            if c['panel'] != panel: continue
            t['executed_cases'].append(cid)
            a, b = rr.get('reference', {}), rr.get('pilot', {})
            status = f"{a.get('status','not_executed')}/{b.get('status','not_executed')}"
            t['pair_statuses'][status] = t['pair_statuses'].get(status, 0)+1
            if a.get('status') == b.get('status') == 'completed':
                t['completed_pairs'] += 1
                key = ('both_success' if a['success'] and b['success'] else
                       'reference_only_wins' if a['success'] else
                       'pilot_only_wins' if b['success'] else 'both_failure')
                t[key] += 1
        paired[panel] = t
    totals['paired'] = paired
    return totals

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'run', 'worker'])
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--label', choices=MODELS)
    p.add_argument('--case', type=int, help='run only this manifest index for both models (bounded smoke)')
    p.add_argument('--timeout-seconds', type=int, default=1800)
    a = p.parse_args()
    if a.mode == 'prepare': prepare(a.manifest); return
    for name in ('ROBOTWIN_SKIP_EXPERT', 'ROBOTWIN_TRAJECTORY_JSONL'):
        if os.environ.get(name): raise ValueError(f'{name} forbidden')
    m = json.loads(a.manifest.read_text()); validate(m)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('ROBOTWIN_'):
            env.pop(key)
    for key in ('XLA_PYTHON_CLIENT_ALLOCATOR', 'XLA_PYTHON_CLIENT_PREALLOCATE',
                'XLA_CLIENT_MEM_FRACTION', 'PYTHONPATH', 'JAX_PLATFORMS'):
        env.pop(key, None)
    env.update(OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache',
               XLA_PYTHON_CLIENT_MEM_FRACTION='0.7')
    env.update(ROBOTWIN_RENDER_BACKEND='rt', ROBOTWIN_RT_DENOISER='oidn',
               ROBOTWIN_OIDN_RUNTIME=str(ROOT/'third_party/oidn-2.5.1/lib'),
               ROBOTWIN_OPENPI_SRC=str(Path(CONFIG['robotwin_root'])/'XPolicyLab/policy/Pi_05/openpi/src'),
               CUDA_VISIBLE_DEVICES='0')
    os.environ.clear(); os.environ.update(env)
    if a.mode == 'worker':
        ref = a.output/'reference'/m['cases'][a.case]['id']/'result.json'
        reference = json.loads(ref.read_text()).get('hashes') if ref.exists() else None
        worker(m, a.label, m['cases'][a.case], a.output/a.label/m['cases'][a.case]['id'], reference)
        return
    if a.output is None: p.error('--output required')
    a.output = a.output.resolve(); a.output.mkdir(parents=False, exist_ok=False)
    write(a.output/'manifest.json', m)
    records = []
    if a.case is not None and not 0 <= a.case < len(m['cases']): p.error('case index out of range')
    for label in MODELS:
        (a.output/label).mkdir()
        for index, case in enumerate(m['cases']):
            if a.case is not None and a.case != index: continue
            directory = a.output/label/case['id']; directory.mkdir()
            with (directory/'worker.log').open('x') as log:
                try:
                    proc = subprocess.Popen([CONFIG['inference_python'], '-I', '-B', str(Path(__file__).resolve()), 'worker',
                           '--manifest', str(a.manifest.resolve()), '--output', str(a.output), '--label', label,
                           '--case', str(index)], env=env, stdout=log, stderr=subprocess.STDOUT,
                           start_new_session=True)
                    try:
                        returncode = proc.wait(timeout=a.timeout_seconds)
                    except BaseException:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try: proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                        raise
                except subprocess.TimeoutExpired:
                    returncode = 'timeout'
            result = directory/'result.json'
            if not result.exists(): write(result, dict(status='worker_error', returncode=returncode))
            records.append((label, case, json.loads(result.read_text())))
            # Exclusive per-step snapshots survive interruption without overwriting evidence.
            write(a.output/f'progress_{len(records):02d}.json', summarize(records))
            if records[-1][2]['status'] in ('worker_error', 'protocol_error'):
                write(a.output/'summary.json', dict(stopped_on_systemic_error=True, totals=summarize(records)))
                return
    write(a.output/'summary.json', summarize(records))

if __name__ == '__main__': main()
