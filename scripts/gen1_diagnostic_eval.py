#!/usr/bin/env python3
"""Bounded original/original diagnostic. prepare is CPU-only; run is explicit.

No retry, replacement, video, or modification of frozen evaluator/simulator files.
State hashes describe observed joint drive-target state, NOT full physics state.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controlled_paired_eval as c

ROOT = c.ROOT
PRIMARY = ROOT/'outputs/controlled_paired_eval_manifest_20261011_v3.json'
TASKS = ('hanging_mug', 'place_mouse_pad')
GIB = 1024**3
LIMITS = dict(output_budget_bytes=2*GIB, min_free_bytes=70*GIB, timeout_seconds=1800)


def selected(primary):
    """Order in frozen primary manifest only; never inspect rollout outcomes."""
    cases = []
    for task in TASKS:
        rows = [x for x in primary['cases'] if x['task'] == task and x['panel'] == 'primary']
        if len(rows) < 2 or len({x['seed'] for x in rows}) != len(rows):
            raise ValueError('invalid primary seed panel')
        for row in rows[:2]:
            for repeat in (1, 2):
                cases.append(dict(id=f"{row['id']}_r{repeat}", pair=row['id'], task=task,
                                  seed=row['seed'], prompt=row['prompt'], panel='diagnostic',
                                  prompt_source=row['prompt_source'], repetition=repeat))
    return cases


def disk_guard(directory, check_size=True):
    directory = Path(directory)
    size = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()) if check_size else 0
    if size >= LIMITS['output_budget_bytes']:
        raise RuntimeError('diagnostic output reached 2 GiB budget')
    if shutil.disk_usage(directory).free < LIMITS['min_free_bytes']:
        raise RuntimeError('diagnostic requires at least 70 GiB free')


def source_code(primary):
    paths = set(primary['code'])
    paths.add(str(Path(__file__).resolve()))
    return {p: c.sha(p) for p in sorted(paths)}


def prepare(output, primary_path=PRIMARY):
    primary_path = Path(primary_path).resolve()
    primary = json.loads(primary_path.read_text())
    # Validate frozen provenance, including parameter shards (CPU I/O only).
    c.validate(primary)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    disk_guard(output.parent, check_size=False)
    m = dict(version=1, primary_manifest=str(primary_path), primary_sha256=c.sha(primary_path),
             model=c.MODELS['reference'], config=c.CONFIG, limits=LIMITS,
             cases=selected(primary), identities=primary['identities']['reference'],
             sources=primary['sources'], code=source_code(primary),
             selection='First two primary rows per task, manifest order, two original repetitions each; outcome-independent.',
             state_semantics='Canonical float32 observed 14-D joint drive-target state; not complete simulator physics state.',
             protocol='Fresh processes, expert gate retained, explicit JAX noise seed 0, fixed prompt, no retries/replacements/videos.')
    output.mkdir(exist_ok=False)
    c.write(output/'manifest.json', m)
    (output/'manifest.json').chmod(0o444)
    c.write(output/'manifest.sha256.json', {'sha256': c.sha(output/'manifest.json')})
    return m


def validate(m, manifest):
    manifest = Path(manifest)
    if c.sha(manifest) != json.loads((manifest.parent/'manifest.sha256.json').read_text())['sha256']:
        raise ValueError('diagnostic manifest identity changed')
    primary_path = Path(m['primary_manifest'])
    if c.sha(primary_path) != m['primary_sha256']:
        raise ValueError('primary manifest changed')
    primary = json.loads(primary_path.read_text())
    if (m['version'] != 1 or m['model'] != c.MODELS['reference'] or
        m['config'] != c.CONFIG or m['limits'] != LIMITS or m['cases'] != selected(primary) or
        m['identities'] != primary['identities']['reference'] or m['sources'] != primary['sources'] or
        set(m['code']) != set(source_code(primary))):
        raise ValueError('diagnostic configuration/selection/identity changed')
    for files in (m['identities'], m['sources'], m['code']):
        for path, digest in files.items():
            if c.sha(path) != digest:
                raise ValueError(f'identity changed: {path}')


def array_hash(value):
    import numpy as np
    a = np.ascontiguousarray(value, dtype='<f4')
    if not np.isfinite(a).all():
        raise ValueError('nonfinite diagnostic array')
    return dict(shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest())


class Trace:
    def __init__(self, directory):
        self.path = directory/'trace.jsonl'
        self.stream = self.path.open('x', buffering=1)
        self.chunk = 0
        self.step = 0

    def emit(self, kind, **fields):
        self.stream.write(json.dumps(dict(kind=kind, **fields), allow_nan=False)+'\n')

    def policy(self, policy):
        c.guard_policy(policy)
        original = policy.infer_chunk
        def infer(**kwargs):
            chunk = original(**kwargs)
            if array_hash(chunk)['shape'] != [50, 14]:
                raise ValueError('requires full 50-action chunk')
            self.emit('chunk', index=self.chunk, before_step=self.step, actions=array_hash(chunk))
            self.chunk += 1
            return chunk
        policy.infer_chunk = infer
        return policy

    def response(self, value):
        if value.get('type') == 'step':
            self.step += 1
            state = value.get('diagnostic_state')
            if state is None and not (value.get('terminated') or value.get('truncated')):
                raise ValueError('missing nonterminal simulator state hash')
            if state is not None and state.get('shape') != [14]:
                raise ValueError('invalid simulator state hash')
            self.emit('step', index=self.step, state=state,
                      terminated=bool(value.get('terminated')), truncated=bool(value.get('truncated')),
                      success=bool(value.get('info', {}).get('is_success')))
        return value


def diagnostic_episode(first, receive, send, make_policy, directory, case):
    trace = Trace(directory)
    trace.emit('gate', seed=first.get('seed'), status=('accepted' if first.get('type') == 'episode_start'
               else first.get('status', 'protocol_error')),
               plan_success=first.get('plan_success'), check_success=first.get('check_success'))
    try:
        result = c.episode(first, lambda: trace.response(receive()), send,
                           lambda: trace.policy(make_policy()), directory, case)
        result['gate'] = 'accepted' if first.get('type') == 'episode_start' else first.get('status', 'unknown')
        return result
    finally:
        trace.stream.close()


def simulator():
    """Read-only IPC observation shim; terminal observation is explicitly absent.

    No extra get_obs/check_success/step calls and no RNG use. Hash only the
    existing observation in the IPC payload. Copy the envelope, never its arrays.
    Timing can differ; these A/A runs use the same instrumented protocol.
    """
    import robotwin_eval_harness as h
    expected = (ROOT/'scripts/robotwin_eval_harness.py').resolve()
    if Path(inspect.getsourcefile(h.send_message)).resolve() != expected:
        raise RuntimeError('unexpected traced simulator source location')
    original_send = h.send_message
    def send(stream, payload):
        if payload.get('type') == 'step':
            obs = payload.get('observation')
            state = array_hash(obs['agent_pos']) if obs is not None else None
            payload = dict(payload, diagnostic_state=state)
        return original_send(stream, payload)
    h.send_message = send
    try:
        return h.main()
    finally:
        h.send_message = original_send


def environment():
    if any(os.environ.get(k) for k in ('ROBOTWIN_SKIP_EXPERT', 'ROBOTWIN_TRAJECTORY_JSONL')):
        raise ValueError('expert bypass/legacy trajectory recording forbidden')
    env = {k: v for k, v in os.environ.items() if not k.startswith('ROBOTWIN_')}
    for k in ('PYTHONPATH', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_ALLOCATOR',
              'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_CLIENT_MEM_FRACTION'):
        env.pop(k, None)
    env.update(OPENPI_DATA_HOME='/root/autodl-tmp/openpi_cache', XLA_PYTHON_CLIENT_MEM_FRACTION='0.7',
               ROBOTWIN_RENDER_BACKEND='rt', ROBOTWIN_RT_DENOISER='oidn',
               ROBOTWIN_OIDN_RUNTIME=str(ROOT/'third_party/oidn-2.5.1/lib'),
               ROBOTWIN_OPENPI_SRC=str(Path(c.CONFIG['robotwin_root'])/'XPolicyLab/policy/Pi_05/openpi/src'),
               CUDA_VISIBLE_DEVICES='0')
    return env


def worker(m, case, directory):
    from robotwin_public_openpi_harness import _configure_sim_renderer_env
    from robotwin_eval_harness import receive_message, send_message
    from pi05_openpi_adapter import PublicPi05Adapter
    env = environment()
    _configure_sim_renderer_env(env)
    env['ROBOTWIN_CONTROLLED_OUTPUT'] = str(directory)
    command = [c.CONFIG['sim_python'], '-I', '-B', str(Path(__file__).resolve()), 'sim',
               '--role', 'sim', '--controlled-case', '--episodes', '1', '--task', case['task'],
               '--seed', str(case['seed']), '--episode-steps', str(c.CONFIG['episode_steps']),
               '--robotwin-root', c.CONFIG['robotwin_root'], '--task-config', c.CONFIG['task_config'],
               '--first-instruction', case['prompt'], '--first-instruction-seed', str(case['seed'])]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             env=env, cwd=c.CONFIG['robotwin_root'])
    try:
        result = diagnostic_episode(receive_message(child.stdout), lambda: receive_message(child.stdout),
                 lambda value: send_message(child.stdin, value),
                 lambda: PublicPi05Adapter(m['model'], device='cuda:0', prompt_prefix='none',
                                          normalization='quantile', policy_seed=0, noise_source='jax'),
                 directory, case)
    except Exception as exc:
        result = dict(status='worker_error', message=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            send_message(child.stdin, {'cmd': 'stop'})
            child.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    if child.returncode != 0:
        result = dict(status='simulator_error', returncode=child.returncode, episode=result)
    c.write(directory/'result.json', result)


def kill_group(proc):
    # Kill descendants even if the group leader already exited.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    proc.wait()


def supervise(command, env, log, output):
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    deadline = time.monotonic()+LIMITS['timeout_seconds']
    try:
        while True:
            disk_guard(output)
            if time.monotonic() >= deadline:
                return dict(status='timeout')
            try:
                rc = proc.wait(timeout=0.5)
                return dict(status='process_exit', returncode=rc)
            except subprocess.TimeoutExpired:
                pass
    except RuntimeError as exc:
        return dict(status='disk_guard', message=str(exc))
    finally:
        kill_group(proc)


def comparison(a, b, ta, tb):
    if a.get('gate') != b.get('gate'):
        return dict(stage='gate', a=a.get('gate'), b=b.get('gate'))
    if a.get('hashes') != b.get('hashes'):
        return dict(stage='initial_observation')
    for i in range(max(len(ta), len(tb))):
        x = ta[i] if i < len(ta) else None
        y = tb[i] if i < len(tb) else None
        if x != y:
            return dict(stage='trajectory', event=i, a=x, b=y)
    for key in ('status', 'success', 'steps'):
        if a.get(key) != b.get(key):
            return dict(stage='outcome', field=key, a=a.get(key), b=b.get(key))
    if a.get('status') != 'completed':
        return dict(stage='not_comparable', reason='No completed rollout pair')
    return None


def summary(m, output, records):
    pairs = []
    for i in range(0, len(m['cases']), 2):
        aa, bb = m['cases'][i:i+2]
        a, b = records.get(aa['id']), records.get(bb['id'])
        if a is None or b is None:
            pairs.append(dict(pair=aa['pair'], status='incomplete'))
            continue
        traces = []
        for case in (aa, bb):
            path = output/case['id']/'trace.jsonl'
            traces.append([json.loads(line) for line in path.read_text().splitlines()] if path.exists() else [])
        difference = comparison(a, b, *traces)
        pairs.append(dict(pair=aa['pair'], outcomes=[a, b], earliest_difference=difference,
                          final_observed_states=[next((e['state'] for e in reversed(t) if e['kind']=='step' and e['state'] is not None), None) for t in traces]))
    return dict(requested_cases=8, executed_cases=len(records), records=records, pairs=pairs,
                limitation='Terminal observation is absent from frozen IPC; final_observed_states is last available nonterminal state. Hashes are not full physics state. Tracing can affect timing: A/A compares the instrumented protocol, not exact reproduction of old runs or efficacy.')


def run(m, manifest, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    disk_guard(output.parent, check_size=False)
    output.mkdir(exist_ok=False)
    c.write(output/'manifest.json', m)
    env = environment()
    records = {}
    for index, case in enumerate(m['cases']):
        directory = output/case['id']
        directory.mkdir()
        command = [c.CONFIG['inference_python'], '-I', '-B', str(Path(__file__).resolve()),
                   'worker', '--manifest', str(manifest.resolve()), '--output', str(output), '--case', str(index)]
        with (directory/'worker.log').open('x') as log:
            process = supervise(command, env, log, output)
        path = directory/'result.json'
        result = json.loads(path.read_text()) if path.exists() else dict(status='worker_error', reason='missing result')
        if process['status'] != 'process_exit' or process.get('returncode') != 0:
            result = dict(status='worker_error', process=process, episode=result)
        records[case['id']] = result
        c.write(output/f'progress_{index+1:02d}.json', summary(m, output, records))
        # Outcome and expert rejection never alter the schedule. Systemic failures abort.
        if result['status'] not in ('completed', 'expert_rejected'):
            break
    c.write(output/'summary.json', summary(m, output, records))
    return 0 if len(records) == 8 and all(r['status'] in ('completed', 'expert_rejected') for r in records.values()) else 1


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        del sys.argv[1]
        return simulator()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'run', 'worker'))
    p.add_argument('--primary-manifest', type=Path, default=PRIMARY)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', type=int)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(args.output, args.primary_manifest)
        return 0
    if args.manifest is None:
        p.error('--manifest required')
    m = json.loads(args.manifest.read_text())
    if args.mode == 'run':
        validate(m, args.manifest)
        return run(m, args.manifest, args.output)
    # Parent verified large shard hashes before launching; worker verifies small code identities.
    if args.case is None or not 0 <= args.case < 8:
        p.error('--case must be 0..7')
    for path, digest in m['code'].items():
        if c.sha(path) != digest:
            raise ValueError('worker source identity changed')
    env = environment()
    os.environ.clear()
    os.environ.update(env)
    worker(m, m['cases'][args.case], args.output/m['cases'][args.case]['id'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
