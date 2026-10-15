#!/usr/bin/env python3
"""Clean diagnostic evaluation with fixed bindings and passive physical traces.

Uses the existing expert gate, task predicates, limits, and 50-action execution.
Results are a controlled diagnostic, not a replacement for the official benchmark.
Every requested seed is recorded, including gate exclusions. No outcome retries.
"""
import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controlled_paired_eval as c
import gen1_raw_paired_eval as binding

ROOT = c.ROOT
SCRIPT = Path(__file__).resolve()
TASKS = ('place_mouse_pad', 'hanging_mug')


def memory_status(root=Path('/sys/fs/cgroup')):
    stats = dict((k, int(v)) for k, v in
                 (line.split() for line in (root/'memory.stat').read_text().splitlines()))
    current = int((root/'memory.current').read_text())
    limit = (root/'memory.max').read_text().strip()
    limit = int(limit) if limit != 'max' else None
    # Only inactive clean file cache is discounted; active cache stays conservative.
    reclaimable = max(0, stats.get('inactive_file', 0) - stats.get('file_dirty', 0)
                      - stats.get('file_writeback', 0) - stats.get('shmem', 0))
    events = dict((k, int(v)) for k, v in
                  (line.split() for line in (root/'memory.events').read_text().splitlines()))
    return dict(current=current, limit=limit, inactive_clean_cache=reclaimable,
                estimated_available=None if limit is None else limit-current+reclaimable,
                events=events)


def release_checkpoint_cache():
    """Advise eviction only for immutable project checkpoint files; delete nothing."""
    total = 0
    for name in ('myungkyu_pi05_jax_robotwin_vanilla_b128_60k', 'gen1_pilot_2000_20261010'):
        for path in (ROOT/'checkpoints'/name).rglob('*'):
            if path.is_file() and not path.is_symlink():
                with path.open('rb') as stream:
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                total += path.stat().st_size
    return total


def physical(task):
    result = {}
    for side in ('left', 'right'):
        result[side] = dict(
            measured_qpos=list(map(float, getattr(task.robot, 'get_'+side+'_arm_real_jointState')()[:6])),
            drive_targets=list(map(float, getattr(task.robot, 'get_'+side+'_arm_jointState')())),
            tcp=list(map(float, getattr(task.robot, 'get_'+side+'_tcp_pose')()[:3])))
    for name in ('mouse', 'target', 'mug', 'rack'):
        actor = getattr(task, name, None)
        if actor is not None:
            pose = actor.get_pose()
            result[name] = dict(position=list(map(float, pose.p)), quaternion=list(map(float, pose.q)))
    return result


def simulator():
    import robotwin_eval_harness as harness
    spec = json.loads(os.environ['GEN1_RAW_TEMPLATE_SPEC'])
    directory = Path(os.environ['ROBOTWIN_CONTROLLED_OUTPUT'])
    repo = Path(c.CONFIG['robotwin_root'])
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo/'scripts')]
    original_import = importlib.import_module
    with contextlib.ExitStack() as stack:
        stream = stack.enter_context((directory/'physical.jsonl').open('x', buffering=1))
        attached = False
        def hooked(name, package=None):
            nonlocal attached
            module = original_import(name, package)
            if name == 'eval_policy_xpolicylab' and not attached:
                attached = True
                stack.enter_context(binding.simulator_shim(module, harness, spec, repo/'description'))
                construct, send = module.class_decorator, harness.send_message
                task = None
                def make(*args, **kwargs):
                    nonlocal task
                    task = construct(*args, **kwargs)
                    return task
                def transmit(channel, payload):
                    kind = payload.get('type')
                    if kind in ('episode_start', 'step'):
                        step = payload.get('info', {}).get('step', 0)
                        stream.write(json.dumps(dict(step=step, physical=physical(task),
                            success=bool(payload.get('info', {}).get('is_success'))))+'\n')
                    return send(channel, payload)
                stack.enter_context(binding.replaced(module, 'class_decorator', make))
                stack.enter_context(binding.replaced(harness, 'send_message', transmit))
            return module
        stack.enter_context(binding.replaced(importlib, 'import_module', hooked))
        return harness.main()


def worker(args, policy_cache=None):
    import numpy as np
    from PIL import Image
    from robotwin_eval_harness import receive_message, send_message
    from robotwin_public_openpi_harness import _observation_for_openpi, _configure_sim_renderer_env
    from pi05_openpi_adapter import PublicPi05Adapter
    spec = binding.template_spec(args.task, args.seed, Path(c.CONFIG['robotwin_root'])/'description')
    env = dict(os.environ, GEN1_RAW_TEMPLATE_SPEC=json.dumps(spec),
               ROBOTWIN_CONTROLLED_OUTPUT=str(args.output))
    _configure_sim_renderer_env(env)
    command = [c.CONFIG['sim_python'], '-I', '-B', str(SCRIPT), 'sim', '--role', 'sim',
               '--controlled-case', '--episodes', '1', '--task', args.task, '--seed', str(args.seed),
               '--episode-steps', '2000', '--robotwin-root', c.CONFIG['robotwin_root'],
               '--task-config', 'demo_clean', '--first-instruction', spec['template'],
               '--first-instruction-seed', str(args.seed)]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
                             cwd=c.CONFIG['robotwin_root'])
    result = dict(status='worker_error')
    try:
        first = receive_message(child.stdout)
        if first['type'] == 'case_status':
            result = {k:v for k,v in first.items() if k != 'type'}
        else:
            bound = first['diagnostic_binding']
            expected = binding.bind_prompt(spec, {'info':bound['fields']}, Path(c.CONFIG['robotwin_root'])/'description')
            binding.require_same_binding(bound, expected)
            prompt = bound['prompt']
            images, state = _observation_for_openpi(first['observation'])
            hashes = c.capture(args.output/'initial', images, state, prompt)
            if policy_cache is not None and policy_cache:
                policy = policy_cache[0]
                import jax
                policy.policy._rng = jax.random.key(0)
                policy._jax_rng = jax.random.key(0)
            else:
                policy = PublicPi05Adapter(args.checkpoint, device='cuda:0', prompt_prefix='none',
                                           normalization='quantile', policy_seed=0, noise_source='jax')
                if policy_cache is not None: policy_cache.append(policy)
            c.guard_policy(policy)
            step = 0
            with (args.output/'actions.jsonl').open('x', buffering=1) as trace:
                while True:
                    # Existing observation only: no extra simulator render/step/check calls.
                    Image.fromarray(images['cam_high']).save(args.output/f'frame_{step:04d}.jpg')
                    chunk = policy.infer_chunk(images=images, state=state, prompt=prompt)
                    trace.write(json.dumps(dict(step=step, state=np.asarray(state).tolist(),
                                                actions=np.asarray(chunk).tolist()))+'\n')
                    done = False
                    for action in chunk:
                        if not np.isfinite(action).all(): raise ValueError('Nonfinite action')
                        send_message(child.stdin, dict(cmd='action', action=np.asarray(action).tolist()))
                        response = receive_message(child.stdout)
                        if response['type'] != 'step': raise RuntimeError(str(response))
                        step += 1
                        if response.get('terminated') or response.get('truncated'):
                            result = dict(status='completed', success=bool(response['info']['is_success']),
                                          steps=step, hashes=hashes, binding=bound)
                            done = True
                            break
                        images, state = _observation_for_openpi(response['observation'])
                    if done: break
    except Exception as exc:
        result = dict(status='worker_error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try: send_message(child.stdin, {'cmd':'stop'}); child.stdin.close()
        except (BrokenPipeError, OSError): pass
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired: child.kill(); child.wait()
        if child.returncode:
            result=dict(status='simulator_error',episode=result)
        result.update(task=args.task, seed=args.seed, checkpoint=args.checkpoint,
                      simulator_returncode=child.returncode)
        c.write(args.output/'result.json', result)


def run(args):
    args.output.mkdir(exist_ok=False)
    records = []
    c.write(args.output/'manifest.json', dict(checkpoint=args.checkpoint, tasks=args.tasks,
            seed_start=args.seed, episodes_per_task=args.episodes, config=c.CONFIG,
            source_sha256=c.sha(SCRIPT), reuse_policy=args.reuse_policy,
            protocol='controlled diagnostic; official expert gate and limits; fixed bound seen instructions; reset JAX keys per episode; no retries'))
    if args.reuse_policy:
        release_checkpoint_cache()
        command=[c.CONFIG['inference_python'],'-I','-u','-B',str(SCRIPT),'panel',
                 '--output',str(args.output),'--checkpoint',args.checkpoint,
                 '--tasks',*args.tasks,'--seed',str(args.seed),'--episodes',str(args.episodes)]
        with (args.output/'panel.log').open('x') as log:
            proc=subprocess.Popen(command,env=binding.runtime_environment(),stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
            baseline=memory_status();started=time.monotonic()
            try:
                while proc.poll() is None:
                    current=memory_status()
                    if any(current['events'][k]>baseline['events'][k] for k in ('oom','oom_kill')):
                        raise RuntimeError('New cgroup OOM event')
                    if current['estimated_available'] is not None and current['estimated_available']<4*2**30:
                        raise RuntimeError('Less than 4 GiB estimated memory available')
                    if time.monotonic()-started>7200:raise TimeoutError('Panel exceeded two hours')
                    time.sleep(1)
            except BaseException:
                binding.kill_group(proc);raise
        if proc.returncode:raise RuntimeError('Panel failed; see panel.log')
        return
    for task in args.tasks:
        for seed in range(args.seed, args.seed+args.episodes):
            directory = args.output/f'{task}_{seed}'
            directory.mkdir()
            release_checkpoint_cache()
            baseline = memory_status()
            command = [c.CONFIG['inference_python'], '-I', '-B', str(SCRIPT), 'worker',
                       '--output', str(directory), '--task', task, '--seed', str(seed),
                       '--checkpoint', args.checkpoint]
            with (directory/'worker.log').open('x') as log:
                proc = subprocess.Popen(command, env=binding.runtime_environment(), stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                started = time.monotonic()
                try:
                    while proc.poll() is None:
                        current = memory_status()
                        if any(current['events'][k] > baseline['events'][k] for k in ('oom','oom_kill')):
                            raise RuntimeError('New cgroup OOM event')
                        if current['estimated_available'] is not None and current['estimated_available'] < 4*2**30:
                            raise RuntimeError('Less than 4 GiB after discounting inactive clean cache')
                        if time.monotonic()-started > 1800: raise TimeoutError('Episode exceeded 1800 seconds')
                        time.sleep(1)
                except BaseException:
                    binding.kill_group(proc)
                    raise
            if proc.returncode: raise RuntimeError(f'Worker failed: {directory}')
            result = json.loads((directory/'result.json').read_text())
            records.append(result)
            c.write(args.output/f'progress_{len(records):02d}.json', dict(records=records))
            print(task, seed, result['status'], result.get('success'), result.get('steps'), flush=True)
    c.write(args.output/'summary.json', dict(records=records,
            completed=sum(r['status']=='completed' for r in records),
            successes=sum(r.get('success',False) for r in records)))


def panel(args):
    cache=[];records=[]
    for task in args.tasks:
        for seed in range(args.seed,args.seed+args.episodes):
            directory=args.output/f'{task}_{seed}'
            directory.mkdir()
            case=argparse.Namespace(task=task,seed=seed,checkpoint=args.checkpoint,output=directory)
            worker(case,cache)
            result=json.loads((directory/'result.json').read_text())
            records.append(result)
            c.write(args.output/f'progress_{len(records):02d}.json',dict(records=records))
            print('RESULT',task,seed,result['status'],result.get('success'),result.get('steps'),flush=True)
    c.write(args.output/'summary.json',dict(records=records,
        completed=sum(r['status']=='completed' for r in records),
        successes=sum(r.get('success',False) for r in records)))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='sim':
        del sys.argv[1]
        return simulator()
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['run','worker','panel'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    p.add_argument('--task', choices=TASKS)
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--reuse-policy', action='store_true')
    args=p.parse_args()
    if args.episodes<=0 or args.episodes>10:p.error('Use between one and ten requested trials per task')
    if args.mode in ('worker','panel'):
        os.environ.update(binding.runtime_environment())
        sys.path[:0]=[str(binding.OPENPI/'src'), str(binding.OPENPI/'packages/openpi-client/src')]
        (worker if args.mode=='worker' else panel)(args)
    else: run(args)


if __name__=='__main__': main()
