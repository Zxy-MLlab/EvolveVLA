#!/usr/bin/env python3
"""External, bounded task-isolated expert collector. Never modifies pinned sources.
Run --check first. --coordinator-approved plus a passed reload audit is mandatory.
Two separate invocations may run different tasks; no shared mutable manifest.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path('/root/autodl-tmp/EvolveVLA')
REPO = ROOT / 'robotwin_repo_ba0fd21'
WRAPPER = ROOT / 'scripts/collection_wrapper_gen1_demo_clean_smoke_20261009.py'
WRAPPER_HASH = '181f25a6cd5e5c433dba13f4c1f5daa50a9563ed4efb4d3abf13166ac975f676'
ACCEPTED = ROOT / 'data/gen1_demo_clean'
BATCH = ROOT / 'data/gen1_task_batches_20261010'
PYTHON = Path('/root/autodl-tmp/conda_envs/robotwin_cu128/bin/python')
AUDIT = ROOT / 'checkpoints/gen1_myungkyu_smoke_cap8_20261009T161534/verification_audit_20261009T162251Z_c66175d3cf11.json'
# 60 GiB untouched reserve + 50 GiB next checkpoint + 2 GiB polling margin.
STOP_GIB = 112
START_GIB = 114
TARGET_LIMIT = 50
ATTEMPT_LIMIT = 1000
SEED_RANGES = {}
ENTRYPOINT = Path(__file__).resolve()
PRUNE_FAILED_RAW = False
spec = importlib.util.spec_from_file_location('vetted_smoke', WRAPPER)
vetted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vetted)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def dump(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def free_gib():
    return shutil.disk_usage(ROOT).free / 2**30


def historical_seeds():
    used = set()
    sources = []
    def extract(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ('seed', 'episode_seed', 'start_seed', 'seed_start') and isinstance(v, int):
                    used.add(v)
                    # Conservatively exclude a whole bounded historical seed-start region.
                    if k in ('start_seed', 'seed_start'):
                        used.update(range(v, min(100000, v + 1000)))
                elif k in ('seeds', 'seed_list', 'known_seeds') and isinstance(v, list):
                    used.update(x for x in v if isinstance(x, int))
                else:
                    extract(v)
        elif isinstance(obj, list):
            for value in obj:
                extract(value)
    for base in [ROOT/'data', ROOT/'results', ROOT/'outputs', ROOT/'robotwin_repo/data',
                 REPO/'data', ROOT/'robotwin_repo_e154702/data']:
        if not base.exists():
            continue
        for parent, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ('.cache', '__pycache__', '.git')]
            for name in files:
                p = Path(parent)/name
                match = re.search(r'seed[_-](\d+)', name)
                if match:
                    used.add(int(match.group(1)))
                if name == 'seed.txt':
                    vals = [int(x) for x in p.read_text().split()]
                    used.update(vals)
                    sources.append({'path': str(p), 'sha256': vetted.digest(p), 'seeds': vals})
                elif p.suffix == '.json':
                    try:
                        extract(json.loads(p.read_text()))
                    except (json.JSONDecodeError, UnicodeError):
                        # Active atomic manifests never partially parse; historical unrelated
                        # malformed JSON is not accepted trajectory input.
                        continue
    # Include attempted smoke seeds even when old failed staging was removed.
    for p in (ROOT/'logs').glob('gen1_demo_clean*.log'):
        used.update(int(x) for x in re.findall(r'\bseed[= :]+(\d+)', p.read_text(errors='replace')))
    return used, sources


def preflight(a):
    require(vetted.digest(WRAPPER) == WRAPPER_HASH, 'Vetted wrapper changed')
    require(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == vetted.REV,
            'Pinned revision changed')
    require(not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'], text=True),
            'Pinned tracked sources modified')
    audit = json.loads(AUDIT.read_text())
    require(audit.get('status') == 'passed', 'Reload verification has not passed')
    require(a.coordinator_approved or a.check, 'Coordinator approval is required to launch')
    tasks_meta = json.loads((ACCEPTED/'tasks44.json').read_text())
    require(vetted.digest(Path(tasks_meta['baseline_path'])) == tasks_meta['baseline_sha256'], 'Baseline selection changed')
    tasks = [x['task'] for x in tasks_meta['tasks']]
    require(len(tasks) == 44 and a.task in tasks, 'Task is not in approved weak-task 44')
    require((REPO/'envs'/f'{a.task}.py').is_file(), 'Task source missing')
    seed_lo, seed_hi = SEED_RANGES.get(a.task, (30000 + tasks.index(a.task) * 1000,
                                             31000 + tasks.index(a.task) * 1000))
    old = json.loads((ACCEPTED/'manifest.json').read_text())
    credited = []
    for row in old['episodes']:
        require(row['success'] and row['plan_success'] and row['replay_success'] and row['source_revision'] == vetted.REV,
                'Invalid accepted-source predicates')
        require(row['config'] == 'demo_clean', 'Invalid accepted config')
        if row['task'] == a.task:
            h5 = next((Path(row['episode_directory'])/'data').glob('*.hdf5'))
            require(vetted.digest(h5) == row['hdf5_sha256'], 'Accepted source hash changed')
            credited.append(row)
    require(0 < a.target <= TARGET_LIMIT and a.target >= len(credited), 'Target outside configured bounds')
    require(0 < a.max_attempts <= ATTEMPT_LIMIT and a.max_new > 0 and a.timeout > 0, 'Invalid bounds')
    require(free_gib() >= START_GIB, f'Need >= {START_GIB} GiB free before next attempt')
    return tasks_meta, (seed_lo, seed_hi), credited


def simulator_env(out):
    env = os.environ.copy()
    lib = ROOT/'third_party/oidn-2.5.1/lib'
    env.update(CUDA_VISIBLE_DEVICES='0', VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',
               ROBOTWIN_RENDER_BACKEND='rt', ROBOTWIN_RT_DENOISER='oidn', PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', LD_PRELOAD=str(lib/'libOpenImageDenoise.so.2'),
               LD_LIBRARY_PATH=str(lib)+':'+env.get('LD_LIBRARY_PATH', ''),
               TMPDIR=str(out/'runtime/tmp'), XDG_CACHE_HOME=str(out/'runtime/cache'),
               CUDA_CACHE_PATH=str(out/'runtime/cuda'), TORCH_EXTENSIONS_DIR=str(out/'runtime/torch'))
    # The parent's CPU-only verifier settings must not leak into simulation.
    env.pop('JAX_PLATFORMS', None)
    return env


def episode(a):
    require(vetted.digest(WRAPPER) == WRAPPER_HASH, 'Vetted wrapper changed')
    require(json.loads(AUDIT.read_text()).get('status') == 'passed' and a.coordinator_approved, 'Gate closed')
    require(a.task in [x['task'] for x in json.loads((ACCEPTED/'tasks44.json').read_text())['tasks']], 'Unknown task')
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import scripts.collect_data as official
    from envs.utils import pkl2hdf5
    stage = Path(a.stage).resolve()
    require(stage.is_relative_to(BATCH/a.task/'.staging'), 'Stage outside isolated root')
    seed_lo, seed_hi = SEED_RANGES.get(a.task, (30000, 74000))
    require(seed_lo <= a.seed < seed_hi, 'Seed outside allocated training region')
    def run(env, cfg):
        cfg.update(save_path=str(stage), need_plan=True, save_data=False, render_freq=0)
        dump(stage/'resolved_config.json', cfg)
        env.setup_demo(now_ep_num=a.index, seed=a.seed, **cfg)
        env.play_once()
        plan = bool(env.plan_success and env.check_success())
        dump(stage/'predicates.json', {'plan_success': plan})
        require(plan, 'Expert planning predicate failed')
        env.save_traj_data(a.index)
        env.close_env()
        cfg.update(need_plan=False, save_data=True)
        env.setup_demo(now_ep_num=a.index, seed=a.seed, **cfg)
        cfg.update(env.load_tran_data(a.index))
        env.set_path_lst(cfg)
        info = env.play_once()
        replay = bool(env.plan_success and env.check_success())
        dump(stage/'predicates.json', {'plan_success': plan, 'replay_success': replay,
                                      'checked_before_close_and_merge': True})
        require(replay, 'Expert replay predicate failed')
        descriptions = official.build_episode_instructions(cfg, info, a.index)
        dump(stage/'scene_info.json', info)
        raw = stage/'.cache'/f'episode{a.index}'
        paths = sorted(raw.glob('*.pkl'), key=lambda p: int(p.stem))
        data = pkl2hdf5.parse_dict_structure(pkl2hdf5.load_pkl_file(paths[0]))
        for p in paths:
            pkl2hdf5.append_data_to_structure(data, pkl2hdf5.load_pkl_file(p))
        (stage/'data').mkdir()
        h5 = stage/'data'/f'episode_{a.index:07d}.hdf5'
        pkl2hdf5.create_xpolicylab_hdf5(data, str(h5), descriptions['seen'], 15)
        report = vetted.validate(h5, raw, descriptions['seen'])
        env.close_env()
        report.update(task=a.task, seed=a.seed, success=True, plan_success=plan, replay_success=replay,
                      source_revision=vetted.REV, config='demo_clean', timestamp=time.time(),
                      config_sha256=vetted.digest(stage/'resolved_config.json'),
                      instruction_sha256=vetted.digest(next((stage/'instruction').glob('*.json'))),
                      wrapper_sha256=WRAPPER_HASH, batch_script_sha256=vetted.digest(ENTRYPOINT))
        dump(stage/'validation.json', report)
    official.run = run
    official.main(task_name=a.task, task_config='demo_clean')


def collect(a):
    meta, (seed_lo, seed_hi), credited = preflight(a)
    used, sources = historical_seeds()
    out = BATCH/a.task
    summary = {'task': a.task, 'worker_root': str(out), 'seed_range': [seed_lo, seed_hi-1],
               'available_seeds': sum(s not in used for s in range(seed_lo, seed_hi)),
               'credited_existing': len(credited), 'target_total': a.target, 'free_gib': free_gib(),
               'stop_floor_gib': STOP_GIB, 'max_new_this_launch': a.max_new,
               'max_attempts_this_launch': a.max_attempts, 'reload_audit': str(AUDIT)}
    if a.check:
        print(json.dumps(summary, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # At most two new workers, including independently launched task invocations.
    slots = []
    slot = None
    for index in range(2):
        handle = (BATCH/f'gpu_slot_{index}.lock').open('a')
        slots.append(handle)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            slot = handle
            break
        except BlockingIOError:
            pass
    require(slot is not None, 'Two collection workers already running')
    for name in ('.staging', 'episodes', 'failed', 'runtime/tmp', 'runtime/cache', 'runtime/cuda', 'runtime/torch'):
        (out/name).mkdir(parents=True, exist_ok=True)
    claims = BATCH/'seed_claims'
    claims.mkdir(exist_ok=True)
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        require(manifest['task'] == a.task and manifest['wrapper_sha256'] == WRAPPER_HASH, 'Manifest identity mismatch')
        require(manifest['credited_manifest_sha256'] == vetted.digest(ACCEPTED/'manifest.json'), 'Credited source changed')
        require(not list((out/'.staging').iterdir()), 'Unfinished stages retained: review before resuming')
        known_dirs = {x['episode_directory'] for x in manifest['episodes']}
        require({str(p) for p in (out/'episodes').iterdir()} == known_dirs, 'Orphan accepted episode: review before resuming')
        for row in manifest['episodes']:
            require(vetted.digest(next((Path(row['episode_directory'])/'data').glob('*.hdf5'))) == row['hdf5_sha256'],
                    'Previously accepted output changed')
    else:
        require(not list((out/'episodes').iterdir()) and not list((out/'.staging').iterdir()), 'Existing unmanifested data')
        manifest = dict(task=a.task, episodes=[], attempts=[], credited_episodes=credited,
                        credited_manifest=str(ACCEPTED/'manifest.json'),
                        credited_manifest_sha256=vetted.digest(ACCEPTED/'manifest.json'),
                        source_revision=vetted.REV, wrapper_sha256=WRAPPER_HASH,
                        batch_script_sha256=vetted.digest(ENTRYPOINT), baseline_sha256=meta['baseline_sha256'],
                        reload_audit=str(AUDIT), reload_audit_sha256=vetted.digest(AUDIT),
                        config='demo_clean', started_at=time.time())
        dump(out/'historical_seed_inventory.json', {'known_seeds': sorted(used), 'sources': sources})
        dump(out/'tasks44.json', meta)
    manifest.update(target_total=a.target, status='running', last_launch=summary)
    env = simulator_env(out)
    dump(out/'simulator_environment.json', {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'VK_ICD_FILENAMES',
         'ROBOTWIN_RENDER_BACKEND', 'ROBOTWIN_RT_DENOISER', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'TMPDIR',
         'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'TORCH_EXTENSIONS_DIR', 'PYTHONDONTWRITEBYTECODE')})
    dump(manifest_path, manifest)
    new = attempts = 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for seed in range(seed_lo, seed_hi):
            if len(credited)+len(manifest['episodes']) >= a.target or new >= a.max_new or attempts >= a.max_attempts:
                break
            if seed in used or (claims/f'seed_{seed}.json').exists():
                continue
            require(free_gib() >= START_GIB, 'Disk start floor reached; preserving all data')
            try:
                with (claims/f'seed_{seed}.json').open('x') as f:
                    json.dump({'task': a.task, 'seed': seed, 'worker_root': str(out), 'claimed_at': time.time()}, f)
                    f.flush()
                    os.fsync(f.fileno())
            except FileExistsError:
                continue
            attempts += 1
            idx = len(credited)+len(manifest['episodes'])
            stage = out/'.staging'/f'seed_{seed}'
            stage.mkdir()
            log = out/f'seed_{seed}.log'
            cmd = [str(PYTHON), '-B', str(ENTRYPOINT), '--episode', '--coordinator-approved',
                   '--task', a.task, '--stage', str(stage), '--index', str(idx), '--seed', str(seed)]
            record = dict(seed=seed, task=a.task, index=idx, log=str(log), stage=str(stage),
                          command=cmd, status='running', started_at=time.time())
            manifest['attempts'].append(record)
            dump(manifest_path, manifest)  # seed recorded before child launch
            with log.open('x') as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                try:
                    record['pid'] = p.pid
                    dump(manifest_path, manifest)
                    print(f'task={a.task} seed={seed} child_pid={p.pid} log={log}', flush=True)
                    while p.poll() is None:
                        if free_gib() < STOP_GIB:
                            raise RuntimeError('Disk runtime floor reached')
                        if time.time()-record['started_at'] > a.timeout:
                            raise TimeoutError('Episode timeout')
                        time.sleep(0.5)
                    record['returncode'] = p.returncode
                except BaseException as exc:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait()
                    record.update(returncode=p.returncode, error=str(exc), status='interrupted')
                    os.rename(stage, out/'failed'/stage.name)
                    record['stage'] = str(out/'failed'/stage.name)
                    dump(manifest_path, manifest)
                    if not isinstance(exc, TimeoutError):
                        raise
            record['elapsed_seconds'] = time.time()-record['started_at']
            if record['returncode'] == 0:
                report = json.loads((stage/'validation.json').read_text())
                require(report['task'] == a.task and report['seed'] == seed and report['success']
                        and report['plan_success'] and report['replay_success'], 'Invalid validation predicates')
                h5 = next((stage/'data').glob('*.hdf5'))
                require(vetted.digest(h5) == report['hdf5_sha256'], 'HDF5 hash mismatch')
                # Only THIS attempt's transient raw, fully checked by the vetted validator.
                # Keep HDF5, traj, config, instructions, hashes, logs and all failed stages.
                shutil.rmtree(stage/'.cache')
                final = out/'episodes'/f'episode_{idx:07d}'
                require(not final.exists(), 'Refusing accepted episode overwrite')
                os.rename(stage, final)
                report['episode_directory'] = str(final)
                manifest['episodes'].append(report)
                record.update(status='accepted', stage=str(final))
                new += 1
            else:
                if stage.exists():
                    os.rename(stage, out/'failed'/stage.name)
                record.update(status='rejected', stage=str(out/'failed'/stage.name))
            # Optional expansion policy: retain failed diagnostics, not bulky raw
            # frames from newly generated, rejected attempts. Never accepted data.
            failed_raw = out/'failed'/stage.name/'.cache'
            if PRUNE_FAILED_RAW and failed_raw.exists():
                require(failed_raw.resolve().is_relative_to((out/'failed').resolve()),
                        'Failed cache outside this task')
                shutil.rmtree(failed_raw)
                record['failed_raw_cache_removed'] = True
            dump(manifest_path, manifest)
        total = len(credited)+len(manifest['episodes'])
        manifest.update(status='complete' if total >= a.target else 'bounded_pause', accepted_total=total)
        dump(manifest_path, manifest)
        print(json.dumps({'status': manifest['status'], 'accepted_total': total, 'new_this_launch': new,
                          'manifest': str(manifest_path), 'free_gib': free_gib()}), flush=True)
        if total < a.target and new < a.max_new:
            raise RuntimeError('Bounded attempt/seed budget exhausted before requested new successes')
    except BaseException as exc:
        manifest.update(status='stopped', error=str(exc))
        dump(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--check', action='store_true', help='CPU-only, no output writes, no simulation')
    parser.add_argument('--coordinator-approved', action='store_true')
    parser.add_argument('--target', type=int, default=50, help='Total per task including accepted smoke')
    parser.add_argument('--max-new', type=int, default=2, help='Bounded pilot success count per invocation')
    parser.add_argument('--max-attempts', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--episode', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--stage', help=argparse.SUPPRESS)
    parser.add_argument('--index', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    a = parser.parse_args()
    episode(a) if a.episode else collect(a)


if __name__ == '__main__':
    main()
