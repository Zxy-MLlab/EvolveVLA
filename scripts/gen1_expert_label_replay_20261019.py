#!/usr/bin/env python3
"""Replay recorded expert labels through the unchanged policy evaluation controller.

prepare is CPU-only. run launches the existing GPU renderer/planner, with no
learned policy loaded. Source states are DRIVE TARGETS, not measured joint qpos.
All hooks are process-local and leave the running evaluation and source files
untouched. A source-verified gate bypass, if requested, is diagnostic only.
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

ROOT = Path('/root/autodl-tmp/EvolveVLA')
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
NORM = ROOT / 'checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k/assets/aloha_agilex_sim/norm_stats.json'
FIELDS = ('left_arm_joint_states', 'left_ee_joint_states',
          'right_arm_joint_states', 'right_ee_joint_states')
CAMERAS = {'cam_high': ('cam_head', 'head_camera'),
           'cam_left_wrist': ('cam_left_wrist', 'left_camera'),
           'cam_right_wrist': ('cam_right_wrist', 'right_camera')}
JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def select_cases(manifest, limit_per_task=10, smoke_only=False):
    counts, selected = {}, []
    for case in manifest['cases']:
        if smoke_only and not case.get('smoke'):
            continue
        task = case['task']
        if counts.get(task, 0) >= limit_per_task:
            continue
        if task not in ('hanging_mug', 'place_mouse_pad'):
            raise ValueError('Unsupported task')
        if Path(case['id']).name != case['id']:
            raise ValueError('Unsafe case id')
        counts[task] = counts.get(task, 0) + 1
        selected.append(case)
    if not selected or len({c['id'] for c in selected}) != len(selected):
        raise ValueError('Empty or duplicate selection')
    return selected


def source_arrays(case):
    import h5py
    import numpy as np
    path = Path(case['source_hdf5'])
    if digest(path) != case.get('source_hdf5_sha256', case.get('source_sha256')):
        raise ValueError('Source HDF5 hash changed')
    with h5py.File(path) as stream:
        arrays = [np.concatenate([stream[f'{group}/{key}'][:] for key in FIELDS], axis=1)
                  .astype(np.float32) for group in ('state', 'action')]
        frequency = int(stream['additional_info/frequency'][()])
        prompts = json.loads(stream['instructions'][()])
    state, action = arrays
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
        raise ValueError('Wrong source shape')
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError('Nonfinite source')
    np.testing.assert_array_equal(state[1:], action[:-1])
    if frequency != 15 or case['prompt'] not in prompts:
        raise ValueError('Source fps or prompt mismatch')
    return state, action


def roundtrip_actions(state, action, norm):
    """Mirror native delta/quantile/pad/unpad/unnormalize/absolute transforms.

    Use the SOURCE anchor every 50 frames; this validates arithmetic only, not
    an on-policy feedback controller. The production transforms are unchanged.
    """
    import numpy as np
    q01 = np.asarray(norm['actions']['q01'], dtype=np.float32)[:14]
    q99 = np.asarray(norm['actions']['q99'], dtype=np.float32)[:14]
    restored = np.empty_like(action)
    for start in range(0, len(action), 50):
        stop = min(start + 50, len(action))
        target = action[np.minimum(np.arange(start, start + 50), len(action)-1)].copy()
        target[:, JOINTS] -= state[start, JOINTS]
        normalized = (target-q01)/(q99-q01+1e-6)*2-1
        padded = np.pad(normalized, ((0, 0), (0, 18)))
        back = (padded[:, :14]+1)/2*(q99-q01+1e-6)+q01
        back[:, JOINTS] += state[start, JOINTS]
        restored[start:stop] = back[:stop-start]
    return restored


def observation_arrays(observation):
    import numpy as np
    images = {}
    for camera, (_, bridge_key) in CAMERAS.items():
        value = np.asarray(observation['pixels'][bridge_key])
        if value.shape == (3, 240, 320):
            value = value.transpose(1, 2, 0)
        if value.dtype != np.uint8:
            if value.max() <= 1.0:
                value = value * 255
            value = np.clip(value, 0, 255).astype(np.uint8)
        if value.shape != (240, 320, 3):
            raise ValueError(f'Unexpected image shape: {camera}: {value.shape}')
        images[camera] = value
    state = np.asarray(observation['agent_pos'], dtype=np.float32).reshape(14)
    return images, state


def source_image_metrics(stream, frame, images):
    import cv2
    import numpy as np
    report = {}
    for camera, (source_camera, _) in CAMERAS.items():
        encoded = bytes(stream[f'vision/{source_camera}/colors'][frame])
        expected = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        actual = images[camera]
        ok, reencoded = cv2.imencode('.jpg', actual)
        report[camera] = dict(mae=float(np.abs(actual.astype(float)-expected).mean()),
                             jpeg_bytes_equal=bool(ok and reencoded.tobytes() == encoded))
    return report


def initial_matches(metrics, state_max_abs, tolerance=8.0):
    return state_max_abs <= 1e-5 and all(m['mae'] <= tolerance for m in metrics.values())


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
    """Enrich existing harness messages; never replace take_action internals."""
    import robotwin_eval_harness as harness
    import gen1_raw_paired_eval as binding
    allow_bypass = os.environ.get('GEN1_ORACLE_GATE_BYPASS') == '1'
    original_import = importlib.import_module
    task, attached = None, False
    gate = dict(status='not_run', bypass_requested=allow_bypass, bypassed=False)
    timing = dict(physics_steps=0, total_physics_steps=0, timestep=None)
    with contextlib.ExitStack() as stack:
        def hooked(name, package=None):
            nonlocal task, attached
            module = original_import(name, package)
            if name != 'eval_policy_xpolicylab' or attached:
                return module
            attached = True
            construct = module.class_decorator
            transmit = harness.send_message
            def make(*args, **kwargs):
                nonlocal task
                task = construct(*args, **kwargs)
                original_play, original_check, original_take = task.play_once, task.check_success, task.take_action
                pending_gate = False
                def play(*a, **kw):
                    nonlocal pending_gate
                    result = original_play(*a, **kw)
                    gate['plan_success'] = bool(task.plan_success)
                    gate['check_success'] = None
                    pending_gate = True
                    if allow_bypass and not task.plan_success:
                        task.plan_success = True
                        gate.update(status='rejected', bypassed=True)
                    return result
                def check(*a, **kw):
                    nonlocal pending_gate
                    value = original_check(*a, **kw)
                    if pending_gate:
                        pending_gate = False
                        gate['check_success'] = bool(value)
                        accepted = gate['plan_success'] and bool(value)
                        gate['status'] = 'passed' if accepted else 'rejected'
                        if allow_bypass and not accepted:
                            gate['bypassed'] = True
                            return True
                    return value
                def take(*a, **kw):
                    previous = sys.getprofile()
                    count = 0
                    scene = task.scene
                    def profile(frame, event, value):
                        nonlocal count
                        if event == 'c_call' and getattr(value, '__name__', '') == 'step' and getattr(value, '__self__', None) is scene:
                            count += 1
                    sys.setprofile(profile)
                    try:
                        return original_take(*a, **kw)
                    finally:
                        sys.setprofile(previous)
                        timing['physics_steps'] = count
                        timing['total_physics_steps'] += count
                        timing['timestep'] = float(scene.get_timestep())
                stack.enter_context(binding.replaced(task, 'play_once', play))
                stack.enter_context(binding.replaced(task, 'check_success', check))
                stack.enter_context(binding.replaced(task, 'take_action', take))
                return task
            def send(channel, payload):
                if payload.get('type') == 'case_status' and payload.get('status') == 'expert_rejected':
                    gate.update(status='rejected', plan_success=payload.get('plan_success'),
                                check_success=payload.get('check_success'))
                extra = dict(oracle_gate=dict(gate))
                if payload.get('type') in ('episode_start', 'step'):
                    extra.update(oracle_physical=physical(task), oracle_timing=dict(timing))
                return transmit(channel, dict(payload, **extra))
            stack.enter_context(binding.replaced(module, 'class_decorator', make))
            stack.enter_context(binding.replaced(harness, 'send_message', send))
            return module
        stack.enter_context(binding.replaced(importlib, 'import_module', hooked))
        return harness.main()


def run_case(case, directory, args):
    import h5py
    import numpy as np
    import controlled_paired_eval as c
    import gen1_raw_paired_eval as binding
    from robotwin_eval_harness import receive_message, send_message
    from robotwin_public_openpi_harness import _configure_sim_renderer_env
    state, action = source_arrays(case)
    env = binding.runtime_environment()
    env.update(ROBOTWIN_CONTROLLED_OUTPUT=str(directory),
               GEN1_ORACLE_GATE_BYPASS='1' if args.allow_gate_bypass else '0')
    _configure_sim_renderer_env(env)
    command = [c.CONFIG['sim_python'], '-I', '-u', '-B', str(SCRIPT), 'sim',
               '--role', 'sim', '--controlled-case', '--episodes', '1', '--task', case['task'],
               '--seed', str(case['seed']), '--episode-index', str(case['episode_index']),
               '--episode-steps', '2000', '--robotwin-root', c.CONFIG['robotwin_root'],
               '--task-config', 'demo_clean', '--first-instruction', case['prompt'],
               '--first-instruction-seed', str(case['seed'])]
    write(directory/'command.json', dict(command=command, action_source='raw HDF5 next drive target'))
    result = dict(id=case['id'], task=case['task'], seed=case['seed'],
                  episode_index=case['episode_index'], source_frames=len(state),
                  status='worker_error', success=False, steps=0)
    started = time.monotonic()
    with (directory/'simulator.log').open('x') as log:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=log, env=env, cwd=c.CONFIG['robotwin_root'], start_new_session=True)
        def timeout(signum, frame):
            raise TimeoutError(f'Case exceeded {args.timeout} seconds')
        prior_alarm = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.timeout)
        try:
            first = receive_message(child.stdout)
            result['expert_gate'] = first.get('oracle_gate')
            if first['type'] == 'case_status':
                result.update({k: v for k, v in first.items() if k not in ('type', 'oracle_gate')})
                return result
            images, reset_state = observation_arrays(first['observation'])
            with h5py.File(case['source_hdf5']) as source, (directory/'trace.jsonl').open('x', buffering=1) as trace:
                metrics = source_image_metrics(source, 0, images)
                initial_error = float(np.abs(reset_state-state[0]).max())
                matched = initial_matches(metrics, initial_error, args.initial_mae_tolerance)
                initial = dict(state_max_abs=initial_error, images=metrics,
                               observed_source_match=matched, physical=first.get('oracle_physical'),
                               limitation='JPEG/observed-state agreement does not prove all hidden physics state equal')
                write(directory/'initial_parity.json', initial)
                np.savez_compressed(directory/'initial_arrays.npz', source_state=state[0], reset_state=reset_state,
                                    **{k: v for k, v in images.items()})
                result['initial_parity'] = initial
                if not matched:
                    result['status'] = 'initial_source_mismatch'
                    return result
                result['status'] = 'source_exhausted'
                max_drive_error, max_real_error = 0., 0.
                for index, target in enumerate(action):
                    send_message(child.stdin, dict(cmd='action', action=target.tolist()))
                    response = receive_message(child.stdout)
                    if response['type'] != 'step':
                        raise RuntimeError(f'Unexpected response: {response.get("type")}')
                    measured = response['oracle_physical']
                    drive = np.asarray(measured['left']['drive_targets'] + measured['right']['drive_targets'])
                    real = np.asarray(measured['left']['measured_qpos'] + measured['right']['measured_qpos'])
                    drive_error = float(np.abs(drive-target).max())
                    real_error = float(np.abs(real-target[list(JOINTS)]).max())
                    max_drive_error = max(max_drive_error, drive_error)
                    max_real_error = max(max_real_error, real_error)
                    row = dict(source_action_index=index, executed_step=index+1,
                               source_target=target.tolist(), physical=measured,
                               drive_target_max_abs_error=drive_error,
                               measured_joint_max_abs_error=real_error,
                               timing=response['oracle_timing'], success=bool(response['info']['is_success']))
                    if response.get('observation') is not None and index+1 < len(state):
                        after_images, _ = observation_arrays(response['observation'])
                        row['source_next_frame_images'] = source_image_metrics(source, index+1, after_images)
                    trace.write(json.dumps(row, allow_nan=False)+'\n')
                    result.update(steps=index+1, success=row['success'],
                                  max_drive_target_error=max_drive_error, max_measured_joint_error=max_real_error,
                                  final_physical=measured, final_timing=response['oracle_timing'])
                    if response.get('terminated') or response.get('truncated'):
                        result['status'] = 'completed' if row['success'] else 'eval_limit_reached'
                        break
        except Exception as error:
            result.update(status='worker_error', error=f'{type(error).__name__}: {error}')
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prior_alarm)
            try:
                send_message(child.stdin, dict(cmd='stop'))
                child.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            result.update(simulator_returncode=child.returncode, elapsed_seconds=time.monotonic()-started)
            write(directory/'result.json', result)
    return result


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        del sys.argv[1]
        return simulator()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-per-task', type=int, default=10)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--allow-gate-bypass', action='store_true')
    parser.add_argument('--initial-mae-tolerance', type=float, default=8.)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if args.limit_per_task < 1 or args.timeout < 1:
        parser.error('Positive limits required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = select_cases(json.loads(args.manifest.read_text()), args.limit_per_task, args.smoke_only)
    norm = json.loads(NORM.read_text())['norm_stats']
    preflight = []
    import numpy as np
    for case in cases:
        state, action = source_arrays(case)
        maximum = float(np.abs(roundtrip_actions(state, action, norm)-action).max())
        if maximum > 1e-5:
            raise ValueError(f'Roundtrip mismatch: {case["id"]}: {maximum}')
        preflight.append(dict(id=case['id'], frames=len(state),
                              next_state_action_exact=True, roundtrip_max_abs=maximum))
    write(args.output/'manifest.json', dict(protocol=__doc__, source_manifest=str(args.manifest.resolve()),
          source_manifest_sha256=digest(args.manifest), runner_sha256=digest(SCRIPT),
          norm_stats=str(NORM), norm_sha256=digest(NORM), cases=cases, preflight=preflight,
          mode=args.mode, allow_gate_bypass=args.allow_gate_bypass,
          initial_mae_tolerance=args.initial_mae_tolerance))
    print('CPU_PREFLIGHT_PASSED', json.dumps(preflight), flush=True)
    if args.mode == 'prepare':
        return
    records = []
    for case in cases:
        directory = args.output/case['id']
        directory.mkdir()
        record = run_case(case, directory, args)
        records.append(record)
        write(args.output/'progress.json', dict(attempted=len(records), requested=len(cases), records=records))
        print('ORACLE_RESULT', json.dumps({k: record.get(k) for k in ('id', 'status', 'success', 'steps', 'elapsed_seconds')}), flush=True)
    write(args.output/'summary.json', dict(records=records, successes=sum(r['success'] for r in records),
          source_matched=sum(r.get('initial_parity', {}).get('observed_source_match', False) for r in records),
          requested=len(cases), gate_bypassed=sum(bool((r.get('expert_gate') or {}).get('bypassed')) for r in records)))


if __name__ == '__main__':
    main()
