#!/usr/bin/env python3
"""Positive control: replay the stored dense expert path on the evaluation reset.

No new planning, source mutation, learned policy, or active-run modification.
Uses the same task configuration and renderer as the sparse label diagnostic;
only the controller changes to the original saved positions and velocities.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import gen1_expert_label_replay_20261019 as replay


class SceneStepCounter:
    """Count the simulator's actual scene.step call sites without replacing it."""
    def __init__(self, source):
        self.source = str(Path(source).resolve())
        self.lines = {i for i, line in enumerate(Path(source).read_text().splitlines(), 1)
                      if line.strip() == 'self.scene.step()'}
        if not self.lines:
            raise ValueError('No explicit physics step call sites found')
        self.count = 0

    def trace(self, frame, event, value):
        if frame.f_code.co_filename != self.source:
            return None
        if event == 'line' and frame.f_lineno in self.lines:
            self.count += 1
        return self.trace


def saved_trajectory(case):
    path = Path(case['source_episode_directory'])/'_traj_data'/f'episode{case["episode_index"]}.pkl'
    if not path.is_file():
        raise ValueError(f'Missing stored planner path: {path}')
    return path


def simulate(case_path, directory, initial_tolerance):
    import h5py
    import numpy as np
    import controlled_paired_eval as controlled
    import robotwin_eval_harness as harness
    case = json.loads(case_path.read_text())
    state, action = replay.source_arrays(case)
    source_path = saved_trajectory(case)
    with source_path.open('rb') as stream:
        trajectory = pickle.load(stream)
    if set(trajectory) != {'left_joint_path', 'right_joint_path'}:
        raise ValueError('Unexpected stored path schema')
    repo = Path(controlled.CONFIG['robotwin_root'])
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo/'scripts')]
    module = importlib.import_module('eval_policy_xpolicylab')
    task_args, _ = module.load_task_args(dict(task_name=case['task'], task_config='demo_clean',
                                            policy_name='robotwin_eval_harness', ckpt_setting='official_pi05'))
    task_args.update(eval_mode=True, render_freq=0, policy_name='robotwin_eval_harness',
                     need_plan=False, save_data=False, save_path=str(directory))
    task_args.pop('eval_video_save_dir', None)
    task = module.class_decorator(case['task'])
    result = dict(id=case['id'], task=case['task'], seed=case['seed'],
                  episode_index=case['episode_index'], controller='stored_dense_expert',
                  source_trajectory=str(source_path), source_trajectory_sha256=replay.digest(source_path),
                  source_frames=len(state), status='worker_error', success=False)
    started = time.monotonic()
    count, maximum, measured_maximum = 0, 0., 0.
    counter = SceneStepCounter(repo/'envs/_base_task.py')
    try:
        task.setup_demo(now_ep_num=case['episode_index'], seed=case['seed'], is_test=True, **task_args)
        task.set_instruction(instruction=case['prompt'])
        images, reset_state = replay.observation_arrays(harness.simulator_observation(task.get_obs()))
        with h5py.File(case['source_hdf5']) as source:
            metrics = replay.source_image_metrics(source, 0, images)
        initial_error = float(np.abs(reset_state-state[0]).max())
        matched = replay.initial_matches(metrics, initial_error, initial_tolerance)
        initial = dict(state_max_abs=initial_error, images=metrics, observed_source_match=matched,
                       physical=replay.physical(task),
                       limitation='Observed initial parity does not prove all hidden physics state equal')
        result['initial_parity'] = initial
        replay.write(directory/'initial_parity.json', initial)
        if not matched:
            result['status'] = 'initial_source_mismatch'
            return
        task.set_path_lst(dict(trajectory, need_plan=False))
        original_capture = task._take_picture
        scene = task.scene
        timestep = float(scene.get_timestep())
        with (directory/'trace.jsonl').open('x', buffering=1) as trace:
            def capture(*args, **kwargs):
                nonlocal count, maximum, measured_maximum
                physical = replay.physical(task)
                drive = np.asarray(physical['left']['drive_targets']+physical['right']['drive_targets'])
                real = np.asarray(physical['left']['measured_qpos']+physical['right']['measured_qpos'])
                expected = state[count] if count < len(state) else action[-1]
                error = float(np.abs(drive-expected).max())
                real_error = float(np.abs(real-expected[list(replay.JOINTS)]).max())
                maximum = max(maximum, error)
                measured_maximum = max(measured_maximum, real_error)
                trace.write(json.dumps(dict(source_capture_index=count, physical=physical,
                    drive_target_max_abs_error=error, measured_joint_max_abs_error=real_error,
                    physics_steps_total=counter.count, simulation_seconds=counter.count*timestep), allow_nan=False)+'\n')
                count += 1
                return original_capture(*args, **kwargs)
            task._take_picture = capture
            prior = sys.gettrace()
            sys.settrace(counter.trace)
            try:
                info = task.play_once()
            finally:
                sys.settrace(prior)
                task._take_picture = original_capture
        result.update(status='completed', success=bool(task.plan_success and task.check_success()),
                      plan_success=bool(task.plan_success), captures=count, expected_captures=len(state)+1,
                      source_capture_count_equal=count == len(state)+1,
                      max_drive_target_error=maximum, max_measured_joint_error=measured_maximum,
                      physics_steps=counter.count, timestep=timestep, simulation_seconds=counter.count*timestep,
                      timing_method='trace actual self.scene.step source call sites; no extra physics calls',
                      final_physical=replay.physical(task), episode_info=info)
    except Exception as error:
        result.update(status='worker_error', error=f'{type(error).__name__}: {error}')
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        replay.write(directory/'result.json', result)
        try:
            task.close_env()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=('run', 'sim'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--limit-per-task', type=int, default=10)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--case-ids', nargs='+')
    parser.add_argument('--initial-mae-tolerance', type=float, default=8.)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if args.mode == 'sim':
        return simulate(args.case, args.output, args.initial_mae_tolerance)
    import controlled_paired_eval as controlled
    import gen1_raw_paired_eval as binding
    from robotwin_public_openpi_harness import _configure_sim_renderer_env
    cases = replay.select_cases(json.loads(args.manifest.read_text()), args.limit_per_task, args.smoke_only)
    if args.case_ids:
        available = {case['id']: case for case in json.loads(args.manifest.read_text())['cases']}
        cases = [available[case_id] for case_id in args.case_ids]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for case in cases:
        saved_trajectory(case)
    replay.write(args.output/'manifest.json', dict(protocol=__doc__, cases=cases,
                 source_manifest=str(args.manifest.resolve()), source_manifest_sha256=replay.digest(args.manifest),
                 script_sha256=replay.digest(SCRIPT), common_replay_script_sha256=replay.digest(replay.SCRIPT)))
    env = binding.runtime_environment()
    _configure_sim_renderer_env(env)
    records = []
    for case in cases:
        directory = args.output/case['id']
        directory.mkdir()
        replay.write(directory/'case.json', case)
        command = [controlled.CONFIG['sim_python'], '-I', '-u', '-B', str(SCRIPT), 'sim',
                   '--case', str(directory/'case.json'), '--output', str(directory),
                   '--initial-mae-tolerance', str(args.initial_mae_tolerance)]
        with (directory/'simulator.log').open('x') as log:
            try:
                process = subprocess.run(command, env=env, cwd=controlled.CONFIG['robotwin_root'],
                                         stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                if not (directory/'result.json').exists():
                    replay.write(directory/'result.json', dict(id=case['id'], task=case['task'],
                                 status='process_error', success=False, returncode=process.returncode))
            except subprocess.TimeoutExpired:
                replay.write(directory/'result.json', dict(id=case['id'], task=case['task'],
                             status='timeout', success=False))
        record = json.loads((directory/'result.json').read_text())
        records.append(record)
        replay.write(args.output/'progress.json', dict(attempted=len(records), requested=len(cases), records=records))
        print('DENSE_RESULT', json.dumps({k: record.get(k) for k in ('id', 'status', 'success', 'max_drive_target_error', 'elapsed_seconds')}), flush=True)
    replay.write(args.output/'summary.json', dict(requested=len(cases), successes=sum(r['success'] for r in records), records=records))


if __name__ == '__main__':
    main()
