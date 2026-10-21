#!/usr/bin/env python3
"""Isolated sparse-label diagnostic v2: exact physics call counts and joint errors.

Reuses the frozen v1 IPC/controller path, replacing only passive observation
hooks. Supports explicit case IDs so smoke cases need not be evaluated again.
"""
import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import gen1_expert_label_replay_20261019 as core
from gen1_expert_dense_replay_20261019 import SceneStepCounter


def simulator():
    import controlled_paired_eval as controlled
    import robotwin_eval_harness as harness
    import gen1_raw_paired_eval as binding
    allow_bypass = os.environ.get('GEN1_ORACLE_GATE_BYPASS') == '1'
    original_import = importlib.import_module
    task, attached = None, False
    gate = dict(status='not_run', bypass_requested=allow_bypass, bypassed=False)
    timing = dict(physics_steps=0, total_physics_steps=0, timestep=None,
                  method='trace actual self.scene.step source call sites; no extra physics calls')
    counter = SceneStepCounter(Path(controlled.CONFIG['robotwin_root'])/'envs/_base_task.py')
    with contextlib.ExitStack() as stack:
        def hooked(name, package=None):
            nonlocal task, attached
            module = original_import(name, package)
            if name != 'eval_policy_xpolicylab' or attached:
                return module
            attached = True
            construct, transmit = module.class_decorator, harness.send_message
            def make(*args, **kwargs):
                nonlocal task
                task = construct(*args, **kwargs)
                original_play, original_check, original_take = task.play_once, task.check_success, task.take_action
                pending_gate = False
                def play(*a, **kw):
                    nonlocal pending_gate
                    result = original_play(*a, **kw)
                    gate.update(plan_success=bool(task.plan_success), check_success=None)
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
                    before = counter.count
                    previous = sys.gettrace()
                    sys.settrace(counter.trace)
                    try:
                        return original_take(*a, **kw)
                    finally:
                        sys.settrace(previous)
                        timing.update(physics_steps=counter.count-before,
                                      total_physics_steps=counter.count,
                                      timestep=float(task.scene.get_timestep()),
                                      simulation_seconds=counter.count*float(task.scene.get_timestep()))
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
                    extra.update(oracle_physical=core.physical(task), oracle_timing=dict(timing))
                return transmit(channel, dict(payload, **extra))
            stack.enter_context(binding.replaced(module, 'class_decorator', make))
            stack.enter_context(binding.replaced(harness, 'send_message', send))
            return module
        stack.enter_context(binding.replaced(importlib, 'import_module', hooked))
        return harness.main()


def joint_tracking_summary(trace_path):
    import numpy as np
    joint_error, gripper_error, real_error, steps = 0., 0., 0., 0
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            row = json.loads(line)
            physical = row['physical']
            drive = np.asarray(physical['left']['drive_targets']+physical['right']['drive_targets'])
            target = np.asarray(row['source_target'])
            difference = np.abs(drive-target)
            joint_error = max(joint_error, float(difference[list(core.JOINTS)].max()))
            gripper_error = max(gripper_error, float(difference[[6, 13]].max()))
            real_error = max(real_error, row['measured_joint_max_abs_error'])
            steps += 1
    return dict(steps=steps, max_joint_drive_target_error=joint_error,
                max_gripper_reported_value_error=gripper_error, max_measured_joint_error=real_error)


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
    parser.add_argument('--case-ids', nargs='+')
    parser.add_argument('--allow-gate-bypass', action='store_true')
    parser.add_argument('--initial-mae-tolerance', type=float, default=8.)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if args.limit_per_task < 1 or args.timeout < 1:
        parser.error('Positive limits required')
    manifest = json.loads(args.manifest.read_text())
    if args.case_ids:
        by_id = {case['id']: case for case in manifest['cases']}
        cases = [by_id[case_id] for case_id in args.case_ids]
        if len(set(args.case_ids)) != len(cases):
            parser.error('Duplicate case IDs')
    else:
        cases = core.select_cases(manifest, args.limit_per_task, args.smoke_only)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    norm = json.loads(core.NORM.read_text())['norm_stats']
    preflight = []
    import numpy as np
    for case in cases:
        state, action = core.source_arrays(case)
        maximum = float(np.abs(core.roundtrip_actions(state, action, norm)-action).max())
        if maximum > 1e-5:
            raise ValueError(f'Roundtrip mismatch: {case["id"]}: {maximum}')
        preflight.append(dict(id=case['id'], frames=len(state), next_state_action_exact=True,
                              roundtrip_max_abs=maximum))
    core.write(args.output/'manifest.json', dict(protocol=__doc__, cases=cases, preflight=preflight,
          source_manifest=str(args.manifest.resolve()), source_manifest_sha256=core.digest(args.manifest),
          code_sha256={str(path): core.digest(path) for path in
                       (SCRIPT, core.SCRIPT, SCRIPT.with_name('gen1_expert_dense_replay_20261019.py'))},
          norm_stats=str(core.NORM), norm_sha256=core.digest(core.NORM),
          allow_gate_bypass=args.allow_gate_bypass, initial_mae_tolerance=args.initial_mae_tolerance))
    print('CPU_PREFLIGHT_PASSED', json.dumps(preflight), flush=True)
    if args.mode == 'prepare':
        return
    core.SCRIPT = SCRIPT  # Spawn this version; frozen v1 file is never edited.
    records = []
    for case in cases:
        directory = args.output/case['id']
        directory.mkdir()
        result = core.run_case(case, directory, args)
        tracking = joint_tracking_summary(directory/'trace.jsonl')
        result['joint_tracking'] = tracking
        core.write(directory/'result.json', result)
        records.append(result)
        core.write(args.output/'progress.json', dict(attempted=len(records), requested=len(cases), records=records))
        print('ORACLE_RESULT', json.dumps({k: result.get(k) for k in ('id', 'status', 'success', 'steps', 'elapsed_seconds')}), flush=True)
    core.write(args.output/'summary.json', dict(records=records, successes=sum(r['success'] for r in records),
        requested=len(cases), source_matched=sum(r.get('initial_parity', {}).get('observed_source_match', False) for r in records),
        gate_bypassed=sum(bool((r.get('expert_gate') or {}).get('bypassed')) for r in records)))


if __name__ == '__main__':
    main()
