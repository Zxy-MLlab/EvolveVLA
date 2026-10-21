#!/usr/bin/env python3
"""Continue retimed source labels after success, recording the true predicate.

This is a persistence diagnostic, not an official success-rate evaluation.
Only early-stop signals inside take_action and the harness are masked. The
official predicate is evaluated unchanged at every original check. Expert
gating remains unchanged; true endpoint values are recorded separately.
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
import gen1_expert_gripper_timing_replay_20261019 as retime
import gen1_expert_label_replay_v2_20261019 as version2


class PredicateRecorder:
    def __init__(self, predicate):
        self.predicate = predicate
        self.in_action = False
        self.action_index = -1
        self.tick_checks = 0
        self.first_true = None
        self.last_true = None
        self.true_checks = 0
        self.events = []
        self.detect_action_scope = False
        self.continuation_enabled = False
        self.last_endpoint_value = None

    def check(self, *args, **kwargs):
        value = bool(self.predicate(*args, **kwargs))
        in_action = self.in_action
        in_harness = False
        if self.detect_action_scope:
            frame = sys._getframe(1)
            while frame is not None:
                if (frame.f_code.co_name == 'take_action'
                    and frame.f_code.co_filename.endswith('/envs/_base_task.py')):
                    in_action = True
                    break
                if frame.f_code.co_name == 'run_simulator' and frame.f_code.co_filename.endswith('/robotwin_eval_harness.py'):
                    in_harness = True
                frame = frame.f_back
        if not in_action:
            if self.continuation_enabled and in_harness:
                self.last_endpoint_value = value
                return False
            return value
        self.tick_checks += 1
        if value:
            event = dict(source_action_index=self.action_index, check_in_action=self.tick_checks)
            if self.first_true is None:
                self.first_true = event
            self.last_true = event
            self.true_checks += 1
        return False


def simulator():
    import gen1_raw_paired_eval as binding
    import robotwin_eval_harness as harness
    directory = Path(os.environ['ROBOTWIN_CONTROLLED_OUTPUT'])
    original_import = importlib.import_module
    attached, recorder, task = False, None, None
    count = 0
    with contextlib.ExitStack() as stack:
        stream = stack.enter_context((directory/'true_success.jsonl').open('x', buffering=1))
        def hooked(name, package=None):
            nonlocal attached, recorder, task
            module = original_import(name, package)
            if name != 'eval_policy_xpolicylab' or attached:
                return module
            attached = True
            construct = module.class_decorator
            transmit = harness.send_message
            def make(*args, **kwargs):
                nonlocal task, recorder
                task = construct(*args, **kwargs)
                # Capture the genuine predicate before v2 installs its gate hook.
                recorder = PredicateRecorder(task.check_success)
                recorder.detect_action_scope = True
                recorder.action_index = 0
                stack.enter_context(binding.replaced(task, 'check_success', recorder.check))
                return task
            def send(channel, payload):
                nonlocal count
                if payload.get('type') == 'episode_start':
                    recorder.continuation_enabled = True
                if payload.get('type') == 'step':
                    # Harness already performed the real outside-action check.
                    value = bool(recorder.last_endpoint_value)
                    row = dict(source_action_index=count, final_after_action_predicate=value,
                               first_true=recorder.first_true, last_true=recorder.last_true,
                               true_tick_checks=recorder.true_checks,
                               physical=core.physical(task))
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    payload = dict(payload, terminated=False,
                                   info=dict(payload['info'], is_success=False))
                    # Suppress only the parent's stop condition; retain the
                    # real predicate in the separate auditable event stream.
                    if payload.get('observation') is None and not payload.get('truncated'):
                        payload['observation'] = harness.simulator_observation(task.now_obs)
                    count += 1
                    recorder.action_index = count
                    recorder.tick_checks = 0
                return transmit(channel, payload)
            stack.enter_context(binding.replaced(module, 'class_decorator', make))
            stack.enter_context(binding.replaced(harness, 'send_message', send))
            return module
        stack.enter_context(binding.replaced(importlib, 'import_module', hooked))
        return retime.simulator()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        del sys.argv[1]
        return simulator()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-ids', nargs='+', required=True)
    parser.add_argument('--gripper-only-ticks', type=int, default=15)
    parser.add_argument('--retime-from-index', type=int, default=319)
    parser.add_argument('--initial-mae-tolerance', type=float, default=8.)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    args.allow_gate_bypass = False
    manifest = json.loads(args.manifest.read_text())
    by_id = {case['id']: case for case in manifest['cases']}
    cases = [by_id[case_id] for case_id in args.case_ids]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(gripper_only_ticks=args.gripper_only_ticks, retime_from_index=args.retime_from_index,
                  joint_epsilon=1e-6, gripper_epsilon=1e-6)
    _, patch = retime.patch_fallback_function(core.ROOT/'robotwin_repo_ba0fd21/envs/_base_task.py', args.gripper_only_ticks)
    core.write(args.output/'manifest.json', dict(protocol=__doc__, cases=cases, intervention=config, method_patch=patch,
        source_manifest=str(args.manifest.resolve()), source_manifest_sha256=core.digest(args.manifest),
        code_sha256={str(path): core.digest(path) for path in (SCRIPT, core.SCRIPT, retime.SCRIPT, version2.SCRIPT,
                     SCRIPT.with_name('gen1_expert_dense_replay_20261019.py'))}))
    core.SCRIPT = SCRIPT
    os.environ['GEN1_RETIME_CONFIG'] = json.dumps(config)
    records = []
    for case in cases:
        os.environ['GEN1_RETIME_CASE'] = json.dumps(case)
        directory = args.output/case['id']
        directory.mkdir()
        record = core.run_case(case, directory, args)
        record['joint_tracking'] = version2.joint_tracking_summary(directory/'trace.jsonl')
        path = directory/'true_success.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        final = events[-1] if events else {}
        record.update(protocol='continue_all_source_labels_after_first_success',
            controller_loop_success_masked=record.pop('success'),
            ever_success=bool(final.get('first_true') or any(row['final_after_action_predicate'] for row in events)),
            final_success=final.get('final_after_action_predicate'),
            first_success=final.get('first_true'), last_success=final.get('last_true'),
            true_tick_checks=final.get('true_tick_checks'), all_source_labels_executed=record.get('steps') == record['source_frames'])
        core.write(directory/'result.json', record)
        records.append(record)
        core.write(args.output/'progress.json', dict(attempted=len(records), requested=len(cases), records=records))
        print('CONTINUE_RESULT', json.dumps({key: record.get(key) for key in ('id','status','ever_success','final_success','steps','elapsed_seconds')}), flush=True)
    core.write(args.output/'summary.json', dict(records=records, requested=len(cases)))


if __name__ == '__main__':
    main()
