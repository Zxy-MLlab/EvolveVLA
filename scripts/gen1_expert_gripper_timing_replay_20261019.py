#!/usr/bin/env python3
"""Diagnostic intervention: shorten selected stationary-arm gripper transitions.

All earlier actions use the original sparse evaluator. For source gripper-only
transitions at or after a fixed index, a process-local AST copy of take_action
changes ONLY its four qpos TOPP fallback durations from 50 to the specified
number of ticks. TOPP flags, arm velocity behavior, gripper interpolation and
success predicates remain identical. This is not a production policy change.
"""
import argparse
import ast
import copy
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import gen1_expert_label_replay_20261019 as core
import gen1_expert_label_replay_v2_20261019 as version2


def eligible_transition(index, start_index, current, target, source_state, joint_epsilon=1e-6, gripper_epsilon=1e-6):
    import numpy as np
    joints = list(core.JOINTS)
    return bool(index >= start_index
        and np.max(np.abs(target[joints]-source_state[joints])) < joint_epsilon
        and np.max(np.abs(target[joints]-current[joints])) < joint_epsilon
        and np.max(np.abs(target[[6, 13]]-source_state[[6, 13]])) > gripper_epsilon)


def patch_fallback_function(source_path, ticks):
    """Copy the actual function AST; modify exactly four qpos fallback literals."""
    tree = ast.parse(Path(source_path).read_text(), filename=str(source_path))
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'take_action']
    if len(candidates) != 1:
        raise ValueError('Expected one take_action function')
    original = candidates[0]
    modified = copy.deepcopy(original)
    branches = [node for node in modified.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                and node.test.left.id == 'action_type' and len(node.test.ops) == 1
                and isinstance(node.test.ops[0], ast.Eq) and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == 'qpos']
    if len(branches) != 1:
        raise ValueError('Expected one qpos controller branch')
    replacements = []
    # Inspect only the qpos body, explicitly excluding its ee elif branch.
    for statement in branches[0].body:
        for node in ast.walk(statement):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('left_n_step', 'right_n_step')
                and isinstance(node.value, ast.Constant) and node.value.value == 50):
                replacements.append(dict(line=node.lineno, variable=node.targets[0].id, before=50, after=ticks))
                node.value.value = ticks
    if len(replacements) != 4 or sorted(r['variable'] for r in replacements) != ['left_n_step']*2+['right_n_step']*2:
        raise ValueError(f'Unexpected fallback rewrite set: {replacements}')
    restored = copy.deepcopy(modified)
    replacement_lines = {r['line'] for r in replacements}
    for node in ast.walk(restored):
        if isinstance(node, ast.Assign) and node.lineno in replacement_lines:
            node.value.value = 50
    if ast.dump(original, include_attributes=True) != ast.dump(restored, include_attributes=True):
        raise ValueError('Retiming changed something besides four fallback constants')
    report = dict(source=str(source_path), source_sha256=core.digest(source_path),
                  replacements=replacements, only_four_qpos_fallback_literals_changed=True,
                  compilation_preserves_source_filename_and_line_numbers=True)
    return modified, report


def retimed_method(original_take, ticks):
    source_path = Path(original_take.__func__.__code__.co_filename).resolve()
    function, report = patch_fallback_function(source_path, ticks)
    namespace = dict(original_take.__func__.__globals__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), 'exec'), namespace)
    return types.MethodType(namespace['take_action'], original_take.__self__), report


def simulator():
    import numpy as np
    import gen1_raw_paired_eval as binding
    case = json.loads(os.environ['GEN1_RETIME_CASE'])
    source_state, source_action = core.source_arrays(case)
    config = json.loads(os.environ['GEN1_RETIME_CONFIG'])
    directory = Path(os.environ['ROBOTWIN_CONTROLLED_OUTPUT'])
    original_import = importlib.import_module
    attached = False
    with contextlib.ExitStack() as stack:
        stream = stack.enter_context((directory/'retiming.jsonl').open('x', buffering=1))
        def hooked(name, package=None):
            nonlocal attached
            module = original_import(name, package)
            if name != 'eval_policy_xpolicylab' or attached:
                return module
            attached = True
            construct = module.class_decorator
            def make(*args, **kwargs):
                task = construct(*args, **kwargs)
                original_take = task.take_action
                modified_take, method_report = retimed_method(original_take, config['gripper_only_ticks'])
                core.write(directory/'retiming_method.json', method_report)
                action_index = 0
                def take(action, *a, **kw):
                    nonlocal action_index
                    index = action_index
                    action_index += 1
                    target = np.asarray(action)
                    np.testing.assert_array_equal(target, source_action[index])
                    current = np.asarray(task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState())
                    active = eligible_transition(index, config['retime_from_index'], current, target,
                                                 source_state[index], config['joint_epsilon'], config['gripper_epsilon'])
                    if not active:
                        return original_take(action, *a, **kw)
                    result = modified_take(action, *a, **kw)
                    stream.write(json.dumps(dict(source_action_index=index,
                        requested_physics_ticks=config['gripper_only_ticks'], method='four_qpos_fallback_durations_only',
                        source_joint_delta=float(np.max(np.abs(target[list(core.JOINTS)]-source_state[index,list(core.JOINTS)]))),
                        source_gripper_delta=(target[[6,13]]-source_state[index,[6,13]]).tolist(),
                        current_gripper_values=current[[6,13]].tolist(), target_gripper_values=target[[6,13]].tolist()), allow_nan=False)+'\n')
                    return result
                stack.enter_context(binding.replaced(task, 'take_action', take))
                return task
            stack.enter_context(binding.replaced(module, 'class_decorator', make))
            return module
        stack.enter_context(binding.replaced(importlib, 'import_module', hooked))
        return version2.simulator()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'sim':
        del sys.argv[1]
        return simulator()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-ids', nargs='+', required=True)
    parser.add_argument('--gripper-only-ticks', type=int, default=15)
    parser.add_argument('--retime-from-index', type=int, default=319)
    parser.add_argument('--joint-epsilon', type=float, default=1e-6)
    parser.add_argument('--gripper-epsilon', type=float, default=1e-6)
    parser.add_argument('--initial-mae-tolerance', type=float, default=8.)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if args.gripper_only_ticks < 1 or args.retime_from_index < 0 or args.joint_epsilon != 1e-6:
        parser.error('Positive ticks, nonnegative start, and fixed joint epsilon 1e-6 required')
    args.allow_gate_bypass = False
    manifest = json.loads(args.manifest.read_text())
    by_id = {case['id']: case for case in manifest['cases']}
    cases = [by_id[case_id] for case_id in args.case_ids]
    if len(set(args.case_ids)) != len(cases):
        parser.error('Duplicate case IDs')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: getattr(args, key) for key in
              ('gripper_only_ticks', 'retime_from_index', 'joint_epsilon', 'gripper_epsilon')}
    preflight = []
    source_path = core.ROOT/'robotwin_repo_ba0fd21/envs/_base_task.py'
    _, method_report = patch_fallback_function(source_path, args.gripper_only_ticks)
    for case in cases:
        states, actions = core.source_arrays(case)
        eligible = [i for i in range(len(actions)) if eligible_transition(i, args.retime_from_index,
                    states[i], actions[i], states[i], args.joint_epsilon, args.gripper_epsilon)]
        preflight.append(dict(id=case['id'], eligible_source_indices=eligible))
    core.write(args.output/'manifest.json', dict(protocol=__doc__, cases=cases, preflight=preflight,
        intervention=config, method_patch=method_report,
        source_manifest=str(args.manifest.resolve()), source_manifest_sha256=core.digest(args.manifest),
        code_sha256={str(path): core.digest(path) for path in (SCRIPT, core.SCRIPT, version2.SCRIPT,
                     SCRIPT.with_name('gen1_expert_dense_replay_20261019.py'))}))
    print('RETIME_PREFLIGHT', json.dumps(preflight), flush=True)
    if args.mode == 'prepare':
        return
    core.SCRIPT = SCRIPT
    records = []
    os.environ['GEN1_RETIME_CONFIG'] = json.dumps(config)
    for case in cases:
        os.environ['GEN1_RETIME_CASE'] = json.dumps(case)
        directory = args.output/case['id']
        directory.mkdir()
        record = core.run_case(case, directory, args)
        record['joint_tracking'] = version2.joint_tracking_summary(directory/'trace.jsonl')
        events_path = directory/'retiming.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        record['retiming'] = dict(config=config, activations=events,
                                 activated_indices=[event['source_action_index'] for event in events],
                                 method_patch=method_report)
        core.write(directory/'result.json', record)
        records.append(record)
        core.write(args.output/'progress.json', dict(requested=len(cases), attempted=len(records), records=records))
        print('RETIME_RESULT', json.dumps({key: record.get(key) for key in ('id','status','success','steps','elapsed_seconds')}), flush=True)
    core.write(args.output/'summary.json', dict(requested=len(cases), successes=sum(r['success'] for r in records), records=records))


if __name__ == '__main__':
    main()
