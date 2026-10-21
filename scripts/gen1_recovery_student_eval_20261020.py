#!/usr/bin/env python3
"""Evaluate learned recovery after one exact recorded failed-policy prefix.

This is a privileged-reset development diagnostic, not an end-to-end success
rate. Policy inputs remain RGB,14D commanded state,original instruction. The
student uses the original deployment controller and original episode limit.
Both comparison models reset inference noise to zero at takeover.
"""
import json
import os
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import gen1_policy_duration_repair_20261020 as duration
import gen1_recovery_takeover_v2_20261020 as recovery

horizon = duration.horizon
SOURCE = horizon.ROOT/'outputs/gen1_control_repair_20261020/recovery_takeover_dev3_v2'
PROTOCOL = SOURCE.parent/'student_recovery_protocol.json'
ACTIVE = {}
ORIGINAL_LOOP = horizon.execution_loop
ORIGINAL_WORKER = horizon.worker


def continuation_loop(policy, first, prefix, send_action, receive_step, decode_observation,
                      record_inference, action_limit=2000, **kwargs):
    case, directory = ACTIVE['case'], ACTIVE['directory']
    if os.environ.get('GEN1_DURATION_MODE') != 'original':
        raise ValueError('Recovery student must use the original deployment controller')
    if prefix != 50:
        raise ValueError('The full50 action execution setting is frozen')
    response = first
    for target in case['prefix_targets']:
        send_action(target)
        response = receive_step()
        if response.get('type') != 'step' or response.get('terminated') or response.get('truncated'):
            raise ValueError('Prefix ended before the frozen takeover boundary')
    physical = recovery.lines(directory/'physical.jsonl')
    timing = recovery.lines(directory/'timing.jsonl')
    expected_steps = case['takeover_boundary_step']
    if (len(physical) != expected_steps+1 or len(timing) != expected_steps
            or [r['step'] for r in physical] != list(range(expected_steps+1))
            or [r['action_index'] for r in timing] != list(range(expected_steps))):
        raise ValueError('Prefix trace missing or noncontiguous')
    comparison = dict(source_case=case['id'], prefix_actions=expected_steps,
        requested_targets_exact=[r['target'] for r in timing] == case['prefix_targets'],
        actual_ticks_exact=[r['physics_steps'] for r in timing] == case['prefix_physics_steps'],
        recorded_physical_exact=not recovery.differences(case['prefix_physical'],
                                                       [r['physical'] for r in physical]),
        no_policy_prediction_before_takeover=True,
        inference_rng='reset JAX key0 at takeover in both compared models',
        source='recorded failed24k policy trace; not expert targets')
    horizon.write_json(directory/'prefix_admission.json', comparison)
    if not all(comparison[k] for k in ('requested_targets_exact','actual_ticks_exact','recorded_physical_exact')):
        raise ValueError('Recorded failed-policy prefix did not reproduce exactly')
    # Use the observation returned by the last original-controller action, with
    # the same instruction binding as the reset message. No extra render/step.
    takeover = dict(first, observation=response['observation'])
    images, state = decode_observation(takeover['observation'])
    takeover_hashes = horizon.evaluation.c.capture(directory/'takeover_input', images, state,
                                                  first['diagnostic_binding']['prompt'])
    def record(row, images):
        record_inference(dict(row, step=row['step']+expected_steps,
                              suffix_step=row['step'], phase='student_recovery'), images)
    result = ORIGINAL_LOOP(policy, takeover, prefix, send_action, receive_step,
        decode_observation, record, action_limit-expected_steps, **kwargs)
    result.update(suffix_steps=result['steps'], steps=result['steps']+expected_steps,
                  prefix_actions=expected_steps, exact_prefix=True,
                  takeover_input_hashes=takeover_hashes,
                  interpretation='conditional student recovery from recorded policy state; not full-episode success')
    return result


def worker(case, prefix, args, cache, directory):
    for path, expected in recovery.read(PROTOCOL)['source_sha256'].items():
        if recovery.digest(path) != expected:
            raise ValueError('Frozen student recovery protocol input changed')
    protocol = recovery.read(SOURCE.parent/'decision_protocol.json')
    if recovery.digest(duration.__file__) != protocol['code_sha256']['gen1_policy_duration_repair_20261020.py']:
        raise ValueError('Frozen duration runner dependency changed')
    source_case = SOURCE/case['id']/'case.json'
    source_result = SOURCE/case['id']/'result.json'
    stored, result = recovery.read(source_case), recovery.read(source_result)
    manifest_case = next(r for r in recovery.read(SOURCE/'manifest.json')['cases'] if r['id'] == case['id'])
    if stored != manifest_case:
        raise ValueError('Prefix case differs from its frozen collector manifest')
    if case['id'] not in recovery.IDS or result.get('exact_prefix') is not True:
        raise ValueError('Recovery source has not verified the exact prefix')
    if any(stored[k] != case[k] for k in ('id','task','seed','episode_index','prompt','source_hdf5_sha256')):
        raise ValueError('Source case identity differs')
    for path, sha in stored['source_files_sha256'].items():
        if recovery.digest(path) != sha:
            raise ValueError('Frozen prefix source changed')
    horizon.write_json(directory/'prefix_provenance.json', dict(
        source_case=str(source_case), source_case_sha256=recovery.digest(source_case),
        source_result=str(source_result), source_result_sha256=recovery.digest(source_result),
        collector_source_sha256=recovery.digest(recovery.__file__),
        protocol=str(PROTOCOL), protocol_sha256=recovery.digest(PROTOCOL),
        experiment=__doc__, training_independence=False))
    ACTIVE.update(case=stored, directory=directory)
    return ORIGINAL_WORKER(case, prefix, args, cache, directory)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='sim':
        del sys.argv[1]
        return duration.simulator()
    for path, expected in recovery.read(PROTOCOL)['source_sha256'].items():
        if recovery.digest(path) != expected:
            raise ValueError(f'Frozen student recovery protocol input changed: {path}')
    with horizon.evaluation.binding.replaced(duration, 'SCRIPT', SCRIPT), \
         horizon.evaluation.binding.replaced(horizon, 'worker', worker), \
         horizon.evaluation.binding.replaced(horizon, 'execution_loop', continuation_loop):
        return duration.main()


if __name__=='__main__':
    main()
