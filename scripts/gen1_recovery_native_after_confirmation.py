#!/usr/bin/env python3
"""Serialize a native-loss ablation after the current independent confirmation.

Runs regardless of the confirmation's policy outcomes. The ablation changes
only the loss relative to the original augmented intervention, selects weights
on held-out action errors, and reserves fresh 600000 seeds for confirmation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
import time

ROOT = Path('/root/autodl-tmp/EvolveVLA')
sys.path.insert(0, str(ROOT/'scripts'))
import gen1_recovery_compare as comparison
from gen1_recovery_next_after_audit import checked_result, process_identity

TRAIN_PYTHON = '/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python'
PRIOR = ROOT/'outputs/gen1_recovery_coverage_confirmation2_20261012'
PRIOR_MANAGER = ROOT/'outputs/gen1_recovery_next_20261012'
OUTPUT = ROOT/'outputs/gen1_recovery_native_queue_20261012'
TRAINING = ROOT/'outputs/gen1_recovery_native_augmented_20261012'
CHECKPOINTS = ROOT/'checkpoints/gen1_recovery_native_augmented_20261012'
CONFIRMATION = ROOT/'outputs/gen1_recovery_native_confirmation_20261012'


def read(path):
    return json.loads(path.read_text())


def verified_confirmation(directory, seed_start, checkpoint):
    result = checked_result(directory/'comparison.json')
    manifest = read(directory/'manifest.json')
    if (manifest['seed_start'] != seed_start
            or manifest['checkpoint'] != str(checkpoint)
            or manifest['target_pairs_per_task'] != 10
            or manifest['development_panel_reused']
            or result['development_panel_reused']
            or result['checkpoint'] != str(checkpoint)):
        raise RuntimeError('Confirmation checkpoint, seed or independence mismatch')
    original, candidate = read(directory/'original.json'), read(directory/'candidate.json')
    for panel in (original, candidate):
        keys = [(r['task'], r['seed']) for r in panel['records']]
        if len(keys) != len(set(keys)):
            raise RuntimeError('Duplicate raw confirmation record')
    if comparison.summarize(original, candidate)['tasks'] != result['tasks']:
        raise RuntimeError('Confirmation does not agree with raw inputs and outcomes')
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--confirmation-supervisor-pid', type=int, required=True)
    args = parser.parse_args()
    identity = process_identity(args.confirmation_supervisor_pid)
    if identity is None:
        raise RuntimeError('Expected live confirmation supervisor when queueing')
    if any(path.exists() for path in (OUTPUT, TRAINING, CHECKPOINTS, CONFIRMATION)):
        raise FileExistsError('Fresh control experiment roots required')
    OUTPUT.mkdir()
    command = [TRAIN_PYTHON, '-I', '-u', '-B',
        str(ROOT/'scripts/gen1_recovery_train.py'), '--full-parameters', '--augment-prompts',
        '--batch-size', '4', '--steps', '2000', '--eval-every', '1000',
        '--learning-rate', '5e-6', '--data-seed', '0',
        '--output', str(TRAINING), '--checkpoint-root', str(CHECKPOINTS)]
    comparison.write(OUTPUT/'plan.json', dict(
        queued_utc=datetime.now(timezone.utc).isoformat(),
        watched_pid=args.confirmation_supervisor_pid, watched_start_time=identity,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        train_source_sha256=hashlib.sha256((ROOT/'scripts/gen1_recovery_train.py').read_bytes()).hexdigest(),
        prerequisite=str(PRIOR), training_command=command,
        rule='After successful second confirmation, run native-loss ablation regardless of its policy outcomes; select on held-out action errors only.',
        matched_intervention=str(ROOT/'outputs/gen1_recovery_augmented_20261012'),
        changed_variable='Restore native 32-dimensional flow-matching loss instead of weighted physical-only loss.',
        fresh_seed_start=600000, clean_only=True, valid_pairs_per_task=10,
        limitation='Does not isolate augmentation itself; training or offline MAE alone cannot establish closed-loop benefit.'))
    print('WAITING_FOR_CONFIRMATION', args.confirmation_supervisor_pid, identity, flush=True)
    try:
        while process_identity(args.confirmation_supervisor_pid) == identity:
            time.sleep(15)
        completed = read(PRIOR_MANAGER/'completed.json')
        if (completed['status'] != 'completed'
                or completed['confirmation'] != str(PRIOR/'comparison.json')):
            raise RuntimeError('Prior manager did not complete the expected confirmation')
        coverage_checkpoint = ROOT/'checkpoints/gen1_recovery_coverage_v2_20261012/2000'
        prior_result = verified_confirmation(PRIOR, 500000, coverage_checkpoint)
        comparison.write(OUTPUT/'prerequisite.json', dict(
            status='passed', assessment=prior_result['outcome_assessment'],
            rule='Outcomes recorded for evidence only; they do not select training configuration.'))
        active_gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout.strip()
        if active_gpu:
            raise RuntimeError(f'GPU still occupied by process IDs: {active_gpu}')
        for source, digest_key in ((ROOT/'scripts/gen1_recovery_train.py', 'train_source_sha256'),
                                   (Path(__file__), 'source_sha256')):
            if hashlib.sha256(source.read_bytes()).hexdigest() != read(OUTPUT/'plan.json')[digest_key]:
                raise RuntimeError('Planned experiment source changed while waiting')
        print('TRAINING_NATIVE_CONTROL', flush=True)
        with (ROOT/'logs/gen1_recovery_native_augmented_20261012.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            comparison.write(OUTPUT/'training_process.json', dict(
                pid=process.pid, start_time=process_identity(process.pid), command=command))
            returncode = process.wait()
        comparison.write(OUTPUT/'training_exit.json', dict(returncode=returncode))
        if returncode or read(TRAINING/'completed.json')['status'] != 'completed':
            raise RuntimeError('Native-loss training did not complete successfully')
        selection = comparison.select(TRAINING, CHECKPOINTS)
        comparison.write(TRAINING/'confirmation_selection.json', selection)
        chosen = Path(selection['selected']['checkpoint'])
        subprocess.run([TRAIN_PYTHON, '-I', '-B',
            str(ROOT/'scripts/gen1_recovery_verify.py'), '--checkpoint', str(chosen),
            '--training', str(TRAINING)], check=True)
        verification = read(chosen/'verification.json')
        if (verification['status'] != 'passed' or not verification['all_finite']
                or not verification['normalization_exact']):
            raise RuntimeError('Selected checkpoint failed production verification')
        print('CONFIRMING_NATIVE_CONTROL', chosen, 'seed=600000', flush=True)
        subprocess.run([sys.executable, '-u', '-B',
            str(ROOT/'scripts/gen1_recovery_confirm.py'), 'run', '--checkpoint', str(chosen),
            '--seed', '600000', '--output', str(CONFIRMATION)], check=True)
        result = verified_confirmation(CONFIRMATION, 600000, chosen)
        comparison.write(OUTPUT/'completed.json', dict(status='completed',
            checkpoint=str(chosen), confirmation=str(CONFIRMATION/'comparison.json'),
            assessment=result['outcome_assessment'], goal_achieved=False,
            note='Requires substantive closed-loop improvement and broader Clean regression assessment.'))
        print('NATIVE_CONTROL_RESULT', json.dumps(result['outcome_assessment']), flush=True)
    except BaseException as error:
        comparison.write(OUTPUT/'failed.json', dict(status='failed',
            error=repr(error), traceback=traceback.format_exc(),
            utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
