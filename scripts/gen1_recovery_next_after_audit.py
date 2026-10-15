#!/usr/bin/env python3
"""Use completed evidence to continue the recovery experiment on one GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/EvolveVLA')
sys.path.insert(0, str(ROOT/'scripts'))
import gen1_recovery_compare as comparison
from gen1_recovery_confirm import assess


def process_identity(pid):
    try:
        # comm may contain spaces or parentheses; fields start after its last ).
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def checked_result(path):
    result = json.loads(path.read_text())
    for task in ('place_mouse_pad', 'hanging_mug'):
        row = result['tasks'][task]
        if row['paired_completed'] != 10 or len(row['pairs']) != 10:
            raise RuntimeError('Completed ten-pair confirmation required')
        if len({pair['seed'] for pair in row['pairs']}) != 10:
            raise RuntimeError('Duplicate confirmation seed')
        expected = dict(
            original_successes=sum(pair['original'] for pair in row['pairs']),
            candidate_successes=sum(pair['candidate'] for pair in row['pairs']),
            original_only=sum(pair['original'] and not pair['candidate'] for pair in row['pairs']),
            candidate_only=sum(pair['candidate'] and not pair['original'] for pair in row['pairs']))
        if any(row[key] != value for key, value in expected.items()):
            raise RuntimeError('Confirmation counts disagree with paired outcomes')
        if any(r['original_status'] not in ('completed', 'expert_rejected')
               or r['candidate_status'] not in ('completed', 'expert_rejected')
               or r['original_status'] == r['candidate_status'] == 'completed'
               for r in row['unpaired']):
            raise RuntimeError('Unresolved evaluation error or initial-input mismatch')
    if result['outcome_assessment'] != assess(result):
        raise RuntimeError('Confirmation assessment disagrees with paired counts')
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--audit-pid', type=int, required=True)
    args = parser.parse_args()
    identity = process_identity(args.audit_pid)
    if identity is None:
        raise RuntimeError('Expected live audit supervisor when queueing')
    output = ROOT/'outputs/gen1_recovery_next_20261012'
    output.mkdir(exist_ok=False)
    comparison.write(output/'plan.json', dict(
        queued_audit_pid=args.audit_pid, process_start_time=identity,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rule='After completed padding audit: if coverage has descriptive gain without task regression, confirm the same model on fresh 500000 seeds; otherwise run the native-loss control and select only by held-out MAE.',
        native_control=dict(initialization='original', full_parameters=True,
            augment_prompts=True, physical_loss=False, steps=2000, batch_size=4,
            learning_rate=5e-6, eval_every=1000, data_seed=0),
        clean_only=True, target_pairs_per_task=10, next_seed_start=500000,
        limitation='Descriptive gain alone does not establish a stable improvement.'))
    deadline = time.monotonic() + 4 * 3600
    while process_identity(args.audit_pid) == identity:
        if time.monotonic() > deadline:
            raise TimeoutError('Audit supervisor did not finish within four hours')
        time.sleep(15)
    result = checked_result(ROOT/'outputs/gen1_recovery_confirmation_20261012/comparison.json')
    padding_path = ROOT/'outputs/gen1_recovery_padding_comparison_20261012.json'
    padding = json.loads(padding_path.read_text())
    if not padding['input_hashes_equal']:
        raise RuntimeError('Padding audit inputs differ')
    # Both supervisors must be finished; the audit serially waited for inference
    # and owns each subsequent GPU diagnostic child until it exits.
    active_gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                                 '--format=csv,noheader,nounits'], check=True,
                                capture_output=True, text=True).stdout.strip()
    if active_gpu:
        raise RuntimeError(f'GPU is still occupied by process IDs: {active_gpu}')
    assessment = result['outcome_assessment']
    print('CONFIRMATION_ASSESSMENT', json.dumps(assessment), flush=True)
    print('PADDING_AUDIT', json.dumps(padding['metrics']), flush=True)
    training = ROOT/'outputs/gen1_recovery_native_augmented_20261012'
    checkpoints = ROOT/'checkpoints/gen1_recovery_native_augmented_20261012'
    if assessment['observed_gain_without_task_regression']:
        chosen = Path(result['checkpoint'])
        next_output = ROOT/'outputs/gen1_recovery_coverage_confirmation2_20261012'
        comparison.write(output/'decision.json', dict(
            action='independent_confirmation', checkpoint=str(chosen),
            assessment=assessment, padding_audit=str(padding_path)))
    else:
        comparison.write(output/'decision.json', dict(
            action='native_loss_control', assessment=assessment,
            padding_audit=str(padding_path),
            rationale='Coverage failed the prespecified no-task-regression gain check. Restore native supervision while matching the earlier augmented run; padding audit alone is not a causal proof.'))
        command = ['/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python', '-I', '-u', '-B',
            str(ROOT/'scripts/gen1_recovery_train.py'), '--full-parameters', '--augment-prompts',
            '--batch-size', '4', '--steps', '2000', '--eval-every', '1000', '--learning-rate', '5e-6',
            '--output', str(training), '--checkpoint-root', str(checkpoints)]
        with (ROOT/'logs/gen1_recovery_native_augmented_20261012.log').open('x') as log:
            print('TRAINING_NATIVE_LOSS', flush=True)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if json.loads((training/'completed.json').read_text())['status'] != 'completed':
            raise RuntimeError('Native-loss training did not complete')
        selection = comparison.select(training, checkpoints)
        comparison.write(training/'confirmation_selection.json', selection)
        chosen = Path(selection['selected']['checkpoint'])
        subprocess.run(['/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python', '-I', '-B',
            str(ROOT/'scripts/gen1_recovery_verify.py'), '--checkpoint', str(chosen),
            '--training', str(training)], check=True)
        next_output = ROOT/'outputs/gen1_recovery_native_confirmation_20261012'
    print('NEXT_CONFIRMATION', str(chosen), str(next_output), flush=True)
    subprocess.run([sys.executable, '-u', str(ROOT/'scripts/gen1_recovery_confirm.py'), 'run',
        '--checkpoint', str(chosen), '--seed', '500000', '--output', str(next_output)], check=True)
    next_result = checked_result(next_output/'comparison.json')
    comparison.write(output/'completed.json', dict(status='completed',
        confirmation=str(next_output/'comparison.json'),
        assessment=next_result['outcome_assessment'],
        goal_achieved=False, note='A human-readable completion audit and broader regression checks remain required.'))
    print('NEXT_RESULT', json.dumps(next_result['outcome_assessment']), flush=True)


if __name__ == '__main__':
    main()
