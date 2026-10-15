#!/usr/bin/env python3
"""Serialize audited preparation, training, verification and fresh Clean panels."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/EvolveVLA')
sys.path.insert(0, str(ROOT / 'scripts'))
from gen1_multitask_data import digest, require
from gen1_recovery_next_after_audit import process_identity, checked_result

PYTHON = '/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python'
TRAINER = ROOT / 'scripts/gen1_multitask_train.py'
TRAINING = ROOT / 'outputs/gen1_multitask_balanced_20261012'
CHECKPOINTS = ROOT / 'checkpoints/gen1_multitask_balanced_20261012'
PREFLIGHT = ROOT / 'outputs/gen1_multitask_loader_20261012'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def gpu_free():
    output = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True).stdout.strip()
    require(not output, f'GPU already occupied: {output}')


def common_arguments(output):
    data = ROOT / 'data'
    return [PYTHON, '-u', str(TRAINER),
            '--repo-id', 'gen1_multitask_coverage_train_20261012',
            '--val-repo-id', 'gen1_multitask_coverage_val_20261012',
            '--lerobot-home', str(data),
            '--split-manifest', str(data / 'gen1_multitask_coverage_20261012/split_manifest.json'),
            '--train-audit', str(data / 'gen1_multitask_coverage_train_20261012/conversion_audit.json'),
            '--val-audit', str(data / 'gen1_multitask_coverage_val_20261012/conversion_audit.json'),
            '--checkpoint-dir', str(CHECKPOINTS), '--provenance-dir', str(output),
            '--steps', '24000', '--save-steps', '1000', '4000', '12000', '24000',
            '--batch-size', '8', '--learning-rate', '3e-6']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--audit-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve().parent == ROOT / 'outputs', 'Unsafe manager output')
    args.output.mkdir(exist_ok=False)
    identity = process_identity(args.audit_pid)
    if identity is None:
        for split in ('train', 'val'):
            path = ROOT / f'data/gen1_multitask_coverage_{split}_20261012/conversion_audit.json'
            audit = json.loads(path.read_text())
            require(audit['status'] == 'passed' and audit['all_frames_decoded'] is True,
                    'No live audit and finalized audit incomplete')
    dependencies = [Path(__file__), TRAINER] + [ROOT / 'scripts' / name for name in (
        'gen1_multitask_data.py', 'gen1_multitask_video.py', 'gen1_multitask_prompts.py',
        'gen1_prompt_augmentation.py', 'gen1_multitask_audit.py', 'gen1_diagnose_pilot.py',
        'gen1_recovery_verify.py', 'gen1_recovery_confirm.py', 'gen1_recovery_eval.py')]
    hashes = {str(path): digest(path) for path in dependencies}
    write(args.output / 'plan.json', dict(audit_pid=args.audit_pid, process_start_time=identity,
        source_sha256=hashes, training_command=common_arguments(TRAINING),
        training=str(TRAINING), checkpoints=str(CHECKPOINTS), preflight=str(PREFLIGHT),
        checkpoint_selection='Equal task mean of four valid-horizon action MAE ratios to original; earlier step breaks ties',
        clean_only=True, pairs_per_task=10, confirmation_seeds=[700000, 800000],
        caveat='Broader-data recovery with several documented interventions, not proof augmentation alone helps; goal remains unachieved'))
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', WANDB_MODE='disabled')

    def run(stage, command, needs_gpu=False):
        for path, sha in hashes.items():
            require(digest(path) == sha, f'Queued source changed: {path}')
        if needs_gpu:
            gpu_free()
        print('START', stage, flush=True)
        with (args.output / f'{stage}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(args.output / f'{stage}_process.json', dict(pid=child.pid,
                process_start_time=process_identity(child.pid), command=command))
            code = child.wait()
        write(args.output / f'{stage}_exit.json', dict(returncode=code,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        require(code == 0, f'{stage} failed; inspect {stage}.log')
        print('FINISHED', stage, flush=True)

    try:
        while identity is not None and process_identity(args.audit_pid) == identity:
            time.sleep(15)
        for split in ('train', 'val'):
            path = ROOT / f'data/gen1_multitask_coverage_{split}_20261012/conversion_audit.json'
            audit = json.loads(path.read_text())
            require(audit['status'] == 'passed' and audit['all_frames_decoded'] is True, 'Audit incomplete')
        run('preflight', common_arguments(PREFLIGHT) + ['--mode', 'loader'])
        preflight = json.loads((PREFLIGHT / 'loader_preflight.json').read_text())
        require(preflight['status'] == 'passed' and preflight['validation_anchors'] == 720, 'Preflight incomplete')
        run('training', common_arguments(TRAINING) + ['--mode', 'train', '--coordinator-approved-gpu'], True)
        complete = json.loads((TRAINING / 'completed.json').read_text())
        require(complete['status'] == 'completed' and complete['steps'] == 24000, 'Training incomplete')
        choice = json.loads((TRAINING / 'selection.json').read_text())['selected']
        checkpoint = Path(choice['checkpoint'])
        require(checkpoint.parent == CHECKPOINTS, 'Foreign checkpoint selected')
        run('verify', [PYTHON, '-u', str(ROOT / 'scripts/gen1_recovery_verify.py'),
                       '--checkpoint', str(checkpoint), '--training', str(TRAINING)])
        results = []
        for number, seed in enumerate((700000, 800000), start=1):
            output = ROOT / f'outputs/gen1_multitask_confirmation{number}_20261012'
            run(f'confirmation{number}', [PYTHON, '-u', str(ROOT / 'scripts/gen1_recovery_confirm.py'),
                'run', '--checkpoint', str(checkpoint), '--seed', str(seed), '--output', str(output)], True)
            result = checked_result(output / 'comparison.json')
            results.append(dict(path=str(output / 'comparison.json'), assessment=result['outcome_assessment']))
            if not result['outcome_assessment']['observed_gain_without_task_regression']:
                break
        write(args.output / 'completed.json', dict(status='completed', selection=choice,
              confirmations=results, goal_achieved=False,
              note='Interpret paired closed-loop results; independent confirmation and broader regression evidence required before claiming a fix'))
    except BaseException as exc:
        write(args.output / 'failed.json', dict(error=f'{type(exc).__name__}: {exc}', goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
