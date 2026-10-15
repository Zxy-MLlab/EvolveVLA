#!/usr/bin/env python3
"""Finish the active diagnostic, extend data coverage, then confirm a fixed model."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/EvolveVLA')
sys.path.insert(0, str(ROOT/'scripts'))
import gen1_recovery_compare as compare


def main():
    previous = ROOT/'outputs/gen1_recovery_augmented_compare_20261012/comparison.json'
    deadline = time.monotonic() + 3600
    while not previous.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Previous panel did not finish within one hour')
        time.sleep(20)
    print('PREVIOUS_COMPARISON', previous.read_text(), flush=True)
    training = ROOT/'outputs/gen1_recovery_coverage_v2_20261012'
    checkpoints = ROOT/'checkpoints/gen1_recovery_coverage_v2_20261012'
    ancestor = ROOT/'checkpoints/gen1_recovery_augmented_20261012/2000'
    with (ROOT/'logs/gen1_recovery_coverage_v2_20261012.log').open('x') as log:
        subprocess.run(['/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python', '-I', '-u', '-B',
            str(ROOT/'scripts/gen1_recovery_train.py'), '--full-parameters', '--augment-prompts',
            '--physical-loss', '--initial-checkpoint', str(ancestor), '--data-seed', '1',
            '--batch-size', '8', '--steps', '4000', '--eval-every', '2000', '--learning-rate', '3e-6',
            '--output', str(training), '--checkpoint-root', str(checkpoints)],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    previous_training = ROOT/'outputs/gen1_recovery_augmented_20261012'
    baseline = json.loads((previous_training/'validation_0000.json').read_text())['metrics']['all']
    ancestor_metrics=json.loads((training/'validation_0000.json').read_text())['metrics']
    candidates = [dict(checkpoint=str(ancestor),training=str(previous_training),step=2000,
        score=sum(ancestor_metrics['all'][k]/baseline[k] for k in compare.KEYS)/4,
        metrics=ancestor_metrics,validation_source=str(training/'validation_0000.json'))]
    for source, root, steps in [(training, checkpoints, [2000, 4000])]:
        for step in steps:
            metrics = json.loads((source/f'validation_{step:04d}.json').read_text())['metrics']
            candidates.append(dict(checkpoint=str(root/str(step)), training=str(source),
                step=step, score=sum(metrics['all'][k]/baseline[k] for k in compare.KEYS)/4,
                metrics=metrics))
    # Earlier listed checkpoint wins a tie; no rollout outcome is consulted.
    best = min(candidates, key=lambda row: row['score'])
    selection = dict(rule='Equal mean of four MAE ratios to original initialization; earliest candidate wins ties; validation only',
                     baseline=baseline, candidates=candidates, selected=best)
    compare.write(training/'confirmation_selection.json', selection)
    chosen = Path(best['checkpoint'])
    if not (chosen/'verification.json').exists():
        subprocess.run(['/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python', '-I', '-B',
            str(ROOT/'scripts/gen1_recovery_verify.py'), '--checkpoint', str(chosen),
            '--training', best['training']], check=True)
    print('CONFIRMATION_SELECTION', json.dumps(selection), flush=True)
    subprocess.run([sys.executable, '-u', str(ROOT/'scripts/gen1_recovery_confirm.py'), 'run',
        '--checkpoint', str(chosen), '--seed', '400000',
        '--output', str(ROOT/'outputs/gen1_recovery_confirmation_20261012')], check=True)


if __name__ == '__main__':
    main()
