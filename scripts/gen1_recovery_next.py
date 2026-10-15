#!/usr/bin/env python3
"""Serialize the second recovery intervention after the first panel finishes."""
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/EvolveVLA')
first=ROOT/'outputs/gen1_recovery_fresh_compare_v3_20261012/comparison.json'
while not first.exists():
    print('Waiting for first controlled panel',flush=True)
    time.sleep(30)
training=ROOT/'outputs/gen1_recovery_augmented_20261012'
checkpoints=ROOT/'checkpoints/gen1_recovery_augmented_20261012'
with (ROOT/'logs/gen1_recovery_augmented_20261012.log').open('x') as log:
    subprocess.run(['/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python','-I','-u','-B',
        str(ROOT/'scripts/gen1_recovery_train.py'),'--full-parameters','--augment-prompts','--physical-loss',
        '--batch-size','4','--steps','2000','--eval-every','1000','--learning-rate','5e-6',
        '--output',str(training),'--checkpoint-root',str(checkpoints)],stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,'-u',str(ROOT/'scripts/gen1_recovery_compare.py'),
    '--training',str(training),'--checkpoints',str(checkpoints),
    '--output',str(ROOT/'outputs/gen1_recovery_augmented_compare_20261012'),
    '--reference',str(ROOT/'outputs/gen1_recovery_fresh_compare_v3_20261012_original')],check=True)
