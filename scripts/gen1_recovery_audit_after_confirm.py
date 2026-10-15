#!/usr/bin/env python3
"""Serialize padding diagnostics after the live confirmation supervisor exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/EvolveVLA')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--supervisor-pid', type=int, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 3 * 3600
    while True:
        try:
            os.kill(args.supervisor_pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Confirmation supervisor did not exit')
        time.sleep(15)
    result = ROOT/'outputs/gen1_recovery_confirmation_20261012/comparison.json'
    if not result.exists():
        raise RuntimeError('Confirmation exited without comparison; resolve before GPU audit')
    print('CONFIRMATION', result.read_text(), flush=True)
    candidates = {
        'original': 'myungkyu_pi05_jax_robotwin_vanilla_b128_60k',
        'augmented': 'gen1_recovery_augmented_20261012/2000',
        'coverage': 'gen1_recovery_coverage_v2_20261012/2000',
    }
    reports = {}
    for label, relative in candidates.items():
        output = ROOT/'outputs'/f'gen1_recovery_padding_{label}_20261012'
        with (ROOT/'logs'/f'gen1_recovery_padding_{label}_20261012.log').open('x') as log:
            subprocess.run(['/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python', '-I', '-u', '-B',
                str(ROOT/'scripts/gen1_recovery_padding_audit.py'), '--checkpoint',
                str(ROOT/'checkpoints'/relative), '--output', str(output)],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        reports[label] = json.loads((output/'report.json').read_text())
        print('AUDITED', label, json.dumps(reports[label]['metrics']), flush=True)
    if len({r['observation_sha256'] for r in reports.values()}) != 1:
        raise RuntimeError('Audit observations differ')
    with (ROOT/'outputs/gen1_recovery_padding_comparison_20261012.json').open('x') as stream:
        json.dump(dict(input_hashes_equal=True,
            metrics={k:v['metrics'] for k,v in reports.items()},
            limitation='Paired held-out denoising diagnostic; not a causal ablation.'), stream, indent=2)


if __name__ == '__main__':
    main()
