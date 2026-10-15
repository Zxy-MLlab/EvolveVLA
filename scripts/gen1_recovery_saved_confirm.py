#!/usr/bin/env python3
"""Confirm the best available saved model after the interrupted coverage run."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/root/autodl-tmp/EvolveVLA')
sys.path.insert(0, str(ROOT/'scripts'))
import gen1_recovery_compare as compare


def main():
    training = ROOT/'outputs/gen1_recovery_coverage_v2_20261012'
    ancestor_training = ROOT/'outputs/gen1_recovery_augmented_20261012'
    baseline = json.loads((ancestor_training/'validation_0000.json').read_text())['metrics']['all']
    candidates = []
    for validation, checkpoint in [
        (training/'validation_0000.json', ROOT/'checkpoints/gen1_recovery_augmented_20261012/2000'),
        (training/'validation_2000.json', ROOT/'checkpoints/gen1_recovery_coverage_v2_20261012/2000'),
    ]:
        metrics = json.loads(validation.read_text())['metrics']
        verification = json.loads((checkpoint/'verification.json').read_text())
        if verification['status'] != 'passed':
            raise ValueError('Unverified checkpoint')
        candidates.append(dict(checkpoint=str(checkpoint), validation=str(validation),
            metrics=metrics, score=sum(metrics['all'][k]/baseline[k] for k in compare.KEYS)/4))
    selected = min(candidates, key=lambda x: x['score'])
    compare.write(training/'interrupted_confirmation_selection.json', dict(
        rule='Same four validation MAE ratios; earlier candidate breaks ties; no rollout selection',
        interruption='Training process disappeared after update 3601; only update 2000 saved; cause unknown',
        scheduled_final_candidate_unavailable=True, candidates=candidates, selected=selected,
        selected_utc=datetime.now(timezone.utc).isoformat()))
    print('SELECTED', json.dumps(selected), flush=True)
    result = subprocess.run([sys.executable, '-u', str(ROOT/'scripts/gen1_recovery_confirm.py'),
        'run', '--checkpoint', selected['checkpoint'], '--seed', '400000',
        '--output', str(ROOT/'outputs/gen1_recovery_confirmation_20261012')])
    compare.write(training/'confirmation_process_exit.json', dict(returncode=result.returncode,
        utc=datetime.now(timezone.utc).isoformat()))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
