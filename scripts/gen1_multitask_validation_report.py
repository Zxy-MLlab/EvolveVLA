#!/usr/bin/env python3
"""Read-only paired-anchor offline diagnostics; never selects a policy."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

KEYS = ('joint_mae', 'gripper_mae', 'first10_joint_mae', 'first10_gripper_mae')


def checked_records(report):
    rows = report['records']
    identities = [(r['task'], r['episode'], r['frame']) for r in rows]
    if len(set(identities)) != len(rows) or len(rows) != report['anchors'] or not rows:
        raise ValueError('Missing or duplicate anchor records')
    if any(not math.isfinite(r[k]) or r[k] < 0 for r in rows for k in KEYS):
        raise ValueError('Invalid action error')
    tasks = sorted({r['task'] for r in rows})
    if set(report['metrics']) != set(tasks) | {'all'}:
        raise ValueError('Metric task set differs from records')
    for task in tasks + ['all']:
        selected = [r for r in rows if task == 'all' or r['task'] == task]
        for key in KEYS:
            actual = statistics.mean(r[key] for r in selected)
            if not math.isclose(report['metrics'][task][key], actual, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('Summary differs from anchor records')
    return {identity: row for identity, row in zip(identities, rows)}


def compare(baseline, candidate):
    if baseline['step'] != 0 or candidate['step'] <= 0:
        raise ValueError('Expected initialization and positive-step candidate')
    before, after = checked_records(baseline), checked_records(candidate)
    if before.keys() != after.keys():
        raise ValueError('Validation anchor panel changed')
    counts = Counter(key[0] for key in before)
    tasks = {}
    for task in sorted(counts):
        metrics = {}
        for key in KEYS:
            old = baseline['metrics'][task][key]
            new = candidate['metrics'][task][key]
            metrics[key] = dict(before=old, after=new, absolute_change=new-old,
                                ratio=new / max(old, 1e-8),
                                percent_change=None if old == 0 else (new/old-1)*100)
        tasks[task] = dict(anchors=counts[task], metrics=metrics,
            any_error_increased=any(m['absolute_change'] > 0 for m in metrics.values()))
    score = statistics.mean(m['ratio'] for task in tasks.values() for m in task['metrics'].values())
    return dict(step=candidate['step'], anchors=len(before), tasks=tasks,
        prespecified_composite_score=score,
        increased_error_tasks={k: [t for t in tasks if tasks[t]['metrics'][k]['absolute_change'] > 0]
                               for k in KEYS},
        limitation='Offline paired expert-action errors, not closed-loop success or causal evidence. No checkpoint selection is changed.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New diagnostic JSON; never overwrite')
    args = parser.parse_args()
    base = args.training / 'validation_0000.json'
    baseline = json.loads(base.read_text())
    checked_records(baseline)
    paths = sorted(args.training.glob('validation_*.json'))
    paths = [p for p in paths if p.stem.removeprefix('validation_').isdigit()]
    comparisons = [compare(baseline, json.loads(p.read_text())) for p in paths if p != base]
    if not comparisons:
        raise ValueError('No post-training validation yet; do not invent improvement')
    report = dict(comparisons=sorted(comparisons, key=lambda r:r['step']),
                  sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for row in report['comparisons']:
        print(json.dumps({k:v for k,v in row.items() if k != 'tasks'}))


if __name__ == '__main__':
    main()
