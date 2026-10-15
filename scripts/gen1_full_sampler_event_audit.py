#!/usr/bin/env python3
"""Replay the pinned task-balanced sampler and count gripper-event exposure.

CPU/read-only audit for the active 18-task run. It reconstructs exactly
steps*batch_size global starts with the configured sampler seed, then labels
whether each start's 50-action target window contains an expert gripper
closing/opening transition. It does not run or modify training.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from gen1_multitask_data import TaskBalancedSampler

ROOT = Path('/root/autodl-tmp/EvolveVLA')
MANIFEST = ROOT / 'data/gen1_multitask_coverage_20261012/split_manifest.json'
AUDIT = ROOT / 'data/gen1_multitask_coverage_train_20261012/conversion_audit.json'
OUT = ROOT / 'outputs/gen1_multitask_balanced_20261012/full_sampler_event_audit_corrected.json'

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def event_flags(grippers, horizon=50):
    changes = np.diff(grippers, axis=0)
    out = {}
    for name, flag in (("closing", changes < -0.05), ("opening", changes > 0.05)):
        prefix = np.concatenate([np.zeros((1, 2), dtype=np.int64), np.cumsum(flag, axis=0)])
        starts = np.arange(len(grippers))
        stops = np.minimum(starts + horizon - 1, len(grippers) - 1)
        out[name] = (prefix[stops] - prefix[starts]) > 0
    return out

def replay_indices(rows, steps, batch_size, seed=1):
    """Reconstruct consumed complete batches, not runtime observations."""
    loader_samples = rows[-1]['stop']
    if steps < 0 or batch_size <= 0 or loader_samples < batch_size:
        raise ValueError('Invalid replay size')
    sampler = TaskBalancedSampler(rows, loader_samples, seed=seed)
    remaining = steps
    while remaining:
        indices = iter(sampler)
        batches = min(remaining, loader_samples // batch_size)
        for _ in range(batches):
            for _ in range(batch_size):
                yield next(indices)
        remaining -= batches

def main():
    from gen1_multitask_data import ordered_episodes, TaskBalancedSampler
    manifest = json.loads(MANIFEST.read_text())
    audit = json.loads(AUDIT.read_text())
    rows = [r for r in ordered_episodes(manifest, audit, 'train')]
    # The active loader's sampler length is the transformed dataset length,
    # not the requested optimizer sample count. Torch DataLoader drops the
    # final incomplete batch (there is none here), then recreates the sampler
    # iterator after exhaustion. Reproduce that exact epoch schedule.
    loader_samples = rows[-1]['stop']
    batch_size = 8
    steps = 24000
    batches_per_epoch = loader_samples // batch_size
    starts = int(steps * batch_size)
    # Build event labels by audited global index.
    events = {}
    interval_starts, interval_stops = [], []
    for row in rows:
        with h5py.File(row['path'], 'r') as f:
            grippers = np.concatenate([f[f'action/{arm}_ee_joint_states'][:] for arm in ('left', 'right')], axis=1)
        if grippers.shape != (row['frames'], 2) or not np.isfinite(grippers).all():
            raise ValueError(f'Invalid gripper data: {row["path"]}')
        flags = event_flags(grippers)
        events[row['episode']] = flags
        interval_starts.append(row['start']); interval_stops.append(row['stop'])
    task_counts = Counter(); task_events = defaultdict(Counter); unique = set()
    for global_index in replay_indices(rows, steps, batch_size):
        i = bisect_right(interval_stops, global_index)
        if i >= len(rows) or global_index < interval_starts[i]:
            raise ValueError('Sampler yielded an index outside audited episode intervals')
        row = rows[i]; frame = global_index - row['start']; flags = events[row['episode']]
        task = row['task']; task_counts[task] += 1; unique.add(global_index)
        closing = flags['closing'][frame]; opening = flags['opening'][frame]
        task_events[task]['closing_either'] += int(closing.any())
        task_events[task]['opening_either'] += int(opening.any())
        task_events[task]['closing_left'] += int(closing[0]); task_events[task]['closing_right'] += int(closing[1])
        task_events[task]['opening_left'] += int(opening[0]); task_events[task]['opening_right'] += int(opening[1])
        task_events[task]['any_transition'] += int((closing | opening).any())
    if sum(task_counts.values()) != starts:
        raise ValueError('Sample count mismatch')
    rows_out = {}
    for task in sorted(task_counts):
        n = task_counts[task]
        rows_out[task] = dict(sampled_starts=n, sampled_share=n / starts, unique_sampled_starts=None,
            events=dict(task_events[task]), event_rate={k: v / n for k, v in task_events[task].items()})
    # Compute unique counts per task in a second deterministic replay without retaining all indices
    seen_by_task=defaultdict(set)
    for global_index in replay_indices(rows, steps, batch_size):
        i = bisect_right(interval_stops, global_index); seen_by_task[rows[i]['task']].add(global_index)
    for task, seen in seen_by_task.items(): rows_out[task]['unique_sampled_starts'] = len(seen)
    result = dict(status='completed', steps=steps, batch_size=batch_size, sampler_seed=1,
        loader_samples=loader_samples, batches_per_epoch=batches_per_epoch,
        sampled_starts=starts,
        unique_starts=len(unique), task_balance=dict(task_counts), tasks=rows_out,
        horizon=50, transition_threshold=0.05,
        inputs={str(MANIFEST): digest(MANIFEST), str(AUDIT): digest(AUDIT),
                str(ROOT/'scripts/gen1_multitask_data.py'): digest(ROOT/'scripts/gen1_multitask_data.py')},
        limitations=[
            'Events label expert action targets, not physical grasp success.',
            'Full-run projection, not observed training exposure or proof training completed.',
            'Source numeric arrays read but not rehashed; source identity relies on prior conversion audit.',
            'Reconstruction does not log prompt choices or optimizer gradient weights.',
            'Overlapping windows are correlated and exposure counts do not prove causal improvement.'
        ])
    with OUT.open('x') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({t: rows_out[t] for t in ('place_mouse_pad','hanging_mug')}, indent=2))

if __name__ == '__main__': main()
