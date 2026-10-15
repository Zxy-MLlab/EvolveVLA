#!/usr/bin/env python3
"""Resume the audited Clean datasets to 1,000 successes per task, no training.

Uses the original official planner/replay/writer and byte-level validator.
New task-disjoint seeds are also disjoint from the historical seed inventory.
Existing train/validation assignments are preserved; no split is rewritten here.
"""
from pathlib import Path
import json
import gen1_collect_task_batch_20261010 as collector

collector.TARGET_LIMIT = 1000
collector.ATTEMPT_LIMIT = 13000
collector.SEED_RANGES = {
    'place_mouse_pad': (74000, 87000),
    'hanging_mug': (87000, 100000),
}
# No future full checkpoint reserve is needed while training is stopped.
# Keep 8 GiB untouched and require 2 GiB for two concurrent raw replay caches.
collector.STOP_GIB = 8
collector.START_GIB = 10
collector.PRUNE_FAILED_RAW = True
collector.ENTRYPOINT = Path(__file__).resolve()

original_preflight = collector.preflight


def preflight(args):
    collector.require(args.task in collector.SEED_RANGES,
                      'Only the two explicitly requested tasks are authorized')
    result = original_preflight(args)
    # Audit the previous 50 records before they count toward the new target.
    path = collector.BATCH / args.task / 'manifest.json'
    manifest = json.loads(path.read_text())
    rows = manifest['credited_episodes'] + manifest['episodes']
    seeds = [row['seed'] for row in rows]
    collector.require(len(seeds) == len(set(seeds)), 'Duplicate accepted seeds')
    for row in rows:
        collector.require(row['task'] == args.task and row['success']
                          and row['plan_success'] and row['replay_success']
                          and row['source_revision'] == collector.vetted.REV
                          and row['config'] == 'demo_clean',
                          'Invalid previous accepted episode')
        h5 = next((Path(row['episode_directory']) / 'data').glob('*.hdf5'))
        collector.require(collector.vetted.digest(h5) == row['hdf5_sha256'],
                          'Previous accepted HDF5 changed')
    return result


collector.preflight = preflight

if __name__ == '__main__':
    collector.main()
