"""Verify subset identity and source allocation on actual finalized numeric data."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from gen1_replay_loader import EpisodeSubset, LOCAL_TRAIN_TASKS, EXPANDED_REPO
from gen1_replay_sampler import ReplaySampler
from gen1_multitask_data import digest

root = Path('/root/autodl-tmp/EvolveVLA')
split_path = root/'data/gen1_multitask_coverage_expanded_20261014/split_manifest.json'
manifest = json.loads(split_path.read_text())
rows = sorted((r for r in manifest['episodes'] if r['split']=='train'), key=lambda r:r['input_link'])
offset = 0
for episode, row in enumerate(rows):
    row.update(episode=episode, start=offset, stop=offset+row['frames'])
    offset += row['frames']
table = pd.concat([pd.read_parquet(p, columns=['index','episode_index','frame_index'])
                   for p in sorted((root/'data'/EXPANDED_REPO/'data').rglob('*.parquet'))]).sort_values('index')
assert len(table)==offset and np.array_equal(table['index'],np.arange(offset))
subset = EpisodeSubset(range(offset), rows, LOCAL_TRAIN_TASKS)
mapped = np.fromiter((subset[i] for i in range(len(subset))),dtype=np.int64)
expected_episodes = {r['episode'] for r in rows if r['task'] in LOCAL_TRAIN_TASKS}
assert set(table.iloc[mapped].episode_index) == expected_episodes
assert len(np.unique(mapped)) == len(mapped)
for row in subset.rows:
    part = table.iloc[mapped[row['start']:row['stop']]]
    assert (part.episode_index==row['episode']).all()
    assert np.array_equal(part.frame_index,np.arange(row['frames']))
official = json.loads((root/'outputs/gen1_official_replay_manifest_20261013.json').read_text())['episodes']
train_rows = [dict(r,source='local') for r in subset.rows]
train_rows += [dict(r,start=r['start']+len(subset),stop=r['stop']+len(subset)) for r in official]
assert len(train_rows)==2890
sampler = ReplaySampler(train_rows, len(subset)+official[-1]['stop'], 28800, allow_local_subset=True)
values = list(sampler)
assert sum(i<len(subset) for i in values)==14400
assert len(sampler.counts)==20
assert all(n==(7200 if source=='local' else 800) for (task,source),n in sampler.counts.items())
report = dict(status='passed', scope='Numeric subset mapping and sampler only; full RGB/mixed-loader preflight still required',
    split_sha256=digest(split_path), local_train_episodes=len(subset.rows),
    local_per_task=dict(Counter(r['task'] for r in subset.rows)),
    official_train_episodes=len(official),official_per_task=dict(Counter(r['task'] for r in official)),
    total_train_episodes=len(train_rows), excluded_local_train_episodes=len(rows)-len(subset.rows),
    local_selected_frames=len(subset), all_selected_frame_indices_verified=True,
    sampler=sampler.report())
with (root/'outputs/two_local_training_scope_20261015.json').open('x') as stream:
    json.dump(report,stream,indent=2)
print(json.dumps(report,indent=2))
