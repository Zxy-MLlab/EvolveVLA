"""CPU-only data gates and deterministic task-balanced sampling.

Task identity comes from audited source trajectories, never LeRobot task_index
(which indexes instruction strings). Importing this module does not touch GPU.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

REVISION = 'ba0fd21ad21c336bedc7bd8eacd3e514860910e2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def validate_split(manifest):
    require(manifest.get('status') == 'passed', 'Unaccepted split')
    tasks = manifest['tasks']
    require(len(tasks) >= 3 and len(set(tasks)) == len(tasks), 'Invalid task list')
    rows = manifest['episodes']
    require(set(r['split'] for r in rows) == {'train', 'val'}, 'Invalid split labels')
    require(set(r['task'] for r in rows) == set(tasks), 'Task coverage mismatch')
    counts = Counter((r['split'], r['task']) for r in rows)
    expected = Counter({(s, t): (manifest['per_task_counts'][s][t]
                                  if isinstance(manifest['per_task_counts'][s], dict)
                                  else manifest['per_task_counts'][s])
                        for s in ('train', 'val') for t in tasks})
    require(all(n > 0 for n in expected.values()) and counts == expected,
            'Unexpected per-task split counts')
    require(dict(Counter(r['split'] for r in rows)) == manifest['counts'], 'Split count mismatch')
    for key in ('path', 'sha256', 'seed'):
        values = [str(Path(r[key]).resolve()) if key == 'path' else r[key] for r in rows]
        require(len(set(values)) == len(rows), f'Duplicate {key}: potential leakage')
    for row in rows:
        require(row['source_revision'] == REVISION, 'Wrong source revision')
        require(all(row[k] is True for k in ('success', 'plan_success', 'replay_success')),
                'Unaccepted expert trajectory')
        require(type(row['seed']) is int and 0 <= row['seed'] < 100000, 'Invalid collection seed')
        require(type(row['frames']) is int and row['frames'] > 1, 'Invalid episode length')
    return rows


def ordered_episodes(manifest, audit, split):
    """Use converter audit order, not manifest order or prompt IDs."""
    rows = validate_split(manifest)
    require(split in ('train', 'val'), 'Unknown split')
    require(audit.get('status') == 'passed' and audit.get('all_frames_decoded') is True,
            'Conversion not fully audited')
    require(audit['fps'] == 15 and audit['state_action_dim'] == 14, 'Changed data semantics')
    expected = {str(Path(r['path']).resolve()): r for r in rows if r['split'] == split}
    sources = audit['sources']
    require(len(sources) == audit['episodes'] == len(expected), 'Wrong episode count')
    seen, result, offset = set(), [], 0
    for episode, source in enumerate(sources):
        path = str(Path(source['path']).resolve())
        require(path in expected and path not in seen, 'Duplicate or foreign source in audit')
        seen.add(path)
        row = expected[path]
        require((source['sha256'], source['frames']) == (row['sha256'], row['frames']),
                'Source hash/length differs from split')
        result.append(dict(row, episode=episode, start=offset, stop=offset + row['frames']))
        offset += row['frames']
    require(offset == audit['frames'], 'Total frame mismatch')
    return result


def audit_dataset(manifest, root, split, converter):
    """Require final conversion and check every parquet frame's index identity."""
    import numpy as np
    import pyarrow.parquet as pq
    root, converter = Path(root), Path(converter)
    audit_path = root / 'conversion_audit.json'
    audit = json.loads(audit_path.read_text())
    require(Path(audit['converter']).resolve() == converter.resolve()
            and audit['converter_sha256'] == digest(converter), 'Pinned converter mismatch')
    rows = ordered_episodes(manifest, audit, split)
    info = json.loads((root / 'meta/info.json').read_text())
    require((info['total_episodes'], info['total_frames'], info['fps']) ==
            (len(rows), rows[-1]['stop'], 15), 'Dataset metadata mismatch')
    for key in ('action', 'observation.state'):
        require(info['features'][key]['shape'] == [14], 'Physical dimension mismatch')
    cameras = {k: v for k, v in info['features'].items() if v['dtype'] == 'video'}
    require(set(cameras) == {f'observation.images.cam_{k}' for k in
                             ('high', 'left_wrist', 'right_wrist')}, 'Camera mapping mismatch')
    columns = ['index', 'episode_index', 'frame_index', 'task_index']
    blocks = [pq.read_table(p, columns=columns).to_pandas()
              for p in sorted((root / 'data').rglob('*.parquet'))]
    require(bool(blocks), 'No data parquet')
    import pandas as pd
    frames = pd.concat(blocks, ignore_index=True).sort_values('index')
    require(np.array_equal(frames['index'].to_numpy(), np.arange(rows[-1]['stop'])),
            'Non-contiguous or duplicated frame indices')
    for row in rows:
        part = frames.iloc[row['start']:row['stop']]
        require((part.episode_index == row['episode']).all(), 'Audit/parquet episode order mismatch')
        require(np.array_equal(part.frame_index.to_numpy(), np.arange(row['frames'])),
                'Episode boundary mismatch')
        require(part.task_index.nunique() == 1, 'Prompt changes within trajectory')
    return rows, dict(status='passed', split=split, episodes=len(rows), frames=len(frames),
                     audit_sha256=digest(audit_path), actual_video_features=cameras)


class TaskBalancedSampler:
    """One draw/task per shuffled cycle, uniform over that task's frames.

Finite iterators restart with a new deterministic epoch seed. No held-out index
can enter because input ranges must consist entirely of contiguous train rows.
Equal task allocation is an explicit intervention, not the official sampler.
"""
    def __init__(self, rows, num_samples, seed=0):
        require(type(num_samples) is int and num_samples > 0, 'Invalid sample count')
        self.num_samples, self.seed, self.epoch = num_samples, seed, 0
        self.ranges = defaultdict(list)
        offset = 0
        for row in rows:
            require(row['split'] == 'train', 'Validation trajectory in training sampler')
            require(row['start'] == offset and row['stop'] > offset
                    and row['stop'] - offset == row['frames'], 'Invalid contiguous frame ranges')
            self.ranges[row['task']].append((row['start'], row['stop']))
            offset = row['stop']
        require(bool(self.ranges), 'Empty training sampler')
        self.tasks = sorted(self.ranges)
        self.ends = {}
        for task, ranges in self.ranges.items():
            total, ends = 0, []
            for start, stop in ranges:
                total += stop - start
                ends.append(total)
            self.ends[task] = ends
        self.counts, self.unique = Counter(), set()

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = random.Random(f'{self.seed}:{self.epoch}')
        self.epoch += 1
        remaining = self.num_samples
        while remaining:
            tasks = self.tasks.copy()
            rng.shuffle(tasks)
            for task in tasks[:remaining]:
                draw = rng.randrange(self.ends[task][-1])
                episode = bisect_right(self.ends[task], draw)
                before = self.ends[task][episode - 1] if episode else 0
                index = self.ranges[task][episode][0] + draw - before
                self.counts[task] += 1
                self.unique.add(index)
                remaining -= 1
                yield index

    def report(self):
        return dict(per_task_samples=dict(self.counts), unique_frames=len(self.unique),
                    samples=sum(self.counts.values()), epochs_started=self.epoch,
                    allocation='shuffled equal-task cycles; uniform frames within each task')


class PhaseAwareTaskSampler(TaskBalancedSampler):
    """Equal-task sampler with bounded expert-phase oversampling.

    ``weights`` is a positive per-frame array in the audited, contiguous
    training-frame order.  Task allocation remains exactly equal; only the
    frame draw within each task changes.  This is a training intervention and
    never changes action targets, inference, or evaluation predicates.
    """
    def __init__(self, rows, num_samples, weights, seed=0):
        import math
        super().__init__(rows, num_samples, seed=seed)
        require(len(weights) == rows[-1]['stop'], 'Phase weights length mismatch')
        self.weight_arrays = {}
        self.weight_totals = {}
        self.phase_counts = Counter()
        for task, ranges in self.ranges.items():
            indices = []
            values = []
            for start, stop in ranges:
                indices.extend(range(start, stop))
                values.extend(float(weights[i]) for i in range(start, stop))
            require(values and all(math.isfinite(v) and v > 0 for v in values),
                    f'Invalid phase weights for {task}')
            cumulative = []
            total = 0.0
            for value in values:
                total += value
                cumulative.append(total)
            self.weight_arrays[task] = (indices, cumulative)
            self.weight_totals[task] = total

    def __iter__(self):
        from bisect import bisect_left
        rng = random.Random(f'{self.seed}:{self.epoch}')
        self.epoch += 1
        remaining = self.num_samples
        while remaining:
            tasks = self.tasks.copy()
            rng.shuffle(tasks)
            for task in tasks[:remaining]:
                indices, cumulative = self.weight_arrays[task]
                draw = rng.random() * cumulative[-1]
                position = bisect_left(cumulative, draw)
                position = min(position, len(indices) - 1)
                index = indices[position]
                self.counts[task] += 1
                self.unique.add(index)
                remaining -= 1
                yield index

    def report(self):
        base = super().report()
        base.update(allocation='shuffled equal-task cycles; bounded phase-weighted frames within each task',
                    phase_weight_totals={k: float(v) for k, v in sorted(self.weight_totals.items())})
        return base
