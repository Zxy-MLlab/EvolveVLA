"""CPU-only equal-task, equal-source replay sampler for concatenated datasets.

Rows use global indices in the concatenated training dataset. Every complete
cycle visits each (task, source) exactly once. Frames within that stratum are
uniform; phase oversampling is deliberately not bundled with replay.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import random
from math import lcm


class ReplaySampler:
    def __init__(self, rows, dataset_length, num_samples, seed=0, allow_local_subset=False):
        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError('Positive integer sample count required')
        if type(dataset_length) is not int or dataset_length <= 0:
            raise ValueError('Positive integer dataset length required')
        self.ranges = defaultdict(list)
        offset = 0
        identities = set()
        for row in sorted(rows, key=lambda r: r['start']):
            if row['split'] != 'train':
                raise ValueError('Held-out trajectory in training sampler')
            if row['source'] not in ('official', 'local'):
                raise ValueError('Unknown replay source')
            start, stop = row['start'], row['stop']
            if (type(start) is not int or type(stop) is not int or
                    start != offset or stop <= start or stop-start != row['frames']):
                raise ValueError('Ranges must cover training data exactly once')
            identity = (row['source'], row['episode'])
            if identity in identities:
                raise ValueError('Duplicate source episode')
            identities.add(identity)
            self.ranges[(row['task'], row['source'])].append((start, stop))
            offset = stop
        if offset != dataset_length:
            raise ValueError('Dataset length differs from audited ranges')
        tasks = {task for task, _ in self.ranges}
        if not allow_local_subset and set(self.ranges) != {(task, source) for task in tasks
                               for source in ('official', 'local')}:
            raise ValueError('Each task requires both official and local training data')
        self.ends = {}
        for key, ranges in self.ranges.items():
            ends, total = [], 0
            for start, stop in ranges:
                total += stop-start
                ends.append(total)
            self.ends[key] = ends
        self.keys = sorted(self.ranges)
        self.cycle_keys = self.keys.copy()
        if allow_local_subset:
            local = [k for k in self.keys if k[1] == 'local']
            official = [k for k in self.keys if k[1] == 'official']
            if not local or not official or not {k[0] for k in local} <= {k[0] for k in official}:
                raise ValueError('Local tasks must be a nonempty subset of official tasks')
            size = lcm(len(local), len(official))
            self.cycle_keys = local * (size // len(local)) + official * (size // len(official))
        self.num_samples, self.seed, self.epoch = num_samples, seed, 0
        self.counts = Counter()

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = random.Random(f'{self.seed}:{self.epoch}')
        self.epoch += 1
        remaining = self.num_samples
        while remaining:
            keys = self.cycle_keys.copy()
            rng.shuffle(keys)
            for key in keys[:remaining]:
                draw = rng.randrange(self.ends[key][-1])
                episode = bisect_right(self.ends[key], draw)
                before = self.ends[key][episode-1] if episode else 0
                index = self.ranges[key][episode][0] + draw-before
                self.counts[key] += 1
                remaining -= 1
                yield index

    def report(self):
        return dict(samples=sum(self.counts.values()), epochs_started=self.epoch,
                    allocation='equal source allocation; equal tasks within each source; uniform frames within stratum',
                    official_fraction_target=0.5,
                    per_task_source=[dict(task=t, source=s, samples=self.counts[t,s])
                                     for t,s in self.keys])
