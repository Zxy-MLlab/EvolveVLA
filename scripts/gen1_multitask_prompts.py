"""Source-grounded prompt augmentation restricted to audited train episodes."""
from collections import Counter
import hashlib
import json
import random

from gen1_multitask_data import digest, require
from gen1_prompt_augmentation import compatible


def prepare(rows):
    import h5py
    pools, report = {}, []
    for row in rows:
        require(row['split'] == 'train', 'Validation prompts cannot enter training')
        require(row['episode'] not in pools, 'Duplicate episode')
        require(digest(row['path']) == row['sha256'], 'Source hash changed')
        with h5py.File(row['path'], 'r') as stream:
            value = stream['instructions'][()]
        prompts = json.loads(value.decode() if isinstance(value, bytes) else value)
        require(isinstance(prompts, list) and all(isinstance(s, str) and s for s in prompts),
                'Invalid source instructions')
        retained = list(dict.fromkeys(s for s in prompts if compatible(row['task'], s)))
        require(bool(retained), 'No compatible source instructions')
        pools[row['episode']] = retained
        report.append(dict(episode=row['episode'], task=row['task'], seed=row['seed'],
                           source=row['path'], sha256=row['sha256'], retained=len(retained),
                           rejected=[s for s in prompts if not compatible(row['task'], s)]))
    return pools, dict(training_only=True, episodes=report, total_retained=sum(map(len, pools.values())),
                       rule='Existing compatible() rule; no generated strings; validation prompts unchanged')


class PromptDataset:
    def __init__(self, base, rows, pools, seed=42017, mode='frame'):
        self.base, self.pools = base, pools
        self.rows = {row['episode']: row for row in rows}
        require(all(row['split'] == 'train' for row in rows), 'Held-out data in prompt wrapper')
        require(set(self.rows) == set(pools), 'Prompt pool/episode mismatch')
        require(mode in ('frame', 'episode'), 'Unknown prompt sampling mode')
        self.rng = random.Random(seed)
        self.mode = mode
        # Episode mode removes within-trajectory language jitter while still
        # selecting from the audited source instruction pool.  The default
        # frame mode preserves the previous experiment exactly.
        self.episode_prompts = {
            episode: pool[int.from_bytes(hashlib.sha256(f'{seed}:{episode}'.encode()).digest()[:8], 'big') % len(pool)]
            for episode, pool in self.pools.items()
        }
        self.counts = Counter()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        episode = int(item['episode_index'])
        row = self.rows[episode]
        require(row['start'] <= index < row['stop'] and int(item['frame_index']) == index - row['start'],
                'Prompt/action episode binding differs from audit')
        item['prompt'] = (self.episode_prompts[episode] if self.mode == 'episode'
                          else self.rng.choice(self.pools[episode]))
        self.counts[episode] += 1
        return item
