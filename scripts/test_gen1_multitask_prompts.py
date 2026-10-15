import hashlib
import random
import unittest

from gen1_multitask_prompts import PromptDataset


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.base = [dict(episode_index=0, frame_index=i, prompt='old', action=[i]) for i in range(3)]
        self.rows = [dict(episode=0, split='train', start=0, stop=3)]
        self.pools = {0: ['first', 'second']}

    def test_reproducible_prompts_preserve_actions_and_base(self):
        first = PromptDataset(self.base, self.rows, self.pools)
        second = PromptDataset(self.base, self.rows, self.pools)
        for index in [0, 1, 2] * 10:
            item = first[index]
            self.assertEqual(item, second[index])
            self.assertIn(item['prompt'], self.pools[0])
            self.assertEqual(item['action'], self.base[index]['action'])
            self.assertEqual(self.base[index]['prompt'], 'old')

    def test_rejects_heldout_and_wrong_pool(self):
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            PromptDataset(self.base, [dict(self.rows[0], split='val')], self.pools)
        with self.assertRaisesRegex(ValueError, 'pool/episode'):
            PromptDataset(self.base, self.rows, {1: ['foreign']})

    def test_rejects_frame_binding_mismatch(self):
        self.base[1]['frame_index'] = 2
        dataset = PromptDataset(self.base, self.rows, self.pools)
        with self.assertRaisesRegex(ValueError, 'binding'):
            dataset[1]

    def test_default_matches_original_rng_sequence(self):
        dataset = PromptDataset(self.base, self.rows, self.pools)
        rng = random.Random(42017)
        for index in [2, 0, 1] * 20:
            self.assertEqual(dataset[index]['prompt'], rng.choice(self.pools[0]))

    def test_episode_prompt_is_fixed_and_preserves_labels(self):
        dataset = PromptDataset(self.base, self.rows, self.pools, mode='episode')
        expected = self.pools[0][int.from_bytes(
            hashlib.sha256(b'42017:0').digest()[:8], 'big') % len(self.pools[0])]
        for index in [2, 0, 1] * 20:
            item = dataset[index]
            self.assertEqual(item['prompt'], expected)
            self.assertEqual(item['action'], self.base[index]['action'])
            self.assertEqual(self.base[index]['prompt'], 'old')

    def test_episode_choice_independent_of_pool_order_and_access_order(self):
        rows = self.rows + [dict(episode=1, split='train', start=3, stop=6)]
        base = self.base + [dict(episode_index=1, frame_index=i, action=[i])
                            for i in range(3)]
        pools = {0: ['first', 'second'], 1: ['third', 'fourth']}
        first = PromptDataset(base, rows, pools, mode='episode')
        second = PromptDataset(base, rows, dict(reversed(list(pools.items()))), mode='episode')
        forward = {i: first[i] for i in range(6)}
        backward = {i: second[i] for i in reversed(range(6))}
        self.assertEqual(forward, backward)
        for i, item in forward.items():
            self.assertIn(item['prompt'], pools[base[i]['episode_index']])

    def test_invalid_prompt_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown prompt'):
            PromptDataset(self.base, self.rows, self.pools, mode='invalid')


if __name__ == '__main__':
    unittest.main()
