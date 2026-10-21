"""CPU tests for action-window isolation, early stopping, and pair eligibility."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import gen1_execution_horizon_ablation_20261019 as ablation


class ExecutionWindowTest(unittest.TestCase):
    def rollout(self, prefix, limit=17, terminate_at=None):
        actions, observations, inferences = [], [], []
        class Policy:
            def infer_chunk(self, *, images, state, prompt):
                observations.append(int(state[0]))
                return np.full((50, 14), len(observations), dtype=np.float32)
        def receive():
            step = len(actions)
            return dict(type='step', observation=step,
                        terminated=step == terminate_at, truncated=False,
                        info={'is_success': step == terminate_at})
        result = ablation.execution_loop(Policy(),
            dict(observation=0, diagnostic_binding={'prompt': 'original'}), prefix,
            actions.append, receive, lambda obs: ({}, np.array([obs])),
            lambda record, images: inferences.append(record), action_limit=limit)
        return result, actions, observations, inferences

    def test_replans_from_latest_observation_and_preserves_action_cap(self):
        for prefix, expected in [(50, [0]), (10, [0, 10]), (5, [0, 5, 10, 15])]:
            with self.subTest(prefix=prefix):
                result, actions, observations, inferences = self.rollout(prefix)
                self.assertEqual(observations, expected)
                self.assertEqual(len(actions), 17)
                self.assertEqual(result['steps'], 17)
                self.assertEqual(result['inference_count'], len(expected))
                self.assertTrue(all(len(row['actions']) == 50 for row in inferences))
                self.assertEqual(sum(row['execute_prefix'] for row in inferences), 17)

    def test_stops_immediately_on_simulator_success(self):
        result, actions, observations, _ = self.rollout(5, terminate_at=11)
        self.assertEqual(len(actions), 11)
        self.assertEqual(observations, [0, 5, 10])
        self.assertTrue(result['success'])
        self.assertEqual(result['inference_count'], 3)

    def test_nonfinite_chunk_sends_no_actions(self):
        class Policy:
            def infer_chunk(self, **kwargs):
                return np.full((50, 14), np.nan)
        sent = []
        with self.assertRaises(ValueError):
            ablation.execution_loop(Policy(), dict(observation=0, diagnostic_binding={'prompt': 'x'}),
                5, sent.append, lambda: None, lambda obs: ({}, np.array([obs])), lambda *a: None)
        self.assertEqual(sent, [])


class IdentityTest(unittest.TestCase):
    def test_mismatch_and_exclusion_do_not_enter_paired_denominator(self):
        rows = []
        for name in ('match', 'mismatch', 'excluded'):
            for prefix in (50, 10, 5):
                rows.append(dict(id=name, task='hanging_mug', execute_prefix=prefix,
                    status='expert_rejected' if name == 'excluded' and prefix == 5 else 'completed',
                    success=prefix != 50, hashes={'image': 'equal'},
                    initial_physical_sha256='different' if name == 'mismatch' and prefix == 5 else 'equal'))
        summary = ablation.summarize(rows, [50, 10, 5])
        self.assertEqual(summary['matched_case_ids'], ['match'])
        self.assertEqual(summary['initial_mismatch_case_ids'], ['mismatch'])
        self.assertEqual(summary['paired_vs_50']['10']['valid_pairs'], 1)
        self.assertEqual(summary['paired_vs_50']['10']['prefix_only_success'], 1)

    def test_source_prompt_and_collection_index_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / 'episode_0000007'
            (episode / 'instruction').mkdir(parents=True)
            source = episode / 'source.hdf5'
            source.write_bytes(b'fixed-source')
            (episode / 'instruction' / 'episode_0000007.json').write_text(json.dumps({'seen': ['source prompt']}))
            row = dict(id='hanging_mug_2', task='hanging_mug', seed=2, episode_index=7,
                prompt='source prompt', source_episode_directory=str(episode), source_hdf5=str(source),
                source_sha256=ablation.digest(source))
            panel = root / 'panel.json'
            panel.write_text(json.dumps({'cases': [row]}))
            self.assertEqual(ablation.load_cases(panel), [row])
            row['prompt'] = 'changed prompt'
            panel.write_text(json.dumps({'cases': [row]}))
            with self.assertRaises(ValueError):
                ablation.load_cases(panel)


if __name__ == '__main__':
    unittest.main()
