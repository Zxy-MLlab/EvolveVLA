import copy
from collections import Counter
import unittest

from gen1_multitask_data import REVISION, TaskBalancedSampler, ordered_episodes, validate_split


def fixture():
    rows = [dict(task=t, split=s, path=f'/source/{t}/{s}', sha256=f'{t}-{s}',
                 seed=i * 2 + j, frames=(i + 1) * 5, source_revision=REVISION,
                 success=True, plan_success=True, replay_success=True)
            for i, t in enumerate(('a', 'b', 'c')) for j, s in enumerate(('train', 'val'))]
    manifest = dict(status='passed', tasks=['a', 'b', 'c'], episodes=rows,
                    per_task_counts={'train': 1, 'val': 1}, counts={'train': 3, 'val': 3})
    sources = [{k: r[k] for k in ('path', 'sha256', 'frames')}
               for r in reversed(rows) if r['split'] == 'train']
    audit = dict(status='passed', all_frames_decoded=True, fps=15, state_action_dim=14,
                 sources=sources, episodes=3, frames=30)
    return manifest, audit


class DataTests(unittest.TestCase):
    def test_expanded_per_task_counts(self):
        manifest, _ = fixture()
        added = dict(manifest['episodes'][0], path='/source/a/extra',
                     sha256='a-extra', seed=100, frames=7)
        manifest['episodes'].append(added)
        manifest['counts']['train'] = 4
        manifest['per_task_counts'] = {
            'train': {'a': 2, 'b': 1, 'c': 1},
            'val': {'a': 1, 'b': 1, 'c': 1},
        }
        self.assertEqual(len(validate_split(manifest)), 7)
        manifest['per_task_counts']['train']['b'] = 2
        with self.assertRaisesRegex(ValueError, 'per-task split'):
            validate_split(manifest)

    def test_converter_order_not_manifest_order(self):
        rows = ordered_episodes(*fixture(), 'train')
        self.assertEqual([r['task'] for r in rows], ['c', 'b', 'a'])
        self.assertEqual([(r['start'], r['stop']) for r in rows], [(0, 15), (15, 25), (25, 30)])

    def test_balanced_reproducible_and_in_range(self):
        rows = ordered_episodes(*fixture(), 'train')
        sampler = TaskBalancedSampler(rows, 3000, seed=9)
        first = list(sampler)
        self.assertEqual(first, list(TaskBalancedSampler(rows, 3000, seed=9)))
        self.assertNotEqual(first, list(sampler))
        self.assertEqual(sampler.report()['per_task_samples'], {'a': 2000, 'b': 2000, 'c': 2000})
        self.assertTrue(all(0 <= index < 30 for index in first))
        self.assertEqual(len(set(first)), 30)
        counts = Counter(first)
        # Different trajectory lengths must not distort equal task allocation.
        self.assertEqual(sum(counts[i] for i in range(15)), 1000)
        self.assertEqual(sum(counts[i] for i in range(25, 30)), 1000)

    def test_partial_cycle(self):
        rows = ordered_episodes(*fixture(), 'train')
        sampler = TaskBalancedSampler(rows, 8)
        self.assertEqual(len(list(sampler)), 8)
        self.assertEqual(sorted(sampler.counts.values()), [2, 3, 3])

    def test_reject_leakage_and_unaccepted(self):
        manifest, audit = fixture()
        for key in ('path', 'seed', 'sha256'):
            altered = copy.deepcopy(manifest)
            altered['episodes'][1][key] = altered['episodes'][0][key]
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                validate_split(altered)
        audit['all_frames_decoded'] = False
        with self.assertRaisesRegex(ValueError, 'fully audited'):
            ordered_episodes(manifest, audit, 'train')

    def test_reject_foreign_or_duplicate_audit_source(self):
        for foreign in (True, False):
            manifest, audit = fixture()
            audit['sources'][0] = (dict(path='/heldout', sha256='x', frames=15) if foreign
                                   else audit['sources'][1].copy())
            with self.assertRaisesRegex(ValueError, 'foreign'):
                ordered_episodes(manifest, audit, 'train')

    def test_reject_validation_sampler(self):
        rows = ordered_episodes(*fixture(), 'train')
        rows[0]['split'] = 'val'
        with self.assertRaisesRegex(ValueError, 'Validation'):
            TaskBalancedSampler(rows, 9)

    def test_reject_range_gap(self):
        rows = ordered_episodes(*fixture(), 'train')
        rows[1]['start'] += 1
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            TaskBalancedSampler(rows, 9)


if __name__ == '__main__':
    unittest.main()
