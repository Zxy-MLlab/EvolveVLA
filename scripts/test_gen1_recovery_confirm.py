import copy
import unittest

from gen1_recovery_confirm import assess, next_cases


class ConfirmationTests(unittest.TestCase):
    def row(self, seed, status='completed', success=False):
        return dict(task='hanging_mug', seed=seed, status=status, success=success,
                    hashes={'camera': 'same'}, binding={'prompt': 'same'})

    def test_supplemental_schedule_is_independent_of_policy_outcomes(self):
        original = {'records': [self.row(400000), self.row(400001)]}
        candidate = {'records': [self.row(400000), self.row(400001, 'expert_rejected')]}
        expected = next_cases(original, candidate, 400000)
        changed = copy.deepcopy(original)
        for row in changed['records']:
            row['success'] = True
        self.assertEqual(expected, next_cases(changed, candidate, 400000))
        mug = [r['seed'] for r in expected if r['task'] == 'hanging_mug']
        self.assertEqual(mug, list(range(400002, 400011)))

    def test_initial_mismatch_and_infrastructure_errors_stop(self):
        original = {'records': [self.row(400000)]}
        candidate = copy.deepcopy(original)
        candidate['records'][0]['hashes']['camera'] = 'different'
        with self.assertRaisesRegex(RuntimeError, 'different initial'):
            next_cases(original, candidate, 400000)
        candidate['records'][0]['status'] = 'worker_error'
        with self.assertRaisesRegex(RuntimeError, 'Infrastructure'):
            next_cases(original, candidate, 400000)

    def test_missing_case_is_not_silently_replaced(self):
        with self.assertRaisesRegex(RuntimeError, 'Missing scheduled'):
            next_cases({'records': [self.row(400000)]}, {'records': []}, 400000)

    def test_descriptive_gain_is_distinct_from_significance(self):
        result = {'tasks': {'mouse': dict(original_only=0, candidate_only=1,
                  original_successes=1, candidate_successes=2, paired_completed=10)}}
        summary = assess(result)
        self.assertTrue(summary['observed_gain_without_task_regression'])
        self.assertEqual(summary['exact_two_sided_paired_p'], 1.0)


if __name__ == '__main__':
    unittest.main()
