import copy
import json
from pathlib import Path
import tempfile
import unittest

from gen1_recovery_confirm import assess
from gen1_recovery_next_after_audit import checked_result


class NextExperimentGateTests(unittest.TestCase):
    def result(self):
        tasks = {}
        for task in ('place_mouse_pad', 'hanging_mug'):
            pairs = [dict(seed=400000+i, original=i == 0, candidate=i in (0, 1))
                     for i in range(10)]
            tasks[task] = dict(paired_completed=10, pairs=pairs,
                original_successes=1, candidate_successes=2,
                original_only=0, candidate_only=1, unpaired=[])
        result = dict(tasks=tasks)
        result['outcome_assessment'] = assess(result)
        return result

    def check(self, result):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'comparison.json'
            path.write_text(json.dumps(result))
            return checked_result(path)

    def test_expert_exclusions_are_allowed_but_mismatches_stop(self):
        result = self.result()
        unmatched = dict(seed=400010, original_status='expert_rejected',
                         candidate_status='completed', initial_equal=False)
        result['tasks']['hanging_mug']['unpaired'] = [unmatched]
        self.check(result)
        unmatched['original_status'] = 'completed'
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            self.check(result)

    def test_partial_duplicate_and_corrupt_counts_cannot_launch_next_job(self):
        valid = self.result()
        self.check(valid)
        for mutation in ('partial', 'duplicate', 'counts', 'assessment'):
            result = copy.deepcopy(valid)
            row = result['tasks']['hanging_mug']
            if mutation == 'partial':
                row['paired_completed'] = 9
            elif mutation == 'duplicate':
                row['pairs'][1]['seed'] = row['pairs'][0]['seed']
            elif mutation == 'counts':
                row['candidate_successes'] = 9
            else:
                result['outcome_assessment']['observed_gain_without_task_regression'] = False
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.check(result)


if __name__ == '__main__':
    unittest.main()
