import copy
import json
from pathlib import Path
import tempfile
import unittest

from gen1_recovery_compare import summarize
from gen1_recovery_confirm import assess
from gen1_recovery_native_after_confirmation import verified_confirmation


class NativeControlPrerequisiteTests(unittest.TestCase):
    def artifacts(self):
        records = [dict(task=task, seed=500000+i, status='completed',
            success=i == 0, hashes={'observation': str(i)}, binding={'prompt': 'same'})
            for task in ('place_mouse_pad', 'hanging_mug') for i in range(10)]
        original = dict(records=records)
        candidate = copy.deepcopy(original)
        result = summarize(original, candidate)
        result.update(checkpoint='/model', development_panel_reused=False)
        result['outcome_assessment'] = assess(result)
        return {'original.json': original, 'candidate.json': candidate,
            'comparison.json': result, 'manifest.json': dict(seed_start=500000,
                checkpoint='/model', target_pairs_per_task=10, development_panel_reused=False)}

    def check(self, artifacts):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in artifacts.items():
                (root/name).write_text(json.dumps(value))
            return verified_confirmation(root, 500000, Path('/model'))

    def test_complete_raw_confirmation_passes(self):
        self.check(self.artifacts())

    def test_raw_input_outcome_and_duplicate_errors_stop_training(self):
        for error in ('hash', 'outcome', 'duplicate'):
            artifacts = self.artifacts()
            records = artifacts['candidate.json']['records']
            if error == 'hash':
                records[0]['hashes']['observation'] = 'different'
            elif error == 'outcome':
                records[0]['success'] = False
            else:
                records.append(copy.deepcopy(records[0]))
            with self.subTest(error=error), self.assertRaises(RuntimeError):
                self.check(artifacts)

    def test_wrong_seeds_checkpoint_or_reused_panel_stops_training(self):
        for key, value in (('seed_start', 400000), ('checkpoint', '/other'),
                           ('development_panel_reused', True)):
            artifacts = self.artifacts()
            artifacts['manifest.json'][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.check(artifacts)


if __name__ == '__main__':
    unittest.main()
