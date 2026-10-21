"""Admission tests prevent missing/rejected/drifted cases inflating paired results."""
import copy
import unittest

import gen1_control_repair_progress_audit_20261020 as audit


class AdmissionTests(unittest.TestCase):
    def result(self, **updates):
        r = dict(status='completed', success=False, simulator_returncode=0,
                 timing_mode='original', timing_summary={})
        r.update(updates)
        return r

    def condition(self, **updates):
        r = dict(id='case', outcome='failure', protocol_eligible=True, success=False,
            initial_physical={'object': [0, 1, 2]}, input_hashes={'state': 'same'},
            checkpoint_identity='weights', checkpoint='/fixed', original_transparency={'status': 'exact'})
        r.update(updates)
        return r

    def test_missing_unfinalized_and_rejected_are_never_failures(self):
        self.assertEqual(audit.classify(None, 'policy'), 'pending')
        self.assertEqual(audit.classify({'status': 'completed', 'success': False}, 'policy'), 'pending')
        self.assertEqual(audit.classify(self.result(status='expert_rejected'), 'policy'), 'expert_rejected')

    def test_nonzero_exit_and_nonboolean_outcome_excluded(self):
        for change in [dict(simulator_returncode=1), dict(simulator_returncode=None), dict(success=0)]:
            self.assertEqual(audit.classify(self.result(**change), 'policy'), 'invalid_terminal_record')

    def test_source_exhausted_is_expert_failure_only(self):
        r = self.result(status='source_exhausted')
        self.assertEqual(audit.classify(r, 'expert'), 'failure')
        self.assertEqual(audit.classify(r, 'policy'), 'infrastructure_or_protocol_error')
        r['success'] = True
        self.assertEqual(audit.classify(r, 'expert'), 'invalid_terminal_record')

    def test_source_audit_and_physical_drift_exclude_pair(self):
        a = self.condition()
        self.assertIn('condition_protocol_exclusion', audit.pair_admission(a, self.condition(protocol_eligible=False), 'policy'))
        self.assertIn('initial_physical_mismatch_or_missing', audit.pair_admission(a, self.condition(initial_physical={'object': [0, 1, 3]}), 'policy'))

    def test_different_checkpoint_or_input_excluded(self):
        a = self.condition()
        self.assertIn('checkpoint_mismatch_or_missing', audit.pair_admission(a, self.condition(checkpoint_identity='different'), 'policy'))
        self.assertIn('initial_observation_hash_mismatch_or_missing', audit.pair_admission(a, self.condition(input_hashes={}), 'policy'))

    def test_archived_cross_process_mismatch_does_not_override_current_pair(self):
        a = self.condition(original_transparency={'status': 'cross_run_mismatch'})
        self.assertEqual(audit.pair_admission(a, self.condition(), 'policy'), [])

    def test_current_full_prediction_difference_excludes_pair(self):
        a = self.condition(task='place_mouse_pad', mode='original', _physical=[{'x': 0}, {'x': 1}],
            _targets=[[0]], _ticks=[15], _predictions=[{'state': [0], 'actions': [[0], [1]]}],
            first_active_action=None, actual_physics_ticks=15)
        b = copy.deepcopy(a)
        b.update(mode='fixed15', first_active_action=0)
        b['_predictions'][0]['actions'][1] = [2]
        r = audit.pair_report(a, b, 'policy')
        self.assertFalse(r['valid_pair'])
        self.assertIn('current_run_first_prediction_mismatch', r['exclusions'])

    def test_one_sided_success_does_not_enter_denominator(self):
        pairs = [dict(id='valid', valid_pair=True, original_success=False, modified_success=True, exclusions=[]),
                 dict(id='rejected', valid_pair=False, original_success=None, modified_success=True, exclusions=['expert_rejected']),
                 dict(id='missing', valid_pair=False, original_success=True, modified_success=None, exclusions=['pending'])]
        r = audit.paired_counts(pairs)
        self.assertEqual((r['valid_pairs'], r['original_successes'], r['modified_successes']), (1, 0, 1))
        self.assertEqual(r['improved_ids'], ['valid'])

    def test_active_short_duration_requires_success_censor(self):
        row = dict(action_index=0, physics_steps=7, total_physics_steps=7, eval_success_latched=False,
            duration_control=dict(qpos_hook_reached=True, mode='fixed15', active=True, scheduled_physics_ticks=15, arms={}))
        self.assertIn('active_duration_not_15_or_valid_success_censor', audit.timing_checks([row], 1, False, 'fixed15'))
        row['eval_success_latched'] = True
        self.assertEqual(audit.timing_checks([row], 1, True, 'fixed15'), [])


if __name__ == '__main__':
    unittest.main()
