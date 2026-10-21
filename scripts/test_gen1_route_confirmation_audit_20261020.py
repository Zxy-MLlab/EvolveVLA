import unittest
from unittest.mock import patch

import gen1_route_confirmation_audit_20261020 as audit


class RouteAuditTests(unittest.TestCase):
    def pair(self, seed, a, b, valid=True, exclusions=None):
        return dict(seed=seed, baseline_success=a, candidate_success=b, valid_pair=valid, exclusions=exclusions or [])

    def test_exact_interval_edges(self):
        self.assertIsNone(audit.interval(0, 0))
        self.assertAlmostEqual(audit.interval(0, 10)[1], 0.3084971078187608)
        self.assertAlmostEqual(audit.interval(10, 10)[0], 0.6915028921812392)

    def test_pending_success_and_rejection_not_failures(self):
        self.assertEqual(audit.classification({'status': 'completed', 'success': True}, False), 'pending')
        self.assertEqual(audit.classification({'status': 'expert_rejected', 'success': False}, True), 'expert_rejected')
        self.assertEqual(audit.classification({'status': 'completed', 'success': True, 'simulator_returncode': 1}, True), 'invalid_terminal_record')

    def test_provisional_rule_requires_complete_panel_and_eight_pairs(self):
        p = [self.pair(i, False, i == 0) for i in range(8)]
        p += [self.pair(8, None, None, False, ['expert_rejected']), self.pair(9, None, None, False, ['pending'])]
        self.assertEqual(audit.summarize(p)['route_recommendation'], 'pending_complete_fixed_panel')
        p[-1]['exclusions'] = ['expert_rejected']
        self.assertEqual(audit.summarize(p)['route_recommendation'], 'provisional_candidate_recommendation')
        p[7].update(valid_pair=False, exclusions=['expert_rejected'])
        self.assertEqual(audit.summarize(p)['route_recommendation'], 'keep_baseline_original')

    def test_one_baseline_only_failure_blocks_route_even_with_net_gain(self):
        p = [self.pair(0, True, False)] + [self.pair(i, False, i < 4) for i in range(1, 10)]
        r = audit.summarize(p)
        self.assertGreater(r['candidate_successes'], r['baseline_successes'])
        self.assertEqual(r['route_recommendation'], 'keep_baseline_original')

    def test_physical_mismatch_excluded_even_when_pixels_match(self):
        a = dict(task='hanging_mug', seed=1, episode_index=0, outcome='failure', success=False,
                 eligible=True, input_hashes={'state': 'same'}, binding={'prompt': 'same'}, initial_physical={'x': 0})
        b = dict(a, outcome='success', success=True, initial_physical={'x': 1})
        r = audit.pair(a, b)
        self.assertFalse(r['valid_pair'])
        self.assertIn('initial_physical_mismatch_or_missing', r['exclusions'])

    def test_manifest_missing_or_runtime_change_cannot_pass(self):
        protocol = dict(checkpoint_identities={'baseline': 'identity'}, checkpoint_paths={'baseline': '/weights'})
        self.assertEqual(audit.manifest_checks({}, protocol, 'baseline', []), {'present': False})
        manifest = dict(checkpoint={'identity_sha256': 'identity', 'path': '/weights'}, cases=[],
            controller='original', reset_episode_index=0, execute_prefix=50, policy_seed=0,
            same_case_retries=0, expert_rejection_replenishment=False, source_sha256={'/code': 'hash'})
        with patch.object(audit, 'sha', return_value='hash'):
            self.assertTrue(all(audit.manifest_checks(manifest, protocol, 'baseline', []).values()))
            manifest['execute_prefix'] = 5
            self.assertFalse(audit.manifest_checks(manifest, protocol, 'baseline', [])['fixed_runtime'])


if __name__ == '__main__':
    unittest.main()
