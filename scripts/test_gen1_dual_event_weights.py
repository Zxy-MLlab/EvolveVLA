import unittest
import numpy as np
from gen1_dual_event_weights import dual_event_weights


class DualEventTests(unittest.TestCase):
    def rows(self, lengths):
        result, offset = [], 0
        for n in lengths:
            result.append(dict(split='train', start=offset, stop=offset+n, frames=n))
            offset += n
        return result

    def test_direct_slice_reference(self):
        rng = np.random.default_rng(17)
        for lengths in ([1], [2, 49], [50, 51, 100]):
            a = rng.random((sum(lengths), 14))
            for horizon in (1, 10, 50):
                expected = []
                offset = 0
                for n in lengths:
                    grips = a[offset:offset+n, [6, 13]]
                    for frame in range(n):
                        event = np.any(np.abs(np.diff(grips[frame:frame+horizon], axis=0)) > .05)
                        late = frame >= np.ceil(.8 * max(n-1, 1))
                        expected.append(1 + float(event) + .5 * late)
                    offset += n
                np.testing.assert_array_equal(dual_event_weights(a, self.rows(lengths), horizon), expected)

    def test_closing_opening_symmetry_and_no_mutation(self):
        a = np.ones((150, 14)); a[60:, 6] = 0
        b = a.copy(); b[:, [6, 13]] = 1-a[:, [6, 13]]
        saved = a.copy()
        w = dual_event_weights(a, self.rows([150]))
        np.testing.assert_array_equal(w, dual_event_weights(b, self.rows([150])))
        np.testing.assert_array_equal(a, saved)
        self.assertEqual(w[10], 1)
        self.assertEqual(w[11], 2)
        self.assertEqual(w[59], 2)
        self.assertEqual(w[60], 1)

    def test_no_cross_episode_event(self):
        a = np.ones((20, 14)); a[10:, 6] = 0
        w = dual_event_weights(a, self.rows([10, 10]))
        expected = np.ones(20); expected[[8, 9, 18, 19]] = 1.5
        np.testing.assert_array_equal(w, expected)

    def test_overlaps_bounded(self):
        a = np.zeros((100, 14)); a[::2, [6, 13]] = 1
        w = dual_event_weights(a, self.rows([100]))
        self.assertLessEqual(w.max(), 2.5)
        self.assertEqual(w[-1], 1.5)

    def test_invalid_inputs(self):
        a = np.ones((10, 14))
        bad_rows = [[], self.rows([9]), [dict(split='val', start=0, stop=10, frames=10)]]
        for rows in bad_rows:
            with self.assertRaises(ValueError): dual_event_weights(a, rows)
        for h in (0, -1, 1.5):
            with self.assertRaises(ValueError): dual_event_weights(a, self.rows([10]), h)
        a[0, 0] = np.nan
        with self.assertRaises(ValueError): dual_event_weights(a, self.rows([10]))


if __name__ == '__main__':
    unittest.main()
