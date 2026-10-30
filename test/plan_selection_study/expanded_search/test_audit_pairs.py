import math
import unittest

from audit_pairs import ten_round_session, passes_session_criterion


class PairAuditTest(unittest.TestCase):
    def test_constant_ratio_and_protocol_criterion(self):
        result = ten_round_session([(0.97, 1.0)] * 10)
        self.assertAlmostEqual(result['geometric_mean_ratio'], 0.97)
        self.assertAlmostEqual(result['ci95_low'], 0.97)
        self.assertAlmostEqual(result['ci95_high'], 0.97)
        self.assertTrue(passes_session_criterion(result))
        self.assertFalse(passes_session_criterion(ten_round_session([(0.99, 1.0)] * 10)))

    def test_bad_pairs_rejected(self):
        with self.assertRaises(ValueError): ten_round_session([(1.0, 1.0)] * 9)
        with self.assertRaises(ValueError): ten_round_session([(math.nan, 1.0)] * 10)
        with self.assertRaises(ValueError): ten_round_session([(0.0, 1.0)] * 10)


if __name__ == '__main__':
    unittest.main()
