import unittest
import enumerate_candidates
import selection


class SelectionTest(unittest.TestCase):
    def test_determinism_cap_identity_and_strata(self):
        identity = {'radix':[16,16], 'logical_registers':16, 'grouped_batch':1}
        all_a, a = selection.select(enumerate_candidates.candidates(256, 16), identity)
        all_b, b = selection.select(enumerate_candidates.candidates(256, 16), identity)
        self.assertEqual([x['candidate_id'] for x in a], [x['candidate_id'] for x in b])
        self.assertEqual(len(a), 64)
        self.assertTrue(all(x['request'] != identity for x in a))
        self.assertTrue(all(x['status'] != 'rejected_static' for x in a))
        self.assertGreater(len({len(x['request']['radix']) for x in a}), 1)
        self.assertEqual(len(all_a), len(all_b))

    def test_rejects_duplicate_ids(self):
        rows = list(enumerate_candidates.candidates(128, 1))
        with self.assertRaises(ValueError): selection.select(rows + rows[:1], {})


if __name__ == '__main__': unittest.main()
