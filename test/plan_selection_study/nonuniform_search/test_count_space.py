"""Small exhaustive checks of static counts, never device admission evidence."""
import json
import unittest

import candidate
import count_space


def without_profile(request):
    return json.dumps({k: v for k, v in request.items() if k != 'profile'},
                      sort_keys=True, separators=(',', ':'))


class CountSpaceTests(unittest.TestCase):
    def test_small_domains_match_full_enumerator_without_duplicates(self):
        for n in (2, 4, 8, 16, 32, 64):
            for batch in (1, 3, 16):
                with self.subTest(n=n, batch=batch):
                    rows = list(candidate.enumerate_candidates(n, batch))
                    ids = [row['candidate_id'] for row in rows]
                    self.assertEqual(len(ids), len(set(ids)))
                    by_profile = {p: [r['request'] for r in rows if r['request']['profile'] == p]
                                  for p in candidate.PROFILES}
                    active = by_profile['nonuniform_active_only']
                    compat = by_profile['uniform_compatibility']
                    uniform = sum(len({r['register_table'][stage] for stage in r['radix']}) == 1
                                  for r in active)
                    active_keys = {without_profile(r) for r in active}
                    compat_keys = {without_profile(r) for r in compat}
                    self.assertEqual(len(active_keys), len(active))
                    self.assertEqual(len(compat_keys), len(compat))
                    result = count_space.count_workload(n, batch)
                    self.assertEqual(result['active_only_uniform_requests'], uniform)
                    self.assertEqual(result['active_only_truly_nonuniform_requests'], len(active) - uniform)
                    self.assertEqual(result['active_only_total_requests'], len(active))
                    self.assertEqual(result['uniform_compatibility_requests'], len(compat))
                    self.assertEqual(result['profile_distinct_total_requests'], len(rows))
                    self.assertEqual(result['profile_ignored_exact_table_intersection'], len(active_keys & compat_keys))
                    self.assertEqual(result['profile_ignored_unique_requests'], len(active_keys | compat_keys))
                    self.assertEqual(result['ordered_decompositions'],
                                     len({tuple(r['request']['radix']) for r in rows}))
                    self.assertEqual(result['grouped_batch_values'], sorted({r['request']['grouped_batch'] for r in rows}))

    def test_repeated_radix_has_one_choice_and_inactive_entries_change_overlap(self):
        rows = [r['request'] for r in candidate.enumerate_candidates(8, 1)
                if r['request']['radix'] == [2, 2, 2]]
        active = [r for r in rows if r['profile'] == 'nonuniform_active_only']
        compat = [r for r in rows if r['profile'] == 'uniform_compatibility']
        self.assertEqual(len(active), 3)
        self.assertEqual(len(compat), 3)
        self.assertEqual(len({without_profile(r) for r in active} &
                             {without_profile(r) for r in compat}), 1)
        self.assertTrue(all(len({r['register_table'][x] for x in r['radix']}) == 1 for r in rows))

    def test_invalid_workloads_are_rejected(self):
        for n, batch in ((1, 1), (3, 1), (128, 0), (128, True), (16384, 1)):
            with self.subTest(n=n, batch=batch), self.assertRaises(ValueError):
                count_space.count_workload(n, batch)


if __name__ == '__main__':
    unittest.main()
