import copy
import hashlib
import itertools
import unittest

import candidate


def request(n=128, batch=16, radix=None, values=None, group=1,
            profile='nonuniform_active_only'):
    radix = [8, 8, 2] if radix is None else radix
    values = {2: 4, 8: 8} if values is None else values
    table = [0] * 33
    for index, value in values.items():
        table[index] = value
    return {'schema': candidate.SCHEMA, 'workload': {'n': n, 'batch': batch},
            'radix': radix, 'register_table': table, 'profile': profile,
            'grouped_batch': group}


class CandidateTests(unittest.TestCase):
    def test_active_min_max_and_thread_dimension(self):
        req = request()
        self.assertEqual(candidate.validate_request(req),
                         {'min_registers': 4, 'max_registers': 8, 'planned_fft_threads': 32})
        record = candidate.candidate_record(req)
        self.assertEqual(record['status'], candidate.STATUS)
        self.assertIn('not scheduler', record['validity'])
        req['register_table'][2] = 8
        self.assertEqual(record['request']['register_table'][2], 4)

    def test_counterexamples_and_strict_fields_are_rejected(self):
        bad = []
        for index, value in ((8, 4), (2, 0), (2, 6), (8, 256), (8, True)):
            item = request(); item['register_table'][index] = value; bad.append(item)
        for group in (3, 0, True, 32):
            item = request(); item['grouped_batch'] = group; bad.append(item)
        for stages in ([8, 8], [8, 8, 3], [True, 8, 2], [32] * 33):
            item = request(); item['radix'] = stages; bad.append(item)
        for name, value in (('schema', 'other/v1'), ('profile', 'implicit'),
                            ('register_table', [0] * 32), ('workload', {'n': 128, 'batch': True})):
            item = request(); item[name] = value; bad.append(item)
        item = request(); item['stage_registers'] = [8, 16, 4]; bad.append(item)
        item = request(); item['register_table'] = {2: 4, 8: 8}; bad.append(item)
        item = request(); item['workload']['n'] = 2 ** 100; bad.append(item)
        for item in bad:
            with self.subTest(item=item), self.assertRaises(ValueError):
                candidate.validate_request(item)

    def test_repeated_radix_has_one_shared_value(self):
        req = request()
        self.assertEqual([req['register_table'][r] for r in req['radix']], [8, 8, 4])
        impossible = copy.deepcopy(req)
        impossible['stage_registers'] = [8, 16, 4]
        with self.assertRaises(ValueError):
            candidate.validate_request(impossible)
        for order in ([8, 8, 2], [2, 8, 8], [8, 2, 8]):
            changed = copy.deepcopy(req); changed['radix'] = order
            self.assertEqual(candidate.validate_request(changed)['planned_fft_threads'], 32)

    def test_environment_contains_full_versioned_workload_and_table(self):
        req = request(); env = candidate.request_environment(req)
        self.assertEqual(env['VKFFT_NONUNIFORM_VERSION'], candidate.SCHEMA)
        self.assertEqual(env['VKFFT_NONUNIFORM_N'], '128')
        self.assertEqual(env['VKFFT_NONUNIFORM_WORKLOAD_BATCH'], '16')
        self.assertEqual(env['VKFFT_NONUNIFORM_RADIX'], '8,8,2')
        self.assertEqual([int(x) for x in env['VKFFT_NONUNIFORM_TABLE'].split(',')], req['register_table'])
        self.assertEqual(env['VKFFT_NONUNIFORM_PROFILE'], req['profile'])
        self.assertEqual(env['VKFFT_NONUNIFORM_GROUP'], '1')
        self.assertEqual(len(env), 7)

    def test_inactive_slots_and_uniform_compatibility(self):
        active = request(); active['register_table'][4] = 4
        with self.assertRaises(ValueError):
            candidate.validate_request(active)
        uniform = request(values={2: 8, 4: 8, 8: 8}, profile='uniform_compatibility')
        self.assertEqual(candidate.validate_request(uniform)['min_registers'], 8)
        for index, value in ((4, 0), (16, 8), (8, 16), (0, 8)):
            changed = copy.deepcopy(uniform); changed['register_table'][index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                candidate.validate_request(changed)
        for wide in (128, 256):
            table = {r: wide for r in candidate.RADICES if wide % r == 0}
            req = request(n=256, radix=[16, 16], values=table, profile='uniform_compatibility')
            self.assertEqual(candidate.validate_request(req)['max_registers'], wide)

    def test_canonical_identity_is_stable_and_includes_all_dimensions(self):
        req = request()
        reordered = {key: copy.deepcopy(req[key]) for key in reversed(list(req))}
        reordered['workload'] = {'batch': 16, 'n': 128}
        self.assertEqual(candidate.canonical_request(req), candidate.canonical_request(reordered))
        self.assertEqual(candidate.candidate_id(req), candidate.candidate_id(reordered))
        self.assertEqual(candidate.candidate_id(req), hashlib.sha256(
            candidate.canonical_request(req).encode('utf8')).hexdigest())
        variants = []
        item = copy.deepcopy(req); item['radix'] = [2, 8, 8]; variants.append(item)
        item = copy.deepcopy(req); item['register_table'][2] = 8; variants.append(item)
        item = copy.deepcopy(req); item['grouped_batch'] = 2; variants.append(item)
        item = copy.deepcopy(req); item['workload']['batch'] = 32; variants.append(item)
        item = copy.deepcopy(req); item['workload']['n'] = 256; item['radix'].append(2); variants.append(item)
        ids = {candidate.candidate_id(x) for x in [req, *variants]}
        self.assertEqual(len(ids), 1 + len(variants))
        same_table_a = request(n=8, batch=1, radix=[2, 2, 2], values={2: 2})
        same_table_b = copy.deepcopy(same_table_a); same_table_b['profile'] = 'uniform_compatibility'
        self.assertEqual(same_table_a['register_table'], same_table_b['register_table'])
        self.assertNotEqual(candidate.candidate_id(same_table_a), candidate.candidate_id(same_table_b))

    def test_enumeration_is_lazy_unique_and_complete_at_small_boundaries(self):
        iterator = candidate.enumerate_candidates(2, 65536)
        self.assertIs(iter(iterator), iterator)
        rows = list(iterator)
        self.assertEqual(len(rows), 34)
        self.assertEqual(len({row['candidate_id'] for row in rows}), len(rows))
        self.assertEqual({row['request']['grouped_batch'] for row in rows},
                         {2 ** i for i in range(17)})
        self.assertEqual(len(list(candidate.enumerate_candidates(4, 3))), 6)
        first = list(itertools.islice(candidate.enumerate_candidates(8192, 65536), 100))
        self.assertEqual(len(first), 100)
        self.assertEqual(len({r['candidate_id'] for r in first}), 100)
        self.assertTrue(all(r['status'] == candidate.STATUS for r in first))
        self.assertEqual(first, list(itertools.islice(candidate.enumerate_candidates(8192, 65536), 100)))

    def test_enumerator_rejects_invalid_inputs_and_profile_lists(self):
        for n, batch in ((1, 1), (8193, 1), (True, 1), (128, 0), (128, 65537), (128, True)):
            with self.subTest(n=n, batch=batch), self.assertRaises(ValueError):
                candidate.enumerate_candidates(n, batch)
        for profiles in ((), ('bad',), ('nonuniform_active_only', 'nonuniform_active_only'), 'uniform_compatibility'):
            with self.subTest(profiles=profiles), self.assertRaises(ValueError):
                candidate.enumerate_candidates(128, 16, profiles)


if __name__ == '__main__':
    unittest.main()
