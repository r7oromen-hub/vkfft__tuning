import copy
import unittest

import search_costs


def fixture():
    rows_hash, budget_hash = 'a'*64, 'b'*64
    contract = {'gpu': 'GPU-test', 'execution_commit': 'c'*40, 'timeout': 30}
    coverage = {'schema': 'frontier-screen-coverage/v1', 'status': 'pass',
                'gpu': contract['gpu'], 'execution_commit': contract['execution_commit'],
                'guard_audit': {'status': 'pass'},
                'evidence_sha256': {'/screen/rows.jsonl': rows_hash, '/screen/budget.jsonl': budget_hash},
                'cases': [{'n': 128, 'batch': 16, 'candidate_id': 'candidate',
                           'request': {'radix': [8, 16], 'logical_registers': 16, 'grouped_batch': 1},
                           'status': 'partial', 'selection_index': 0}]}
    rows = [{'record_type': 'session_start', 'contract': copy.deepcopy(contract)}]
    def identity(phase, candidate):
        return {'n': 128, 'batch': 16, 'candidate_id': candidate, 'phase': phase, 'round': 0}
    def state(last, inflight=None):
        return {'contract': copy.deepcopy(contract), 'seconds': 30, 'boot': 'boot-one',
                'start': 100.0, 'last': float(last), 'inflight': copy.deepcopy(inflight)}
    ledger = [{'event': 'start', 'state': state(100)}]
    for phase, candidate, begin, end, wall in [('identity', 'original', 101, 103, 1.5),
                                               ('validation', 'candidate', 106, 110, 3.5),
                                               ('timing', 'candidate', 115, 121, 5.0)]:
        item = identity(phase, candidate)
        if phase == 'validation':
            rows.append({'record_type': 'candidate_start', 'contract': copy.deepcopy(contract),
                         'n': 128, 'batch': 16, 'candidate_id': candidate,
                         'request': copy.deepcopy(coverage['cases'][0]['request'])})
        ledger.append({'event': 'work_start', 'state': state(begin, item)})
        ledger.append({'event': 'work_finish', 'state': state(end), 'identity': item, 'status': 'ok'})
        rows.append({'record_type': 'execution', 'contract': copy.deepcopy(contract),
                     'workload': {'n': 128, 'batch': 16}, 'candidate_id': candidate,
                     'phase': phase, 'round': 0, 'status': 'ok', 'wall_seconds': wall})
    denied = identity('timing', 'candidate')
    ledger.append({'event': 'denied', 'state': state(125), 'identity': denied})
    rows.append({'record_type': 'budget_exhausted', 'contract': copy.deepcopy(contract)})
    return coverage, rows, ledger, rows_hash, budget_hash


def paired_fixture(include_pair):
    coverage, rows, ledger, rows_hash, budget_hash = fixture()
    candidate = next(r for r in rows if r.get('phase') == 'timing')
    candidate['order'] = 0
    original = copy.deepcopy(candidate)
    original.update(candidate_id='original', order=1, wall_seconds=2.5)
    rows.insert(-1, original)
    if include_pair:
        rows.insert(-1, {'record_type': 'screen_pair', 'contract': copy.deepcopy(rows[0]['contract']),
                         'n': 128, 'batch': 16, 'candidate_id': 'candidate', 'round': 0})
    item = {'n': 128, 'batch': 16, 'candidate_id': 'original', 'phase': 'timing', 'round': 0}
    state = copy.deepcopy(ledger[-2]['state'])
    state.update(last=121.2, inflight=copy.deepcopy(item))
    ledger.insert(-1, {'event': 'work_start', 'state': state})
    state = copy.deepcopy(state)
    state.update(last=124.2, inflight=None)
    ledger.insert(-1, {'event': 'work_finish', 'state': state, 'identity': item, 'status': 'ok'})
    return coverage, rows, ledger, rows_hash, budget_hash


class SearchCostsTests(unittest.TestCase):
    def test_elapsed_work_gap_and_phase_partition(self):
        result = search_costs.analyze_costs(*fixture())
        self.assertEqual(result['interval']['start_monotonic'], 100)
        self.assertEqual(result['interval']['end_monotonic'], 125)
        self.assertEqual(result['interval']['elapsed_seconds'], 25)
        self.assertEqual(result['interval']['work_seconds'], 12)
        self.assertEqual(result['interval']['gap_seconds'], 13)
        self.assertEqual(result['phase_seconds']['identity'], 2)
        self.assertEqual(result['phase_seconds']['validation'], 4)
        self.assertEqual(result['phase_seconds']['timing'], 6)
        self.assertIn('candidate_costs', result)
        self.assertIn('unattributed', result)

    def test_only_uniquely_paired_original_is_attributed(self):
        result = search_costs.analyze_costs(*paired_fixture(True))
        candidate = result['candidate_costs'][0]
        self.assertAlmostEqual(candidate['own_work_seconds'], 10)
        self.assertAlmostEqual(candidate['paired_original_timing_seconds'], 3)
        self.assertAlmostEqual(candidate['attributed_work_seconds'], 13)
        self.assertEqual(result['unattributed']['unpaired_original_timing_seconds'], 0)
        self.assertAlmostEqual(candidate['observed_span']['seconds'], 18.2)
        self.assertAlmostEqual(candidate['observed_span']['between_work_seconds'], 5.2)

    def test_incomplete_pair_keeps_original_timing_unattributed(self):
        result = search_costs.analyze_costs(*paired_fixture(False))
        candidate = result['candidate_costs'][0]
        self.assertEqual(candidate['own_work_seconds'], 10)
        self.assertEqual(candidate['paired_original_timing_seconds'], 0)
        self.assertEqual(candidate['attributed_work_seconds'], 10)
        self.assertAlmostEqual(result['unattributed']['unpaired_original_timing_seconds'], 3)

    def test_source_hash_or_contract_mismatch_rejected(self):
        coverage, rows, ledger, rows_hash, budget_hash = fixture()
        for bad in [(coverage, rows, ledger, 'f'*64, budget_hash),
                    (coverage, rows, ledger, rows_hash, 'f'*64),
                    (coverage, rows, ledger, rows_hash, rows_hash)]:
            with self.subTest(args=bad[-2:]), self.assertRaises(ValueError):
                search_costs.analyze_costs(*bad)
        bad = copy.deepcopy(rows)
        bad[1]['contract']['gpu'] = 'GPU-other'
        with self.assertRaises(ValueError):
            search_costs.analyze_costs(coverage, bad, ledger, rows_hash, budget_hash)
        bad = copy.deepcopy(coverage)
        bad['execution_commit'] = 'd'*40
        with self.assertRaises(ValueError):
            search_costs.analyze_costs(bad, rows, ledger, rows_hash, budget_hash)

    def test_boot_changed_or_time_moves_backwards_rejected(self):
        coverage, rows, ledger, rows_hash, budget_hash = fixture()
        bad = copy.deepcopy(ledger)
        bad[2]['state']['boot'] = 'boot-two'
        with self.assertRaises(ValueError):
            search_costs.analyze_costs(coverage, rows, bad, rows_hash, budget_hash)
        bad = copy.deepcopy(ledger)
        bad[2]['state']['last'] = 100.5
        with self.assertRaises(ValueError):
            search_costs.analyze_costs(coverage, rows, bad, rows_hash, budget_hash)

    def test_overlapping_or_unfinished_work_rejected(self):
        coverage, rows, ledger, rows_hash, budget_hash = fixture()
        bad = copy.deepcopy(ledger)
        bad.insert(2, copy.deepcopy(bad[3]))
        bad[2]['state']['last'] = 102
        with self.assertRaises(ValueError):
            search_costs.analyze_costs(coverage, rows, bad, rows_hash, budget_hash)
        bad = copy.deepcopy(ledger[:-2])
        with self.assertRaises(ValueError):
            search_costs.analyze_costs(coverage, rows[:-1], bad, rows_hash, budget_hash)

    def test_execution_cannot_exceed_its_ledger_interval(self):
        coverage, rows, ledger, rows_hash, budget_hash = fixture()
        for value in (2.00001, -1, float('inf')):
            with self.subTest(value=value):
                bad = copy.deepcopy(rows)
                bad[1]['wall_seconds'] = value
                with self.assertRaises(ValueError):
                    search_costs.analyze_costs(coverage, bad, ledger, rows_hash, budget_hash)

    def test_execution_identity_and_status_must_match_ledger(self):
        coverage, rows, ledger, rows_hash, budget_hash = fixture()
        for field, value in (('candidate_id', 'wrong'), ('status', 'failed'), ('round', 1)):
            with self.subTest(field=field):
                bad=copy.deepcopy(rows)
                bad[1][field]=value
                with self.assertRaises(ValueError):
                    search_costs.analyze_costs(coverage, bad, ledger, rows_hash, budget_hash)


if __name__ == '__main__':
    unittest.main()
