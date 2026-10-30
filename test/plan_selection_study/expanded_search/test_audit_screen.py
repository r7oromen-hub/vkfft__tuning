import copy
import hashlib
import json
import random
import unittest
import audit_screen


class ScreenAuditTest(unittest.TestCase):
    def fixture(self):
        rows = []
        for round_no in range(3):
            payload = json.dumps([20261030,16,1,'abc','screen_timing',round_no,'random',42],separators=(',',':')).encode()
            order = ['original', 'candidate']
            random.Random(int.from_bytes(hashlib.sha256(payload).digest(),'big')).shuffle(order)
            for index, role in enumerate(order):
                duration = 1.0 if role == 'original' else 1.1
                cid = role if role == 'original' else 'abc'
                rows.append({'record_type': 'execution', 'guard_run_id': f'{role}{round_no}',
                    'candidate_id': cid, 'phase': 'screen_timing', 'round': round_no, 'order': index,
                    'returncode': 0, 'status': 'ok',
                    'workload': {'n': 16, 'batch': 1, 'seed': 42, 'pattern': 'random', 'repeats': 31,
                                 'warmup': 5, 'warmup_ms': 1000, 'graph_repeats': 2},
                    'result': {'status': 'ok', 'warmup_actual': 5, 'warmup_wall_ms': 1001, 'graph_repeats': 2, 'graph_kernel_count': 2, 'graph_event_record_count': 2,
                               'exec_ms_samples': [duration]*31, 'raw_interval_ms_samples': [duration*2]*31}})
            rows.append({'record_type': 'screen_pair', 'n': 16, 'batch': 1, 'candidate_id': 'abc',
                         'round': round_no, 'order': order,
                         'original_guard_run_id': f'original{round_no}', 'candidate_guard_run_id': f'candidate{round_no}',
                         'original_median_ms': 1.0, 'candidate_median_ms': 1.1, 'candidate_over_original': 1.1})
        rows.append({'record_type': 'candidate_disposition', 'n': 16, 'batch': 1, 'candidate_id': 'abc',
                     'status': 'screen_complete', 'pair_ratios': [1.1]*3, 'median_ratio': 1.1})
        return rows

    def test_ratio_recomputed_and_default_fallback(self):
        result = audit_screen.audit(self.fixture())
        self.assertEqual(result['status'], 'pass', result['errors'])
        self.assertEqual(result['workload_best_screen_only'][0]['screen_selected'], 'default')
        self.assertEqual(result['workload_best_screen_only'][0]['screen_fallback_ratio'], 1.0)

    def test_missing_pair_and_forged_median_fail(self):
        rows = self.fixture()
        bad = [r for r in rows if not (r.get('record_type') == 'screen_pair' and r['round'] == 1)]
        self.assertEqual(audit_screen.audit(bad)['status'], 'fail')
        bad = copy.deepcopy(rows)
        next(r for r in bad if r.get('record_type') == 'screen_pair')['candidate_median_ms'] = 0.5
        self.assertEqual(audit_screen.audit(bad)['status'], 'fail')


if __name__ == '__main__': unittest.main()
