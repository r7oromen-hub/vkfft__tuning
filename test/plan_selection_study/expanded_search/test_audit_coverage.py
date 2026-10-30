import unittest
import audit_coverage


class CoverageTest(unittest.TestCase):
    def fixture(self):
        protocol = {'workloads': [{'n': 16, 'batch': 1}, {'n': 32, 'batch': 4096}]}
        rows = []
        for n, b, cid, base, radix, regs, group, swap in [
                (16, 1, 'a', [4, 4], [2, 8], 8, 1, 0),
                (32, 4096, 'b', [8, 4], [4, 4, 2], 4, 4, 1)]:
            rows.extend([
                {'record_type': 'workload_status', 'n': n, 'batch': b, 'status': 'admission_exercised'},
                {'record_type': 'execution', 'candidate_id': 'expanded_default', 'returncode': 0,
                 'workload': {'n': n, 'batch': b}, 'result': {'structural_features': {'kernels': [{'stageRadix': base}]}}},
                {'record_type': 'execution', 'candidate_id': cid, 'returncode': 0,
                 'workload': {'n': n, 'batch': b, 'study_mode': 'describe'},
                 'result': {'precompile_kernels': [{'raw_descriptor': {'axisSwapped': swap}}]}},
                {'record_type': 'candidate_disposition', 'n': n, 'batch': b, 'candidate_id': cid,
                 'request': {'radix': radix, 'logical_registers': regs, 'grouped_batch': group}, 'status': 'numerically_valid'}])
        return rows, protocol

    def test_multiple_candidates_supply_coverage(self):
        rows, p = self.fixture()
        result = audit_coverage.coverage(rows, p, 'pass')
        self.assertEqual(result['status'], 'ready_for_review')
        self.assertTrue(all(result['required_dimensions'].values()))
        self.assertFalse(result['observed_dimensions']['tested_R2'])
        self.assertFalse(result['observed_dimensions']['tested_R64'])

    def test_integrity_and_missing_coverage_block(self):
        rows, p = self.fixture()
        self.assertEqual(audit_coverage.coverage(rows, p, 'fail')['status'], 'coverage_incomplete')
        rows = [r for r in rows if r.get('candidate_id') != 'b']
        self.assertEqual(audit_coverage.coverage(rows, p, 'pass')['status'], 'coverage_incomplete')


if __name__ == '__main__':
    unittest.main()
