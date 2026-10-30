import unittest
import audit_admission


class AdmissionAuditTest(unittest.TestCase):
    def fixture(self):
        protocol = {'gpu_uuid': 'GPU-test', 'workloads': [{'n': 16, 'batch': 1}],
                    'patterns': ['random'], 'validation_seeds': [42]}
        rows = [{'record_type': 'admission_protocol', 'protocol': protocol, 'sha256': {}},
                {'record_type': 'admission_complete', 'status': 'budget_exhausted'}]
        common = {'guard_id': 'g', 'gpu_uuid': 'GPU-test'}
        guard = [dict(common, event='guard_start', own_signal_method='pidfd_with_rechecked_PPID_exe_starttime'),
                 dict(common, event='guard_complete', success=True, clean=True)]
        return rows, guard, protocol

    def test_partial_run_is_incomplete(self):
        rows, guard, protocol = self.fixture()
        result = audit_admission.audit(rows, guard, [], protocol)
        self.assertEqual(result['status'], 'incomplete', result['errors'])
        self.assertEqual(result['missing_workloads'], [(16, 1)])

    def test_missing_guard_or_incident_fails(self):
        rows, guard, protocol = self.fixture()
        self.assertEqual(audit_admission.audit(rows, guard[:-1], [], protocol)['status'], 'fail')
        self.assertEqual(audit_admission.audit(rows, guard, [{'incident': True}], protocol)['status'], 'fail')

    def test_unregistered_success_and_claimed_coverage_fail(self):
        rows, guard, protocol = self.fixture()
        rows.insert(1, {'record_type': 'candidate_disposition', 'n': 16, 'batch': 1,
                        'candidate_id': 'not-run', 'status': 'numerically_valid', 'checks': 1})
        result = audit_admission.audit(rows, guard, [], protocol)
        self.assertEqual(result['status'], 'fail')
        self.assertTrue(any('coverage incomplete' in x for x in result['errors']))

    def test_protocol_mismatch_fails(self):
        rows, guard, protocol = self.fixture()
        rows[0]['protocol'] = {'gpu_uuid': 'different'}
        self.assertEqual(audit_admission.audit(rows, guard, [], protocol)['status'], 'fail')

    def test_identity_uses_precompile_axis_orientation(self):
        regs = [8 if i in (2, 4, 8) else 0 for i in range(33)]
        result = {'n': 128, 'batch': 4096,
                  'structural_features': {'numAxisUploads': 1, 'kernels': [
                      {'stageRadix': [8, 8, 2], 'groupedBatch': 8, 'localSize': [8, 16, 1]}]},
                  'precompile_kernels': [{'raw_descriptor': {'logical_registers_per_thread': 8,
                      'min_logical_registers_per_thread': 8, 'registerBoost': 1,
                      'registers_per_radix': regs, 'axisSwapped': 1}}]}
        self.assertEqual(audit_admission.default_identity(result),
                         {'radix': [8, 8, 2], 'logical_registers': 8, 'grouped_batch': 8})
        result['precompile_kernels'][0]['raw_descriptor']['axisSwapped'] = 0
        self.assertIsNone(audit_admission.default_identity(result))


if __name__ == '__main__':
    unittest.main()
