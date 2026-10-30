import copy
import unittest
import audit_guard


class GuardAuditTest(unittest.TestCase):
    def fixture(self):
        shared = {'guard_id': 'g', 'gpu_uuid': 'GPU-x'}
        active = {'guard_run_id': 'g:1', 'phase': 'screen_timing', 'workload_id': 'n16_b1',
                  'candidate_id': 'c', 'round': 0, 'order': 0}
        row = dict(record_type='execution', guard_id='g', guard_gpu_uuid='GPU-x', guard_clean=True,
                   guard_run_id='g:1', command=['/bin/test'], **{k: active[k] for k in ('phase', 'workload_id', 'candidate_id', 'round', 'order')})
        sample = dict(shared, event='sample', active_run=active, ok=True, gpu={'uuid': 'GPU-x'},
                      violation_reasons=[], foreign_compute_pids=[], monitor_error=None)
        events = [dict(shared, event='guard_start', own_signal_method='pidfd_with_rechecked_PPID_exe_starttime'),
                  dict(shared, event='run_one_start', active_run=active), dict(sample, boundary='before'),
                  dict(shared, event='own_harness_registered', active_run=active, pidfd_retained=True, identity={'exe': '/bin/test'}),
                  dict(sample, boundary='after'), dict(shared, event='run_one_end', guard_run_id='g:1'),
                  dict(shared, event='guard_complete', clean=True, success=True)]
        return [row], events

    def test_complete_guard_passes(self):
        rows, events = self.fixture()
        self.assertEqual(audit_guard.audit(rows, events, [], 'GPU-x')['status'], 'pass')

    def test_guard_gap_or_incident_fails(self):
        rows, events = self.fixture()
        self.assertEqual(audit_guard.audit(rows, events[:-1], [], 'GPU-x')['status'], 'fail')
        self.assertEqual(audit_guard.audit(rows, events, [{'incident': True}], 'GPU-x')['status'], 'fail')
        bad = copy.deepcopy(rows); bad[0]['candidate_id'] = 'wrong'
        self.assertEqual(audit_guard.audit(bad, events, [], 'GPU-x')['status'], 'fail')


if __name__ == '__main__': unittest.main()
