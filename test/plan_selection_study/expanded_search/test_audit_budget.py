import copy
import json
from pathlib import Path
import tempfile
import unittest

import audit_budget
import budget


class BudgetAuditTest(unittest.TestCase):
    def fixture(self):
        tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(tmp.cleanup)
        now = [100.0]
        b = budget.Budget(tmp.name, clock=lambda: now[0], get_boot=lambda: 'boot')
        b.start()
        self.assertTrue(b.may_start('build', expected_upper_seconds=50, job='x'))
        now[0] += 5
        b.finish_work('ok')
        now[0] += 1
        b.advance('complete')
        rows = [json.loads(x) for x in b.ledger.read_text(encoding='utf8').splitlines()]
        protocol = {'stage_budget_seconds': dict(budget.PHASES)}
        return rows, protocol

    def test_real_budget_ledger_replays(self):
        rows, p = self.fixture()
        result = audit_budget.audit(rows, p)
        self.assertEqual(result['status'], 'pass', result['errors'])
        self.assertEqual(result['elapsed_seconds'], 6)

    def test_missing_start_or_tampered_reserve_fails(self):
        rows, p = self.fixture()
        self.assertEqual(audit_budget.audit(rows[1:], p)['status'], 'fail')
        bad = copy.deepcopy(rows)
        bad[1]['remaining_seconds'] = 999999
        self.assertEqual(audit_budget.audit(bad, p)['status'], 'fail')

    def test_invalid_boot_or_overlapping_work_fails(self):
        rows, p = self.fixture()
        bad = copy.deepcopy(rows)
        bad[1]['state']['boot_id'] = 'another'
        self.assertEqual(audit_budget.audit(bad, p)['status'], 'fail')
        bad = copy.deepcopy(rows)
        bad.insert(2, copy.deepcopy(bad[1]))
        self.assertEqual(audit_budget.audit(bad, p)['status'], 'fail')

    def test_unbudgeted_guard_execution_fails(self):
        rows, p = self.fixture()
        execution = [{'record_type': 'execution', 'guard_run_id': 'unbudgeted'}]
        self.assertEqual(audit_budget.audit(rows, p, execution)['status'], 'fail')

    def test_pair_run_ids_reconciled(self):
        rows, p = self.fixture()
        rows[1]['kind'] = rows[1]['state']['inflight']['kind'] = 'search_pair'
        rows[2]['kind'] = 'search_pair'
        rows[2]['guard_run_ids'] = ['left', 'right']
        execution = [{'record_type': 'execution', 'guard_run_id': x} for x in ('left', 'right')]
        self.assertEqual(audit_budget.audit(rows, p, execution)['status'], 'pass')
        self.assertEqual(audit_budget.audit(rows, p, execution[:1])['status'], 'fail')


if __name__ == '__main__':
    unittest.main()
