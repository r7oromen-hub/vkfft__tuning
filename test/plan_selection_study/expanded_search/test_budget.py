import json
from pathlib import Path
import tempfile
import unittest
from budget import Budget, TOTAL


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.now = 100.0
        self.boot = 'boot-a'

    def make(self):
        return Budget(self.tmp.name, clock=lambda: self.now, get_boot=lambda: self.boot)

    def test_phase_deadline_restart_and_inflight(self):
        b = self.make(); b.start()
        self.assertTrue(b.may_start('candidate', expected_upper_seconds=10, candidate_id='a'))
        self.now += 5395
        b.finish_work('ok')
        self.assertFalse(b.may_start('candidate', expected_upper_seconds=10, candidate_id='near_deadline'))
        self.now += 6
        b = self.make()
        self.assertFalse(b.may_start('candidate', expected_upper_seconds=10, candidate_id='b'))
        b.checkpoint('inflight_completed', candidate_id='a')
        self.assertTrue(b.advance('phase_deadline'))
        self.assertEqual(b.state['phase'], 'search')
        self.assertTrue(b.may_start('candidate', expected_upper_seconds=10, candidate_id='c'))
        self.now = 100 + TOTAL + 1
        b.finish_work('ok')
        self.assertFalse(b.may_start('candidate', expected_upper_seconds=10, candidate_id='d'))
        self.assertFalse(b.advance('global_deadline'))

    def test_crash_between_ledger_and_snapshot_uses_ledger(self):
        b = self.make(); b.start()
        stale = b.path.read_text(encoding='utf8')
        self.now += 1; b.advance('early_complete')
        b.path.write_text(stale, encoding='utf8')
        self.assertEqual(self.make().state['phase'], 'search')

    def test_reboot_or_bad_ledger_fails_closed(self):
        b = self.make(); b.start(); self.boot = 'boot-b'
        with self.assertRaises(RuntimeError): self.make()

    def test_unresolved_inflight_recovery_never_resubmits(self):
        b = self.make(); b.start()
        self.assertTrue(b.may_start('pair', expected_upper_seconds=20, pair_id='p1'))
        self.now += 30
        b = self.make()
        with self.assertRaises(RuntimeError): b.may_start('pair', expected_upper_seconds=20, pair_id='p1')
        with self.assertRaises(RuntimeError): b.advance('interrupted')
        self.boot = 'boot-a'
        with b.ledger.open('a', encoding='utf8') as out: out.write('{broken\n')
        with self.assertRaises(RuntimeError): self.make()


if __name__ == '__main__':
    unittest.main()
