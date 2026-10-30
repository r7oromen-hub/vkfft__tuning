import unittest
from unittest.mock import patch
import audit_confirmation


class ConfirmationAuditTest(unittest.TestCase):
    def fixture(self):
        plan={'sha256':{'original':'o','expanded':'e'},'selection_seed':20261030,'workloads':[]}
        rows=[{'record_type':'confirmation_protocol','session_id':'a','session_kind':'first','plan_sha256':'plan',
               'sha256':plan['sha256'],'gpu':'GPU-x','rounds':10},
              {'record_type':'confirmation_complete','session_id':'a','status':'completed'}]
        return rows,plan

    def test_empty_completed_session_has_no_speedup(self):
        rows,plan=self.fixture()
        with patch.object(audit_confirmation.audit_guard,'audit',return_value={'errors':[]}):
            result=audit_confirmation.audit(rows,[],[],plan,'plan')
        self.assertEqual(result['status'],'pass')
        self.assertEqual(result['complete'],[])

    def test_missing_terminal_or_bad_plan_fails(self):
        rows,plan=self.fixture()
        self.assertEqual(audit_confirmation.audit(rows[:-1],[],[],plan,'plan')['status'],'fail')
        with patch.object(audit_confirmation.audit_guard,'audit',return_value={'errors':[]}):
            self.assertEqual(audit_confirmation.audit(rows,[],[],plan,'wrong')['status'],'fail')

    def test_reserved_for_second_is_a_valid_terminal(self):
        rows,plan=self.fixture();rows[-1]['status']='reserved_for_second'
        with patch.object(audit_confirmation.audit_guard,'audit',return_value={'errors':[]}):
            self.assertEqual(audit_confirmation.audit(rows,[],[],plan,'plan')['status'],'pass')


if __name__=='__main__':unittest.main()
