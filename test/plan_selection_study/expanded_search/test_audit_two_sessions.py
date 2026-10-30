import unittest
import audit_two_sessions as audit


class TwoSessionAuditTest(unittest.TestCase):
    def fixture(self):
        item={'n':128,'batch':1,'candidate_id':'c','role':'screen_faster','geometric_mean_ratio':0.97,
              'passes_practical_and_ci':True}
        first={'status':'pass','session_kind':'first','session_id':'one','complete':[item]}
        second={'status':'pass','session_kind':'second','session_id':'two','complete':[dict(item)]}
        gate={'schema':'expanded-first-session-gate/v1','status':'pass','session_id':'one','plan_sha256':'p',
              'audit_sha256':'a','evidence_sha256':{'rows':'first-rows'},'eligible':[dict(item)]}
        header={'session_id':'two','plan_sha256':'p','first_gate_sha256':'g','first_results_sha256':'first-rows'}
        return first,second,gate,header

    def test_two_distinct_complete_sessions(self):
        result=audit.combine(*self.fixture(),'g','p','a')
        self.assertEqual(result['status'],'pass')
        self.assertEqual(result['candidates'][0]['status'],'confirmed_under_preregistered_rule')

    def test_reused_session_and_control_promotion_fail(self):
        first,second,gate,header=self.fixture();second['session_id']='one'
        self.assertEqual(audit.combine(first,second,gate,header,'g','p','a')['status'],'fail')
        first,second,gate,header=self.fixture();gate['eligible'][0]['role']='default_faster_control'
        self.assertEqual(audit.combine(first,second,gate,header,'g','p','a')['status'],'fail')


if __name__=='__main__':unittest.main()
