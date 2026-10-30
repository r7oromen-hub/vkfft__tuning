import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import audit_search


class SearchIntegrityTest(unittest.TestCase):
    def test_missing_protocol_cannot_pass(self):
        result = audit_search.audit([], [], [], '.')
        self.assertEqual(result['status'], 'fail')

    def test_missing_candidate_or_original_evidence_fails(self):
        self.assertFalse(audit_search.numerical({}, {'radix':[4,4], 'logical_registers':4, 'grouped_batch':1}, 'GPU-x'))
        self.assertFalse(audit_search.original_numerical({}, 'GPU-x'))

    def test_budget_interrupted_identity_and_partial_manifest_are_uncovered(self):
        header={'record_type':'search_protocol','protocol':{'gpu_uuid':'GPU-x','batches':[1],'lengths':[16]},'sha256':{}}
        terminal={'record_type':'search_complete','status':'budget_exhausted','workload_statuses':[
            {'n':16,'batch':1,'status':'identity_pending_or_incomplete'}]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            (Path(temp)/'n16_b1.jsonl').write_text('{partial',encoding='utf8')
            with patch.object(audit_search.audit_guard,'audit',return_value={'errors':[]}), \
                 patch.object(audit_search.audit_screen,'audit',return_value={'errors':[]}):
                result=audit_search.audit([header,terminal],[],[],temp)
        self.assertEqual(result['status'],'pass',result['errors'])
        self.assertEqual(result['counts']['terminal_identity_pending_or_incomplete'],1)
        self.assertTrue(result['incomplete_workloads'][0]['partial_manifest_present'])


if __name__ == '__main__': unittest.main()
