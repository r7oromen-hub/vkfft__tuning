import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_search_results as subject


class SearchResultTest(unittest.TestCase):
    def test_separates_rejected_finished_and_budget_partial(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp);n,b=128,1
            selected=[{'record_type':'candidate_selected','n':n,'batch':b,'candidate_id':str(i),'request':{}} for i in range(3)]
            rows=[]
            for i in range(3):
                rows.append(selected[i]);rows.append({'record_type':'execution','phase':'candidate_describe','candidate_id':str(i),'workload':{'n':n,'batch':b}})
                rows.append({'record_type':'candidate_start','n':n,'batch':b,'candidate_id':str(i)})
                if i<2:rows.append({'record_type':'candidate_disposition','n':n,'batch':b,'candidate_id':str(i),
                                     'status':'screen_complete' if i==0 else 'scheduler_or_resource_rejected',**({'median_ratio':1.05} if i==0 else {})})
            rows.append({'record_type':'search_complete','status':'budget_exhausted','workload_statuses':[]})
            path=root/'rows';path.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf8')
            manifest=root/'n128_b1.jsonl';manifest.write_text(''.join(json.dumps({'record_type':'candidate_manifest','status':'pending_scheduler_and_device_validation',
                         'selection':'selected'})+'\n' for _ in range(4)),encoding='utf8')
            old={'selected_coverage':[{'n':n,'batch':b,'selected_count':4}],'counts':{},'cases':[]}
            with patch.object(subject.audit_old_space,'analyze',return_value=old):r=subject.summarize(path,root)
            self.assertEqual(r['manifest_selected_requests'],4);self.assertEqual(r['candidate_selected_events'],3)
            self.assertEqual(r['describe_execution_count'],3);self.assertEqual(r['selected_without_describe_execution'],1)
            self.assertEqual(r['requested_without_disposition'],1);self.assertEqual(r['candidate_dispositions']['scheduler_or_resource_rejected'],1)
            self.assertEqual(r['cases'][0]['fallback_best_screen_ratio'],1.0)


if __name__=='__main__':unittest.main()
