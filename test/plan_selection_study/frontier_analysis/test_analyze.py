import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import analyze


def candidate_id(request):
    return hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def fixture():
    specs=[(128,16,[8,16],16,1,.90),(128,16,[16,8],16,1,.99),
           (128,16,[8,16],32,1,1.10),(128,16,[8,16],128,1,.97),
           (128,16,[8,16],16,2,1.03),(256,1,[16,16],16,1,1.02)]
    contract={'gpu':'GPU-test','execution_commit':'a'*40}
    rows=[{'record_type':'session_start','contract':contract},
          {'record_type':'workload_status','n':128,'batch':16,'status':'identity_passed',
           'request':{'radix':[8,16],'logical_registers':16,'grouped_batch':1}},
          {'record_type':'workload_status','n':256,'batch':1,'status':'identity_passed',
           'request':{'radix':[16,16],'logical_registers':16,'grouped_batch':1}}]; cases=[]; indices={}
    for n,batch,radix,r,group,ratio in specs:
        request={'radix':radix,'logical_registers':r,'grouped_batch':group};cid=candidate_id(request)
        key=(n,batch);index=indices.get(key,0);indices[key]=index+1
        cases.append({'n':n,'batch':batch,'candidate_id':cid,'request':request,
                      'selection_index':index,'status':'screen_complete','median_ratio':ratio})
        rows.append({'record_type':'candidate_start','n':n,'batch':batch,'candidate_id':cid,'request':request})
        for round_no in range(3):
            for order,name in enumerate(('original',cid)):
                result={'exec_ms_samples':[1.0 if name=='original' else ratio],
                        'structural_features':{'kernels':[]},'compiled_diagnostics':[]}
                rows.append({'record_type':'execution','phase':'timing','candidate_id':name,'round':round_no,
                             'order':order,'workload':{'n':n,'batch':batch},'result':result})
            rows.append({'record_type':'screen_pair','n':n,'batch':batch,'candidate_id':cid,
                         'round':round_no,'ratio':ratio})
        rows.append({'record_type':'candidate_disposition','n':n,'batch':batch,'candidate_id':cid,
                     'request':request,'status':'screen_complete','pair_ratios':[ratio]*3,'median_ratio':ratio})
    unseen={'radix':[8,16],'logical_registers':256,'grouped_batch':1}
    cases.append({'n':128,'batch':16,'candidate_id':candidate_id(unseen),'request':unseen,
                  'selection_index':5,'status':'selected_not_started'})
    for row in rows: row.setdefault('contract',contract)
    coverage={'schema':'frontier-screen-coverage/v1','status':'pass','gpu':'GPU-test','execution_commit':'a'*40,
              'guard_audit':{'status':'pass'},
              'cases':cases,'counts':{'selected':len(cases),'screen_complete':len(specs),
                                     'selected_not_started':1,'partial':0,'scheduler_rejected':0,'resource_rejected':0},
              'evidence_sha256':{'/remote/search/rows.jsonl':'e'*64}}
    return coverage,rows,'e'*64


class AnalysisTests(unittest.TestCase):
    def test_register_prototype_denominator_counts_static_requests_only(self):
        self.assertEqual(analyze.prototype_static_supported(128,1,(2,4,8,16,32,64)),244)
        self.assertEqual(analyze.prototype_static_supported(128,1,(128,256)),61)
        self.assertEqual(analyze.prototype_static_supported(128,16,(128,256)),305)

    def test_cli_binds_raw_file_hash_and_writes_report(self):
        coverage,rows,_=fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);raw=root/'custom-screen.jsonl';audit=root/'coverage.json';output=root/'analysis.json'
            raw.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf8')
            coverage['evidence_sha256']={'/remote/custom-screen.jsonl':analyze.sha(raw)}
            audit.write_text(json.dumps(coverage),encoding='utf8')
            run=subprocess.run([sys.executable,str(Path(analyze.__file__)), '--coverage',str(audit),
                                '--rows',str(raw),'--output',str(output),'--seed','7'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            report=json.loads(output.read_text(encoding='utf8'))
            self.assertEqual(report['input_sha256']['rows'],analyze.sha(raw))
            self.assertEqual(report['overall']['screen_complete_count'],6)

    def test_fractions_use_only_screen_complete_candidates(self):
        coverage,rows,digest=fixture();result=analyze.analyze(coverage,rows,digest,42)
        overall=result['overall']
        self.assertEqual(overall['screen_complete_count'],6)
        self.assertEqual(overall['faster_count'],3)
        self.assertAlmostEqual(overall['faster_fraction'],3/6)
        self.assertEqual(overall['at_least_2pct_count'],2)
        self.assertAlmostEqual(overall['at_least_2pct_fraction'],2/6)
        workload=next(w for w in result['workloads'] if (w['n'],w['batch'])==(128,16))
        self.assertEqual(workload['outcomes']['screen_complete_count'],5)
        self.assertAlmostEqual(workload['best_screened']['ratio'],.9)
        slower=next(w for w in result['workloads'] if (w['n'],w['batch'])==(256,1))
        self.assertEqual(slower['best_with_default']['kind'],'default')
        self.assertEqual(slower['best_with_default']['ratio'],1.0)
        for space in result['register_subspaces'].values():
            self.assertIn('n',space['best_screened']);self.assertIn('batch',space['best_screened'])

    def test_register_subspace_counts_distinguish_unmeasured(self):
        coverage,rows,digest=fixture();result=analyze.analyze(coverage,rows,digest,42)
        old=result['register_subspaces']['r_le_64'];wide=result['register_subspaces']['r_128_256']
        self.assertEqual((old['selected_count'],old['screen_complete_count']),(5,5))
        self.assertEqual((wide['selected_count'],wide['screen_complete_count'],wide['selected_not_started']),(2,1,1))
        self.assertAlmostEqual(wide['completion_fraction'],.5)
        self.assertEqual(wide['scheduler_rejected']+wide['resource_rejected'],0)
        self.assertEqual(next(x for x in wide['by_workload'] if (x['n'],x['batch'])==(128,16))['prototype_static_supported_count'],305)

    def test_matched_groups_preserve_workload_and_fixed_dimensions(self):
        coverage,rows,digest=fixture();matched=analyze.analyze(coverage,rows,digest,42)['matched']
        self.assertEqual((matched['fixed_r_group_radix']['group_count'],matched['fixed_r_group_radix']['pair_count']),(1,1))
        self.assertEqual((matched['fixed_radix_layout']['group_count'],matched['fixed_radix_layout']['pair_count']),(1,6))
        self.assertEqual((matched['fixed_radix_r_group']['group_count'],matched['fixed_radix_r_group']['pair_count']),(1,1))
        self.assertEqual((matched['fixed_radix_group_r']['group_count'],matched['fixed_radix_group_r']['pair_count']),(1,3))

    def test_replay_is_deterministic_and_default_is_fallback(self):
        coverage,rows,digest=fixture()
        first=analyze.analyze(coverage,rows,digest,999)['replay']
        self.assertEqual(first,analyze.analyze(coverage,rows,digest,999)['replay'])
        self.assertEqual(first['seed'],999)
        for workload in first['workloads']:
            points=workload['points'];self.assertEqual(points[0]['step'],0);self.assertEqual(points[0]['best_ratio'],1.0)
            self.assertTrue(all(b['best_ratio']<=a['best_ratio'] for a,b in zip(points,points[1:])))
        slower=next(w for w in first['workloads'] if (w['n'],w['batch'])==(256,1))
        self.assertEqual(slower['points'][-1]['best_ratio'],1.0)

    def test_rejects_mismatched_evidence_duplicate_or_extra_disposition(self):
        coverage,rows,digest=fixture()
        with self.assertRaises(ValueError):analyze.analyze(coverage,rows,'f'*64,1)
        bad=copy.deepcopy(rows);bad.append(copy.deepcopy(next(x for x in rows if x.get('record_type')=='candidate_disposition')))
        with self.assertRaises(ValueError):analyze.analyze(coverage,bad,digest,1)
        bad=copy.deepcopy(rows);item=copy.deepcopy(next(x for x in rows if x.get('record_type')=='candidate_disposition'))
        item['candidate_id']='f'*64;bad.append(item)
        with self.assertRaises(ValueError):analyze.analyze(coverage,bad,digest,1)

    def test_unstarted_workload_does_not_imply_observed_default(self):
        coverage,rows,digest=fixture()
        request={'radix':[8,8,8],'logical_registers':8,'grouped_batch':1}
        coverage['cases'].append({'n':512,'batch':1,'candidate_id':candidate_id(request),'request':request,
                                  'selection_index':0,'status':'selected_not_started'})
        coverage['counts']['selected']+=1;coverage['counts']['selected_not_started']+=1
        report=analyze.analyze(coverage,rows,digest,1)
        work=next(w for w in report['workloads'] if (w['n'],w['batch'])==(512,1))
        self.assertIsNone(work['best_screened']);self.assertIsNone(work['best_with_default'])
        self.assertEqual(work['default_status'],'not_observed')

    def test_unsupported_identity_preserves_observed_default(self):
        coverage,rows,digest=fixture()
        request={'radix':[8,8,8],'logical_registers':8,'grouped_batch':1}
        coverage['cases'].append({'n':512,'batch':1,'candidate_id':candidate_id(request),'request':request,
                                  'selection_index':0,'status':'selected_not_started',
                                  'not_started_reason':'unsupported_identity'})
        coverage['counts']['selected']+=1;coverage['counts']['selected_not_started']+=1
        rows.append({'record_type':'workload_status','n':512,'batch':1,'status':'unsupported_identity',
                     'reason':'nonuniform_register_table','contract':rows[0]['contract']})
        work=next(w for w in analyze.analyze(coverage,rows,digest,1)['workloads'] if (w['n'],w['batch'])==(512,1))
        self.assertIsNone(work['best_screened'])
        self.assertEqual(work['default_status'],'unsupported_identity')
        self.assertEqual(work['best_with_default']['kind'],'default')
        self.assertEqual(work['best_with_default']['ratio'],1.0)


if __name__=='__main__':unittest.main()
