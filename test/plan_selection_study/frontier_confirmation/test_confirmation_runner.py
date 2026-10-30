import hashlib
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frontier_confirmation_runner',HERE/'runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
evidence=runner.evidence


class RunnerTests(unittest.TestCase):
    def run_case(self, fail=False):
        tmp=tempfile.TemporaryDirectory(dir=HERE);self.addCleanup(tmp.cleanup);base=Path(tmp.name)
        req={'radix':[8,16],'logical_registers':16,'grouped_batch':1}
        cid=hashlib.sha256(json.dumps(req,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        signature=evidence.signature_hash('same')
        proof={'cases':[dict(n=128,batch=16,candidate_id=cid,request=req,
                            screen_candidate_signature_sha256=signature,screen_original_signature_sha256=signature)]}
        contract={'binaries':{'original':'a','expanded':'b'},'plan_sha256':'d'*64}
        args=SimpleNamespace(gpu='GPU-test',output=base/'out',original=base/'o',expanded=base/'e',
                             timeout=10,session_kind='first')
        calls=[]
        class Ledger:
            def start(self,*a): return True
            def finish(self,*a): pass
        def one(args,work,candidate,phase,round_no,order):
            calls.append((phase,candidate[0],work['pattern'],work['seed'],round_no,order))
            result={'raw_interval_ms_samples':[2.0],'warmup_actual':5,'warmup_wall_ms':1000,
                    'exec_ms_samples':[1.0]}
            return {'record_type':'execution','status':'ok','result':result,'workload':work,
                    'candidate_id':candidate[0],'phase':phase,'round':round_no,'order':order,
                    'guard_run_id':f'{phase}-{round_no}-{order}-{candidate[0]}'}
        def check(row,*a,**kw):
            if fail and row['phase']=='confirmation_timing' and row['candidate_id']!='original': raise ValueError('numerical failure')
            return row['result']
        with mock.patch.dict(os.environ,{'VKFFT_STUDY_GUARD_UUID':'GPU-test','VKFFT_STUDY_GUARD_ID':'g'}), \
             mock.patch.object(runner.suite,'run_one',side_effect=one), \
             mock.patch.object(runner.validation,'check',side_effect=check), \
             mock.patch.object(runner.validation,'signature',return_value='same'), \
             mock.patch.object(runner.comparison,'compare'), \
             mock.patch.object(runner.frontier,'representable',return_value=(req,None)), \
             mock.patch.object(runner.search_controller,'source_signature',return_value='same'):
            if fail:
                with self.assertRaisesRegex(ValueError,'numerical failure'):runner.run_session(args,Ledger(),contract,proof)
            else:runner.run_session(args,Ledger(),contract,proof)
        return calls,runner.frontier.read_log(args.output)

    def test_ten_new_pairs_all_random_validated(self):
        calls,rows=self.run_case();timings=[x for x in calls if x[0]=='confirmation_timing']
        self.assertEqual(len(timings),20)
        self.assertTrue(all(x[2:4]==('random',42) for x in timings))
        self.assertEqual(len([r for r in rows if r['record_type']=='confirmation_pair']),10)
        self.assertEqual([r['status'] for r in rows if r['record_type']=='session_complete'],['complete'])

    def test_failed_numeric_check_stops_without_session_completion(self):
        calls,rows=self.run_case(fail=True)
        self.assertLess(len([x for x in calls if x[0]=='confirmation_timing']),20)
        self.assertFalse(any(r['record_type']=='session_complete' for r in rows))

    def test_pair_order_is_stable_and_session_specific(self):
        first=[evidence.pair_order('a'*64,'first',128,16,'b'*64,i) for i in range(10)]
        self.assertEqual(first,[evidence.pair_order('a'*64,'first',128,16,'b'*64,i) for i in range(10)])
        self.assertNotEqual(first,[evidence.pair_order('a'*64,'second',128,16,'b'*64,i) for i in range(10)])

    def test_screen_hash_mismatch_is_rejected_before_results(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            paths={name:Path(tmp)/name for name in ('rows','guard','incidents','budget')}
            for path in paths.values():path.write_text('',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                evidence.audit_screen({'schema':'frontier-confirmation-plan/v1','source_sha256':{}},paths,{})

    def test_screen_reaudit_detects_incomplete_candidate(self):
        fixture_path=HERE.parent/'frontier_execution'/'test_admission_gate.py'
        fixture_spec=importlib.util.spec_from_file_location('frontier_gate_test_fixture',fixture_path)
        fixture=importlib.util.module_from_spec(fixture_spec)
        with mock.patch.dict(sys.modules,{'runner':evidence.frontier}): fixture_spec.loader.exec_module(fixture)
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            base=Path(tmp);rows,ledger,manifest,commit,req,binaries=fixture.GateTests().make_case(base)
            paths={name:base/f'{name}.jsonl' for name in ('rows','guard','incidents','budget')}
            def write(path,items):path.write_text(''.join(json.dumps(x)+'\n' for x in items),encoding='utf8')
            write(paths['rows'],rows);write(paths['guard'],[{'event':'test'}]);write(paths['incidents'],[]);write(paths['budget'],ledger)
            cid=hashlib.sha256(json.dumps(req,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            plan={'schema':'frontier-confirmation-plan/v1','gpu':'GPU-test',
                  'source_sha256':{k:evidence.frontier.sha(v) for k,v in paths.items()},
                  'cases':[{'n':128,'batch':16,'candidate_id':cid,'request':req}]}
            with mock.patch.object(evidence.audit_guard,'audit',return_value={'status':'pass'}), \
                 mock.patch.object(evidence.validation,'check',side_effect=lambda row,*a,**kw:row['result']), \
                 mock.patch.object(evidence.validation,'check_describe',side_effect=lambda row,*a:row['result']), \
                 mock.patch.object(evidence.validation,'signature',return_value='same'), \
                 mock.patch.object(evidence.comparison,'compare'), \
                 mock.patch.object(evidence.frontier,'representable',return_value=(req,None)), \
                 mock.patch.object(evidence.search_controller,'source_signature',return_value='same'):
                result=evidence.audit_screen(plan,paths,binaries)
                self.assertEqual(result['cases'][0]['screen_ratio'],.9)
                control=copy.deepcopy(plan);control['cases'][0]['role']='negative_control'
                other_identity={'radix':[16,8],'logical_registers':16,'grouped_batch':1}
                with mock.patch.object(evidence.frontier,'representable',return_value=(other_identity,None)):
                    with self.assertRaisesRegex(ValueError,'role'):
                        evidence.audit_screen(control,paths,binaries)
                    slower=copy.deepcopy(rows)
                    for row in slower:
                        if row.get('phase')=='timing' and row['candidate_id']==cid:
                            row['result']['exec_ms_samples']=[1.1];row['stdout']=json.dumps(row['result'])
                        elif row.get('record_type')=='screen_pair':row['ratio']=1.1
                        elif row.get('record_type')=='candidate_disposition':
                            row['pair_ratios']=[1.1]*3;row['median_ratio']=1.1
                    write(paths['rows'],slower)
                    control['source_sha256']['rows']=evidence.frontier.sha(paths['rows'])
                    result=evidence.audit_screen(control,paths,binaries)
                    self.assertEqual((result['cases'][0]['role'],result['cases'][0]['screen_ratio']),('negative_control',1.1))
                    control['cases'][0]['role']='winner'
                    with self.assertRaisesRegex(ValueError,'role'):
                        evidence.audit_screen(control,paths,binaries)
                rows.pop();write(paths['rows'],rows)
                plan['source_sha256']['rows']=evidence.frontier.sha(paths['rows'])
                with self.assertRaisesRegex(ValueError,'disposition'):
                    evidence.audit_screen(plan,paths,binaries)


if __name__=='__main__':unittest.main()
