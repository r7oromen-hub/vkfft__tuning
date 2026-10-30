import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import coverage_audit as coverage

HERE = Path(__file__).resolve().parent


def cid(request):
    return hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class CoverageTests(unittest.TestCase):
    def test_independent_domain_cardinality(self):
        self.assertEqual(coverage.domain_cardinality(128,1),
                         {'enumerated':488,'static_supported':305,'static_rejected':183})
        # N=2 has exactly the [2] radix sequence and only R=2 is eligible.
        self.assertEqual(coverage.domain_cardinality(2,12),
                         {'enumerated':24,'static_supported':3,'static_rejected':21})
        single=coverage.domain_cardinality(128,1)
        grouped=coverage.domain_cardinality(128,16)
        self.assertEqual(grouped,{k:5*v for k,v in single.items()})

    def fixture(self, base):
        path=HERE.parent/'frontier_execution'/'test_admission_gate.py'
        spec=importlib.util.spec_from_file_location('coverage_gate_fixture',path)
        module=importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules,{'runner':coverage.base}): spec.loader.exec_module(module)
        rows,ledger,manifest,commit,request,binaries=module.GateTests().make_case(base)
        second={'radix':[16,8],'logical_registers':16,'grouped_batch':1}
        third={'radix':[8,4,4],'logical_registers':8,'grouped_batch':1}
        unseen={'radix':[8,16],'logical_registers':16,'grouped_batch':2}
        rejected={'radix':[8,16],'logical_registers':8,'grouped_batch':1}
        records=[json.loads(line) for line in manifest.read_text(encoding='utf8').splitlines()]
        records.extend([
            {'candidate_id':cid(second),'request':second,'selection':'selected','selection_index':1,
             'status':'pending_scheduler_and_device_validation','reason':None},
            {'candidate_id':cid(third),'request':third,'selection':'selected','selection_index':2,
             'status':'pending_scheduler_and_device_validation','reason':None},
            {'candidate_id':cid(unseen),'request':unseen,'selection':'not_selected',
             'status':'pending_scheduler_and_device_validation','reason':None},
            {'candidate_id':cid(rejected),'request':rejected,'selection':'rejected_static',
             'status':'rejected_static','reason':'radix_exceeds_registers'}])
        records[0].update(enumerated_count=5,selected_count=3,static_rejected_count=1,pending_count=4)
        manifest.write_text(''.join(json.dumps(x)+'\n' for x in records),encoding='utf8')
        contract=rows[0]['contract'];contract['manifests']['m']=coverage.base.sha(manifest);contract['timeout']=10
        for row in rows:
            row.update(session_id='s',guard_id='g')
            if row.get('record_type')=='execution': row['workload']['study_mode']='describe' if row['phase']=='describe' else 'execute'
        for i,event in enumerate(ledger): event['state'].update(start=0,seconds=1000,last=i,boot='boot')
        return rows,ledger,manifest,binaries,request,second

    def verify(self,rows,ledger,manifest,binaries,request,identity_override=None):
        with mock.patch.object(coverage.audit_guard,'audit',return_value={'status':'pass','errors':[]}), \
             mock.patch.object(coverage.validation,'check',side_effect=lambda row,*a,**kw:row['result']), \
             mock.patch.object(coverage.validation,'check_describe',side_effect=lambda row,*a:row['result']), \
             mock.patch.object(coverage.validation,'signature',return_value='same'), \
             mock.patch.object(coverage.comparison,'compare'), \
             mock.patch.object(coverage.base,'representable',return_value=identity_override or (request,None)), \
             mock.patch.object(coverage.search_controller,'source_signature',return_value='same'):
            return coverage.audit_coverage(rows,[{'event':'guard_start','guard_id':'g','gpu_uuid':'GPU-test'}],[],ledger,[manifest],'GPU-test',binaries)

    def deny(self,rows,ledger,identity):
        rows.append({'record_type':'budget_exhausted','contract':rows[0]['contract']})
        ledger.append({'event':'denied','identity':identity,'state':{'contract':rows[0]['contract'],'inflight':None,
                                                                   'start':0,'seconds':1000,'last':1000,'boot':'boot'}})

    def test_unstarted_and_static_rejections_are_distinct(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            rows,ledger,manifest,binaries,request,second=self.fixture(Path(tmp))
            rows.append(dict(record_type='candidate_start',n=128,batch=16,candidate_id=cid(second),request=second,
                             session_id='s',guard_id='g',contract=rows[0]['contract']))
            self.deny(rows,ledger,dict(n=128,batch=16,candidate_id=cid(second),phase='describe',round=0))
            report=self.verify(rows,ledger,manifest,binaries,request)
            self.assertEqual(report['status'],'pass',report)
            self.assertEqual(report['counts']['screen_complete'],1)
            self.assertEqual(report['counts']['selected_not_started'],1)
            self.assertEqual(report['counts']['partial'],1)
            self.assertEqual(report['counts']['not_selected'],1)
            self.assertEqual(report['counts']['static_rejected'],1)
            self.assertFalse(report['all_workload_inventories_complete'])

    def test_missing_disposition_is_partial_at_budget_boundary(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            rows,ledger,manifest,binaries,request,_=self.fixture(Path(tmp))
            rows.pop();rows.pop();last=rows.pop();ledger.pop();ledger.pop()
            self.deny(rows,ledger,dict(n=128,batch=16,candidate_id=last['candidate_id'],phase=last['phase'],round=last['round']))
            report=self.verify(rows,ledger,manifest,binaries,request)
            self.assertEqual(report['status'],'pass',report)
            self.assertEqual(report['counts']['partial'],1)
            self.assertEqual(report['counts']['selected_not_started'],2)

    def test_missing_disposition_without_boundary_fails(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            rows,ledger,manifest,binaries,request,_=self.fixture(Path(tmp));rows.pop()
            report=self.verify(rows,ledger,manifest,binaries,request)
            self.assertEqual(report['status'],'fail')

    def test_duplicate_and_unknown_candidate_fail(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            rows,ledger,manifest,binaries,request,second=self.fixture(Path(tmp))
            self.deny(rows,ledger,dict(n=128,batch=16,candidate_id=cid(second),phase='describe',round=0))
            duplicate=copy.deepcopy(next(r for r in rows if r.get('record_type')=='candidate_disposition'))
            rows.insert(-1,duplicate);self.assertEqual(self.verify(rows,ledger,manifest,binaries,request)['status'],'fail')
            rows.pop(-2)
            rows.insert(-1,dict(record_type='candidate_start',n=128,batch=16,candidate_id='unknown',request={},
                                session_id='s',guard_id='g',contract=rows[0]['contract']))
            self.assertEqual(self.verify(rows,ledger,manifest,binaries,request)['status'],'fail')

    def test_forged_known_rejection_is_not_accepted(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            rows,ledger,manifest,binaries,request,second=self.fixture(Path(tmp))
            self.deny(rows,ledger,dict(n=128,batch=16,candidate_id=cid(second),phase='describe',round=0))
            rows.insert(-1,dict(record_type='candidate_start',n=128,batch=16,candidate_id=cid(second),request=second,
                                session_id='s',guard_id='g',contract=rows[0]['contract']))
            rows.insert(-1,dict(record_type='candidate_disposition',n=128,batch=16,candidate_id=cid(second),request=second,
                                status='scheduler_rejected',session_id='s',guard_id='g',contract=rows[0]['contract']))
            self.assertEqual(self.verify(rows,ledger,manifest,binaries,request)['status'],'fail')

    def test_unsupported_identity_leaves_selected_unmeasured(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            rows,ledger,manifest,binaries,request,_=self.fixture(Path(tmp))
            del rows[3:];del ledger[5:]
            rows.append(dict(record_type='workload_status',n=128,batch=16,status='unsupported_identity',
                             reason='register_boost_not_one',session_id='s',guard_id='g',contract=rows[0]['contract']))
            report=self.verify(rows,ledger,manifest,binaries,request,(None,'register_boost_not_one'))
            self.assertEqual(report['status'],'pass',report)
            self.assertEqual(report['counts']['selected_not_started'],3)
            self.assertTrue(all(c.get('not_started_reason')=='unsupported_identity' for c in report['cases']))

    def test_forged_early_budget_denial_fails(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            rows,ledger,manifest,binaries,request,second=self.fixture(Path(tmp))
            rows.append(dict(record_type='candidate_start',n=128,batch=16,candidate_id=cid(second),request=second,
                             session_id='s',guard_id='g',contract=rows[0]['contract']))
            self.deny(rows,ledger,dict(n=128,batch=16,candidate_id=cid(second),phase='describe',round=0))
            ledger[-1]['state']['last']=len(ledger)+1
            self.assertEqual(self.verify(rows,ledger,manifest,binaries,request)['status'],'fail')

    def test_unmeasured_candidate_cannot_be_labeled_invalid(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            rows,ledger,manifest,binaries,request,second=self.fixture(Path(tmp))
            rows.append(dict(record_type='candidate_start',n=128,batch=16,candidate_id=cid(second),request=second,
                             session_id='s',guard_id='g',contract=rows[0]['contract']))
            rows.append(dict(record_type='candidate_disposition',n=128,batch=16,candidate_id=cid(second),request=second,
                             status='numerically_invalid',session_id='s',guard_id='g',contract=rows[0]['contract']))
            self.deny(rows,ledger,dict(n=128,batch=16,candidate_id=cid(second),phase='describe',round=0))
            self.assertEqual(self.verify(rows,ledger,manifest,binaries,request)['status'],'fail')

    def test_guarded_rejection_requires_matching_request_environment(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            rows,ledger,manifest,binaries,request,second=self.fixture(Path(tmp))
            contract=rows[0]['contract']; name=cid(second)
            rows.append(dict(record_type='candidate_start',n=128,batch=16,candidate_id=name,request=second,
                             session_id='s',guard_id='g',contract=contract))
            template=next(r for r in rows if r.get('record_type')=='execution' and r.get('candidate_id')=='original')
            failure=copy.deepcopy(template)
            failure.update(phase='describe',candidate_id=name,round=0,order=0,status='failed',returncode=1,
                           guard_clean=True,guard_gpu_uuid='GPU-test',executable_sha256=contract['binaries']['expanded'],
                           guard_run_id='g:failure',direct_environment=coverage.validation.request_env(second))
            failure['workload']['study_mode']='describe'
            failure['result']={'status':'error','failure_phase':'initialize',
                               'error':'initializeVkFFT failed with code 3002','vkfft_result_code':3002,
                               'expanded_rejection_reason':'splitter_shared'}
            failure['stdout']=json.dumps(failure['result']); rows.append(failure)
            def budget_pair(row):
                identity={'n':128,'batch':16,'candidate_id':row['candidate_id'],'phase':row['phase'],'round':row['round']}
                def state(inflight):return {'contract':contract,'inflight':inflight,'start':0,'seconds':1000,
                                            'last':len(ledger),'boot':'boot'}
                ledger.append({'event':'work_start','state':state(identity)})
                ledger.append({'event':'work_finish','state':state(None),'identity':identity,'status':row['status']})
            budget_pair(failure)
            post=copy.deepcopy(template);post.update(phase='post_rejection',round=0,order=0,direct_environment={})
            rows.append(post);budget_pair(post)
            rows.append(dict(record_type='candidate_disposition',n=128,batch=16,candidate_id=name,request=second,
                             status='scheduler_rejected',diagnostic='splitter_shared',session_id='s',guard_id='g',contract=contract))
            third={'radix':[8,4,4],'logical_registers':8,'grouped_batch':1}
            rows.append(dict(record_type='candidate_start',n=128,batch=16,candidate_id=cid(third),request=third,
                             session_id='s',guard_id='g',contract=contract))
            self.deny(rows,ledger,dict(n=128,batch=16,candidate_id=cid(third),phase='describe',round=0))
            report=self.verify(rows,ledger,manifest,binaries,request)
            self.assertEqual(report['status'],'pass',report)
            self.assertEqual(report['counts']['scheduler_rejected'],1)
            failure['direct_environment']['VKFFT_EXPANDED_REGS']='8'
            self.assertEqual(self.verify(rows,ledger,manifest,binaries,request)['status'],'fail')


if __name__=='__main__':unittest.main()
