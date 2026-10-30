import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import audit


def case_fixture(root, kind='first'):
    binaries = {'original':root/'original','expanded':root/'expanded'}
    for name,path in binaries.items(): path.write_text(name,encoding='utf8')
    request={'radix':[8,16],'logical_registers':16,'grouped_batch':1}
    case={'n':128,'batch':16,'candidate_id':'candidate','request':request}
    plan={'schema':'frontier-confirmation-plan/v1','gpu':'GPU-test','source_sha256':{'rows':'screen'},'cases':[case]}
    proofs={(128,16,'candidate'):{**case,'screen_candidate_signature_sha256':'cand','screen_original_signature_sha256':'orig'}}
    commit='a'*40; phash='b'*64
    contract={'schema':'frontier-confirmation/v1','gpu':'GPU-test','execution_commit':commit,'vkfft_commit':'c'*40,
              'binaries':{name:audit.base.sha(path) for name,path in binaries.items()},'plan_sha256':phash,
              'source_sha256':plan['source_sha256'],'session_kind':kind}
    first_hashes={'rows':'r','guard':'g','incidents':'i','budget':'b'}
    if kind=='second':
        contract['first_session']={'session_id':'session-first','guard_id':'guard-first','evidence_sha256':first_hashes}
    sid='session-'+kind; gid='guard-'+kind; rows=[]
    ledger=[{'event':'start','state':{'contract':contract,'inflight':None}}]
    def emit(row): rows.append(dict(row,contract=contract,session_id=sid,guard_id=gid))
    def execution(phase,name,round_no=0,order=0):
        rid=gid+':'+str(sum(r.get('record_type')=='execution' for r in rows))
        result={'raw_interval_ms_samples':[2.5],'exec_ms_samples':[1.0 if name=='original' else .95],
                'warmup_actual':5,'warmup_wall_ms':1000}
        work={'n':128,'batch':16,'pattern':'random','seed':42,'graph_repeats':64}
        row={'record_type':'execution','phase':phase,'candidate_id':name,'round':round_no,'order':order,
             'workload':work,'result':result,'stdout':json.dumps(result),'status':'ok','returncode':0,
             'executable_sha256':contract['binaries']['original' if name=='original' else 'expanded'],'guard_run_id':rid}
        emit(row)
        ident={'n':128,'batch':16,'candidate_id':name,'phase':phase,'round':round_no}
        ledger.append({'event':'work_start','state':{'contract':contract,'inflight':ident}})
        ledger.append({'event':'work_finish','state':{'contract':contract,'inflight':None},'identity':ident,'status':'ok'})
        return rid
    emit({'record_type':'session_start'})
    for name in ('original','expanded_default','identity'): execution('identity',name)
    emit({'record_type':'workload_status','n':128,'batch':16,'status':'identity_passed','request':request})
    execution('calibration','original')
    emit({'record_type':'candidate_start',**case,'screen_original_signature_sha256':'orig','screen_candidate_signature_sha256':'cand'})
    def ordering(ph,sk,n,b,cid,r): return ['original','candidate'] if r%2==0 else ['candidate','original']
    for r in range(10):
        ids={}
        for ix,name in enumerate(ordering(phash,kind,128,16,'candidate',r)):
            ids[name]=execution('confirmation_timing','original' if name=='original' else 'candidate',r,ix)
        emit({'record_type':'confirmation_pair','n':128,'batch':16,'candidate_id':'candidate','round':r,
              'original_guard_run_id':ids['original'],'candidate_guard_run_id':ids['candidate'],
              'original_ms':1.0,'candidate_ms':.95})
    emit({'record_type':'candidate_disposition',**case,'status':'confirmation_complete'})
    emit({'record_type':'session_complete','status':'complete','candidate_count':1})
    guard=[{'guard_id':gid,'utc':'2026-10-31T00:00:00+00:00' if kind=='first' else '2026-10-31T01:00:00+00:00'},
           {'guard_id':gid,'utc':'2026-10-31T00:30:00+00:00' if kind=='first' else '2026-10-31T01:30:00+00:00'}]
    return rows,guard,ledger,plan,phash,commit,binaries,proofs,request,ordering


class AuditTests(unittest.TestCase):
    def check(self, fixture):
        rows,guards,ledger,plan,phash,commit,binaries,proofs,request,ordering=fixture
        with mock.patch.object(audit.audit_guard,'audit',return_value={'status':'pass','errors':[]}), \
             mock.patch.object(audit.validation,'check',side_effect=lambda row,*a,**k:row['result']), \
             mock.patch.object(audit.validation,'signature',return_value='same'), \
             mock.patch.object(audit.comparison,'compare'), \
             mock.patch.object(audit.base,'representable',return_value=(request,None)), \
             mock.patch.object(audit,'signature_sha',side_effect=lambda result,original:'orig' if original else 'cand'):
            return audit.audit_session(rows,guards,[],ledger,plan,phash,'GPU-test',commit,binaries,proofs,ordering)

    def test_raw_session_and_incomplete_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            f=case_fixture(Path(tmp)); result=self.check(f)
            self.assertEqual(result['status'],'pass',result)
            self.assertTrue(result['results'][0]['practical'])
            tampered=copy.deepcopy(f); tampered[0].pop()
            self.assertEqual(self.check(tampered)['status'],'fail')
            tampered=copy.deepcopy(f); pair=next(x for x in tampered[0] if x.get('record_type')=='confirmation_pair')
            pair['candidate_ms']=.5
            self.assertEqual(self.check(tampered)['status'],'fail')
            tampered=copy.deepcopy(f); row=next(x for x in tampered[0] if x.get('phase')=='confirmation_timing')
            row['stdout']='{}'
            self.assertEqual(self.check(tampered)['status'],'fail')
            tampered=copy.deepcopy(f); pair=next(x for x in tampered[0] if x.get('record_type')=='confirmation_pair')
            pair['candidate_guard_run_id']=pair['original_guard_run_id']
            self.assertEqual(self.check(tampered)['status'],'fail')

    def test_two_sessions_require_distinct_complete_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            one=self.check(case_fixture(Path(tmp),'first')); two=self.check(case_fixture(Path(tmp),'second'))
            self.assertEqual(one['status'],'pass',one); self.assertEqual(two['status'],'pass',two)
            hashes={'rows':'r','guard':'g','incidents':'i','budget':'b'}
            combined=audit.audit_two(one,two,hashes)
            self.assertEqual(combined['status'],'pass',combined)
            self.assertTrue(combined['results'][0]['both_sessions_practical'])
            slower=copy.deepcopy(two); slower['results'][0]['practical']=False
            self.assertFalse(audit.audit_two(one,slower,hashes)['results'][0]['both_sessions_practical'])
            wrong_binding=copy.deepcopy(two); wrong_binding['first_session']['evidence_sha256']['rows']='changed'
            self.assertEqual(audit.audit_two(one,wrong_binding,hashes)['status'],'fail')
            two['guard_id']=one['guard_id']; self.assertEqual(audit.audit_two(one,two,hashes)['status'],'fail')


if __name__ == '__main__': unittest.main()
