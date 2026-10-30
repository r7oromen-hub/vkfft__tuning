import copy
import hashlib
import json
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_rotation as a
from test_rotation_controller import case, valid_row


class RotationAuditTest(unittest.TestCase):
    def test_gate_hash_and_reconfirmed_candidates(self):
        c=case();plan={'sha256':{'original':'o','expanded':'e'},'workloads':[{'n':c['n'],'batch':c['batch'],'candidate':c}]}
        sessions={'first':{'status':'pass','session_id':'f'},'second':{'status':'pass','session_id':'s'}}
        combined={'status':'pass','candidates':[{'n':c['n'],'batch':c['batch'],'candidate_id':c['candidate_id'],
                                                 'status':'confirmed_under_preregistered_rule'}]}
        budget={'status':'pass','phase':'confirmation','inflight':None}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            paths={side:{key:Path(temp)/(side+'_'+key) for key in ('rows','guard','incidents','budget')} for side in ('first','second')}
            for side in paths.values():
                for path in side.values():path.write_text('',encoding='utf8')
            hashes={'plan':'p','protocol':'protocol','first_gate':'fg','first_audit':'fa','second_audit':'sa','combined_audit':'ca',
                    'sessions':{side:{key:a.sha(path) for key,path in files.items()} for side,files in paths.items()}}
            fg={'evidence_sha256':hashes['sessions']['first'],'binaries_sha256':plan['sha256']}
            gate={'schema':'expanded-two-session-gate/v1','status':'pass','plan_sha256':'p','protocol_sha256':'protocol',
                  'first_gate_sha256':'fg','first_audit_sha256':'fa','second_audit_sha256':'sa','combined_audit_sha256':'ca',
                  'session_evidence_sha256':hashes['sessions'],'binaries_sha256':plan['sha256'],'confirmed':[c]}
            report={'status':'pass','first_session':sessions['first'],'second_session':sessions['second'],
                    'first_budget':budget,'second_budget':budget,'combined':combined}
            with patch.object(a.audit_confirmation,'audit',side_effect=[sessions['first'],sessions['second']]*2), \
                 patch.object(a.audit_two_sessions,'combine',return_value=combined):
                ok=a.verify_gate(plan,gate,fg,sessions['first'],sessions['second'],report,paths,hashes)
                self.assertEqual(ok['status'],'pass',ok['errors'])
                bad=copy.deepcopy(gate);bad['confirmed'][0]['request']['logical_registers']=16
                self.assertEqual(a.verify_gate(plan,bad,fg,sessions['first'],sessions['second'],report,paths,hashes)['status'],'fail')

    def test_slot_graph_pool_and_samples_tampering(self):
        c=case();rotation={'slots':2,'total_bytes':131072,'graph_repeats':64,'l2_bytes':32768}
        with patch.object(a.audit_search,'source_signature',side_effect=lambda _r,original:{'original':True} if original else {'candidate':True}):
            for original in (True,False):
                row=valid_row(original);row['workload'].update(graph_repeats=64,repeats=31,warmup=5,warmup_ms=1000)
                self.assertEqual(a.check_execution(row,c,rotation,'GPU-test',original=original,timing=True),[])
                for kind in ('slot','graph','pool','samples'):
                    x=copy.deepcopy(row)
                    if kind=='slot':x['result']['rotation_final_validation']['slots'].pop()
                    if kind=='graph':x['result']['rotation_graph_kernel_arguments_and_order_verified']=False
                    if kind=='pool':x['result']['rotation_input_pool_bytes']-=8
                    if kind=='samples':x['result']['raw_interval_ms_samples'][0]=0
                    self.assertTrue(a.check_execution(x,c,rotation,'GPU-test',original=original,timing=True),kind)

    def test_ten_pairs_and_incomplete(self):
        c=case();key=(c['n'],c['batch'],c['candidate_id']);rot={'slots':2,'total_bytes':131072,'graph_repeats':64,'l2_bytes':32768}
        plan={'selection_seed':42,'sha256':{'original':'hot-o','expanded':'hot-e'}};gate={'confirmed':[c]};session='session'
        h={'record_type':'rotation_protocol','session_id':session,'gpu':'GPU-test','rounds':10,'confirmed':[c],
           'hot_binary_sha256':plan['sha256'],'sha256':{'original':'o','expanded':'e'}}
        rows=[h]
        def execution(original,rid,phase,round_no,order):
            row=valid_row(original);row.update(record_type='execution',session_id=session,guard_run_id=rid,
                candidate_id='original' if original else key[2],phase=phase,round=round_no,order=order,
                executable_sha256='o' if original else 'e')
            row['workload'].update(graph_repeats=64,repeats=1 if phase=='rotation_calibration' else 31,warmup=5,warmup_ms=1000)
            if phase=='rotation_calibration':
                row['result']['exec_ms_samples']=row['result']['exec_ms_samples'][:1];row['result']['raw_interval_ms_samples']=[2.0]
            row['stdout']=json.dumps(row['result'])+'\n';rows.append(row)
        execution(True,'cal','rotation_calibration',0,0)
        rows.append(dict(record_type='rotation_calibration_complete',n=key[0],batch=key[1],candidate_id=key[2],
                         guard_run_id='cal',rotation=rot,raw_interval_ms=2.0))
        for i in range(10):
            seed=json.dumps([42,*key,'rotation',i,42],separators=(',',':')).encode();order=['original','candidate']
            random.Random(int.from_bytes(hashlib.sha256(seed).digest(),'big')).shuffle(order)
            execution(True,'o'+str(i),'rotation_timing',i,order.index('original'))
            execution(False,'c'+str(i),'rotation_timing',i,order.index('candidate'))
            rows.append(dict(record_type='rotation_pair',session_id=session,n=key[0],batch=key[1],candidate_id=key[2],round=i,
                             order=order,rotation=rot,original_guard_run_id='o'+str(i),candidate_guard_run_id='c'+str(i),
                             original_median_ms=2/64,candidate_median_ms=2/64,candidate_over_original=1.0))
        rows.extend([dict(record_type='rotation_disposition',n=key[0],batch=key[1],candidate_id=key[2],status='complete',
                          rotation=rot,paired_ratios=[1.0]*10,median_ratio=1.0),
                     dict(record_type='rotation_complete',session_id=session,status='completed',incomplete_or_not_started=[])])
        hot=[{'record_type':'execution','candidate_id':'original','workload':{'n':key[0],'batch':key[1]},
              'result':{'gpu':{'uuid':'GPU-test','l2_cache_bytes':32768}}}]
        with patch.object(a.audit_guard,'audit',return_value={'errors':[]}), \
             patch.object(a.audit_budget,'audit',return_value={'errors':[],'phase':'extension','inflight':None}), \
             patch.object(a.audit_search,'source_signature',side_effect=lambda _r,original:{'original':True} if original else {'candidate':True}):
            result=a.audit_records(rows,[],[],gate,plan,[],{},[],hot)
            self.assertEqual(result['status'],'pass',result['errors']);self.assertEqual(len(result['complete']),1)
            altered=copy.deepcopy(rows);next(x for x in altered if x.get('record_type')=='rotation_pair')['candidate_over_original']=0.5
            self.assertEqual(a.audit_records(altered,[],[],gate,plan,[],{},[],hot)['status'],'fail')
            self.assertEqual(a.audit_records([],[],[],gate,plan,[],{},[],hot)['status'],'uncovered')


if __name__=='__main__':unittest.main()
