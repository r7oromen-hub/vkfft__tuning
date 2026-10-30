import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import unittest
import confirmation_controller as confirm


class FakeBudget:
    def __init__(self):self.state={'phase':'confirmation','stopped':False,'inflight':None,'phase_start_monotonic':time.monotonic()};self.count=0
    def may_start(self,kind,**kw):
        if self.state['inflight']:raise AssertionError('overlap')
        self.state['inflight']=kw;return True
    def finish_work(self,status,**kw):self.state['inflight']=None;self.count+=1


class ConfirmationTest(unittest.TestCase):
    def test_first_session_pairs_and_gate_for_second(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp);o=root/'original';e=root/'expanded';o.write_bytes(b'o');e.write_bytes(b'e')
            sig={'signature':1};candidate={'candidate_id':'c','request':{'radix':[16,8],'logical_registers':16,'grouped_batch':1},
                 'graph_repeats':64,'original_signature':sig,'candidate_signature':sig}
            plan={'schema':'expanded-confirmation-plan/v1','screen_gate_sha256':'audited','selection_seed':20261030,
                  'binaries':{'original':str(o),'expanded':str(e)},'sha256':{'original':confirm.sha(o),'expanded':confirm.sha(e)},
                  'workloads':[{'n':128,'batch':1,'candidate':candidate,'role':'screen_faster'}],
                  'confirmation_order':[{'n':128,'batch':1,'candidate_id':'c','role':'screen_faster'}]}
            planpath=root/'plan.json';planpath.write_text(json.dumps(plan));out=root/'first.jsonl';fake=FakeBudget();calls=[]
            def run(args,w,c,phase,round_no,order):
                calls.append((c[0],round_no,order));v=1.0 if c[0]=='original' else .9
                return {'record_type':'execution','guard_clean':True,'guard_run_id':f'r{len(calls)}','candidate_id':c[0],
                        'result':{'exec_ms_samples':[v]*31,'warmup_actual':5,'warmup_wall_ms':1000}}
            with patch.dict(os.environ,{'VKFFT_STUDY_GUARD_ID':'g','VKFFT_STUDY_GUARD_UUID':'GPU-test'}), \
                 patch.object(confirm.budget,'Budget',return_value=fake),patch.object(confirm.suite,'run_one',side_effect=run), \
                 patch.object(confirm.validation,'check',side_effect=lambda row,*x,**y:row['result']), \
                 patch.object(confirm.search_controller,'source_signature',return_value=sig):
                confirm.main(['--gpu','GPU-test','--budget-dir',str(root),'--plan',str(planpath),'--output',str(out),'--session-kind','first'])
                records=[json.loads(x) for x in out.read_text().splitlines()]
                self.assertEqual(len([x for x in records if x['record_type']=='confirmation_pair']),10)
                self.assertEqual(fake.count,10);self.assertEqual(records[-1]['status'],'completed')
                with self.assertRaises(SystemExit):
                    confirm.main(['--gpu','GPU-test','--budget-dir',str(root),'--plan',str(planpath),'--output',str(root/'second'),
                                  '--session-kind','second','--first-results',str(out)])

    def test_second_uses_audited_priority_and_excludes_controls(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp);o=root/'original';e=root/'expanded';o.write_bytes(b'o');e.write_bytes(b'e')
            sig={'signature':1};workloads=[];order=[]
            for n,cid,role in ((128,'a','screen_faster'),(256,'b','screen_faster'),(512,'ctrl','default_faster_control')):
                req={'radix':[2],'logical_registers':2,'grouped_batch':1}
                cand={'candidate_id':cid,'request':req,'graph_repeats':64,'original_signature':sig,'candidate_signature':sig}
                workloads.append({'n':n,'batch':1,'candidate':cand,'role':role});order.append({'n':n,'batch':1,'candidate_id':cid,'role':role})
            plan={'schema':'expanded-confirmation-plan/v1','screen_gate_sha256':'audited','selection_seed':20261030,
                  'binaries':{'original':str(o),'expanded':str(e)},'sha256':{'original':confirm.sha(o),'expanded':confirm.sha(e)},
                  'workloads':workloads,'confirmation_order':order}
            pp=root/'plan.json';pp.write_text(json.dumps(plan));first=root/'first';fg=root/'first_guard';fi=root/'first_incidents';fb=root/'first_budget';fa=root/'first_audit'
            first.write_text(json.dumps({'record_type':'confirmation_protocol','session_kind':'first','session_id':'s','plan_sha256':confirm.sha(pp)})+'\n'+
                             json.dumps({'record_type':'confirmation_complete','status':'completed'})+'\n')
            for path in (fg,fi,fb,fa):path.write_text('fixture')
            gate={'schema':'expanded-first-session-gate/v1','status':'pass','session_id':'s','plan_sha256':confirm.sha(pp),
                  'evidence_sha256':{'rows':confirm.sha(first),'guard':confirm.sha(fg),'incidents':confirm.sha(fi),'budget':confirm.sha(fb)},
                  'audit_sha256':confirm.sha(fa),'binaries_sha256':plan['sha256'],
                  'eligible':[{'n':128,'batch':1,'candidate_id':'a','geometric_mean_ratio':.90,'passes_practical_and_ci':False},
                              {'n':256,'batch':1,'candidate_id':'b','geometric_mean_ratio':.95,'passes_practical_and_ci':True},
                              {'n':512,'batch':1,'candidate_id':'ctrl','geometric_mean_ratio':.80,'passes_practical_and_ci':True}]}
            gatepath=root/'gate';gatepath.write_text(json.dumps(gate));calls=[]
            def run(args,w,c,phase,round_no,order):
                calls.append(c[0]);return {'record_type':'execution','guard_run_id':f'r{len(calls)}','candidate_id':c[0],
                    'result':{'exec_ms_samples':[1]*31,'warmup_actual':5,'warmup_wall_ms':1000}}
            args=['--gpu','GPU-test','--budget-dir',str(root),'--plan',str(pp),'--output',str(root/'second'),'--session-kind','second',
                  '--first-results',str(first),'--first-gate',str(gatepath),'--first-audit',str(fa),'--first-guard',str(fg),
                  '--first-incidents',str(fi),'--first-budget-snapshot',str(fb)]
            with patch.dict(os.environ,{'VKFFT_STUDY_GUARD_ID':'g','VKFFT_STUDY_GUARD_UUID':'GPU-test'}), \
                 patch.object(confirm.budget,'Budget',return_value=FakeBudget()),patch.object(confirm.suite,'run_one',side_effect=run), \
                 patch.object(confirm.validation,'check',side_effect=lambda row,*x,**y:row['result']), \
                 patch.object(confirm.search_controller,'source_signature',return_value=sig):confirm.main(args)
            candidates=[x for x in calls if x!='original']
            self.assertEqual(candidates,['b']*10+['a']*10)


if __name__=='__main__':unittest.main()
