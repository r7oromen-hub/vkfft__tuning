import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import admission_gate as gate
import runner


class GateTests(unittest.TestCase):
    def make_case(self, base):
        req = {'radix':[8,16], 'logical_registers':16, 'grouped_batch':1}
        cid = hashlib.sha256(json.dumps(req,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        manifest = base/'m.jsonl'
        lines = [{'schema':'frontier-manifest/v1','workload':{'n':128,'batch':16}},
                 {'request':req,'candidate_id':cid,'selection':'selected','selection_index':0,
                  'status':'pending_scheduler_and_device_validation'}]
        manifest.write_text('\n'.join(json.dumps(x) for x in lines)+'\n',encoding='utf8')
        commit='a'*40
        binaries={'original':base/'original','expanded':base/'expanded'}
        for name,path in binaries.items(): path.write_text(name,encoding='utf8')
        contract={'gpu':'GPU-test','execution_commit':commit,'manifests':{'m':runner.sha(manifest)},
                  'binaries':{name:runner.sha(path) for name,path in binaries.items()}}
        rows=[]; ledger=[{'event':'start','state':{'contract':contract,'inflight':None}}]
        def emit(row): rows.append(dict(contract=contract,**row))
        def execution(phase,name,pattern='random',seed=42,round_no=0,order=0,graph=64):
            descriptor={'canonical_descriptor':{},'source_sha256':'x','source_bytes':1,'compiler_input':{}}
            result={'precompile_kernels':[descriptor], 'raw_interval_ms_samples':[2.1],
                    'exec_ms_samples':[1.0 if name=='original' else 0.9], 'warmup_actual':5,'warmup_wall_ms':1000}
            work={'n':128,'batch':16,'pattern':pattern,'seed':seed,'graph_repeats':graph}
            emit(dict(record_type='execution',phase=phase,candidate_id=name,round=round_no,order=order,
                      workload=work,result=result,stdout=json.dumps(result),status='ok',returncode=0,
                      executable_sha256=contract['binaries']['original' if name=='original' else 'expanded']))
            identity={'n':128,'batch':16,'candidate_id':name,'phase':phase,'round':round_no}
            ledger.append({'event':'work_start','state':{'contract':contract,'inflight':identity}})
            ledger.append({'event':'work_finish','state':{'contract':contract,'inflight':None},'identity':identity,'status':'ok'})
        emit(dict(record_type='session_start'))
        for name in ('original','expanded_default','identity'): execution('identity',name)
        emit(dict(record_type='workload_status',n=128,batch=16,status='identity_passed',request=req))
        execution('calibration','original')
        emit(dict(record_type='candidate_start',n=128,batch=16,candidate_id=cid,request=req))
        execution('describe',cid)
        for pattern in runner.PATTERNS:
            for seed in runner.SEEDS: execution('validation',cid,pattern,seed)
        for round_no in range(3):
            for order,name in enumerate(runner.seeded_order(128,16,cid,round_no)):
                execution('timing','original' if name=='original' else cid,round_no=round_no,order=order)
            emit(dict(record_type='screen_pair',n=128,batch=16,candidate_id=cid,round=round_no,ratio=.9))
        emit(dict(record_type='candidate_disposition',n=128,batch=16,candidate_id=cid,status='screen_complete',request=req,pair_ratios=[.9]*3,median_ratio=.9))
        return rows,ledger,manifest,commit,req,binaries

    def test_gate_accepts_complete_and_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            rows,ledger,manifest,commit,request,binaries=self.make_case(Path(tmp))
            with mock.patch.object(gate.audit_guard,'audit',return_value={'status':'pass'}), \
                 mock.patch.object(gate.validation,'check',side_effect=lambda row,*a,**k:row['result']), \
                 mock.patch.object(gate.validation,'check_describe',side_effect=lambda row,*a:row['result']), \
                 mock.patch.object(gate.validation,'signature',return_value='same'), \
                 mock.patch.object(gate.comparison,'compare'), \
                 mock.patch.object(gate.runner,'representable',return_value=(request,None)), \
                 mock.patch.object(gate.search_controller,'source_signature',return_value='same'):
                def audit(): return gate.audit(rows,[{'event':'guard'}],[],ledger,[manifest],'GPU-test',commit,1,binaries)
                result = audit(); self.assertEqual(result['status'],'pass',result)
                last=rows.pop()
                self.assertEqual(audit()['status'],'fail')
                rows.append(last)
                row=next(x for x in rows if x.get('phase')=='validation')
                row['stdout']='{}'
                self.assertEqual(audit()['status'],'fail')
                row['stdout']=json.dumps(row['result'])
                rows[-1]['status']='numerically_invalid'
                self.assertEqual(audit()['status'],'fail')


if __name__ == '__main__': unittest.main()
