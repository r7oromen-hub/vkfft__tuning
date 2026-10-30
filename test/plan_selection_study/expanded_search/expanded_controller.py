"""Guard-only execution of the separately registered admission stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import budget
import identity
import run_suite as suite
import validation
sys.path.insert(1, str(HERE.parent / 'original_compare'))
import original_compare_controller as original_comparison


class BudgetStop(Exception): pass


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True);p.add_argument('--expanded-exe',required=True);p.add_argument('--original-exe',required=True)
    p.add_argument('--budget-dir',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--admission-protocol',required=True,type=Path);p.add_argument('--timeout',type=int,default=300)
    a=p.parse_args(argv)
    if os.environ.get('VKFFT_STUDY_GUARD_UUID') != a.gpu or not os.environ.get('VKFFT_STUDY_GUARD_ID'):
        p.error('guard required')
    b=budget.Budget(a.budget_dir)
    if b.state['phase'] != 'admission' or b.state['stopped']: p.error('budget is not in admission')
    protocol=json.loads(a.admission_protocol.read_text(encoding='utf8'))
    if protocol['gpu_uuid'] != a.gpu: p.error('protocol GPU mismatch')
    expanded=str(Path(a.expanded_exe).resolve(strict=True)); original=str(Path(a.original_exe).resolve(strict=True))
    hashes={name:sha(path) for name,path in [('expanded',expanded),('original',original)]}
    if expanded == original: p.error('distinct original and expanded binaries required')
    session=uuid.uuid4().hex
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as out:
        def emit(row):
            out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
        emit(dict(record_type='admission_protocol',session_id=session,protocol=protocol,
                  protocol_sha256=sha(a.admission_protocol),binaries={'expanded':expanded,'original':original},sha256=hashes))
        def one(n,batch,name,request=None,pattern='random',seed=42,mode='execute'):
            work_id=uuid.uuid4().hex
            if not b.may_start('admission_process',expected_upper_seconds=a.timeout+30,
                               work_id=work_id,n=n,batch=batch,candidate_id=name,pattern=pattern,seed=seed,mode=mode):
                raise BudgetStop()
            binary='original' if name == 'original' else 'expanded'
            a.exe=original if binary=='original' else expanded;a.exe_sha256=hashes[binary];a.session_id=session
            w=dict(n=n,batch=batch,seed=seed,config={},pattern=pattern,graph_repeats=64,
                   warmup=0,warmup_ms=0,repeats=1,study_mode=mode)
            env=validation.request_env(request) if request else {}
            row=suite.run_one(a,w,(name,env),'admission',0,0)
            emit(row);b.finish_work(row['status'],work_id=work_id,guard_run_id=row.get('guard_run_id'))
            return row
        statuses=[]
        try:
            for work in protocol['workloads']:
                n,batch=work['n'],work['batch']
                o=one(n,batch,'original');original_result=validation.check(o,a.gpu,original=True)
                d=one(n,batch,'expanded_default');default=validation.check(d,a.gpu)
                original_comparison.compare(default,original_result)
                req,reason=identity.representable(default)
                if req is None:
                    emit(dict(record_type='workload_status',n=n,batch=batch,status='unsupported_identity',reason=reason));statuses.append('unsupported_identity');continue
                ir=one(n,batch,'identity',req);identity_result=validation.check(ir,a.gpu,request=req)
                if validation.signature(default) != validation.signature(identity_result):
                    raise ValueError(f'identity mismatch for n={n}, batch={batch}')
                for request in work['requests']:
                    name=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                    desc=one(n,batch,name,request,mode='describe')
                    if desc.get('returncode') != 0:
                        r=desc.get('result',{})
                        if validation.known_rejection(desc,a.gpu):
                            emit(dict(record_type='candidate_disposition',n=n,batch=batch,candidate_id=name,request=request,
                                      status='scheduler_or_resource_rejected',reason=r.get('error')))
                            validation.check(one(n,batch,'original'),a.gpu,original=True)
                            continue
                        raise ValueError('unexpected describe failure')
                    dr=validation.check_describe(desc,a.gpu,request);k=dr['structural_features']['kernels'][0]
                    valid=0
                    for pattern in protocol['patterns']:
                        for seed in protocol['validation_seeds']:
                            base=one(n,batch,'original',pattern=pattern,seed=seed)
                            validation.check(base,a.gpu,original=True)
                            row=one(n,batch,name,request,pattern,seed)
                            if validation.known_rejection(row,a.gpu):
                                emit(dict(record_type='candidate_disposition',n=n,batch=batch,candidate_id=name,request=request,
                                          status='compiled_resource_or_scheduler_rejected',reason=row['result'].get('error')))
                                validation.check(one(n,batch,'original'),a.gpu,original=True)
                                break
                            validation.check(row,a.gpu,request=request);valid+=1
                        else: continue
                        break
                    if valid == len(protocol['patterns'])*len(protocol['validation_seeds']):
                        emit(dict(record_type='candidate_disposition',n=n,batch=batch,candidate_id=name,request=request,status='numerically_valid',checks=valid,
                                  actual_local_size=k['localSize']))
                emit(dict(record_type='workload_status',n=n,batch=batch,status='admission_exercised'));statuses.append('admission_exercised')
            emit(dict(record_type='admission_complete',status='completed_cases',workload_statuses=statuses))
        except BudgetStop:
            emit(dict(record_type='admission_complete',status='budget_exhausted',workload_statuses=statuses))


if __name__ == '__main__': main()
