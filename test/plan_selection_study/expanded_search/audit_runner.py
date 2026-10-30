"""Budget and isolate independent screen or first-session CPU audits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import budget
import device_lock
import runner

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',required=True,choices=('screen','first'))
    for name in ('state','rows','guard','incidents','budget-snapshot','protocol','audit','gate'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--previous-rows',required=True,type=Path,nargs='+')
    p.add_argument('--manifests',type=Path);p.add_argument('--plan',type=Path)
    a=p.parse_args()
    if a.kind=='screen' and (a.manifests is None or a.plan is not None):p.error('screen requires manifests only')
    if a.kind=='first' and (a.plan is None or a.manifests is not None):p.error('first requires plan only')
    script=HERE/('make_screen_gate.py' if a.kind=='screen' else 'make_first_session_gate.py')
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+runner.GPU+'.lock')) as lock:
        b=budget.Budget(a.state)
        if b.state['phase']!='confirmation' or b.state['stopped'] or b.state.get('inflight'):
            raise ValueError('confirmation phase unavailable for independent audit')
        ident=uuid.uuid4().hex
        pending=a.gate.with_name(a.gate.name+'.pending-'+ident)
        if a.gate.exists() or pending.exists() or a.audit.exists():raise FileExistsError('audit/gate path already exists')
        cmd=[sys.executable,str(script),'--rows',str(a.rows),'--guard',str(a.guard),'--incidents',str(a.incidents),
             '--budget',str(a.budget_snapshot),'--protocol',str(a.protocol),'--audit',str(a.audit),'--gate',str(pending),
             '--manifests' if a.kind=='screen' else '--plan',str(a.manifests if a.kind=='screen' else a.plan),
             '--previous-rows',*[str(x) for x in a.previous_rows]]
        sources={x.name:runner.file_sha(x) for x in sorted(HERE.glob('audit_*.py'))}
        sources[script.name]=runner.file_sha(script)
        if not b.may_start('independent_'+a.kind+'_audit',expected_upper_seconds=650,work_id=ident,
                           argv=cmd,source_hashes=sources):
            raise ValueError('insufficient confirmation budget for audit')
        start=time.monotonic()
        try:
            proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=600,pass_fds=(lock.fd,))
            result=dict(argv=cmd,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,elapsed_seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired as error:
            def as_text(x):return x.decode('utf8','replace') if isinstance(x,bytes) else (x or '')
            log=a.audit.with_name(a.audit.name+'.launcher.json')
            with log.open('x',encoding='utf8') as out:
                json.dump(dict(argv=cmd,timeout=True,stdout=as_text(error.stdout),stderr=as_text(error.stderr),
                               elapsed_seconds=time.monotonic()-start),out,sort_keys=True);out.write('\n');out.flush();os.fsync(out.fileno())
            b.finish_work('timeout',work_id=ident,log=str(log));return 2
        except BaseException as error:
            b.finish_work('exception',work_id=ident,error=f'{type(error).__name__}: {error}');raise
        log=a.audit.with_name(a.audit.name+'.launcher.json')
        with log.open('x',encoding='utf8') as out:
            json.dump(result,out,sort_keys=True,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
        if any(runner.file_sha(HERE/name)!=digest for name,digest in sources.items()):
            b.finish_work('failed',work_id=ident,error='audit source changed during run',log=str(log));return 2
        if proc.returncode!=0:
            b.finish_work('failed',work_id=ident,exit_code=proc.returncode,log=str(log),pending_gate=str(pending));return proc.returncode
        gate=json.loads(pending.read_text(encoding='utf8'))
        if gate.get('status')!='pass' or gate.get('audit_sha256')!=runner.file_sha(a.audit):
            b.finish_work('failed',work_id=ident,error='audit result or report hash invalid',log=str(log),pending_gate=str(pending));return 2
        for path in (a.audit,pending):
            with path.open('rb+') as f:os.fsync(f.fileno())
        gate_sha=runner.file_sha(pending)
        b.finish_work('completed',work_id=ident,exit_code=0,log=str(log),pending_gate=str(pending),
                      pending_gate_sha256=gate_sha,audit_sha256=gate['audit_sha256'],final_gate=str(a.gate))
        # Hard-link is atomic and refuses to overwrite a prior gate.
        os.link(pending,a.gate)
        b.checkpoint('independent_audit_gate_published',work_id=ident,gate=str(a.gate),sha256=gate_sha)
        return proc.returncode


if __name__=='__main__':sys.exit(main())
