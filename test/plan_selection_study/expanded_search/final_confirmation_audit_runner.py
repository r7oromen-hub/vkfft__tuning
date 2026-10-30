"""Budget, freeze and atomically publish second-session audit or two-session gate."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',required=True,choices=('second','two'))
    p.add_argument('--state',required=True,type=Path);p.add_argument('--audit',required=True,type=Path);p.add_argument('--plan',required=True,type=Path)
    for name in ('rows','guard','incidents','gate','protocol','first-gate','first-audit','first-rows','first-guard','first-incidents','first-budget',
                 'second-audit','second-rows','second-guard','second-incidents','second-budget'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--previous-rows',type=Path,nargs='+')
    a=p.parse_args()
    required=('rows','guard','incidents') if a.kind=='second' else ('gate','protocol','first_gate','first_audit','first_rows','first_guard',
              'first_incidents','first_budget','second_audit','second_rows','second_guard','second_incidents','second_budget','previous_rows')
    if any(getattr(a,name) is None for name in required):p.error('required evidence missing for requested audit')
    script=HERE/('audit_confirmation.py' if a.kind=='second' else 'make_two_session_gate.py')
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+runner.GPU+'.lock')) as lock:
        b=budget.Budget(a.state)
        if b.state['phase']!='confirmation' or b.state['stopped'] or b.state.get('inflight'):raise ValueError('confirmation budget unavailable')
        ident=uuid.uuid4().hex
        pending=a.audit.with_name(a.audit.name+'.pending-'+ident)
        pending_gate=a.gate.with_name(a.gate.name+'.pending-'+ident) if a.gate else None
        if a.audit.exists() or pending.exists() or (a.gate and (a.gate.exists() or pending_gate.exists())):raise FileExistsError('output exists')
        if a.kind=='second':
            cmd=[sys.executable,str(script),'--rows',str(a.rows),'--guard',str(a.guard),'--incidents',str(a.incidents),'--plan',str(a.plan)]
        else:
            events=[json.loads(x) for x in b.ledger.read_text(encoding='utf8').splitlines()]
            prior=[x for x in events if x.get('event')=='work_finished' and x.get('kind')=='independent_second_audit' and x.get('status')=='completed']
            if len(prior)!=1 or prior[0].get('final_audit')!=str(a.second_audit) or prior[0].get('audit_sha256')!=runner.file_sha(a.second_audit):
                raise ValueError('second audit was not successfully budgeted and published')
            cmd=[sys.executable,str(script),'--plan',str(a.plan),'--protocol',str(a.protocol),'--audit',str(pending),'--gate',str(pending_gate)]
            for name in required:
                if name in ('gate','protocol','previous_rows'):continue
                cmd.extend(['--'+name.replace('_','-'),str(getattr(a,name))])
            cmd.extend(['--previous-rows',*[str(x) for x in a.previous_rows]])
        sources={x.name:runner.file_sha(x) for x in sorted(HERE.glob('audit_*.py'))}
        sources[script.name]=runner.file_sha(script);sources[Path(__file__).name]=runner.file_sha(__file__)
        if not b.may_start('independent_'+a.kind+'_audit',expected_upper_seconds=650,work_id=ident,argv=cmd,source_hashes=sources):
            raise ValueError('insufficient confirmation budget for audit')
        start=time.monotonic()
        try:
            proc=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=600,pass_fds=(lock.fd,))
            result=dict(argv=cmd,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,elapsed_seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired as error:
            text=lambda x:x.decode('utf8','replace') if isinstance(x,bytes) else (x or '')
            log=a.audit.with_name(a.audit.name+'.launcher.json')
            with log.open('x',encoding='utf8') as out:
                json.dump(dict(argv=cmd,timeout=True,stdout=text(error.stdout),stderr=text(error.stderr)),out,sort_keys=True);out.write('\n');out.flush();os.fsync(out.fileno())
            b.finish_work('timeout',work_id=ident,log=str(log));return 2
        except BaseException as error:
            b.finish_work('exception',work_id=ident,error=f'{type(error).__name__}: {error}');raise
        log=a.audit.with_name(a.audit.name+'.launcher.json')
        with log.open('x',encoding='utf8') as out:json.dump(result,out,sort_keys=True);out.write('\n');out.flush();os.fsync(out.fileno())
        if proc.returncode!=0 or any(runner.file_sha(HERE/name)!=digest for name,digest in sources.items()):
            b.finish_work('failed',work_id=ident,exit_code=proc.returncode,log=str(log));return proc.returncode or 2
        if a.kind=='second':
            report=json.loads(proc.stdout)
            if report.get('status')!='pass' or report.get('session_kind')!='second':
                b.finish_work('failed',work_id=ident,error='invalid second-session audit',log=str(log));return 2
            with pending.open('x',encoding='utf8') as out:json.dump(report,out,sort_keys=True,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
        else:
            report=json.loads(pending.read_text(encoding='utf8'));gate=json.loads(pending_gate.read_text(encoding='utf8'))
            if report.get('status')!='pass' or gate.get('status')!='pass' or gate.get('combined_audit_sha256')!=runner.file_sha(pending):
                b.finish_work('failed',work_id=ident,error='two-session audit or gate invalid',log=str(log));return 2
        for path in (pending,pending_gate):
            if path is not None:
                with path.open('rb+') as f:os.fsync(f.fileno())
        audit_sha=runner.file_sha(pending);gate_sha=runner.file_sha(pending_gate) if pending_gate else None
        b.finish_work('completed',work_id=ident,log=str(log),pending_audit=str(pending),final_audit=str(a.audit),audit_sha256=audit_sha,
                      pending_gate=str(pending_gate) if pending_gate else None,final_gate=str(a.gate) if a.gate else None,gate_sha256=gate_sha)
        os.link(pending,a.audit)
        if pending_gate:os.link(pending_gate,a.gate)
        b.checkpoint('final_confirmation_audit_published',work_id=ident,audit=str(a.audit),audit_sha256=audit_sha,
                     gate=str(a.gate) if a.gate else None,gate_sha256=gate_sha)
        return 0


if __name__=='__main__':sys.exit(main())
