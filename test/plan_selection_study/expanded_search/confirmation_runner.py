"""Rebuild an independently gated plan and launch one guarded confirmation session."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import budget
import confirmation_plan
import device_lock
import runner

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2];GPU=runner.GPU
SOURCES=('confirmation_runner.py','confirmation_plan.py','confirmation_controller.py','phase_gpu_guard.py',
         'search_controller.py','run_suite.py','validation.py','budget.py','device_lock.py')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('state','results','search','gate','guard','incidents','budget-snapshot','protocol','manifest-dir','audit','plan'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--session-kind',required=True,choices=('first','second'))
    p.add_argument('--first-results',type=Path);p.add_argument('--first-gate',type=Path);p.add_argument('--first-audit',type=Path);p.add_argument('--first-guard',type=Path)
    p.add_argument('--first-incidents',type=Path);p.add_argument('--first-budget-snapshot',type=Path)
    a=p.parse_args();env=os.environ.copy();env['PATH']='/usr/local/cuda-13.0/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']='/usr/local/cuda-13.0/lib64:'+env.get('LD_LIBRARY_PATH','')
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+GPU+'.lock')) as lock:
        b=budget.Budget(a.state)
        if b.state['phase']!='confirmation' or b.state['stopped'] or b.state.get('inflight'):raise ValueError('confirmation budget unavailable')
        if b.remaining()<=690:raise ValueError('insufficient budget for guard and a paired round')
        # Independently rebuild the exact plan from the gated raw search data.
        work_id=uuid.uuid4().hex
        if not b.may_start('confirmation_plan_verification',expected_upper_seconds=600,work_id=work_id,
                           session_kind=a.session_kind,search=str(a.search),gate=str(a.gate)):
            raise ValueError('insufficient budget to verify plan')
        try:
            expected=confirmation_plan.verified_plan(a)
        except BaseException as error:
            b.finish_work('exception',work_id=work_id,error=f'{type(error).__name__}: {error}');raise
        b.finish_work('completed',work_id=work_id)
        for name,path in expected['binaries'].items():
            if runner.file_sha(path)!=expected['sha256'][name]:raise ValueError('confirmation binary changed: '+name)
        if a.plan.exists():
            if json.loads(a.plan.read_text(encoding='utf8'))!=expected:raise ValueError('confirmation plan differs from independently reconstructed plan')
        else:
            with a.plan.open('x',encoding='utf8') as out:
                json.dump(expected,out,sort_keys=True,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
        if not any(item.get('candidate') for item in expected['workloads']):
            b.checkpoint('confirmation_no_candidates',plan_sha256=runner.file_sha(a.plan));return 0
        if a.session_kind=='second' and any(x is None for x in (a.first_results,a.first_gate,a.first_audit,a.first_guard,a.first_incidents,a.first_budget_snapshot)):
            raise ValueError('second session needs independently audited first-session evidence')
        if a.session_kind=='first' and any(x is not None for x in (a.first_results,a.first_gate,a.first_audit,a.first_guard,a.first_incidents,a.first_budget_snapshot)):
            raise ValueError('unexpected first-session evidence')
        sources={name:runner.file_sha(HERE/name) for name in SOURCES}
        events=[json.loads(x) for x in b.ledger.read_text(encoding='utf8').splitlines()]
        publications=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='independent_audit_gate_published' and
                      x.get('gate')==str(a.gate) and x.get('sha256')==runner.file_sha(a.gate)]
        if len(publications)!=1:raise ValueError('screen gate lacks unique budgeted publication')
        launches=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='confirmation_guard_launch']
        exits=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='confirmation_guard_exit']
        if any(x.get('session_kind')==a.session_kind for x in launches):
            raise ValueError('confirmation session kind already launched; no selective repeats')
        if a.session_kind=='second':
            first=[x for x in launches if x.get('session_kind')=='first']
            first_exit=[x for x in exits if x.get('session_kind')=='first']
            if (len(first)!=1 or len(first_exit)!=1 or first[0]['session_id']!=first_exit[0]['session_id'] or
                first_exit[0].get('exit_code')!=0):raise ValueError('first guard did not complete exactly once')
            def opt(argv,name):return argv[argv.index(name)+1]
            if any(Path(opt(first[0]['argv'],flag)).resolve()!=path.resolve() for flag,path in
                   (('--output',a.first_results),('--log',a.first_guard),('--incidents',a.first_incidents))):
                raise ValueError('first-session evidence is not the launched session')
            with a.first_results.open('r',encoding='utf8') as f:first_header=json.loads(f.readline())
            gate=json.loads(a.first_gate.read_text(encoding='utf8'))
            evidence={'rows':runner.file_sha(a.first_results),'guard':runner.file_sha(a.first_guard),
                      'incidents':runner.file_sha(a.first_incidents),'budget':runner.file_sha(a.first_budget_snapshot)}
            if (gate.get('schema')!='expanded-first-session-gate/v1' or gate.get('status')!='pass' or
                gate.get('session_id')!=first_header.get('session_id') or gate.get('plan_sha256')!=runner.file_sha(a.plan) or
                gate.get('evidence_sha256')!=evidence or gate.get('binaries_sha256')!=expected['sha256'] or
                gate.get('audit_sha256')!=runner.file_sha(a.first_audit)):
                raise ValueError('first-session gate/evidence mismatch')
            first_publications=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='independent_audit_gate_published' and
                                x.get('gate')==str(a.first_gate) and x.get('sha256')==runner.file_sha(a.first_gate)]
            if len(first_publications)!=1:raise ValueError('first-session gate lacks budgeted publication')
        frozen=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='confirmation_source_freeze']
        if frozen:
            if len(frozen)!=1 or frozen[0]['source_hashes']!=sources or frozen[0]['plan_sha256']!=runner.file_sha(a.plan):
                raise ValueError('confirmation source or plan changed after freeze')
        else:b.checkpoint('confirmation_source_freeze',source_hashes=sources,plan_sha256=runner.file_sha(a.plan))
        if b.remaining()<=690:raise ValueError('insufficient budget after audit and plan generation')
        token=uuid.uuid4().hex
        cmd=[sys.executable,str(HERE/'phase_gpu_guard.py'),'--kind','confirmation','--gpu',GPU,
             '--log',str(a.results/f'confirmation_{a.session_kind}_{token}.guard.jsonl'),
             '--incidents',str(a.results/f'confirmation_{a.session_kind}_{token}.incidents.jsonl'),
             '--budget-dir',str(a.state),'--plan',str(a.plan),'--output',str(a.results/f'confirmation_{a.session_kind}_{token}.jsonl'),
             '--session-kind',a.session_kind]
        if a.session_kind=='second':
            for name in ('first_results','first_gate','first_audit','first_guard','first_incidents','first_budget_snapshot'):
                cmd.extend(['--'+name.replace('_','-'),str(getattr(a,name))])
        b.checkpoint('confirmation_guard_launch',session_id=token,session_kind=a.session_kind,argv=cmd,
                     plan_sha256=runner.file_sha(a.plan),source_hashes=sources)
        proc=subprocess.run(cmd,cwd=ROOT,env=env,pass_fds=(lock.fd,))
        b=budget.Budget(a.state);b.checkpoint('confirmation_guard_exit',session_id=token,session_kind=a.session_kind,exit_code=proc.returncode)
        return proc.returncode


if __name__=='__main__':sys.exit(main())
