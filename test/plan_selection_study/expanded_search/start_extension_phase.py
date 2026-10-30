"""Advance a fully audited two-session confirmation into extension budget."""
import argparse
import json
import os
from pathlib import Path

import budget
import confirmation_plan
import device_lock
import runner
from start_confirmation_phase import option


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','gate','plan','protocol','first-gate','first-audit','second-audit','combined-audit',
                 'first-rows','first-guard','first-incidents','first-budget','second-rows','second-guard','second-incidents','second-budget','snapshot'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+runner.GPU+'.lock')):
        b=budget.Budget(a.state)
        if b.state['phase'] not in ('confirmation','extension') or b.state['stopped'] or b.state.get('inflight'):
            raise ValueError('confirmation not at closed boundary')
        gate=json.loads(a.gate.read_text(encoding='utf8'));plan=json.loads(a.plan.read_text(encoding='utf8'))
        evidence={}
        for kind in ('first','second'):
            evidence[kind]={x:confirmation_plan.digest(getattr(a,kind+'_'+x)) for x in ('rows','guard','incidents','budget')}
        expected={'plan_sha256':confirmation_plan.digest(a.plan),'protocol_sha256':confirmation_plan.digest(a.protocol),
                  'first_gate_sha256':confirmation_plan.digest(a.first_gate),'first_audit_sha256':confirmation_plan.digest(a.first_audit),
                  'second_audit_sha256':confirmation_plan.digest(a.second_audit),'combined_audit_sha256':confirmation_plan.digest(a.combined_audit),
                  'session_evidence_sha256':evidence,'binaries_sha256':plan['sha256']}
        if gate.get('schema')!='expanded-two-session-gate/v1' or gate.get('status')!='pass' or any(gate.get(k)!=v for k,v in expected.items()):
            raise ValueError('two-session gate or evidence mismatch')
        for name,path in plan['binaries'].items():
            if runner.file_sha(path)!=plan['sha256'][name]:raise ValueError('confirmed binary changed')
        events=[json.loads(x) for x in b.ledger.read_text(encoding='utf8').splitlines()]
        publications=[e for e in events if e.get('event')=='checkpoint' and e.get('kind')=='final_confirmation_audit_published' and
                      e.get('gate')==str(a.gate) and e.get('gate_sha256')==confirmation_plan.digest(a.gate)]
        if len(publications)!=1:raise ValueError('two-session gate lacks unique budgeted publication')
        for kind in ('first','second'):
            starts=[e for e in events if e.get('event')=='checkpoint' and e.get('kind')=='confirmation_guard_launch' and e.get('session_kind')==kind]
            ends=[e for e in events if e.get('event')=='checkpoint' and e.get('kind')=='confirmation_guard_exit' and e.get('session_kind')==kind]
            if len(starts)!=1 or len(ends)!=1 or starts[0]['session_id']!=ends[0]['session_id'] or ends[0].get('exit_code')!=0:
                raise ValueError(kind+' session launcher not uniquely successful')
            argv=starts[0]['argv']
            for flag,name in (('--output','rows'),('--log','guard'),('--incidents','incidents')):
                if Path(option(argv,flag)).resolve()!=getattr(a,kind+'_'+name).resolve():raise ValueError(kind+' session evidence path mismatch')
        ledger=b.ledger.read_bytes()
        if a.snapshot.exists():
            snapshot=a.snapshot.read_bytes()
            if b.state['phase']=='confirmation' and snapshot!=ledger:raise ValueError('existing snapshot does not match confirmation boundary')
            if b.state['phase']=='extension':
                if not ledger.startswith(snapshot):raise ValueError('snapshot not ledger prefix')
                rest=ledger[len(snapshot):].splitlines();first=json.loads(rest[0]) if rest else {}
                if first.get('event')!='phase_advance' or first.get('previous_phase')!='confirmation':raise ValueError('snapshot does not end at phase boundary')
        else:
            if b.state['phase']!='confirmation':raise ValueError('extension has no confirmation snapshot')
            with a.snapshot.open('xb') as out:out.write(ledger);out.flush();os.fsync(out.fileno())
        if b.state['phase']=='confirmation' and not b.advance('two_session_audit_complete'):raise ValueError('global deadline passed')
        existing=[e for e in events if e.get('event')=='checkpoint' and e.get('kind')=='confirmation_budget_snapshot_final']
        if existing:
            if (len(existing)!=1 or existing[0].get('sha256')!=confirmation_plan.digest(a.snapshot) or
                existing[0].get('gate_sha256')!=confirmation_plan.digest(a.gate) or
                existing[0].get('confirmed_count')!=len(gate['confirmed'])):raise ValueError('snapshot checkpoint mismatch')
            return
        b.checkpoint('confirmation_budget_snapshot_final',path=str(a.snapshot),sha256=confirmation_plan.digest(a.snapshot),
                     gate_sha256=confirmation_plan.digest(a.gate),confirmed_count=len(gate['confirmed']))


if __name__=='__main__':main()
