"""Freeze the second confirmation budget boundary after its guarded exit."""
import argparse
import json
import os
from pathlib import Path

import budget
import confirmation_plan
import device_lock
import runner
from start_confirmation_phase import first_json,last_json,option


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('state','rows','guard','incidents','snapshot'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+runner.GPU+'.lock')):
        b=budget.Budget(a.state)
        if b.state['phase']!='confirmation' or b.state['stopped'] or b.state.get('inflight'):raise ValueError('confirmation is not at clean boundary')
        head=first_json(a.rows);end=last_json(a.rows);gs=first_json(a.guard);ge=last_json(a.guard)
        if (head.get('record_type')!='confirmation_protocol' or head.get('session_kind')!='second' or
            end.get('record_type')!='confirmation_complete' or end.get('session_id')!=head.get('session_id') or
            end.get('status') not in ('completed','budget_exhausted') or gs.get('event')!='guard_start' or
            ge.get('event')!='guard_complete' or gs.get('guard_id')!=ge.get('guard_id') or
            gs.get('gpu_uuid')!=runner.GPU or ge.get('gpu_uuid')!=runner.GPU or ge.get('success') is not True or
            ge.get('clean') is not True or ge.get('script_exit_code')!=0 or a.incidents.stat().st_size or
            not gs.get('script','').endswith('/confirmation_controller.py')):raise ValueError('second guard/session not cleanly complete')
        events=[json.loads(x) for x in b.ledger.read_text(encoding='utf8').splitlines()]
        starts=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='confirmation_guard_launch' and x.get('session_kind')=='second']
        exits=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='confirmation_guard_exit' and x.get('session_kind')=='second']
        if len(starts)!=1 or len(exits)!=1 or starts[0]['session_id']!=exits[0]['session_id'] or exits[0]['exit_code']!=0:
            raise ValueError('second launcher boundary missing or duplicated')
        argv=starts[0]['argv']
        if any(Path(option(argv,flag)).resolve()!=path.resolve() for flag,path in (('--output',a.rows),('--log',a.guard),('--incidents',a.incidents))):
            raise ValueError('second launcher paths differ')
        if Path(option(gs['script_args'],'--output')).resolve()!=a.rows.resolve():raise ValueError('second guard output path differs')
        with a.rows.open('r',encoding='utf8') as f:
            next(f);second=json.loads(next(f))
        if second.get('record_type')=='execution' and (second.get('guard_id')!=gs['guard_id'] or second.get('session_id')!=head['session_id']):
            raise ValueError('second controller and guard identities differ')
        ledger=b.ledger.read_bytes();existing=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='second_confirmation_budget_snapshot']
        if a.snapshot.exists():
            snapshot=a.snapshot.read_bytes()
            if existing:
                expected={'sha256':confirmation_plan.digest(a.snapshot),'rows_sha256':confirmation_plan.digest(a.rows),
                          'guard_sha256':confirmation_plan.digest(a.guard),'incidents_sha256':confirmation_plan.digest(a.incidents),
                          'confirmation_session':head['session_id'],'guard_id':gs['guard_id']}
                if len(existing)!=1 or not ledger.startswith(snapshot) or any(existing[0].get(k)!=v for k,v in expected.items()):raise ValueError('snapshot checkpoint mismatch')
                return
            if snapshot!=ledger:raise ValueError('existing snapshot differs from ledger boundary')
        else:
            with a.snapshot.open('xb') as out:out.write(ledger);out.flush();os.fsync(out.fileno())
        b.checkpoint('second_confirmation_budget_snapshot',path=str(a.snapshot),sha256=confirmation_plan.digest(a.snapshot),
                     confirmation_session=head['session_id'],guard_id=gs['guard_id'],rows_sha256=confirmation_plan.digest(a.rows),
                     guard_sha256=confirmation_plan.digest(a.guard),incidents_sha256=confirmation_plan.digest(a.incidents))


if __name__=='__main__':main()
