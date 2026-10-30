"""Close a clean search and begin confirmation without resetting the total clock."""
import argparse
import json
import os
from pathlib import Path

import budget
import confirmation_plan
import device_lock
import runner


def last_json(path):
    # Read only a bounded tail; the last record is a compact terminal record.
    with path.open('rb') as f:
        f.seek(0,2);size=f.tell();f.seek(max(0,size-1024*1024));data=f.read()
    lines=[x for x in data.splitlines() if x.strip()]
    if not lines:raise ValueError('missing terminal record')
    return json.loads(lines[-1])


def first_json(path):
    with path.open('r',encoding='utf8') as f:return json.loads(f.readline())


def option(argv,name):
    i=argv.index(name)
    return argv[i+1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',required=True,type=Path)
    p.add_argument('--rows',required=True,type=Path);p.add_argument('--guard',required=True,type=Path)
    p.add_argument('--incidents',required=True,type=Path);p.add_argument('--budget-snapshot',required=True,type=Path)
    a=p.parse_args()
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+runner.GPU+'.lock')):
        b=budget.Budget(a.state)
        if b.state['phase'] not in ('search','confirmation') or b.state['stopped'] or b.state.get('inflight'):raise ValueError('search is not cleanly stopped')
        head=first_json(a.rows);end=last_json(a.rows);gs=first_json(a.guard);g=last_json(a.guard)
        if (end.get('record_type')!='search_complete' or end.get('status') not in ('completed_selected','budget_exhausted') or
            g.get('event')!='guard_complete' or g.get('success') is not True or g.get('clean') is not True or
            g.get('script_exit_code')!=0 or a.incidents.stat().st_size!=0 or
            head.get('record_type')!='search_protocol' or not head.get('session_id') or
            gs.get('event')!='guard_start' or gs.get('guard_id')!=g.get('guard_id') or
            gs.get('gpu_uuid')!=runner.GPU or g.get('gpu_uuid')!=runner.GPU or
            not gs.get('script','').endswith('/search_controller.py')):
            raise ValueError('search/guard did not finish cleanly; independent investigation required')
        events=[json.loads(x) for x in b.ledger.read_text(encoding='utf8').splitlines()]
        starts=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='search_guard_launch']
        exits=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='search_guard_exit']
        if (len(starts)!=1 or len(exits)!=1 or exits[0].get('exit_code')!=0 or
            starts[0].get('session_id')!=exits[0].get('session_id')):raise ValueError('missing or mismatched successful launcher boundary')
        argv=starts[0]['argv']
        if (Path(option(argv,'--output')).resolve()!=a.rows.resolve() or Path(option(argv,'--log')).resolve()!=a.guard.resolve() or
            Path(option(argv,'--incidents')).resolve()!=a.incidents.resolve() or
            Path(option(gs['script_args'],'--output')).resolve()!=a.rows.resolve()):raise ValueError('search launcher/guard paths differ')
        with a.rows.open('r',encoding='utf8') as f:
            next(f)
            second=json.loads(next(f))
        if second.get('record_type')=='execution' and (second.get('guard_id')!=gs['guard_id'] or second.get('session_id')!=head['session_id']):
            raise ValueError('search controller and guard identities differ')
        # The snapshot captures the complete search ledger before the advance.
        # If interrupted after fsync, it is accepted only if it is the exact
        # pre-advance ledger; no file is overwritten on recovery.
        ledger=b.ledger.read_bytes()
        if a.budget_snapshot.exists():
            snapshot=a.budget_snapshot.read_bytes()
            if b.state['phase']=='search' and snapshot!=ledger:raise ValueError('existing snapshot differs from search ledger')
            if b.state['phase']=='confirmation':
                if not ledger.startswith(snapshot):raise ValueError('snapshot is not ledger prefix')
                rest=ledger[len(snapshot):].splitlines()
                first_after=json.loads(rest[0]) if rest else {}
                if first_after.get('event')!='phase_advance' or first_after.get('previous_phase')!='search':
                    raise ValueError('snapshot does not end at search/confirmation boundary')
        else:
            if b.state['phase']!='search':raise ValueError('confirmation started without budget snapshot')
            with a.budget_snapshot.open('xb') as out:
                out.write(ledger);out.flush();os.fsync(out.fileno())
        if b.state['phase']=='search' and not b.advance('search_guard_complete'):raise ValueError('global deadline passed')
        existing=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='search_budget_snapshot']
        if existing:
            expected={'sha256':confirmation_plan.digest(a.budget_snapshot),'search_session':head['session_id'],'guard_id':gs['guard_id'],
                      'rows_sha256':confirmation_plan.digest(a.rows),'guard_sha256':confirmation_plan.digest(a.guard),
                      'incidents_sha256':confirmation_plan.digest(a.incidents)}
            if len(existing)!=1 or any(existing[0].get(k)!=v for k,v in expected.items()):raise ValueError('snapshot checkpoint mismatch')
            return
        b.checkpoint('search_budget_snapshot',path=str(a.budget_snapshot),sha256=runner.file_sha(a.budget_snapshot),
                     search_session=head['session_id'],guard_id=gs['guard_id'],rows_sha256=confirmation_plan.digest(a.rows),
                     guard_sha256=confirmation_plan.digest(a.guard),incidents_sha256=confirmation_plan.digest(a.incidents))


if __name__=='__main__':main()
