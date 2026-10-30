"""Recover an audited gate only after a recorded successful budget finish."""
import argparse
import json
import os
from pathlib import Path

import budget
import device_lock
import runner

HERE=Path(__file__).resolve().parent


def option(argv,name):return argv[argv.index(name)+1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',required=True,type=Path);p.add_argument('--work-id',required=True)
    a=p.parse_args()
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+runner.GPU+'.lock')):
        b=budget.Budget(a.state)
        if b.state['phase']!='confirmation' or b.state['stopped'] or b.state.get('inflight'):
            raise ValueError('confirmation budget unavailable or unresolved work')
        events=[json.loads(x) for x in b.ledger.read_text(encoding='utf8').splitlines()]
        start=[e for e in events if e.get('event')=='work_start_allowed' and e.get('work_id')==a.work_id and e.get('kind') in ('independent_screen_audit','independent_first_audit')]
        end=[e for e in events if e.get('event')=='work_finished' and e.get('work_id')==a.work_id and e.get('status')=='completed']
        if len(start)!=1 or len(end)!=1:raise ValueError('missing unique successful audit budget record')
        s,e=start[0],end[0]
        for name,digest in s['source_hashes'].items():
            if runner.file_sha(HERE/name)!=digest:raise ValueError('audit source changed since execution: '+name)
        pending=Path(e['pending_gate']);final=Path(e['final_gate']);audit=Path(option(s['argv'],'--audit'))
        gate=json.loads(pending.read_text(encoding='utf8'))
        if (runner.file_sha(pending)!=e['pending_gate_sha256'] or runner.file_sha(audit)!=e['audit_sha256'] or
            gate.get('status')!='pass' or gate.get('audit_sha256')!=e['audit_sha256']):
            raise ValueError('completed audit evidence hash mismatch')
        if final.exists():
            if runner.file_sha(final)!=e['pending_gate_sha256']:raise ValueError('existing final gate differs')
        else:os.link(pending,final)
        published=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='independent_audit_gate_published' and x.get('work_id')==a.work_id]
        if published:
            if len(published)!=1 or published[0].get('sha256')!=e['pending_gate_sha256']:raise ValueError('gate publication record conflict')
            return
        b.checkpoint('independent_audit_gate_published',work_id=a.work_id,gate=str(final),sha256=e['pending_gate_sha256'],recovered=True)


if __name__=='__main__':main()
