"""Recover publication of a completed second or two-session CPU audit."""
import argparse
import json
import os
from pathlib import Path

import budget
import device_lock
import runner

HERE=Path(__file__).resolve().parent


def publish(source,target,digest):
    source,target=Path(source),Path(target)
    if runner.file_sha(source)!=digest:raise ValueError('pending evidence hash mismatch')
    if target.exists():
        if runner.file_sha(target)!=digest:raise ValueError('existing published evidence differs')
    else:os.link(source,target)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',required=True,type=Path);p.add_argument('--work-id',required=True)
    a=p.parse_args()
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+runner.GPU+'.lock')):
        b=budget.Budget(a.state)
        if b.state['phase']!='confirmation' or b.state['stopped'] or b.state.get('inflight'):raise ValueError('confirmation unavailable')
        events=[json.loads(x) for x in b.ledger.read_text(encoding='utf8').splitlines()]
        starts=[x for x in events if x.get('event')=='work_start_allowed' and x.get('work_id')==a.work_id and x.get('kind') in ('independent_second_audit','independent_two_audit')]
        ends=[x for x in events if x.get('event')=='work_finished' and x.get('work_id')==a.work_id and x.get('status')=='completed']
        if len(starts)!=1 or len(ends)!=1:raise ValueError('missing unique successful budgeted audit')
        for name,digest in starts[0]['source_hashes'].items():
            if runner.file_sha(HERE/name)!=digest:raise ValueError('source changed since audit: '+name)
        end=ends[0];report=json.loads(Path(end['pending_audit']).read_text(encoding='utf8'))
        if report.get('status')!='pass':raise ValueError('pending audit did not pass')
        if end.get('pending_gate'):
            gate=json.loads(Path(end['pending_gate']).read_text(encoding='utf8'))
            if gate.get('status')!='pass' or gate.get('combined_audit_sha256')!=end['audit_sha256']:raise ValueError('pending gate is invalid')
        publish(end['pending_audit'],end['final_audit'],end['audit_sha256'])
        if end.get('pending_gate'):publish(end['pending_gate'],end['final_gate'],end['gate_sha256'])
        done=[x for x in events if x.get('event')=='checkpoint' and x.get('kind')=='final_confirmation_audit_published' and x.get('work_id')==a.work_id]
        if done:
            if len(done)!=1 or done[0].get('audit_sha256')!=end['audit_sha256'] or done[0].get('gate_sha256')!=end.get('gate_sha256'):
                raise ValueError('publication checkpoint conflict')
            return
        b.checkpoint('final_confirmation_audit_published',work_id=a.work_id,audit=end['final_audit'],audit_sha256=end['audit_sha256'],
                     gate=end.get('final_gate'),gate_sha256=end.get('gate_sha256'),recovered=True)


if __name__=='__main__':main()
