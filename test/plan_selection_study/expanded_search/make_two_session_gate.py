#!/usr/bin/env python3
"""Create a rotation gate only for independently confirmed two-session wins."""
import argparse
import json
from pathlib import Path

import audit_budget
import audit_confirmation
import audit_two_sessions


def read(path):return audit_confirmation.read(path)
def sha(path):return audit_confirmation.sha(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','protocol','first-gate','first-audit','first-rows','first-guard','first-incidents','first-budget',
                 'second-audit','second-rows','second-guard','second-incidents','second-budget','audit','gate'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous-rows',type=Path,nargs='+',required=True,help='all admission and search execution JSONLs')
    a=p.parse_args()
    if a.audit.exists() or a.gate.exists():raise FileExistsError('audit and gate must be new files')
    plan=json.loads(a.plan.read_text(encoding='utf8'));protocol=json.loads(a.protocol.read_text(encoding='utf8'))
    first_gate=json.loads(a.first_gate.read_text(encoding='utf8'));first_report=json.loads(a.first_audit.read_text(encoding='utf8'))
    second_report=json.loads(a.second_audit.read_text(encoding='utf8'))
    frows=read(a.first_rows);srows=read(a.second_rows);prior=[]
    for path in a.previous_rows:prior.extend(read(path))
    fcomputed=audit_confirmation.audit(frows,read(a.first_guard),read(a.first_incidents),plan,sha(a.plan))
    scomputed=audit_confirmation.audit(srows,read(a.second_guard),read(a.second_incidents),plan,sha(a.plan))
    fbudget=audit_budget.audit(read(a.first_budget),protocol,prior+frows)
    sbudget=audit_budget.audit(read(a.second_budget),protocol,prior+frows+srows)
    header=next((x for x in srows if x.get('record_type')=='confirmation_protocol'),{})
    combined=audit_two_sessions.combine(fcomputed,scomputed,first_gate,header,sha(a.first_gate),sha(a.plan),sha(a.first_audit))
    evidence={'first':{'rows':sha(a.first_rows),'guard':sha(a.first_guard),'incidents':sha(a.first_incidents),'budget':sha(a.first_budget)},
              'second':{'rows':sha(a.second_rows),'guard':sha(a.second_guard),'incidents':sha(a.second_incidents),'budget':sha(a.second_budget)}}
    errors=[]
    if first_gate.get('evidence_sha256')!=evidence['first']:errors.append('first gate evidence hash mismatch')
    if first_report.get('status')!='pass' or first_report.get('first_session')!=fcomputed or first_report.get('budget')!=fbudget:
        errors.append('stored first audit differs from independent recomputation')
    stored_second=second_report.get('second_session',second_report)
    if second_report.get('status')!='pass' or stored_second!=scomputed:errors.append('stored second audit differs from independent recomputation')
    if any(x.get('status')!='pass' for x in (fcomputed,scomputed,fbudget,sbudget,combined)):
        errors.append('a session, budget, or combined independent audit failed')
    if any(x.get('phase')!='confirmation' or x.get('inflight') is not None for x in (fbudget,sbudget)):
        errors.append('budget snapshots are not closed confirmation boundaries')
    audit={'status':'pass' if not errors else 'fail','errors':errors,'first_session':fcomputed,'second_session':scomputed,
           'first_budget':fbudget,'second_budget':sbudget,'combined':combined}
    with a.audit.open('x',encoding='utf8') as f:json.dump(audit,f,ensure_ascii=False,indent=2);f.write('\n')
    candidates={(w['n'],w['batch'],w['candidate']['candidate_id']):w['candidate'] for w in plan['workloads'] if w.get('candidate')}
    confirmed=[]
    for x in combined.get('candidates',[]):
        if x['status']!='confirmed_under_preregistered_rule':continue
        key=(x['n'],x['batch'],x['candidate_id']);c=candidates[key]
        confirmed.append({'n':key[0],'batch':key[1],'candidate_id':key[2],'request':c['request'],
                          'graph_repeats':c['graph_repeats'],'original_signature':c['original_signature'],
                          'candidate_signature':c['candidate_signature']})
    gate={'schema':'expanded-two-session-gate/v1','status':'pass' if not errors else 'fail','plan_sha256':sha(a.plan),'protocol_sha256':sha(a.protocol),
          'first_gate_sha256':sha(a.first_gate),'first_audit_sha256':sha(a.first_audit),'second_audit_sha256':sha(a.second_audit),
          'combined_audit_sha256':sha(a.audit),'session_evidence_sha256':evidence,'binaries_sha256':plan['sha256'],
          'confirmed':confirmed,'scope':'only two-session preregistered winners; empty confirmed means rotation not applicable'}
    with a.gate.open('x',encoding='utf8') as f:json.dump(gate,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':gate['status'],'confirmed_count':len(confirmed),'errors':errors},ensure_ascii=False))
    return 0 if gate['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
