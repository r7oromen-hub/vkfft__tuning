#!/usr/bin/env python3
"""Combine independently audited confirmation sessions without post-hoc promotion."""
import argparse
import json
from pathlib import Path

import audit_confirmation


def combine(first, second, first_gate, second_header, gate_sha, plan_sha, first_audit_sha):
    errors=[]
    if first.get('status')!='pass' or second.get('status')!='pass':errors.append('a session did not pass independent audit')
    if first.get('session_kind')!='first' or second.get('session_kind')!='second' or not first.get('session_id') or first.get('session_id')==second.get('session_id'):
        errors.append('sessions are missing, same or have wrong roles')
    if (first_gate.get('schema')!='expanded-first-session-gate/v1' or first_gate.get('status')!='pass' or
            first_gate.get('session_id')!=first.get('session_id') or first_gate.get('plan_sha256')!=plan_sha or
            first_gate.get('audit_sha256')!=first_audit_sha):
        errors.append('first-session gate does not bind the audited session/plan')
    if (second_header.get('session_id')!=second.get('session_id') or second_header.get('plan_sha256')!=plan_sha or
            second_header.get('first_gate_sha256')!=gate_sha or
            second_header.get('first_results_sha256')!=first_gate.get('evidence_sha256',{}).get('rows')):
        errors.append('second session does not bind the first-session evidence/gate')
    key=lambda x:(x['n'],x['batch'],x['candidate_id'])
    f={key(x):x for x in first.get('complete',[])};s={key(x):x for x in second.get('complete',[])}
    eligible={key(x):x for x in first_gate.get('eligible',[])}
    if len(f)!=len(first.get('complete',[])) or len(s)!=len(second.get('complete',[])) or len(eligible)!=len(first_gate.get('eligible',[])):
        errors.append('duplicate candidate in confirmation audit or gate')
    if not set(eligible)<=set(f) or not set(s)<=set(eligible):errors.append('second session candidate not independently eligible')
    for k,item in eligible.items():
        if k in f and (item.get('role')!='screen_faster' or item.get('geometric_mean_ratio')!=f[k].get('geometric_mean_ratio') or
                item.get('passes_practical_and_ci') is not f[k].get('passes_practical_and_ci')):
            errors.append(f'{k}: eligible summary differs from first audit')
    ranked=sorted(first_gate.get('eligible',[]),key=lambda x:(not x['passes_practical_and_ci'],x['geometric_mean_ratio'],x['n'],x['batch'],x['candidate_id']))
    observed=second.get('candidate_order_observed',second.get('complete',[]))
    if [key(x) for x in observed] != [key(x) for x in ranked[:len(observed)]]:
        errors.append('second-session candidate order differs from frozen first-session ranking')
    result=[]
    for k in sorted(set(f)|set(s)):
        a,b=f.get(k),s.get(k)
        confirmed=bool(a and b and a.get('role')=='screen_faster' and a.get('passes_practical_and_ci') is True and b.get('passes_practical_and_ci') is True)
        result.append({'n':k[0],'batch':k[1],'candidate_id':k[2],
                       'status':'default_faster_control' if a and a.get('role')=='default_faster_control' else
                                'confirmed_under_preregistered_rule' if confirmed else 'unconfirmed',
                       'first':a,'second':b})
    return {'status':'fail' if errors else 'pass','errors':errors,'candidates':result,
            'scope':'same-GPU session evidence; block t intervals do not correct selection/multiple comparisons or establish cross-device generalization'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('first-audit','second-audit','first-gate','second-rows','plan'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();load=lambda path:json.loads(path.read_text(encoding='utf8'))
    second_header=next(x for x in audit_confirmation.read(a.second_rows) if x.get('record_type')=='confirmation_protocol')
    first_report=load(a.first_audit); second_report=load(a.second_audit)
    first=first_report.get('first_session',first_report);second=second_report.get('second_session',second_report)
    result=combine(first,second,load(a.first_gate),second_header,
                   audit_confirmation.sha(a.first_gate),audit_confirmation.sha(a.plan),audit_confirmation.sha(a.first_audit))
    if first_report.get('status')!='pass' or second_report.get('status')!='pass':
        result['status']='fail';result['errors'].append('first or second audit wrapper failed')
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
