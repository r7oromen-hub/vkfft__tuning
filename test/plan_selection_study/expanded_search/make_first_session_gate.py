#!/usr/bin/env python3
"""Create a second-session gate from independently audited first-session data."""
import argparse
import json
from pathlib import Path

import audit_budget
import audit_confirmation


def digest(path):return audit_confirmation.sha(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rows','guard','incidents','budget','protocol','plan','audit','gate'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous-rows',type=Path,nargs='+',required=True,
                   help='all earlier admission and search JSONLs whose guard IDs precede this session')
    a=p.parse_args()
    if a.audit.exists() or a.gate.exists():raise FileExistsError('audit and gate must be new files')
    rows=audit_confirmation.read(a.rows);plan=json.loads(a.plan.read_text(encoding='utf8'))
    first=audit_confirmation.audit(rows,audit_confirmation.read(a.guard),audit_confirmation.read(a.incidents),plan,digest(a.plan))
    prior=[]
    for path in a.previous_rows:prior.extend(audit_confirmation.read(path))
    budget=audit_budget.audit(audit_confirmation.read(a.budget),json.loads(a.protocol.read_text(encoding='utf8')),prior+rows)
    errors=[]
    if first.get('status')!='pass' or first.get('session_kind')!='first':errors.append('first-session independent audit failed')
    if budget.get('status')!='pass' or budget.get('phase')!='confirmation' or budget.get('inflight') is not None:
        errors.append('first-session budget snapshot is invalid or has work in flight')
    report={'status':'pass' if not errors else 'fail','errors':errors,'first_session':first,'budget':budget}
    with a.audit.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    eligible=sorted((x for x in first.get('complete',[]) if x['role']=='screen_faster' and x['geometric_mean_ratio']<1),
                    key=lambda x:(not x['passes_practical_and_ci'],x['geometric_mean_ratio'],x['n'],x['batch'],x['candidate_id']))
    gate={'schema':'expanded-first-session-gate/v1','status':'pass' if not errors else 'fail',
          'session_id':first.get('session_id'),'plan_sha256':digest(a.plan),
          'evidence_sha256':{'rows':digest(a.rows),'guard':digest(a.guard),'incidents':digest(a.incidents),'budget':digest(a.budget)},
          'binaries_sha256':plan.get('sha256'),'eligible':eligible,'audit_sha256':digest(a.audit),'counts':first.get('counts',{}),
          'scope':'eligible means screened-faster role and independently audited first-session geometric ratio <1; criteria-passing candidates are ranked first; controls excluded'}
    with a.gate.open('x',encoding='utf8') as f:json.dump(gate,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':gate['status'],'audit':str(a.audit),'gate':str(a.gate),'errors':errors},ensure_ascii=False))
    return 0 if gate['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
