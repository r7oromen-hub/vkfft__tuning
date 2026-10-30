#!/usr/bin/env python3
"""Create an evidence-bound confirmation gate from a finished search snapshot."""
import argparse
import json
from pathlib import Path

import audit_budget
import audit_search


def digest(path): return audit_search.sha(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rows','guard','incidents','budget','protocol','manifests','audit','gate'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous-rows',type=Path,nargs='+',required=True,
                   help='all earlier admission JSONLs whose budgeted run IDs precede search')
    a=p.parse_args()
    if a.audit.exists() or a.gate.exists():
        raise FileExistsError('audit and gate must be new files')
    rows=audit_search.read(a.rows);guard=audit_search.read(a.guard);incidents=audit_search.read(a.incidents)
    prior=[]
    for path in a.previous_rows: prior.extend(audit_search.read(path))
    protocol=json.loads(a.protocol.read_text(encoding='utf8'))
    search=audit_search.audit(rows,guard,incidents,a.manifests)
    budget=audit_budget.audit(audit_search.read(a.budget),protocol,prior+rows)
    headers=[r for r in rows if r.get('record_type')=='search_protocol']
    errors=[]
    if len(headers)!=1: errors.append('missing unique search header')
    elif headers[0].get('protocol')!=protocol or headers[0].get('protocol_sha256')!=digest(a.protocol):
        errors.append('search protocol does not match frozen protocol')
    manifest_hashes={}
    for r in rows:
        if r.get('record_type')=='workload_status' and r.get('status')=='identity_passed':
            key=f"n{r['n']}_b{r['batch']}";path=a.manifests/(key+'.jsonl')
            if key in manifest_hashes or not path.is_file(): errors.append('missing or duplicate manifest '+key)
            else: manifest_hashes[key]=digest(path)
    if search.get('status')!='pass' or budget.get('status')!='pass': errors.append('independent search or budget audit failed')
    if budget.get('inflight') is not None or budget.get('phase')!='search': errors.append('budget snapshot is not a closed search boundary')
    report={'status':'pass' if not errors else 'fail','errors':errors,'search':search,'budget':budget}
    with a.audit.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    gate={'schema':'expanded-screen-gate/v1','status':'pass' if not errors else 'fail',
          'search_session':headers[0].get('session_id') if len(headers)==1 else None,
          'evidence_sha256':{'rows':digest(a.rows),'guard':digest(a.guard),'incidents':digest(a.incidents),
                             'budget':digest(a.budget),'protocol':digest(a.protocol)},
          'manifest_sha256':manifest_hashes,'binaries_sha256':headers[0].get('sha256') if len(headers)==1 else None,
          'audit_sha256':digest(a.audit),'counts':search.get('counts',{}),
          'scope':'complete screened candidates only; selected-but-unstarted/partial are not invalid; no confirmed speedup'}
    with a.gate.open('x',encoding='utf8') as f:json.dump(gate,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':gate['status'],'audit':str(a.audit),'gate':str(a.gate),'errors':errors},ensure_ascii=False))
    return 0 if gate['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
