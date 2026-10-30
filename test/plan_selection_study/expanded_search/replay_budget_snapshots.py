#!/usr/bin/env python3
"""Replay three immutable ledgers against all preceding guarded execution IDs."""
import argparse
import json
from pathlib import Path
import audit_budget


def ids(paths):
    out=[]
    for path in paths:
        with Path(path).open(encoding='utf8') as source:
            for line in source:
                row=json.loads(line)
                if row.get('record_type')=='execution':out.append({'record_type':'execution','guard_run_id':row.get('guard_run_id')})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','search-budget','first-budget','second-budget','search-rows','first-rows','second-rows','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--admission-rows',nargs='+',required=True,type=Path);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text(encoding='utf8'));prior=ids(a.admission_rows+[a.search_rows]);results={}
    for name,ledger_path,extra in (('search',a.search_budget,[]),('first',a.first_budget,[a.first_rows]),('second',a.second_budget,[a.second_rows])):
        prior+=ids(extra)
        with ledger_path.open(encoding='utf8') as source:ledger=[json.loads(x) for x in source if x.strip()]
        results[name]=audit_budget.audit(ledger,protocol,prior)
    with a.output.open('x',encoding='utf8') as out:json.dump(results,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps(results,ensure_ascii=False,indent=2));return 0 if all(x['status']=='pass' for x in results.values()) else 1


if __name__=='__main__':raise SystemExit(main())
