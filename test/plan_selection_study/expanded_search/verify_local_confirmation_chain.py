#!/usr/bin/env python3
"""Verify local copies of the confirmation gate, reports, budgets and binaries."""
import argparse
import json
from pathlib import Path
import audit_rotation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','protocol','gate','first-gate','first-audit','second-audit','combined-audit','budget-replay',
                 'first-rows','first-guard','first-incidents','first-budget','second-rows','second-guard','second-incidents','second-budget',
                 'original-binary','expanded-binary','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();load=audit_rotation.load;sha=audit_rotation.sha;plan=load(a.plan)
    files={name:{key:getattr(a,name+'_'+key) for key in ('rows','guard','incidents','budget')} for name in ('first','second')}
    hashes={'plan':sha(a.plan),'protocol':sha(a.protocol),'first_gate':sha(a.first_gate),'first_audit':sha(a.first_audit),
            'second_audit':sha(a.second_audit),'combined_audit':sha(a.combined_audit),
            'sessions':{name:{key:sha(value) for key,value in values.items()} for name,values in files.items()}}
    combined=load(a.combined_audit);result=audit_rotation.verify_gate(plan,load(a.gate),load(a.first_gate),
                                            load(a.first_audit),load(a.second_audit),combined,files,hashes)
    replay=load(a.budget_replay)
    for name in ('first','second'):
        if replay[name]!=combined[name+'_budget']:result['errors'].append(name+' budget replay differs')
    if {'original':sha(a.original_binary),'expanded':sha(a.expanded_binary)}!=plan['sha256']:
        result['errors'].append('local binary digests differ from frozen plan')
    result['status']='fail' if result['errors'] else 'pass'
    with a.output.open('x',encoding='utf8') as out:json.dump(result,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps({'status':result['status'],'errors':result['errors'],'confirmed':len(result['confirmed'])},ensure_ascii=False))
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
