#!/usr/bin/env python3
"""Freeze explicit winners and negative controls after re-auditing screen evidence."""
import argparse
import json
from pathlib import Path

import evidence


def build(spec, paths, binaries):
    if set(spec)!={'gpu','cases'}: raise ValueError('spec must contain exactly gpu and cases')
    cases=[dict(case,role=case.get('role','winner')) for case in spec['cases']]
    plan={'schema':'frontier-confirmation-plan/v1','gpu':spec['gpu'],'cases':cases,
          'source_sha256':{name:evidence.frontier.sha(path) for name,path in paths.items()}}
    proof=evidence.audit_screen(plan,paths,binaries)
    plan['screen_execution_commit']=proof['screen_contract']['execution_commit']
    plan['screen_binaries']=proof['screen_contract']['binaries']
    plan['screen_winners']=[case for case in proof['cases'] if case['role']=='winner']
    plan['screen_negative_controls']=[case for case in proof['cases'] if case['role']=='negative_control']
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for name in ('rows','guard','incidents','budget'):p.add_argument('--screen-'+name,type=Path,required=True)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--expanded',type=Path,required=True)
    args=p.parse_args();spec=json.loads(args.spec.read_text(encoding='utf8'))
    paths={name:getattr(args,'screen_'+name) for name in ('rows','guard','incidents','budget')}
    plan=build(spec,paths,{'original':args.original,'expanded':args.expanded})
    plan['selection_spec_sha256']=evidence.frontier.sha(args.spec)
    with args.output.open('x',encoding='utf8') as out:
        json.dump(plan,out,sort_keys=True,indent=2,allow_nan=False);out.write('\n')


if __name__=='__main__':main()
