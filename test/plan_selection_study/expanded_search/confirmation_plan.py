"""Freeze proposed confirmation cases from complete, raw paired screen rows."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_relevant(path):
    """Bound retained memory to screen evidence rather than all validation rows."""
    rows=[]
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);kind=r.get('record_type')
            if (kind in ('search_protocol','search_complete','screen_pair','workload_status') or
                kind=='candidate_disposition' and r.get('status')=='screen_complete' or
                kind=='execution' and r.get('phase')=='screen_timing'):
                rows.append(r)
    return rows


def make(rows, source_sha256, gate_sha256=None):
    protocols=[r for r in rows if r.get('record_type')=='search_protocol']
    terminals=[r for r in rows if r.get('record_type')=='search_complete']
    if len(protocols)!=1 or len(terminals)!=1 or terminals[0].get('status') not in ('budget_exhausted','completed_selected'):
        raise ValueError('search must finish cleanly or at budget boundary')
    protocol=protocols[0];p=protocol['protocol']
    executions={r['guard_run_id']:r for r in rows if r.get('record_type')=='execution' and r.get('guard_run_id')}
    if len(executions)!=sum(r.get('record_type')=='execution' and bool(r.get('guard_run_id')) for r in rows):
        raise ValueError('duplicate execution guard ID')
    groups={}
    for d in (r for r in rows if r.get('record_type')=='candidate_disposition' and r.get('status')=='screen_complete'):
        key=(d['n'],d['batch'],d['candidate_id'])
        if key in groups:raise ValueError('duplicate candidate disposition')
        pairs=[r for r in rows if r.get('record_type')=='screen_pair' and (r['n'],r['batch'],r['candidate_id'])==key]
        if len(pairs)!=3 or sorted(x['round'] for x in pairs)!=[0,1,2]:raise ValueError('missing or duplicate screen rounds')
        ratios=[]
        for pair in sorted(pairs,key=lambda r:r['round']):
            o=executions[pair['original_guard_run_id']];c=executions[pair['candidate_guard_run_id']]
            if not (o['guard_clean'] is True and c['guard_clean'] is True and o['candidate_id']=='original' and
                    c['candidate_id']==d['candidate_id'] and o['phase']==c['phase']=='screen_timing' and
                    o['round']==c['round']==pair['round'] and {o['order'],c['order']}=={0,1} and
                    (o['workload']['n'],o['workload']['batch'])==(d['n'],d['batch']) and
                    (c['workload']['n'],c['workload']['batch'])==(d['n'],d['batch'])):
                raise ValueError('screen pair execution mismatch')
            om=statistics.median(o['result']['exec_ms_samples']);cm=statistics.median(c['result']['exec_ms_samples'])
            ratio=cm/om
            if not (math.isfinite(ratio) and ratio>0 and math.isclose(om,pair['original_median_ms'],rel_tol=1e-9) and
                    math.isclose(cm,pair['candidate_median_ms'],rel_tol=1e-9) and
                    math.isclose(ratio,pair['candidate_over_original'],rel_tol=1e-9)):
                raise ValueError('screen pair summary mismatch')
            ratios.append(ratio)
        if (len(d['pair_ratios'])!=3 or any(not math.isclose(x,y,rel_tol=1e-9) for x,y in zip(ratios,d['pair_ratios'])) or
            not math.isclose(statistics.median(ratios),d['median_ratio'],rel_tol=1e-9)):
            raise ValueError('screen disposition summary mismatch')
        groups[key]=d
    workloads=[]
    for batch in p['batches']:
        for n in p['lengths']:
            options=[d for (nn,bb,_),d in groups.items() if (nn,bb)==(n,batch)]
            best=min(options,key=lambda d:(d['median_ratio'],d['candidate_id'])) if options else None
            workloads.append(dict(n=n,batch=batch,screened_count=len(options),best=best))
    faster=sorted((w for w in workloads if w['best'] is not None and w['best']['median_ratio']<1),
                  key=lambda w:(w['best']['median_ratio'],w['n'],w['batch'],w['best']['candidate_id']))
    # At most two near-tie examples where the default won screening, to check
    # the default fallback; these are not positive claims.
    controls=sorted((w for w in workloads if w['best'] is not None and w['best']['median_ratio']>=1),
                    key=lambda w:(w['best']['median_ratio'],w['n'],w['batch'],w['best']['candidate_id']))[:2]
    order=faster+controls
    selected={(w['n'],w['batch']):('screen_faster' if w in faster else 'default_faster_control') for w in order}
    for item in workloads:
        role=selected.get((item['n'],item['batch']));best=item.pop('best')
        item['candidate']=best if role else None;item['role']=role
        item['fallback_reason']=None if role else ('no_screened_candidate' if best is None else 'default_retained_not_selected_for_control')
    confirmation_order=[{'n':w['n'],'batch':w['batch'],'candidate_id':w['candidate']['candidate_id'],'role':w['role']} for w in order]
    return dict(schema='expanded-confirmation-plan/v1',search_sha256=source_sha256,screen_gate_sha256=gate_sha256,search_session=protocol['session_id'],
                binaries=protocol['binaries'],sha256=protocol['sha256'],selection_seed=p['selection_seed'],
                workloads=workloads,confirmation_order=confirmation_order,
                interpretation='faster screened winners first, then at most two default-faster controls; no performance conclusion until independent confirmation')


def verified_plan(a):
    rows=read_relevant(a.search)
    gate=json.loads(a.gate.read_text(encoding='utf8'))
    expected={'rows':digest(a.search),'guard':digest(a.guard),'incidents':digest(a.incidents),
              'budget':digest(a.budget_snapshot),'protocol':digest(a.protocol)}
    header=next((r for r in rows if r.get('record_type')=='search_protocol'),None)
    if (gate.get('schema')!='expanded-screen-gate/v1' or gate.get('status')!='pass' or
        gate.get('evidence_sha256')!=expected or gate.get('audit_sha256')!=digest(a.audit) or
        not header or gate.get('search_session')!=header['session_id'] or gate.get('binaries_sha256')!=header['sha256']):
        raise ValueError('independent screen gate or evidence hash mismatch')
    expected_manifests={f"n{r['n']}_b{r['batch']}":r['manifest_sha256'] for r in rows
                        if r.get('record_type')=='workload_status' and r.get('status')=='identity_passed'}
    if gate.get('manifest_sha256')!=expected_manifests:raise ValueError('manifest set disagrees with screened workloads')
    for name,h in expected_manifests.items():
        if digest(a.manifest_dir/(name+'.jsonl'))!=h:raise ValueError('manifest hash mismatch: '+name)
    return make(rows,digest(a.search),digest(a.gate))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--search',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--gate',required=True,type=Path);p.add_argument('--guard',required=True,type=Path);p.add_argument('--incidents',required=True,type=Path)
    p.add_argument('--budget-snapshot',required=True,type=Path);p.add_argument('--protocol',required=True,type=Path)
    p.add_argument('--manifest-dir',required=True,type=Path);p.add_argument('--audit',required=True,type=Path)
    a=p.parse_args();result=verified_plan(a)
    with a.output.open('x',encoding='utf8') as out:json.dump(result,out,sort_keys=True,indent=2);out.write('\n')


if __name__=='__main__':main()
