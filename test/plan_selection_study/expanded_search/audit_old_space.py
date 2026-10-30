#!/usr/bin/env python3
"""Mark structural membership in the former endpoint-preserving search.

This is not a runtime equivalence proof: generated source, options and complete
plan state still need comparison to claim an old/new implementation is the same.
"""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path


def membership(request, baseline, actual_local=None):
    try:
        n,batch=baseline['n'],baseline['batch']
        k=baseline['structural_features']['kernels'][0]
        d=baseline['precompile_kernels'][0]['raw_descriptor']
        stages=k['stageRadix'];seq=request['radix'];g=request['grouped_batch'];r=request['logical_registers']
        if d['registerBoost'] != 1:
            return {'membership':'not_comparable','reason':'expanded_boost1_cannot_represent_baseline_boost'}
        if (baseline['structural_features']['numAxisUploads']!=1 or len(baseline['structural_features']['kernels'])!=1 or
                d['fftDim']!=n or d['complexSize']!=8 or d['axisSwapped'] not in (0,1)):
            return {'membership':'not_comparable','reason':'baseline_outside_old_enumerator'}
        if (not seq or math.prod(seq)!=n or any(x not in (2,4,8,16,32) for x in seq) or
                seq[0]!=stages[0] or seq[-1]!=stages[-1] or (len(stages)<=2 and seq!=stages)):
            return {'membership':'outside_old_space','reason':'fixed_endpoints_or_two_stage_rule'}
        if g<1 or g>min(batch,1024) or g&(g-1) or batch%g:
            return {'membership':'outside_old_space','reason':'group_not_enumerated'}
        regs=d['registers_per_radix']; active=seq
        if any(regs[x]<=0 or regs[x]%x for x in active):
            return {'membership':'outside_old_space','reason':'old_radix_register_precheck'}
        if (min(regs[x] for x in active)!=d['min_logical_registers_per_thread'] or
                max(regs[x] for x in active)!=d['logical_registers_per_thread']):
            return {'membership':'outside_old_space','reason':'old_register_envelope_precheck'}
        # The expanded candidate uses uniform R and reconstructs the full power-of-two table.
        expanded=[r if i in (2,4,8,16,32) and r%i==0 else 0 for i in range(33)]
        if regs!=expanded or d['min_logical_registers_per_thread']!=r or d['logical_registers_per_thread']!=r:
            return {'membership':'outside_old_space','reason':'register_table_or_envelope_changed'}
        local=k['localSize'];swap=d['axisSwapped'];threads=local[1 if swap else 0]
        if local[2]!=1 or local[0 if swap else 1]!=k['groupedBatch'] or threads!=n//r:
            return {'membership':'outside_old_space','reason':'fft_thread_dimension_changed'}
        shared=baseline['resolved_config']['sharedMemorySize']
        if g*threads>baseline['resolved_config']['maxThreadsNum'] or g*n>shared//8:
            return {'membership':'outside_old_space','reason':'old_thread_or_shared_memory_precheck'}
        block=[g,threads,1] if g>1 and g*n<shared//8 else [threads,g,1]
        if actual_local is not None and block!=actual_local:
            return {'membership':'outside_old_space','reason':'actual_block_differs_from_old_request'}
        return {'membership':'old_structural_subset','reason':None,'expected_old_block':block,
                'caveat':'requires source/options/runtime-state match before claiming execution equivalence'}
    except (KeyError,TypeError,IndexError,ValueError,ZeroDivisionError) as exc:
        return {'membership':'not_comparable','reason':'missing_or_invalid_evidence: '+str(exc)}


def analyze(rows, manifest_dir=None):
    baselines={};requests={};layouts={};groups=defaultdict(list);counts=Counter()
    for x in rows:
        kind=x.get('record_type')
        if kind=='execution' and x.get('phase')=='identity_control' and x.get('candidate_id')=='expanded_default' and x.get('returncode')==0:
            w=x['workload'];baselines[(w['n'],w['batch'])]=x['result']
        elif kind=='candidate_selected':requests[(x['n'],x['batch'],x['candidate_id'])]=x['request']
        elif kind=='candidate_layout':layouts[(x['n'],x['batch'],x['candidate_id'])]=x['local_size']
        elif kind=='candidate_disposition' and x.get('status')=='screen_complete':groups[(x['n'],x['batch'])].append(x)
    selected_coverage=[]
    if manifest_dir is not None:
        for (n,batch),base in sorted(baselines.items()):
            path=Path(manifest_dir)/f'n{n}_b{batch}.jsonl'
            if not path.is_file():
                selected_coverage.append({'n':n,'batch':batch,'status':'manifest_missing'})
                continue
            categories=Counter(); selected_count=0
            with path.open(encoding='utf8') as source:
                for line in source:
                    row=json.loads(line)
                    if row.get('selection')!='selected': continue
                    selected_count+=1;key=(n,batch,row['candidate_id'])
                    mark=membership(row['request'],base,layouts.get(key))
                    categories[mark['membership']]+=1
            selected_coverage.append({'n':n,'batch':batch,'status':'descriptive_only',
                                      'selected_count':selected_count,'structural_membership_counts':dict(categories)})
    cases=[]
    for (n,batch),items in sorted(groups.items()):
        old=[];new=[]
        for x in items:
            key=(n,batch,x['candidate_id'])
            if (n,batch) not in baselines or key not in requests:
                mark={'membership':'not_comparable','reason':'missing_baseline_or_request'}
            else:mark=membership(requests[key],baselines[(n,batch)],layouts.get(key))
            counts[mark['membership']]+=1
            value=x['median_ratio']; entry={'candidate_id':key[2],'screen_ratio':value,**mark}
            if mark['membership']=='old_structural_subset':old.append(entry)
            elif mark['membership']=='outside_old_space':new.append(entry)
        cases.append({'n':n,'batch':batch,'screen_complete':len(items),
                      'old_structural_count':len(old),'expanded_only_count':len(new),
                      'best_old_structural':min(old,key=lambda x:x['screen_ratio']) if old else None,
                      'best_expanded_only':min(new,key=lambda x:x['screen_ratio']) if new else None})
    return {'counts':dict(counts),'cases':cases,'selected_coverage':selected_coverage,
            'scope':'descriptive partition of the same completed screen points; no matched-budget old-search comparison; no source-equivalence claim; incomplete/untested requests excluded'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rows',type=Path);p.add_argument('--manifests-dir',type=Path);a=p.parse_args()
    with a.rows.open(encoding='utf8') as f:rows=[json.loads(x) for x in f if x.strip()]
    print(json.dumps(analyze(rows,a.manifests_dir),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
