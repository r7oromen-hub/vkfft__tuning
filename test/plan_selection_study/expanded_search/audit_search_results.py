#!/usr/bin/env python3
"""Streaming descriptive summary of completed screens; no confirmation claim."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import statistics

import audit_old_space
import enumerate_candidates


def summarize(path, manifests):
    compact=[];groups=defaultdict(list);counts=Counter();terminal=[];selected=[];dispositions=set();started=set();described=set();validated=defaultdict(set)
    with Path(path).open(encoding='utf8') as source:
        for line in source:
            row=json.loads(line);kind=row.get('record_type')
            if kind=='candidate_disposition':
                status=row.get('status');counts[status]+=1
                key=(row['n'],row['batch'],row['candidate_id'])
                if key in dispositions:raise ValueError('duplicate disposition')
                dispositions.add(key)
                if status=='screen_complete':
                    ratio=row['median_ratio']
                    if not isinstance(ratio,(int,float)) or not math.isfinite(ratio) or ratio<=0:raise ValueError('invalid completed screening ratio')
                    groups[(row['n'],row['batch'])].append((ratio,row['candidate_id']))
                    compact.append({k:row[k] for k in ('record_type','n','batch','candidate_id','status','median_ratio')})
            elif kind in ('candidate_selected','candidate_layout'):
                compact.append(row)
                if kind=='candidate_selected':selected.append((row['n'],row['batch'],row['candidate_id']))
            elif kind=='candidate_start':started.add((row['n'],row['batch'],row['candidate_id']))
            elif kind=='execution' and row.get('phase') in ('candidate_describe','candidate_validation') and row.get('candidate_id')!='original':
                w=row['workload'];key=(w['n'],w['batch'],row['candidate_id'])
                if row['phase']=='candidate_describe':described.add(key)
                elif row.get('status')=='ok':validated[key].add((w['pattern'],w['seed']))
            elif kind=='execution' and row.get('phase')=='identity_control' and row.get('candidate_id')=='expanded_default':
                compact.append({k:row[k] for k in ('record_type','phase','candidate_id','returncode','workload','result')})
            elif kind=='search_complete':terminal.append(row)
    if len(terminal)!=1:raise ValueError('search terminal missing or duplicated')
    if len(selected)!=len(set(selected)):raise ValueError('duplicate selected candidate request')
    old=audit_old_space.analyze(compact,manifests)
    baselines={(r['workload']['n'],r['workload']['batch']):r['result'] for r in compact if r.get('record_type')=='execution'}
    requests={(r['n'],r['batch'],r['candidate_id']):r['request'] for r in compact if r.get('record_type')=='candidate_selected'}
    layouts={(r['n'],r['batch'],r['candidate_id']):r['local_size'] for r in compact if r.get('record_type')=='candidate_layout'}
    selected_set=set(selected)
    if not dispositions<=selected_set or not started<=selected_set:raise ValueError('unselected candidate has start or disposition')
    manifest_selected=sum(x.get('selected_count',0) for x in old['selected_coverage']);manifest_counts=Counter();manifest_cases=[]
    for workload in old['selected_coverage']:
        n,b=workload['n'],workload['batch'];path=Path(manifests)/f'n{n}_b{b}.jsonl';kinds=Counter()
        if not path.is_file():continue
        with path.open(encoding='utf8') as source:
            for line in source:
                item=json.loads(line)
                if item.get('record_type')=='candidate_manifest':
                    kinds['prototype_mathematical_requests']+=1
                    kinds['static_status_'+str(item.get('status'))]+=1
                    kinds['selection_'+str(item.get('selection'))]+=1
        manifest_counts.update(kinds)
        manifest_cases.append({'n':n,'batch':b,'ordered_radix_decompositions':sum(1 for _ in enumerate_candidates.decompositions(n)),**dict(kinds)})
    cases=[]
    for key,values in sorted(groups.items()):
        ratios=sorted(x[0] for x in values);best=min(values)
        cases.append({'n':key[0],'batch':key[1],'completed':len(values),'faster_than_default_count':sum(x<1 for x in ratios),
                      'at_least_2pct_screen_count':sum(x<=.98 for x in ratios),'min_ratio':ratios[0],
                      'median_ratio':statistics.median(ratios),'max_ratio':ratios[-1],'all_ratios_sorted':ratios,
                      'best_candidate_id':best[1], 'fallback_best_screen_ratio':min(1.0,best[0]),
                      'best_old_space_membership':audit_old_space.membership(requests[(*key,best[1])],baselines[key],layouts.get((*key,best[1])))
                          if key in baselines and (*key,best[1]) in requests else {'membership':'not_comparable','reason':'missing evidence'}})
    all_ratios=[ratio for group in groups.values() for ratio,_ in group]
    return {'terminal_status':terminal[0].get('status'),'terminal_workload_statuses':terminal[0].get('workload_statuses'),
            'candidate_dispositions':dict(counts),'manifest_counts':dict(manifest_counts),'manifest_cases':manifest_cases,
            'manifest_selected_requests':manifest_selected,'candidate_selected_events':len(selected),
            'selected_without_event':manifest_selected-len(selected),'describe_execution_count':len(described),
            'selected_without_describe_execution':manifest_selected-len(described),
            'candidates_with_15_successful_validation_rows':sum(len(v)==15 for v in validated.values()),
            'requested_without_disposition':len(selected_set-dispositions),'describe_returned_without_disposition':len(started-dispositions),
            'screened_total':len(all_ratios),
            'screen_faster_count':sum(x<1 for x in all_ratios),'screen_at_least_2pct_count':sum(x<=.98 for x in all_ratios),
            'workloads_with_screen_faster':sum(x['min_ratio']<1 for x in cases),'workloads_with_2pct_screen':sum(x['min_ratio']<=.98 for x in cases),
            'workloads_with_default_fallback':sum(x['min_ratio']>=1 for x in cases),'cases':cases,'old_space':old,
            'interpretation':'screening medians only, selected against the original default; faster points require independent confirmation; fallback is a descriptive per-workload minimum and not a deployable selector',
            'coverage':'stratified selection oversamples stage-count/radix/register extremes; no equal-budget old/new search comparison and no global optimum claim'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',required=True,type=Path);p.add_argument('--manifests',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    result=summarize(args.rows,args.manifests)
    with args.output.open('x',encoding='utf8') as out:json.dump(result,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','old_space','terminal_workload_statuses')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
