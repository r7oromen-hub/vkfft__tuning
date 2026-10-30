#!/usr/bin/env python3
"""Descriptive analysis of an audited, budget-limited frontier screen."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    """Retain only screen summaries and timing samples after the audit has bound the file."""
    retained=[]
    with Path(path).open(encoding='utf8') as source:
        for line in source:
            if not line.strip():continue
            row=json.loads(line);kind=row.get('record_type')
            if kind=='execution':
                if row.get('phase') not in ('identity','calibration','timing'):continue
                row.pop('stdout',None);row.pop('stderr',None)
            elif kind not in ('session_start','workload_status','candidate_start','screen_pair',
                              'candidate_disposition','budget_exhausted'):
                continue
            retained.append(row)
    return retained


def need(condition, message):
    if not condition: raise ValueError(message)


def key(row): return (row['n'],row['batch'],row['candidate_id'])


def ratio_stats(items):
    n=len(items);fast=sum(c['ratio']<1 for c in items);two=sum(c['ratio']<=.98 for c in items)
    return {'screen_complete_count':n,'faster_count':fast,'faster_fraction':fast/n if n else None,
            'at_least_2pct_count':two,'at_least_2pct_fraction':two/n if n else None}


def prototype_static_supported(n, batch, registers):
    """Count this prototype's static requests, not device-valid FFT plans."""
    need(type(n) is int and n>=2 and n<=8192 and not n & (n-1) and
         type(batch) is int and 1<=batch<=65536, 'invalid prototype workload')
    exponent=n.bit_length()-1;groups=0;g=1
    while g<=batch:
        if batch%g==0:groups+=1
        g*=2
    total=0
    for r in registers:
        if r>n:continue
        max_radix_exp=min(5,r.bit_length()-1)
        counts=[1]+[0]*exponent
        for power in range(1,exponent+1):
            counts[power]=sum(counts[power-k] for k in range(1,min(power,max_radix_exp)+1))
        total+=counts[exponent]*groups
    return total


def select_best(items):
    return min(items,key=lambda item:(item['ratio'],item['candidate_id'])) if items else None


def point(item):
    if item is None:return None
    return {k:item[k] for k in ('n','batch','candidate_id','request','ratio','observed_plan')}


def observed_plan(result):
    """Only copy directly observed fields; a request is not an observed layout."""
    kernels=(result.get('structural_features') or {}).get('kernels') or []
    diagnostics=result.get('compiled_diagnostics') or []
    pre=result.get('precompile_kernels') or []
    kernel=kernels[0] if len(kernels)==1 else {}
    compiled=diagnostics[0] if len(diagnostics)==1 else {}
    snapshot=pre[0] if len(pre)==1 else {}
    raw=snapshot.get('raw_descriptor') or {}
    return {'stage_radix':kernel.get('stageRadix'),'local_size':kernel.get('localSize'),
            'grouped_batch':kernel.get('groupedBatch'),'logical_registers_per_thread':raw.get('logical_registers_per_thread'),
            'register_boost':raw.get('registerBoost'),'dynamic_shared_bytes':raw.get('dynamic_shared_bytes'),
            'cuda_registers_per_thread':compiled.get('cuda_num_regs'),'cuda_local_bytes':compiled.get('cuda_local_bytes'),
            'cuda_static_shared_bytes':compiled.get('cuda_static_shared_bytes'),'source_sha256':snapshot.get('source_sha256')}


def parse_complete(case, segment):
    n,b,cid=key(case);dispositions=[r for r in segment if r.get('record_type')=='candidate_disposition']
    need(len(dispositions)==1 and dispositions[0]['status']=='screen_complete' and dispositions[0]['request']==case['request'],
         'completed candidate disposition differs from coverage')
    timings=[r for r in segment if r.get('record_type')=='execution' and r.get('phase')=='timing']
    records=[r for r in segment if r.get('record_type')=='screen_pair']
    need(len(timings)==6 and len(records)==3 and {r['round'] for r in records}=={0,1,2}, 'screen pairs incomplete')
    ratios=[];plan=None
    for round_no in range(3):
        pair=[r for r in timings if r['round']==round_no]
        need(len(pair)==2 and {r['candidate_id'] for r in pair}=={'original',cid} and
             {r['order'] for r in pair}=={0,1}, 'invalid paired execution')
        values={}
        for row in pair:
            work=row['workload'];need((work['n'],work['batch'])==(n,b), 'paired workload mismatch')
            samples=row['result']['exec_ms_samples']
            need(isinstance(samples,list) and samples and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in samples),
                 'invalid paired samples')
            values['original' if row['candidate_id']=='original' else 'candidate']=statistics.median(samples)
            if row['candidate_id']==cid and plan is None:plan=observed_plan(row['result'])
        ratio=values['candidate']/values['original']; ratios.append(ratio)
        pair_record=next(r for r in records if r['round']==round_no)
        need(key(pair_record)==(n,b,cid) and math.isclose(pair_record['ratio'],ratio,rel_tol=1e-12), 'recorded ratio differs')
    score=statistics.median(ratios);d=dispositions[0]
    need(d['pair_ratios']==ratios and math.isclose(d['median_ratio'],score,rel_tol=1e-12) and
         math.isclose(case['median_ratio'],score,rel_tol=1e-12), 'coverage or screen median differs')
    return dict(n=n,batch=b,candidate_id=cid,request=case['request'],ratio=score,
                pair_ratios=ratios,observed_plan=plan)


def group_matches(items, selector, varying):
    grouped=defaultdict(list)
    for item in items:grouped[selector(item)].append(item)
    groups=[]
    for fixed,members in grouped.items():
        if len({varying(item) for item in members})<2:continue
        ordered=sorted(members,key=lambda x:(x['ratio'],x['candidate_id']))
        best,worst=ordered[0],ordered[-1]
        groups.append({'n':best['n'],'batch':best['batch'],'fixed':list(fixed[2:]),
                       'member_count':len(members),'pair_count':len(members)*(len(members)-1)//2,
                       'slowest_over_fastest':worst['ratio']/best['ratio'],
                       'members':[point(item) for item in ordered]})
    groups.sort(key=lambda group:(group['n'],group['batch'],json.dumps(group['fixed'],sort_keys=True)))
    by_work=defaultdict(lambda:Counter())
    for group in groups:
        count=by_work[(group['n'],group['batch'])]
        count['group_count']+=1;count['pair_count']+=group['pair_count'];count['candidate_occurrences']+=group['member_count']
    return {'group_count':len(groups),'pair_count':sum(g['pair_count'] for g in groups),
            'candidate_occurrences':sum(g['member_count'] for g in groups),
            'by_workload':[dict(n=n,batch=b,**dict(count)) for (n,b),count in sorted(by_work.items())],
            'groups':groups}


def replay(items, seed):
    by_work=defaultdict(list)
    for item in items:by_work[(item['n'],item['batch'])].append(item)
    curves=[]
    for (n,b),members in sorted(by_work.items()):
        def rank(item):
            payload=json.dumps([seed,n,b,item['candidate_id']],separators=(',',':')).encode()
            return hashlib.sha256(payload).hexdigest(),item['candidate_id']
        best=1.0;points=[{'step':0,'best_ratio':best,'candidate_id':None}]
        for step,item in enumerate(sorted(members,key=rank),1):
            best=min(best,item['ratio'])
            points.append({'step':step,'candidate_id':item['candidate_id'],'candidate_ratio':item['ratio'],'best_ratio':best})
        curves.append({'n':n,'batch':b,'points':points})
    return {'seed':seed,'workloads':curves,
            'scope':'offline fixed-seed reordering of already completed screen points, with default ratio 1 at step zero; excludes rejected, partial and unmeasured requests and does not simulate cost or online search'}


def analyze(coverage, rows, rows_sha256, seed):
    need(coverage.get('schema')=='frontier-screen-coverage/v1' and coverage.get('status')=='pass', 'coverage audit did not pass')
    need((coverage.get('guard_audit') or {}).get('status')=='pass', 'coverage guard audit did not pass')
    hashes=coverage.get('evidence_sha256') or {}
    matches=[name for name,value in hashes.items() if value==rows_sha256]
    need(len(matches)==1, 'raw rows are not uniquely bound to the coverage report')
    need(type(seed) is int and seed>=0, 'seed must be a nonnegative integer')
    need(rows and rows[0].get('record_type')=='session_start', 'missing screen session header')
    contract=rows[0]['contract']
    need(contract.get('gpu')==coverage.get('gpu') and contract.get('execution_commit')==coverage.get('execution_commit') and
         all(row.get('contract')==contract for row in rows), 'screen identity or contract mismatch')
    cases=coverage['cases']; selected={};allowed=('screen_complete','scheduler_rejected','resource_rejected','partial','selected_not_started')
    for case in cases:
        k=key(case)
        need(k not in selected and case['status'] in allowed, 'duplicate or unknown coverage candidate')
        need(type(case['request'].get('logical_registers')) is int and
             case['request']['logical_registers'] in (2,4,8,16,32,64,128,256), 'unsupported register subdomain')
        encoded=json.dumps(case['request'],sort_keys=True,separators=(',',':')).encode()
        need(hashlib.sha256(encoded).hexdigest()==case['candidate_id'], 'coverage candidate request hash mismatch')
        selected[k]=case
    status_counts=Counter(case['status'] for case in cases);counts=coverage['counts']
    need(len(selected)==counts['selected'] and all(status_counts[status]==counts[status] for status in allowed),
         'coverage selected partition mismatch')
    segments={};active=None
    for row in rows:
        if row.get('record_type')=='candidate_start':
            k=key(row);need(k in selected and k not in segments and row['request']==selected[k]['request'], 'unknown, duplicate or mismatched candidate start')
            segments[k]=[row];active=k
        elif row.get('record_type')=='execution' and row.get('phase') in ('identity','calibration'):
            active=None
        elif active is not None:segments[active].append(row)
        elif row.get('record_type') in ('screen_pair','candidate_disposition'):
            raise ValueError('orphan screen disposition or pair')
    completed=[]
    for k,case in selected.items():
        segment=segments.get(k)
        if case['status']=='selected_not_started':
            need(segment is None, 'unstarted candidate has execution evidence');continue
        need(segment is not None, 'started candidate missing from rows')
        dispositions=[r for r in segment if r.get('record_type')=='candidate_disposition']
        if case['status']=='partial':
            need(not dispositions, 'partial candidate has a disposition');continue
        need(len(dispositions)==1 and key(dispositions[0])==k and dispositions[0]['status']==case['status'] and
             dispositions[0]['request']==case['request'], 'disposition does not match coverage')
        if case['status']=='screen_complete':completed.append(parse_complete(case,segment))
    need(len(completed)==counts['screen_complete'], 'completed count mismatch')
    workload_keys=sorted({(case['n'],case['batch']) for case in cases} |
                         {(r['n'],r['batch']) for r in rows if r.get('record_type')=='workload_status'})
    workload_reports=[]
    for n,b in workload_keys:
        scoped=[c for c in completed if (c['n'],c['batch'])==(n,b)];best=select_best(scoped)
        statuses=[r for r in rows if r.get('record_type')=='workload_status' and (r['n'],r['batch'])==(n,b)]
        need(len(statuses)<=1 and (not statuses or statuses[0]['status'] in ('identity_passed','unsupported_identity')),
             'invalid workload default status')
        default_status=statuses[0]['status'] if statuses else 'not_observed'
        need(default_status!='not_observed' or not scoped, 'completed candidate lacks default evidence')
        with_default=({'kind':'candidate',**point(best)} if best and best['ratio']<1 else
                      {'kind':'default','ratio':1.0} if statuses else None)
        workload_reports.append({'n':n,'batch':b,'selected_count':sum((c['n'],c['batch'])==(n,b) for c in cases),
                                 'default_status':default_status,'outcomes':ratio_stats(scoped),
                                 'best_screened':point(best),'best_with_default':with_default})
    spaces={}
    for name,include in (('r_le_64',lambda r:r<=64),('r_128_256',lambda r:r in (128,256))):
        chosen=[c for c in cases if include(c['request']['logical_registers'])]
        scores=[c for c in completed if include(c['request']['logical_registers'])]
        c=Counter(x['status'] for x in chosen);total=len(chosen)
        regs=(2,4,8,16,32,64) if name=='r_le_64' else (128,256)
        denominator=sum(prototype_static_supported(n,b,regs) for n,b in workload_keys)
        spaces[name]={'selected_count':total,**{status:c[status] for status in allowed},
                      'screen_complete_count':len(scores),'completion_fraction':len(scores)/total if total else None,
                      'prototype_static_supported_count':denominator,
                      'completed_over_prototype_static_supported':len(scores)/denominator if denominator else None,
                      'best_screened':point(select_best(scores)),
                      'by_workload':[{'n':n,'batch':b,'selected_count':sum((x['n'],x['batch'])==(n,b) for x in chosen),
                                      'screen_complete_count':len(s:=[x for x in scores if (x['n'],x['batch'])==(n,b)]),
                                      'prototype_static_supported_count':prototype_static_supported(n,b,regs),
                                      'completed_over_prototype_static_supported':len(s)/d if
                                          (d:=prototype_static_supported(n,b,regs)) else None,
                                      'best_screened':point(select_best(s))}
                                     for n,b in workload_keys]}
    def wr(item):return (item['n'],item['batch'])
    def radix(item):return tuple(item['request']['radix'])
    def reg(item):return item['request']['logical_registers']
    def group(item):return item['request']['grouped_batch']
    matched={'fixed_r_group_radix':group_matches(completed,lambda i:(*wr(i),reg(i),group(i)),radix),
             'fixed_radix_layout':group_matches(completed,lambda i:(*wr(i),radix(i)),lambda i:(reg(i),group(i))),
             'fixed_radix_r_group':group_matches(completed,lambda i:(*wr(i),radix(i),reg(i)),group),
             'fixed_radix_group_r':group_matches(completed,lambda i:(*wr(i),radix(i),group(i)),reg)}
    return {'schema':'frontier-screen-analysis/v1','gpu':coverage['gpu'],'execution_commit':coverage['execution_commit'],
            'source_rows_sha256':rows_sha256,'coverage_counts':counts,'overall':ratio_stats(completed),
            'workloads':workload_reports,'register_subspaces':spaces,'matched':matched,'replay':replay(completed,seed),
            'scope':{'denominator':'completed selected candidates only, including any measured identity; default reference at 1 is not an extra candidate',
                     'register_subspaces':'R<=64 is a subdomain of this prototype, not the historical fixed-endpoint search; static-supported denominators are independent counts within the represented workloads, not device-valid counts or the VkFFT-wide space',
                     'measurement':'3-round screening median; no confirmation, training, full-space optimum or online search claim',
                     'comparisons':'matched requests describe whole-plan differences; source, compiled resources and layout may change together, so no parameter causality is inferred'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage',type=Path,required=True);parser.add_argument('--rows',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=20261031)
    args=parser.parse_args();coverage_bytes=args.coverage.read_bytes()
    coverage=json.loads(coverage_bytes.decode('utf8'))
    rows_hash=sha(args.rows);rows=read_rows(args.rows)
    need(sha(args.rows)==rows_hash, 'raw rows changed while being read')
    result=analyze(coverage,rows,rows_hash,args.seed)
    result['input_sha256']={'coverage':hashlib.sha256(coverage_bytes).hexdigest(),'rows':rows_hash}
    with args.output.open('x',encoding='utf8') as out:
        json.dump(result,out,sort_keys=True,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({'status':'complete','screen_complete':result['overall']['screen_complete_count'],
                      'output':str(args.output)},sort_keys=True))


if __name__=='__main__':main()
