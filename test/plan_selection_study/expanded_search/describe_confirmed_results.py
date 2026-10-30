#!/usr/bin/env python3
"""Describe audited plans and both session intervals without attributing causality."""
import argparse
import json
from pathlib import Path


def structure(signature):
    k=signature['structural_features']['kernels'][0];d=signature['canonical_descriptor'];c=signature['compiled_diagnostics'][0]
    return {'radix':k['stageRadix'],'logical_registers':d['logical_registers_per_thread'],
            'minimum_logical_registers':d['min_logical_registers_per_thread'],'grouped_batch':k['groupedBatch'],
            'local_size':k['localSize'],'register_boost':k['registerBoost'],
            'dynamic_shared_bytes':k['dynamic_shared_bytes'],'cuda_static_shared_bytes':c['cuda_static_shared_bytes'],
            'cuda_registers_per_thread':c['cuda_num_regs'],'cuda_local_bytes_per_thread':c['cuda_local_bytes'],
            'cuda_kernel_thread_limit':c['cuda_max_threads_per_block'],'source_sha256':signature['source_sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','combined-audit','screen-summary','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();plan=json.loads(a.plan.read_text(encoding='utf8'));report=json.loads(a.combined_audit.read_text(encoding='utf8'))
    summary=json.loads(a.screen_summary.read_text(encoding='utf8'));combined=report.get('combined',report)
    if report.get('status')!='pass' or combined.get('status')!='pass':raise ValueError('combined audit must pass')
    plans={(w['n'],w['batch'],w['candidate']['candidate_id']):w['candidate'] for w in plan['workloads'] if w.get('candidate')}
    cases={(x['n'],x['batch']):x for x in summary['cases']};results=[]
    for item in combined['candidates']:
        key=(item['n'],item['batch'],item['candidate_id']);candidate=plans[key];s=cases[key[:2]]
        if s['best_candidate_id']!=key[2]:raise ValueError('confirmation candidate differs from frozen screened best')
        result={'n':key[0],'batch':key[1],'candidate_id':key[2],'status':item['status'],
                'screen_ratio':candidate['median_ratio'],'old_space_membership':s['best_old_space_membership'],
                'original':structure(candidate['original_signature']),'candidate':structure(candidate['candidate_signature']),
                'first':item['first'],'second':item['second']}
        for side in ('first','second'):
            if result[side] is not None:
                result[side]=dict(result[side],speedup_factor=1/result[side]['geometric_mean_ratio'])
        results.append(result)
    output={'results':results,'scope':'same-GPU plan-level comparisons; changed radix/group/register/layout may be coupled; no causal component attribution'}
    with a.output.open('x',encoding='utf8') as out:json.dump(output,out,ensure_ascii=False,indent=2);out.write('\n')


if __name__=='__main__':main()
