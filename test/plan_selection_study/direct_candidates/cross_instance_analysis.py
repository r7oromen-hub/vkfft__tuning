#!/usr/bin/env python3
"""Compare audited screening grids across two environments; never pool samples."""
import argparse, collections, hashlib, json, math, statistics
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
import audit_joint


def get_run(root, stem, exe):
    base=root/stem
    values=[]
    for suffix in ('_rows.jsonl','_guard.jsonl','_incidents.jsonl'):
        rows,errors=audit_joint.load(Path(str(base)+suffix))
        if errors: raise ValueError(errors)
        values.append(rows)
    exe_sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    checked=audit_joint.audit(*values,exe_sha)
    if checked['status']!='complete' or checked['errors']: raise ValueError(checked['errors'])
    scores={json.dumps(x['request'],sort_keys=True):x['screen_ratio'] for x in checked['screened']}
    sources={}
    for row in values[0]:
        if row.get('phase')=='numerical' and row.get('candidate_id') and row.get('status')=='ok':
            sources.setdefault(row['candidate_id'],row['result']['precompile_kernels'][0]['source_sha256'])
    source_by_request={json.dumps(x['request'],sort_keys=True):sources.get(x['candidate_id']) for x in checked['screened']}
    p=checked['protocol']
    return checked,scores,source_by_request,{'gpu':p['gpu'],'n':p['n'],'batch':p['batch'],'exe_sha256':exe_sha}


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-root',required=True,type=Path);p.add_argument('--new-root',required=True,type=Path)
    p.add_argument('--stem',default='joint_n1024_b16');p.add_argument('--old-exe',required=True,type=Path);p.add_argument('--new-exe',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    old,oscores,osources,ometa=get_run(a.old_root,a.stem,a.old_exe)
    new,nscores,nsources,nmeta=get_run(a.new_root,a.stem,a.new_exe)
    assert (ometa['n'],ometa['batch'])==(nmeta['n'],nmeta['batch'])
    common=sorted(set(oscores)&set(nscores))
    if len(common)<3: raise ValueError('fewer than three mutually validated candidates')
    old_v=[oscores[x] for x in common]; new_v=[nscores[x] for x in common]
    def cross(scores):
        d={(tuple(json.loads(k)['radix']),json.loads(k)['grouped_batch']):v for k,v in scores.items()}
        aa=(8,4,4,4,2);bb=(8,8,2,4,2)
        if not all((r,g) in d for r in (aa,bb) for g in (4,8)): return None
        r4=d[aa,4]/d[bb,4];r8=d[aa,8]/d[bb,8]
        return {'a':[8,4,4,4,2],'b':[8,8,2,4,2],'a_over_b_g4':r4,'a_over_b_g8':r8,
                'ratio_of_ratios':r8/r4,'rank_reversal':(r4-1)*(r8-1)<0}
    diffs=[abs(math.log(nscores[k]/oscores[k])) for k in common]
    by_group={}
    for group in sorted({json.loads(k)['grouped_batch'] for k in common}):
        members=[k for k in common if json.loads(k)['grouped_batch']==group]
        if len(members)>=3:
            by_group[str(group)]={'candidates':len(members),
                'spearman':float(spearmanr([oscores[k] for k in members],[nscores[k] for k in members]).statistic),
                'kendall_tau':float(kendalltau([oscores[k] for k in members],[nscores[k] for k in members]).statistic)}
    result={'old':ometa,'new':nmeta,'common_candidates':len(common),'old_only':len(set(oscores)-set(nscores)),
            'new_only':len(set(nscores)-set(oscores)),
            'spearman_screening':float(spearmanr(old_v,new_v).statistic),
            'kendall_tau_screening':float(kendalltau(old_v,new_v).statistic),
            'within_group_rank_correlations':by_group,
            'median_abs_log_relative_ratio_difference':statistics.median(diffs),
            'same_generated_source_hash_count':sum(osources[k] is not None and osources[k]==nsources[k] for k in common),
            'old_interaction_screen':cross(oscores),'new_interaction_screen':cross(nscores),
            'limitations':'Three screening pairs per point, not independent confirmation of every point; new GPU instance and CUDA toolkit both changed; old and new samples are not pooled.'}
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
