#!/usr/bin/env python3
"""Audit a single ten-round confirmation session from raw guarded executions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import audit_guard
import audit_pairs
import audit_search


def read(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def near(a,b):
    return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12)


def audit(rows,guard,incidents,plan,plan_sha):
    errors=[];heads=[x for x in rows if x.get('record_type')=='confirmation_protocol']
    ends=[x for x in rows if x.get('record_type')=='confirmation_complete']
    if len(heads)!=1 or len(ends)!=1:
        return {'status':'fail','errors':['missing or duplicate confirmation protocol/terminal']}
    h=heads[0];session=h.get('session_id');kind=h.get('session_kind');gpu=h.get('gpu')
    if (kind not in ('first','second') or not session or h.get('plan_sha256')!=plan_sha or h.get('sha256')!=plan['sha256'] or
            h.get('rounds')!=10 or ends[0].get('session_id')!=session or ends[0].get('status') not in ('completed','budget_exhausted','reserved_for_second')):
        errors.append('session, plan, binary hash or terminal mismatch')
    ga=audit_guard.audit(rows,guard,incidents,gpu);errors.extend('guard: '+x for x in ga['errors'])
    executions={};pairs=defaultdict(dict);counts=Counter()
    for r in (x for x in rows if x.get('record_type')=='execution'):
        rid=r.get('guard_run_id');binary='original' if r.get('candidate_id')=='original' else 'expanded'
        try:
            if not rid or rid in executions or r['session_id']!=session or r['executable_sha256']!=h['sha256'][binary]:
                raise ValueError('execution ID, session or binary mismatch')
            if [json.loads(x) for x in r['stdout'].splitlines() if x.strip()] != [r['result']]:
                raise ValueError('stdout/result mismatch')
            executions[rid]=r
        except (ValueError,TypeError,KeyError) as exc: errors.append(f'{rid}: {exc}')
    candidates={(x['n'],x['batch'],x['candidate']['candidate_id']):x['candidate'] for x in plan['workloads'] if x.get('candidate')}
    order_entries=plan.get('confirmation_order',[])
    order_keys=[(x['n'],x['batch'],x['candidate_id']) for x in order_entries]
    roles={(x['n'],x['batch'],x['candidate_id']):x.get('role') for x in order_entries}
    if (len(order_keys)!=len(set(order_keys)) or set(order_keys)!=set(candidates) or
            any(role not in ('screen_faster','default_faster_control') for role in roles.values()) or
            sum(role=='default_faster_control' for role in roles.values())>2):
        errors.append('confirmation order/role differs from the registered plan')
    for p in (x for x in rows if x.get('record_type')=='confirmation_pair'):
        key=(p.get('n'),p.get('batch'),p.get('candidate_id'));round_no=p.get('round')
        try:
            if key not in candidates or p['session_id']!=session or p['session_kind']!=kind or round_no not in range(10) or round_no in pairs[key]:
                raise ValueError('unregistered candidate, session or duplicate round')
            orig=executions[p['original_guard_run_id']];cand=executions[p['candidate_guard_run_id']]
            expected=['original','candidate']
            seed=json.dumps([plan['selection_seed'],kind,*key,'confirmation',round_no,42],separators=(',',':')).encode()
            random.Random(int.from_bytes(hashlib.sha256(seed).digest(),'big')).shuffle(expected)
            if (p['order']!=expected or orig['candidate_id']!='original' or cand['candidate_id']!=key[2] or
                    orig['order']!=expected.index('original') or cand['order']!=expected.index('candidate') or
                    any(r['phase']!='confirmation_timing' or r['round']!=round_no or
                        (r['workload']['n'],r['workload']['batch'],r['workload']['seed'],r['workload']['pattern'])!=(key[0],key[1],42,'random') for r in (orig,cand))):
                raise ValueError('pair identity, workload or seeded order mismatch')
            req=candidates[key]['request']
            if not audit_search.original_numerical(orig,gpu) or not audit_search.numerical(cand,req,gpu):
                raise ValueError('numerical or compiled resource evidence failed')
            for r,is_original in ((orig,True),(cand,False)):
                w,result=r['workload'],r['result'];v=result['exec_ms_samples'];raw=result['raw_interval_ms_samples']
                if (w['repeats']!=31 or w['warmup']<5 or w['warmup_ms']<1000 or result['warmup_actual']<5 or result['warmup_wall_ms']<1000 or
                        result['graph_repeats']!=w['graph_repeats'] or result['graph_kernel_count']!=w['graph_repeats'] or result['graph_event_record_count']!=2 or
                        len(v)!=31 or len(raw)!=31 or min(raw)<0.5 or not all(math.isfinite(x) and x>0 for x in v+raw) or
                        not all(math.isclose(x,y/w['graph_repeats'],rel_tol=2e-6,abs_tol=1e-9) for x,y in zip(v,raw)) or
                        audit_search.source_signature(result,is_original)!=candidates[key]['original_signature' if is_original else 'candidate_signature']):
                    raise ValueError('timing sample or source/plan/resource mismatch')
            om=statistics.median(orig['result']['exec_ms_samples']);cm=statistics.median(cand['result']['exec_ms_samples'])
            if not (near(p['original_median_ms'],om) and near(p['candidate_median_ms'],cm) and near(p['candidate_over_original'],cm/om)):
                raise ValueError('pair summary differs from raw measurements')
            pairs[key][round_no]=(cm,om);counts['audited_pairs']+=1
        except (KeyError,TypeError,ValueError,ZeroDivisionError) as exc:errors.append(f'{key}/{round_no}: {exc}')
    encountered=[]
    for pair in (x for x in rows if x.get('record_type')=='confirmation_pair'):
        key=(pair.get('n'),pair.get('batch'),pair.get('candidate_id'))
        if key in encountered and encountered[-1] != key:
            errors.append('confirmation revisits a prior candidate')
        elif key not in encountered:
            encountered.append(key)
    if kind == 'first' and encountered != order_keys[:len(encountered)]:
        errors.append('first confirmation candidate order differs from preregistered plan')
    complete=[];seen=set()
    for d in (x for x in rows if x.get('record_type')=='confirmation_disposition'):
        key=(d.get('n'),d.get('batch'),d.get('candidate_id'))
        try:
            if key in seen or d['session_id']!=session or d['session_kind']!=kind or d['status']!='complete' or set(pairs[key])!=set(range(10)):
                raise ValueError('invalid, repeated or incomplete confirmation disposition')
            seen.add(key);blocks=[pairs[key][i] for i in range(10)];result=audit_pairs.ten_round_session(blocks)
            ratios=[a/b for a,b in blocks]
            if len(d['paired_ratios'])!=10 or any(not near(a,b) for a,b in zip(ratios,d['paired_ratios'])) or not near(statistics.median(ratios),d['median_ratio']):
                raise ValueError('disposition ratio mismatch')
            complete.append({'n':key[0],'batch':key[1],'candidate_id':key[2],'role':roles.get(key),**result,
                             'passes_practical_and_ci':audit_pairs.passes_session_criterion(result)})
        except (KeyError,TypeError,ValueError) as exc:errors.append(f'{key}: {exc}')
    counts['completed_candidates']=len(complete)
    counts['partial_candidate_pairs']=sum(len(v) for k,v in pairs.items() if k not in seen)
    return {'status':'fail' if errors else 'pass','errors':errors,'session_id':session,'session_kind':kind,
            'counts':dict(counts),'complete':complete,
            'candidate_order_observed':[{'n':k[0],'batch':k[1],'candidate_id':k[2]} for k in pairs],
            'scope':'one session only; positive general claim requires independent second session and budget audit; no correction for candidate selection'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rows','guard','incidents','plan'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=audit(read(a.rows),read(a.guard),read(a.incidents),json.loads(a.plan.read_text(encoding='utf8')),sha(a.plan))
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
