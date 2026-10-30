#!/usr/bin/env python3
"""Descriptive analysis of a local, immutable search/guard JSONL snapshot."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    with path.open(encoding='utf8') as source:
        return [json.loads(line) for line in source if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()


def median(values):return statistics.median(values) if values else None


def analyze(rows, guard):
    executions=[r for r in rows if r.get('record_type')=='execution']
    groups=defaultdict(list)
    for x in executions:
        name=x.get('candidate_id'); role=name if name in ('original','expanded_default','identity') else 'candidate'
        groups[(x.get('phase'),role)].append(x)
    summary=[]
    fields=('context_bootstrap_or_recheck_ms','reference_and_host_input_ms','candidate_external_setup_ms',
            'initialize_ms','graph_build_ms','warmup_wall_ms','candidate_wall_ms')
    for (phase,role),all_rows in sorted(groups.items()):
        valid=[x for x in all_rows if x.get('status')=='ok' and x.get('returncode')==0]
        metrics={}
        for field in fields:
            values=[x['result'][field] for x in valid if isinstance(x.get('result',{}).get(field),(int,float))]
            metrics[field]={'n':len(values),'median':median(values),'sum':sum(values)}
        for stage in ('nvrtc_compile','shader_codegen'):
            values=[x['result']['init_trace']['stage_totals'][stage]['inclusive_ms'] for x in valid
                    if stage in x.get('result',{}).get('init_trace',{}).get('stage_totals',{})]
            metrics[stage+'_trace_ms']={'n':len(values),'median':median(values),'sum':sum(values)}
        walls=[1000*x['wall_seconds'] for x in valid if isinstance(x.get('wall_seconds'),(int,float))]
        summary.append({'phase':phase,'role':role,'rows':len(all_rows),'successful_rows':len(valid),
                        'process_wall_ms':{'n':len(walls),'median':median(walls),'sum':sum(walls)},'metrics':metrics})
    times=defaultdict(dict)
    for event in guard:
        rid=event.get('guard_run_id') or (event.get('active_run') or {}).get('guard_run_id')
        if rid and event.get('event') in ('run_one_start','run_one_end'):
            times[rid][event['event']]=event['utc']
    overhead=defaultdict(list)
    for x in executions:
        t=times.get(x.get('guard_run_id'),{})
        if set(t)=={'run_one_start','run_one_end'} and isinstance(x.get('wall_seconds'),(int,float)):
            start=datetime.fromisoformat(t['run_one_start']);end=datetime.fromisoformat(t['run_one_end'])
            overhead[x['phase']].append(1000*((end-start).total_seconds()-x['wall_seconds']))
    guard_overhead={k:{'n':len(v),'median_ms':median(v),'sum_ms':sum(v)} for k,v in sorted(overhead.items())}
    during=defaultdict(list)
    for x in guard:
        if x.get('event')=='sample' and x.get('boundary')=='during' and x.get('ok') is True:
            phase=(x.get('active_run') or {}).get('phase','no_active')
            u=x.get('gpu',{}).get('utilization_gpu_pct')
            if isinstance(u,(int,float)):during[phase].append(u)
    utilization={k:{'samples':len(v),'zero_samples':sum(x==0 for x in v),'median_percent':median(v),'max_percent':max(v)} for k,v in sorted(during.items())}
    return {'executions':len(executions),'phase_counts':dict(Counter(x.get('phase') for x in executions)),
            'phase_and_role':summary,'guard_envelope_minus_process':guard_overhead,'during_gpu_utilization_samples':utilization,
            'caveats':['snapshot can end mid-run and overweights startup; not a full-run profile',
                       'context bootstrap is cuInit/cudaSetDevice/cudaFree-null wall time; not isolated context creation',
                       'NVRTC trace exists only for instrumented runs; original controls have no compile trace',
                       'guard envelope minus subprocess wall includes boundary sampling, Python/IPC and serialization, not only guard overhead',
                       'GPU utilization is nvidia-smi sample-window busy time, not occupancy, and long runs are oversampled',
                       'components and medians cannot be summed as an exact wall-clock decomposition']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',type=Path,required=True);p.add_argument('--guard',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=analyze(read(a.rows),read(a.guard));report['inputs_sha256']={'rows':sha(a.rows),'guard':sha(a.guard)}
    with a.output.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':main()
