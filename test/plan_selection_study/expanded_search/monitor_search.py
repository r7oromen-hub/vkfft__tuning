"""Incremental, non-invasive progress snapshot; never drives the GPU."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',required=True,type=Path)
    p.add_argument('--guard',required=True,type=Path);p.add_argument('--incidents',required=True,type=Path);p.add_argument('--budget',required=True,type=Path)
    p.add_argument('--cursor',required=True,type=Path);a=p.parse_args()
    state=json.loads(a.cursor.read_text(encoding='utf8')) if a.cursor.exists() else {
        'offset':0,'counts':{},'terminal':None,'workloads':{},'selected':{}}
    counts=Counter(state['counts'])
    with a.rows.open('rb') as f:
        if f.seek(0,2)<state['offset']:raise ValueError('search log truncated')
        f.seek(state['offset'])
        while True:
            start=f.tell();line=f.readline()
            if not line or not line.endswith(b'\n'):
                state['offset']=start;break
            r=json.loads(line);kind=r.get('record_type')
            if kind=='workload_status':
                counts['workload_'+str(r.get('status'))]+=1
                if r.get('status')=='identity_passed':
                    key=f"n{r['n']}_b{r['batch']}"
                    state['workloads'][key]={'selected_count':r['manifest_complete']['selected'],'screened':0,'disposed_indices':[]}
            elif kind in ('candidate_selected','candidate_start'):
                counts[kind]+=1
                if kind=='candidate_selected':state['selected'][f"n{r['n']}_b{r['batch']}:{r['candidate_id']}"]=r['selection_index']
            elif kind=='execution' and r.get('returncode')==1:
                result=r.get('result',{});error=result.get('error')
                if error=='initializeVkFFT failed with code 3002':counts['rejection_code_3002']+=1
                elif error=='expanded candidate exceeds compiled function or device resource limit':counts['rejection_compiled_resource']+=1
                elif result.get('status')=='correctness_failed' or result.get('failure_phase')=='single_forward_validation':counts['numerical_failed_execution']+=1
            elif kind=='candidate_disposition':
                counts['candidate_'+str(r.get('status'))]+=1
                key=f"n{r['n']}_b{r['batch']}";index=state['selected'].get(key+':'+r['candidate_id'])
                if key in state['workloads'] and index is not None:
                    state['workloads'][key]['disposed_indices'].append(index)
                    if r.get('status')=='screen_complete':state['workloads'][key]['screened']+=1
                if r.get('status')=='screen_complete' and r.get('median_ratio',1)>=0 and r['median_ratio']<1:
                    counts['screen_faster_unconfirmed']+=1
            elif kind=='search_complete':state['terminal']={'status':r.get('status'),'error':r.get('error')}
    state['counts']=dict(counts)
    a.cursor.parent.mkdir(parents=True,exist_ok=True);tmp=a.cursor.with_suffix('.tmp')
    with tmp.open('w',encoding='utf8') as f:json.dump(state,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,a.cursor)
    b=json.loads(a.budget.read_text(encoding='utf8'))
    incident_count=sum(1 for line in a.incidents.read_text(encoding='utf8').splitlines() if line.strip())
    with a.guard.open('rb') as f:
        f.seek(0,2);end=f.tell();f.seek(max(0,end-512*1024));tail=f.read()
    telemetry=None
    for line in reversed(tail.splitlines()):
        try:r=json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
        if r.get('event')=='sample' and r.get('gpu'):
            telemetry={'utc':r.get('utc'),'gpu_utilization_pct':r['gpu'].get('utilization_gpu_pct'),
                       'memory_used_mib':r['gpu'].get('memory_used_mib'),
                       'active_phase':(r.get('active_run') or {}).get('phase'),'boundary':r.get('boundary')}
            break
    workloads=list(state['workloads'].values());screened=[x['screened'] for x in workloads if x['screened']]
    complete_rounds=0
    if workloads:
        for i in range(64):
            if all(i>=w['selected_count'] or i in w['disposed_indices'] for w in workloads):complete_rounds+=1
            else:break
    print(json.dumps({'counts':state['counts'],'terminal':state['terminal'],'incident_lines':incident_count,
                      'screened_workload_count':len(screened),'screened_per_workload_min':min(screened) if screened else 0,
                      'screened_per_workload_max':max(screened) if screened else 0,'completed_candidate_rounds':complete_rounds,
                      'phase':b['phase'],'telemetry':telemetry,'phase_remaining_seconds':round(b['phase_deadline_monotonic']-time.monotonic()),
                      'total_remaining_seconds':round(b['global_deadline_monotonic']-time.monotonic())},sort_keys=True))


if __name__=='__main__':main()
