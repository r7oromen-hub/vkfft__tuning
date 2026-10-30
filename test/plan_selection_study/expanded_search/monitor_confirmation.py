"""Read only new confirmation events and the latest existing guard sample."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rows','guard','incidents','budget','cursor'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();state=json.loads(a.cursor.read_text(encoding='utf8')) if a.cursor.exists() else {'offset':0,'counts':{},'terminal':None}
    counts=Counter(state['counts'])
    with a.rows.open('rb') as f:
        if f.seek(0,2)<state['offset']:raise ValueError('confirmation log truncated')
        f.seek(state['offset'])
        while True:
            start=f.tell();line=f.readline()
            if not line or not line.endswith(b'\n'):state['offset']=start;break
            r=json.loads(line);kind=r.get('record_type')
            if kind=='confirmation_protocol':state['session_kind']=r['session_kind']
            elif kind=='confirmation_pair':counts['paired_rounds']+=1
            elif kind=='confirmation_disposition':
                counts['completed_candidates']+=1
                if r.get('median_ratio',1)<1:counts['median_faster_unconfirmed']+=1
            elif kind=='confirmation_complete':state['terminal']={'status':r.get('status'),'error':r.get('error')}
    state['counts']=dict(counts);tmp=a.cursor.with_suffix('.tmp')
    with tmp.open('w',encoding='utf8') as f:json.dump(state,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(tmp,a.cursor)
    with a.guard.open('rb') as f:f.seek(0,2);end=f.tell();f.seek(max(0,end-512*1024));tail=f.read()
    sample=None
    for line in reversed(tail.splitlines()):
        try:r=json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
        if r.get('event')=='sample' and r.get('gpu'):
            sample={'utc':r['utc'],'gpu_utilization_pct':r['gpu'].get('utilization_gpu_pct'),
                    'memory_used_mib':r['gpu'].get('memory_used_mib'),'active_phase':(r.get('active_run') or {}).get('phase')};break
    b=json.loads(a.budget.read_text(encoding='utf8'))
    print(json.dumps({'session_kind':state.get('session_kind'),'counts':state['counts'],'terminal':state['terminal'],
                      'incident_lines':sum(1 for x in a.incidents.read_text(encoding='utf8').splitlines() if x.strip()),
                      'telemetry':sample,'phase':b['phase'],'phase_remaining_seconds':round(b['phase_deadline_monotonic']-time.monotonic()),
                      'total_remaining_seconds':round(b['global_deadline_monotonic']-time.monotonic())},sort_keys=True))


if __name__=='__main__':main()
