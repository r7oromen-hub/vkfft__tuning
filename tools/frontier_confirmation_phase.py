"""Freeze screened winners and explicit negative controls for two guard sessions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
GPU='GPU-3559ec6b-6278-8f93-0706-5f8a85c4911b'


def select_cases(rows, negative_controls):
    """Pick per-workload winners and exactly the negative controls specified before confirmation."""
    identities={(r['n'],r['batch']):r['request'] for r in rows
                if r.get('record_type')=='workload_status' and r.get('status')=='identity_passed'}
    completed={}; best={}
    for row in rows:
        if row.get('record_type')!='candidate_disposition' or row.get('status')!='screen_complete':continue
        workload=(row['n'],row['batch']);key=(*workload,row['candidate_id'])
        if key in completed:raise ValueError('duplicate screened candidate disposition')
        completed[key]=row
        if workload not in identities:raise ValueError('screened workload lacks identity evidence')
        if row['request']==identities[workload]:continue
        if row['median_ratio']<1 and (workload not in best or
                (row['median_ratio'],row['candidate_id'])<(best[workload]['median_ratio'],best[workload]['candidate_id'])):
            best[workload]=row
    winners=sorted(best.values(),key=lambda r:(r['median_ratio'],r['n'],r['batch'],r['candidate_id']))
    selected=[dict({k:row[k] for k in ('n','batch','candidate_id','request')},role='winner') for row in winners]
    seen=set()
    for text in negative_controls:
        pieces=text.split(':')
        if (len(pieces)!=3 or not pieces[0].isdigit() or not pieces[1].isdigit() or
                len(pieces[2])!=64 or any(c not in '0123456789abcdef' for c in pieces[2])):
            raise ValueError('negative control must be N:BATCH:lowercase_candidate_sha256')
        key=(int(pieces[0]),int(pieces[1]),pieces[2])
        if key in seen:raise ValueError('duplicate negative control')
        seen.add(key);row=completed.get(key)
        if (row is None or not row['median_ratio']>1 or
                row['request']==identities.get((key[0],key[1]))):
            raise ValueError('negative control must be a completed non-identity candidate slower than default')
        selected.append(dict({k:row[k] for k in ('n','batch','candidate_id','request')},role='negative_control'))
    if not selected:raise ValueError('no screened winner or selected negative control to confirm')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',required=True)
    p.add_argument('--negative-control',action='append',default=[],metavar='N:BATCH:CANDIDATE_ID',
                   help='explicitly include a screened candidate slower than the default; may be repeated')
    args=p.parse_args()
    if not args.source_run.startswith('frontier-20261031-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.source_run):
        raise ValueError('invalid source run ID')
    output=Path(os.environ['EXPERIMENT_ARTIFACT_DIR'])
    prior=ROOT/'experiments/artifacts/frontier-20261031-admission-001'
    source=ROOT/'experiments/artifacts'/args.source_run
    record=json.loads((ROOT/'experiments/records'/(args.source_run+'.json')).read_text())
    if record['status']!='completed':raise ValueError('source run not terminal and completed')
    if (source/'admission.jsonl').exists():
        paths={'rows':source/'admission.jsonl',**{k:source/(k+'.jsonl') for k in ('guard','incidents','budget')}}
    else:
        paths={k:source/'search'/(k+'.jsonl') for k in ('rows','guard','incidents','budget')}
    rows=[json.loads(x) for x in paths['rows'].read_text().splitlines()]
    if not rows or rows[0]['contract']['gpu']!=GPU:raise ValueError('source GPU differs')
    spec={'gpu':GPU,'cases':select_cases(rows,args.negative_control)}
    specpath=output/'selection.json';specpath.write_text(json.dumps(spec,indent=2)+'\n')
    start=json.loads((prior/'campaign_start.json').read_text())
    first=json.loads((prior/'budget.jsonl').read_text().splitlines()[0])['state']
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=first['boot']:raise ValueError('campaign boot changed')
    if time.monotonic()<start['start_monotonic']:raise ValueError('clock moved backwards')
    deadline=min(time.monotonic()+3600,start['start_monotonic']+start['total_reserved_wall_seconds'])
    if deadline-time.monotonic()<1320:raise ValueError('not enough budget for two guarded sessions')
    folder=ROOT/'test/plan_selection_study/frontier_confirmation'
    original=prior/'build/vkfft_frontier_original';expanded=prior/'build/vkfft_frontier_v2'
    env=dict(os.environ)
    env['PATH']='/usr/local/cuda-13.0/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']='/usr/local/cuda-13.0/lib64:'+env.get('LD_LIBRARY_PATH','')
    common=[]
    for k,path in paths.items():common+=['--screen-'+k,str(path)]
    common+=['--original',str(original),'--expanded',str(expanded)]
    plan=output/'plan.json'
    subprocess.run([sys.executable,str(folder/'make_plan.py'),'--spec',str(specpath),'--output',str(plan),*common],cwd=ROOT,env=env,check=True)
    first_paths={}
    for kind in ('first','second'):
        dest=output/kind;dest.mkdir()
        remaining=deadline-time.monotonic()
        seconds=remaining/2 if kind=='first' else remaining
        if seconds<660:raise ValueError('insufficient confirmation session budget')
        cmd=[sys.executable,str(folder/'runner.py'),'--plan',str(plan),'--session-kind',kind,*common,
             '--gpu',GPU,'--source-file',str(Path(__file__).resolve()),
             '--artifact-file',str(prior/'build/compile_commands.json'),
             '--generated-provenance',str(prior/'generated/expanded_provenance.json'),
             '--output',str(dest/'rows.jsonl'),'--log',str(dest/'guard.jsonl'),
             '--incidents',str(dest/'incidents.jsonl'),'--budget-ledger',str(dest/'budget.jsonl'),
             '--lock','/run/lock/vkfft-study-'+GPU+'.lock','--seconds',str(seconds)]
        if kind=='second':
            for k,path in first_paths.items():cmd+=['--first-'+k,str(path)]
        (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
        if kind=='first':first_paths={k:dest/(k+'.jsonl') for k in ('rows','guard','incidents','budget')}
    cmd=[sys.executable,str(folder/'audit.py'),'--plan',str(plan),*common,'--gpu',GPU,
         '--execution-commit',os.environ['EXPERIMENT_CODE_COMMIT'],'--output',str(output/'two_session_audit.json')]
    for kind in ('first','second'):
        for k in ('rows','guard','incidents','budget'):cmd+=['--'+kind+'-'+k,str(output/kind/(k+'.jsonl'))]
    return subprocess.run(cmd,cwd=ROOT,env=env).returncode


if __name__=='__main__':raise SystemExit(main())
