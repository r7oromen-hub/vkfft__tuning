"""Execute bounded exhaustive or round-robin frontier phases after audited admission."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT/'test/plan_selection_study/frontier_search'
CONTROL = ROOT/'test/plan_selection_study/frontier_execution/runner.py'
GPU = 'GPU-3559ec6b-6278-8f93-0706-5f8a85c4911b'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def available_seconds(phase_start, phase_budget, deadline, now):
    """Account for manifest construction before admitting another guarded run."""
    return min(phase_budget - (now-phase_start), deadline-now-7200)


def prioritize_anchors(rows):
    selected = [r for r in rows if r['selection']=='selected']
    def priority(row):
        labels=set(row['selection_reasons'])
        return (0 if 'historical' in labels else 1 if 'identity' in labels else
                2 if 'identity_neighbor' in labels else 3, row['selection_index'])
    for index,row in enumerate(sorted(selected,key=priority)):
        row['selection_index']=index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('exhaust128','breadth'),required=True)
    args = p.parse_args()
    output = Path(os.environ['EXPERIMENT_ARTIFACT_DIR'])
    previous = ROOT/'experiments/artifacts/frontier-20261031-admission-001'
    gatepath = ROOT/'experiments/evidence/frontier-20261031-admission-gate.json'
    gate = json.loads(gatepath.read_text())
    if gate.get('status') != 'pass' or gate['gpu'] != GPU or gate['candidates'] != 7:
        raise ValueError('fresh-device admission gate missing')
    expected = {Path(k.replace('\\','/')).name:v for k,v in gate['evidence_sha256'].items()}
    for name,h in expected.items():
        if sha(previous/name) != h: raise ValueError('admission evidence changed: '+name)
    start = json.loads((previous/'campaign_start.json').read_text())
    # Monotonic clock and boot must match the original admission ledger.
    first = json.loads((previous/'budget.jsonl').read_text().splitlines()[0])['state']
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != first['boot']:
        raise ValueError('campaign host rebooted; budget requires review')
    deadline = start['start_monotonic'] + start['total_reserved_wall_seconds']
    if time.monotonic() < start['start_monotonic']: raise ValueError('monotonic clock moved backwards')
    remaining = deadline-time.monotonic()
    # Reserve one hour for confirmation and one for the separate knob prototype.
    seconds = min(7200 if args.phase=='exhaust128' else 10800,remaining-7200)
    if seconds < 660: raise ValueError('campaign budget cannot start this phase')
    env = dict(os.environ)
    env['PATH']='/usr/local/cuda-13.0/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']='/usr/local/cuda-13.0/lib64:'+env.get('LD_LIBRARY_PATH','')
    spec = importlib.util.spec_from_file_location('frontier_manifest',FRONT/'manifest.py')
    mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

    def manifest(n,b,identity,historical,limit,directory):
        directory.mkdir(parents=True,exist_ok=True)
        header,rows=mod.build(n,b,identity,historical,limit)
        prioritize_anchors(rows)
        header['execution_priority']='historical, identity, identity_neighbor, remaining selection order'
        dest=directory/f'n{n}_b{b}.manifest.jsonl'
        with dest.open('x') as f:
            for row in [header,*rows]:f.write(json.dumps(row,sort_keys=True)+'\n')
        return dest,header

    def run(manifests,directory,budget,rr=False):
        directory.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(CONTROL),'--gpu',GPU,
             '--original',str(previous/'build/vkfft_frontier_original'),
             '--expanded',str(previous/'build/vkfft_frontier_v2'),
             '--source-file',str(ROOT/'experiments/frontier_20261031_protocol.json'),
             '--source-file',str(Path(__file__).resolve()),'--source-file',str(gatepath),
             '--artifact-file',str(previous/'build/compile_commands.json'),
             '--generated-provenance',str(previous/'generated/expanded_provenance.json'),
             '--output',str(directory/'rows.jsonl'),'--budget-ledger',str(directory/'budget.jsonl'),
             '--log',str(directory/'guard.jsonl'),'--incidents',str(directory/'incidents.jsonl'),
             '--lock','/run/lock/vkfft-study-'+GPU+'.lock','--seconds',str(budget),'--timeout','300']
        if rr:cmd+=['--round-robin']
        for path in manifests:cmd+=['--manifest',str(path)]
        (directory/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        r=subprocess.run(cmd,cwd=ROOT,env=env)
        if r.returncode: raise RuntimeError('guarded phase failed: '+str(r.returncode))
        return [json.loads(x) for x in (directory/'rows.jsonl').read_text().splitlines()]

    phase_start=time.monotonic()
    summaries=[]
    if args.phase=='exhaust128':
        path,header=manifest(128,1,None,[],1000000,output/'manifests')
        summaries.append(header)
        search_seconds=available_seconds(phase_start,seconds,deadline,time.monotonic())
        if search_seconds<660:raise ValueError('no search budget after manifest generation')
        rows=run([path],output/'search',search_seconds)
    else:
        workloads=[(n,b) for b in (1,16,256,4096) for n in (128,256,512,1024,2048,4096,8192)]
        paths=[manifest(n,b,None,[],0,output/'defaults_manifests')[0] for n,b in workloads]
        defaults_seconds=min(1800,available_seconds(phase_start,seconds,deadline,time.monotonic()))
        if defaults_seconds<660:raise ValueError('no default-admission budget after manifest generation')
        baseline=run(paths,output/'defaults',defaults_seconds)
        identities={(r['n'],r['batch']):r['request'] for r in baseline if r.get('record_type')=='workload_status' and r.get('status')=='identity_passed'}
        if len(identities)!=28:raise ValueError('not all fresh-device defaults admitted')
        historical={
            (128,16):[{'radix':[8,16],'logical_registers':16,'grouped_batch':1}],
            (128,4096):[{'radix':[8,2,8],'logical_registers':16,'grouped_batch':16}],
            (256,4096):[{'radix':[16,16],'logical_registers':16,'grouped_batch':1}],
            (1024,256):[{'radix':[8,4,4,8],'logical_registers':16,'grouped_batch':1}],
        }
        paths=[]
        for n,b in workloads:
            path,header=manifest(n,b,identities[n,b],historical.get((n,b),[]),128,output/'manifests')
            paths.append(path);summaries.append(header)
        remaining_phase=available_seconds(phase_start,seconds,deadline,time.monotonic())
        if remaining_phase<660:raise ValueError('no search budget after defaults')
        rows=run(paths,output/'search',remaining_phase,True)
    dispositions=[r for r in rows if r.get('record_type')=='candidate_disposition']
    complete=sum(r['status']=='screen_complete' for r in dispositions)
    mandatory=[]
    selected_keys=set()
    manifest_paths=[path] if args.phase=='exhaust128' else paths
    for manifest_path in manifest_paths:
        inventory=[json.loads(x) for x in manifest_path.read_text().splitlines()]
        workload=inventory[0]['workload']
        for item in inventory[1:]:
            if item['selection']=='selected':
                selected_keys.add((workload['n'],workload['batch'],item['candidate_id']))
            if item['selection']=='selected' and set(item['selection_reasons']) & {'historical','identity','identity_neighbor'}:
                mandatory.append((workload['n'],workload['batch'],item['candidate_id']))
    disposed={(r['n'],r['batch'],r['candidate_id']) for r in dispositions}
    missing=[list(key) for key in mandatory if key not in disposed]
    status={'phase':args.phase,'gpu':GPU,'execution_commit':os.environ['EXPERIMENT_CODE_COMMIT'],
            'phase_elapsed_seconds':time.monotonic()-phase_start,'budget_seconds':seconds,
            'selected_requests':sum(h['selected_count'] for h in summaries),
            'disposed_requests':len(dispositions),'screen_complete':complete,
            'budget_exhausted':any(r.get('record_type')=='budget_exhausted' for r in rows),
            'all_selected_disposed':disposed==selected_keys and len(dispositions)==len(disposed),
            'mandatory_selected_count':len(mandatory),'mandatory_missing':missing,
            'all_mandatory_disposed':not missing,
            'performance_confirmation':'not performed in this phase','inventories':summaries}
    (output/'phase_summary.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({k:v for k,v in status.items() if k!='inventories'}))


if __name__=='__main__':main()
