"""Freeze screened winners and explicit negative controls for two guard sessions."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frontier_campaign_inputs',ROOT/'tools/frontier_campaign_inputs.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


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


def screen_source(root, ctx, source_run):
    """Bind the selected screening run to its committed record and admission."""
    record,source,_=campaign.load_completed_run(root,source_run,ctx['protocol'])
    if (source/'admission.jsonl').exists():
        paths={'rows':source/'admission.jsonl',**{k:source/(k+'.jsonl') for k in ('guard','incidents','budget')}}
    else:
        paths={k:source/'search'/(k+'.jsonl') for k in ('rows','guard','incidents','budget')}
    campaign.verify_artifacts(root,record,list(paths.values()))
    rows=[json.loads(x) for x in paths['rows'].read_text(encoding='utf8').splitlines() if x.strip()]
    if not rows or rows[0]['contract']['gpu']!=ctx['gpu'] or rows[0]['contract']['execution_commit']!=record['execution_commit']:
        raise ValueError('source GPU or execution commit differs from its record')
    source_contract=rows[0]['contract'];admission=ctx['admission_contract']
    if (any(row.get('contract')!=source_contract for row in rows) or
        source_contract['binaries']!=admission['binaries'] or source_contract['vkfft_commit']!=admission['vkfft_commit'] or
        source_contract['generated_provenance']['sha256']!=admission['generated_provenance']['sha256']):
        raise ValueError('source binary, VkFFT, generated provenance or contract differs from admission')
    ledger=[json.loads(line) for line in paths['budget'].read_text(encoding='utf8').splitlines() if line.strip()]
    if not ledger or ledger[0].get('event')!='start' or ledger[-1]['state'].get('inflight') is not None:
        raise ValueError('source budget is incomplete')
    initial=ledger[0]['state'];previous=initial['start']
    deadline=ctx['campaign_start']['start_monotonic']+ctx['config']['total_reserved_wall_seconds']
    if (initial['boot']!=ctx['budget_state']['boot'] or initial['start']<ctx['budget_state']['start'] or
        initial['start']>=deadline):raise ValueError('source belongs to a different campaign or boot')
    for event in ledger:
        state=event['state'];last=state['last']
        if (state.get('contract')!=source_contract or state['boot']!=initial['boot'] or state['start']!=initial['start'] or
            type(last) not in (int,float) or not math.isfinite(last) or last<previous):
            raise ValueError('source budget contract, boot or monotonic clock differs')
        previous=last
    if source_run!=ctx['admission_run']:
        for tracked in (ctx['protocol'],ctx['gate']):
            if source_contract['source_files'].get(str(tracked))!=sha(tracked):
                raise ValueError('source was not bound to the selected protocol and admission gate')
    return paths,rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'experiments/frontier_20261031_protocol.json')
    p.add_argument('--admission-run',default='frontier-20261031-admission-001')
    p.add_argument('--gate',type=Path,default=ROOT/'experiments/evidence/frontier-20261031-admission-gate.json')
    p.add_argument('--negative-control',action='append',default=[],metavar='N:BATCH:CANDIDATE_ID',
                   help='explicitly include a screened candidate slower than the default; may be repeated')
    args=p.parse_args(argv)
    ctx=campaign.load_campaign(ROOT,args.protocol,args.admission_run,args.gate)
    gpu=ctx['gpu'];cuda_root=ctx['cuda_root']
    output=Path(os.environ['EXPERIMENT_ARTIFACT_DIR'])
    prior=ctx['admission_dir']
    paths,rows=screen_source(ROOT,ctx,args.source_run)
    spec={'gpu':gpu,'cases':select_cases(rows,args.negative_control)}
    specpath=output/'selection.json';specpath.write_text(json.dumps(spec,indent=2)+'\n')
    deadline=min(time.monotonic()+3600,campaign.validate_clock(ctx))
    if deadline-time.monotonic()<1320:raise ValueError('not enough budget for two guarded sessions')
    folder=ROOT/'test/plan_selection_study/frontier_confirmation'
    original=prior/'build/vkfft_frontier_original';expanded=prior/'build/vkfft_frontier_v2'
    env=dict(os.environ)
    env['PATH']=str(PurePosixPath(cuda_root)/'bin')+':'+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(PurePosixPath(cuda_root)/'lib64')+':'+env.get('LD_LIBRARY_PATH','')
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
             '--gpu',gpu,'--source-file',str(Path(__file__).resolve()),
             '--source-file',str(ctx['protocol']),'--source-file',str(ctx['gate']),
             '--source-file',str(ctx['record_path']),
             '--artifact-file',str(prior/'build/compile_commands.json'),
             '--generated-provenance',str(prior/'generated/expanded_provenance.json'),
             '--output',str(dest/'rows.jsonl'),'--log',str(dest/'guard.jsonl'),
             '--incidents',str(dest/'incidents.jsonl'),'--budget-ledger',str(dest/'budget.jsonl'),
             '--lock','/run/lock/vkfft-study-'+gpu+'.lock','--seconds',str(seconds)]
        if kind=='second':
            for k,path in first_paths.items():cmd+=['--first-'+k,str(path)]
        (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
        if kind=='first':first_paths={k:dest/(k+'.jsonl') for k in ('rows','guard','incidents','budget')}
    cmd=[sys.executable,str(folder/'audit.py'),'--plan',str(plan),*common,'--gpu',gpu,
         '--execution-commit',os.environ['EXPERIMENT_CODE_COMMIT'],'--output',str(output/'two_session_audit.json')]
    for kind in ('first','second'):
        for k in ('rows','guard','incidents','budget'):cmd+=['--'+kind+'-'+k,str(output/kind/(k+'.jsonl'))]
    return subprocess.run(cmd,cwd=ROOT,env=env).returncode


if __name__=='__main__':raise SystemExit(main())
