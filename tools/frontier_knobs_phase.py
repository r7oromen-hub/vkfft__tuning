"""Build and screen joint-knob configurations after an audited v2 search."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
import frontier_campaign_inputs as campaign_inputs

DEFAULT_SOURCE_RUN='frontier-20261031-exhaust128-001'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf8') as source:
        return [json.loads(line) for line in source if line.strip()]


def verify_source(root, source_run, campaign):
    """Bind a completed, guarded screen to the selected admission campaign."""
    record,artifacts,record_path=campaign_inputs.load_completed_run(root,source_run,campaign['protocol'])
    paths={key:artifacts/'search'/(key+'.jsonl') for key in ('rows','guard','incidents','budget')}
    campaign_inputs.verify_artifacts(root,record,list(paths.values()))
    rows,guard,incidents,budget=(read_jsonl(paths[key]) for key in ('rows','guard','incidents','budget'))
    if not rows or not guard or not budget:
        raise ValueError('source search evidence missing')
    contract=rows[0].get('contract') or {}
    if (contract.get('protocol')!='frontier-execution/v1' or contract.get('gpu')!=campaign['gpu'] or
        contract.get('execution_commit')!=record.get('execution_commit') or
        any(row.get('contract')!=contract for row in rows)):
        raise ValueError('source search GPU, execution commit, or contract differs')
    sources=contract.get('source_files') or {}
    for source in (campaign['protocol'],campaign['gate']):
        if sources.get(str(Path(source).resolve()))!=sha(source):
            raise ValueError('source search is not bound to the selected protocol and admission gate')
    expected_binaries={name:sha(campaign['admission_dir']/'build'/filename) for name,filename in
                       (('original','vkfft_frontier_original'),('expanded','vkfft_frontier_v2'))}
    if contract.get('binaries')!=expected_binaries:
        raise ValueError('source search did not use the selected admission binaries')
    admission_contract=campaign['admission_contract']
    if contract.get('vkfft_commit') != admission_contract.get('vkfft_commit'):
        raise ValueError('source search uses a different VkFFT commit than the selected admission')
    if contract.get('generated_provenance',{}).get('sha256') != admission_contract.get('generated_provenance',{}).get('sha256'):
        raise ValueError('source search did not use the selected admission generated provenance')
    if (budget[0].get('event')!='start' or budget[-1]['state'].get('inflight') is not None or
        any(item['state'].get('contract')!=contract or item['state'].get('boot')!=campaign['budget_state']['boot']
            for item in budget)):
        raise ValueError('source budget is incomplete or from a different campaign boot')
    initial=budget[0]['state']
    if initial['start']<campaign['budget_state']['start']:
        raise ValueError('source search predates the selected admission campaign')
    previous=initial['start']
    for item in budget:
        state=item['state']
        if state.get('start')!=initial['start'] or state.get('last',-1)<previous:
            raise ValueError('source budget clock is inconsistent')
        previous=state['last']
    if time.monotonic()<previous:
        raise ValueError('source budget clock is ahead of the current campaign clock')
    executions=[row for row in rows if row.get('record_type')=='execution']
    starts=[item for item in budget if item.get('event')=='work_start']
    ends=[item for item in budget if item.get('event')=='work_finish']
    expected_events=['start']+['work_start','work_finish']*len(executions)
    if budget[-1].get('event')=='denied':expected_events.append('denied')
    if len(starts)!=len(ends) or len(ends)!=len(executions) or [item.get('event') for item in budget]!=expected_events:
        raise ValueError('source budget work sequence does not match executions')
    for start,end,row in zip(starts,ends,executions):
        work=end['identity'];w=row['workload']
        if (start['state']['inflight']!=work or end['status']!=row['status'] or
            (work['n'],work['batch'],work['candidate_id'],work['phase'],work['round'])!=
            (w['n'],w['batch'],row['candidate_id'],row['phase'],row['round'])):
            raise ValueError('source budget work identity differs from execution')
    spec=importlib.util.spec_from_file_location('frontier_knobs_source_guard_audit',
        Path(root)/'test/plan_selection_study/expanded_search/audit_guard.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    result=audit.audit(rows,guard,incidents,campaign['gpu'])
    if result.get('status')!='pass' or result.get('counts',{}).get('executions',0)<1:
        raise ValueError('source search guard audit failed: '+str(result.get('errors')))
    return {'run':source_run,'record':str(record_path),'record_sha256':sha(record_path),
            'execution_commit':record['execution_commit'],'artifact_sha256':{name:sha(path) for name,path in paths.items()},
            'guard_audit':result}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default=campaign_inputs.DEFAULT_PROTOCOL)
    parser.add_argument('--admission-run',default=campaign_inputs.DEFAULT_ADMISSION_RUN)
    parser.add_argument('--gate',default=campaign_inputs.DEFAULT_GATE)
    parser.add_argument('--source-run',default=DEFAULT_SOURCE_RUN)
    args=parser.parse_args(argv)
    output=Path(os.environ['EXPERIMENT_ARTIFACT_DIR'])
    campaign=campaign_inputs.load_campaign(ROOT,args.protocol,args.admission_run,args.gate)
    source=verify_source(ROOT,args.source_run,campaign)
    gpu=campaign['gpu'];cuda_root=campaign['cuda_root']
    campaign_deadline=campaign_inputs.validate_clock(campaign)
    now=time.monotonic()
    # At most one hour for this extension, reserving one hour for confirmation.
    deadline=min(now+3600,campaign_deadline-3600)
    if deadline-now<1200:raise ValueError('insufficient global extension budget')
    metadata={'gpu':gpu,'cuda_root':cuda_root,'protocol':str(campaign['protocol']),
              'protocol_sha256':sha(campaign['protocol']),'admission_run':campaign['admission_run'],
              'admission_execution_commit':campaign['execution_commit'],'gate':str(campaign['gate']),
              'gate_sha256':sha(campaign['gate']),'source':source,'campaign_deadline_monotonic':campaign_deadline}
    with (output/'campaign_inputs.json').open('x',encoding='utf8') as dest:
        json.dump(metadata,dest,indent=2,sort_keys=True);dest.write('\n')
    env=dict(os.environ)
    env['PATH']=str(PurePosixPath(cuda_root)/'bin')+':'+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(PurePosixPath(cuda_root)/'lib64')+':'+env.get('LD_LIBRARY_PATH','')
    knobs=ROOT/'test/plan_selection_study/frontier_knobs'
    generated=output/'generated';build=output/'build'
    lockpath=Path('/run/lock')/('vkfft-study-'+gpu+'.lock')
    sys.path.insert(0,str(ROOT/'test/plan_selection_study/expanded_search'))
    import device_lock
    commands=[
        [sys.executable,str(knobs/'prepare_headers.py'),'--source',str(ROOT/'VkFFT/vkFFT'),'--output',str(generated/'include')],
        ['cmake','-S',str(ROOT/'test/plan_selection_study/frontier_search'),'-B',str(build),
         '-DCMAKE_BUILD_TYPE=Release','-DFRONTIER_GENERATED_ROOT='+str(generated),
         '-DCUDAToolkit_ROOT='+str(cuda_root),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
        ['cmake','--build',str(build),'--parallel','2'],
        ['ctest','--test-dir',str(build),'--output-on-failure'],
        [sys.executable,str(knobs/'make_manifest.py'),'--spec',str(ROOT/'experiments/frontier_knobs_n128_b16.json'),
         '--output',str(output/'manifest.json')],
    ]
    with device_lock.DeviceLock(lockpath):
        check=subprocess.run(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        if check.stdout.strip():raise ValueError('another GPU process is active')
        for i,cmd in enumerate(commands):
            timeout=min(600,deadline-time.monotonic()-660)
            if timeout<=0:raise ValueError('extension deadline cannot admit another build operation')
            r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=timeout)
            (output/f'build_step_{i}.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,
                stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
            if r.returncode:return r.returncode
    parent=subprocess.check_output(['git','rev-parse',os.environ['EXPERIMENT_CODE_COMMIT']+'^'],cwd=ROOT,text=True).strip()
    upstream=subprocess.check_output(['git','-C',str(ROOT/'VkFFT'),'rev-parse','HEAD'],text=True).strip()
    provenance=generated/'expanded_provenance.json'
    original=build/'vkfft_frontier_original';expanded=build/'vkfft_frontier_v2'
    attestation=output/'build_attestation.json'
    cmd=[sys.executable,str(knobs/'make_attestation.py'),'--code-commit',parent,'--vkfft-commit',upstream,
         '--generated-provenance',str(provenance),'--original',str(original),'--expanded',str(expanded),'--output',str(attestation)]
    for command in commands:cmd+=['--build-command',json.dumps(command)]
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    remaining=deadline-time.monotonic()
    if remaining<660:raise ValueError('no guarded extension budget left')
    cmd=[sys.executable,str(knobs/'runner.py'),'--gpu',gpu,'--original',str(original),'--expanded',str(expanded),
         '--manifest',str(output/'manifest.json'),'--generated-provenance',str(provenance),'--build-attestation',str(attestation),
         '--output',str(output/'rows.jsonl'),'--budget-ledger',str(output/'budget.jsonl'),'--log',str(output/'guard.jsonl'),
         '--incidents',str(output/'incidents.jsonl'),'--lock',str(lockpath),'--seconds',str(remaining)]
    (output/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    return subprocess.run(cmd,cwd=ROOT,env=env).returncode


if __name__=='__main__':raise SystemExit(main())
