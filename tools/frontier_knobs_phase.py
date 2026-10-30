"""Build and screen 30 explicitly frozen joint-knob configurations, after v2 search."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
GPU='GPU-3559ec6b-6278-8f93-0706-5f8a85c4911b'


def main():
    output=Path(os.environ['EXPERIMENT_ARTIFACT_DIR'])
    prior=ROOT/'experiments/artifacts/frontier-20261031-admission-001'
    exhausted=ROOT/'experiments/artifacts/frontier-20261031-exhaust128-001/search'
    record=json.loads((ROOT/'experiments/records/frontier-20261031-exhaust128-001.json').read_text())
    if record['status']!='completed':raise ValueError('prior GPU search is not terminal and completed')
    guards=[json.loads(x) for x in (exhausted/'guard.jsonl').read_text().splitlines()]
    if (exhausted/'incidents.jsonl').stat().st_size or not any(r.get('event')=='guard_complete' and r.get('clean') is True for r in guards):
        raise ValueError('prior guard requires audit')
    start=json.loads((prior/'campaign_start.json').read_text())
    first=json.loads((prior/'budget.jsonl').read_text().splitlines()[0])['state']
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=first['boot']:raise ValueError('campaign boot changed')
    now=time.monotonic()
    if now<start['start_monotonic']:raise ValueError('clock moved backwards')
    # At most one hour for this extension, reserving one hour for confirmation.
    deadline=min(now+3600,start['start_monotonic']+start['total_reserved_wall_seconds']-3600)
    if deadline-now<1200:raise ValueError('insufficient global extension budget')
    env=dict(os.environ)
    env['PATH']='/usr/local/cuda-13.0/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']='/usr/local/cuda-13.0/lib64:'+env.get('LD_LIBRARY_PATH','')
    knobs=ROOT/'test/plan_selection_study/frontier_knobs'
    generated=output/'generated';build=output/'build'
    lockpath=Path('/run/lock')/('vkfft-study-'+GPU+'.lock')
    sys.path.insert(0,str(ROOT/'test/plan_selection_study/expanded_search'))
    import device_lock
    commands=[
        [sys.executable,str(knobs/'prepare_headers.py'),'--source',str(ROOT/'VkFFT/vkFFT'),'--output',str(generated/'include')],
        ['cmake','-S',str(ROOT/'test/plan_selection_study/frontier_search'),'-B',str(build),
         '-DCMAKE_BUILD_TYPE=Release','-DFRONTIER_GENERATED_ROOT='+str(generated),
         '-DCUDAToolkit_ROOT=/usr/local/cuda-13.0','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
        ['cmake','--build',str(build),'--parallel','2'],
        ['ctest','--test-dir',str(build),'--output-on-failure'],
        [sys.executable,str(knobs/'make_manifest.py'),'--spec',str(ROOT/'experiments/frontier_knobs_n128_b16.json'),
         '--output',str(output/'manifest.json')],
    ]
    with device_lock.DeviceLock(lockpath):
        check=subprocess.run(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
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
    cmd=[sys.executable,str(knobs/'runner.py'),'--gpu',GPU,'--original',str(original),'--expanded',str(expanded),
         '--manifest',str(output/'manifest.json'),'--generated-provenance',str(provenance),'--build-attestation',str(attestation),
         '--output',str(output/'rows.jsonl'),'--budget-ledger',str(output/'budget.jsonl'),'--log',str(output/'guard.jsonl'),
         '--incidents',str(output/'incidents.jsonl'),'--lock',str(lockpath),'--seconds',str(remaining)]
    (output/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    return subprocess.run(cmd,cwd=ROOT,env=env).returncode


if __name__=='__main__':raise SystemExit(main())
