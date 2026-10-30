"""Continue the same admission budget with a separately preserved repair build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import budget
import device_lock
import runner

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GPU=runner.GPU
SOURCES=tuple(runner.SOURCE_FILES)+('runner_repair.py',)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',required=True,type=Path);p.add_argument('--results',required=True,type=Path)
    p.add_argument('--action',required=True,choices=('build-repair','run-repair'))
    a=p.parse_args()
    env=os.environ.copy();env['PATH']='/usr/local/cuda-13.0/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']='/usr/local/cuda-13.0/lib64:'+env.get('LD_LIBRARY_PATH','')
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+GPU+'.lock')) as lock:
        b=budget.Budget(a.state)
        if b.state is None or b.state['phase']!='admission' or b.state['stopped'] or b.state.get('inflight'):
            raise RuntimeError('admission unavailable or unresolved work remains')
        generated=a.state/'generated_repair_v2';build=a.state/'build_repair_v2'
        if a.action=='build-repair':
            sources={name:runner.file_sha(HERE/name) for name in SOURCES}
            b.checkpoint('repair_source_freeze',reason='post-initialize axisSwapped reset caused identity validation failure',source_hashes=sources)
            commands=[
                [sys.executable,str(HERE/'prepare_headers.py'),'--source',str(ROOT/'VkFFT/vkFFT'),'--output',str(generated/'include')],
                ['cmake','-S',str(HERE),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                 '-DEXPANDED_GENERATED_ROOT='+str(generated),'-DCUDAToolkit_ROOT=/usr/local/cuda-13.0','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
                ['cmake','--build',str(build),'--parallel','2'],
                ['ctest','--test-dir',str(build),'--output-on-failure']]
            for index,cmd in enumerate(commands):
                job=uuid.uuid4().hex
                if not b.may_start('repair_build',expected_upper_seconds=900,job_id=job,argv=cmd): return 2
                start=time.monotonic()
                try:
                    proc=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=850,pass_fds=(lock.fd,))
                    result=dict(job_id=job,argv=cmd,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,elapsed_seconds=time.monotonic()-start)
                except BaseException as error:
                    b.finish_work('exception',error=f'{type(error).__name__}: {error}');raise
                log=a.results/f'repair_build_{index}_{job}.json'
                with log.open('x',encoding='utf8') as out:
                    json.dump(result,out,sort_keys=True,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
                b.finish_work('ok' if proc.returncode==0 else 'failed',result_path=str(log),exit_code=proc.returncode)
                if proc.returncode: return proc.returncode
            if any(runner.file_sha(HERE/name)!=value for name,value in sources.items()): raise RuntimeError('source changed during build')
            exe=build/'vkfft_expanded_candidate'
            b.checkpoint('repair_build_complete',binary=str(exe),sha256=runner.file_sha(exe),source_hashes=sources,
                         provenance=str(generated/'expanded_provenance.json'),compile_commands=str(build/'compile_commands.json'))
            return 0
        if b.remaining() <= 360: raise RuntimeError('insufficient admission budget for guard')
        events=[json.loads(line) for line in b.ledger.read_text(encoding='utf8').splitlines()]
        built=[e for e in events if e.get('event')=='checkpoint' and e.get('kind')=='repair_build_complete']
        expanded=build/'vkfft_expanded_candidate';original=ROOT/'build_original_compare/vkfft_original_control'
        if len(built)!=1 or runner.file_sha(expanded)!=built[0]['sha256']: raise RuntimeError('repair build hash mismatch')
        for name,digest in built[0]['source_hashes'].items():
            if runner.file_sha(HERE/name)!=digest: raise RuntimeError('repair source mismatch: '+name)
        if runner.file_sha(original)!='3ca7ee798d8962b11e501923b86a70eba3d74d019d2e4e77f86aa9b466cae445': raise RuntimeError('original binary hash mismatch')
        session=uuid.uuid4().hex
        cmd=[sys.executable,str(HERE/'gpu_guard.py'),'--gpu',GPU,'--log',str(a.results/f'admission_v2_{session}.guard.jsonl'),
             '--incidents',str(a.results/f'admission_v2_{session}.incidents.jsonl'),'--expanded-exe',str(expanded),
             '--original-exe',str(original),'--budget-dir',str(a.state),'--output',str(a.results/f'admission_v2_{session}.jsonl'),
             '--admission-protocol',str(HERE/'admission_protocol.json')]
        b.checkpoint('repair_guard_launch',session_id=session,argv=cmd)
        proc=subprocess.run(cmd,cwd=ROOT,env=env,pass_fds=(lock.fd,))
        b=budget.Budget(a.state);b.checkpoint('repair_guard_exit',session_id=session,exit_code=proc.returncode)
        return proc.returncode


if __name__=='__main__': sys.exit(main())
