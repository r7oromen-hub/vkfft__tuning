"""Exclusive device reservation and persistent-budget admission launcher."""
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GPU='GPU-492bdfa1-90bc-e9d3-a759-c2c9ea7f629e'
SOURCE_FILES=('prepare_headers.py','candidate.hpp','admission_protocol.json','study_protocol.json',
              'runner.py','gpu_guard.py','expanded_controller.py','run_suite.py','validation.py',
              'identity.py','budget.py','device_lock.py','CMakeLists.txt')


def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',required=True,type=Path);p.add_argument('--results',required=True,type=Path)
    p.add_argument('--action',required=True,choices=('start-admission','run-admission'))
    a=p.parse_args();a.state.mkdir(parents=True,exist_ok=True);a.results.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PATH']='/usr/local/cuda-13.0/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']='/usr/local/cuda-13.0/lib64:'+env.get('LD_LIBRARY_PATH','')
    # Shared across every state directory used for this physical GPU.
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+GPU+'.lock')) as lock:
        b=budget.Budget(a.state)
        if a.action=='start-admission':
            b.start()
            sources={name:file_sha(HERE/name) for name in SOURCE_FILES}
            b.checkpoint('source_freeze',source_hashes=sources)
            generated=a.state/'generated'; build=a.state/'build'
            commands=[
                [sys.executable,str(HERE/'prepare_headers.py'),'--source',str(ROOT/'VkFFT/vkFFT'),'--output',str(generated/'include')],
                ['cmake','-S',str(HERE),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                 '-DEXPANDED_GENERATED_ROOT='+str(generated),'-DCUDAToolkit_ROOT=/usr/local/cuda-13.0',
                 '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
                ['cmake','--build',str(build),'--parallel','2'],
                ['ctest','--test-dir',str(build),'--output-on-failure']]
            for index,cmd in enumerate(commands):
                job_id=uuid.uuid4().hex
                if not b.may_start('build',expected_upper_seconds=900,job_id=job_id,argv=cmd):
                    print('admission budget does not permit another build step',file=sys.stderr);return 2
                started=time.monotonic()
                try:
                    proc=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=850,pass_fds=(lock.fd,))
                    result=dict(job_id=job_id,argv=cmd,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,
                                elapsed_seconds=time.monotonic()-started)
                except BaseException as error:
                    b.finish_work('exception',error=f'{type(error).__name__}: {error}')
                    raise
                log=a.results/f'build_{index}_{job_id}.json'
                with log.open('x',encoding='utf8') as out:
                    json.dump(result,out,sort_keys=True,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
                b.finish_work('ok' if proc.returncode==0 else 'failed',result_path=str(log),exit_code=proc.returncode)
                if proc.returncode: return proc.returncode
            exe=build/'vkfft_expanded_candidate'
            if any(file_sha(HERE/name)!=digest for name,digest in sources.items()):
                raise RuntimeError('source/protocol changed during build')
            b.checkpoint('build_complete',binary=str(exe),sha256=file_sha(exe),source_hashes=sources,
                         provenance=str(generated/'expanded_provenance.json'),compile_commands=str(build/'compile_commands.json'))
            return 0
        if b.state is None or b.state['phase']!='admission' or b.state['stopped'] or b.state.get('inflight'):
            raise RuntimeError('admission budget unavailable or unresolved work remains')
        if b.remaining() <= 360: raise RuntimeError('insufficient admission budget to start guarded session')
        expanded=a.state/'build/vkfft_expanded_candidate'
        original=ROOT/'build_original_compare/vkfft_original_control'
        events=[json.loads(line) for line in b.ledger.read_text(encoding='utf8').splitlines()]
        built=[e for e in events if e.get('event')=='checkpoint' and e.get('kind')=='build_complete']
        if len(built)!=1 or file_sha(expanded)!=built[0]['sha256']:
            raise RuntimeError('expanded executable does not match unique frozen build checkpoint')
        for name,expected in built[0]['source_hashes'].items():
            if file_sha(HERE/name)!=expected:
                raise RuntimeError('frozen source/protocol changed after build: '+name)
        if file_sha(original) != '3ca7ee798d8962b11e501923b86a70eba3d74d019d2e4e77f86aa9b466cae445':
            raise RuntimeError('original control binary hash differs from frozen evidence')
        session=uuid.uuid4().hex
        cmd=[sys.executable,str(HERE/'gpu_guard.py'),'--gpu',GPU,
             '--log',str(a.results/f'admission_{session}.guard.jsonl'),
             '--incidents',str(a.results/f'admission_{session}.incidents.jsonl'),
             '--expanded-exe',str(expanded),'--original-exe',str(original),
             '--budget-dir',str(a.state),'--output',str(a.results/f'admission_{session}.jsonl'),
             '--admission-protocol',str(HERE/'admission_protocol.json')]
        b.checkpoint('guard_launch',session_id=session,argv=cmd)
        proc=subprocess.run(cmd,cwd=ROOT,env=env,pass_fds=(lock.fd,))
        b=budget.Budget(a.state)
        b.checkpoint('guard_exit',session_id=session,exit_code=proc.returncode)
        return proc.returncode


if __name__=='__main__': sys.exit(main())
