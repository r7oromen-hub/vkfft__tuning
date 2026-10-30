"""Reserve extension budget and build/run only a gated rotation experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import budget
import device_lock
import rotation_evidence
import runner

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2];GPU=runner.GPU
SOURCES=('prepare_rotation.py','rotation_pool.hpp','CMakeLists_rotation.txt','test_rotation_pool.cpp',
         'test_rotation_generation.py','test_rotation_controller.py','rotation_cpu/CMakeLists.txt','rotation_evidence.py','rotation_controller.py',
         'rotation_gpu_guard.py','rotation_runner.py','run_suite.py','search_controller.py','validation.py',
         'prepare_headers.py','enumerate_candidates.py','selection.py','manifest.py','identity.py',
         'audit_confirmation.py','audit_two_sessions.py','audit_search.py','audit_guard.py','audit_admission.py',
         'audit_screen.py','audit_space.py','audit_pairs.py','audit_budget.py','budget.py','device_lock.py','runner.py',
         '../original_compare/original_compare_guard.py','../original_compare/original_compare_controller.py')


def tree_sha(path):
    path=Path(path).resolve(strict=True);digest=hashlib.sha256()
    for item in sorted((x for x in path.rglob('*') if x.is_file()),key=lambda x:x.relative_to(path).as_posix()):
        rel=item.relative_to(path).as_posix().encode();digest.update(len(rel).to_bytes(4,'big'));digest.update(rel)
        digest.update(bytes.fromhex(runner.file_sha(item)))
    return digest.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','results','expanded-generated-root','plan','gate','protocol','first-gate','first-audit','second-audit','combined-audit',
                 'first-rows','first-guard','first-incidents','first-budget','second-rows','second-guard','second-incidents','second-budget'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--timeout',type=int,default=600)
    a=p.parse_args(argv)
    if not 1<=a.timeout<=3600:p.error('invalid timeout')
    a.results.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PATH']='/usr/local/cuda-13.0/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']='/usr/local/cuda-13.0/lib64:'+env.get('LD_LIBRARY_PATH','')
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+GPU+'.lock')) as lock:
        b=budget.Budget(a.state)
        if b.state['phase']!='extension' or b.state['stopped'] or b.state.get('inflight'):
            raise ValueError('extension budget unavailable')
        work=uuid.uuid4().hex
        if not b.may_start('rotation_prebuild_gate_audit',expected_upper_seconds=600,work_id=work):
            b.checkpoint('rotation_uncovered',reason='insufficient_extension_budget_for_gate_audit');return 2
        try: plan,gate,cases,l2=rotation_evidence.verify(a)
        except BaseException as error:
            b.finish_work('exception',work_id=work,error=f'{type(error).__name__}: {error}');raise
        b.finish_work('completed',work_id=work,gate_sha256=runner.file_sha(a.gate),confirmed_count=len(cases))
        if not cases:
            b.checkpoint('rotation_uncovered',reason='no_two_session_confirmed_winners',gate_sha256=runner.file_sha(a.gate));return 0
        for name,path in plan['binaries'].items():
            if runner.file_sha(path)!=plan['sha256'][name]:raise ValueError('confirmed hot binary changed: '+name)
        generated=a.expanded_generated_root.resolve(strict=True)
        events=[json.loads(line) for line in b.ledger.read_text(encoding='utf8').splitlines()]
        builds=[x for x in events if x.get('event')=='checkpoint' and x.get('kind') in ('build_complete','repair_build_complete') and
                x.get('sha256')==plan['sha256']['expanded'] and Path(x.get('binary','')).resolve()==Path(plan['binaries']['expanded']).resolve()]
        if len(builds)!=1 or Path(builds[0]['provenance']).resolve()!=generated/'expanded_provenance.json':
            raise ValueError('expanded generated root is not the confirmed hot build')
        # Freeze current inputs. Tree digests do not themselves prove upstream
        # revision; original source/compile provenance must be audited separately.
        inventory=uuid.uuid4().hex
        if not b.may_start('rotation_source_inventory',expected_upper_seconds=420,work_id=inventory):
            b.checkpoint('rotation_uncovered',reason='insufficient_extension_budget_for_source_inventory');return 2
        try:
            freeze={name:runner.file_sha(HERE/name) for name in SOURCES}
            input_digest={'expanded_generated_tree':tree_sha(generated),'original_header_tree':tree_sha(ROOT/'VkFFT/vkFFT'),
                          'instrumentation_reference':runner.file_sha(HERE.parent/'instrumentation/cpu_reference.hpp'),
                          'instrumentation_source':runner.file_sha(HERE.parent/'instrumentation/vkfft_instrumented_benchmark.cpp'),
                          'instrumentation_trace':runner.file_sha(HERE.parent/'instrumentation/trace.hpp'),
                          'instrumentation_prepare':runner.file_sha(HERE.parent/'instrumentation/prepare_headers.py')}
        except BaseException as error:
            b.finish_work('exception',work_id=inventory,error=f'{type(error).__name__}: {error}');raise
        b.finish_work('completed',work_id=inventory)
        b.checkpoint('rotation_source_freeze',source_hashes=freeze,input_tree_sha256=input_digest,
                     gate_sha256=runner.file_sha(a.gate),confirmed_hot_sha256=plan['sha256'])
        source=a.state/'rotation_generated';build=a.state/'rotation_build';cpu_build=a.state/'rotation_cpu_build'
        if any(path.exists() for path in (source,build,cpu_build)):raise FileExistsError('rotation build paths must be new')
        commands=[
            [sys.executable,str(HERE/'prepare_rotation.py'),'--expanded-generated-root',str(generated),'--output',str(source)],
            ['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
             '-DROTATION_GENERATED_ROOT='+str(source),'-DEXPANDED_GENERATED_ROOT='+str(generated),
             '-DVKFFT_INCLUDE_ROOT='+str(ROOT/'VkFFT/vkFFT'),'-DINSTRUMENTATION_ROOT='+str(HERE.parent/'instrumentation'),
             '-DCUDAToolkit_ROOT=/usr/local/cuda-13.0','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
            ['cmake','--build',str(build),'--parallel','2'],
            ['cmake','-S',str(HERE/'rotation_cpu'),'-B',str(cpu_build),'-DCMAKE_BUILD_TYPE=Release'],
            ['cmake','--build',str(cpu_build),'--parallel','2'],
            ['ctest','--test-dir',str(cpu_build),'--output-on-failure'],
            [sys.executable,str(HERE/'test_rotation_controller.py')]]
        for index,cmd in enumerate(commands):
            work=uuid.uuid4().hex
            if not b.may_start('rotation_build',expected_upper_seconds=420,work_id=work,argv=cmd):
                b.checkpoint('rotation_uncovered',reason='insufficient_extension_budget_for_build',command_index=index);return 2
            try:
                proc=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=400,pass_fds=(lock.fd,))
            except BaseException as error:
                b.finish_work('exception',work_id=work,error=f'{type(error).__name__}: {error}');raise
            path=a.results/f'rotation_build_{index}_{work}.json'
            with path.open('x',encoding='utf8') as out:
                json.dump({'argv':cmd,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr},out,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
            b.finish_work('ok' if proc.returncode==0 else 'failed',work_id=work,result_path=str(path),exit_code=proc.returncode)
            if proc.returncode:
                b.checkpoint('rotation_uncovered',reason='rotation_build_failed',result_path=str(path));return proc.returncode
        provenance_work=uuid.uuid4().hex
        if not b.may_start('rotation_build_provenance',expected_upper_seconds=420,work_id=provenance_work):
            b.checkpoint('rotation_uncovered',reason='insufficient_extension_budget_for_build_provenance');return 2
        try:
            after={'expanded_generated_tree':tree_sha(generated),'original_header_tree':tree_sha(ROOT/'VkFFT/vkFFT'),
                   'instrumentation_reference':runner.file_sha(HERE.parent/'instrumentation/cpu_reference.hpp'),
                   'instrumentation_source':runner.file_sha(HERE.parent/'instrumentation/vkfft_instrumented_benchmark.cpp'),
                   'instrumentation_trace':runner.file_sha(HERE.parent/'instrumentation/trace.hpp'),
                   'instrumentation_prepare':runner.file_sha(HERE.parent/'instrumentation/prepare_headers.py')}
            if any(runner.file_sha(HERE/name)!=digest for name,digest in freeze.items()) or input_digest!=after:
                raise RuntimeError('rotation source or input trees changed during build')
            commands_json=json.loads((build/'compile_commands.json').read_text(encoding='utf8'))
            original_commands=[x['command'] for x in commands_json if 'vkfft_rotation_original.dir' in x['command']]
            orig_include=str((ROOT/'VkFFT/vkFFT').resolve());gen_include=str((generated/'include').resolve())
            if len(original_commands)!=1 or orig_include not in original_commands[0] or gen_include not in original_commands[0] or original_commands[0].index(orig_include)>original_commands[0].index(gen_include):
                raise ValueError('original header search precedence not established')
            deps=[x for x in build.rglob('*.o.d') if 'vkfft_rotation_original.dir' in x.as_posix()]
            if len(deps)!=1:raise ValueError('original compiler dependency file missing or ambiguous')
            deptext=deps[0].read_text(encoding='utf8')
            if str((generated/'include/vkFFT').resolve()) in deptext or str((generated/'include/vkFFT.h').resolve()) in deptext or str((ROOT/'VkFFT/vkFFT/vkFFT.h').resolve()) not in deptext:
                raise ValueError('original compiler dependencies do not prove original include path')
            binaries={'original':build/'vkfft_rotation_original','expanded':build/'vkfft_rotation_expanded'}
            hashes={name:runner.file_sha(path) for name,path in binaries.items()}
        except BaseException as error:
            b.finish_work('exception',work_id=provenance_work,error=f'{type(error).__name__}: {error}');raise
        b.finish_work('completed',work_id=provenance_work,sha256=hashes)
        b.checkpoint('rotation_build_complete',binaries={k:str(v) for k,v in binaries.items()},sha256=hashes,
                     source_hashes=freeze,input_tree_sha256=input_digest,provenance=str(source/'rotation_provenance.json'),
                     compile_commands=str(build/'compile_commands.json'),original_dependencies=str(deps[0]))
        # Keep enough stage time for guarded gate re-verification, one
        # calibration and one complete pair before starting the session.
        if b.remaining()<=600+3*(a.timeout+30)+30:
            b.checkpoint('rotation_uncovered',reason='insufficient_extension_budget_for_guard_and_pair');return 2
        token=uuid.uuid4().hex
        cmd=[sys.executable,str(HERE/'rotation_gpu_guard.py'),'--gpu',GPU,
             '--log',str(a.results/f'rotation_{token}.guard.jsonl'),'--incidents',str(a.results/f'rotation_{token}.incidents.jsonl'),
             '--budget-dir',str(a.state),'--output',str(a.results/f'rotation_{token}.jsonl'),
             '--rotation-original-exe',str(binaries['original']),'--rotation-expanded-exe',str(binaries['expanded']),
             '--rotation-original-sha256',hashes['original'],'--rotation-expanded-sha256',hashes['expanded'],'--timeout',str(a.timeout)]
        for name in ('plan','gate','protocol','first_gate','first_audit','second_audit','combined_audit','first_rows','first_guard','first_incidents','first_budget',
                     'second_rows','second_guard','second_incidents','second_budget'):
            cmd.extend(['--'+name.replace('_','-'),str(getattr(a,name))])
        b.checkpoint('rotation_guard_launch',session_id=token,argv=cmd,source_hashes=freeze,input_tree_sha256=input_digest)
        proc=subprocess.run(cmd,cwd=ROOT,env=env,pass_fds=(lock.fd,))
        b=budget.Budget(a.state);b.checkpoint('rotation_guard_exit',session_id=token,exit_code=proc.returncode)
        return proc.returncode


if __name__=='__main__':sys.exit(main())
