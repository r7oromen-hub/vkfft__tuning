"""Advance an independently admitted study into one exclusive search session."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import budget
import device_lock
import runner
import search_controller

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GPU=runner.GPU
FILES=('search_runner.py','phase_gpu_guard.py','search_controller.py','manifest.py','selection.py',
       'enumerate_candidates.py','audit_admission.py','search_protocol_v2.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',required=True,type=Path);p.add_argument('--results',required=True,type=Path)
    p.add_argument('--admission-stem',required=True);p.add_argument('--admission-gate',required=True,type=Path)
    a=p.parse_args()
    env=os.environ.copy();env['PATH']='/usr/local/cuda-13.0/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']='/usr/local/cuda-13.0/lib64:'+env.get('LD_LIBRARY_PATH','')
    with device_lock.DeviceLock(Path('/run/lock')/('vkfft-study-'+GPU+'.lock')) as lock:
        b=budget.Budget(a.state)
        if b.state is None or b.state['phase']!='admission' or b.state['stopped'] or b.state.get('inflight'):
            raise RuntimeError('cannot advance: admission budget not available')
        rows=a.results/(a.admission_stem+'.jsonl');guard=a.results/(a.admission_stem+'.guard.jsonl')
        incidents=a.results/(a.admission_stem+'.incidents.jsonl');protocol=HERE/'admission_protocol.json'
        expanded=a.state/'build_repair_v2/vkfft_expanded_candidate';original=ROOT/'build_original_compare/vkfft_original_control'
        hashes={'expanded':runner.file_sha(expanded),'original':runner.file_sha(original)}
        evidence={'rows':runner.file_sha(rows),'guard':runner.file_sha(guard),'incidents':runner.file_sha(incidents),'protocol':runner.file_sha(protocol)}
        gate=json.loads(a.admission_gate.read_text(encoding='utf8'))
        search_controller.check_admission_gate(gate,evidence,hashes)
        events=[json.loads(line) for line in b.ledger.read_text(encoding='utf8').splitlines()]
        built=[e for e in events if e.get('event')=='checkpoint' and e.get('kind')=='repair_build_complete']
        if len(built)!=1 or built[0]['sha256']!=hashes['expanded']: raise RuntimeError('expanded binary does not match admission build')
        for name,digest in built[0]['source_hashes'].items():
            if runner.file_sha(HERE/name)!=digest: raise RuntimeError('admission source changed: '+name)
        if hashes['original']!='3ca7ee798d8962b11e501923b86a70eba3d74d019d2e4e77f86aa9b466cae445':
            raise RuntimeError('original control binary changed')
        sources={name:runner.file_sha(HERE/name) for name in FILES}
        b.checkpoint('search_source_freeze',source_hashes=sources,gate_sha256=runner.file_sha(a.admission_gate),evidence_sha256=evidence)
        if not b.advance('independent_admission_gate_pass'): raise RuntimeError('global deadline already passed')
        if b.remaining()<=660: raise RuntimeError('insufficient search budget for guarded operation')
        session=uuid.uuid4().hex
        cmd=[sys.executable,str(HERE/'phase_gpu_guard.py'),'--kind','search','--gpu',GPU,
             '--log',str(a.results/f'search_{session}.guard.jsonl'),'--incidents',str(a.results/f'search_{session}.incidents.jsonl'),
             '--expanded-exe',str(expanded),'--original-exe',str(original),'--budget-dir',str(a.state),
             '--output',str(a.results/f'search_{session}.jsonl'),'--manifest-dir',str(a.results/f'manifests_{session}'),
             '--study-protocol',str(HERE/'search_protocol_v2.json'),'--admission-protocol',str(protocol),
             '--admission-rows',str(rows),'--admission-guard',str(guard),'--admission-incidents',str(incidents),
             '--admission-gate',str(a.admission_gate)]
        b.checkpoint('search_guard_launch',session_id=session,argv=cmd,source_hashes=sources)
        proc=subprocess.run(cmd,cwd=ROOT,env=env,pass_fds=(lock.fd,))
        b=budget.Budget(a.state);b.checkpoint('search_guard_exit',session_id=session,exit_code=proc.returncode)
        return proc.returncode


if __name__=='__main__':sys.exit(main())
