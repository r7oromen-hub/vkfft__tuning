#!/usr/bin/env python3
"""Offline build/budget audit for the v2 rotation attempt that never ran GPU."""
import json
from pathlib import Path
import audit_budget
import audit_rotation as base
import audit_rotation_v2 as repair
import replay_budget_snapshots


HERE=Path(__file__).resolve().parent;D=HERE/'results_20261030';R=D/'rotation_immutable_local';RUN=R/'runtime_20261030';RESULT=R/'results_20261030'


def main():
    ledger=base.read(RESULT/'budget_final_snapshot.jsonl');protocol=base.load(HERE/'study_protocol.json')
    prior=[D/'admission_dc66fec2f0ec471fb6723e44967a4ceb.jsonl',D/'admission_v2_b6fcc72e9ff2413783c1b5710a9ba963.jsonl',
           D/'search_immutable_local/search_7c86f6f3e3ea4e29ad024a63743d3872.jsonl',
           D/'confirmation_first_d0a3389193ab406c97c149c0752597f3.jsonl',D/'confirmation_second_d3e5f3c437f24030830f8cb865247b9a.jsonl']
    budget=audit_budget.audit(ledger,protocol,replay_budget_snapshots.ids(prior));errors=list(budget['errors'])
    builds=[x for x in ledger if x.get('event')=='checkpoint' and x.get('kind')=='rotation_repair_build_complete']
    freeze=[x for x in ledger if x.get('event')=='checkpoint' and x.get('kind')=='rotation_repair_source_freeze']
    blocked=[x for x in ledger if x.get('event')=='checkpoint' and x.get('kind')=='rotation_uncovered' and
             x.get('reason')=='insufficient_extension_budget_for_guard_and_pair']
    if len(builds)!=1 or len(freeze)!=1 or len(blocked)!=1:raise ValueError('missing or duplicate build/freeze/uncovered checkpoint')
    b=builds[0];oldfail=RESULT/'rotation_build_2_ab1cf182335d47118a0bcf61c0aa9c59.json'
    oldsrc=RUN/'rotation_generated/rotating_benchmark.cpp';oldprov=RUN/'rotation_generated/rotation_provenance.json'
    oldsteps=[x for x in ledger if x.get('event')=='work_finished' and x.get('kind')=='rotation_build'][:3]
    oldargv=oldsteps[1]['identity']['argv']
    original_prefix=next(x.split('=',1)[1] for x in oldargv if x.startswith('-DVKFFT_INCLUDE_ROOT='))
    generated_prefix=next(x.split('=',1)[1] for x in oldargv if x.startswith('-DEXPANDED_GENERATED_ROOT='))+'/include'
    new=RUN/'rotation_generated_repair_v2';build=RUN/'rotation_build_repair_v2';generated=RUN/'generated_repair_v2'
    build_report=repair.repair_audit_build(ledger,{'sha256':b['sha256']},build/'vkfft_rotation_original',build/'vkfft_rotation_expanded',
        new,generated/'expanded_benchmark.cpp',build/'compile_commands.json',
        build/'CMakeFiles/vkfft_rotation_original.dir/rotating_benchmark.cpp.o.d',original_prefix,generated_prefix,
        HERE,generated,HERE.parents[2]/'VkFFT/vkFFT',HERE.parent/'instrumentation',oldfail,oldsrc,oldprov)
    errors.extend(build_report['errors'])
    start=ledger.index(freeze[0]);end=ledger.index(b)
    steps=[x for x in ledger[start:end] if x.get('event')=='work_finished' and x.get('kind')=='rotation_build']
    for i,step in enumerate(steps):
        name=Path(step['result_path']).name;log=base.load(RESULT/name)
        if log.get('returncode')!=step['exit_code'] or log.get('argv')!=step['identity']['argv'] or log['returncode']!=0:
            errors.append('repair build log mismatch: '+name)
        if i==5 and ('100% tests passed, 0 tests failed out of 5' not in log.get('stdout','')):
            errors.append('five CPU CTest successes not evidenced')
    if any(x.get('kind') in ('rotation_guard_launch','rotation_repair_guard_launch') for x in ledger):
        errors.append('rotation GPU guard unexpectedly launched')
    if any(x.get('kind') in ('rotation_calibration','rotation_pair','rotation_gate_verification') and x.get('event')=='work_start_allowed' for x in ledger):
        errors.append('rotation GPU execution unexpectedly budgeted')
    if ledger[-1]['event']!='budget_stop' or not ledger[-1]['state'].get('stopped') or ledger[-1]['state'].get('inflight') is not None:
        errors.append('budget terminal is not stopped cleanly')
    if base.load(D/'confirmation_local_chain.json')['status']!='pass':errors.append('confirmation gate chain is not audited')
    result={'status':'fail' if errors else 'pass','errors':errors,'budget':budget,'build':build_report,
            'rotation_gpu_status':'uncovered_no_guard_or_execution',
            'reason':blocked[0]['reason'],'remaining_seconds':blocked[0]['remaining_seconds'],
            'default_timeout_reservation_seconds':600+3*(600+30)+30,
            'reservation_caveat':'2520 seconds assumes the runner default timeout of 600 seconds; actual launcher argv was not frozen in an independent file',
            'scope':'CUDA host compilation and CPU mock tests only; no GPU graph, numerical or performance result'}
    target=D/'rotation_uncovered_local_audit.json'
    with target.open('x',encoding='utf8') as out:json.dump(result,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps({k:result[k] for k in ('status','errors','rotation_gpu_status','reason','remaining_seconds')},ensure_ascii=False))
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
