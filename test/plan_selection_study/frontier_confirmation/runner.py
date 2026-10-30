#!/usr/bin/env python3
"""Run one new, independently guarded 10-round frontier confirmation session."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

import evidence
frontier=evidence.frontier
suite=frontier.suite
validation=evidence.validation
search_controller=evidence.search_controller
comparison=evidence.comparison
guard_common=frontier.guard_common


def run_session(args, ledger, contract, proof):
    if os.environ.get('VKFFT_STUDY_GUARD_UUID') != args.gpu or not os.environ.get('VKFFT_STUDY_GUARD_ID'):
        raise ValueError('GPU guard required')
    session=uuid.uuid4().hex
    def emit(row):
        out=dict(row);out.update(session_id=session,guard_id=os.environ['VKFFT_STUDY_GUARD_ID'],contract=contract)
        frontier.append(args.output,out)
    emit(dict(record_type='session_start',gpu=args.gpu,session_kind=args.session_kind))
    def one(n,batch,name,req=None,*,phase='identity',graph=64,round_no=0,order=0,timing=False):
        if not ledger.start(dict(n=n,batch=batch,candidate_id=name,phase=phase,round=round_no),args.timeout+30):
            raise frontier.BudgetStop()
        original=name=='original'
        args.exe=str(args.original if original else args.expanded)
        args.exe_sha256=contract['binaries']['original' if original else 'expanded'];args.session_id=session
        w=dict(n=n,batch=batch,seed=42,config={},pattern='random',graph_repeats=graph,
               warmup=5 if timing else 0,warmup_ms=1000 if timing else 0,repeats=31 if timing else 1,study_mode='execute')
        row=suite.run_one(args,w,(name,validation.request_env(req) if req else {}),phase,round_no,order)
        emit(row);ledger.finish(row['status']);return row
    contexts={}
    for case in proof['cases']:
        n,batch,cid,request=(case['n'],case['batch'],case['candidate_id'],case['request'])
        key=(n,batch)
        if key not in contexts:
            original=validation.check(one(n,batch,'original'),args.gpu,original=True)
            default=validation.check(one(n,batch,'expanded_default'),args.gpu)
            comparison.compare(default,original)
            identity,reason=frontier.representable(default)
            if identity is None: raise ValueError('unsupported identity: '+str(reason))
            replay=validation.check(one(n,batch,'identity',identity),args.gpu,request=identity)
            if validation.signature(default)!=validation.signature(replay): raise ValueError('identity replay mismatch')
            orig_sig=search_controller.source_signature(original,original=True)
            orig_hash=evidence.signature_hash(orig_sig)
            if orig_hash!=case['screen_original_signature_sha256']: raise ValueError('original differs from screened plan')
            emit(dict(record_type='workload_status',n=n,batch=batch,status='identity_passed',request=identity))
            graph=64
            while True:
                result=validation.check(one(n,batch,'original',phase='calibration',graph=graph),args.gpu,original=True)
                if search_controller.source_signature(result,original=True)!=orig_sig: raise ValueError('calibration plan drift')
                if min(result['raw_interval_ms_samples'])>=2: break
                if graph>=2048: raise ValueError('calibration failed')
                graph*=2
            contexts[key]=(orig_sig,orig_hash,graph)
        orig_sig,orig_hash,graph=contexts[key]
        if orig_hash!=case['screen_original_signature_sha256']: raise ValueError('screen original signatures differ')
        emit(dict(record_type='candidate_start',n=n,batch=batch,candidate_id=cid,request=request,
                  screen_original_signature_sha256=orig_hash,
                  screen_candidate_signature_sha256=case['screen_candidate_signature_sha256']))
        for round_no in range(10):
            results={}; rows={}
            for ix,name in enumerate(evidence.pair_order(contract['plan_sha256'],args.session_kind,n,batch,cid,round_no)):
                is_original=name=='original';variant='original' if is_original else cid
                row=one(n,batch,variant,None if is_original else request,phase='confirmation_timing',graph=graph,
                        round_no=round_no,order=ix,timing=True)
                result=validation.check(row,args.gpu,original=is_original,request=None if is_original else request,timing=True)
                if result['warmup_actual']<5 or result['warmup_wall_ms']<1000: raise ValueError('short warmup')
                sig=search_controller.source_signature(result,original=is_original)
                if evidence.signature_hash(sig)!=(orig_hash if is_original else case['screen_candidate_signature_sha256']):
                    raise ValueError('confirmation plan/source/resources differ from screen')
                results[name]=statistics.median(result['exec_ms_samples']); rows[name]=row
            emit(dict(record_type='confirmation_pair',n=n,batch=batch,candidate_id=cid,round=round_no,
                      candidate_guard_run_id=rows['candidate']['guard_run_id'],original_guard_run_id=rows['original']['guard_run_id'],
                      candidate_ms=results['candidate'],original_ms=results['original']))
        emit(dict(record_type='candidate_disposition',n=n,batch=batch,candidate_id=cid,request=request,status='confirmation_complete'))
    emit(dict(record_type='session_complete',status='complete',candidate_count=len(proof['cases'])))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--session-kind',choices=('first','second'),required=True)
    for name in ('rows','guard','incidents','budget'):
        p.add_argument('--screen-'+name,type=Path,required=True)
        p.add_argument('--first-'+name,type=Path)
    p.add_argument('--gpu',required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--expanded',type=Path,required=True)
    p.add_argument('--source-file',type=Path,action='append',required=True);p.add_argument('--artifact-file',type=Path,action='append',default=[])
    p.add_argument('--generated-provenance',type=Path,required=True)
    for name in ('output','budget-ledger','log','incidents','lock'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=float,required=True);p.add_argument('--timeout',type=int,default=300)
    p.add_argument('--idle-seconds',type=float,default=30)
    args=p.parse_args(argv)
    if (os.name!='posix' or not hasattr(os,'pidfd_open') or not args.gpu.startswith('GPU-') or ',' in args.gpu or
        not math.isfinite(args.seconds) or args.seconds<=args.idle_seconds or args.timeout<=0 or not 0<=args.idle_seconds<=3600):
        p.error('Linux pidfd, one GPU UUID, and positive valid budget/timeout required')
    args.original=args.original.resolve(strict=True);args.expanded=args.expanded.resolve(strict=True)
    if args.original==args.expanded:p.error('distinct binaries required')
    paths=[args.output,args.budget_ledger,args.log,args.incidents,args.lock]
    if len({x.resolve() for x in paths})!=len(paths):p.error('output paths must be distinct')
    for path in paths:path.parent.mkdir(parents=True,exist_ok=True)
    evidence_paths={name:getattr(args,'screen_'+name) for name in ('rows','guard','incidents','budget')}
    plan=json.loads(args.plan.read_text(encoding='utf8'))
    if plan['gpu']!=args.gpu:raise ValueError('plan GPU differs')
    proof=evidence.audit_screen(plan,evidence_paths,{'original':args.original,'expanded':args.expanded})
    root=Path(frontier.git(Path(__file__).parent,'rev-parse','--show-toplevel'))
    for source in (Path(__file__).resolve(),Path(evidence.__file__).resolve()):
        frontier.git(root,'ls-files','--error-unmatch',str(source))
    contract=dict(frontier.provenance(root,args.source_file,args.artifact_file),schema='frontier-confirmation/v1',gpu=args.gpu,
                  binaries={'original':frontier.sha(args.original),'expanded':frontier.sha(args.expanded)},
                  plan_sha256=frontier.sha(args.plan),source_sha256=proof['source_sha256'],session_kind=args.session_kind)
    contract['generated_provenance']=frontier.verify_generated(root,args.generated_provenance,contract['vkfft_commit'])
    screen_contract=proof['screen_contract']
    if (screen_contract['vkfft_commit']!=contract['vkfft_commit'] or
        screen_contract['generated_provenance']['sha256']!=contract['generated_provenance']['sha256']):
        raise ValueError('generated source or VkFFT commit differs from screening')
    first_paths={name:getattr(args,'first_'+name) for name in ('rows','guard','incidents','budget')}
    if args.session_kind=='first' and any(first_paths.values()):
        raise ValueError('first session must not supply earlier confirmation evidence')
    if args.session_kind=='second':
        if not all(first_paths.values()): raise ValueError('second session requires all first-session evidence')
        audit_path=Path(__file__).with_name('audit.py')
        frontier.git(root,'ls-files','--error-unmatch',str(audit_path))
        spec=importlib.util.spec_from_file_location('frontier_confirmation_audit',audit_path)
        audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        first_rows=frontier.read_log(first_paths['rows'])
        result=audit.audit_session(first_rows,frontier.read_log(first_paths['guard']),frontier.read_log(first_paths['incidents']),
                    frontier.read_log(first_paths['budget']),plan,contract['plan_sha256'],args.gpu,contract['execution_commit'],
                    {'original':args.original,'expanded':args.expanded},
                    {(c['n'],c['batch'],c['candidate_id']):c for c in proof['cases']},evidence.pair_order)
        if result.get('status')!='pass' or result.get('session_kind')!='first':
            raise ValueError('first session audit failed: '+str(result.get('errors')))
        contract['first_session']={'session_id':result['session_id'],'guard_id':result['guard_id'],
                                   'evidence_sha256':{name:frontier.sha(path) for name,path in first_paths.items()}}
    import device_lock
    with device_lock.DeviceLock(args.lock):
        if any(path.exists() for path in (args.output,args.budget_ledger,args.log,args.incidents)):
            raise ValueError('confirmation requires fresh output, budget and guard files; no selective retries')
        ledger=frontier.Ledger(args.budget_ledger,contract,args.seconds)
        args.output.touch(exist_ok=False);args.log.touch(exist_ok=False);args.incidents.touch(exist_ok=False)
        guard=guard_common.Guard(argparse.Namespace(gpu=args.gpu,log=args.log,incidents=args.incidents,idle_seconds=args.idle_seconds))
        old_popen,old_run=subprocess.Popen,suite.run_one
        keys=('CUDA_VISIBLE_DEVICES','VKFFT_STUDY_GUARD_ID','VKFFT_STUDY_GUARD_UUID','VKFFT_STUDY_GUARD_LOG')
        saved={k:os.environ.get(k) for k in keys}
        try:
            os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,VKFFT_STUDY_GUARD_ID=guard.guard_id,
                              VKFFT_STUDY_GUARD_UUID=args.gpu,VKFFT_STUDY_GUARD_LOG=str(args.log.resolve()))
            guard.emit('guard_start',script=str(Path(__file__).resolve()),idle_seconds=args.idle_seconds,
                       memory_limit_mib=guard_common.MEMORY_LIMIT_MIB,monitor_interval_seconds=1,
                       parent_pid=os.getpid(),own_signal_method='pidfd_with_rechecked_PPID_exe_starttime')
            guard.idle_check();subprocess.Popen=guard.popen(old_popen);suite.run_one=guard.wrapped(old_run);guard.start()
            try:run_session(args,ledger,contract,proof)
            except frontier.BudgetStop:
                frontier.append(args.output,dict(record_type='budget_exhausted',contract=contract))
            guard.sample('final');guard.check();guard.close();guard.check()
            if guard.thread is not None and guard.thread.is_alive():raise guard_common.GuardAbort('monitor failed to stop')
            guard.emit('guard_complete',success=True,clean=True,script_exit_code=0,sample_count=guard.sample_count)
        except BaseException as error:
            if not guard.abort.is_set():guard.trip([f'guard_or_target_exception:{type(error).__name__}:{error}'])
            guard.close();guard.emit('guard_abort',incident=True,success=False,clean=False,
                                      error=f'{type(error).__name__}: {error}',sample_count=guard.sample_count)
            raise
        finally:
            guard.close();subprocess.Popen,suite.run_one=old_popen,old_run
            for key,value in saved.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value


if __name__=='__main__':main()
