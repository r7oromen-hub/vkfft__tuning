"""Guard-only, budgeted, paired screening of frozen expanded candidate requests."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_admission
import budget
import identity
import manifest
import run_suite as suite
import validation
sys.path.insert(1, str(HERE.parent/'original_compare'))
import original_compare_controller as original_comparison

PATTERNS = ('zero', 'impulse', 'constant', 'quarter-tone', 'random')
REQUIRED_ADMISSION = ('fully_disposed_preregistered_workloads', 'first_radix_changed',
                      'last_radix_changed', 'radix_multiset_changed', 'multiple_register_values',
                      'both_axis_orientations', 'at_least_two_lengths', 'small_and_large_batch')


class BudgetStop(Exception): pass


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_admission_gate(gate, evidence_hashes, binary_hashes):
    if (gate.get('schema') != 'expanded-admission-gate/v1' or gate.get('status') != 'pass' or
        gate.get('integrity_status') != 'pass' or
        any(gate.get('required_dimensions', {}).get(name) is not True for name in REQUIRED_ADMISSION) or
        gate.get('evidence_sha256') != evidence_hashes or gate.get('binaries_sha256') != binary_hashes):
        raise ValueError('admission gate missing, incomplete, unapproved, or bound to different evidence')


def seeded_order(seed, *parts):
    key = json.dumps([seed,*parts], separators=(',',':')).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(),'big'))
    order = ['original','candidate']; rng.shuffle(order)
    return order


def workload_order(protocol):
    return [(n,b) for b in protocol['batches'] for n in protocol['lengths']]


def candidate_rounds(contexts, ordered_workloads):
    maximum = max((len(contexts[k]['selected']) for k in ordered_workloads if k in contexts), default=0)
    for index in range(maximum):
        for key in ordered_workloads:
            if key in contexts and index < len(contexts[key]['selected']):
                yield key, contexts[key]['selected'][index]


def source_signature(result, *, original):
    p = (result['postinit_source_kernels'] if original else result['precompile_kernels'])[0]
    return {'structural_features':result['structural_features'], 'compiled_diagnostics':result['compiled_diagnostics'],
            'resolved_config':result['resolved_config'], 'canonical_descriptor':p['canonical_descriptor'],
            'source_sha256':p['source_sha256'], 'source_bytes':p['source_bytes'],
            'compiler_input':p['expected_compiler_input'] if original else p['compiler_input'],
            'actual_compiler_options':p['actual_compiler_options']}


def numerical_failure(row, gpu, request):
    """Recognize only explicit, complete numerical failures; other errors stop the run."""
    try:
        r,w=row['result'],row['workload']
        if (row.get('guard_clean') is not True or row.get('guard_gpu_uuid') != gpu or
            row.get('returncode') != 1 or row.get('status') != 'failed' or
            r.get('gpu',{}).get('uuid') != gpu or r['gpu'].get('cuda_visible_devices') != gpu or
            row.get('direct_environment') != validation.request_env(request) or
            (r['n'],r['batch'],r['seed'],r['pattern']) != (w['n'],w['batch'],w['seed'],w['pattern']) or
            r.get('header_instrumentation') is not True or r.get('requested_config') != {'makeForwardPlanOnly':1} or
            r.get('validation_output_initialization') != '0xff_bytes_fp32_nan_before_single_and_final_timed_forward' or
            r.get('single_forward_input_unchanged') is not True or
            r.get('descriptor_validation',{}).get('all_match') is not True): return False
        plan=r['structural_features']; k=plan['kernels'][0]; p=r['precompile_kernels'][0];d=p['raw_descriptor']
        local=k['localSize'];swap=d['axisSwapped'];lim=r['expanded_resource_validation']
        if (plan['numAxisUploads'] != 1 or len(plan['kernels']) != 1 or len(r['precompile_kernels']) != 1 or
            p['compiler_options_match'] is not True or k['stageRadix'] != request['radix'] or
            k['groupedBatch'] != request['grouped_batch'] or d['logical_registers_per_thread'] != request['logical_registers'] or
            d['registerBoost'] != 1 or swap not in (0,1) or len(local) != 3 or local[2] != 1 or
            local[1 if swap else 0] != w['n']//request['logical_registers'] or local[0 if swap else 1] != request['grouped_batch'] or
            not 0 <= lim['dynamic_shared_bytes'] <= lim['function_dynamic_limit_bytes'] or
            not 0 <= lim['static_shared_bytes'] or lim['dynamic_shared_bytes']+lim['static_shared_bytes'] > lim['device_shared_limit_bytes'] or
            lim['threads'] > lim['function_thread_limit'] or lim['threads'] != math.prod(local)): return False
        if r.get('status') == 'correctness_failed':
            if r.get('input_unchanged') is not True or r['single_forward_correctness']['passed'] is not True: return False
            c=r['correctness']
        elif (r.get('status')=='error' and r.get('failure_phase')=='single_forward_validation' and
              r.get('error')=='Single out-of-place forward failed immutable-input or numerical validation'):
            c=r['single_forward_correctness']
        else: return False
        expected_reference = ('independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity'
                              if w['pattern']=='random' else 'analytic_pattern_reference_with_independent_CPU_FFT_crosscheck')
        return (c.get('passed') is False and c.get('criterion') == 'criterion_v2' and c.get('reference') == expected_reference and
                c.get('checked_elements') == w['n']*w['batch'] and
                math.isclose(c['l2_scaled_error_threshold'],1e-5,rel_tol=1e-12) and
                math.isclose(c['linf_input_scaled_error_threshold'],1e-4,rel_tol=1e-12) and
                isinstance(c.get('nonfinite_count'),int) and isinstance(c.get('failing_batches'),int))
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError): return False


def run(a, *, budget_obj=None):
    if os.environ.get('VKFFT_STUDY_GUARD_UUID') != a.gpu or not os.environ.get('VKFFT_STUDY_GUARD_ID'):
        raise ValueError('guard required')
    b = budget_obj or budget.Budget(a.budget_dir)
    if b.state['phase'] != 'search' or b.state['stopped'] or b.state.get('inflight'):
        raise ValueError('budget is not in an available search phase')
    protocol = json.loads(a.study_protocol.read_text(encoding='utf8'))
    if (protocol['schema'] != 'expanded-search-protocol/v1' or protocol['gpu_uuid'] != a.gpu or
        protocol['screen_pair_rounds'] != 3 or protocol['max_nondefault_selected_requests_per_workload'] != 64 or
        protocol['validation_seeds'] != [42,53,71] or protocol['lengths'] != [128,256,512,1024,2048,4096,8192] or
        protocol['batches'] != [1,16,256,4096] or not protocol['failed_selected_requests_are_not_replaced']):
        raise ValueError('unexpected or modified registered study protocol')
    admission_protocol=json.loads(a.admission_protocol.read_text(encoding='utf8'))
    rows=audit_admission.load(a.admission_rows); guard=audit_admission.load(a.admission_guard)
    incidents=audit_admission.load(a.admission_incidents)
    audited=audit_admission.audit(rows,guard,incidents,admission_protocol,sha(a.admission_protocol))
    if audited['status'] != 'pass': raise ValueError('admission audit did not pass')
    headers=[row for row in rows if row.get('record_type')=='admission_protocol']
    expanded=str(Path(a.expanded_exe).resolve(strict=True)); original=str(Path(a.original_exe).resolve(strict=True))
    hashes={'expanded':sha(expanded),'original':sha(original)}
    if expanded == original or len(headers)!=1 or headers[0]['sha256'] != hashes:
        raise ValueError('search executable differs from audited admission')
    evidence_hashes={'rows':sha(a.admission_rows),'guard':sha(a.admission_guard),
                     'incidents':sha(a.admission_incidents),'protocol':sha(a.admission_protocol)}
    gate=json.loads(a.admission_gate.read_text(encoding='utf8'))
    check_admission_gate(gate,evidence_hashes,hashes)
    ordered=workload_order(protocol);session=uuid.uuid4().hex
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.manifest_dir.mkdir(parents=True,exist_ok=True)
    contexts={}; disposition={}; terminal='error'
    with a.output.open('x',encoding='utf8') as out:
        def emit(row):
            out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
        emit(dict(record_type='search_protocol',session_id=session,protocol=protocol,protocol_sha256=sha(a.study_protocol),
                  admission_rows_sha256=sha(a.admission_rows),admission_guard_sha256=sha(a.admission_guard),
                  admission_incidents_sha256=sha(a.admission_incidents),admission_protocol_sha256=sha(a.admission_protocol),
                  admission_gate_sha256=sha(a.admission_gate),admission_gate=gate,
                  admission_audit=audited,binaries={'expanded':expanded,'original':original},sha256=hashes,
                  ordered_workloads=[{'n':n,'batch':batch} for n,batch in ordered]))

        def gate(kind, count, **info):
            work_id=uuid.uuid4().hex
            if not b.may_start(kind,expected_upper_seconds=count*(a.timeout+30),work_id=work_id,**info):
                raise BudgetStop()
            return work_id

        def execute(n,batch,name,request,phase,round_no,order,pattern,seed,R,*,timing=False,mode='execute'):
            binary='original' if name=='original' else 'expanded'
            a.exe=original if binary=='original' else expanded;a.exe_sha256=hashes[binary];a.session_id=session
            w=dict(n=n,batch=batch,seed=seed,config={},pattern=pattern,graph_repeats=R,
                   warmup=5 if timing else 0,warmup_ms=1000 if timing else 0,repeats=31 if timing else 1,study_mode=mode)
            env=validation.request_env(request) if request else {}
            row=suite.run_one(a,w,(name,env),phase,round_no,order);emit(row)
            return row

        def one(n,batch,name,request,phase,*,pattern='random',seed=42,R=64,mode='execute'):
            work_id=gate('search_process',1,n=n,batch=batch,candidate_id=name,phase=phase,mode=mode)
            try:
                row=execute(n,batch,name,request,phase,0,0,pattern,seed,R,mode=mode)
            except BaseException as error:
                b.finish_work('exception',work_id=work_id,error=f'{type(error).__name__}: {error}')
                raise
            b.finish_work(row['status'],work_id=work_id,guard_run_id=row.get('guard_run_id'))
            return row

        def pair(n,batch,cid,request,phase,round_no,pattern,seed,R,*,timing=False):
            order=seeded_order(protocol['selection_seed'],n,batch,cid,phase,round_no,pattern,seed)
            work_id=gate('search_pair',2,n=n,batch=batch,candidate_id=cid,phase=phase,round=round_no,pattern=pattern,seed=seed)
            result={}
            try:
                for ix,name in enumerate(order):
                    variant='original' if name=='original' else cid
                    result[name]=execute(n,batch,variant,request if name=='candidate' else None,
                                         phase,round_no,ix,pattern,seed,R,timing=timing)
            except BaseException as error:
                b.finish_work('exception',work_id=work_id,error=f'{type(error).__name__}: {error}')
                raise
            b.finish_work('completed',work_id=work_id,guard_run_ids=[result[x].get('guard_run_id') for x in order])
            return order,result

        try:
            # Freeze control identity and selected requests before any candidate timing.
            for n,batch in ordered:
                disposition[(n,batch)]='identity_pending_or_incomplete'
                original_result=validation.check(one(n,batch,'original',None,'identity_control'),a.gpu,original=True)
                default=validation.check(one(n,batch,'expanded_default',None,'identity_control'),a.gpu)
                original_comparison.compare(default,original_result)
                req,reason=identity.representable(default)
                if req is None:
                    disposition[(n,batch)]='unsupported_identity'
                    emit(dict(record_type='workload_status',n=n,batch=batch,status='unsupported_identity',reason=reason));continue
                replay=validation.check(one(n,batch,'identity',req,'identity_control'),a.gpu,request=req)
                if validation.signature(default) != validation.signature(replay): raise ValueError(f'identity mismatch n={n}, batch={batch}')
                path=a.manifest_dir/f'n{n}_b{batch}.jsonl'
                manifest_work=gate('manifest_write',1,n=n,batch=batch,path=str(path))
                try:
                    selected, complete=manifest.make(path,n,batch,req,seed=protocol['selection_seed'],limit=64)
                except BaseException as error:
                    b.finish_work('exception',work_id=manifest_work,error=f'{type(error).__name__}: {error}',partial_path=str(path))
                    raise
                b.finish_work('completed',work_id=manifest_work,path=str(path),sha256=sha(path))
                contexts[(n,batch)]={'selected':selected,'identity':req,'manifest':str(path),'R':None,
                                     'original_signature':source_signature(original_result,original=True)}
                disposition[(n,batch)]='identity_passed'
                emit(dict(record_type='workload_status',n=n,batch=batch,status='identity_passed',identity=req,
                          manifest=str(path),manifest_sha256=sha(path),manifest_complete=complete))
            # Calibrate the original denominator independently for each workload.
            for n,batch in ordered:
                if (n,batch) not in contexts: continue
                R=64
                while True:
                    row=one(n,batch,'original',None,'calibration',R=R)
                    r=validation.check(row,a.gpu,original=True)
                    if source_signature(r,original=True) != contexts[(n,batch)]['original_signature']:
                        raise ValueError('calibration original plan/source/resources changed')
                    if min(r['raw_interval_ms_samples']) >= 2.0: break
                    if R >= 2048: raise ValueError(f'calibration failed n={n}, batch={batch}')
                    R=min(2048,R*2)
                contexts[(n,batch)]['R']=R
                emit(dict(record_type='calibration_complete',n=n,batch=batch,graph_repeats=R,
                          guard_run_id=row['guard_run_id'],raw_interval_ms=min(r['raw_interval_ms_samples'])))
            for (n,batch), selected in candidate_rounds(contexts,ordered):
                request=selected['request'];cid=selected['candidate_id'];R=contexts[(n,batch)]['R']
                emit(dict(record_type='candidate_selected',n=n,batch=batch,candidate_id=cid,request=request,
                          selection_index=selected['selection_index'],planned_fft_threads=selected['planned_fft_threads']))
                row=one(n,batch,cid,request,'candidate_describe',R=R,mode='describe')
                emit(dict(record_type='candidate_start',n=n,batch=batch,candidate_id=cid,guard_run_id=row.get('guard_run_id')))
                if validation.known_rejection(row,a.gpu):
                    validation.check(one(n,batch,'original',None,'post_rejection_control',R=R),a.gpu,original=True)
                    emit(dict(record_type='candidate_disposition',n=n,batch=batch,candidate_id=cid,status='scheduler_or_resource_rejected',reason=row['result']['error']));continue
                described=validation.check_describe(row,a.gpu,request)
                emit(dict(record_type='candidate_layout',n=n,batch=batch,candidate_id=cid,
                          local_size=described['structural_features']['kernels'][0]['localSize'],
                          raw_axis_swapped=described['precompile_kernels'][0]['raw_descriptor']['axisSwapped'],
                          source_sha256=described['precompile_kernels'][0]['source_sha256']))
                reason=None
                for pattern in PATTERNS:
                    for seed in protocol['validation_seeds']:
                        order,pr=pair(n,batch,cid,request,'candidate_validation',0,pattern,seed,R)
                        ov=validation.check(pr['original'],a.gpu,original=True)
                        if source_signature(ov,original=True) != contexts[(n,batch)]['original_signature']:
                            raise ValueError('validation original plan/source/resources changed')
                        if validation.known_rejection(pr['candidate'],a.gpu): reason='scheduler_or_resource_rejected'
                        elif numerical_failure(pr['candidate'],a.gpu,request): reason='numerically_invalid'
                        else:
                            cv=validation.check(pr['candidate'],a.gpu,request=request)
                            dp=described['precompile_kernels'][0];cp=cv['precompile_kernels'][0]
                            if any(cp[field] != dp[field] for field in ('canonical_descriptor','source_sha256','source_bytes','compiler_input')):
                                raise ValueError('describe and validation codegen differ')
                        if reason: break
                    if reason: break
                if reason:
                    validation.check(one(n,batch,'original',None,'post_rejection_control',R=R),a.gpu,original=True)
                    emit(dict(record_type='candidate_disposition',n=n,batch=batch,candidate_id=cid,status=reason));continue
                ratios=[]; original_sig=None; candidate_sig=None
                for round_no in range(protocol['screen_pair_rounds']):
                    order,pr=pair(n,batch,cid,request,'screen_timing',round_no,'random',42,R,timing=True)
                    orig=validation.check(pr['original'],a.gpu,original=True,timing=True)
                    cand=validation.check(pr['candidate'],a.gpu,request=request,timing=True)
                    if any(r['warmup_actual'] < 5 or r['warmup_wall_ms'] < 1000 for r in (orig,cand)):
                        raise ValueError('actual warmup shorter than screening protocol')
                    osig=source_signature(orig,original=True);csig=source_signature(cand,original=False)
                    if osig != contexts[(n,batch)]['original_signature']:
                        raise ValueError('timing original plan/source/resources changed')
                    if original_sig is not None and (original_sig != osig or candidate_sig != csig):
                        raise ValueError('screen plan/source/resources changed across rounds')
                    original_sig,candidate_sig=osig,csig
                    dp=described['precompile_kernels'][0]
                    if any(csig[field] != dp[field] for field in ('canonical_descriptor','source_sha256','source_bytes','compiler_input')):
                        raise ValueError('describe and execute codegen differ')
                    om=statistics.median(orig['exec_ms_samples']);cm=statistics.median(cand['exec_ms_samples']);ratio=cm/om
                    ratios.append(ratio)
                    emit(dict(record_type='screen_pair',session_id=session,n=n,batch=batch,candidate_id=cid,round=round_no,
                              order=order,original_guard_run_id=pr['original']['guard_run_id'],candidate_guard_run_id=pr['candidate']['guard_run_id'],
                              original_median_ms=om,candidate_median_ms=cm,candidate_over_original=ratio))
                emit(dict(record_type='candidate_disposition',n=n,batch=batch,candidate_id=cid,status='screen_complete',
                          request=request,graph_repeats=R,pair_ratios=ratios,median_ratio=statistics.median(ratios),
                          original_signature=original_sig,candidate_signature=candidate_sig))
            terminal='completed_selected'
        except BudgetStop:
            terminal='budget_exhausted'
        except BaseException as error:
            emit(dict(record_type='search_complete',status='error',error=f'{type(error).__name__}: {error}',
                      workload_statuses=[{'n':n,'batch':batch,'status':disposition.get((n,batch),'not_started')} for n,batch in ordered]))
            raise
        emit(dict(record_type='search_complete',status=terminal,
                  workload_statuses=[{'n':n,'batch':batch,'status':disposition.get((n,batch),'not_started')} for n,batch in ordered]))
    return terminal


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True);p.add_argument('--expanded-exe',required=True);p.add_argument('--original-exe',required=True)
    p.add_argument('--budget-dir',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--manifest-dir',required=True,type=Path)
    p.add_argument('--study-protocol',required=True,type=Path);p.add_argument('--admission-protocol',required=True,type=Path)
    p.add_argument('--admission-rows',required=True,type=Path);p.add_argument('--admission-guard',required=True,type=Path);p.add_argument('--admission-incidents',required=True,type=Path)
    p.add_argument('--admission-gate',required=True,type=Path)
    p.add_argument('--timeout',type=int,default=300)
    a=p.parse_args(argv)
    if not 1 <= a.timeout <= 3600: p.error('timeout must be 1..3600 seconds')
    return run(a)


if __name__=='__main__':
    raise SystemExit('use search_gpu_guard; unguarded GPU execution is disabled')
