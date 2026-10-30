"""Guard-only, budgeted multi-address rotation of independently confirmed winners."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import uuid

import budget
import rotation_evidence
import run_suite as suite
import search_controller
import validation


class BudgetStop(Exception): pass


def check(row,gpu,case,rotation,*,original,timing):
    r,w=row['result'],row['workload'];n,batch=case['n'],case['batch']
    validation.need(row.get('guard_clean') is True and row.get('guard_gpu_uuid')==gpu and
                    row.get('status')=='ok' and row.get('returncode')==0 and r.get('status')=='ok', 'rotation guard or process failed')
    validation.need(r['gpu']['uuid']==gpu and r['gpu']['cuda_visible_devices']==gpu and
                    (r['n'],r['batch'],r['seed'],r['pattern'])==(n,batch,42,'random'), 'rotation workload/GPU mismatch')
    validation.need(r['header_instrumentation'] is (not original) and r['requested_config']=={'makeForwardPlanOnly':1} and
                    row['direct_environment']==({} if original else validation.request_env(case['request'])), 'rotation binary/request mismatch')
    validation.need(search_controller.source_signature(r,original=original)==case['original_signature' if original else 'candidate_signature'],
                    'rotation plan/source/resource drift from confirmed hot candidate')
    slots=rotation['slots'];R=rotation['graph_repeats'];bytes_per_buffer=n*batch*8
    validation.need(r.get('rotation_mode')=='multi_address_graph_random_unique_scaled_inputs' and
                    r.get('rotation_pool_slots')==slots and r.get('rotation_pool_total_bytes')==2*slots*bytes_per_buffer and
                    r.get('rotation_input_pool_bytes')==slots*bytes_per_buffer and
                    r.get('rotation_output_pool_bytes')==slots*bytes_per_buffer and
                    r.get('rotation_device_l2_bytes')==rotation['l2_bytes'] and slots*bytes_per_buffer>rotation['l2_bytes'] and
                    r.get('rotation_graph_kernel_arguments_and_order_verified') is True and
                    r.get('rotation_distinct_input_representatives_verified') is True and
                    r.get('rotation_output_initialization')=='0xff_bytes_fp32_nan_before_initial_and_separate_post_timing_validation_graph',
                    'rotation pool or graph address evidence invalid')
    validation.need(r.get('graph_repeats')==R and r.get('graph_kernel_count')==R and r.get('graph_event_record_count')==2 and
                    r.get('graph_node_count')==R+2 and R%slots==0 and
                    r.get('timing_scope')=='multi_address_graph_forward_same_stream_no_transfers_or_validation_inside_event_interval',
                    'rotation graph protocol mismatch')
    for stage in ('rotation_initial_validation','rotation_final_validation'):
        report=r[stage]; entries=report['slots']
        validation.need(report.get('passed') is True and report.get('slots_checked')==slots and
                        report.get('elements_per_slot')==n*batch and len(entries)==slots and
                        [x['slot'] for x in entries]==list(range(slots)), 'rotation full-pool evidence incomplete')
        for entry in entries:
            c=entry['correctness']
            validation.need(entry.get('input_unchanged') is True and c.get('passed') is True and
                            c.get('criterion')=='criterion_v2' and c.get('checked_elements')==n*batch and
                            c.get('reference')=='independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity' and
                            c.get('nonfinite_count')==0 and c.get('failing_batches')==0 and
                            math.isclose(c['l2_scaled_error_threshold'],1e-5,rel_tol=1e-12) and
                            math.isclose(c['linf_input_scaled_error_threshold'],1e-4,rel_tol=1e-12),
                            'rotation slot input or numerical evidence failed')
    for name in ('rotation_first_graph_interval_ms','rotation_final_validation_graph_interval_ms'):
        validation.need(isinstance(r.get(name),(float,int)) and math.isfinite(r[name]) and r[name]>0, 'rotation validation interval invalid')
    if not original:
        k=r['structural_features']['kernels'][0];p=r['precompile_kernels'][0];d=p['raw_descriptor'];q=case['request']
        swap=d['axisSwapped'];local=k['localSize'];lim=r['expanded_resource_validation']
        validation.need(r['descriptor_validation']['all_match'] is True and p['compiler_options_match'] is True and
                        k['stageRadix']==q['radix'] and k['groupedBatch']==q['grouped_batch'] and
                        d['logical_registers_per_thread']==q['logical_registers'] and d['registerBoost']==1 and
                        swap in (0,1) and local[2]==1 and local[1 if swap else 0]==n//q['logical_registers'] and
                        local[0 if swap else 1]==q['grouped_batch'] and
                        0<=lim['dynamic_shared_bytes']<=lim['function_dynamic_limit_bytes'] and 0<=lim['static_shared_bytes'] and
                        lim['dynamic_shared_bytes']+lim['static_shared_bytes']<=lim['device_shared_limit_bytes'] and
                        lim['threads']==math.prod(local) and lim['threads']<=lim['function_thread_limit'],
                        'rotation candidate plan or resource validation failed')
    else:
        validation.need(r['init_trace']['available'] is False and len(r['postinit_source_kernels'])==1,
                        'original rotation instrumentation claim invalid')
    samples=r['exec_ms_samples'];raw=r['raw_interval_ms_samples'];expected=31 if timing else 1
    validation.need(len(samples)==len(raw)==expected and r['warmup_actual']>=5 and r['warmup_wall_ms']>=1000 and
                    all(math.isfinite(x) and x>0 for x in samples+raw) and
                    all(math.isclose(x,y/R,rel_tol=2e-6,abs_tol=1e-9) for x,y in zip(samples,raw)) and
                    (not timing or min(raw)>=.5), 'rotation timing or warmup invalid')
    return r


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True);p.add_argument('--budget-dir',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--rotation-original-exe',required=True,type=Path);p.add_argument('--rotation-expanded-exe',required=True,type=Path)
    p.add_argument('--rotation-original-sha256',required=True);p.add_argument('--rotation-expanded-sha256',required=True)
    for name in ('plan','gate','protocol','first-gate','first-audit','second-audit','combined-audit',
                 'first-rows','first-guard','first-incidents','first-budget',
                 'second-rows','second-guard','second-incidents','second-budget'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--timeout',type=int,default=600)
    a=p.parse_args(argv)
    if not 1<=a.timeout<=3600:p.error('timeout must be 1..3600')
    if os.environ.get('VKFFT_STUDY_GUARD_UUID')!=a.gpu or not os.environ.get('VKFFT_STUDY_GUARD_ID'):p.error('guard required')
    b=budget.Budget(a.budget_dir)
    if b.state['phase']!='extension' or b.state['stopped'] or b.state.get('inflight'):p.error('extension budget unavailable')
    work_id=uuid.uuid4().hex
    if not b.may_start('rotation_gate_verification',expected_upper_seconds=600,work_id=work_id):p.error('insufficient extension budget for gate audit')
    try: plan,gate,cases,l2=rotation_evidence.verify(a)
    except BaseException as error:
        b.finish_work('exception',work_id=work_id,error=f'{type(error).__name__}: {error}');raise
    b.finish_work('completed',work_id=work_id,gate_sha256=rotation_evidence.sha(a.gate))
    if not cases:p.error('no confirmed winners; rotation is not authorized')
    original=str(a.rotation_original_exe.resolve(strict=True));expanded=str(a.rotation_expanded_exe.resolve(strict=True))
    hashes={'original':rotation_evidence.sha(original),'expanded':rotation_evidence.sha(expanded)}
    if original==expanded or hashes!={'original':a.rotation_original_sha256,'expanded':a.rotation_expanded_sha256}:
        p.error('rotation binary hash mismatch')
    session=uuid.uuid4().hex;a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as out:
        def emit(row):
            out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');out.flush();os.fsync(out.fileno())
        emit(dict(record_type='rotation_protocol',session_id=session,gpu=a.gpu,rounds=10,plan_sha256=rotation_evidence.sha(a.plan),
                  two_session_gate_sha256=rotation_evidence.sha(a.gate),hot_binary_sha256=plan['sha256'],
                  binaries={'original':original,'expanded':expanded},sha256=hashes,confirmed=cases,
                  timing_scope='separate multi-address working-set experiment; not a cold-cache or end-to-end claim'))
        def invoke(case,rotation,name,phase,round_no,order,timing):
            a.exe=original if name=='original' else expanded;a.exe_sha256=hashes[name if name=='original' else 'expanded'];a.session_id=session
            w=dict(n=case['n'],batch=case['batch'],seed=42,config={},pattern='random',graph_repeats=rotation['graph_repeats'],
                   warmup=5,warmup_ms=1000,repeats=31 if timing else 1,study_mode='execute')
            row=suite.run_one(a,w,('original' if name=='original' else case['candidate_id'],
                                   {} if name=='original' else validation.request_env(case['request'])),phase,round_no,order)
            emit(row);return row
        def allowed(kind,count,**detail):
            work_id=uuid.uuid4().hex
            if not b.may_start(kind,expected_upper_seconds=count*(a.timeout+30),work_id=work_id,**detail):raise BudgetStop()
            return work_id
        terminal='completed';disposed=set();active=None
        def disposition(key,status,**details):
            emit(dict(record_type='rotation_disposition',n=key[0],batch=key[1],candidate_id=key[2],status=status,**details))
            disposed.add(key)
        try:
            for case in cases:
                key=(case['n'],case['batch'],case['candidate_id']);active=key;gpu_record,l2_bytes=l2[key]
                if gpu_record!=a.gpu:raise ValueError('confirmed GPU differs from rotation target')
                rotation=rotation_evidence.pool(case['n']*case['batch']*8,l2_bytes,case['graph_repeats'])
                if rotation is None:
                    disposition(key,'uncovered_pool_or_graph_limit');continue
                rotation['l2_bytes']=l2_bytes
                while True:
                    work_id=allowed('rotation_calibration',1,n=key[0],batch=key[1],candidate_id=key[2],graph_repeats=rotation['graph_repeats'])
                    try: row=invoke(case,rotation,'original','rotation_calibration',0,0,False)
                    except BaseException as error:
                        b.finish_work('exception',work_id=work_id,error=f'{type(error).__name__}: {error}');raise
                    b.finish_work(row['status'],work_id=work_id,guard_run_id=row.get('guard_run_id'))
                    if row.get('returncode')!=0:
                        r=row.get('result',{});error=r.get('error','')
                        if (row.get('guard_clean') is True and row.get('guard_gpu_uuid')==a.gpu and
                            r.get('status')=='error' and error in ('rotation_unsupported_pool_or_graph_size','rotation_unsupported_device_memory',
                              'rotation graph kernel argument inspection unavailable')):
                            disposition(key,'uncovered',reason=error);break
                        disposition(key,'uncovered_calibration_failure',reason=error)
                        raise ValueError('unexpected rotation calibration failure')
                    try: result=check(row,a.gpu,case,rotation,original=True,timing=False)
                    except BaseException as error:
                        disposition(key,'uncovered_calibration_validation_failure',reason=f'{type(error).__name__}: {error}');raise
                    if min(result['raw_interval_ms_samples'])>=2.0:
                        emit(dict(record_type='rotation_calibration_complete',n=key[0],batch=key[1],candidate_id=key[2],
                                  rotation=rotation,guard_run_id=row['guard_run_id'],raw_interval_ms=min(result['raw_interval_ms_samples'])));break
                    if rotation['graph_repeats']>=2048:
                        disposition(key,'uncovered_calibration_interval');break
                    rotation['graph_repeats']*=2
                else: raise AssertionError('unreachable')
                if row.get('returncode')!=0 or min(result['raw_interval_ms_samples'])<2.0:continue
                ratios=[]
                for round_no in range(10):
                    work_id=allowed('rotation_pair',2,n=key[0],batch=key[1],candidate_id=key[2],round=round_no)
                    order=search_controller.seeded_order(plan['selection_seed'],*key,'rotation',round_no,42);pair={}
                    try:
                        for ix,name in enumerate(order):pair[name]=invoke(case,rotation,name,'rotation_timing',round_no,ix,True)
                    except BaseException as error:
                        b.finish_work('exception',work_id=work_id,error=f'{type(error).__name__}: {error}');raise
                    b.finish_work('completed',work_id=work_id,guard_run_ids=[pair[x].get('guard_run_id') for x in order])
                    try:
                        original_result=check(pair['original'],a.gpu,case,rotation,original=True,timing=True)
                        candidate_result=check(pair['candidate'],a.gpu,case,rotation,original=False,timing=True)
                    except BaseException as error:
                        disposition(key,'uncovered_pair_failure',reason=f'{type(error).__name__}: {error}',round=round_no);raise
                    om=statistics.median(original_result['exec_ms_samples']);cm=statistics.median(candidate_result['exec_ms_samples']);ratio=cm/om
                    ratios.append(ratio)
                    emit(dict(record_type='rotation_pair',session_id=session,n=key[0],batch=key[1],candidate_id=key[2],round=round_no,
                              order=order,rotation=rotation,original_guard_run_id=pair['original']['guard_run_id'],
                              candidate_guard_run_id=pair['candidate']['guard_run_id'],original_median_ms=om,candidate_median_ms=cm,candidate_over_original=ratio))
                disposition(key,'complete',rotation=rotation,paired_ratios=ratios,median_ratio=statistics.median(ratios))
        except BudgetStop: terminal='budget_exhausted'
        except BaseException as error:
            emit(dict(record_type='rotation_complete',session_id=session,status='error',error=f'{type(error).__name__}: {error}',
                      incomplete_or_not_started=[{'n':c['n'],'batch':c['batch'],'candidate_id':c['candidate_id']} for c in cases
                                                 if (c['n'],c['batch'],c['candidate_id']) not in disposed]));raise
        emit(dict(record_type='rotation_complete',session_id=session,status=terminal,
                  incomplete_or_not_started=[{'n':c['n'],'batch':c['batch'],'candidate_id':c['candidate_id']} for c in cases
                                             if (c['n'],c['batch'],c['candidate_id']) not in disposed]))


if __name__=='__main__':raise SystemExit('use rotation_gpu_guard; unguarded GPU execution is disabled')
