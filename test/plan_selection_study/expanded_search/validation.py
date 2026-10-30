"""Fail-closed checks of benchmark evidence for single-upload experiments."""
import math


def need(ok, reason):
    if not ok: raise ValueError(reason)


def request_env(req):
    return {'VKFFT_EXPANDED_RADIX': ','.join(map(str, req['radix'])),
            'VKFFT_EXPANDED_REGS': str(req['logical_registers']),
            'VKFFT_EXPANDED_BATCH': str(req['grouped_batch'])}


def signature(result):
    p = result['precompile_kernels'][0]
    return (result['structural_features'], result['compiled_diagnostics'], result['resolved_config'],
            p['canonical_descriptor'], p['source_sha256'],
            p['source_bytes'], p['compiler_input'], p['actual_compiler_options'])


def check(row, gpu, *, original=False, request=None, timing=False):
    need(row.get('guard_clean') is True and row.get('guard_gpu_uuid') == gpu, 'guard not clean or GPU mismatch')
    need(row.get('returncode') == 0 and row.get('status') == 'ok', 'process failed')
    r, w = row['result'], row['workload']
    need(r.get('status') == 'ok' and r['gpu']['uuid'] == gpu and r['gpu']['cuda_visible_devices'] == gpu, 'result or GPU mismatch')
    need((r['n'],r['batch'],r['seed'],r['pattern']) == (w['n'],w['batch'],w['seed'],w['pattern']), 'workload mismatch')
    need(r.get('requested_config') == {'makeForwardPlanOnly':1}, 'unexpected requested config')
    need(r.get('header_instrumentation') is (not original), 'incorrect binary instrumentation')
    need(r.get('validation_output_initialization') == '0xff_bytes_fp32_nan_before_single_and_final_timed_forward', 'output poisoning missing')
    need(r.get('single_forward_input_unchanged') is True and r.get('input_unchanged') is True, 'input changed')
    for field in ('single_forward_correctness','correctness'):
        c = r[field]
        expected_reference = ('independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity'
                              if w['pattern'] == 'random' else 'analytic_pattern_reference_with_independent_CPU_FFT_crosscheck')
        need(c.get('passed') is True and c.get('nonfinite_count') == 0 and c.get('failing_batches') == 0 and
             c.get('checked_elements') == w['n']*w['batch'] and c.get('reference') == expected_reference and
             c.get('criterion') == 'criterion_v2' and
             math.isclose(c['l2_scaled_error_threshold'],1e-5,rel_tol=1e-12) and
             math.isclose(c['linf_input_scaled_error_threshold'],1e-4,rel_tol=1e-12), 'numerical check failed')
    s = r['structural_features']; need(s['numAxisUploads'] == 1 and len(s['kernels']) == 1, 'outside single-upload scope')
    need(r.get('graph_repeats') == w['graph_repeats'] and r.get('graph_kernel_count') == w['graph_repeats'] and
         r.get('graph_event_record_count') == 2, 'graph mismatch')
    samples, raw = r['exec_ms_samples'], r['raw_interval_ms_samples']
    need(len(samples) == len(raw) == w['repeats'] and all(math.isfinite(v) and v > 0 for v in samples+raw), 'invalid timings')
    need(all(math.isclose(x,y/w['graph_repeats'],rel_tol=2e-6,abs_tol=1e-9) for x,y in zip(samples,raw)), 'timing normalization mismatch')
    if timing:
        need(w['repeats'] == 31 and w['warmup'] >= 5 and w['warmup_ms'] >= 1000 and min(raw) >= .5, 'timing protocol mismatch')
    if original:
        need(len(r['postinit_source_kernels']) == 1 and r['init_trace']['available'] is False, 'original source evidence missing')
    else:
        need(r['descriptor_validation']['all_match'] is True and len(r['precompile_kernels']) == 1 and
             r['precompile_kernels'][0]['compiler_options_match'] is True, 'precompile descriptor mismatch')
    if request:
        need(row['direct_environment'] == request_env(request), 'request environment mismatch')
        k=s['kernels'][0]; d=r['precompile_kernels'][0]['raw_descriptor']
        local=k['localSize']; swap=d['axisSwapped']
        need(k['stageRadix'] == request['radix'] and k['groupedBatch'] == request['grouped_batch'] and
             d['logical_registers_per_thread'] == request['logical_registers'] and d['registerBoost'] == 1 and
             swap in (0,1) and len(local)==3 and local[2]==1 and
             local[1 if swap else 0] == w['n']//request['logical_registers'] and
             local[0 if swap else 1] == request['grouped_batch'],
             'actual plan differs from request')
        limits=r['expanded_resource_validation']
        need(0 <= limits['dynamic_shared_bytes'] <= limits['function_dynamic_limit_bytes'] and
             0 <= limits['static_shared_bytes'] and
             limits['dynamic_shared_bytes']+limits['static_shared_bytes'] <= limits['device_shared_limit_bytes'] and
             limits['threads'] <= limits['function_thread_limit'] and limits['threads']==math.prod(local), 'compiled resource check failed')
    else:
        need(not row['direct_environment'], 'unexpected override on default')
    return r


def known_rejection(row, gpu):
    r = row.get('result', {})
    return (row.get('guard_clean') is True and row.get('guard_gpu_uuid') == gpu and row.get('returncode') == 1 and
            r.get('status') == 'error' and r.get('failure_phase') == 'initialize' and
            r.get('error') in ('initializeVkFFT failed with code 3002',
                               'expanded candidate exceeds compiled function or device resource limit'))


def check_describe(row, gpu, request):
    need(row.get('guard_clean') is True and row.get('guard_gpu_uuid') == gpu and row.get('returncode') == 0, 'describe guard or process failure')
    need(row.get('direct_environment') == request_env(request), 'describe environment mismatch')
    r=row['result'];w=row['workload']
    need(r.get('status')=='ok' and r.get('study_mode')=='describe' and r.get('describe_only') is True and
         r['gpu']['uuid']==gpu and (r['n'],r['batch'])==(w['n'],w['batch']), 'describe identity mismatch')
    counters=r['init_trace']['counters']
    need(all(counters[x]==0 for x in ('nvrtc_compile','module_load','driver_kernel_launch')) and
         r['descriptor_validation']['all_match'] is True and r['structural_features']['numAxisUploads']==1 and
         len(r['structural_features']['kernels'])==len(r['precompile_kernels'])==1, 'describe execution or structure mismatch')
    k=r['structural_features']['kernels'][0];d=r['precompile_kernels'][0]['raw_descriptor']
    local=k['localSize'];swap=d['axisSwapped']
    need(k['stageRadix']==request['radix'] and k['groupedBatch']==request['grouped_batch'] and
         d['logical_registers_per_thread']==request['logical_registers'] and d['registerBoost']==1 and
         swap in (0,1) and len(local)==3 and local[2]==1 and
         local[1 if swap else 0]==w['n']//request['logical_registers'] and
         local[0 if swap else 1]==request['grouped_batch'],
         'describe request or actual structure mismatch')
    return r
