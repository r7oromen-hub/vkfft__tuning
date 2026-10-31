"""Validate guarded single-process evidence for the proposed nonuniform planner.

This checks records, not the guard ledger, a 15-case matrix, binary provenance or
independent timing confirmation. A controller and an evidence audit must bind those.
"""
import math

import candidate


PATTERNS = ('zero', 'impulse', 'constant', 'quarter-tone', 'random')
TRACE_FIELDS = {
    'VKFFT_NONUNIFORM_VERSION': 'nonuniform_request_version',
    'VKFFT_NONUNIFORM_N': 'nonuniform_requested_n',
    'VKFFT_NONUNIFORM_WORKLOAD_BATCH': 'nonuniform_requested_workload_batch',
    'VKFFT_NONUNIFORM_RADIX': 'nonuniform_requested_radix',
    'VKFFT_NONUNIFORM_TABLE': 'nonuniform_requested_register_table',
    'VKFFT_NONUNIFORM_PROFILE': 'nonuniform_requested_profile',
    'VKFFT_NONUNIFORM_GROUP': 'nonuniform_requested_group',
}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    return (type(value) is str and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value))


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def nonnegative_int(value):
    return type(value) is int and value >= 0


def _one(items, description):
    need(type(items) is list and len(items) == 1 and type(items[0]) is dict, description)
    return items[0]


def _common(row, gpu, original, request, describe):
    need(type(gpu) is str and bool(gpu), 'missing GPU identity')
    need(row.get('guard_clean') is True and row.get('guard_gpu_uuid') == gpu and
         row.get('returncode') == 0 and row.get('status') == 'ok', 'guard or process failure')
    r, w = row['result'], row['workload']
    need(r.get('status') == 'ok' and r['gpu']['uuid'] == gpu and
         r['gpu']['cuda_visible_devices'] == gpu, 'result or GPU mismatch')
    need((r['n'], r['batch'], r['seed'], r['pattern']) ==
         (w['n'], w['batch'], w['seed'], w['pattern']), 'workload mismatch')
    need(type(w['n']) is int and type(w['batch']) is int and w['n'] > 0 and w['batch'] > 0 and
         type(w['seed']) is int and w['pattern'] in PATTERNS, 'invalid workload')
    need(r.get('requested_config') == {'makeForwardPlanOnly': 1} and
         r.get('header_instrumentation') is (not original), 'requested config or binary instrumentation mismatch')
    need(r.get('study_mode') == ('describe' if describe else 'execute') and
         r.get('describe_only') is describe, 'study mode mismatch')
    need(r.get('backend') == 'CUDA_NVRTC' and r.get('precision') == 'complex_fp32' and
         r.get('placement') == 'outofplace' and r.get('forward_direction') == -1, 'FFT scope mismatch')
    shape = r['structural_features']
    need(shape.get('numAxisUploads') == 1, 'requires one upload')
    kernel = _one(shape.get('kernels'), 'requires one kernel')
    if request is None:
        need(row.get('direct_environment') == {}, 'missing or unexpected override on default')
        need(not any(field in r for field in TRACE_FIELDS.values()), 'default contains request telemetry')
    else:
        need(not original, 'original binary cannot execute a candidate request')
        candidate.validate_request(request)
        need(request['workload'] == {'n': w['n'], 'batch': w['batch']}, 'request workload mismatch')
        env = candidate.request_environment(request)
        need(row.get('direct_environment') == env, 'candidate environment mismatch')
        need(all(r.get(field) == env[key] for key, field in TRACE_FIELDS.items()),
             'request telemetry mismatch')
    return r, w, kernel


def _source(snapshot, kernel, original=False, describe=False):
    source = snapshot.get('source_sha256')
    need(snapshot.get('axis') == 0 and snapshot.get('upload') == 0 and
         digest(source) and type(snapshot.get('source_bytes')) is int and snapshot['source_bytes'] > 0,
         'missing source fingerprint')
    field = 'expected_compiler_input' if original else 'compiler_input'
    compiler = snapshot[field]
    need(compiler.get('source_sha256') == source and compiler.get('function') == kernel.get('function') and
         type(kernel.get('function')) is str and bool(kernel['function']) and
         digest(compiler.get('original_compile_header_sha256')),
         'compiler input/source fingerprint mismatch')
    options = compiler.get('options')
    need(type(options) is list and options and all(type(x) is str and x for x in options),
         'missing compiler options')
    actual = snapshot.get('actual_compiler_options', 'missing')
    if original or describe:
        need(actual is None, 'unexpected actual compiler options')
    else:
        need(snapshot.get('compiler_options_match') is True and actual == options,
             'actual compiler options differ')


def _snapshots(result, kernel, request, describe):
    snapshot = _one(result.get('precompile_kernels'), 'missing precompile snapshot')
    _source(snapshot, kernel, describe=describe)
    pre, post, canonical = (snapshot[k] for k in ('precodegen_raw_descriptor', 'raw_descriptor', 'canonical_descriptor'))
    pair = _one(result['descriptor_validation'].get('pairs'), 'missing post-initialization descriptor pair')
    need(result['descriptor_validation'].get('all_match') is True and pair.get('equal') is True and
         pair.get('precompile_canonical') == canonical and pair.get('postinit_canonical') == canonical,
         'precompile/post-initialization descriptors differ')
    for field in ('fftDim', 'FFTdim', 'complexSize', 'numAxisUploads', 'numStages', 'stageRadix',
                  'registers_per_radix', 'min_logical_registers_per_thread', 'logical_registers_per_thread',
                  'registerBoost', 'groupedBatch', 'localSize', 'axisBlock', 'axisSwapped',
                  'useRader', 'useBluesteinFFT'):
        need(field in pre and field in post and pre[field] == post[field], 'pre/post-codegen plan differs: ' + field)
    swap, local = post['axisSwapped'], post['localSize']
    block = post['axisBlock']
    need(type(swap) is int and swap in (0, 1) and type(local) is list and len(local) == 3 and
         all(type(x) is int and x > 0 for x in local) and
         type(block) is list and len(block) == 4 and all(type(x) is int and x > 0 for x in block),
         'invalid raw layout')
    normalized = list(block)
    if swap:
        normalized[0], normalized[1] = normalized[1], normalized[0]
    need(canonical.get('axisBlock') == normalized and kernel.get('axisBlock') == normalized and
         canonical.get('localSize') == local and kernel.get('localSize') == local,
         'canonical/final layout mismatch')
    for field in ('stageRadix', 'groupedBatch', 'logical_registers_per_thread',
                  'min_logical_registers_per_thread', 'registerBoost', 'fftDim', 'numStages'):
        need(field in post and canonical.get(field) == post[field] and kernel.get(field) == post[field],
             'post-codegen/final plan differs: ' + field)
    for field in ('readToRegisters', 'writeFromRegisters', 'dynamic_shared_bytes'):
        need(field in post and canonical.get(field) == post[field] and kernel.get(field) == post[field],
             'post-codegen/final metadata differs: ' + field)
    need(canonical.get('registers_per_radix') == post['registers_per_radix'], 'canonical table mismatch')
    if request is not None:
        derived = candidate.validate_request(request)
        n, group = request['workload']['n'], request['grouped_batch']
        need(post['FFTdim'] == 1 and post['fftDim'] == n and post['complexSize'] == 8 and
             post['numAxisUploads'] == 1 and post['registerBoost'] == 1 and
             post['useRader'] == 0 and post['useBluesteinFFT'] == 0 and
             post['numStages'] == len(request['radix']) and post['stageRadix'] == request['radix'] and
             post['registers_per_radix'] == request['register_table'] and
             post['min_logical_registers_per_thread'] == derived['min_registers'] and
             post['logical_registers_per_thread'] == derived['max_registers'] and
             post['groupedBatch'] == group and local[2] == 1 and
             local[1 if swap else 0] == derived['planned_fft_threads'] and
             local[0 if swap else 1] == group and normalized[:3] == [derived['planned_fft_threads'], group, 1],
             'actual plan differs from nonuniform request')
    return snapshot


def _compiled(result, kernel, request):
    diag = _one(result.get('compiled_diagnostics'), 'missing compiled diagnostics')
    for field in ('cuda_num_regs', 'cuda_local_bytes', 'cuda_static_shared_bytes', 'cuda_max_threads_per_block'):
        need(nonnegative_int(diag.get(field)), 'missing or invalid compiled resource: ' + field)
    need(diag['cuda_max_threads_per_block'] > 0, 'invalid function thread limit')
    if request is None:
        return
    limits = result['expanded_resource_validation']
    for field in ('cuda_registers_per_thread', 'cuda_local_bytes', 'dynamic_shared_bytes', 'static_shared_bytes',
                  'function_dynamic_limit_bytes', 'device_shared_limit_bytes', 'threads', 'function_thread_limit'):
        need(nonnegative_int(limits.get(field)), 'missing or invalid candidate resource: ' + field)
    need(limits['cuda_registers_per_thread'] == diag['cuda_num_regs'] and
         limits['cuda_local_bytes'] == diag['cuda_local_bytes'] and
         limits['static_shared_bytes'] == diag['cuda_static_shared_bytes'] and
         limits['function_thread_limit'] == diag['cuda_max_threads_per_block'] and
         limits['dynamic_shared_bytes'] == kernel['dynamic_shared_bytes'] and
         limits['dynamic_shared_bytes'] <= limits['function_dynamic_limit_bytes'] and
         limits['dynamic_shared_bytes'] + limits['static_shared_bytes'] <= limits['device_shared_limit_bytes'] and
         0 < limits['threads'] <= limits['function_thread_limit'] and
         limits['threads'] == math.prod(kernel['localSize']), 'compiled resource check failed')


def _check(row, gpu, original, request, timing):
    r, w, kernel = _common(row, gpu, original, request, False)
    need(r.get('validation_output_initialization') == '0xff_bytes_fp32_nan_before_single_and_final_timed_forward' and
         r.get('single_forward_input_unchanged') is True and r.get('input_unchanged') is True,
         'output poisoning or immutable input check failed')
    for field in ('single_forward_correctness', 'correctness'):
        result = r[field]
        reference = ('independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity'
                     if w['pattern'] == 'random' else 'analytic_pattern_reference_with_independent_CPU_FFT_crosscheck')
        need(result.get('passed') is True and type(result.get('nonfinite_count')) is int and result['nonfinite_count'] == 0 and
             type(result.get('failing_batches')) is int and result['failing_batches'] == 0 and
             type(result.get('checked_elements')) is int and result['checked_elements'] == w['n'] * w['batch'] and
             result.get('reference') == reference and result.get('criterion') == 'criterion_v2' and
             math.isclose(result['l2_scaled_error_threshold'], 1e-5, rel_tol=1e-12) and
             math.isclose(result['linf_input_scaled_error_threshold'], 1e-4, rel_tol=1e-12),
             'numerical check failed')
    graph = w['graph_repeats']
    need(type(graph) is int and graph > 0 and r.get('graph_repeats') == graph and
         r.get('graph_kernel_count') == graph and r.get('graph_event_record_count') == 2, 'graph mismatch')
    samples, raw = r['exec_ms_samples'], r['raw_interval_ms_samples']
    need(type(w['repeats']) is int and w['repeats'] > 0 and type(samples) is list and type(raw) is list and
         len(samples) == len(raw) == w['repeats'] and all(positive(x) for x in samples + raw) and
         all(math.isclose(x, y / graph, rel_tol=2e-6, abs_tol=1e-9) for x, y in zip(samples, raw)),
         'invalid timing samples')
    if timing:
        need(w['repeats'] == 31 and w['warmup'] >= 5 and w['warmup_ms'] >= 1000 and min(raw) >= .5,
             'timing protocol mismatch')
    if original:
        need(r['init_trace']['available'] is False and r.get('precompile_kernels') == [], 'original trace mismatch')
        snapshot = _one(r.get('postinit_source_kernels'), 'missing original retained-source snapshot')
        need(snapshot.get('snapshot_stage') == 'post_initialize_retained_source', 'original snapshot stage mismatch')
        _source(snapshot, kernel, original=True)
    else:
        _snapshots(r, kernel, request, False)
    _compiled(r, kernel, request)
    return r


def check(row, gpu, *, original=False, request=None, timing=False):
    """Check one execute row; candidate requests are never masqueraded as defaults."""
    try:
        return _check(row, gpu, original, request, timing)
    except (KeyError, TypeError, IndexError, ZeroDivisionError, OverflowError) as error:
        raise ValueError('missing or invalid execution evidence') from error


def check_describe(row, gpu, request):
    """Check one noncompiled describe row without inventing compiled evidence."""
    try:
        need(request is not None, 'describe requires a request')
        r, w, kernel = _common(row, gpu, False, request, True)
        counts = r['init_trace']['counters']
        need(all(counts[name] == 0 for name in ('nvrtc_compile', 'module_load', 'driver_kernel_launch')),
             'describe unexpectedly compiled, loaded or launched')
        _snapshots(r, kernel, request, True)
        return r
    except (KeyError, TypeError, IndexError, ZeroDivisionError, OverflowError) as error:
        raise ValueError('missing or invalid describe evidence') from error
