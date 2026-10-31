"""Representable default replay and three-binary identity evidence.

The caller must first validate the actual execution rows, including GPU guard,
numeric checks, environment and timing, with the table-aware validator.  These
functions take benchmark result objects; they do not manufacture or alter rows.
"""
import copy

import candidate
import validation


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_int(value):
    return type(value) is int and value > 0


def representable(default):
    """Return an exact request or (None, reason); never synthesize table slots.

    A default can be representable without having passed a device check.  The
    caller must retain and validate its real guarded execution evidence.
    """
    try:
        if default.get('status') != 'ok' or default.get('header_instrumentation') is not True:
            return None, 'not_successful_instrumented_default'
        shape = default['structural_features']
        kernels, snapshots = shape['kernels'], default['precompile_kernels']
        if shape['numAxisUploads'] != 1 or len(kernels) != 1 or len(snapshots) != 1:
            return None, 'requires_one_upload_and_kernel'
        kernel = kernels[0]
        raw = snapshots[0]['raw_descriptor']
        n, batch = default['n'], default['batch']
        radix, table, group = kernel['stageRadix'], raw['registers_per_radix'], kernel['groupedBatch']
        if type(raw['registerBoost']) is not int or raw['registerBoost'] != 1:
            return None, 'register_boost_not_one'
        expected_scope = {'FFTdim': 1, 'numAxisUploads': 1, 'fftDim': n, 'complexSize': 8,
                          'useRader': 0, 'useBluesteinFFT': 0, 'performR2C': 0, 'performDCT': 0,
                          'performDST': 0, 'inverse': 0, 'actualInverse': 0, 'axis': 0,
                          'upload': 0, 'precision': 0}
        if any(type(raw[field]) is not int or raw[field] != value for field, value in expected_scope.items()):
            return None, 'outside_restricted_scope'
        if (kernel['stageRadix'] != raw['stageRadix'] or raw['numStages'] != len(radix) or
                raw['groupedBatch'] != group or kernel['localSize'] != raw['localSize']):
            return None, 'inconsistent_observed_structure'
        request = {'schema': candidate.SCHEMA, 'workload': {'n': n, 'batch': batch},
                   'radix': copy.deepcopy(radix), 'register_table': copy.deepcopy(table),
                   'profile': None, 'grouped_batch': group}
        derived = None
        # Deterministic preference preserves the frontier's full uniform table
        # when it also happens to satisfy the active-only profile.
        for profile in ('uniform_compatibility', 'nonuniform_active_only'):
            request['profile'] = profile
            try:
                derived = candidate.validate_request(request)
                break
            except (ValueError, TypeError, KeyError):
                pass
        if derived is None:
            return None, 'unsupported_table_or_request_profile'
        if (raw['min_logical_registers_per_thread'] != derived['min_registers'] or
                raw['logical_registers_per_thread'] != derived['max_registers'] or
                kernel['min_logical_registers_per_thread'] != derived['min_registers'] or
                kernel['logical_registers_per_thread'] != derived['max_registers'] or
                type(kernel['registerBoost']) is not int or kernel['registerBoost'] != 1):
            return None, 'register_envelope_mismatch'
        swap, local = raw['axisSwapped'], raw['localSize']
        if (type(swap) is not int or swap not in (0, 1) or type(local) is not list or len(local) != 3 or
                not all(_positive_int(x) for x in local) or local[2] != 1 or
                local[1 if swap else 0] != derived['planned_fft_threads'] or
                local[0 if swap else 1] != group):
            return None, 'unsupported_local_layout'
        # In the instrumentation harness axisBlock is normalized after swap;
        # raw descriptor/localSize retain the pre-normalization orientation.
        block = kernel['axisBlock']
        if (type(block) is not list or len(block) != 4 or
                block[:3] != [derived['planned_fft_threads'], group, 1] or
                block[3] != n):
            return None, 'unsupported_normalized_axis_block'
        return request, None
    except (KeyError, TypeError, IndexError, ValueError, AttributeError):
        return None, 'missing_or_invalid_default_evidence'


def _sha(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(char in '0123456789abcdef' for char in value))


def _common(result, *, original):
    _need(isinstance(result, dict) and result.get('status') == 'ok', 'result did not succeed')
    _need(result.get('header_instrumentation') is (not original), 'incorrect binary instrumentation')
    _need(result.get('requested_config') == {'makeForwardPlanOnly': 1}, 'non-default configuration')
    shape = result['structural_features']
    _need(shape['numAxisUploads'] == 1 and len(shape['kernels']) == 1, 'not one kernel/upload')
    diagnostics = result['compiled_diagnostics']
    _need(isinstance(diagnostics, list) and len(diagnostics) == 1, 'compiled diagnostics missing')
    resources = diagnostics[0]
    for key in ('cuda_local_bytes', 'cuda_num_regs', 'cuda_static_shared_bytes', 'cuda_max_threads_per_block'):
        _need(type(resources.get(key)) is int and resources[key] >= 0, 'compiled resource evidence missing')
    _need(resources['cuda_max_threads_per_block'] > 0, 'invalid compiled thread limit')
    config = dict(result['resolved_config'])
    _need(config.pop('keepShaderCode', None) == (1 if original else 0), 'source-retention flag mismatch')
    field = 'postinit_source_kernels' if original else 'precompile_kernels'
    snapshots = result[field]
    _need(isinstance(snapshots, list) and len(snapshots) == 1, 'source snapshot missing')
    snap = snapshots[0]
    _need(_sha(snap.get('source_sha256')) and _positive_int(snap.get('source_bytes')) and
          isinstance(snap.get('canonical_descriptor'), dict) and snap['canonical_descriptor'],
          'invalid source fingerprint or descriptor')
    _need(snap.get('axis') == 0 and snap.get('upload') == 0, 'source snapshot axis mismatch')
    if original:
        _need(result.get('init_trace', {}).get('available') is False and
              snap.get('snapshot_stage') == 'post_initialize_retained_source' and
              'actual_compiler_options' in snap and snap['actual_compiler_options'] is None,
              'original retained-source evidence missing or options falsely claimed')
        compiler = snap['expected_compiler_input']
    else:
        _need(result.get('descriptor_validation', {}).get('all_match') is True and
              snap.get('compiler_options_match') is True, 'instrumented descriptor/options mismatch')
        compiler = snap['compiler_input']
        _need(isinstance(snap.get('actual_compiler_options'), list) and
              snap['actual_compiler_options'] == compiler.get('options'), 'actual compiler options missing')
    options = compiler.get('options')
    expected = validation.expected_architecture_option(result['gpu'])
    _need(options == [expected] and
          compiler.get('source_sha256') == snap['source_sha256'] and
          compiler.get('function') == shape['kernels'][0].get('function') and
          _sha(compiler.get('original_compile_header_sha256')), 'compiler input evidence missing')
    if not original:
        _need(compiler.get('architecture_option') == expected and type(compiler.get('backend')) is int and
              compiler['backend'] == 1, 'compiler architecture or backend mismatch')
    return {'shape': shape, 'diagnostics': diagnostics, 'config': config,
            'descriptor': snap['canonical_descriptor'], 'source_sha256': snap['source_sha256'],
            'source_bytes': snap['source_bytes'], 'compiler_options': options,
            'original_compile_header_sha256': compiler['original_compile_header_sha256'],
            'compiler_input': compiler, 'actual_options': snap['actual_compiler_options']}


def compare_three(original, expanded_default, identity):
    """Compare three already-validated results; return proof with explicit limits.

    Original actual NVRTC options were not captured: only expected options from
    the restricted uninstrumented compiler path are compared to captured options.
    """
    try:
        baseline_request, reason = representable(expanded_default)
        _need(baseline_request is not None, 'default not representable: ' + str(reason))
        identity_request, reason = representable(identity)
        _need(identity_request == baseline_request, 'identity request or layout differs: ' + str(reason))
        environment = candidate.request_environment(baseline_request)
        result_fields = {
            'VKFFT_NONUNIFORM_VERSION': 'nonuniform_request_version',
            'VKFFT_NONUNIFORM_N': 'nonuniform_requested_n',
            'VKFFT_NONUNIFORM_WORKLOAD_BATCH': 'nonuniform_requested_workload_batch',
            'VKFFT_NONUNIFORM_RADIX': 'nonuniform_requested_radix',
            'VKFFT_NONUNIFORM_TABLE': 'nonuniform_requested_register_table',
            'VKFFT_NONUNIFORM_PROFILE': 'nonuniform_requested_profile',
            'VKFFT_NONUNIFORM_GROUP': 'nonuniform_requested_group',
        }
        _need(all(identity.get(field) == environment[key] for key, field in result_fields.items()),
              'identity requested fields do not match default replay')
        original_sig = _common(original, original=True)
        default_sig = _common(expanded_default, original=False)
        identity_sig = _common(identity, original=False)
        for result in (expanded_default, identity):
            _need((result['n'], result['batch'], result['seed'], result['pattern']) ==
                  (original['n'], original['batch'], original['seed'], original['pattern']) and
                  result['gpu']['uuid'] == original['gpu']['uuid'] and result['gpu']['uuid'],
                  'GPU or workload mismatch')
        for name in ('shape', 'diagnostics', 'config', 'descriptor', 'source_sha256', 'source_bytes',
                     'compiler_options', 'original_compile_header_sha256'):
            _need(original_sig[name] == default_sig[name] == identity_sig[name], 'three-way ' + name + ' mismatch')
        _need(default_sig['compiler_input'] == identity_sig['compiler_input'] and
              default_sig['actual_options'] == identity_sig['actual_options'],
              'instrumented compiler inputs or actual options differ')
        return {'status': 'match', 'request': copy.deepcopy(baseline_request),
                'candidate_id': candidate.candidate_id(baseline_request),
                'source_sha256': original_sig['source_sha256'],
                'source_bytes': original_sig['source_bytes'],
                'original_compiler_options': 'expected from restricted original path; actual options unavailable',
                'scope': 'structural/source/resource identity only; caller must validate guard, numerical and runtime evidence'}
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError('missing or invalid identity evidence') from exc
