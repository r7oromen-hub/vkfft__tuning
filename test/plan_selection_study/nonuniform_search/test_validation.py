"""Synthetic schema checks only; none of these rows is GPU evidence."""
import copy
import unittest

import candidate
import identity
import validation


GPU = 'GPU-fixture'
SOURCE = 'a' * 64
HEADER = 'b' * 64
OPTIONS = ['--gpu-architecture=sm_89']


def make_request():
    table = [0] * 33
    table[2], table[8] = 4, 8
    return {'schema': candidate.SCHEMA, 'workload': {'n': 128, 'batch': 16},
            'radix': [8, 8, 2], 'register_table': table,
            'profile': 'nonuniform_active_only', 'grouped_batch': 2}


def make_row(request=None, *, role='candidate', describe=False):
    request = copy.deepcopy(make_request() if request is None else request)
    values = candidate.validate_request(request)
    n, batch = request['workload']['n'], request['workload']['batch']
    group = request['grouped_batch']; threads = values['planned_fft_threads']
    local = [group, threads, 1] if group > 1 else [threads, 1, 1]
    axis_block = [threads, group, 1, n]
    raw_axis_block = [group, threads, 1, n] if group > 1 else axis_block[:]
    desc = {'FFTdim': 1, 'complexSize': 8, 'numAxisUploads': 1, 'numStages': len(request['radix']),
            'stageRadix': request['radix'][:], 'fftDim': n, 'groupedBatch': group,
            'logical_registers_per_thread': values['max_registers'],
            'min_logical_registers_per_thread': values['min_registers'],
            'registers_per_radix': request['register_table'][:], 'registerBoost': 1,
            'localSize': local[:], 'axisBlock': raw_axis_block,
            'axisSwapped': int(group > 1), 'useRader': 0, 'useBluesteinFFT': 0,
            'performR2C': 0, 'performDCT': 0, 'performDST': 0,
            'dynamic_shared_bytes': 4096, 'readToRegisters': 0, 'writeFromRegisters': 0,
            'axis': 0, 'upload': 0}
    canonical = copy.deepcopy(desc)
    canonical.pop('axisSwapped')
    canonical['axisBlock'] = axis_block[:]
    precodegen = copy.deepcopy(desc)
    precodegen['dynamic_shared_bytes'] = 0
    kernel = {'stageRadix': request['radix'][:], 'numStages': len(request['radix']),
              'fftDim': n, 'groupedBatch': group, 'localSize': local[:], 'axisBlock': axis_block[:],
              'logical_registers_per_thread': values['max_registers'],
              'min_logical_registers_per_thread': values['min_registers'], 'registerBoost': 1,
              'dynamic_shared_bytes': 4096, 'readToRegisters': 0, 'writeFromRegisters': 0,
              'function': 'VkFFT_main', 'upload': 0}
    compiler = {'source_sha256': SOURCE, 'options': OPTIONS[:],
                'original_compile_header_sha256': HEADER,
                'function': 'VkFFT_main', 'architecture_option': OPTIONS[0], 'backend': 1}
    pre = {'axis': 0, 'upload': 0, 'raw_descriptor': copy.deepcopy(desc),
           'precodegen_raw_descriptor': precodegen, 'canonical_descriptor': canonical,
           'source_sha256': SOURCE, 'source_bytes': 19099, 'compiler_input': compiler,
           'actual_compiler_options': None if describe else OPTIONS[:], 'compiler_options_match': True}
    correctness = {'passed': True, 'nonfinite_count': 0, 'failing_batches': 0,
                   'checked_elements': n * batch,
                   'reference': 'independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity',
                   'criterion': 'criterion_v2', 'l2_scaled_error_threshold': 1e-5,
                   'linf_input_scaled_error_threshold': 1e-4}
    repeats, graph = 31, 2048
    work = {'n': n, 'batch': batch, 'seed': 42, 'pattern': 'random', 'graph_repeats': graph,
            'repeats': repeats, 'warmup': 5, 'warmup_ms': 1000,
            'study_mode': 'describe' if describe else 'execute', 'config': {}}
    original = role == 'original'
    result = {'status': 'ok', 'gpu': {'uuid': GPU, 'cuda_visible_devices': GPU},
              'n': n, 'batch': batch, 'seed': 42, 'pattern': 'random',
              'backend': 'CUDA_NVRTC', 'precision': 'complex_fp32', 'placement': 'outofplace',
              'forward_direction': -1,
              'study_mode': work['study_mode'], 'describe_only': describe,
              'requested_config': {'makeForwardPlanOnly': 1},
              'resolved_config': {'keepShaderCode': int(original)},
              'header_instrumentation': not original,
              'validation_output_initialization': '0xff_bytes_fp32_nan_before_single_and_final_timed_forward',
              'input_unchanged': True, 'single_forward_input_unchanged': True,
              'correctness': copy.deepcopy(correctness), 'single_forward_correctness': copy.deepcopy(correctness),
              'structural_features': {'numAxisUploads': 1, 'kernels': [kernel]},
              'compiled_diagnostics': [] if describe else [
                  {'upload': 0, 'cuda_num_regs': 54, 'cuda_local_bytes': 0,
                   'cuda_static_shared_bytes': 0, 'cuda_max_threads_per_block': threads * group}],
              'graph_repeats': graph, 'graph_kernel_count': graph, 'graph_event_record_count': 2,
              'exec_ms_samples': [.0005] * repeats, 'raw_interval_ms_samples': [1.024] * repeats,
              'init_trace': {'available': False} if original else {
                  'counters': {'nvrtc_compile': 0 if describe else 1, 'module_load': 0 if describe else 1,
                               'driver_kernel_launch': 0}},
              'descriptor_validation': {'all_match': True, 'pairs': [
                  {'axis': 0, 'upload': 0, 'equal': True, 'precompile_canonical': copy.deepcopy(canonical),
                   'postinit_canonical': copy.deepcopy(canonical)}]},
              'precompile_kernels': [pre]}
    if not describe:
        result['expanded_resource_validation'] = {
            'cuda_registers_per_thread': 54, 'cuda_local_bytes': 0,
            'dynamic_shared_bytes': 4096, 'static_shared_bytes': 0,
            'function_dynamic_limit_bytes': 49152, 'device_shared_limit_bytes': 101376,
            'threads': threads * group, 'function_thread_limit': threads * group}
    env = candidate.request_environment(request) if role == 'candidate' else {}
    if env:
        for suffix, env_key in (('version', 'VERSION'), ('requested_n', 'N'),
                                ('requested_workload_batch', 'WORKLOAD_BATCH'), ('requested_radix', 'RADIX'),
                                ('requested_register_table', 'TABLE'), ('requested_profile', 'PROFILE'),
                                ('requested_group', 'GROUP')):
            result['nonuniform_request_version' if suffix == 'version' else 'nonuniform_' + suffix] = env['VKFFT_NONUNIFORM_' + env_key]
    if original:
        result['precompile_kernels'] = []
        result.pop('expanded_resource_validation', None)
        result['postinit_source_kernels'] = [{'axis': 0, 'upload': 0,
            'canonical_descriptor': copy.deepcopy(canonical), 'source_sha256': SOURCE, 'source_bytes': 19099,
            'expected_compiler_input': {'source_sha256': SOURCE, 'options': OPTIONS[:],
                                        'original_compile_header_sha256': HEADER, 'function': 'VkFFT_main'},
            'snapshot_stage': 'post_initialize_retained_source',
            'actual_compiler_options': None}]
    return {'guard_clean': True, 'guard_gpu_uuid': GPU, 'returncode': 0, 'status': 'ok',
            'direct_environment': env, 'workload': work, 'result': result}


def changed(row, path, value):
    row = copy.deepcopy(row)
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return row


class ValidationTests(unittest.TestCase):
    def test_valid_candidate_execute_and_describe_are_distinct(self):
        req = make_request()
        row = make_row(req)
        self.assertEqual(validation.check(row, GPU, request=req, timing=True), row['result'])
        default = make_row(req, role='default')
        self.assertEqual(validation.check(default, GPU, timing=True), default['result'])
        original = make_row(req, role='original')
        self.assertEqual(validation.check(original, GPU, original=True, timing=True), original['result'])
        description = make_row(req, describe=True)
        self.assertEqual(validation.check_describe(description, GPU, req), description['result'])
        with self.assertRaises((ValueError, KeyError, TypeError)):
            validation.check(description, GPU, request=req, timing=True)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            validation.check_describe(row, GPU, req)

    def test_candidate_environment_guard_and_trace_cannot_be_spoofed(self):
        req = make_request(); row = make_row(req)
        mutations = [
            (('guard_clean',), False), (('guard_gpu_uuid',), 'GPU-other'),
            (('returncode',), 1), (('status',), 'error'),
            (('direct_environment', 'VKFFT_NONUNIFORM_TABLE'), '0'),
            (('result', 'nonuniform_requested_register_table'), '0'),
            (('result', 'nonuniform_request_version'), 'wrong/v1'),
            (('workload', 'n'), 256), (('result', 'gpu', 'uuid'), 'GPU-other'),
            (('result', 'gpu', 'cuda_visible_devices'), 'GPU-other'),
            (('result', 'header_instrumentation'), False),
        ]
        for path, value in mutations:
            with self.subTest(path=path), self.assertRaises((ValueError, KeyError, TypeError)):
                validation.check(changed(row, path, value), GPU, request=req, timing=True)
        # An injected row must never be accepted as a default by removing its request.
        with self.assertRaises((ValueError, KeyError, TypeError)):
            validation.check(row, GPU, request=None)
        # Deleting the row environment still leaves the generated request telemetry.
        with self.assertRaises((ValueError, KeyError, TypeError)):
            validation.check(changed(row, ('direct_environment',), {}), GPU, request=None)

    def test_planning_and_all_descriptor_snapshots_are_checked(self):
        req = make_request(); row = make_row(req)
        p = ('result', 'precompile_kernels', 0)
        cases = [
            (p + ('precodegen_raw_descriptor', 'registers_per_radix', 2), 8),
            (p + ('precodegen_raw_descriptor', 'min_logical_registers_per_thread'), 8),
            (p + ('precodegen_raw_descriptor', 'stageRadix'), [2, 8, 8]),
            (p + ('raw_descriptor', 'registers_per_radix', 4), 8),
            (p + ('raw_descriptor', 'logical_registers_per_thread'), 4),
            (p + ('raw_descriptor', 'registerBoost'), 2),
            (p + ('raw_descriptor', 'groupedBatch'), 1),
            (p + ('raw_descriptor', 'axisSwapped'), 0),
            (p + ('raw_descriptor', 'localSize'), [2, 16, 1]),
            (('result', 'structural_features', 'kernels', 0, 'stageRadix'), [2, 8, 8]),
            (('result', 'structural_features', 'kernels', 0, 'axisBlock'), [16, 2, 1, 128]),
            (('result', 'descriptor_validation', 'pairs', 0, 'postinit_canonical', 'registers_per_radix', 2), 8),
            (('result', 'descriptor_validation', 'all_match'), False),
        ]
        for path, value in cases:
            with self.subTest(path=path), self.assertRaises((ValueError, KeyError, TypeError)):
                validation.check(changed(row, path, value), GPU, request=req)
            with self.subTest(describe_path=path), self.assertRaises((ValueError, KeyError, TypeError)):
                validation.check_describe(changed(make_row(req, describe=True), path, value), GPU, req)

    def test_source_options_and_compiled_resource_checks(self):
        req = make_request(); row = make_row(req); p = ('result', 'precompile_kernels', 0)
        cases = [
            (p + ('source_sha256',), 'not-a-sha'),
            (p + ('source_bytes',), 0),
            (p + ('axis',), 1),
            (p + ('compiler_input', 'source_sha256'), 'c' * 64),
            (p + ('compiler_input', 'function'), 'wrong_kernel'),
            (p + ('compiler_input', 'original_compile_header_sha256'), 'bad'),
            (p + ('actual_compiler_options',), ['--wrong']),
            (p + ('compiler_options_match',), False),
            (('result', 'compiled_diagnostics', 0, 'cuda_num_regs'), None),
            (('result', 'compiled_diagnostics', 0, 'cuda_local_bytes'), None),
            (('result', 'expanded_resource_validation', 'cuda_registers_per_thread'), 53),
            (('result', 'expanded_resource_validation', 'cuda_local_bytes'), -1),
            (('result', 'expanded_resource_validation', 'dynamic_shared_bytes'), 1000000),
            (('result', 'expanded_resource_validation', 'threads'), 128),
            (('result', 'expanded_resource_validation', 'function_thread_limit'), 1),
        ]
        for path, value in cases:
            with self.subTest(path=path), self.assertRaises((ValueError, KeyError, TypeError)):
                validation.check(changed(row, path, value), GPU, request=req)
        description = make_row(req, describe=True)
        for path, value in [
                (('result', 'precompile_kernels', 0, 'actual_compiler_options'), OPTIONS[:]),
                (('result', 'init_trace', 'counters', 'nvrtc_compile'), 1),
                (('result', 'init_trace', 'counters', 'module_load'), 1),
                (('result', 'init_trace', 'counters', 'driver_kernel_launch'), 1)]:
            with self.subTest(describe_path=path), self.assertRaises((ValueError, KeyError, TypeError)):
                validation.check_describe(changed(description, path, value), GPU, req)

    def test_correctness_and_timing_fail_closed(self):
        req = make_request(); row = make_row(req)
        cases = [
            (('result', 'validation_output_initialization'), 'unpoisoned'),
            (('result', 'input_unchanged'), False),
            (('result', 'single_forward_input_unchanged'), False),
            (('result', 'correctness', 'passed'), False),
            (('result', 'single_forward_correctness', 'checked_elements'), 127),
            (('result', 'correctness', 'nonfinite_count'), 1),
            (('result', 'correctness', 'failing_batches'), 1),
            (('result', 'correctness', 'reference'), 'not-independent'),
            (('result', 'correctness', 'l2_scaled_error_threshold'), .1),
            (('result', 'graph_kernel_count'), 1),
            (('result', 'raw_interval_ms_samples'), [.01] * 31),
            (('result', 'exec_ms_samples'), [float('nan')] * 31),
            (('workload', 'warmup_ms'), 0),
        ]
        for path, value in cases:
            with self.subTest(path=path), self.assertRaises((ValueError, KeyError, TypeError)):
                validation.check(changed(row, path, value), GPU, request=req, timing=True)


class IdentityTests(unittest.TestCase):
    def test_representability_preserves_exact_table_and_layout(self):
        req = make_request(); result = make_row(req, role='default')['result']
        replay, reason = identity.representable(result)
        self.assertIsNone(reason)
        self.assertEqual(replay, req)
        bad = copy.deepcopy(result)
        bad['precompile_kernels'][0]['raw_descriptor']['registers_per_radix'][4] = 8
        replay, reason = identity.representable(bad)
        self.assertIsNone(replay); self.assertTrue(reason)
        bad = copy.deepcopy(result)
        bad['precompile_kernels'][0]['raw_descriptor']['registerBoost'] = 2
        replay, reason = identity.representable(bad)
        self.assertIsNone(replay); self.assertTrue(reason)

    def test_three_way_comparison_and_original_options_are_expected_only(self):
        req = make_request()
        original = make_row(req, role='original')['result']
        default = make_row(req, role='default')['result']
        replay = make_row(req, role='candidate')['result']
        proof = identity.compare_three(original, default, replay)
        self.assertIsInstance(proof, dict)
        mutations = [
            ('original', ('postinit_source_kernels', 0, 'source_sha256'), 'c' * 64),
            ('original', ('postinit_source_kernels', 0, 'expected_compiler_input', 'options'), ['wrong']),
            ('original', ('postinit_source_kernels', 0, 'actual_compiler_options'), OPTIONS[:]),
            ('default', ('precompile_kernels', 0, 'source_sha256'), 'c' * 64),
            ('identity', ('precompile_kernels', 0, 'actual_compiler_options'), ['wrong']),
            ('identity', ('compiled_diagnostics', 0, 'cuda_num_regs'), 55),
            ('identity', ('structural_features', 'kernels', 0, 'stageRadix'), [2, 8, 8]),
        ]
        for target, path, value in mutations:
            data = {'original': copy.deepcopy(original), 'default': copy.deepcopy(default),
                    'identity': copy.deepcopy(replay)}
            ptr = data[target]
            for k in path[:-1]: ptr = ptr[k]
            ptr[path[-1]] = value
            with self.subTest(target=target, path=path), self.assertRaises((ValueError, KeyError, TypeError)):
                identity.compare_three(data['original'], data['default'], data['identity'])


if __name__ == '__main__':
    unittest.main()
