import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


requests = load('joint_requests_test', HERE / 'requests.py')
validation = load('joint_validation_test', HERE / 'validation.py')


def make_row(role, req, *, lut=None, coalesced=None):
    lut = req['use_lut'] if lut is None else lut
    coalesced = req['coalesced_memory'] if coalesced is None else coalesced
    actual_lut = lut or -1
    actual_coalesced = coalesced or 32
    w = {'n': 256, 'batch': 16, 'seed': 42, 'pattern': 'random', 'graph_repeats': 1, 'repeats': 1}
    desc = {'LUT': int(actual_lut == 1), 'bufferLUTSize': 64}
    raw = dict(desc, logical_registers_per_thread=16, registerBoost=1, axisSwapped=0)
    compiled = {'registers': 10}
    input_data = {'original_compile_header_sha256': 'header'}
    pre = {'canonical_descriptor': desc, 'raw_descriptor': raw, 'source_sha256': 'source', 'source_bytes': 42,
           'compiler_input': input_data, 'actual_compiler_options': ['opts'], 'compiler_options_match': True}
    post = {'canonical_descriptor': desc, 'source_sha256': 'source', 'source_bytes': 42,
            'expected_compiler_input': dict(input_data, options=['opts']), 'actual_compiler_options': None}
    correctness = {'passed': True, 'nonfinite_count': 0, 'failing_batches': 0, 'checked_elements': 4096,
                   'reference': 'independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity',
                   'criterion': 'criterion_v2', 'l2_scaled_error_threshold': 1e-5, 'linf_input_scaled_error_threshold': 1e-4}
    original = role == 'original'
    result = {'status': 'ok', 'gpu': {'uuid': 'gpu', 'cuda_visible_devices': 'gpu'},
              'n': 256, 'batch': 16, 'seed': 42, 'pattern': 'random',
              'requested_config': requests.expected_config(lut, coalesced),
              'resolved_config': {'useLUT': actual_lut, 'coalescedMemory': actual_coalesced, 'keepShaderCode': int(original)},
              'header_instrumentation': not original, 'validation_output_initialization': '0xff_bytes_fp32_nan_before_single_and_final_timed_forward',
              'input_unchanged': True, 'single_forward_input_unchanged': True,
              'correctness': correctness, 'single_forward_correctness': correctness,
              'structural_features': {'numAxisUploads': 1, 'kernels': [{'stageRadix': [16, 16], 'groupedBatch': 4, 'localSize': [16, 4, 1]}]},
              'compiled_diagnostics': compiled, 'graph_repeats': 1, 'graph_kernel_count': 1, 'graph_event_record_count': 2,
              'exec_ms_samples': [1.0], 'raw_interval_ms_samples': [1.0],
              'init_trace': {'available': False}, 'postinit_source_kernels': [post], 'precompile_kernels': [pre],
              'descriptor_validation': {'all_match': True},
              'expanded_resource_validation': {'dynamic_shared_bytes': 1, 'static_shared_bytes': 1,
                  'function_dynamic_limit_bytes': 10, 'device_shared_limit_bytes': 10,
                  'threads': 64, 'function_thread_limit': 1024}}
    env = requests.invocation(256, 16, req)['env'] if role == 'identity' else {}
    return {'guard_clean': True, 'guard_gpu_uuid': 'gpu', 'returncode': 0, 'status': 'ok',
            'direct_environment': env, 'workload': w, 'result': result}


class KnobsTests(unittest.TestCase):
    def setUp(self):
        self.req = {'radix': [16, 16], 'logical_registers': 16, 'grouped_batch': 4,
                    'use_lut': 1, 'coalesced_memory': 64}

    def test_cartesian_identity_and_static_rejections(self):
        base = list(requests.v2.candidates(16, 2))
        rows = list(requests.enumerate_requests(16, 2))
        self.assertEqual(len(rows), len(base) * 15)
        self.assertEqual(len({r['candidate_id'] for r in rows}), len(rows))
        summary = requests.inventory_summary(16, 2)
        self.assertEqual(summary['enumerated_count'], len(rows))
        self.assertEqual(summary['pending_count'], sum(not row['reason'] for row in rows))
        self.assertEqual(summary['static_rejected_count'], sum(bool(row['reason']) for row in rows))
        for row in rows:
            self.assertEqual(requests.static_reason(16, 2, row['request']), row['reason'])
        cid = requests.candidate_id(256, 16, self.req)
        self.assertNotEqual(cid, requests.v2.candidate_id({k: self.req[k] for k in ('radix', 'logical_registers', 'grouped_batch')}))
        for update in ({'use_lut': 0}, {'coalesced_memory': 0}):
            self.assertNotEqual(cid, requests.candidate_id(256, 16, dict(self.req, **update)))
        self.assertNotEqual(cid, requests.candidate_id(256, 32, self.req))
        with self.assertRaises(ValueError):
            requests.candidate_id(256, 16, dict(self.req, use_lut=True))

    def test_same_knob_controls_and_baseline(self):
        contract = requests.comparison_contract(256, 16, self.req)
        self.assertEqual(contract['original_no_knob_performance_denominator']['args'], [])
        for name in ('matched_original_default', 'matched_experimental_default', 'matched_identity_replay'):
            self.assertEqual(contract[name]['args'], ['--use-lut', '1', '--coalesced-memory', '64'])
        self.assertFalse(contract['matched_experimental_default']['env'])
        self.assertTrue(contract['matched_identity_replay']['env'])
        rows = [make_row(role, self.req) for role in ('original', 'default', 'identity')]
        self.assertEqual(validation.check_matched_identity(*rows, 'gpu', self.req)['status'], 'pass')
        validation.check_performance_baseline(make_row('original', self.req, lut=0, coalesced=0), 'gpu')
        with self.assertRaises(ValueError):
            validation.check_performance_baseline(rows[0], 'gpu')

    def test_mismatch_and_legacy_checks_fail_closed(self):
        row = make_row('identity', self.req)
        validation.check_execute(row, 'gpu', lut=1, coalesced=64, request=self.req)
        mutations = [('requested_config', {'makeForwardPlanOnly': 1}),
                     ('resolved_config', {'useLUT': -1, 'coalescedMemory': 64}),
                     ('input_unchanged', False)]
        for field, value in mutations:
            changed = copy.deepcopy(row); changed['result'][field] = value
            with self.assertRaises(ValueError):
                validation.check_execute(changed, 'gpu', lut=1, coalesced=64, request=self.req)
        changed = copy.deepcopy(row); changed['result']['precompile_kernels'][0]['raw_descriptor']['LUT'] = 0
        with self.assertRaises(ValueError):
            validation.check_execute(changed, 'gpu', lut=1, coalesced=64, request=self.req)
        changed = copy.deepcopy(row); changed['result']['expanded_resource_validation']['threads'] = 1000
        with self.assertRaises(ValueError):
            validation.check_execute(changed, 'gpu', lut=1, coalesced=64, request=self.req)
        self.assertEqual(requests.expected_config(0, 0), {'makeForwardPlanOnly': 1})
        self.assertEqual(requests.expected_config(-1, 16)['useLUT'], -1)

    def test_identity_rejects_mismatched_workload_and_source(self):
        rows = [make_row(role, self.req) for role in ('original', 'default', 'identity')]
        rows[2]['workload']['seed'] = 53
        with self.assertRaisesRegex(ValueError, 'workloads differ'):
            validation.check_matched_identity(*rows, 'gpu', self.req)
        rows[2]['workload']['seed'] = 42
        rows[2]['result']['precompile_kernels'][0]['source_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identity replay differs'):
            validation.check_matched_identity(*rows, 'gpu', self.req)


if __name__ == '__main__':
    unittest.main()
