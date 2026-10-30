import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
original_path = sys.path[:]
try:
    sys.path.insert(0, str(HERE.parent / 'expanded_search'))
    spec = importlib.util.spec_from_file_location('frontier_rotation_pool_controller_test', HERE.parent / 'expanded_search' / 'rotation_controller.py')
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
finally:
    sys.path[:] = original_path


def fixture(original):
    req = {'radix': [8, 4, 8], 'logical_registers': 8, 'grouped_batch': 4}
    case = {'n': 256, 'batch': 16, 'request': req, 'original_signature': 'original', 'candidate_signature': 'candidate'}
    per = 256 * 16 * 8; slots = 2; repeat = 64
    diag = {'passed': True, 'criterion': 'criterion_v2', 'checked_elements': 4096,
            'reference': 'independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity',
            'nonfinite_count': 0, 'failing_batches': 0,
            'l2_scaled_error_threshold': 1e-5, 'linf_input_scaled_error_threshold': 1e-4}
    check = {'passed': True, 'slots_checked': slots, 'elements_per_slot': 4096,
             'slots': [{'slot': i, 'input_unchanged': True, 'correctness': copy.deepcopy(diag)} for i in range(slots)]}
    result = {'status': 'ok', 'gpu': {'uuid': 'GPU-test', 'cuda_visible_devices': 'GPU-test'},
              'n': 256, 'batch': 16, 'seed': 42, 'pattern': 'random', 'header_instrumentation': not original,
              'requested_config': {'makeForwardPlanOnly': 1}, 'rotation_mode': 'multi_address_graph_random_unique_scaled_inputs',
              'rotation_pool_slots': slots, 'rotation_pool_total_bytes': 2*slots*per,
              'rotation_input_pool_bytes': slots*per, 'rotation_output_pool_bytes': slots*per,
              'rotation_device_l2_bytes': per, 'rotation_graph_kernel_arguments_and_order_verified': True,
              'rotation_distinct_input_representatives_verified': True,
              'rotation_output_initialization': '0xff_bytes_fp32_nan_before_initial_and_separate_post_timing_validation_graph',
              'graph_repeats': repeat, 'graph_kernel_count': repeat, 'graph_event_record_count': 2, 'graph_node_count': repeat+2,
              'timing_scope': 'multi_address_graph_forward_same_stream_no_transfers_or_validation_inside_event_interval',
              'rotation_initial_validation': copy.deepcopy(check), 'rotation_final_validation': copy.deepcopy(check),
              'rotation_first_graph_interval_ms': 2.0, 'rotation_final_validation_graph_interval_ms': 2.0,
              'warmup_actual': 5, 'warmup_wall_ms': 1000, 'exec_ms_samples': [2/repeat]*31,
              'raw_interval_ms_samples': [2.0]*31, 'init_trace': {'available': False},
              'postinit_source_kernels': [{}], 'descriptor_validation': {'all_match': True},
              'structural_features': {'numAxisUploads': 1, 'kernels': [{'stageRadix': req['radix'], 'groupedBatch': 4, 'localSize': [32, 4, 1]}]},
              'precompile_kernels': [{'compiler_options_match': True,
                                      'raw_descriptor': {'axisSwapped': 0, 'logical_registers_per_thread': 8, 'registerBoost': 1}}],
              'expanded_resource_validation': {'dynamic_shared_bytes': 100, 'function_dynamic_limit_bytes': 200,
                  'static_shared_bytes': 0, 'device_shared_limit_bytes': 200, 'threads': 128, 'function_thread_limit': 1024}}
    row = {'result': result, 'workload': {'n': 256, 'batch': 16}, 'status': 'ok', 'returncode': 0,
           'guard_clean': True, 'guard_gpu_uuid': 'GPU-test',
           'direct_environment': {} if original else control.validation.request_env(req)}
    rotation = {'slots': slots, 'graph_repeats': repeat, 'l2_bytes': per}
    return row, case, rotation


class RotationValidationTests(unittest.TestCase):
    def test_both_roles_require_all_slots_and_graph_evidence(self):
        with patch.object(control.search_controller, 'source_signature',
                          side_effect=lambda _result, original: 'original' if original else 'candidate'):
            for original in (True, False):
                row, case, rotation = fixture(original)
                control.check(row, 'GPU-test', case, rotation, original=original, timing=True)
                for mode in ('initial_input', 'final_output', 'missing_slot', 'l2', 'graph'):
                    bad = copy.deepcopy(row)['result']
                    if mode == 'initial_input': bad['rotation_initial_validation']['slots'][0]['input_unchanged'] = False
                    if mode == 'final_output': bad['rotation_final_validation']['slots'][1]['correctness']['passed'] = False
                    if mode == 'missing_slot': bad['rotation_final_validation']['slots'].pop()
                    if mode == 'l2': bad['rotation_device_l2_bytes'] *= 2
                    if mode == 'graph': bad['rotation_graph_kernel_arguments_and_order_verified'] = False
                    with self.assertRaises(ValueError):
                        control.check(dict(row, result=bad), 'GPU-test', case, rotation, original=original, timing=True)


if __name__ == '__main__':
    unittest.main()
