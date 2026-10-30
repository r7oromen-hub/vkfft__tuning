import sys
import copy
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_probe
import compare_snapshots


class ProbeGenerationTest(unittest.TestCase):
    def test_anchors_and_output_are_isolated(self):
        source = HERE.parents[3] / 'VkFFT/vkFFT'
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            output = Path(temp) / 'include'
            prepare_probe.prepare(source, output)
            plan = (output/'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h').read_text(encoding='utf8')
            self.assertLess(plan.index('capture_precodegen(app, axis)'), plan.index('invoke("shader_codegen"'))
            trace = (output/'trace.hpp').read_text(encoding='utf8')
            self.assertIn('precodegen_raw_descriptor', trace)
            self.assertIn('precodegen.clear()', trace)
            with self.assertRaises(ValueError):
                prepare_probe.prepare(source, output)


class ComparisonTest(unittest.TestCase):
    def record(self):
        return {'status': 'ok', 'study_mode': 'describe', 'describe_only': True, 'n': 256, 'batch': 16,
                'descriptor_validation': {'all_match': True},
                'init_trace': {'counters': {'nvrtc_compile': 0, 'module_load': 0, 'driver_kernel_launch': 0},
                               'stage_totals': {'shader_codegen': {'calls': 1}}},
                'precompile_kernels': [{'axis': 0, 'upload': 0,
                    'precodegen_raw_descriptor': {'axis': 0, 'upload': 0, 'radix': [8, 4], 'dynamic_shared_bytes': 0},
                    'raw_descriptor': {'axis': 0, 'upload': 0, 'radix': [8, 4], 'dynamic_shared_bytes': 1024}}]}

    def test_reports_values_and_sample_specific_unchanged_fields(self):
        result = compare_snapshots.compare([self.record()])
        self.assertEqual(result['kernel_pairs'], 1)
        self.assertEqual(result['changed_fields']['dynamic_shared_bytes'][0]['precompile'], 1024)
        self.assertEqual(result['unchanged_in_all_observed_pairs'], ['axis', 'radix', 'upload'])

    def test_rejects_incomplete_or_non_describe_evidence(self):
        record = self.record()
        record['init_trace']['counters']['nvrtc_compile'] = 1
        with self.assertRaises(ValueError):
            compare_snapshots.compare([record])
        record = self.record()
        del record['precompile_kernels'][0]['precodegen_raw_descriptor']['radix']
        with self.assertRaises(ValueError):
            compare_snapshots.compare([record])
        with self.assertRaises(ValueError):
            compare_snapshots.compare([])

    def test_control_comparison_requires_equal_source_and_structure(self):
        probe = self.record()
        probe.update({'requested_config': {}, 'placement': 'outofplace', 'backend': 'CUDA_NVRTC', 'precision': 'complex_fp32',
                      'cuda_runtime_version': 12060, 'cuda_driver_version': 12060, 'nvrtc_version': '12.6',
                      'gpu': {'uuid': 'GPU-test', 'compute_capability': '8.0'}})
        kernel = probe['precompile_kernels'][0]
        kernel.update({'canonical_descriptor': kernel['raw_descriptor'], 'source_sha256': 'abc', 'source_bytes': 3, 'compiler_input': {}})
        control = copy.deepcopy(probe)
        del control['precompile_kernels'][0]['precodegen_raw_descriptor']
        self.assertEqual(compare_snapshots.compare_control([probe], [control])['record_count'], 1)
        control['precompile_kernels'][0]['source_sha256'] = 'different'
        with self.assertRaises(ValueError):
            compare_snapshots.compare_control([probe], [control])


if __name__ == '__main__':
    unittest.main()
