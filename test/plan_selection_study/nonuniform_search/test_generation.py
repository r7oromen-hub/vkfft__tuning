import hashlib
from pathlib import Path
import tempfile
import unittest

import prepare_headers


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class GenerationTests(unittest.TestCase):
    def test_isolated_planning_order_trace_and_harness(self):
        source = ROOT / 'VkFFT/vkFFT'
        original_scheduler = source / 'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h'
        before = hashlib.sha256(original_scheduler.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            output = Path(tmp) / 'include'
            provenance = prepare_headers.prepare(source, output)
            scheduler = (output / 'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h').read_text(encoding='utf8')
            splitter = (output / 'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h').read_text(encoding='utf8')
            plan = (output / 'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h').read_text(encoding='utf8')
            harness = (Path(tmp) / 'expanded_benchmark.cpp').read_text(encoding='utf8')
            seed = scheduler.index('expanded_candidate::seed(')
            optimize = scheduler.index('res = VkFFTOptimizeRadixKernels(', seed)
            order = scheduler.index('expanded_candidate::reorder(', optimize)
            minmax = scheduler.index('res = VkFFTMinMaxRegisterCheck(', order)
            consistent = scheduler.index('expanded_candidate::consistent(', minmax)
            self.assertTrue(seed < optimize < order < minmax < consistent)
            self.assertIn('!expanded_candidate::enabled() && min_registers_per_thread != registers_per_thread', scheduler)
            self.assertIn('fftDim.data.i / request.min_regs', splitter)
            self.assertNotIn('fftDim.data.i / request.regs', splitter)
            self.assertIn('request.workload_batch != (long long)FFTPlan->actualFFTSizePerAxis[0][1]', splitter)
            pre = plan.index('vkfft_trace::capture_precodegen(app, axis)')
            codegen = plan.index('invoke("shader_codegen"')
            self.assertLess(plan.index('expanded_candidate::consistent('), pre)
            self.assertLess(pre, codegen)
            self.assertGreater(plan.rindex('expanded_candidate::consistent('), codegen)
            self.assertIn('VKFFT_NONUNIFORM_TABLE', harness)
            self.assertIn('valid(request, o.n, o.batch)', harness)
            self.assertIn('axis.axisBlock[0] != (pfUINT)(o.n/request.min_regs)', harness)
            self.assertIn('nonuniform requests require the generated planner binary', harness)
            self.assertIn('expanded_resource_validation', harness)
            self.assertIn('poison single-forward output', harness)
            self.assertIn('precodegen_raw_descriptor', (output / 'trace.hpp').read_text(encoding='utf8'))
            self.assertIn('nonuniform_v1', provenance)
            for relative, digest in provenance['expanded'].items():
                self.assertEqual(hashlib.sha256((Path(tmp) / relative).read_bytes()).hexdigest(), digest)
            self.assertEqual((output / 'candidate.hpp').read_bytes(), (HERE / 'candidate.hpp').read_bytes())
            with self.assertRaises(ValueError):
                prepare_headers.prepare(source, output)
        self.assertEqual(hashlib.sha256(original_scheduler.read_bytes()).hexdigest(), before)

    def test_changed_generated_anchors_fail_closed(self):
        for transform in (prepare_headers.transform_splitter, prepare_headers.transform_plan,
                          prepare_headers.transform_benchmark):
            with self.subTest(transform=transform.__name__), self.assertRaises(ValueError):
                transform('unexpected generated source')


if __name__ == '__main__':
    unittest.main()
