import tempfile
from pathlib import Path
import unittest
import prepare_headers

HERE=Path(__file__).resolve().parent


class GenerationTest(unittest.TestCase):
    def test_isolation_and_planning_order(self):
        source=HERE.parents[2]/'VkFFT/vkFFT'
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            output=Path(temp)/'include'
            prepare_headers.prepare(source,output)
            scheduler=(output/'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h').read_text(encoding='utf8')
            start=scheduler.index('expanded_candidate::seed(')
            optimize=scheduler.index('res = VkFFTOptimizeRadixKernels(',start)
            reorder=scheduler.index('expanded_candidate::reorder(',optimize)
            minmax=scheduler.index('res = VkFFTMinMaxRegisterCheck(',reorder)
            consistent=scheduler.index('expanded_candidate::consistent(',minmax)
            self.assertTrue(start < optimize < reorder < minmax < consistent)
            self.assertIn('if (!expanded_candidate::enabled()) {\n\tif (((registers_per_thread_per_radix[32]',scheduler)
            self.assertIn('!expanded_candidate::enabled() && min_registers_per_thread != registers_per_thread',scheduler)
            benchmark=(Path(temp)/'expanded_benchmark.cpp').read_text(encoding='utf8')
            self.assertIn('expanded_resource_validation',benchmark)
            self.assertIn('poison single-forward output',benchmark)
            plan=(output/'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h').read_text(encoding='utf8')
            self.assertLess(plan.index('capture_precodegen(app, axis)'),plan.index('invoke("shader_codegen"'))
            self.assertLess(plan.index('invoke("shader_codegen"'),plan.index('expanded_candidate::layout_matches('))
            self.assertLess(plan.index('expanded_candidate::layout_matches('),plan.index('vkfft_trace::capture_precompile(app, axis)'))
            self.assertIn('precodegen_raw_descriptor',(output/'trace.hpp').read_text(encoding='utf8'))
            with self.assertRaises(ValueError): prepare_headers.prepare(source,output)


if __name__=='__main__': unittest.main()
