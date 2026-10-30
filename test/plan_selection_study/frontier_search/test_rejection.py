from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'expanded_search'))
import prepare_headers
import rejection_instrumentation


class RejectionGenerationTests(unittest.TestCase):
    def test_real_generation_instruments_every_prototype_return(self):
        source = HERE.parents[2]/'VkFFT/vkFFT'
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            output = Path(temp)/'include'
            prepare_headers.prepare(source, output)
            rejection_instrumentation.instrument(output)
            scheduler = (output/'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h').read_text(encoding='utf8')
            splitter = (output/'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h').read_text(encoding='utf8')
            plan = (output/'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h').read_text(encoding='utf8')
            benchmark = (Path(temp)/'expanded_benchmark.cpp').read_text(encoding='utf8')
            for item in ('scheduler_axis', 'scheduler_invalid_seed', 'scheduler_stage_multiset', 'scheduler_register_table_changed'):
                self.assertIn('expanded_rejection::record("'+item+'")', scheduler)
            for item in ('splitter_invalid_request', 'splitter_fft_threads', 'splitter_zero_dimension',
                         'splitter_thread_limit', 'splitter_shared_memory_limit'):
                self.assertIn('expanded_rejection::record("'+item+'")', splitter)
            for item in ('plan_workgroup_dimension_0', 'plan_workgroup_dimension_2', 'plan_stage_count',
                         'plan_stage_order', 'plan_register_table_changed', 'plan_local_layout'):
                self.assertIn('expanded_rejection::record("'+item+'")', plan)
            self.assertIn('if (resFFT != VKFFT_SUCCESS) return resFFT;', plan)
            self.assertIn('expanded_rejection::reset();', benchmark)
            self.assertIn('result.add("expanded_rejection_reason", "null")', benchmark)
            self.assertIn('VKFFT_FRONTIER_REJECTION:', (output/'rejection.hpp').read_text())
            with self.assertRaisesRegex(ValueError, 'anchor'):
                rejection_instrumentation.instrument(output)


if __name__ == '__main__':
    unittest.main()
