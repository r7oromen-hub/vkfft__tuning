import tempfile
import unittest
from pathlib import Path
import prepare_direct_headers as gen

class GenerationTests(unittest.TestCase):
    def test_anchors_and_isolation(self):
        source = Path(__file__).resolve().parents[3] / 'VkFFT' / 'vkFFT'
        scheduler = source / 'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h'
        original = scheduler.read_bytes()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as td:
            out = Path(td) / 'generated' / 'include'
            gen.prepare(source, out)
            self.assertEqual(original, scheduler.read_bytes())
            generated = (out / scheduler.relative_to(source)).read_text(encoding='utf8')
            self.assertEqual(generated.count('direct_candidate::reorder('), 1)
            split = (out / 'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h').read_text(encoding='utf8')
            self.assertEqual(split.count('direct_candidate::batch()'), 1)
            plan = (out / 'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h').read_text(encoding='utf8')
            self.assertIn('if (resFFT != VKFFT_SUCCESS) return resFFT;', plan)
            self.assertEqual(plan.count('direct_candidate::match_block('), 1)
            bench = (out.parent / 'direct_benchmark.cpp').read_text(encoding='utf8')
            self.assertEqual(bench.count('direct_candidate::match_stages('), 1)
            self.assertEqual(bench.count('cudaMemsetAsync(resources.buffer, 0xff, bytes, resources.stream)'), 2)
            self.assertLess(bench.index('"poison single-forward output"'), bench.index('"synchronize one-forward validation"'))
            self.assertLess(bench.index('"poison final validation output"'), bench.index('"record start"'))
            self.assertLess(bench.index('"poison final validation output"'), bench.index('execute(); cuda_check(cudaStreamSynchronize(resources.stream), "synchronize graph forward")'))
            self.assertTrue((out.parent / 'direct_candidate_provenance.json').is_file())

    def test_duplicate_or_missing_anchor_fails(self):
        with self.assertRaises(RuntimeError): gen.replace_once('a a', 'a', 'b')
        with self.assertRaises(RuntimeError): gen.replace_once('a', 'x', 'b')

if __name__ == '__main__': unittest.main()
