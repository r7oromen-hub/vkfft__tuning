import tempfile
import unittest
from pathlib import Path
import prepare_direct_headers as gen
import enumerate_joint as enum

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
            self.assertEqual(generated.count('direct_candidate::refactor('), 1)
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

    def test_joint_enumeration(self):
        result = dict(n=256,batch=4,resolved_config={'sharedMemorySize':49152,'maxThreadsNum':1024},
            structural_features={'numAxisUploads':1,'kernels':[{'stageRadix':[8,8,4], 'localSize':[8,2,1], 'groupedBatch':2}]},
            precompile_kernels=[{'raw_descriptor':{'fftDim':256,'complexSize':8,'axisSwapped':0,'registerBoost':1,'registers_per_radix':[0,0,8,0,8,0,0,0,8]+[0]*24,'min_logical_registers_per_thread':8,'logical_registers_per_thread':8}}])
        rows=list(enum.candidates(result))
        self.assertEqual(len({x['candidate_id'] for x in rows}),len(rows))
        requests=[x['request'] for x in rows]
        self.assertIn({'radix':[8,2,2,2,4], 'grouped_batch':1, 'block':[8,1,1]}, requests)
        self.assertIn({'radix':[8,8,4], 'grouped_batch':2, 'block':[2,8,1]}, requests)
        self.assertTrue(all(x['status'].startswith('pending_') for x in rows))
        self.assertTrue(all(__import__('math').prod(x['radix']) == 256 for x in requests))
        result['structural_features']['numAxisUploads']=2
        with self.assertRaises(ValueError): list(enum.candidates(result))

    def test_duplicate_or_missing_anchor_fails(self):
        with self.assertRaises(RuntimeError): gen.replace_once('a a', 'a', 'b')
        with self.assertRaises(RuntimeError): gen.replace_once('a', 'x', 'b')

if __name__ == '__main__': unittest.main()
