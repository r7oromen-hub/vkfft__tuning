import tempfile
from pathlib import Path
import unittest
import prepare_headers
import prepare_rotation

HERE=Path(__file__).resolve().parent


class RotationGenerationTest(unittest.TestCase):
    def test_generated_graph_and_verification_boundaries(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root=Path(temp); generated=root/'expanded'; output=root/'rotation'
            prepare_headers.prepare(HERE.parents[2]/'VkFFT/vkFFT',generated/'include')
            before=(generated/'expanded_benchmark.cpp').read_bytes()
            metadata=prepare_rotation.prepare(generated,output)
            self.assertEqual((generated/'expanded_benchmark.cpp').read_bytes(),before)
            self.assertEqual(len(metadata['expanded_benchmark_sha256']),64)
            code=(output/'rotating_benchmark.cpp').read_text(encoding='utf8')
            self.assertIn('cudaGraphKernelNodeGetParams',code)
            self.assertIn('cudaGraphNodeGetDependencies',code)
            self.assertIn('launch.inputBuffer=&pool.inputs[slot];launch.buffer=&pool.outputs[slot];',code)
            self.assertIn('rotation_graph_kernel_arguments_and_order_verified',code)
            self.assertIn('rotation_distinct_input_representatives_verified',code)
            self.assertIn('rotation_initial_validation',code)
            self.assertIn('rotation_final_validation',code)
            self.assertLess(code.index('fill_host(i);'),code.index('cudaStreamBeginCapture('))
            self.assertLess(code.index('cudaMemsetAsync(output,0xff'),code.index('cudaStreamBeginCapture('))
            self.assertIn('phase="rotation_final_validation";poison();execute_graph();',code)
            timing=code.split('phase="execution";',1)[1].split('phase="rotation_final_validation";',1)[0]
            self.assertNotIn('poison()',timing)
            self.assertIn('rotation original control forbids expanded request',code)
            with self.assertRaises(ValueError): prepare_rotation.prepare(generated,output)
            with self.assertRaises(ValueError): prepare_rotation.prepare(generated,generated/'nested')


if __name__=='__main__': unittest.main()
