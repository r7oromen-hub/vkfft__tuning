import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rotation = load('frontier_rotation_prepare_test', HERE / 'prepare.py')
frontier = load('frontier_rotation_base_test', STUDY / 'frontier_search/prepare_headers.py')


class PrepareTests(unittest.TestCase):
    def test_generated_frontier_rotation_preserves_inputs_and_records_full_hashes(self):
        original = REPO / 'VkFFT/vkFFT'
        instrumentation = STUDY / 'instrumentation'
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            generated, output, legacy_output = root / 'frontier', root / 'rotation', root / 'legacy'
            frontier.prepare(original, generated / 'include')
            before = {name: rotation.tree_inventory(path) for name, path in
                      (('frontier_generated', generated), ('original_vkfft', original), ('instrumentation', instrumentation))}
            legacy_provenance = rotation.load_rotation_v2().prepare(generated, legacy_output)
            provenance = rotation.prepare(generated, output, original, instrumentation)
            info = provenance['frontier_rotation']
            self.assertEqual(info['schema'], 'frontier-rotation-generated/v1')
            self.assertEqual(info['device_validation'], 'not performed')
            self.assertEqual(info['inputs'], before)
            for name, digest in info['output_files'].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(json.loads((output / 'rotation_provenance.json').read_text(encoding='utf8')), provenance)
            self.assertEqual(before['frontier_generated'], rotation.tree_inventory(generated))
            text = (output / 'rotating_benchmark.cpp').read_text(encoding='utf8')
            old = (legacy_output / 'rotating_benchmark.cpp').read_text(encoding='utf8')
            runtime_query = ('cudaKernelNodeParams params{};\n'
                             '                cuda_check(cudaGraphKernelNodeGetParams(nodes[i],&params), "rotation graph kernel parameters");')
            driver_query = ('CUDA_KERNEL_NODE_PARAMS params{};\n'
                            '                driver_check(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(nodes[i]),&params), "rotation graph kernel parameters");')
            self.assertEqual(old.count(runtime_query), 1)
            self.assertEqual(text, old.replace(runtime_query, driver_query, 1))
            self.assertEqual(provenance['frontier_legacy_rotating_benchmark_sha256'], legacy_provenance['rotating_benchmark.cpp'])
            self.assertEqual(provenance['rotating_benchmark.cpp'], hashlib.sha256((output/'rotating_benchmark.cpp').read_bytes()).hexdigest())
            self.assertIn('if(!params.kernelParams || !params.kernelParams[0] || !params.kernelParams[1])', text)
            self.assertEqual(text.count('rotation_graph_dependencies(nodes[i]'), 2)
            self.assertNotIn('cudaGraphNodeGetDependencies(nodes[i]', text)
            for marker in ('rotation_device_l2_bytes', 'rotation_pool_total_bytes', 'rotation_pool_slots',
                           'rotation_graph_kernel_arguments_and_order_verified',
                           'rotation_initial_validation', 'rotation_final_validation', 'input_unchanged'):
                self.assertIn(marker, text)
            cmake = (output / 'CMakeLists.txt').read_text(encoding='utf8')
            self.assertLess(cmake.index('PRIVATE "${VKFFT_INCLUDE_ROOT}"'),
                            cmake.index('PRIVATE "${ROTATION_GENERATED_ROOT}"'))
            with self.assertRaisesRegex(ValueError, 'new or empty'):
                rotation.prepare(generated, output, original, instrumentation)

    def test_rejects_tampering_and_output_inside_input(self):
        original = REPO / 'VkFFT/vkFFT'
        instrumentation = STUDY / 'instrumentation'
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            generated = root / 'frontier'
            frontier.prepare(original, generated / 'include')
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                rotation.prepare(generated, generated / 'new-rotation', original, instrumentation)
            path = generated / 'include/candidate.hpp'
            path.write_text(path.read_text(encoding='utf8') + '\n// tampered\n', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                rotation.prepare(generated, root / 'rotation', original, instrumentation)
            self.assertFalse((root / 'rotation').exists())


if __name__ == '__main__':
    unittest.main()
