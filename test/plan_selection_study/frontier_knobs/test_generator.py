import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frontier_knobs_generator_test', HERE / 'prepare_headers.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class GeneratorTests(unittest.TestCase):
    def test_isolated_generation_and_full_provenance(self):
        source = HERE.parents[2] / 'VkFFT/vkFFT'
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            output = Path(tmp) / 'include'
            generator.prepare(source, output)
            benchmark = (Path(tmp) / 'expanded_benchmark.cpp').read_text(encoding='utf8')
            self.assertIn('o.aim || o.boost || o.nonpow2 || o.lut4 || o.swap2 || o.swap3)', benchmark)
            self.assertNotIn('o.boost || o.lut || o.nonpow2 || o.coalesced', benchmark)
            self.assertIn('resources.app.configuration.useLUT != o.lut', benchmark)
            self.assertIn('resources.app.configuration.coalescedMemory != (pfUINT)o.coalesced', benchmark)
            self.assertIn('expanded_resource_validation', benchmark)
            plan = (output / 'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h').read_text(encoding='utf8')
            self.assertIn('expanded_candidate::layout_matches', plan)
            self.assertIn('expanded_rejection_reason', benchmark)
            candidate = (output / 'candidate.hpp').read_text(encoding='utf8')
            self.assertIn('boost != 1', candidate)
            self.assertIn('r.regs > 256', candidate)
            provenance = json.loads((Path(tmp) / 'expanded_provenance.json').read_text(encoding='utf8'))
            self.assertEqual(provenance['frontier_knobs']['generator_sha256'], generator.sha(HERE / 'prepare_headers.py'))
            self.assertEqual(provenance['frontier_knobs']['parent_v2_generator_sha256'], generator.sha(generator.V2 / 'prepare_headers.py'))
            self.assertNotEqual(provenance['frontier_knobs']['parent_generated_benchmark_sha256'], generator.sha(Path(tmp) / 'expanded_benchmark.cpp'))
            for rel, digest in provenance['expanded'].items():
                self.assertEqual(hashlib.sha256((Path(tmp) / rel).read_bytes()).hexdigest(), digest)
            with self.assertRaises(ValueError):
                generator.prepare(source, output)

    def test_existing_sibling_files_are_not_overwritten(self):
        source = HERE.parents[2] / 'VkFFT/vkFFT'
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            sentinel = Path(tmp) / 'expanded_benchmark.cpp'
            sentinel.write_text('keep', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'already contains'):
                generator.prepare(source, Path(tmp) / 'include')
            self.assertEqual(sentinel.read_text(encoding='utf8'), 'keep')

    def test_generated_headers_match_v2_exactly(self):
        source = HERE.parents[2] / 'VkFFT/vkFFT'
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            parent = Path(tmp) / 'parent'
            knobs = Path(tmp) / 'knobs'
            generator.v2.prepare(source, parent / 'include')
            generator.prepare(source, knobs / 'include')
            parent_prov = json.loads((parent / 'expanded_provenance.json').read_text(encoding='utf8'))
            knobs_prov = json.loads((knobs / 'expanded_provenance.json').read_text(encoding='utf8'))
            for rel, digest in parent_prov['expanded'].items():
                if rel != 'expanded_benchmark.cpp':
                    self.assertEqual(knobs_prov['expanded'][rel], digest, rel)
            self.assertEqual(knobs_prov['frontier_knobs']['parent_generated_benchmark_sha256'],
                             generator.sha(parent / 'expanded_benchmark.cpp'))


if __name__ == '__main__':
    unittest.main()
