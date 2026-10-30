import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frontier_v2_prepare', HERE / 'prepare_headers.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


class V2GenerationTests(unittest.TestCase):
    def test_isolated_v2_generation_and_provenance(self):
        source = HERE.parents[2] / 'VkFFT/vkFFT'
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            output = Path(tmp) / 'include'
            v2.prepare(source, output)
            candidate = (output / 'candidate.hpp').read_text(encoding='utf8')
            self.assertIn('r.regs > 256', candidate)
            self.assertIn('for (int radix = 0; radix < 33; ++radix)', candidate)
            self.assertIn('r.regs % radix == 0 ? r.regs : 0', candidate)
            scheduler = (output / 'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h').read_text(encoding='utf8')
            self.assertIn('expanded_rejection::record("scheduler_register_table_changed")', scheduler)
            self.assertIn('expanded_candidate::consistent(', scheduler)
            benchmark = (Path(tmp) / 'expanded_benchmark.cpp').read_text(encoding='utf8')
            self.assertIn('expanded_resource_validation', benchmark)
            self.assertIn('expanded_rejection_reason', benchmark)
            provenance = json.loads((Path(tmp) / 'expanded_provenance.json').read_text(encoding='utf8'))
            for rel, digest in provenance['expanded'].items():
                self.assertEqual(hashlib.sha256((Path(tmp) / rel).read_bytes()).hexdigest(), digest)
            self.assertEqual(provenance['frontier_v2']['generated_rejection_header_sha256'],
                             hashlib.sha256((output / 'rejection.hpp').read_bytes()).hexdigest())
            with self.assertRaises(ValueError):
                v2.prepare(source, output)


if __name__ == '__main__':
    unittest.main()
