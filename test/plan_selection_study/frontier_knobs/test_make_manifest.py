import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('make_knobs_manifest_test', HERE / 'make_manifest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.spec = {'workload': {'n': 256, 'batch': 16},
                     'plans': [{'radix': [16, 16], 'logical_registers': 16, 'grouped_batch': 4}],
                     'lut_values': [-1, 0, 1], 'coalesced_values': [0, 16, 32, 64, 128]}

    def test_product_and_versioned_ids(self):
        result = module.build(self.spec)
        self.assertEqual(len(result['candidates']), 15)
        self.assertEqual(result['schema'], 'frontier-knobs-screen/v1')
        self.assertEqual(len({r['candidate_id'] for r in result['candidates']}), 15)
        for row in result['candidates']:
            self.assertEqual(row['candidate_id'], module.requests.candidate_id(256, 16, row['request']))

    def test_no_silent_dropping(self):
        bad = dict(self.spec, lut_values=[0, 0])
        with self.assertRaisesRegex(ValueError, 'duplicate knob'):
            module.build(bad)
        bad = dict(self.spec, plans=[self.spec['plans'][0]] * 3)
        with self.assertRaisesRegex(ValueError, 'limit'):
            module.build(bad)
        bad = dict(self.spec, plans=[{'radix': [16, 16], 'logical_registers': 8, 'grouped_batch': 4}])
        with self.assertRaisesRegex(ValueError, 'radix_exceeds_registers'):
            module.build(bad)

    def test_cli_exclusive_write(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            source = Path(tmp)/'spec.json'; output = Path(tmp)/'manifest.json'
            source.write_text(json.dumps(self.spec), encoding='utf8')
            command = [sys.executable, str(HERE/'make_manifest.py'), '--spec', str(source), '--output', str(output)]
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(len(json.loads(output.read_text(encoding='utf8'))['candidates']), 15)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
