"""Regression against a complete independently collected historical comparison."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'direct_candidates/results/joint_newhost_20261030'
STEM = 'original_compare_n256_b4096_01'
FILES = [STEM + suffix for suffix in ('.jsonl', '.guard.jsonl', '.incidents.jsonl')] + [
    'vkfft_direct_candidate', 'vkfft_original_control', 'build_original.log',
    'original_compile_commands.json', 'original_compile_dependencies.d']


class AuditRegression(unittest.TestCase):
    def test_historical_evidence_and_mismatch_rejection(self):
        if not all((SOURCE / name).is_file() for name in FILES):
            self.skipTest('complete historical evidence not present')
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            for name in FILES:
                shutil.copyfile(SOURCE / name, root / name)
            command = [sys.executable, str(HERE/'audit_original_compare_parameterized.py'), str(root),
                       '--stem', STEM, '--output-stem', 'independent_audit', '--n', '256', '--batch', '4096',
                       '--graph-repeats', '512', '--winner-radix', '16,16', '--winner-group', '1',
                       '--winner-block', '16,1,1', '--default-radix', '16,16', '--default-group', '8',
                       '--default-block', '8,16,1']
            passed = subprocess.run(command, capture_output=True, text=True, encoding='utf8')
            self.assertEqual(passed.returncode, 0, passed.stderr)
            result = json.loads((root/'independent_audit.json').read_text(encoding='utf8'))
            prior = json.loads((SOURCE/'original_compare_n256_b4096_audit.json').read_text(encoding='utf8'))
            self.assertEqual(result['expected']['batch'], 4096)
            self.assertEqual(len(result['winner_over_original']), 10)
            self.assertEqual(result['winner_over_original'], prior['winner_over_original'])
            self.assertEqual(result['joint_over_original'], prior['joint_over_original'])
            self.assertIn('N=256、batch=4096', (root/'independent_audit_zh.md').read_text(encoding='utf8'))
            # An already-existing output is rejected, and wrong independent expected parameters fail.
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            for key, wrong in (('--batch', '1024'), ('--graph-repeats', '1024'), ('--winner-group', '4'), ('--default-group', '4')):
                altered = command.copy()
                altered[altered.index('--output-stem')+1] = 'wrong'
                altered[altered.index(key)+1] = wrong
                self.assertNotEqual(subprocess.run(altered, capture_output=True).returncode, 0, key)
                self.assertFalse((root/'wrong.json').exists(), key)
            records = [json.loads(line) for line in (root/(STEM+'.jsonl')).read_text(encoding='utf8').splitlines()]
            records[-1]['winner_over_original_median_ratios'].pop()
            (root/(STEM+'.jsonl')).write_text('\n'.join(json.dumps(row) for row in records)+'\n',encoding='utf8')
            altered = command.copy(); altered[altered.index('--output-stem')+1] = 'truncated'
            self.assertNotEqual(subprocess.run(altered, capture_output=True).returncode, 0)
            self.assertFalse((root/'truncated.json').exists())


if __name__ == '__main__':
    unittest.main()
