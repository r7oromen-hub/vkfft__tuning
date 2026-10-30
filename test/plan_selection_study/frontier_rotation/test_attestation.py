import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class AttestationTests(unittest.TestCase):
    def test_exclusive_attestation_binds_inputs_and_binary_hashes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            (root/'rotation_provenance.json').write_text('{}', encoding='utf8')
            original = root/'original'; expanded = root/'expanded'
            original.write_bytes(b'o'); expanded.write_bytes(b'e')
            output = root/'build.json'
            command = [sys.executable, str(HERE/'make_build_attestation.py'), '--code-commit', 'a'*40,
                       '--vkfft-commit', 'b'*40, '--rotation-root', str(root), '--original', str(original),
                       '--expanded', str(expanded), '--build-command', 'cmake --build build', '--output', str(output)]
            subprocess.run(command, check=True, capture_output=True)
            doc = json.loads(output.read_text(encoding='utf8'))
            self.assertEqual(doc['binaries']['original'], hashlib.sha256(b'o').hexdigest())
            self.assertEqual(doc['binaries']['expanded'], hashlib.sha256(b'e').hexdigest())
            self.assertEqual(doc['code_commit'], 'a'*40)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
