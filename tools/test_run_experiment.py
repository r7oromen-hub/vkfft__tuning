import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run_experiment import run


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'user.email', 'test@example.invalid')
        (self.root / '.gitignore').write_text('experiments/artifacts/\n')
        (self.root / 'protocol.json').write_text('{}\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True).strip()

    def launch(self, identity, code):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(self.root, identity, self.root / 'protocol.json', [sys.executable, '-c', code])

    def test_success_commit_and_artifacts(self):
        self.assertEqual(self.launch('ok', 'print(42)'), 0)
        record = json.loads((self.root / 'experiments/records/ok.json').read_text())
        self.assertEqual(record['execution_commit'], self.git('rev-parse', 'HEAD^'))
        self.assertEqual(record['status'], 'completed')
        self.assertTrue(record['artifacts'])
        self.assertEqual(self.git('status', '--porcelain'), '')
        with self.assertRaises(ValueError):
            self.launch('ok', 'print(42)')

    def test_failure_preserves_result_commit(self):
        self.assertEqual(self.launch('failed', 'raise SystemExit(7)'), 7)
        record = json.loads((self.root / 'experiments/records/failed.json').read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(self.git('rev-list', '--count', 'HEAD'), '3')

    def test_dirty_source_rejected(self):
        (self.root / 'protocol.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError):
            self.launch('dirty', 'print(42)')

    def test_source_mutation_invalidates_run(self):
        self.assertEqual(self.launch('mutated', "from pathlib import Path; Path('protocol.json').write_text('changed')"), 2)
        record = json.loads((self.root / 'experiments/records/mutated.json').read_text())
        self.assertEqual(record['status'], 'source_changed_during_run')


if __name__ == '__main__':
    unittest.main()
