import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import frontier_postscreen_phase as phase


class PostscreenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.source_run = 'breadth-004'
        self.run_id = 'postscreen-001'
        self.commit = 'a' * 40
        self.parent = 'b' * 40
        self.source_commit = 'c' * 40
        self.protocol = self.put('experiments/protocol.json', '{}')
        self.gate = self.put('experiments/gate.json', '{}')
        self.admission = self.root / 'experiments/artifacts/admission'
        self.original = self.put('experiments/artifacts/admission/build/vkfft_frontier_original', 'original')
        self.expanded = self.put('experiments/artifacts/admission/build/vkfft_frontier_v2', 'expanded')
        self.source = self.root / 'experiments/artifacts' / self.source_run
        self.output = self.root / 'experiments/artifacts' / self.run_id
        self.output.mkdir(parents=True)
        self.gpu = 'GPU-example'
        self.admission_contract = {'binaries': {'original': phase.campaign_inputs.sha(self.original),
                                                 'expanded': phase.campaign_inputs.sha(self.expanded)},
                                   'vkfft_commit': 'd' * 40,
                                   'generated_provenance': {'sha256': 'e' * 64}}
        self.contract = dict(copy.deepcopy(self.admission_contract), protocol='frontier-execution/v1',
                             gpu=self.gpu, execution_commit=self.source_commit,
                             source_files={str(self.protocol): phase.campaign_inputs.sha(self.protocol),
                                           str(self.gate): phase.campaign_inputs.sha(self.gate)})
        self.context = {'protocol': self.protocol, 'gate': self.gate, 'gpu': self.gpu,
                        'admission_dir': self.admission, 'admission_contract': self.admission_contract}
        for kind, manifest_dir in (('defaults', 'defaults_manifests'), ('search', 'manifests')):
            for name in phase.LOG_NAMES:
                contents = (json.dumps({'record_type': 'session_start', 'contract': self.contract}) + '\n'
                            if name == 'rows' else '{}\n')
                self.put(f'experiments/artifacts/{self.source_run}/{kind}/{name}.jsonl', contents)
            self.put(f'experiments/artifacts/{self.source_run}/{manifest_dir}/n128_b1.manifest.jsonl', '{}\n')
        self.record_path = self.put(f'experiments/records/{self.source_run}.json', '{}')
        self.record = {'id': self.source_run, 'execution_commit': self.source_commit}
        self.reindex()

        info = {'path': self.protocol.relative_to(self.root).as_posix(),
                'bytes': self.protocol.stat().st_size, 'sha256': phase.campaign_inputs.sha(self.protocol)}
        self.declaration = {'schema': 'git-experiment/v1', 'id': self.run_id, 'status': 'declared',
                            'execution_commit': None, 'parent_code_commit': self.parent, 'protocol': info}
        self.put(f'experiments/records/{self.run_id}.json', json.dumps(self.declaration))
        self.execution = dict(self.declaration, execution_commit=self.commit, started_at='2026-01-01')
        self.put(f'experiments/artifacts/{self.run_id}/execution.json', json.dumps(self.execution))
        self.env = {'EXPERIMENT_ID': self.run_id, 'EXPERIMENT_CODE_COMMIT': self.commit,
                    'EXPERIMENT_ARTIFACT_DIR': str(self.output)}

    def put(self, rel, content):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf8')
        return path

    def reindex(self):
        self.record['artifacts'] = [
            {'path': p.relative_to(self.root).as_posix(), 'bytes': p.stat().st_size,
             'sha256': phase.campaign_inputs.sha(p)}
            for p in sorted(self.source.rglob('*')) if p.is_file()]

    def load_source(self):
        with mock.patch.object(phase.campaign_inputs, 'load_completed_run',
                               return_value=(self.record, self.source, self.record_path)):
            return phase.source_inputs(self.root, self.context, self.source_run)

    def registered(self):
        def git(root, *args):
            return self.commit if args == ('rev-parse', 'HEAD') else self.parent
        with mock.patch.dict(phase.os.environ, self.env), \
             mock.patch.object(phase.campaign_inputs, 'git', side_effect=git), \
             mock.patch.object(phase.campaign_inputs, 'tracked'):
            return phase.registered_output(self.root, self.protocol, self.source_run)

    def test_registered_execution_matches_declaration(self):
        self.assertEqual(self.registered(), self.output)

    def test_registered_rejects_changed_execution_and_source_output(self):
        execution_file = self.output / 'execution.json'
        for field, bad in (('execution_commit', 'f' * 40), ('started_at', ''), ('protocol', {})):
            with self.subTest(field=field):
                value = dict(self.execution, **{field: bad})
                execution_file.write_text(json.dumps(value), encoding='utf8')
                with self.assertRaises(ValueError):
                    self.registered()
        execution_file.write_text(json.dumps(self.execution), encoding='utf8')
        self.env['EXPERIMENT_ARTIFACT_DIR'] = str(self.source)
        with self.assertRaises(ValueError):
            self.registered()
        self.env['EXPERIMENT_ID'] = self.source_run
        with self.assertRaises(ValueError):
            self.registered()

    def test_source_inventory_and_manifests_are_verified(self):
        _, _, sessions = self.load_source()
        self.assertEqual([p.name for p in sessions['defaults']['manifests']], ['n128_b1.manifest.jsonl'])
        self.assertEqual([p.name for p in sessions['search']['manifests']], ['n128_b1.manifest.jsonl'])
        (self.source / 'search/guard.jsonl').write_text('changed\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'artifact missing or changed'):
            self.load_source()

    def test_source_contract_mismatches_are_rejected_even_when_rehashed(self):
        for field in ('gpu', 'execution_commit', 'binaries', 'vkfft_commit', 'generated_provenance', 'source_files'):
            with self.subTest(field=field):
                contract = copy.deepcopy(self.contract)
                contract[field] = {} if isinstance(contract[field], dict) else 'wrong'
                rows = self.source / 'search/rows.jsonl'
                rows.write_text(json.dumps({'record_type': 'session_start', 'contract': contract}) + '\n', encoding='utf8')
                self.reindex()
                with self.assertRaises(ValueError):
                    self.load_source()

    def run_main(self, side_effect=None):
        record, record_path, sessions = self.load_source()
        with mock.patch.object(phase, 'ROOT', self.root), \
             mock.patch.object(phase.campaign_inputs, 'load_campaign', return_value=self.context), \
             mock.patch.object(phase, 'registered_output', return_value=self.output), \
             mock.patch.object(phase, 'source_inputs', return_value=(record, record_path, sessions)), \
             mock.patch.object(phase, 'run_step', side_effect=side_effect) as step:
            phase.main(['--source-run', self.source_run, '--protocol', str(self.protocol),
                        '--admission-run', 'admission', '--gate', str(self.gate)])
            return step.call_args_list, sessions

    def test_main_runs_exact_offline_commands_in_order(self):
        calls, sessions = self.run_main()
        expected = []
        for kind in ('defaults', 'search'):
            cmd = [sys.executable, str(phase.AUDIT)]
            for name in phase.LOG_NAMES:
                cmd += ['--' + name, str(sessions[kind]['paths'][name])]
            cmd += ['--original', str(self.original), '--expanded', str(self.expanded)]
            for path in sessions[kind]['manifests']:
                cmd += ['--manifest', str(path)]
            cmd += ['--gpu', self.gpu, '--output', str(self.output / (kind + '_coverage.json'))]
            expected.append(mock.call(self.output, kind + '_coverage', cmd))
        coverage = str(self.output / 'search_coverage.json')
        rows = str(sessions['search']['paths']['rows'])
        for name, script, extra in (
                ('analysis', 'analyze.py', []),
                ('old_space_partition', 'old_space_partition.py', []),
                ('search_costs', 'search_costs.py', ['--budget', str(sessions['search']['paths']['budget'])]),
                ('aggregate', 'aggregate_confirmed.py', [])):
            expected.append(mock.call(self.output, name, [sys.executable, str(phase.ANALYSIS / script),
                '--coverage', coverage, '--rows', rows, *extra, '--output', str(self.output / (name + '.json'))]))
        self.assertEqual(calls, expected)

    def test_coverage_failure_stops_subsequent_steps(self):
        record, record_path, sessions = self.load_source()
        for index, kind in enumerate(('defaults', 'search')):
            with self.subTest(kind=kind), \
                 mock.patch.object(phase, 'ROOT', self.root), \
                 mock.patch.object(phase.campaign_inputs, 'load_campaign', return_value=self.context), \
                 mock.patch.object(phase, 'registered_output', return_value=self.output), \
                 mock.patch.object(phase, 'source_inputs', return_value=(record, record_path, sessions)), \
                 mock.patch.object(phase, 'run_step', side_effect=[None] * index + [subprocess.CalledProcessError(1, 'audit')]) as step:
                with self.assertRaises(subprocess.CalledProcessError):
                    phase.main(['--source-run', self.source_run])
                self.assertEqual(step.call_count, index + 1)
                self.assertEqual(step.call_args.args[1], kind + '_coverage')


if __name__ == '__main__':
    unittest.main()
