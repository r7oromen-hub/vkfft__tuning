import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('knobs_runner_tests', HERE / 'runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Ledger:
    def __init__(self, maximum=None): self.work = []; self.finished = []; self.maximum = maximum
    def start(self, identity, expected):
        if self.maximum is not None and len(self.work) >= self.maximum: return False
        self.work.append(identity); return True
    def finish(self, status): self.finished.append(status)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.req = {'radix': [16, 16], 'logical_registers': 16, 'grouped_batch': 4,
                    'use_lut': 1, 'coalesced_memory': 64}
        self.cid = runner.requests.candidate_id(256, 16, self.req)
        self.args = types.SimpleNamespace(gpu='GPU-test', original=Path('original'), expanded=Path('candidate'),
                                          timeout=2, output=Path('unused'))
        self.contract = {'binaries': {'original': 'original_hash', 'expanded': 'candidate_hash'}}
        self.rows = [(256, 16, [{'candidate_id': self.cid, 'request': self.req}])]
        self.events, self.calls = [], []

    def fake_one(self, args, workload, candidate, phase, round_no, order):
        name, env = candidate
        self.calls.append((name, workload.copy(), env.copy(), phase, round_no, order))
        original = name == 'original'
        result = {'role': 'original' if original else 'candidate',
                  'raw_interval_ms_samples': [2.5], 'exec_ms_samples': [1.0 if original else .8],
                  'warmup_actual': 5, 'warmup_wall_ms': 1000,
                  'precompile_kernels': [{'canonical_descriptor': {}, 'source_sha256': 'source',
                                          'source_bytes': 10, 'compiler_input': {}}]}
        if getattr(self, 'fail_validation', False) and phase == 'validation':
            result['single_forward_correctness'] = {'passed': False}
        if getattr(self, 'reject_describe', False) and phase == 'describe':
            result.update(status='error', failure_phase='initialize', error='initializeVkFFT failed with code 3002',
                          expanded_rejection_reason='splitter_thread_limit')
            return {'status': 'failed', 'returncode': 1, 'guard_clean': True, 'guard_gpu_uuid': 'GPU-test',
                    'stderr': 'VKFFT_FRONTIER_REJECTION:{"reason":"splitter_thread_limit"}\n', 'result': result}
        return {'status': 'ok', 'returncode': 0, 'result': result}

    def run_screen(self, ledger):
        with patch.dict(os.environ, {'VKFFT_STUDY_GUARD_UUID': 'GPU-test', 'VKFFT_STUDY_GUARD_ID': 'guard'}), \
             patch.object(runner, 'run_one', side_effect=self.fake_one), \
             patch.object(runner.v2, 'append', side_effect=lambda path, row: self.events.append(row)), \
             patch.object(runner.v2, 'representable', return_value=({'radix': [16, 16], 'logical_registers': 16, 'grouped_batch': 4}, None)), \
             patch.object(runner.validation, 'check_execute', side_effect=lambda row, *a, **kw: row['result']), \
             patch.object(runner.validation, 'check_describe', side_effect=lambda row, *a, **kw: row['result']), \
             patch.object(runner.validation, 'check_matched_identity'), \
             patch.object(runner.validation.comparison, 'compare'), \
             patch.object(runner.search, 'source_signature', side_effect=lambda result, **kw: result['role']):
            return runner.screen(self.args, ledger, self.contract, self.rows[0])

    def test_full_small_screen_uses_baseline_and_matched_controls(self):
        ledger = Ledger(); self.run_screen(ledger)
        phases = [call[3] for call in self.calls]
        self.assertEqual(phases.count('baseline'), 1)
        self.assertEqual(phases.count('calibration'), 1)
        self.assertEqual(phases.count('matched_identity'), 3)
        self.assertEqual(phases.count('describe'), 1)
        validations = [call for call in self.calls if call[3] == 'validation']
        self.assertEqual({(call[1]['pattern'], call[1]['seed']) for call in validations},
                         {(p, s) for p in runner.PATTERNS for s in runner.SEEDS})
        timings = [call for call in self.calls if call[3] == 'timing']
        self.assertEqual(len(timings), 6)
        for name, workload, env, _, _, _ in timings:
            if name == 'original': self.assertEqual((workload['knob_args'], env), ([], {}))
            else: self.assertEqual(workload['knob_args'], ['--use-lut', '1', '--coalesced-memory', '64'])
        self.assertEqual(len(ledger.work), len(self.calls))
        self.assertEqual(len(ledger.finished), len(self.calls))
        self.assertEqual(len([e for e in self.events if e.get('record_type') == 'screen_pair']), 3)
        self.assertEqual(self.events[-1]['status'], 'screen_complete')

    def test_numerical_failure_stops_without_further_gpu_work(self):
        self.fail_validation = True; ledger = Ledger()
        with self.assertRaises(runner.v2.NumericalFailure): self.run_screen(ledger)
        self.assertEqual(self.calls[-1][3], 'validation')
        self.assertEqual(sum(c[3] == 'validation' for c in self.calls), 1)
        self.assertEqual(self.events[-1]['record_type'], 'numerical_stop')
        self.assertEqual(len(ledger.work), len(ledger.finished))

    def test_scheduler_rejection_is_specific_and_does_not_run_candidate(self):
        self.reject_describe = True; self.run_screen(Ledger())
        self.assertEqual(self.calls[-1][3], 'describe')
        self.assertEqual(self.events[-1]['status'], 'scheduler_rejected')
        self.assertEqual(self.events[-1]['reason'], 'splitter_thread_limit')
        row = self.fake_one(self.args, {}, ('x', {}), 'describe', 0, 0)
        row['stderr'] = ''
        self.assertIsNone(runner.scheduler_rejection(row, 'GPU-test'))

    def test_budget_denial_prevents_subprocess(self):
        with self.assertRaises(runner.v2.BudgetStop): self.run_screen(Ledger(maximum=1))
        self.assertEqual(len(self.calls), 1)

    def test_knobs_and_candidate_environment_are_transmitted(self):
        workload = {'n': 256, 'batch': 16, 'seed': 42, 'pattern': 'random', 'graph_repeats': 1,
                    'warmup': 0, 'warmup_ms': 0, 'repeats': 1, 'study_mode': 'execute',
                    'knob_args': ['--use-lut', '-1', '--coalesced-memory', '64']}
        env = runner.requests.invocation(256, 16, self.req)['env']
        self.args.exe = Path('candidate'); self.args.exe_sha256 = 'digest'; self.args.session_id = 'session'
        process = types.SimpleNamespace(returncode=0, stdout='{"status":"ok"}\n', stderr='')
        with patch.object(runner.v2, 'sha', return_value='digest'), \
             patch.object(runner.subprocess, 'run', return_value=process) as invoke, \
             patch.dict(os.environ, {'VKFFT_DIRECT_RADIX': 'stale'}):
            row = runner.run_one(self.args, workload, ('candidate', env), 'validation', 0, 0)
        command = invoke.call_args.args[0]; child_env = invoke.call_args.kwargs['env']
        self.assertEqual(command[-4:], ['--use-lut', '-1', '--coalesced-memory', '64'])
        self.assertNotIn('VKFFT_DIRECT_RADIX', child_env)
        self.assertEqual(child_env['VKFFT_EXPANDED_REGS'], '16')
        self.assertEqual(row['command'], command)

    def test_manifest_and_build_attestation_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path = Path(tmp) / 'manifest.json'
            manifest = {'schema': 'frontier-knobs-screen/v1', 'workload': {'n': 256, 'batch': 16},
                        'candidates': [{'candidate_id': self.cid, 'request': self.req}]}
            path.write_text(json.dumps(manifest), encoding='utf8')
            self.assertEqual(runner.load_manifest(path)[0:2], (256, 16))
            manifest['candidates'].append(manifest['candidates'][0])
            path.write_text(json.dumps(manifest), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'duplicate'): runner.load_manifest(path)
            path.write_text('{}', encoding='utf8')
            with self.assertRaises(ValueError): runner.verify_attestation(path, 'a', 'b', 'c', {})
            attestation = {'schema': 'frontier-knobs-build/v1', 'code_commit': 'parent', 'vkfft_commit': 'vk',
                           'generated_provenance_sha256': 'generated', 'binaries': {'original': 'x', 'expanded': 'y'},
                           'build_commands': ['cmake --build build']}
            path.write_text(json.dumps(attestation), encoding='utf8')
            self.assertEqual(runner.verify_attestation(path, 'parent', 'vk', 'generated', attestation['binaries'])['document'], attestation)
            with self.assertRaises(ValueError):
                runner.verify_attestation(path, 'different', 'vk', 'generated', attestation['binaries'])

    def test_declaration_binds_code_before_run_commit(self):
        record = {'schema': 'git-experiment/v1', 'id': 'case', 'parent_code_commit': 'parent', 'status': 'declared'}
        with patch.object(runner.v2, 'git', side_effect=['parent', 'experiments/records/case.json', json.dumps(record)]):
            self.assertEqual(runner.verify_declaration('root', 'run_commit', 'case'), 'parent')
        with patch.object(runner.v2, 'git', side_effect=['parent', 'other.py']):
            with self.assertRaisesRegex(ValueError, 'single run declaration'):
                runner.verify_declaration('root', 'run_commit', 'case')


if __name__ == '__main__':
    unittest.main()
