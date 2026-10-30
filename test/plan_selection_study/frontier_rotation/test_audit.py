import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load('frontier_rotation_raw_audit_test', HERE/'audit.py')
fixtures = load('frontier_rotation_raw_pool_fixture', HERE/'test_pool_validation.py')


class FakeLedger:
    def __init__(self, contract):
        self.contract = contract; self.rows = [{'event': 'start', 'state': self.state(None)}]; self.inflight = None
    def state(self, inflight): return {'contract': self.contract, 'inflight': inflight, 'last': 100}
    def start(self, work, _expected):
        self.inflight = work; self.rows.append({'event': 'work_start', 'state': self.state(work)}); return True
    def finish(self, status):
        self.rows.append({'event': 'work_finish', 'status': status, 'identity': self.inflight, 'state': self.state(None)})
        self.inflight = None


class AuditTests(unittest.TestCase):
    def test_context_rechecks_campaign_build_and_each_deadline(self):
        campaign = {'start': 100, 'deadline': 1000, 'boot': 'boot'}
        build = {'binaries': {'original': 'o', 'expanded': 'e'}}
        contract = {'execution_commit': 'execution', 'code_commit': 'parent', 'source_files': {},
                    'rotation_build': build, 'campaign': campaign, 'deadline_monotonic': 800,
                    'stage_seconds': 900, 'timeout': 100}
        def state(last, inflight=None):
            return {'contract': contract, 'boot': 'boot', 'start': 100, 'seconds': 700,
                    'last': last, 'inflight': inflight}
        budget = [{'event': 'start', 'state': state(100)},
                  {'event': 'work_start', 'state': state(200, {'work': 1})},
                  {'event': 'work_finish', 'state': state(210)}]
        args = types.SimpleNamespace(rotation_experiment_id='case', rotation_original='o', rotation_expanded='e',
            rotation_root='r', generated_root='g', original_include_root='i', instrumentation_root='j',
            build_attestation='a', compile_commands='c', original_dependencies='d', campaign_record='cr',
            campaign_start='cs', admission_budget='ab', gpu='GPU-test')
        with patch.object(audit.runner, 'verify_declaration', return_value='parent'), \
             patch.object(audit.frontier, 'git', side_effect=lambda _root, *a: 'vk' if a[0] == 'rev-parse' else ''), \
             patch.object(audit.runner, 'verify_rotation_build', return_value=build), \
             patch.object(audit.runner, 'campaign', return_value=campaign):
            result = audit.verify_context('root', contract, budget, {'vkfft_commit': 'vk'}, args)
            self.assertEqual(result['campaign'], campaign)
            changed = copy.deepcopy(budget)
            changed[1]['state']['last'] = 700
            changed[2]['state']['last'] = 710
            with self.assertRaisesRegex(ValueError, 'reserve'):
                audit.verify_context('root', contract, changed, {'vkfft_commit': 'vk'}, args)
            changed = copy.deepcopy(budget)
            changed[1]['state']['boot'] = 'different'
            with self.assertRaisesRegex(ValueError, 'boot'):
                audit.verify_context('root', contract, changed, {'vkfft_commit': 'vk'}, args)
            changed = [budget[0], budget[2], budget[1]]
            with self.assertRaisesRegex(ValueError, 'alternating'):
                audit.verify_context('root', contract, changed, {'vkfft_commit': 'vk'}, args)

    def test_unsupported_pool_is_reported_uncovered_without_measurement(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp); original = root/'original'; expanded = root/'expanded'
            original.write_bytes(b'o'); expanded.write_bytes(b'e')
            case = {'n': 128, 'batch': 1, 'candidate_id': 'small', 'request': {'radix': [8, 16],
                    'logical_registers': 16, 'grouped_batch': 1}, 'rotation': None,
                    'hot_original_signature': 'original', 'hot_candidate_signature': 'candidate',
                    'status': 'unsupported_pool_or_graph_limit'}
            proof = {'plan_sha256': 'plan', 'cases': [case]}
            bins = {'original': original, 'expanded': expanded}
            contract = {'schema': 'frontier-rotation-session/v1', 'gpu': 'GPU-test', 'eligibility': proof,
                        'rotation_build': {'binaries': {k: audit.frontier.sha(v) for k, v in bins.items()}}, 'deadline_monotonic': 1000}
            args = types.SimpleNamespace(gpu='GPU-test', rotation_original=original, rotation_expanded=expanded,
                                         timeout=3, output=root/'rows')
            rows = []; ledger = FakeLedger(contract)
            with patch.dict(os.environ, {'VKFFT_STUDY_GUARD_ID': 'guard', 'VKFFT_STUDY_GUARD_UUID': 'GPU-test'}), \
                 patch.object(audit.runner.frontier, 'append', side_effect=lambda _path, row: rows.append(row)), \
                 patch.object(audit.runner.suite, 'run_one') as execute:
                audit.runner.run_session(args, ledger, contract, proof)
                execute.assert_not_called()
            with patch.object(audit.eligibility.audit.audit_guard, 'audit', return_value={'status': 'pass'}):
                result = audit.audit_rows(rows, [{}], [], ledger.rows, proof, 'GPU-test', bins)
            self.assertEqual((result['measured_count'], result['uncovered_count']), (0, 1))
            self.assertEqual(result['results'][0]['status'], 'uncovered_pool_or_graph_limit')

    def test_raw_ten_pairs_and_full_pool_failure_detection(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp); original = root/'original'; expanded = root/'expanded'
            original.write_bytes(b'o'); expanded.write_bytes(b'e')
            req = {'radix': [8, 4, 8], 'logical_registers': 8, 'grouped_batch': 4}
            case = {'n': 256, 'batch': 16, 'candidate_id': 'candidate', 'request': req,
                    'hot_original_signature': 'original', 'hot_candidate_signature': 'candidate',
                    'rotation': {'slots': 2, 'total_bytes': 131072, 'graph_repeats': 64, 'l2_bytes': 32768},
                    'l2_bytes': 32768, 'status': 'eligible'}
            proof = {'plan_sha256': 'plan', 'cases': [case]}
            hashes = {'original': audit.frontier.sha(original), 'expanded': audit.frontier.sha(expanded)}
            contract = {'schema': 'frontier-rotation-session/v1', 'gpu': 'GPU-test', 'eligibility': proof,
                        'rotation_build': {'binaries': hashes}, 'deadline_monotonic': 1000}
            args = types.SimpleNamespace(gpu='GPU-test', rotation_original=original, rotation_expanded=expanded,
                                         timeout=3, output=root/'rows')
            rows = []; ledger = FakeLedger(contract); calls = []
            def fake_run(_args, workload, candidate, phase, round_no, order):
                role = candidate[0] == 'original'
                row, _, _ = fixtures.fixture(role)
                result = row['result']
                raw = 2.0 if role else 1.5
                count = workload['repeats']
                result['raw_interval_ms_samples'] = [raw] * count
                result['exec_ms_samples'] = [raw / 64] * count
                calls.append(1)
                row.update(record_type='execution', candidate_id=candidate[0], workload=workload, phase=phase,
                           round=round_no, order=order, guard_run_id=f'run-{len(calls)}',
                           executable_sha256=hashes['original' if role else 'expanded'], stdout=json.dumps(result))
                return row
            with patch.dict(os.environ, {'VKFFT_STUDY_GUARD_ID': 'guard', 'VKFFT_STUDY_GUARD_UUID': 'GPU-test'}), \
                 patch.object(audit.runner.frontier, 'append', side_effect=lambda _path, row: rows.append(row)), \
                 patch.object(audit.runner.suite, 'run_one', side_effect=fake_run), \
                 patch.object(audit.runner.rotation.search_controller, 'source_signature',
                              side_effect=lambda _r, original: 'original' if original else 'candidate'):
                audit.runner.run_session(args, ledger, contract, proof)
                with patch.object(audit.eligibility.audit.audit_guard, 'audit', return_value={'status': 'pass'}):
                    report = audit.audit_rows(rows, [{}], [], ledger.rows, proof, 'GPU-test',
                                              {'original': original, 'expanded': expanded})
                    self.assertEqual((report['measured_count'], report['uncovered_count']), (1, 0))
                    self.assertEqual(report['results'][0]['status'], 'measured')
                    changed = copy.deepcopy(rows)
                    timing = next(r for r in changed if r.get('phase') == 'rotation_timing')
                    timing['result']['rotation_final_validation']['slots'][0]['input_unchanged'] = False
                    timing['stdout'] = json.dumps(timing['result'])
                    with self.assertRaises(ValueError):
                        audit.audit_rows(changed, [{}], [], ledger.rows, proof, 'GPU-test',
                                         {'original': original, 'expanded': expanded})
                    changed = copy.deepcopy(rows)
                    pair = next(r for r in changed if r.get('record_type') == 'rotation_pair')
                    pair['ratio'] *= 2
                    with self.assertRaisesRegex(ValueError, 'pair summary'):
                        audit.audit_rows(changed, [{}], [], ledger.rows, proof, 'GPU-test',
                                         {'original': original, 'expanded': expanded})


if __name__ == '__main__':
    unittest.main()
