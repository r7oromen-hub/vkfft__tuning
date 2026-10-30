import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frontier_rotation_eligibility_test', HERE / 'eligibility.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


class EligibilityTests(unittest.TestCase):
    def test_pool_above_actual_l2_or_explicitly_unsupported(self):
        self.assertEqual(e.pool(1024, 4096, 64), {'slots': 8, 'total_bytes': 16384, 'graph_repeats': 64, 'l2_bytes': 4096})
        self.assertIsNone(e.pool(1024, 4 * 1024 * 1024, 64))
        self.assertIsNone(e.pool(800 * 1024 * 1024, 1, 64))
        with self.assertRaises(ValueError): e.pool(1024, 1024, 63)

    def test_only_raw_reaudited_positive_winner_is_eligible(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            plan = root / 'plan.json'; plan.write_text('{}', encoding='utf8')
            binaries = {}
            for name in ('original', 'expanded'):
                binaries[name] = root / name; binaries[name].write_bytes(name.encode())
            req = {'radix': [16, 16], 'logical_registers': 16, 'grouped_batch': 4}
            winner = {'n': 256, 'batch': 16, 'candidate_id': 'winner', 'request': req,
                      'role': 'winner', 'positive_confirmation': True}
            negative = dict(winner, candidate_id='negative', role='negative_control', positive_confirmation=False)
            genhash = 'a' * 64
            screen = {'gpu': 'GPU-test', 'screen_contract': {'gpu': 'GPU-test', 'vkfft_commit': 'vk',
                                                            'generated_provenance': {'sha256': genhash}},
                      'cases': [dict(winner, screen_original_signature_sha256='hash-original',
                                     screen_candidate_signature_sha256='hash-candidate')]}
            paths = {}
            for side in ('screen', 'first', 'second'):
                paths[side] = {}
                for name in ('rows', 'guard', 'incidents', 'budget'):
                    path = root / f'{side}_{name}'; path.write_text('', encoding='utf8'); paths[side][name] = path
            rows = [{'record_type': 'session_start', 'contract': {'generated_provenance': {'sha256': genhash}}},
                    {'record_type': 'candidate_start', 'n': 256, 'batch': 16, 'candidate_id': 'winner'}]
            for round_no in range(10):
                for role in ('original', 'winner'):
                    rows.append({'record_type': 'execution', 'phase': 'confirmation_timing', 'round': round_no,
                                 'candidate_id': role, 'workload': {'graph_repeats': 64},
                                 'result': {'sig': 'original' if role == 'original' else 'candidate',
                                            'gpu': {'uuid': 'GPU-test', 'l2_cache_bytes': 32768}}})
            rows.append({'record_type': 'candidate_disposition', 'n': 256, 'batch': 16, 'candidate_id': 'winner'})
            for side in ('first', 'second'):
                paths[side]['rows'].write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf8')
            reports = [{'status': 'pass', 'session_kind': side, 'vkfft_commit': 'vk'} for side in ('first', 'second')]
            with patch.object(e.evidence, 'audit_screen', return_value=screen), \
                 patch.object(e.audit, 'audit_session', side_effect=reports), \
                 patch.object(e.audit, 'audit_two', return_value={'status': 'pass', 'results': [winner, negative]}), \
                 patch.object(e.search, 'source_signature', side_effect=lambda r, original: r['sig']), \
                 patch.object(e.evidence, 'signature_hash', side_effect=lambda x: 'hash-' + x):
                result = e.verify(plan, paths['screen'], paths['first'], paths['second'], binaries, 'GPU-test', 'commit')
            self.assertEqual(len(result['cases']), 1)
            c = result['cases'][0]
            self.assertEqual(c['candidate_id'], 'winner')
            self.assertEqual(c['rotation']['slots'], 2)
            self.assertEqual(c['hot_candidate_signature'], 'candidate')
            rows[2]['result']['gpu']['l2_cache_bytes'] += 1
            paths['second']['rows'].write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf8')
            with patch.object(e.evidence, 'audit_screen', return_value=screen), \
                 patch.object(e.audit, 'audit_session', side_effect=reports), \
                 patch.object(e.audit, 'audit_two', return_value={'status': 'pass', 'results': [winner]}):
                with self.assertRaisesRegex(ValueError, 'L2 changed'):
                    e.verify(plan, paths['screen'], paths['first'], paths['second'], binaries, 'GPU-test', 'commit')

    def test_incomplete_segment_is_not_a_winner(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            e.segment([{'record_type': 'candidate_start', 'n': 1, 'batch': 1, 'candidate_id': 'a'}], (1, 1, 'a'))


if __name__ == '__main__':
    unittest.main()
