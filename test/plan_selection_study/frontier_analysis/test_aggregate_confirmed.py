import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import aggregate_confirmed as aggregate
from test_analyze import fixture


def stats(ratio):
    return {'rounds': 10, 'geometric_mean_ratio': ratio, 'ci95_low': ratio,
            'ci95_high': ratio, 'candidate_over_original_by_round': [ratio] * 10,
            'latency_reduction': 1 - ratio,
            'caveat': 'within-session t interval; independent-block assumption; does not correct screening or multiple comparison'}


def inputs(first_ratio=.96, second_ratio=.90):
    coverage, rows, rows_hash = fixture()
    # Keep the screened winner separate from the identity control.
    next(r for r in rows if r.get('record_type') == 'workload_status' and r['n'] == 128)['request'] = {
        'radix': [8, 16], 'logical_registers': 64, 'grouped_batch': 1}
    winner = coverage['cases'][0]
    key = {k: winner[k] for k in ('n', 'batch', 'candidate_id', 'request')}
    proof = dict(key, role='winner', screen_ratio=.90,
                 screen_candidate_signature_sha256='1'*64, screen_original_signature_sha256='2'*64)
    source = {'rows': rows_hash, 'guard': '3'*64, 'incidents': '4'*64, 'budget': '5'*64}
    binaries = {'original': '6'*64, 'expanded': '7'*64}
    for row in rows:
        row['contract'] = dict(row['contract'], binaries=binaries, vkfft_commit='f'*40)
    plan_hash = '8'*64
    first_hashes = {'rows': '9'*64, 'guard': 'a'*64, 'incidents': 'b'*64, 'budget': 'c'*64}
    first_stats, second_stats = stats(first_ratio), stats(second_ratio)

    def session(kind, result, session_id, guard_id, started, completed):
        meets = result['latency_reduction'] >= .02 and result['ci95_high'] < 1
        return {'schema': 'frontier-confirmation-session-audit/v1', 'status': 'pass',
                'gpu': coverage['gpu'], 'execution_commit': 'd'*40, 'vkfft_commit': 'f'*40,
                'binaries': binaries, 'plan_sha256': plan_hash, 'source_sha256': source,
                'session_kind': kind, 'session_id': session_id, 'guard_id': guard_id,
                'first_session': None, 'guard_started_utc': started, 'guard_completed_utc': completed,
                'guard_run_ids': [guard_id + '-run'], 'guard_audit': {'status': 'pass'},
                'results': [dict(key, role='winner', statistics=result,
                                 meets_practical_threshold=meets, practical=meets, default_faster=False)]}

    first = session('first', first_stats, 'first-id', 'first-guard',
                    '2026-10-31T00:00:00+00:00', '2026-10-31T00:01:00+00:00')
    second = session('second', second_stats, 'second-id', 'second-guard',
                     '2026-10-31T00:02:00+00:00', '2026-10-31T00:03:00+00:00')
    second['first_session'] = {'session_id': 'first-id', 'guard_id': 'first-guard', 'evidence_sha256': first_hashes}
    one = first['results'][0]['practical']; two = second['results'][0]['practical']
    combined_case = dict(key, role='winner', first=first_stats, second=second_stats,
                         first_meets_practical_threshold=one, second_meets_practical_threshold=two,
                         first_practical=one, second_practical=two,
                         same_improvement_direction=first_ratio < 1 and second_ratio < 1,
                         both_sessions_practical=one and two, positive_confirmation=one and two,
                         both_sessions_default_faster=False, negative_control_unexpected_improvement=False)
    combined = {'schema': 'frontier-confirmation-two-session-audit/v1', 'status': 'pass',
                'gpu': coverage['gpu'], 'plan_sha256': plan_hash, 'results': [combined_case],
                'scope': 'two ordered independent guarded sessions on one GPU; no multiple-testing adjustment',
                'role_interpretation': 'positive_confirmation is reserved for winners; negative controls report default-faster and unexpected-improvement outcomes separately'}
    report = {'schema': 'frontier-confirmation-audit-report/v1', 'status': 'pass',
              'screen': {'gpu': coverage['gpu'], 'screen_contract': rows[0]['contract'],
                         'source_sha256': source, 'cases': [proof]},
              'first': first, 'second': second, 'combined': combined,
              'evidence_sha256': {'/remote/search/rows.jsonl': rows_hash}}
    return coverage, rows, rows_hash, report, 'b'*64


def identity_cases(include_slow_nondefault):
    coverage, rows, digest = fixture()
    # Fixture's first case equals the observed identity; a sub-1 ratio here is
    # deliberately retained to check that identity measurement noise is excluded.
    indices = [0, 2, 4] if include_slow_nondefault else [0]
    selected = [coverage['cases'][i] for i in indices]
    ids = {case['candidate_id'] for case in selected}
    retained = [r for r in rows if r.get('record_type') == 'session_start' or
                (r.get('record_type') == 'workload_status' and (r['n'], r['batch']) == (128, 16))]
    accept = False
    for row in rows:
        if row.get('record_type') == 'candidate_start':
            accept = row['n'] == 128 and row['batch'] == 16 and row['candidate_id'] in ids
        if accept:
            retained.append(row)
    coverage['cases'] = selected
    coverage['counts'] = {'selected': len(selected), 'screen_complete': len(selected),
                          'selected_not_started': 0, 'partial': 0,
                          'scheduler_rejected': 0, 'resource_rejected': 0}
    return coverage, retained, digest, None, None


class AggregateConfirmedTests(unittest.TestCase):
    def test_identity_alone_is_not_nondefault_coverage(self):
        result = aggregate.aggregate(*identity_cases(False))
        work = next(x for x in result['workloads'] if (x['n'], x['batch']) == (128, 16))
        self.assertEqual(work['identity_screen_complete_count'], 1)
        self.assertEqual(work['nondefault_screen_complete_count'], 0)
        self.assertEqual(work['coverage_reason'], 'identity_only')
        self.assertEqual(work['confirmation_status'], 'not_covered')
        self.assertIsNone(work['screen_best_with_default'])
        self.assertEqual(work['fallback']['kind'], 'default')
        self.assertEqual(result['workload_stats']['covered_count'], 0)
        self.assertEqual(result['screen_summary']['known_count'], 0)
        missing = next(x for x in result['workloads'] if (x['n'], x['batch']) == (256, 1))
        self.assertEqual(missing['coverage_reason'], 'no_complete_candidates')

    def test_only_slow_nondefault_candidates_count_as_covered_but_default_wins(self):
        result = aggregate.aggregate(*identity_cases(True))
        work = next(x for x in result['workloads'] if (x['n'], x['batch']) == (128, 16))
        self.assertEqual(work['identity_screen_complete_count'], 1)
        self.assertEqual(work['nondefault_screen_complete_count'], 2)
        self.assertEqual(work['coverage_reason'], 'nondefault_screened')
        self.assertEqual(work['confirmation_status'], 'no_screen_winner')
        self.assertEqual(work['screen_best_with_default']['kind'], 'default')
        self.assertEqual(work['screen_best_with_default']['ratio'], 1)
        self.assertEqual(result['workload_stats']['covered_count'], 1)
        self.assertEqual(result['screen_summary']['known_count'], 1)
        self.assertEqual(result['candidate_stats']['screen_complete_count'], 3)
        self.assertEqual(result['candidate_stats']['faster_count'], 1)
        self.assertAlmostEqual(result['candidate_stats']['faster_fraction'], 1/3)
        self.assertEqual(result['nondefault_candidate_stats']['screen_complete_count'], 2)
        self.assertEqual(result['nondefault_candidate_stats']['faster_count'], 0)
        self.assertEqual(result['nondefault_candidate_stats']['faster_fraction'], 0)

    def test_incomplete_matrix_has_no_campaign_average(self):
        result = aggregate.aggregate(*inputs())
        self.assertEqual(len(result['workloads']), 28)
        self.assertFalse(result['fallback_summary']['complete_28'])
        self.assertIsNone(result['fallback_summary']['mean_latency_reduction'])
        self.assertIsNone(result['fallback_summary']['geometric_mean_speedup'])
        unobserved = next(x for x in result['workloads'] if x['default_status'] == 'not_observed')
        self.assertEqual(unobserved['confirmation_status'], 'not_covered')
        self.assertIsNone(unobserved['fallback'])

    def test_confirmed_equal_weight_sessions_and_default_fallback(self):
        result = aggregate.aggregate(*inputs())
        work = next(x for x in result['workloads'] if (x['n'], x['batch']) == (128, 16))
        self.assertEqual(work['confirmation_status'], 'confirmed_practical')
        self.assertEqual(work['fallback']['kind'], 'confirmed_candidate')
        # Equal weight in log space, independent of the screening median of .90.
        self.assertAlmostEqual(work['confirmation']['equal_weight_geometric_mean_ratio'], math.sqrt(.96*.90))
        work = next(x for x in result['workloads'] if (x['n'], x['batch']) == (256, 1))
        self.assertEqual(work['confirmation_status'], 'not_covered')
        self.assertEqual(work['coverage_reason'], 'identity_only')
        self.assertEqual(work['fallback']['kind'], 'default')

    def test_unconfirmed_and_nonpractical_must_fall_back_to_default(self):
        coverage, rows, digest, _, report_hash = inputs()
        no_report = aggregate.aggregate(coverage, rows, digest, None, None)
        work = next(x for x in no_report['workloads'] if (x['n'], x['batch']) == (128, 16))
        self.assertEqual(work['confirmation_status'], 'not_confirmed')
        self.assertEqual(work['fallback']['kind'], 'default')

    def test_all_default_workloads_equal_weighted_28(self):
        coverage, rows, digest, report, report_hash = inputs()
        actual = aggregate.analyze.analyze(coverage, rows, digest, 20261031)
        present = {(w['n'], w['batch']): w for w in actual['workloads']}
        for n, batch in aggregate.WORKLOADS:
            if (n, batch) not in present:
                actual['workloads'].append({'n': n, 'batch': batch, 'default_status': 'identity_passed',
                    'outcomes': {'screen_complete_count': 0}, 'best_screened': None, 'best_with_default': None})
        with patch.object(aggregate.analyze, 'analyze', return_value=actual):
            result = aggregate.aggregate(coverage, rows, digest, report, report_hash)
        summary = result['fallback_summary']
        ratio = math.sqrt(.96 * .90)
        self.assertTrue(summary['complete_28'])
        self.assertEqual(summary['known_count'], 28)
        self.assertEqual(result['workload_stats']['denominator'], 28)
        self.assertAlmostEqual(summary['mean_latency_reduction'], (1 - ratio) / 28)
        self.assertAlmostEqual(summary['geometric_mean_speedup'], math.exp(-math.log(ratio) / 28))
        result = aggregate.aggregate(*inputs(.99, .97))
        work = next(x for x in result['workloads'] if (x['n'], x['batch']) == (128, 16))
        self.assertEqual(work['confirmation_status'], 'confirmed_not_practical')
        self.assertEqual(work['fallback']['kind'], 'default')

    def test_mismatched_source_and_gpu_or_commits_are_rejected(self):
        for where, field, value in [('screen', 'gpu', 'GPU-other'),
                                    ('first', 'gpu', 'GPU-other'),
                                    ('second', 'execution_commit', 'f'*40)]:
            with self.subTest(where=where, field=field):
                coverage, rows, digest, report, report_hash = inputs()
                report[where][field] = value
                with self.assertRaises(ValueError):
                    aggregate.aggregate(coverage, rows, digest, report, report_hash)
        for field, value in [('rows', 'f'*64)]:
            coverage, rows, digest, report, report_hash = inputs()
            report['screen']['source_sha256'][field] = value
            with self.assertRaises(ValueError):
                aggregate.aggregate(coverage, rows, digest, report, report_hash)
        coverage, rows, digest, report, report_hash = inputs()
        report['screen']['screen_contract'] = dict(report['screen']['screen_contract'], execution_commit='f'*40)
        with self.assertRaises(ValueError):
            aggregate.aggregate(coverage, rows, digest, report, report_hash)
        with self.assertRaises(ValueError):
            aggregate.aggregate(coverage, rows, 'f'*64, report, report_hash)

    def test_incomplete_or_forged_confirmation_cannot_pass(self):
        for change in ('status', 'second', 'combined', 'combined_score'):
            with self.subTest(change=change):
                coverage, rows, digest, report, report_hash = inputs()
                if change == 'status': report['status'] = 'fail'
                elif change == 'second': report['second'] = None
                elif change == 'combined': report['combined'] = None
                else: report['combined']['results'][0]['positive_confirmation'] = False
                with self.assertRaises(ValueError):
                    aggregate.aggregate(coverage, rows, digest, report, report_hash)

    def test_confirmation_file_hash_and_raw_reaudit_are_both_required(self):
        _, _, _, report, _ = inputs()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            def file(name):
                path = root / name
                path.write_text(name, encoding='utf8')
                return path
            screen = {k: file('screen-' + k) for k in ('rows', 'guard', 'incidents', 'budget')}
            first = {k: file('first-' + k) for k in ('rows', 'guard', 'incidents', 'budget')}
            second = {k: file('second-' + k) for k in ('rows', 'guard', 'incidents', 'budget')}
            binaries = {k: file(k) for k in ('original', 'expanded')}
            plan = root / 'plan.json'
            contract = report['screen']['screen_contract']
            contract['generated_provenance'] = {'sha256': 'd'*64}
            plan.write_text(json.dumps({'screen_execution_commit': contract['execution_commit'],
                                        'screen_binaries': contract['binaries']}), encoding='utf8')
            hashes = {k: aggregate.sha(v) for k, v in first.items()}
            report['second']['first_session']['evidence_sha256'] = hashes
            report['combined'] = aggregate.confirmation_audit.audit_two(report['first'], report['second'], hashes)
            paths = [plan, *screen.values(), *first.values(), *second.values(), *binaries.values()]
            report['evidence_sha256'] = {str(p): aggregate.sha(p) for p in paths}
            logs = [{'contract': dict(contract, execution_commit=report['first']['execution_commit'])}]
            with patch.object(aggregate.evidence, 'audit_screen', return_value=report['screen']), \
                 patch.object(aggregate.evidence.frontier, 'read_log', return_value=logs), \
                 patch.object(aggregate.confirmation_audit, 'audit_session', side_effect=[report['first'], report['second']]):
                self.assertIs(aggregate.confirm_files(report, plan, screen, first, second, binaries), report)
            # Hash validation precedes the expensive audit and catches altered raw evidence.
            screen['rows'].write_text('tampered', encoding='utf8')
            with self.assertRaises(ValueError):
                aggregate.confirm_files(report, plan, screen, first, second, binaries)
            screen['rows'].write_text('screen-rows', encoding='utf8')
            wrong = copy.deepcopy(report['screen']); wrong['gpu'] = 'GPU-other'
            with patch.object(aggregate.evidence, 'audit_screen', return_value=wrong):
                with self.assertRaises(ValueError):
                    aggregate.confirm_files(report, plan, screen, first, second, binaries)
            wrong_session = copy.deepcopy(report['first']); wrong_session['status'] = 'fail'
            with patch.object(aggregate.evidence, 'audit_screen', return_value=report['screen']), \
                 patch.object(aggregate.evidence.frontier, 'read_log', return_value=logs), \
                 patch.object(aggregate.confirmation_audit, 'audit_session', return_value=wrong_session):
                with self.assertRaises(ValueError):
                    aggregate.confirm_files(report, plan, screen, first, second, binaries)


if __name__ == '__main__':
    unittest.main()
