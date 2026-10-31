"""Synthetic CPU evidence for the offline auditor; never GPU admission evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import audit
import candidate
import identity
import test_validation


GPU = test_validation.GPU
BOOT = '12345678-1234-1234-1234-123456789abc'
SESSION = 'session-fixture'
GUARD = 'guard-fixture'
PATTERNS = ('zero', 'impulse', 'constant', 'quarter-tone', 'random')
SEEDS = (42, 53, 71)


def protocol():
    return {'schema': 'nonuniform-admission-protocol/v1', 'gpu_uuid': GPU,
            'hostname': 'fixture', 'boot_id': BOOT,
            'machine_id': '0' * 32, 'seconds': 1000, 'idle_seconds': 1,
            'timeout': 10, 'graph_repeats': 64,
            'patterns': list(PATTERNS), 'seeds': list(SEEDS)}


def contract():
    return {'schema': 'nonuniform-admission-execution/v1', 'boot_id': BOOT,
            'gpu': GPU, 'campaign_deadline_monotonic': 1100.0,
            'binaries': {'original': 'a' * 64, 'expanded': 'b' * 64}}


def item():
    request = test_validation.make_request()
    return {'request': request, 'candidate_id': candidate.candidate_id(request)}


def work_row(req, cid, phase, *, role='candidate', mode='execute', pattern='random', seed=42):
    """Shape actual controller work and retain the validator's real fixture result."""
    row = test_validation.make_row(req, role=role, describe=mode == 'describe')
    row.update(record_type='execution', phase=phase, candidate_id=cid,
               session_id=SESSION, guard_id=GUARD, round=0, order=0,
               workload_id='n128_b16', guard_run_id=GUARD + ':' + phase + ':' + str(seed) + ':' + pattern)
    w, r = row['workload'], row['result']
    w.update(seed=seed, pattern=pattern, graph_repeats=64, repeats=1,
             warmup=0, warmup_ms=0, study_mode=mode, config={})
    r.update(candidate_id=cid, seed=seed, pattern=pattern, graph_repeats=64,
             warmup=0, warmup_ms_requested=0, warmup_actual=0, warmup_wall_ms=0,
             graph_expected_kernel_count=64, graph_kernel_count=64,
             exec_ms_samples=[.001], raw_interval_ms_samples=[.064])
    ref = ('independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity'
           if pattern == 'random' else 'analytic_pattern_reference_with_independent_CPU_FFT_crosscheck')
    for field in ('correctness', 'single_forward_correctness'):
        r[field]['reference'] = ref
    row['executable_sha256'] = ('a' if role == 'original' else 'b') * 64
    row['stdout'] = json.dumps(r, allow_nan=False)
    row['stderr'] = ''
    return row


def transcript():
    entry = item(); req, cid = entry['request'], entry['candidate_id']
    rows = [dict(record_type='session_start', session_id=SESSION, guard_id=GUARD,
                 gpu=GPU, candidate_count=1),
            dict(record_type='candidate_start', session_id=SESSION, guard_id=GUARD,
                 n=128, batch=16, candidate_id=cid, request=copy.deepcopy(req))]
    description = work_row(req, cid, 'describe', mode='describe')
    original = work_row(req, 'original', 'identity', role='original')
    default = work_row(req, 'expanded_default', 'identity', role='default')
    replay = work_row(req, 'identity', 'identity')
    replay['guard_run_id'] = GUARD + ':identity:replay'
    default['guard_run_id'] = GUARD + ':identity:default'
    original['guard_run_id'] = GUARD + ':identity:original'
    rows.extend([description, original, default, replay])
    proof = identity.compare_three(original['result'], default['result'], replay['result'])
    rows.append(dict(record_type='identity_passed', session_id=SESSION, guard_id=GUARD,
                     n=128, batch=16, candidate_id=cid, proof=proof))
    for pattern in PATTERNS:
        for seed in SEEDS:
            rows.append(work_row(req, cid, 'validation', pattern=pattern, seed=seed))
    rows.extend([dict(record_type='candidate_disposition', session_id=SESSION, guard_id=GUARD,
                      n=128, batch=16, candidate_id=cid, request=copy.deepcopy(req),
                      status='checks_passed_pending_audit', validation_count=15,
                      interpretation='single-session checks only; independent guard/ledger/source audit still required'),
                 dict(record_type='session_checks_complete', session_id=SESSION, guard_id=GUARD,
                      status='pending_independent_audit')])
    return rows, [entry]


def with_ledger(rows, c=None):
    c = contract() if c is None else c
    start = 100.0
    state = {'contract': c, 'seconds': 1000, 'boot': BOOT,
             'start': start, 'last': start, 'inflight': None}
    ledger = [{'event': 'start', 'state': copy.deepcopy(state)}]
    clock = start + 1
    for row in rows:
        row['contract'] = c
        if row['record_type'] not in ('execution', 'launch_denied'):
            continue
        w = row['workload']
        key = {'n': w['n'], 'batch': w['batch'], 'candidate_id': row['candidate_id'],
               'phase': row['phase'], 'round': row['round']}
        state.update(last=clock, inflight=key)
        ledger.append({'event': 'work_start', 'state': copy.deepcopy(state)})
        row['started_monotonic'] = clock + .1
        row['ended_monotonic'] = clock + .7
        row['wall_seconds'] = .6
        state.update(last=clock + 1, inflight=None)
        ledger.append({'event': 'work_finish', 'identity': key, 'status': row['status'],
                       'state': copy.deepcopy(state)})
        clock += 2
    return ledger


def denied_transcript(*, late):
    """A clean budget boundary at the first candidate, with or without a guard run."""
    full, items = transcript()
    entry = items[0]; req, cid = entry['request'], entry['candidate_id']
    c = contract(); c['campaign_deadline_monotonic'] = 143.0
    rows = copy.deepcopy(full[:2])
    if late:
        denied = copy.deepcopy(next(r for r in full if r.get('phase') == 'describe'))
        denied.update(record_type='launch_denied', status='budget_denied_before_launch',
                      reason='absolute_stage_or_campaign_deadline', guard_run_id=GUARD + ':denied',
                      command=['/tmp/generated-binary'], validation_errors=[])
        for name in ('stdout', 'stderr', 'returncode', 'result'):
            denied.pop(name, None)
        rows.append(denied)
    rows.extend([dict(record_type='candidate_disposition', session_id=SESSION, guard_id=GUARD,
                      n=128, batch=16, candidate_id=cid, request=copy.deepcopy(req),
                      status='not_started', reason='budget_denied'),
                 dict(record_type='budget_exhausted', session_id=SESSION, guard_id=GUARD)])
    ledger = with_ledger(rows, c)
    if late:
        denied = next(r for r in rows if r.get('record_type') == 'launch_denied')
        denied.update(started_monotonic=130.0, ended_monotonic=130.5, wall_seconds=.5)
        ledger[-1]['state']['last'] = 130.7
    else:
        state = copy.deepcopy(ledger[0]['state'])
        state['last'] = 104.0
        key = {'n': 128, 'batch': 16, 'candidate_id': cid, 'phase': 'describe', 'round': 0}
        ledger.append({'event': 'denied', 'identity': key,
                       'reason': 'absolute_stage_or_campaign_deadline', 'state': state})
    return rows, items, ledger, c


def denied_guard(row):
    """Guard-wrapped deadline denial: no child was ever registered or launched."""
    common = {'guard_id': GUARD, 'gpu_uuid': GPU}
    active = {'guard_run_id': row['guard_run_id'], 'phase': row['phase'],
              'workload_id': row['workload_id'], 'candidate_id': row['candidate_id'],
              'round': row['round'], 'order': row['order'], 'expected_exe': row['command'][0]}
    def sample(boundary, a, start, end):
        return dict(common, event='sample', boundary=boundary, active_run=a, ok=True,
                    gpu={'uuid': GPU, 'memory_used_mib': 64}, compute_apps=[],
                    foreign_compute_pids=[], violation_reasons=[], monitor_error=None,
                    idle_monitor_delay=False, sample_started_monotonic=start,
                    sample_completed_monotonic=end, query_wall_seconds=end-start)
    return [dict(common, event='guard_start', parent_pid=100, idle_seconds=1,
                 own_signal_method='pidfd_with_rechecked_PPID_exe_starttime'),
            sample('idle', None, 125.0, 125.1),
            dict(common, event='idle_complete', clean=True, observed_idle_seconds=1),
            dict(common, event='run_one_start', active_run=active, run_transition_monotonic=129.0),
            sample('before', active, 129.1, 129.2),
            sample('after', active, 130.6, 130.7),
            dict(common, event='run_one_end', guard_run_id=row['guard_run_id'], clean=True,
                 status='budget_denied_before_launch', workload_id=row['workload_id'],
                 candidate_id=row['candidate_id'], run_transition_monotonic=130.8),
            sample('final', None, 131.0, 131.1),
            dict(common, event='guard_complete', clean=True, success=True, script_exit_code=0)]


class TranscriptAuditTests(unittest.TestCase):
    def test_complete_identity_and_all_fifteen_real_rows(self):
        rows, items = transcript()
        report = audit.audit_transcript(rows, items, GPU, protocol())
        self.assertIsInstance(report, dict)

    def test_one_missing_or_duplicated_matrix_row_is_not_a_pass(self):
        rows, items = transcript()
        index = next(i for i, row in enumerate(rows) if row.get('phase') == 'validation')
        for replacement in ('missing', 'duplicate'):
            changed = copy.deepcopy(rows)
            if replacement == 'missing':
                changed.pop(index)
            else:
                changed.insert(index, copy.deepcopy(changed[index]))
            with self.subTest(replacement=replacement), self.assertRaises((ValueError, KeyError, TypeError)):
                audit.audit_transcript(changed, items, GPU, protocol())

    def test_forged_candidate_request_identity_or_disposition_rejected(self):
        rows, items = transcript()
        cases = []
        bad = copy.deepcopy(rows)
        next(r for r in bad if r.get('phase') == 'validation')['result']['nonuniform_requested_group'] = '1'
        cases.append(bad)
        bad = copy.deepcopy(rows)
        next(r for r in bad if r.get('record_type') == 'identity_passed')['proof']['candidate_id'] = 'f' * 64
        cases.append(bad)
        bad = copy.deepcopy(rows)
        next(r for r in bad if r.get('record_type') == 'candidate_disposition')['validation_count'] = 14
        cases.append(bad)
        for changed in cases:
            with self.subTest(changed=cases.index(changed)), self.assertRaises((ValueError, KeyError, TypeError)):
                audit.audit_transcript(changed, items, GPU, protocol())

    def test_budget_boundary_is_not_completed_candidate(self):
        for late in (False, True):
            rows, items, _, _ = denied_transcript(late=late)
            with self.subTest(late=late):
                report = audit.audit_transcript(rows, items, GPU, protocol())
                self.assertFalse(report['complete_session'])
                self.assertEqual(report['cases'][0]['status'], 'not_started')
                forged = copy.deepcopy(rows)
                next(r for r in forged if r.get('record_type') == 'candidate_disposition')['status'] = 'checks_passed_pending_audit'
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    audit.audit_transcript(forged, items, GPU, protocol())

    def test_unsupported_identity_requires_observed_default_evidence(self):
        rows, items, _, _ = denied_transcript(late=False)
        disposition = next(r for r in rows if r.get('record_type') == 'candidate_disposition')
        disposition.update(status='unsupported_identity', reason='unsupported_table_or_request_profile')
        rows[-1] = dict(record_type='session_checks_complete', session_id=SESSION, guard_id=GUARD,
                        status='pending_independent_audit')
        # No describe, original or default exists. The state cannot be established.
        with self.assertRaises((ValueError, KeyError, TypeError)):
            audit.audit_transcript(rows, items, GPU, protocol())


class LedgerAuditTests(unittest.TestCase):
    def test_completed_ledger_binds_every_work_item(self):
        rows, _ = transcript(); c = contract(); ledger = with_ledger(rows, c)
        self.assertIsInstance(audit.audit_ledger(rows, ledger, c, protocol()), dict)

    def test_missing_finish_or_different_work_identity_fails(self):
        rows, _ = transcript(); c = contract(); ledger = with_ledger(rows, c)
        missing = copy.deepcopy(ledger); missing.pop(2)
        changed = copy.deepcopy(ledger); changed[2]['identity']['candidate_id'] = 'wrong'
        for bad in (missing, changed):
            with self.subTest(bad=bad[:3]), self.assertRaises((ValueError, KeyError, TypeError)):
                audit.audit_ledger(rows, bad, c, protocol())

    def test_time_regression_unclosed_inflight_or_changed_boot_fails(self):
        rows, _ = transcript(); c = contract(); ledger = with_ledger(rows, c)
        cases = []
        bad = copy.deepcopy(ledger); bad[2]['state']['last'] = 99; cases.append(bad)
        bad = copy.deepcopy(ledger); bad[-1]['state']['inflight'] = {'phase': 'unfinished'}; cases.append(bad)
        bad = copy.deepcopy(ledger); bad[-1]['state']['boot'] = 'another-boot'; cases.append(bad)
        for bad in cases:
            with self.subTest(bad=cases.index(bad)), self.assertRaises((ValueError, KeyError, TypeError)):
                audit.audit_ledger(rows, bad, c, protocol())

    def test_initial_inflight_work_is_never_a_fresh_ledger(self):
        rows, _ = transcript(); c = contract(); ledger = with_ledger(rows, c)
        ledger[0]['state']['inflight'] = {'phase': 'unresolved_prior_gpu_work'}
        with self.assertRaises((ValueError, KeyError, TypeError)):
            audit.audit_ledger(rows, ledger, c, protocol())

    def test_ledger_rejects_work_start_without_timeout_reserve(self):
        rows, _ = transcript(); c = contract(); ledger = with_ledger(rows, c)
        ledger[1]['state']['last'] = 1095.0
        with self.assertRaises((ValueError, KeyError, TypeError)):
            audit.audit_ledger(rows, ledger, c, protocol())

    def test_clean_early_and_late_budget_boundaries_remain_unmeasured(self):
        for late in (False, True):
            rows, _, ledger, c = denied_transcript(late=late)
            with self.subTest(late=late):
                self.assertIsInstance(audit.audit_ledger(rows, ledger, c, protocol()), dict)
                bad = copy.deepcopy(rows); bad.pop()  # no terminal budget_exhausted
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    audit.audit_ledger(bad, ledger, c, protocol())
        rows, _, ledger, c = denied_transcript(late=True)
        forged = copy.deepcopy(ledger)
        forged[-1]['status'] = 'ok'
        with self.assertRaises((ValueError, KeyError, TypeError)):
            audit.audit_ledger(rows, forged, c, protocol())


class GuardAuditTests(unittest.TestCase):
    def test_launch_denied_has_guard_span_but_no_registered_child(self):
        rows, _, _, _ = denied_transcript(late=True)
        row = next(r for r in rows if r.get('record_type') == 'launch_denied')
        guard = denied_guard(row)
        self.assertIsInstance(audit.audit_guard_rows(rows, guard, [], GPU), dict)
        for mutation in ('registered', 'foreign', 'missing_after', 'wrong_candidate'):
            bad = copy.deepcopy(guard)
            if mutation == 'registered':
                bad.insert(5, dict(event='own_harness_registered', guard_id=GUARD,
                    gpu_uuid=GPU, active_run=bad[3]['active_run'], pidfd_retained=True,
                    identity={'pid': 101, 'ppid': 100, 'start_ticks': 1, 'exe': row['command'][0]}))
            elif mutation == 'foreign':
                next(e for e in bad if e.get('boundary') == 'before')['foreign_compute_pids'] = [9]
            elif mutation == 'missing_after':
                bad = [e for e in bad if e.get('boundary') != 'after']
            else:
                next(e for e in bad if e.get('event') == 'run_one_start')['active_run']['candidate_id'] = 'other'
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError, TypeError)):
                audit.audit_guard_rows(rows, bad, [], GPU)

    def test_denied_interval_cannot_hide_bad_sample_or_incident(self):
        rows, _, _, _ = denied_transcript(late=True)
        row = next(r for r in rows if r.get('record_type') == 'launch_denied')
        guard = denied_guard(row)
        self.assertIsInstance(audit.audit_guard_rows(rows, guard, [], GPU), dict)
        for mutation in ('memory', 'slow_query', 'incident', 'abort', 'quarantine'):
            bad = copy.deepcopy(guard)
            bad_rows = copy.deepcopy(rows)
            sample = next(e for e in bad if e.get('boundary') == 'before')
            if mutation == 'memory':
                sample['gpu']['memory_used_mib'] = 3000
            elif mutation == 'slow_query':
                sample.update(query_wall_seconds=2.1,
                              sample_started_monotonic=129.1, sample_completed_monotonic=131.2)
                denied = next(r for r in bad_rows if r.get('record_type') == 'launch_denied')
                denied.update(started_monotonic=132.0, ended_monotonic=132.5, wall_seconds=.5)
                next(e for e in bad if e.get('boundary') == 'after').update(
                    sample_started_monotonic=132.6, sample_completed_monotonic=132.7)
                next(e for e in bad if e.get('event') == 'run_one_end')['run_transition_monotonic'] = 132.8
                next(e for e in bad if e.get('boundary') == 'final').update(
                    sample_started_monotonic=133.0, sample_completed_monotonic=133.1)
            else:
                kind = {'incident': 'incident', 'abort': 'guard_abort',
                        'quarantine': 'quarantined_run'}[mutation]
                bad.insert(5, dict(event=kind, incident=True, guard_id=GUARD, gpu_uuid=GPU,
                                   active_run=copy.deepcopy(bad[3]['active_run'])))
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError, TypeError)):
                audit.audit_guard_rows(bad_rows, bad, [], GPU)

    def test_unmatched_or_overlapping_guard_run_cannot_be_hidden(self):
        rows, _, _, _ = denied_transcript(late=True)
        row = next(r for r in rows if r.get('record_type') == 'launch_denied')
        guard = denied_guard(row)
        for mutation in ('missing_id', 'overlap'):
            bad = copy.deepcopy(guard)
            if mutation == 'missing_id':
                extra = dict(event='run_one_start', guard_id=GUARD, gpu_uuid=GPU,
                             active_run={}, run_transition_monotonic=128.5)
            else:
                active = copy.deepcopy(bad[3]['active_run']); active['guard_run_id'] = GUARD + ':unrecorded'
                extra = dict(event='run_one_start', guard_id=GUARD, gpu_uuid=GPU,
                             active_run=active, run_transition_monotonic=129.05)
            bad.insert(4, extra)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError, TypeError)):
                audit.audit_guard_rows(rows, bad, [], GPU)


class RawEvidenceTests(unittest.TestCase):
    def test_stdout_must_equal_recorded_result_and_command_must_bind_binary(self):
        rows, _ = transcript(); c = contract(); with_ledger(rows, c)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            paths = {'original': Path(temp)/'original', 'expanded': Path(temp)/'expanded'}
            for path in paths.values(): path.write_text('synthetic binary', encoding='utf8')
            for i, row in enumerate(r for r in rows if r.get('record_type') == 'execution'):
                w, name = row['workload'], row['candidate_id']
                exe = paths['original' if name == 'original' else 'expanded'].resolve()
                row.update(invocation_id='invocation-'+str(i), cuda_visible_devices=GPU,
                           candidate_environment_policy='nonuniform, expanded and direct prefixes cleared before explicit override',
                           validation_errors=[])
                row['command'] = [str(exe), '--n', str(w['n']), '--batch', str(w['batch']),
                    '--seed', str(w['seed']), '--timing-mode', 'graph', '--placement', 'outofplace',
                    '--graph-repeats', str(w['graph_repeats']), '--warmup', '0', '--warmup-ms', '0',
                    '--repeats', '1', '--pattern', w['pattern'], '--study-mode', w['study_mode'],
                    '--candidate-id', name]
            self.assertIsInstance(audit.audit_raw_rows(rows, c, protocol(), paths), dict)
            for mutation in ('stdout', 'sha', 'command', 'invocation'):
                changed = copy.deepcopy(rows)
                selected = [r for r in changed if r.get('record_type') == 'execution']
                if mutation == 'stdout': selected[0]['stdout'] = '{"status":"fabricated"}'
                elif mutation == 'sha': selected[0]['executable_sha256'] = 'f' * 64
                elif mutation == 'command': selected[0]['command'][0] = str(paths['original'].resolve())
                else: selected[1]['invocation_id'] = selected[0]['invocation_id']
                with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError, TypeError)):
                    audit.audit_raw_rows(changed, c, protocol(), paths)


class DeclaredCommandTests(unittest.TestCase):
    def test_direct_python_exact_flags_and_recorded_paths_are_required(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp).resolve()
            names = ('protocol', 'manifest', 'original', 'expanded', 'generated_provenance',
                     'build_attestation', 'compile_commands', 'original_dependencies',
                     'rows', 'budget', 'guard', 'incidents')
            # Files need not exist: audit_declared_command compares resolved paths.
            args = SimpleNamespace(**{name: root / ('input with spaces ' + name) for name in names})
            flags = {'--protocol': args.protocol, '--manifest': args.manifest,
                     '--original': args.original, '--expanded': args.expanded,
                     '--generated-provenance': args.generated_provenance,
                     '--build-attestation': args.build_attestation,
                     '--compile-commands': args.compile_commands,
                     '--original-dependencies': args.original_dependencies,
                     '--output': args.rows, '--budget-ledger': args.budget,
                     '--log': args.guard, '--incidents': args.incidents}
            c = contract()
            lock = (Path('/run/lock') / ('vkfft-study-' + c['gpu'] + '.lock')).resolve()
            command = ['python3', str((audit.HERE / 'admission.py').resolve())]
            for flag, path in flags.items(): command.extend((flag, str(path)))
            command.extend(('--campaign-deadline-monotonic', str(c['campaign_deadline_monotonic']),
                            '--lock', str(lock)))
            audit.audit_declared_command(command, args, c, root)
            cases = []
            cases.append(['bash', '-c', 'exec "$@"', *command])
            cases.append([*command, '--extra', 'unexpected'])
            bad = command.copy(); bad[bad.index('--campaign-deadline-monotonic') + 1] = '9999'; cases.append(bad)
            bad = command.copy(); bad[bad.index('--output') + 1] = str(root / 'different-output'); cases.append(bad)
            for i, bad in enumerate(cases):
                with self.subTest(case=i), self.assertRaises((ValueError, KeyError, TypeError)):
                    audit.audit_declared_command(bad, args, c, root)


if __name__ == '__main__':
    unittest.main()
