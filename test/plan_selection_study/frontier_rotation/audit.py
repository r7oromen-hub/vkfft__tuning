#!/usr/bin/env python3
"""Independently re-audit a guarded rotation session from its raw evidence."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
_saved_path = sys.path[:]
try:
    spec = importlib.util.spec_from_file_location('frontier_rotation_runner_audit', HERE / 'runner.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
finally:
    sys.path[:] = _saved_path
frontier = runner.frontier
eligibility = runner.eligibility


def need(condition, message):
    if not condition:
        raise ValueError(message)


def key(row):
    return row['n'], row['batch'], row['candidate_id']


def verify_pair(k, proof, timing, record, final_plan):
    round_no = record['round']
    need(key(record) == k, 'rotation pair candidate mismatch')
    pair = sorted([r for r in timing if r['round'] == round_no], key=lambda r: r['order'])
    order = runner.search.seeded_order(20261031, proof['plan_sha256'], *k, 'rotation', round_no)
    expected_ids = ['original' if role == 'original' else k[2] for role in order]
    need(len(pair) == 2 and [r['order'] for r in pair] == [0, 1] and
         [r['candidate_id'] for r in pair] == expected_ids and list(record['order']) == list(order) and
         record['rotation'] == final_plan and all(r['status'] == 'ok' and
         r['workload']['graph_repeats'] == final_plan['graph_repeats'] for r in pair), 'rotation pair order or plan mismatch')
    by_id = {r['candidate_id']: r for r in pair}
    original, candidate = by_id['original'], by_id[k[2]]
    om = statistics.median(original['result']['exec_ms_samples'])
    cm = statistics.median(candidate['result']['exec_ms_samples'])
    ratio = cm / om
    need(math.isfinite(ratio) and ratio > 0 and record['original_guard_run_id'] == original['guard_run_id'] and
         record['candidate_guard_run_id'] == candidate['guard_run_id'] and
         all(math.isclose(record[field], value, rel_tol=1e-12) for field, value in
             (('original_median_ms', om), ('candidate_median_ms', cm), ('ratio', ratio))), 'rotation pair summary mismatch')
    return ratio, (cm, om)


def verify_context(root, contract, budget, proof, args):
    """Recheck the campaign, source/build association and absolute budget offline."""
    need(budget and budget[0].get('event') == 'start', 'budget start missing')
    execution_commit = contract['execution_commit']
    parent = runner.verify_declaration(root, execution_commit, args.rotation_experiment_id)
    need(parent == contract['code_commit'], 'rotation declaration and build code commit differ')
    # Audit against the frozen source, not merely the run's recorded hashes.
    frontier.git(root, 'diff', '--quiet', parent, '--',
                 'test/plan_selection_study/frontier_rotation', 'test/plan_selection_study/frontier_confirmation',
                 'test/plan_selection_study/frontier_execution', 'test/plan_selection_study/expanded_search',
                 'test/plan_selection_study/original_compare', 'test/plan_selection_study/instrumentation',
                 'test/plan_selection_study/frontier_search', 'VkFFT')
    need(frontier.git(root, 'rev-parse', parent + ':VkFFT') == proof['vkfft_commit'] and
         frontier.git(Path(root) / 'VkFFT/vkFFT', 'rev-parse', 'HEAD') == proof['vkfft_commit'],
         'committed or checked-out VkFFT revision differs from hot evidence')
    frontier.git(Path(root) / 'VkFFT/vkFFT', 'diff', '--quiet')
    frontier.git(Path(root) / 'VkFFT/vkFFT', 'diff', '--cached', '--quiet')
    for path, digest in contract['source_files'].items():
        need(frontier.sha(path) == digest, 'recorded source hash changed: ' + path)
    rotation_bins = {'original': args.rotation_original, 'expanded': args.rotation_expanded}
    build = runner.verify_rotation_build(root, args.rotation_root, args.generated_root, args.original_include_root,
        args.instrumentation_root, proof, rotation_bins, args.build_attestation, args.compile_commands,
        args.original_dependencies, parent)
    need(build == contract['rotation_build'], 'rotation build evidence differs from run contract')
    initial = budget[0]['state']
    need(all(type(value) in (int, float) and math.isfinite(value) for value in
             (initial['start'], initial['seconds'], contract['deadline_monotonic'], contract['stage_seconds'], contract['timeout'])) and
         initial['seconds'] > 0 and contract['stage_seconds'] > 0 and contract['timeout'] > 0,
         'nonfinite or invalid budget contract')
    # Auditing after the campaign has expired is valid; use the recorded start
    # and boot only to recheck the immutable original campaign artifacts.
    campaign = runner.campaign(root, args.campaign_record, args.campaign_start, args.admission_budget,
                               args.gpu, now=contract['campaign']['start'], boot=initial['boot'])
    need(campaign == contract['campaign'], 'campaign source or deadline mismatch')
    deadline = contract['deadline_monotonic']
    need(deadline <= campaign['deadline'] and deadline <= initial['start'] + contract['stage_seconds'],
         'stage deadline exceeds campaign or stage budget')
    previous = initial['start']
    expected_event = 'start'
    for entry in budget:
        state = entry['state']
        need(entry.get('event') == expected_event, 'budget events are not strictly alternating')
        if expected_event in ('start', 'work_finish'):
            need(state.get('inflight') is None, 'budget state unexpectedly in flight')
        else:
            need(isinstance(state.get('inflight'), dict), 'work start lacks in-flight identity')
        need(state['contract'] == contract and state['boot'] == campaign['boot'] and
             state['start'] == initial['start'] and state['seconds'] == initial['seconds'] and
             state['last'] >= previous, 'budget boot, contract or monotonic timeline mismatch')
        if entry.get('event') == 'work_start':
            need(state['last'] + contract['timeout'] + 30 < deadline,
                 'work started without remaining global deadline reserve')
        previous = state['last']
        expected_event = 'work_finish' if expected_event == 'work_start' else 'work_start'
    need(budget[-1]['state'].get('inflight') is None, 'unresolved budget item')
    return {'campaign': campaign, 'rotation_build': build, 'execution_commit': execution_commit}


def audit_rows(rows, guard, incidents, budget, proof, gpu, rotation_binaries):
    """A passing status concerns evidence integrity, not confirmation or cache misses."""
    need(rows and guard and budget, 'missing rotation evidence')
    contract = rows[0]['contract']
    need(all(row.get('contract') == contract for row in rows), 'mixed rotation contracts')
    need(contract.get('schema') == 'frontier-rotation-session/v1' and contract.get('gpu') == gpu and
         contract.get('eligibility') == proof, 'rotation contract or hot proof mismatch')
    need({name: frontier.sha(path) for name, path in rotation_binaries.items()} == contract['rotation_build']['binaries'],
         'rotation binary hashes differ')
    guard_report = eligibility.audit.audit_guard.audit(rows, guard, incidents, gpu)
    need(guard_report['status'] == 'pass', 'guard audit failed: ' + str(guard_report.get('errors')))
    starts = [r for r in rows if r.get('record_type') == 'rotation_session_start']
    ends = [r for r in rows if r.get('record_type') == 'rotation_session_complete']
    need(len(starts) == len(ends) == 1 and ends[0].get('status') in ('complete', 'complete_with_uncovered', 'no_eligible_winner'),
         'rotation session is partial or lacks a terminal record')
    session, gid = starts[0]['session_id'], starts[0]['guard_id']
    need(all(r.get('session_id') == session and r.get('guard_id') == gid for r in rows), 'mixed session or guard IDs')
    executions = [r for r in rows if r.get('record_type') == 'execution']
    need(budget[0].get('event') == 'start' and budget[-1]['state'].get('inflight') is None and
         all(r['state']['contract'] == contract for r in budget), 'budget state mismatch or unfinished work')
    started = [r for r in budget if r.get('event') == 'work_start']
    finished = [r for r in budget if r.get('event') == 'work_finish']
    need(len(started) == len(finished) == len(executions), 'budget/execution count mismatch')
    for begin, end, row in zip(started, finished, executions):
        work, w = end['identity'], row['workload']
        need(begin['state']['inflight'] == work and end['status'] == row['status'] and
             (work['n'], work['batch'], work['candidate_id'], work['phase'], work['round']) ==
             (w['n'], w['batch'], row['candidate_id'], row['phase'], row['round']), 'budget work linkage mismatch')
        need(begin['state']['last'] <= contract['deadline_monotonic'], 'work started after global deadline')
        parsed = [json.loads(line) for line in row['stdout'].splitlines() if line.strip()]
        need(len(parsed) == 1 and parsed[0] == row['result'], 'execution stdout/result mismatch')
        role = 'original' if row['candidate_id'] == 'original' else 'expanded'
        need(row['executable_sha256'] == contract['rotation_build']['binaries'][role], 'execution binary mismatch')
        need(not runner.numerical_failure(row), 'rotation numerical failure')
    expected = {key(case): case for case in proof['cases']}
    need(len(expected) == len(proof['cases']), 'duplicate confirmed cases')
    segments, active = {}, None
    for row in rows:
        if row.get('record_type') == 'rotation_case_start':
            k = key(row)
            need(active is None and k in expected and k not in segments and row['request'] == expected[k]['request'] and
                 row.get('eligibility_status') == expected[k]['status'],
                 'unexpected or duplicate rotation case start')
            segments[k] = [row]; active = k
        elif active is not None:
            segments[active].append(row)
            if row.get('record_type') == 'rotation_disposition':
                need(key(row) == active, 'rotation disposition belongs to another case')
                active = None
    need(active is None and set(segments) == set(expected), 'missing or partial candidate segments')
    need(len([r for r in rows if r.get('record_type') == 'rotation_disposition']) == len(expected) and
         len([r for r in rows if r.get('record_type') == 'rotation_pair']) ==
         sum(1 for part in segments.values() for row in part if row.get('record_type') == 'rotation_pair'),
         'orphan pair or disposition record')
    used = set()
    results = []
    for k, case in expected.items():
        part = segments[k]
        dispositions = [r for r in part if r.get('record_type') == 'rotation_disposition']
        need(len(dispositions) == 1, 'missing or duplicate rotation disposition')
        disposition = dispositions[0]
        need(disposition['request'] == case['request'], 'disposition request mismatch')
        status = disposition.get('status')
        part_exec = [r for r in part if r.get('record_type') == 'execution']
        need(all((r['workload']['n'], r['workload']['batch']) == k[:2] and
                 r['phase'] in ('rotation_calibration', 'rotation_timing') and
                 r['candidate_id'] in ('original', k[2]) for r in part_exec), 'rotation execution workload/candidate mismatch')
        used.update(r['guard_run_id'] for r in part_exec)
        if case['rotation'] is None:
            need(status == 'uncovered_pool_or_graph_limit' and not part_exec, 'unsupported pool has device work or wrong disposition')
            results.append({'n': k[0], 'batch': k[1], 'candidate_id': k[2], 'status': 'uncovered_pool_or_graph_limit'})
            continue
        base = case['rotation']
        legacy = {'n': k[0], 'batch': k[1], 'request': case['request'],
                  'original_signature': case['hot_original_signature'], 'candidate_signature': case['hot_candidate_signature']}
        for row in part_exec:
            repetitions = row['workload']['graph_repeats']
            need(type(repetitions) is int and repetitions >= base['graph_repeats'] and repetitions <= 2048 and
                 not repetitions & (repetitions-1) and repetitions % base['slots'] == 0, 'invalid rotation graph repetitions')
            plan = dict(base, graph_repeats=repetitions)
            if row.get('status') == 'ok':
                runner.rotation.check(row, gpu, legacy, plan, original=row['candidate_id'] == 'original',
                                      timing=row['phase'] == 'rotation_timing')
            else:
                need(status == 'uncovered' and runner.uncovered(row, gpu, case, plan) == disposition.get('reason'),
                     'failed rotation run is not the reported unsupported condition')
        if status != 'complete':
            need(status in ('uncovered', 'uncovered_calibration_interval'), 'unknown rotation disposition')
            timing = [r for r in part_exec if r['phase'] == 'rotation_timing']
            pairs = [r for r in part if r.get('record_type') == 'rotation_pair']
            need(len(pairs) < 10 and {r['round'] for r in pairs} == set(range(len(pairs))),
                 'uncovered pair summaries are not a complete prefix')
            calibrated = [r for r in part if r.get('record_type') == 'rotation_calibration_complete']
            if timing:
                need(len(calibrated) == 1, 'timing started without successful calibration')
                final_plan = calibrated[0]['rotation']
                cal = [r for r in part_exec if r['phase'] == 'rotation_calibration']
                need(cal and cal[-1]['guard_run_id'] == calibrated[0]['guard_run_id'] and
                     cal[-1]['candidate_id'] == 'original' and cal[-1]['status'] == 'ok' and
                     cal[-1]['workload']['graph_repeats'] == final_plan['graph_repeats'] and
                     min(cal[-1]['result']['raw_interval_ms_samples']) >= 2, 'uncovered timing lacks valid calibration')
                for record in pairs:
                    verify_pair(k, proof, timing, record, final_plan)
                tail = [r for r in timing if r['round'] == len(pairs)]
                need(len(timing) == 2 * len(pairs) + len(tail) and len(tail) in (1, 2) and
                     tail[-1]['status'] != 'ok' and tail[-1] is part_exec[-1], 'uncovered timing tail mismatch')
                order = runner.search.seeded_order(20261031, proof['plan_sha256'], *k, 'rotation', len(pairs))
                need([r['order'] for r in tail] == list(range(len(tail))) and
                     [r['candidate_id'] for r in tail] == ['original' if role == 'original' else k[2]
                                                            for role in order[:len(tail)]] and
                     all(r['workload']['graph_repeats'] == final_plan['graph_repeats'] for r in tail),
                     'uncovered timing order mismatch')
            else:
                need(not pairs and not calibrated, 'pair or calibration summary without timings')
            if status == 'uncovered':
                last_plan = dict(base, graph_repeats=part_exec[-1]['workload']['graph_repeats']) if part_exec else None
                need(part_exec and part_exec[-1]['status'] != 'ok' and runner.uncovered(part_exec[-1], gpu, case, last_plan) == disposition['reason'],
                     'uncovered reason not supported by final execution')
            else:
                need(part_exec and part_exec[-1]['phase'] == 'rotation_calibration' and
                     part_exec[-1]['workload']['graph_repeats'] == 2048 and
                     min(part_exec[-1]['result']['raw_interval_ms_samples']) < 2, 'calibration limit not proven')
            results.append({'n': k[0], 'batch': k[1], 'candidate_id': k[2], 'status': status,
                            'reason': disposition.get('reason')})
            continue
        calibrated = [r for r in part if r.get('record_type') == 'rotation_calibration_complete']
        need(len(calibrated) == 1 and key(calibrated[0]) == k and calibrated[0]['rotation'] == disposition['rotation'],
             'calibration summary mismatch')
        final_plan = calibrated[0]['rotation']
        calibration = [r for r in part_exec if r['phase'] == 'rotation_calibration']
        need(calibration and calibration[-1]['guard_run_id'] == calibrated[0]['guard_run_id'] and
             calibration[-1]['workload']['graph_repeats'] == final_plan['graph_repeats'] and
             min(calibration[-1]['result']['raw_interval_ms_samples']) >= 2, 'calibration evidence mismatch')
        expected_repeats = base['graph_repeats']
        for row in calibration:
            need(row['candidate_id'] == 'original' and row['workload']['graph_repeats'] == expected_repeats,
                 'calibration sequence mismatch')
            if row is not calibration[-1]:
                need(min(row['result']['raw_interval_ms_samples']) < 2, 'unnecessary calibration repeat')
            expected_repeats *= 2
        timing = [r for r in part_exec if r['phase'] == 'rotation_timing']
        pairs = [r for r in part if r.get('record_type') == 'rotation_pair']
        need(len(timing) == 20 and len(pairs) == 10 and {r['round'] for r in pairs} == set(range(10)),
             'ten rotation pairs are incomplete')
        ratios, measurements = [], []
        for round_no in range(10):
            record = next(r for r in pairs if r['round'] == round_no)
            ratio, measurement = verify_pair(k, proof, timing, record, final_plan)
            ratios.append(ratio); measurements.append(measurement)
        need(disposition['paired_ratios'] == ratios and math.isclose(disposition['median_ratio'], statistics.median(ratios), rel_tol=1e-12),
             'rotation disposition ratio mismatch')
        stats = eligibility.audit.audit_pairs.ten_round_session(measurements)
        results.append({'n': k[0], 'batch': k[1], 'candidate_id': k[2], 'status': 'measured',
                        'rotation': final_plan, 'statistics': stats})
    need(used == {r['guard_run_id'] for r in executions}, 'orphan rotation execution')
    measured = sum(r['status'] == 'measured' for r in results)
    uncovered_count = len(results) - measured
    expected_status = 'no_eligible_winner' if not results else 'complete_with_uncovered' if uncovered_count else 'complete'
    need(ends[0]['status'] == expected_status and ends[0]['case_count'] == len(results) and
         ends[0]['measured_count'] == measured and ends[0]['uncovered_count'] == uncovered_count,
         'session coverage summary mismatch')
    return {'schema': 'frontier-rotation-audit/v1', 'status': 'pass', 'gpu': gpu,
            'guard_audit': guard_report, 'results': results, 'measured_count': measured,
            'uncovered_count': uncovered_count,
            'scope': 'independent raw-evidence audit of separate multi-address timing; no cache-miss or cold-cache claim'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'hot-original', 'hot-expanded', 'rotation-original', 'rotation-expanded',
                 'rotation-root', 'generated-root', 'original-include-root', 'instrumentation-root',
                 'build-attestation', 'compile-commands', 'original-dependencies',
                 'campaign-record', 'campaign-start', 'admission-budget',
                 'rows', 'guard', 'incidents', 'budget', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    for side in ('screen', 'first', 'second'):
        for name in ('rows', 'guard', 'incidents', 'budget'):
            p.add_argument('--'+side+'-'+name, type=Path, required=True)
    p.add_argument('--gpu', required=True); p.add_argument('--confirmation-commit', required=True)
    p.add_argument('--rotation-experiment-id', required=True)
    a = p.parse_args()
    try:
        paths = {side: {name: getattr(a, side+'_'+name) for name in ('rows', 'guard', 'incidents', 'budget')}
                 for side in ('screen', 'first', 'second')}
        proof = eligibility.verify(a.plan, paths['screen'], paths['first'], paths['second'],
                                   {'original': a.hot_original, 'expanded': a.hot_expanded}, a.gpu, a.confirmation_commit)
        rows, guard, incidents, budget = (frontier.read_log(getattr(a, name)) for name in ('rows', 'guard', 'incidents', 'budget'))
        need(rows, 'missing rotation rows')
        root = Path(frontier.git(HERE, 'rev-parse', '--show-toplevel'))
        context = verify_context(root, rows[0]['contract'], budget, proof, a)
        report = audit_rows(rows, guard, incidents, budget, proof, a.gpu,
                            {'original': a.rotation_original, 'expanded': a.rotation_expanded})
        report['verified_context'] = context
        inputs = [a.plan, a.rows, a.guard, a.incidents, a.budget, a.hot_original, a.hot_expanded,
                  a.rotation_original, a.rotation_expanded, a.build_attestation, a.compile_commands,
                  a.original_dependencies, a.campaign_record, a.campaign_start, a.admission_budget,
                  *(path for side in paths.values() for path in side.values())]
        report['input_sha256'] = {str(Path(path).resolve()): frontier.sha(path) for path in inputs}
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError, OSError, StopIteration, subprocess.SubprocessError) as error:
        report = {'schema': 'frontier-rotation-audit/v1', 'status': 'fail', 'errors': [f'{type(error).__name__}: {error}']}
    with a.output.open('x', encoding='utf8') as out:
        json.dump(report, out, sort_keys=True, indent=2, allow_nan=False); out.write('\n')
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
