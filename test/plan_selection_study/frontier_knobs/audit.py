#!/usr/bin/env python3
"""Offline audit of one frontier-knobs session, including clean budget truncation.

This verifies the supplied bytes against the session contract; callers must also
authenticate those bytes against the completed experiment's artifact inventory.
No benchmark or GPU operation is performed.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('frontier_knobs_audit_runner', HERE / 'runner.py')
guard_audit = load('frontier_knobs_guard_audit', HERE.parent / 'expanded_search/audit_guard.py')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_log(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def check_ledger(ledger, executions, contract):
    need(ledger and ledger[0].get('event') == 'start', 'missing budget start')
    initial = ledger[0]['state']
    start, seconds, boot = initial['start'], initial['seconds'], initial['boot']
    need(type(start) in (int, float) and math.isfinite(start) and
         type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0 and
         isinstance(boot, str) and bool(boot) and initial['inflight'] is None, 'invalid budget origin')
    events = ['start'] + ['work_start', 'work_finish'] * len(executions)
    denied = ledger[-1].get('event') == 'denied'
    if denied: events.append('denied')
    need([x.get('event') for x in ledger] == events, 'budget execution sequence mismatch')
    last = start
    for entry in ledger:
        state = entry['state']; clock = state['last']
        need(state.get('contract') == contract and state.get('start') == start and
             state.get('seconds') == seconds and state.get('boot') == boot and
             type(clock) in (int, float) and math.isfinite(clock) and clock >= last,
             'budget contract or clock changed')
        last = clock
    for index, row in enumerate(executions):
        begin, finish = ledger[1 + 2 * index:3 + 2 * index]
        work = finish['identity']; w = row['workload']
        need(begin['state']['inflight'] == work and finish['state']['inflight'] is None and
             finish['status'] == row['status'] and
             work == {'n': w['n'], 'batch': w['batch'], 'candidate_id': row['candidate_id'],
                      'phase': row['phase'], 'round': row['round']}, 'budget work differs from execution')
    need(ledger[-1]['state']['inflight'] is None, 'budget has unfinished work')
    if denied:
        need(last + contract['timeout'] + 30 >= start + seconds, 'premature budget denial')
    return ledger[-1].get('identity') if denied else None


class BudgetBoundary(Exception):
    pass


def audit(rows, guard, incidents, ledger, manifest, gpu, binaries):
    """Audit one exact runner trace; budget-denied work remains unmeasured."""
    need(rows, 'missing rows')
    contract = rows[0]['contract']
    need(contract.get('protocol') == 'frontier-knobs-execution/v1' and contract.get('gpu') == gpu and
         contract.get('binaries') == binaries and type(contract.get('timeout')) is int and contract['timeout'] > 0,
         'execution contract, GPU or binary hash mismatch')
    need(all(row.get('contract') == contract for row in rows), 'mixed contracts')
    result = guard_audit.audit(rows, guard, incidents, gpu)
    need(result.get('status') == 'pass', 'guard audit failed: ' + str(result.get('errors')))
    executions = [row for row in rows if row.get('record_type') == 'execution']
    denied = check_ledger(ledger, executions, contract)
    need((rows[-1].get('record_type') == 'budget_exhausted') == (denied is not None),
         'budget denial and terminal row disagree')
    n, batch, candidates = manifest
    gid = guard[0]['guard_id']
    sessions = {row.get('session_id') for row in rows if row.get('record_type') != 'budget_exhausted'}
    need(len(sessions) <= 1 and None not in sessions and
         all(row.get('guard_id') == gid for row in rows), 'mixed session or guard identity')
    index = 0
    outcomes = []
    admitted = set()
    active = None

    def take(kind):
        nonlocal index
        need(index < len(rows), 'truncated evidence without a budget boundary')
        row = rows[index]
        need(row.get('record_type') == kind, 'unexpected evidence order: expected ' + kind)
        index += 1
        return row

    def execute(name, phase, *, request=None, lut=0, coalesced=0, graph=64,
                pattern='random', seed=42, timing=False, mode='execute', round_no=0, order=0):
        nonlocal index
        identity = {'n': n, 'batch': batch, 'candidate_id': name, 'phase': phase, 'round': round_no}
        if index < len(rows) and rows[index].get('record_type') == 'budget_exhausted':
            need(denied == identity and index == len(rows) - 1, 'denied work identity or boundary mismatch')
            index += 1
            raise BudgetBoundary()
        row = take('execution')
        knob_args = ['--use-lut', str(lut), '--coalesced-memory', str(coalesced)] if lut or coalesced else []
        env = {}
        if request is not None:
            inv = runner.requests.invocation(n, batch, request)
            knob_args, env = inv['args'], inv['env']
        workload = {'n': n, 'batch': batch, 'seed': seed, 'pattern': pattern, 'graph_repeats': graph,
                    'warmup': 5 if timing else 0, 'warmup_ms': 1000 if timing else 0,
                    'repeats': 31 if timing else 1, 'study_mode': mode, 'knob_args': knob_args}
        need(row.get('workload') == workload and row.get('workload_id') == f'n{n}_b{batch}' and
             row.get('candidate_id') == name and row.get('phase') == phase and
             row.get('round') == round_no and row.get('order') == order and
             row.get('direct_environment') == env and
             row.get('executable_sha256') == binaries['original' if name == 'original' else 'expanded'],
             'execution identity, workload or binary mismatch')
        command = row['command']
        expected = ['--n', str(n), '--batch', str(batch), '--study-mode', mode, '--placement', 'outofplace']
        if mode == 'execute':
            expected += ['--seed', str(seed), '--timing-mode', 'graph', '--graph-repeats', str(graph),
                         '--warmup', str(workload['warmup']), '--warmup-ms', str(workload['warmup_ms']),
                         '--repeats', str(workload['repeats']), '--pattern', pattern]
        need(isinstance(command, list) and command[1:] == expected + knob_args and
             [json.loads(line) for line in row['stdout'].splitlines() if line.strip()] == [row['result']],
             'execution command or raw stdout mismatch')
        need(not runner.numerical_failure(row), 'numerical failure in execution evidence')
        return row

    def checked(name, phase, **kwargs):
        row = execute(name, phase, **kwargs)
        result = runner.validation.check_execute(row, gpu, lut=kwargs.get('lut', 0),
                    coalesced=kwargs.get('coalesced', 0), original=name == 'original',
                    request=kwargs.get('request'), timing=kwargs.get('timing', False))
        return row, result

    boundary = False
    try:
        _, baseline = checked('original', 'baseline')
        base_sig = runner.search.source_signature(baseline, original=True)
        graph = 64
        while True:
            _, value = checked('original', 'calibration', graph=graph)
            need(runner.search.source_signature(value, original=True) == base_sig, 'baseline plan drift')
            if min(value['raw_interval_ms_samples']) >= 2: break
            need(graph < 2048, 'calibration exceeded protocol')
            graph *= 2
        row = take('baseline_admitted')
        need((row.get('n'), row.get('batch'), row.get('graph_repeats'), row.get('performance_denominator')) ==
             (n, batch, graph, 'original_no_knob_default'), 'baseline admission mismatch')
        for item in candidates:
            req, cid = item['request'], item['candidate_id']
            lut, coal = req['use_lut'], req['coalesced_memory']
            group = (lut, coal)
            if group not in admitted:
                ro, original = checked('original', 'matched_identity', lut=lut, coalesced=coal, graph=graph)
                rd, default = checked('experimental_default', 'matched_identity', lut=lut, coalesced=coal, graph=graph)
                runner.validation.comparison.compare(default, original)
                identity, reason = runner.v2.representable(default)
                need(identity is not None, 'unsupported same-knob identity: ' + str(reason))
                identity.update(use_lut=lut, coalesced_memory=coal)
                rr, _ = checked('identity', 'matched_identity', request=identity, lut=lut, coalesced=coal, graph=graph)
                runner.validation.check_matched_identity(ro, rd, rr, gpu, identity)
                matched = take('matched_identity_passed')
                need(matched.get('knobs') == list(group) and matched.get('identity') == identity,
                     'matched identity record mismatch')
                admitted.add(group)
            active = item
            candidate = take('candidate_start')
            need((candidate.get('n'), candidate.get('batch'), candidate.get('candidate_id'), candidate.get('request')) ==
                 (n, batch, cid, req), 'candidate order or request mismatch')
            row = execute(cid, 'describe', request=req, lut=lut, coalesced=coal, graph=graph, mode='describe')
            reason = runner.scheduler_rejection(row, gpu)
            if reason: need(row.get('status') == 'failed', 'rejection status mismatch')
            if not reason:
                described = runner.validation.check_describe(row, gpu, req)
                dp = described['precompile_kernels'][0]
                signature = None
                for pattern in runner.PATTERNS:
                    for seed in runner.SEEDS:
                        row = execute(cid, 'validation', request=req, lut=lut, coalesced=coal,
                                      graph=graph, pattern=pattern, seed=seed)
                        reason = runner.scheduler_rejection(row, gpu)
                        if reason: need(row.get('status') == 'failed', 'rejection status mismatch')
                        if reason: break
                        value = runner.validation.check_execute(row, gpu, lut=lut, coalesced=coal, request=req)
                        cp = value['precompile_kernels'][0]
                        need(all(cp[k] == dp[k] for k in ('canonical_descriptor', 'source_sha256', 'source_bytes', 'compiler_input')),
                             'describe and validation code differ')
                        sig = runner.search.source_signature(value, original=False)
                        need(signature is None or signature == sig, 'validation plan drift')
                        signature = sig
                    if reason: break
            if reason:
                disposition = take('candidate_disposition')
                need(disposition.get('candidate_id') == cid and disposition.get('status') == 'scheduler_rejected' and
                     disposition.get('reason') == reason, 'scheduler rejection disposition mismatch')
                outcomes.append({'candidate_id': cid, 'request': req, 'status': 'scheduler_rejected', 'reason': reason})
                active = None
                continue
            ratios = []
            for round_no in range(3):
                pair = {}
                for order, role in enumerate(runner.v2.seeded_order(n, batch, cid, round_no)):
                    name = 'original' if role == 'original' else cid
                    _, value = checked(name, 'timing', request=None if role == 'original' else req,
                        lut=0 if role == 'original' else lut, coalesced=0 if role == 'original' else coal,
                        graph=graph, timing=True, round_no=round_no, order=order)
                    need(value['warmup_actual'] >= 5 and value['warmup_wall_ms'] >= 1000 and
                         runner.search.source_signature(value, original=role == 'original') ==
                         (base_sig if role == 'original' else signature), 'timing warmup or plan drift')
                    pair[role] = value
                ratio = statistics.median(pair['candidate']['exec_ms_samples']) / statistics.median(pair['original']['exec_ms_samples'])
                need(math.isfinite(ratio) and ratio > 0, 'invalid timing ratio')
                record = take('screen_pair')
                need(record.get('candidate_id') == cid and record.get('round') == round_no and
                     record.get('denominator') == 'original_no_knob_default' and
                     math.isclose(record['ratio'], ratio, rel_tol=1e-12), 'screen pair ratio mismatch')
                ratios.append(ratio)
            disposition = take('candidate_disposition')
            need(disposition.get('candidate_id') == cid and disposition.get('request') == req and
                 disposition.get('status') == 'screen_complete' and disposition.get('pair_ratios') == ratios and
                 disposition.get('median_ratio') == statistics.median(ratios), 'candidate disposition mismatch')
            outcomes.append({'candidate_id': cid, 'request': req, 'status': 'screen_complete',
                             'pair_ratios': ratios, 'median_ratio': statistics.median(ratios)})
            active = None
    except BudgetBoundary:
        boundary = True
    need(index == len(rows) and boundary == (denied is not None), 'extra evidence or unused budget denial')
    completed = len(outcomes)
    if active is not None:
        outcomes.append({'candidate_id': active['candidate_id'], 'request': active['request'], 'status': 'partial'})
    for item in candidates[completed + (1 if active is not None else 0):]:
        outcomes.append(dict(item, status='not_started'))
    return {'schema': 'frontier-knobs-audit/v1', 'status': 'pass', 'gpu': gpu,
            'execution_commit': contract.get('execution_commit'), 'workload': {'n': n, 'batch': batch},
            'budget_boundary': boundary, 'candidates': outcomes, 'guard': result,
            'scope': {'performance': 'screening against original no-knob default, not independent confirmation',
                      'rejections': 'scheduler rejection lacks an independent post-rejection control',
                      'guard': 'periodic sampling cannot rule out between-sample interference'}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('rows', 'guard', 'incidents', 'budget', 'manifest', 'original', 'expanded',
                 'generated-provenance', 'build-attestation', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpu', required=True)
    a = p.parse_args()
    paths = {name: getattr(a, name.replace('-', '_')) for name in
             ('rows', 'guard', 'incidents', 'budget', 'manifest', 'original', 'expanded',
              'generated-provenance', 'build-attestation')}
    hashes = {name: sha(path) for name, path in paths.items()}
    rows, guard, incidents, ledger = (read_log(paths[name]) for name in ('rows', 'guard', 'incidents', 'budget'))
    need(rows, 'missing rows')
    contract = rows[0]['contract']
    binaries = {key: hashes[key] for key in ('original', 'expanded')}
    need((contract.get('manifest') or {}).get('sha256') == hashes['manifest'] and
         (contract.get('generated_provenance') or {}).get('sha256') == hashes['generated-provenance'] and
         (contract.get('build_attestation') or {}).get('sha256') == hashes['build-attestation'],
         'manifest, generated provenance or attestation hash differs')
    attestation = runner.verify_attestation(a.build_attestation, contract['code_commit'], contract['vkfft_commit'],
                                             hashes['generated-provenance'], binaries)
    need(attestation['document'] == contract['build_attestation']['document'], 'attestation document differs')
    report = audit(rows, guard, incidents, ledger, runner.load_manifest(a.manifest), a.gpu, binaries)
    need({name: sha(path) for name, path in paths.items()} == hashes, 'input changed during audit')
    report['evidence_sha256'] = hashes
    with a.output.open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    print(json.dumps({'status': report['status'], 'budget_boundary': report['budget_boundary'], 'output': str(a.output)}))


if __name__ == '__main__':
    main()
