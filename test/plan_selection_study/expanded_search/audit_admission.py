#!/usr/bin/env python3
"""Independent offline admission evidence audit; never runs a GPU command."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def load(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def check(condition, message, errors):
    if not condition:
        errors.append(message)


def run_id(event):
    return event.get('guard_run_id') or (event.get('active_run') or {}).get('guard_run_id')


def default_identity(result):
    """Independently decide whether the uniform-register prototype can express it."""
    f = result['structural_features']; k = f['kernels'][0]
    d = result['precompile_kernels'][0]['raw_descriptor']
    n, batch = result['n'], result['batch']
    r, group, seq = d['logical_registers_per_thread'], k['groupedBatch'], k['stageRadix']
    if f['numAxisUploads'] != 1 or len(f['kernels']) != 1 or d['registerBoost'] != 1:
        return None
    if r not in (2, 4, 8, 16, 32, 64) or d['min_logical_registers_per_thread'] != r or n % r:
        return None
    expected = [r if x in (2, 4, 8, 16, 32) and r % x == 0 else 0 for x in range(33)]
    if d['registers_per_radix'] != expected or not seq or math.prod(seq) != n or any(x not in (2, 4, 8, 16, 32) or x > r or r % x for x in seq):
        return None
    if group < 1 or group & (group-1) or batch % group:
        return None
    local, swap = k['localSize'], d['axisSwapped']
    if swap not in (0, 1) or len(local) != 3 or local[2] != 1 or local[1 if swap else 0] != n // r or local[0 if swap else 1] != group:
        return None
    return {'radix': seq, 'logical_registers': r, 'grouped_batch': group}


def audit(rows, guard, incidents, protocol, protocol_sha256=None):
    errors = []
    expected_gpu = protocol['gpu_uuid']
    headers = [x for x in rows if x.get('record_type') == 'admission_protocol']
    terminals = [x for x in rows if x.get('record_type') == 'admission_complete']
    starts = [x for x in guard if x.get('event') == 'guard_start']
    ends = [x for x in guard if x.get('event') == 'guard_complete']
    check(len(headers) == len(terminals) == len(starts) == len(ends) == 1, 'missing or repeated protocol, terminal or guard boundary', errors)
    if not headers or not starts:
        return {'status': 'fail', 'errors': errors, 'counts': {}}
    header, start = headers[0], starts[0]
    check(header.get('protocol') == protocol, 'logged admission protocol differs from preregistration', errors)
    if protocol_sha256 is not None:
        check(header.get('protocol_sha256') == protocol_sha256, 'protocol file hash differs from preregistration', errors)
    guard_id = start.get('guard_id')
    check(start.get('gpu_uuid') == expected_gpu and bool(guard_id), 'guard start GPU or identity mismatch', errors)
    check(start.get('own_signal_method') == 'pidfd_with_rechecked_PPID_exe_starttime', 'guard process ownership method mismatch', errors)
    check(not incidents, 'guard incident log is nonempty', errors)
    check(not any(g.get('incident') or g.get('event') in ('guard_abort', 'quarantined_run', 'termination_error') for g in guard), 'guard incident or abort', errors)
    if ends:
        check(ends[0].get('success') is True and ends[0].get('clean') is True and ends[0].get('guard_id') == guard_id, 'guard did not finish cleanly', errors)
    for event in guard:
        check(event.get('guard_id') == guard_id and event.get('gpu_uuid') == expected_gpu, 'guard event identity mismatch', errors)
        if event.get('event') == 'sample':
            check(event.get('ok') is True and not event.get('violation_reasons') and not event.get('foreign_compute_pids') and
                  event.get('monitor_error') is None and event.get('gpu', {}).get('uuid') == expected_gpu,
                  'unclean guard sample', errors)
    events = defaultdict(list)
    for event in guard:
        rid = run_id(event)
        if rid:
            events[rid].append(event)
    executions = [r for r in rows if r.get('record_type') == 'execution']
    run_ids, invocation_ids = set(), set()
    by_workload = defaultdict(list)
    counts, precode_changes = Counter(), Counter()
    for row in executions:
        rid = row.get('guard_run_id'); iid = row.get('invocation_id')
        check(bool(rid) and rid not in run_ids and bool(iid) and iid not in invocation_ids, 'missing or duplicate run ID', errors)
        run_ids.add(rid); invocation_ids.add(iid)
        check(row.get('guard_clean') is True and row.get('guard_gpu_uuid') == expected_gpu and row.get('guard_id') == guard_id, f'{rid}: guard row mismatch', errors)
        ev = events.get(rid, [])
        for kind in ('run_one_start', 'run_one_end', 'own_harness_registered'):
            check(sum(x.get('event') == kind for x in ev) == 1, f'{rid}: missing/duplicate {kind}', errors)
        for boundary in ('before', 'after'):
            check(sum(x.get('event') == 'sample' and x.get('boundary') == boundary for x in ev) == 1, f'{rid}: missing/duplicate {boundary} sample', errors)
        registered = [x for x in ev if x.get('event') == 'own_harness_registered']
        if registered:
            check(registered[0].get('pidfd_retained') is True and registered[0].get('identity', {}).get('exe') == row.get('command', [None])[0], f'{rid}: child identity mismatch', errors)
        try:
            parsed = [json.loads(x) for x in row.get('stdout', '').splitlines() if x.strip()]
            check(parsed == [row['result']], f'{rid}: stdout differs from parsed result', errors)
            w, r = row['workload'], row['result']
            key = (w['n'], w['batch'])
            check(row.get('workload_id') == f"n{key[0]}_b{key[1]}", f'{rid}: workload ID mismatch', errors)
            by_workload[key].append(row)
            if row.get('returncode') != 0:
                counts['failed_or_rejected_process'] += 1
                continue
            check(row.get('status') == 'ok' and r.get('status') == 'ok', f'{rid}: non-success result', errors)
            check(r.get('gpu', {}).get('uuid') == expected_gpu and r.get('gpu', {}).get('cuda_visible_devices') == expected_gpu, f'{rid}: result GPU mismatch', errors)
            check((r.get('n'), r.get('batch')) == key, f'{rid}: result workload mismatch', errors)
            binary = 'original' if row.get('candidate_id') == 'original' else 'expanded'
            check(row.get('executable_sha256') == header.get('sha256', {}).get(binary), f'{rid}: executable hash mismatch', errors)
            check(r.get('header_instrumentation') is (binary == 'expanded'), f'{rid}: instrumentation mismatch', errors)
            if binary == 'expanded':
                p = r['precompile_kernels'][0]
                pre, post = p['precodegen_raw_descriptor'], p['raw_descriptor']
                check(pre.keys() == post.keys(), f'{rid}: pre-codegen and precompile descriptor schemas differ', errors)
                for field in pre.keys() & post.keys():
                    if pre[field] != post[field]:
                        precode_changes[field] += 1
                counts['precode_descriptor_pairs'] += 1
            env = row.get('direct_environment', {})
            if env:
                check(set(env) == {'VKFFT_EXPANDED_RADIX', 'VKFFT_EXPANDED_REGS', 'VKFFT_EXPANDED_BATCH'}, f'{rid}: override fields mismatch', errors)
                requested_radix = list(map(int, env['VKFFT_EXPANDED_RADIX'].split(',')))
                regs, group = int(env['VKFFT_EXPANDED_REGS']), int(env['VKFFT_EXPANDED_BATCH'])
                requested = {'radix': requested_radix, 'logical_registers': regs, 'grouped_batch': group}
                requested_id = hashlib.sha256(json.dumps(requested, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                check(row.get('candidate_id') == 'identity' or row.get('candidate_id') == requested_id,
                      f'{rid}: candidate ID does not match request', errors)
                k = r['structural_features']['kernels'][0]
                d = r['precompile_kernels'][0]['raw_descriptor']
                swap = d['axisSwapped']; local = k['localSize']
                check(k['stageRadix'] == requested_radix and k['groupedBatch'] == group and
                      d['logical_registers_per_thread'] == d['min_logical_registers_per_thread'] == regs and d['registerBoost'] == 1 and
                      swap in (0, 1) and local[2] == 1 and local[1 if swap else 0] == w['n']//regs and local[0 if swap else 1] == group,
                      f'{rid}: actual plan differs from request', errors)
            if w.get('study_mode') == 'describe':
                check(r.get('describe_only') is True, f'{rid}: not a describe result', errors)
                counters = r['init_trace']['counters']
                check(all(counters.get(x) == 0 for x in ('nvrtc_compile', 'module_load', 'driver_kernel_launch')), f'{rid}: describe executed compile/launch', errors)
                counts['describe_success'] += 1
                continue
            check((r.get('seed'), r.get('pattern')) == (w['seed'], w['pattern']), f'{rid}: input mismatch', errors)
            check(r.get('single_forward_input_unchanged') is True and r.get('input_unchanged') is True, f'{rid}: input changed', errors)
            check(r.get('validation_output_initialization') == '0xff_bytes_fp32_nan_before_single_and_final_timed_forward', f'{rid}: output poison missing', errors)
            for name in ('single_forward_correctness', 'correctness'):
                c = r[name]
                ref = ('independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity' if w['pattern'] == 'random'
                       else 'analytic_pattern_reference_with_independent_CPU_FFT_crosscheck')
                check(c.get('passed') is True and c.get('nonfinite_count') == 0 and c.get('failing_batches') == 0 and
                      c.get('checked_elements') == w['n'] * w['batch'] and c.get('reference') == ref and c.get('criterion') == 'criterion_v2' and
                      math.isclose(c['l2_scaled_error_threshold'], 1e-5, rel_tol=1e-12) and math.isclose(c['linf_input_scaled_error_threshold'], 1e-4, rel_tol=1e-12),
                      f'{rid}: {name} incomplete or failed', errors)
            if row.get('direct_environment'):
                limit = r['expanded_resource_validation']
                check(0 <= limit['dynamic_shared_bytes'] <= limit['function_dynamic_limit_bytes'] and 0 <= limit['static_shared_bytes'] and
                      limit['dynamic_shared_bytes'] + limit['static_shared_bytes'] <= limit['device_shared_limit_bytes'] and
                      limit['threads'] <= limit['function_thread_limit'], f'{rid}: compiled resources invalid', errors)
            counts['execute_success'] += 1
        except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
            errors.append(f'{rid}: incomplete or malformed evidence: {exc}')
    for disposition in [r for r in rows if r.get('record_type') == 'candidate_disposition']:
        key = (disposition.get('n'), disposition.get('batch'))
        candidate = disposition.get('candidate_id')
        if disposition.get('status') == 'numerically_valid':
            runs = [r for r in by_workload[key] if r.get('candidate_id') == candidate and r.get('returncode') == 0 and r['workload'].get('study_mode') == 'execute']
            observed = Counter((r['workload']['pattern'], r['workload']['seed']) for r in runs)
            want = {(p, s): 1 for p in protocol['patterns'] for s in protocol['validation_seeds']}
            check(dict(observed) == want and disposition.get('checks') == len(want), f'{key}/{candidate}: claimed numerical coverage incomplete', errors)
        counts['disposition_' + str(disposition.get('status'))] += 1
    for status in [r for r in rows if r.get('record_type') == 'workload_status']:
        counts['workload_' + str(status.get('status'))] += 1
        key = (status.get('n'), status.get('batch'))
        originals = [r['result'] for r in by_workload[key] if r.get('candidate_id') == 'original' and r.get('returncode') == 0]
        defaults = [r['result'] for r in by_workload[key] if r.get('candidate_id') == 'expanded_default' and r.get('returncode') == 0]
        identities = [r['result'] for r in by_workload[key] if r.get('candidate_id') == 'identity' and r.get('returncode') == 0]
        check(bool(originals) and len(defaults) == 1, f'{key}: original/expanded default controls missing', errors)
        if originals and defaults:
            try:
                o, d = originals[0], defaults[0]
                op, dp = o['postinit_source_kernels'][0], d['precompile_kernels'][0]
                dc, oc = dict(d['resolved_config']), dict(o['resolved_config'])
                check(dc.pop('keepShaderCode', None) == 0 and oc.pop('keepShaderCode', None) == 1 and dc == oc,
                      f'{key}: original and expanded resolved config mismatch', errors)
                check(d['structural_features'] == o['structural_features'] and d['compiled_diagnostics'] == o['compiled_diagnostics'] and
                      dp['source_sha256'] == op['source_sha256'] and dp['source_bytes'] == op['source_bytes'] and dp['canonical_descriptor'] == op['canonical_descriptor'] and
                      op['expected_compiler_input']['options'] == dp['actual_compiler_options'] and
                      op['expected_compiler_input']['original_compile_header_sha256'] == dp['compiler_input']['original_compile_header_sha256'],
                      f'{key}: original/expanded controls differ', errors)
                if status.get('status') == 'admission_exercised':
                    expected_identity = default_identity(d)
                    check(expected_identity is not None, f'{key}: exercised though default is not representable', errors)
                    check(len(identities) == 1, f'{key}: identity control missing', errors)
                    if identities:
                        i = identities[0]; ip = i['precompile_kernels'][0]
                        identity_rows = [r for r in by_workload[key] if r.get('candidate_id') == 'identity' and r.get('returncode') == 0]
                        expected_env = {'VKFFT_EXPANDED_RADIX': ','.join(map(str, expected_identity['radix'])),
                                        'VKFFT_EXPANDED_REGS': str(expected_identity['logical_registers']),
                                        'VKFFT_EXPANDED_BATCH': str(expected_identity['grouped_batch'])} if expected_identity else None
                        check(identity_rows[0].get('direct_environment') == expected_env, f'{key}: identity request differs from baseline', errors)
                        check(i['structural_features'] == d['structural_features'] and i['compiled_diagnostics'] == d['compiled_diagnostics'] and i['resolved_config'] == d['resolved_config'] and
                              all(ip[x] == dp[x] for x in ('canonical_descriptor', 'source_sha256', 'source_bytes', 'compiler_input', 'actual_compiler_options')),
                              f'{key}: identity differs from expanded default', errors)
                elif status.get('status') == 'unsupported_identity':
                    check(not identities and default_identity(d) is None, f'{key}: incorrect unsupported identity classification', errors)
                else:
                    errors.append(f'{key}: unknown admission status')
            except (KeyError, TypeError, IndexError) as exc:
                errors.append(f'{key}: control evidence incomplete: {exc}')
        if status.get('status') == 'admission_exercised':
            registered = next((w['requests'] for w in protocol['workloads'] if (w['n'], w['batch']) == key), [])
            expected_requests = {hashlib.sha256(json.dumps(req, sort_keys=True, separators=(',', ':')).encode()).hexdigest(): req for req in registered}
            dispositions = [r for r in rows if r.get('record_type') == 'candidate_disposition' and (r.get('n'), r.get('batch')) == key]
            actual = {r.get('candidate_id'): r.get('request') for r in dispositions}
            check(len(actual) == len(dispositions) and actual == expected_requests, f'{key}: preregistered candidate disposition missing or duplicated', errors)
            for item in dispositions:
                cid = item['candidate_id']
                related = [r for r in by_workload[key] if r.get('candidate_id') == cid]
                check(bool(related), f'{key}/{cid}: disposition without execution evidence', errors)
                if item.get('status') == 'numerically_valid':
                    check(any(r['workload'].get('study_mode') == 'describe' and r.get('returncode') == 0 for r in related),
                          f'{key}/{cid}: numerically valid without successful describe', errors)
                elif item.get('status') not in ('scheduler_or_resource_rejected', 'compiled_resource_or_scheduler_rejected'):
                    errors.append(f'{key}/{cid}: unknown disposition status')
                else:
                    known = {'initializeVkFFT failed with code 3002', 'expanded candidate exceeds compiled function or device resource limit'}
                    check(any(r.get('returncode') == 1 and r.get('result', {}).get('status') == 'error' and
                              r.get('result', {}).get('failure_phase') == 'initialize' and r.get('result', {}).get('error') in known
                              for r in related), f'{key}/{cid}: rejection lacks a recognized failure', errors)
    statuses = [r for r in rows if r.get('record_type') == 'workload_status']
    observed = {(r.get('n'), r.get('batch')) for r in statuses}
    expected = {(r['n'], r['batch']) for r in protocol['workloads']}
    check(len(observed) == len(statuses) and observed <= expected, 'duplicate or unregistered workload status', errors)
    incomplete = observed != expected or not terminals or terminals[0].get('status') != 'completed_cases'
    return {'status': 'fail' if errors else 'incomplete' if incomplete else 'pass', 'errors': errors, 'counts': dict(counts),
            'precode_changed_fields': dict(precode_changes),
            'missing_workloads': sorted(expected - observed),
            'scope': 'admission log and guard only; inspect unsupported, rejected and missing work separately; no performance claim',
            'sampling_caveat': 'periodic GPU sampling cannot exclude brief interference between samples'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=Path, required=True)
    p.add_argument('--guard', type=Path, required=True)
    p.add_argument('--incidents', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    a = p.parse_args()
    result = audit(load(a.rows), load(a.guard), load(a.incidents), json.loads(a.protocol.read_text(encoding='utf8')),
                   hashlib.sha256(a.protocol.read_bytes()).hexdigest())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
