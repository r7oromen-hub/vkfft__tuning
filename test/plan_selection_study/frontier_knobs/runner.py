#!/usr/bin/env python3
"""Small, guarded joint-knob admission and paired screening; Linux CUDA only."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


requests = load('knob_requests_controller', HERE / 'requests.py')
validation = load('knob_validation_controller', HERE / 'validation.py')
v2 = load('knob_v2_execution_controller', HERE.parent / 'frontier_execution' / 'runner.py')
guard_common = v2.guard_common
search = v2.search_controller

PATTERNS = ('zero', 'impulse', 'constant', 'quarter-tone', 'random')
SEEDS = (42, 53, 71)
FIELDS = ('VKFFT_DIRECT_RADIX', 'VKFFT_DIRECT_BATCH', 'VKFFT_DIRECT_BLOCK',
          'VKFFT_EXPANDED_RADIX', 'VKFFT_EXPANDED_REGS', 'VKFFT_EXPANDED_BATCH')
MAX_CANDIDATES = 32


def load_manifest(path):
    doc = json.loads(Path(path).read_text(encoding='utf8'))
    if set(doc) != {'schema', 'workload', 'candidates'} or doc['schema'] != 'frontier-knobs-screen/v1':
        raise ValueError('unexpected small-screen manifest schema')
    work = doc['workload']
    if not isinstance(work, dict) or set(work) != {'n', 'batch'}:
        raise ValueError('invalid manifest workload')
    n, batch = work['n'], work['batch']
    rows = doc['candidates']
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CANDIDATES:
        raise ValueError('small-screen manifest must contain 1..32 candidates')
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'candidate_id', 'request'}:
            raise ValueError('invalid manifest row')
        req = row['request']
        if requests.static_reason(n, batch, req):
            raise ValueError('manifest contains static rejection')
        cid = requests.candidate_id(n, batch, req)
        if row['candidate_id'] != cid or cid in seen:
            raise ValueError('invalid or duplicate candidate ID')
        seen.add(cid)
    return n, batch, rows


def verify_generated(root, path, vkfft_commit):
    verified = v2.verify_generated(root, path, vkfft_commit)
    doc = json.loads(Path(path).read_text(encoding='utf8'))
    knobs = doc.get('frontier_knobs', {})
    source = Path(root) / 'test/plan_selection_study/frontier_knobs/prepare_headers.py'
    if (knobs.get('schema') != 'frontier-knobs-generated/v1' or
        knobs.get('generator_sha256') != v2.sha(source) or
        knobs.get('parent_v2_generator_sha256') != v2.sha(Path(root) / 'test/plan_selection_study/frontier_search/prepare_headers.py') or
        knobs.get('generated_benchmark_sha256') != v2.sha(Path(path).parent / 'expanded_benchmark.cpp')):
        raise ValueError('joint-knob generated provenance mismatch')
    return verified


def verify_declaration(root, execution_commit, experiment_id):
    """The run declaration is created after binaries were built from its parent."""
    if not experiment_id or not all(ch.isalnum() or ch in '_.-' for ch in experiment_id):
        raise ValueError('invalid experiment ID')
    parent = v2.git(root, 'rev-parse', execution_commit + '^')
    record = f'experiments/records/{experiment_id}.json'
    changed = v2.git(root, 'diff', '--name-only', parent, execution_commit).splitlines()
    if changed != [record]:
        raise ValueError('execution commit is not a single run declaration')
    doc = json.loads(v2.git(root, 'show', execution_commit + ':' + record))
    if (doc.get('schema') != 'git-experiment/v1' or doc.get('id') != experiment_id or
        doc.get('parent_code_commit') != parent or doc.get('status') != 'declared'):
        raise ValueError('execution declaration does not bind parent code commit')
    return parent


def verify_attestation(path, code_commit, upstream_commit, generated_sha, originals):
    """Bind declared build inputs to exact source and binary bytes; retain the attestation."""
    doc = json.loads(Path(path).read_text(encoding='utf8'))
    if (doc.get('schema') != 'frontier-knobs-build/v1' or doc.get('code_commit') != code_commit or
        doc.get('vkfft_commit') != upstream_commit or doc.get('generated_provenance_sha256') != generated_sha or
        doc.get('binaries') != originals or not isinstance(doc.get('build_commands'), list) or
        not doc['build_commands'] or not all(isinstance(x, str) and x for x in doc['build_commands'])):
        raise ValueError('build attestation does not bind sources, commands and binary hashes')
    return {'path': str(Path(path).resolve()), 'sha256': v2.sha(Path(path)), 'document': doc,
            'limitation': 'attested build commands and binary/source hash association, not independent reproduction'}


def run_one(args, workload, candidate, phase, round_no, order):
    """One child process; wrapper provides device monitoring and evidence boundaries."""
    name, override = candidate
    mode = workload['study_mode']
    command = [str(args.exe), '--n', str(workload['n']), '--batch', str(workload['batch']),
               '--study-mode', mode, '--placement', 'outofplace']
    if mode == 'execute':
        command += ['--seed', str(workload['seed']), '--timing-mode', 'graph',
                    '--graph-repeats', str(workload['graph_repeats']), '--warmup', str(workload['warmup']),
                    '--warmup-ms', str(workload['warmup_ms']), '--repeats', str(workload['repeats']),
                    '--pattern', workload['pattern']]
    elif mode != 'describe':
        raise ValueError('unsupported mode')
    command += workload['knob_args']
    env = os.environ.copy()
    for key in FIELDS:
        env.pop(key, None)
    if set(override) - set(FIELDS):
        raise ValueError('unrecognized override')
    env.update(override)
    start = time.monotonic()
    row = dict(record_type='execution', session_id=args.session_id, invocation_id=uuid.uuid4().hex,
               phase=phase, workload_id=f"n{workload['n']}_b{workload['batch']}", workload=workload,
               candidate_id=name, round=round_no, order=order, command=command, direct_environment=override,
               executable_sha256=args.exe_sha256, validation_errors=[], started_monotonic=start,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        if v2.sha(Path(args.exe)) != args.exe_sha256:
            raise RuntimeError('frozen binary changed')
        proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=args.timeout, check=False)
        row.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        parsed = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        if len(parsed) != 1 or not isinstance(parsed[0], dict):
            raise ValueError('expected one JSON object on stdout')
        row['result'] = parsed[0]
        row['status'] = 'ok' if proc.returncode == 0 else 'failed'
        if proc.returncode:
            row['validation_errors'].append(f'harness_exit:{proc.returncode}')
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        row.update(status='failed', error=f'{type(error).__name__}: {error}')
        row['validation_errors'].append(row['error'])
        if isinstance(error, subprocess.TimeoutExpired):
            row.update(stdout=str(error.stdout or ''), stderr=str(error.stderr or ''))
    row['ended_monotonic'] = time.monotonic()
    row['wall_seconds'] = row['ended_monotonic'] - start
    row['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return row


def numerical_failure(row):
    result = row.get('result') or {}
    return (result.get('status') == 'correctness_failed' or
            (result.get('single_forward_correctness') or {}).get('passed') is False or
            (result.get('correctness') or {}).get('passed') is False)


def scheduler_rejection(row, gpu):
    result = row.get('result') or {}
    reason = result.get('expanded_rejection_reason')
    marker = 'VKFFT_FRONTIER_REJECTION:' + json.dumps({'reason': reason}, separators=(',', ':'))
    if (row.get('guard_clean') is True and row.get('guard_gpu_uuid') == gpu and row.get('returncode') == 1 and
        result.get('status') == 'error' and result.get('failure_phase') == 'initialize' and
        result.get('error') == 'initializeVkFFT failed with code 3002' and
        isinstance(reason, str) and reason and marker in row.get('stderr', '').splitlines()):
        return reason
    return None


def screen(args, ledger, contract, manifest):
    if os.environ.get('VKFFT_STUDY_GUARD_UUID') != args.gpu or not os.environ.get('VKFFT_STUDY_GUARD_ID'):
        raise ValueError('live GPU guard required')
    n, batch, candidates = manifest
    args.session_id = uuid.uuid4().hex

    def emit(row):
        v2.append(args.output, dict(row, session_id=args.session_id, guard_id=os.environ['VKFFT_STUDY_GUARD_ID'], contract=contract))

    def one(name, request=None, *, lut=0, coalesced=0, mode='execute', phase='identity', pattern='random', seed=42,
            graph=64, repeats=1, round_no=0, order=0, timing=False):
        if not ledger.start(dict(n=n, batch=batch, candidate_id=name, phase=phase, round=round_no), args.timeout + 30):
            raise v2.BudgetStop()
        original = name == 'original'
        args.exe = args.original if original else args.expanded
        args.exe_sha256 = contract['binaries']['original' if original else 'expanded']
        knob_args = ['--use-lut', str(lut), '--coalesced-memory', str(coalesced)] if lut or coalesced else []
        if request is not None:
            invocation = requests.invocation(n, batch, request)
            knob_args, env = invocation['args'], invocation['env']
        else:
            env = {}
        workload = dict(n=n, batch=batch, seed=seed, pattern=pattern, graph_repeats=graph,
                        warmup=5 if timing else 0, warmup_ms=1000 if timing else 0, repeats=repeats,
                        study_mode=mode, knob_args=knob_args)
        row = run_one(args, workload, (name, env), phase, round_no, order)
        emit(row)
        ledger.finish(row['status'])
        if numerical_failure(row):
            emit(dict(record_type='numerical_stop', candidate_id=name, phase=phase, result=row.get('result')))
            raise v2.NumericalFailure('numerical failure requires offline audit; no further device work')
        return row

    def execute_checked(name, request=None, **kw):
        row = one(name, request, **kw)
        lut, coalesced = kw.get('lut', 0), kw.get('coalesced', 0)
        result = validation.check_execute(row, args.gpu, lut=lut, coalesced=coalesced,
                                          original=name == 'original', request=request, timing=kw.get('timing', False))
        return row, result

    baseline_row, baseline = execute_checked('original', phase='baseline')
    baseline_sig = search.source_signature(baseline, original=True)
    graph = 64
    while True:
        _, result = execute_checked('original', phase='calibration', graph=graph)
        if search.source_signature(result, original=True) != baseline_sig:
            raise ValueError('baseline plan drift')
        if min(result['raw_interval_ms_samples']) >= 2:
            break
        if graph >= 2048:
            raise ValueError('unable to calibrate original graph to 2 ms')
        graph *= 2
    emit(dict(record_type='baseline_admitted', n=n, batch=batch, graph_repeats=graph,
              performance_denominator='original_no_knob_default'))

    admitted = set()
    for item in candidates:
        req, cid = item['request'], item['candidate_id']
        lut, coalesced = req['use_lut'], req['coalesced_memory']
        group = (lut, coalesced)
        if group not in admitted:
            original_row, original = execute_checked('original', lut=lut, coalesced=coalesced, phase='matched_identity', graph=graph)
            default_row, default = execute_checked('experimental_default', lut=lut, coalesced=coalesced, phase='matched_identity', graph=graph)
            validation.comparison.compare(default, original)
            identity, reason = v2.representable(default)
            if identity is None:
                emit(dict(record_type='unsupported_identity', knobs=group, reason=reason))
                raise ValueError('same-knob default cannot be represented by this prototype')
            identity.update(use_lut=lut, coalesced_memory=coalesced)
            replay_row, _ = execute_checked('identity', identity, lut=lut, coalesced=coalesced, phase='matched_identity', graph=graph)
            validation.check_matched_identity(original_row, default_row, replay_row, args.gpu, identity)
            emit(dict(record_type='matched_identity_passed', knobs=group, identity=identity))
            admitted.add(group)
        emit(dict(record_type='candidate_start', n=n, batch=batch, candidate_id=cid, request=req))
        row = one(cid, req, lut=lut, coalesced=coalesced, phase='describe', mode='describe', graph=graph)
        reason = scheduler_rejection(row, args.gpu)
        if reason:
            emit(dict(record_type='candidate_disposition', candidate_id=cid, status='scheduler_rejected', reason=reason))
            continue
        described = validation.check_describe(row, args.gpu, req)
        dp = described['precompile_kernels'][0]
        validated_sig = None
        for pattern in PATTERNS:
            for seed in SEEDS:
                row = one(cid, req, lut=lut, coalesced=coalesced, phase='validation', pattern=pattern, seed=seed, graph=graph)
                reason = scheduler_rejection(row, args.gpu)
                if reason:
                    break
                result = validation.check_execute(row, args.gpu, lut=lut, coalesced=coalesced, request=req)
                cp = result['precompile_kernels'][0]
                if any(cp[k] != dp[k] for k in ('canonical_descriptor', 'source_sha256', 'source_bytes', 'compiler_input')):
                    raise ValueError('describe and validation code differ')
                signature = search.source_signature(result, original=False)
                if validated_sig is not None and signature != validated_sig:
                    raise ValueError('validation plan drift')
                validated_sig = signature
            if reason:
                break
        if reason:
            emit(dict(record_type='candidate_disposition', candidate_id=cid, status='scheduler_rejected', reason=reason))
            continue
        ratios = []
        for round_no in range(3):
            results = {}
            for order, role in enumerate(v2.seeded_order(n, batch, cid, round_no)):
                name = 'original' if role == 'original' else cid
                row = one(name, None if role == 'original' else req, lut=0 if role == 'original' else lut,
                          coalesced=0 if role == 'original' else coalesced, phase='timing', graph=graph,
                          repeats=31, round_no=round_no, order=order, timing=True)
                result = validation.check_execute(row, args.gpu, lut=0 if role == 'original' else lut,
                    coalesced=0 if role == 'original' else coalesced, original=role == 'original',
                    request=None if role == 'original' else req, timing=True)
                if result['warmup_actual'] < 5 or result['warmup_wall_ms'] < 1000:
                    raise ValueError('timing warmup below protocol')
                sig = search.source_signature(result, original=role == 'original')
                if sig != (baseline_sig if role == 'original' else validated_sig):
                    raise ValueError('timing plan differs from validated plan')
                results[role] = result
            ratio = statistics.median(results['candidate']['exec_ms_samples']) / statistics.median(results['original']['exec_ms_samples'])
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError('invalid paired ratio')
            ratios.append(ratio)
            emit(dict(record_type='screen_pair', candidate_id=cid, round=round_no, ratio=ratio,
                      denominator='original_no_knob_default'))
        emit(dict(record_type='candidate_disposition', candidate_id=cid, request=req, status='screen_complete',
                  pair_ratios=ratios, median_ratio=statistics.median(ratios)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True); parser.add_argument('--original', required=True, type=Path)
    parser.add_argument('--expanded', required=True, type=Path); parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--generated-provenance', required=True, type=Path); parser.add_argument('--build-attestation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path); parser.add_argument('--budget-ledger', required=True, type=Path)
    parser.add_argument('--log', required=True, type=Path); parser.add_argument('--incidents', required=True, type=Path)
    parser.add_argument('--lock', required=True, type=Path, help='shared lock used by all experiments on this device')
    parser.add_argument('--seconds', required=True, type=float); parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--idle-seconds', type=float, default=30)
    args = parser.parse_args(argv)
    if os.name != 'posix' or not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal') or not args.gpu.startswith('GPU-') or ',' in args.gpu:
        parser.error('Linux pidfd and one GPU UUID required')
    if not math.isfinite(args.seconds) or not math.isfinite(args.idle_seconds) or args.seconds <= args.idle_seconds or args.timeout <= 0 or not 0 <= args.idle_seconds <= 3600:
        parser.error('invalid deadline, timeout or idle time')
    for field in ('original', 'expanded', 'manifest', 'generated_provenance', 'build_attestation'):
        setattr(args, field, getattr(args, field).resolve(strict=True))
    if args.original == args.expanded:
        parser.error('two distinct binaries required')
    paths = [args.output, args.budget_ledger, args.log, args.incidents, args.lock]
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error('evidence and lock paths must differ')
    root = Path(v2.git(HERE, 'rev-parse', '--show-toplevel'))
    source_files = [HERE / x for x in ('runner.py', 'requests.py', 'validation.py', 'prepare_headers.py')]
    contract = v2.provenance(root, source_files)
    if not os.environ.get('EXPERIMENT_CODE_COMMIT') or os.environ['EXPERIMENT_CODE_COMMIT'] != contract['execution_commit']:
        raise ValueError('must run from tools/run_experiment.py with matching execution commit')
    code_commit = verify_declaration(root, contract['execution_commit'], os.environ.get('EXPERIMENT_ID'))
    hashes = {'original': v2.sha(args.original), 'expanded': v2.sha(args.expanded)}
    generated = verify_generated(root, args.generated_provenance, contract['vkfft_commit'])
    attestation = verify_attestation(args.build_attestation, code_commit, contract['vkfft_commit'], generated['sha256'], hashes)
    manifest = load_manifest(args.manifest)
    contract.update(code_commit=code_commit, gpu=args.gpu, binaries=hashes, generated_provenance=generated, build_attestation=attestation,
                    manifest={'path':str(args.manifest), 'sha256':v2.sha(args.manifest)},
                    timeout=args.timeout, protocol='frontier-knobs-execution/v1')
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE.parent / 'expanded_search'))
    import device_lock
    global run_one
    with device_lock.DeviceLock(args.lock):
        if any(p.exists() for p in (args.output, args.budget_ledger, args.log, args.incidents)):
            raise ValueError('existing evidence requires offline audit; automatic resume is disabled')
        ledger = v2.Ledger(args.budget_ledger, contract, args.seconds)
        for path in (args.log, args.incidents):
            path.touch(exist_ok=False)
        guard_args = argparse.Namespace(gpu=args.gpu, log=args.log, incidents=args.incidents, idle_seconds=args.idle_seconds)
        guard = guard_common.Guard(guard_args)
        original_popen, original_run = subprocess.Popen, run_one
        keys = ('CUDA_VISIBLE_DEVICES', 'VKFFT_STUDY_GUARD_ID', 'VKFFT_STUDY_GUARD_UUID', 'VKFFT_STUDY_GUARD_LOG')
        saved = {key: os.environ.get(key) for key in keys}
        try:
            os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, VKFFT_STUDY_GUARD_ID=guard.guard_id,
                              VKFFT_STUDY_GUARD_UUID=args.gpu, VKFFT_STUDY_GUARD_LOG=str(args.log.resolve()))
            guard.emit('guard_start', script=str(HERE / 'runner.py'), idle_seconds=args.idle_seconds,
                       memory_limit_mib=guard_common.MEMORY_LIMIT_MIB, monitor_interval_seconds=1,
                       parent_pid=os.getpid(), own_signal_method='pidfd_with_rechecked_PPID_exe_starttime')
            guard.idle_check(); subprocess.Popen = guard.popen(original_popen); run_one = guard.wrapped(original_run); guard.start()
            try:
                screen(args, ledger, contract, manifest)
            except v2.BudgetStop:
                v2.append(args.output, dict(record_type='budget_exhausted', contract=contract, guard_id=guard.guard_id))
            guard.sample('final'); guard.check(); guard.close(); guard.check()
            if guard.thread is not None and guard.thread.is_alive():
                raise guard_common.GuardAbort('monitor failed to stop')
            guard.emit('guard_complete', success=True, clean=True, script_exit_code=0, sample_count=guard.sample_count)
        except BaseException as error:
            if not guard.abort.is_set():
                guard.trip([f'guard_or_target_exception:{type(error).__name__}:{error}'])
            guard.close(); guard.emit('guard_abort', incident=True, success=False, clean=False,
                                      error=f'{type(error).__name__}: {error}', sample_count=guard.sample_count)
            raise
        finally:
            guard.close(); subprocess.Popen, run_one = original_popen, original_run
            for key, value in saved.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value


if __name__ == '__main__':
    main()
