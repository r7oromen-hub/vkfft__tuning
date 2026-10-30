#!/usr/bin/env python3
"""Guarded, resumable screening of committed frontier experiments on one GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
EXPANDED = HERE.parent / 'expanded_search'
sys.path.insert(0, str(EXPANDED))
import run_suite as suite
import validation
import search_controller
sys.path.insert(1, str(HERE.parent / 'original_compare'))
import original_compare_controller as comparison
import original_compare_guard as guard_common

PATTERNS = ('zero', 'impulse', 'constant', 'quarter-tone', 'random')
SEEDS = (42, 53, 71)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def provenance(root, sources, artifacts=()):
    root = Path(root).resolve(strict=True)
    for repo in (root, root / 'VkFFT' / 'vkFFT'):
        subprocess.run(['git', '-C', str(repo), 'diff', '--quiet'], check=True)
        subprocess.run(['git', '-C', str(repo), 'diff', '--cached', '--quiet'], check=True)
    # Running uncommitted controller code would defeat execution-commit tracking.
    required = [HERE / 'runner.py', EXPANDED / 'run_suite.py', EXPANDED / 'validation.py',
                EXPANDED / 'search_controller.py', EXPANDED / 'device_lock.py',
                HERE.parent / 'original_compare' / 'original_compare_guard.py',
                HERE.parent / 'original_compare' / 'original_compare_controller.py']
    required += [HERE.parent / 'frontier_search' / name for name in
                 ('prepare_headers.py', 'candidate.hpp', 'rejection_instrumentation.py', 'rejection.hpp')]
    for path in required: git(root, 'ls-files', '--error-unmatch', str(path))
    paths = [Path(p).resolve(strict=True) for p in sources]
    if not paths:
        raise ValueError('at least one build/source provenance file is required')
    vkfft = root / 'VkFFT' / 'vkFFT'
    for path in paths:
        repo = vkfft if path.is_relative_to(vkfft) else root
        if not path.is_relative_to(repo): raise ValueError('source must be inside a committed repository')
        git(repo, 'ls-files', '--error-unmatch', str(path))
    return {'execution_commit': git(root, 'rev-parse', 'HEAD'),
            'vkfft_commit': git(root / 'VkFFT' / 'vkFFT', 'rev-parse', 'HEAD'),
            'source_files': {str(p): sha(p) for p in paths},
            'build_artifacts': {str(Path(p).resolve(strict=True)): sha(p) for p in artifacts}}


def verify_generated(root, path, vkfft_commit):
    path = Path(path).resolve(strict=True); doc = json.loads(path.read_text(encoding='utf8'))
    if doc.get('upstream_commit') != vkfft_commit or not doc.get('expanded') or not doc.get('frontier_v2'):
        raise ValueError('generated provenance is missing or targets a different VkFFT commit')
    for rel, expected in doc['expanded'].items():
        target = (path.parent / rel).resolve(strict=True)
        if not target.is_relative_to(path.parent) or sha(target) != expected:
            raise ValueError('generated source hash mismatch: ' + rel)
    front = doc['frontier_v2']; source_dir = root / 'test' / 'plan_selection_study' / 'frontier_search'
    for field, filename in (('generator_sha256','prepare_headers.py'), ('candidate_sha256','candidate.hpp'),
                            ('rejection_instrumentation_sha256','rejection_instrumentation.py'),
                            ('rejection_header_sha256','rejection.hpp')):
        if sha(source_dir / filename) != front.get(field): raise ValueError('generator source hash mismatch: ' + filename)
    if sha(path.parent / 'include' / 'rejection.hpp') != front.get('generated_rejection_header_sha256'):
        raise ValueError('generated rejection header mismatch')
    return {'path':str(path), 'sha256':sha(path), 'generated_files':doc['expanded']}


def load_manifest(path):
    with Path(path).open(encoding='utf8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or rows[0].get('schema') != 'frontier-manifest/v1':
        raise ValueError('invalid manifest header')
    w = rows[0]['workload']; n, batch = w['n'], w['batch']
    seen, selected = set(), []
    for row in rows[1:]:
        req = row['request']; cid = hashlib.sha256(json.dumps(req, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if row.get('candidate_id') != cid or cid in seen:
            raise ValueError('invalid or duplicate manifest candidate id')
        seen.add(cid)
        if row.get('selection') == 'selected':
            if row.get('status') != 'pending_scheduler_and_device_validation':
                raise ValueError('selected candidate is not statically eligible')
            selected.append(row)
    selected.sort(key=lambda row: row['selection_index'])
    if [r['selection_index'] for r in selected] != list(range(len(selected))):
        raise ValueError('selection indices must be contiguous from zero')
    return n, batch, selected


def read_log(path):
    if not path.exists(): return []
    with path.open(encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


def append(path, row):
    with path.open('a', encoding='utf8') as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


class Ledger:
    """One wall-clock deadline across resumes; fail closed on an unresolved item."""
    def __init__(self, path, contract, seconds, *, clock=time.monotonic, boot=None):
        self.path, self.clock = Path(path), clock
        self.boot = boot or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        rows = read_log(self.path)
        if rows:
            self.state = rows[-1]['state']
            if (self.state['contract'] != contract or self.state['seconds'] != seconds or
                self.state['boot'] != self.boot or self.clock() < self.state['last'] or self.state['inflight']):
                raise ValueError('budget contract, boot, monotonic clock, or in-flight item requires audit')
        else:
            now = self.clock()
            self.state = dict(contract=contract, seconds=seconds, boot=self.boot, start=now, last=now, inflight=None)
            self.event('start')

    def event(self, event, **details):
        self.state['last'] = self.clock()
        append(self.path, dict(event=event, state=self.state.copy(), **details))

    def start(self, identity, expected):
        if self.state['inflight']: raise ValueError('unresolved in-flight item')
        if self.clock() + expected >= self.state['start'] + self.state['seconds']:
            self.event('denied', identity=identity); return False
        self.state['inflight'] = identity; self.event('work_start'); return True

    def finish(self, status):
        if not self.state['inflight']: raise ValueError('no in-flight item')
        item = self.state['inflight']; self.state['inflight'] = None
        self.event('work_finish', identity=item, status=status)


class BudgetStop(Exception): pass


def seeded_order(n, batch, cid, round_no):
    key = json.dumps([20261031, n, batch, cid], separators=(',', ':')).encode()
    first = random.Random(int.from_bytes(hashlib.sha256(key).digest(), 'big')).randrange(2)
    return ('original', 'candidate') if (first + round_no) % 2 == 0 else ('candidate', 'original')


def representable(default):
    f = default['structural_features']
    if f['numAxisUploads'] != 1 or len(f['kernels']) != 1: return None, 'requires_one_upload_and_kernel'
    k = f['kernels'][0]; d = default['precompile_kernels'][0]['raw_descriptor']
    radix, r, group = k['stageRadix'], d['logical_registers_per_thread'], k['groupedBatch']
    if d['registerBoost'] != 1: return None, 'register_boost_not_one'
    if r not in (2, 4, 8, 16, 32, 64, 128, 256) or d['min_logical_registers_per_thread'] != r:
        return None, 'nonuniform_or_unsupported_register_envelope'
    expected = [r if i in (2, 4, 8, 16, 32) and r % i == 0 else 0 for i in range(33)]
    if d['registers_per_radix'] != expected: return None, 'nonuniform_register_table'
    if (math.prod(radix) != default['n'] or group < 1 or group & (group-1) or default['batch'] % group or
        default['n'] % r or any(x not in (2, 4, 8, 16, 32) or x > r or r % x for x in radix)):
        return None, 'unsupported_radix_register_or_batch'
    swap, local = d['axisSwapped'], k['localSize']
    if (swap not in (0, 1) or len(local) != 3 or local[2] != 1 or
        local[1 if swap else 0] != default['n']//r or local[0 if swap else 1] != group):
        return None, 'unsupported_local_layout'
    return {'radix':radix, 'logical_registers':r, 'grouped_batch':group}, None


def completed_candidates(rows, guard_rows):
    clean = {r['guard_id'] for r in guard_rows if r.get('event') == 'guard_complete' and
             r.get('success') is True and r.get('clean') is True}
    return {(r['n'], r['batch'], r['candidate_id']) for r in rows
            if r.get('guard_id') in clean and r.get('record_type') == 'candidate_disposition' and
            r.get('status') in ('screen_complete', 'scheduler_rejected', 'resource_rejected', 'numerically_invalid')}


def screen(args, ledger, contract, rows):
    if os.environ.get('VKFFT_STUDY_GUARD_UUID') != args.gpu or not os.environ.get('VKFFT_STUDY_GUARD_ID'):
        raise ValueError('GPU guard required')
    session = uuid.uuid4().hex
    def emit(row):
        out = dict(row); out.update(session_id=session, guard_id=os.environ['VKFFT_STUDY_GUARD_ID'], contract=contract)
        append(args.output, out)
    emit(dict(record_type='session_start', gpu=args.gpu))
    complete = completed_candidates(rows, read_log(args.log))

    def one(n, batch, name, req=None, *, mode='execute', phase='validation', pattern='random', seed=42, repeats=1, graph=64, round_no=0, order=0, timing=False):
        if not ledger.start(dict(n=n, batch=batch, candidate_id=name, phase=phase, round=round_no), args.timeout + 30):
            raise BudgetStop()
        args.exe = str(args.original if name == 'original' else args.expanded)
        args.exe_sha256 = contract['binaries']['original' if name == 'original' else 'expanded']
        args.session_id = session
        w = dict(n=n, batch=batch, seed=seed, config={}, pattern=pattern, graph_repeats=graph,
                 warmup=5 if timing else 0, warmup_ms=1000 if timing else 0, repeats=repeats, study_mode=mode)
        try:
            row = suite.run_one(args, w, (name, validation.request_env(req) if req else {}), phase, round_no, order)
            emit(row)
            ledger.finish(row['status'])
            return row
        except BaseException:
            # A crash between execution and durable evidence requires manual audit.
            raise

    def checked(*a, original=False, request=None, **kw):
        row = one(*a, **kw)
        return validation.check(row, args.gpu, original=original, request=request)

    for manifest_path in args.manifest:
        n, batch, selected = load_manifest(manifest_path)
        original = checked(n, batch, 'original', original=True, phase='identity')
        default = checked(n, batch, 'expanded_default', phase='identity')
        comparison.compare(default, original)
        req, reason = representable(default)
        if req is None:
            emit(dict(record_type='workload_status', n=n, batch=batch, status='unsupported_identity', reason=reason)); continue
        replay = checked(n, batch, 'identity', req, request=req, phase='identity')
        if validation.signature(default) != validation.signature(replay): raise ValueError('default identity replay mismatch')
        original_sig = search_controller.source_signature(original, original=True)
        emit(dict(record_type='workload_status', n=n, batch=batch, status='identity_passed', request=req))
        graph = 64
        while True:
            result = checked(n, batch, 'original', original=True, phase='calibration', graph=graph)
            if search_controller.source_signature(result, original=True) != original_sig: raise ValueError('original plan drift')
            if min(result['raw_interval_ms_samples']) >= 2: break
            if graph >= 2048: raise ValueError('calibration failed')
            graph *= 2
        for item in selected:
            cid, request = item['candidate_id'], item['request']
            if (n, batch, cid) in complete: continue
            emit(dict(record_type='candidate_start', n=n, batch=batch, candidate_id=cid, request=request))
            row = one(n, batch, cid, request, phase='describe', mode='describe', graph=graph)
            reason = None
            if validation.known_rejection(row, args.gpu):
                reason = 'scheduler_rejected' if row['result']['error'].endswith('3002') else 'resource_rejected'
            else:
                described = validation.check_describe(row, args.gpu, request)
                dp = described['precompile_kernels'][0]
                validated_sig = None
                for pattern in PATTERNS:
                    for seed in SEEDS:
                        row = one(n, batch, cid, request, phase='validation', pattern=pattern, seed=seed, graph=graph)
                        if validation.known_rejection(row, args.gpu):
                            reason = 'scheduler_rejected' if row['result']['error'].endswith('3002') else 'resource_rejected'
                        elif search_controller.numerical_failure(row, args.gpu, request): reason = 'numerically_invalid'
                        else:
                            result = validation.check(row, args.gpu, request=request)
                            cp = result['precompile_kernels'][0]
                            if any(cp[k] != dp[k] for k in ('canonical_descriptor', 'source_sha256', 'source_bytes', 'compiler_input')):
                                raise ValueError('describe and validation code differ')
                            sig = search_controller.source_signature(result, original=False)
                            if validated_sig is not None and validated_sig != sig: raise ValueError('validation plan drift')
                            validated_sig = sig
                        if reason: break
                    if reason: break
            if reason:
                result = checked(n, batch, 'original', original=True, phase='post_rejection', graph=graph)
                if search_controller.source_signature(result, original=True) != original_sig: raise ValueError('post-rejection control drift')
                emit(dict(record_type='candidate_disposition', n=n, batch=batch, candidate_id=cid, request=request, status=reason,
                          diagnostic=row.get('result', {}).get('expanded_rejection_reason', 'unattributed'))); continue
            ratios = []
            candidate_sig = None
            for round_no in range(3):
                results = {}
                for ix, name in enumerate(seeded_order(n, batch, cid, round_no)):
                    variant = 'original' if name == 'original' else cid
                    row = one(n, batch, variant, None if name == 'original' else request, phase='timing', graph=graph,
                              repeats=31, round_no=round_no, order=ix, timing=True)
                    results[name] = validation.check(row, args.gpu, original=name == 'original',
                                                     request=request if name == 'candidate' else None, timing=True)
                    if results[name]['warmup_actual'] < 5 or results[name]['warmup_wall_ms'] < 1000: raise ValueError('short warmup')
                if search_controller.source_signature(results['original'], original=True) != original_sig: raise ValueError('timing original drift')
                sig = search_controller.source_signature(results['candidate'], original=False)
                if sig != validated_sig: raise ValueError('validation and timing plan differ')
                if candidate_sig is not None and candidate_sig != sig: raise ValueError('candidate plan drift')
                candidate_sig = sig
                cp = results['candidate']['precompile_kernels'][0]
                if any(cp[k] != dp[k] for k in ('canonical_descriptor', 'source_sha256', 'source_bytes', 'compiler_input')): raise ValueError('timing code drift')
                ratio = statistics.median(results['candidate']['exec_ms_samples']) / statistics.median(results['original']['exec_ms_samples'])
                if not math.isfinite(ratio) or ratio <= 0: raise ValueError('invalid timing ratio')
                ratios.append(ratio)
                emit(dict(record_type='screen_pair', n=n, batch=batch, candidate_id=cid, round=round_no, ratio=ratio))
            emit(dict(record_type='candidate_disposition', n=n, batch=batch, candidate_id=cid, request=request,
                      status='screen_complete', pair_ratios=ratios, median_ratio=statistics.median(ratios)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', required=True); p.add_argument('--original', type=Path, required=True)
    p.add_argument('--expanded', type=Path, required=True); p.add_argument('--manifest', type=Path, action='append', required=True)
    p.add_argument('--source-file', type=Path, action='append', required=True)
    p.add_argument('--artifact-file', type=Path, action='append', default=[])
    p.add_argument('--generated-provenance', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--budget-ledger', type=Path, required=True)
    p.add_argument('--log', type=Path, required=True); p.add_argument('--incidents', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True); p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--timeout', type=int, default=300); p.add_argument('--idle-seconds', type=float, default=30)
    args = p.parse_args(argv)
    if os.name != 'posix' or not hasattr(os, 'pidfd_open') or not args.gpu.startswith('GPU-') or ',' in args.gpu:
        p.error('Linux pidfd and a single GPU UUID required')
    if not math.isfinite(args.seconds) or args.seconds <= args.idle_seconds or args.timeout <= 0 or not 0 <= args.idle_seconds <= 3600:
        p.error('invalid budget, timeout, or idle duration')
    args.original = args.original.resolve(strict=True); args.expanded = args.expanded.resolve(strict=True)
    if args.original == args.expanded: p.error('distinct binaries required')
    paths = [args.output, args.budget_ledger, args.log, args.incidents, args.lock]
    if len({x.resolve() for x in paths}) != len(paths): p.error('output paths must differ')
    for x in paths: x.parent.mkdir(parents=True, exist_ok=True)
    root = Path(git(HERE, 'rev-parse', '--show-toplevel'))
    contract = dict(provenance(root, args.source_file, args.artifact_file), gpu=args.gpu,
                    binaries={'original':sha(args.original), 'expanded':sha(args.expanded)},
                    manifests={str(x.resolve(strict=True)):sha(x) for x in args.manifest},
                    timeout=args.timeout, protocol='frontier-execution/v1')
    contract['generated_provenance'] = verify_generated(root, args.generated_provenance, contract['vkfft_commit'])
    for path in args.manifest: load_manifest(path)
    import device_lock
    with device_lock.DeviceLock(args.lock):
        if args.budget_ledger.exists() and (not args.output.exists() or not args.log.exists() or not args.incidents.exists()):
            raise ValueError('resume evidence missing; manual audit required')
        old_rows = read_log(args.output)
        if old_rows and any(row.get('contract') != contract for row in old_rows): raise ValueError('output contract mismatch')
        ledger = Ledger(args.budget_ledger, contract, args.seconds)
        guard_args = argparse.Namespace(gpu=args.gpu, log=args.log, incidents=args.incidents, idle_seconds=args.idle_seconds)
        for path in (args.log, args.incidents): path.touch(exist_ok=True)
        guard = guard_common.Guard(guard_args)
        old_popen, old_run = subprocess.Popen, suite.run_one
        keys = ('CUDA_VISIBLE_DEVICES', 'VKFFT_STUDY_GUARD_ID', 'VKFFT_STUDY_GUARD_UUID', 'VKFFT_STUDY_GUARD_LOG')
        saved = {k:os.environ.get(k) for k in keys}
        try:
            os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, VKFFT_STUDY_GUARD_ID=guard.guard_id,
                              VKFFT_STUDY_GUARD_UUID=args.gpu, VKFFT_STUDY_GUARD_LOG=str(args.log.resolve()))
            guard.emit('guard_start', script=str(HERE / 'runner.py'), idle_seconds=args.idle_seconds,
                       memory_limit_mib=guard_common.MEMORY_LIMIT_MIB, monitor_interval_seconds=1,
                       parent_pid=os.getpid(), own_signal_method='pidfd_with_rechecked_PPID_exe_starttime')
            guard.idle_check(); subprocess.Popen = guard.popen(old_popen); suite.run_one = guard.wrapped(old_run); guard.start()
            try: screen(args, ledger, contract, old_rows)
            except BudgetStop: append(args.output, dict(record_type='budget_exhausted', contract=contract))
            guard.sample('final'); guard.check(); guard.close(); guard.check()
            if guard.thread is not None and guard.thread.is_alive(): raise guard_common.GuardAbort('monitor failed to stop')
            guard.emit('guard_complete', success=True, clean=True, script_exit_code=0, sample_count=guard.sample_count)
        except BaseException as error:
            if not guard.abort.is_set(): guard.trip([f'guard_or_target_exception:{type(error).__name__}:{error}'])
            guard.close(); guard.emit('guard_abort', incident=True, success=False, clean=False,
                                      error=f'{type(error).__name__}: {error}', sample_count=guard.sample_count)
            raise
        finally:
            guard.close(); subprocess.Popen, suite.run_one = old_popen, old_run
            for k, value in saved.items():
                if value is None: os.environ.pop(k, None)
                else: os.environ[k] = value


if __name__ == '__main__': main()
