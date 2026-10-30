#!/usr/bin/env python3
"""Guarded, globally budgeted multi-address measurement of confirmed winners."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
EXPANDED = STUDY / 'expanded_search'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eligibility = load('frontier_rotation_eligibility_runner', HERE / 'eligibility.py')
prepare = load('frontier_rotation_prepare_runner', HERE / 'prepare.py')
frontier = eligibility.frontier
sys.path.insert(0, str(EXPANDED))
try:
    rotation = load('frontier_rotation_legacy_controller', EXPANDED / 'rotation_controller.py')
finally:
    sys.path.pop(0)
suite = frontier.suite
search = eligibility.search
guard_common = frontier.guard_common


def need(value, message):
    if not value:
        raise ValueError(message)


def campaign(root, record_path, campaign_start, admission_budget, gpu, *, now=None, boot=None):
    """Bind the original campaign and budget to its committed admission record."""
    root = Path(root).resolve(strict=True)
    record_path = Path(record_path).resolve(strict=True)
    need(record_path.is_relative_to(root), 'campaign record is outside the committed checkout')
    frontier.git(root, 'ls-files', '--error-unmatch', str(record_path))
    record = json.loads(record_path.read_text(encoding='utf8'))
    need(record.get('schema') == 'git-experiment/v1' and record.get('status') == 'completed', 'admission record is not completed')
    listed = record.get('artifacts')
    need(isinstance(listed, list), 'admission artifact list missing')
    for input_path in (campaign_start, admission_budget):
        path = Path(input_path).resolve(strict=True)
        need(path.is_relative_to(root), 'campaign artifact is outside checkout')
        rel = path.relative_to(root).as_posix()
        matches = [item for item in listed if item.get('path') == rel]
        need(len(matches) == 1 and matches[0].get('sha256') == frontier.sha(path) and
             matches[0].get('bytes') == path.stat().st_size, 'campaign artifact hash mismatch: ' + rel)
    start = json.loads(Path(campaign_start).read_text(encoding='utf8'))
    entries = frontier.read_log(admission_budget)
    need(entries and entries[0].get('event') == 'start', 'admission budget has no start')
    first = entries[0]['state']
    need(entries[-1]['state'].get('inflight') is None, 'admission budget has unresolved work')
    need(start.get('gpu') == gpu and first['contract'].get('gpu') == gpu, 'campaign GPU mismatch')
    need(start.get('execution_commit') == record.get('execution_commit') == first['contract'].get('execution_commit'),
         'campaign admission commit mismatch')
    protocol = record.get('protocol') or {}
    protocol_path = (root / protocol.get('path', '')).resolve(strict=True)
    need(protocol_path.is_relative_to(root) and frontier.sha(protocol_path) == protocol.get('sha256'),
         'campaign protocol hash mismatch')
    cfg = json.loads(protocol_path.read_text(encoding='utf8'))
    need(cfg.get('schema') == 'frontier-campaign/v1' and cfg.get('gpu_uuid') == gpu and
         cfg.get('total_reserved_wall_seconds') == start.get('total_reserved_wall_seconds'), 'campaign protocol mismatch')
    begun, duration = start.get('start_monotonic'), start.get('total_reserved_wall_seconds')
    need(type(begun) in (int, float) and math.isfinite(begun) and type(duration) in (int, float) and
         math.isfinite(duration) and duration > 0, 'invalid campaign time')
    now = time.monotonic() if now is None else now
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip() if boot is None else boot
    need(first['boot'] == boot and all(row['state'].get('boot') == boot for row in entries), 'campaign boot changed')
    need(now >= begun and first['start'] >= begun and first['last'] >= first['start'], 'campaign monotonic clock mismatch')
    deadline = begun + duration
    need(now < deadline, 'global campaign deadline exhausted')
    return {'start': begun, 'deadline': deadline, 'boot': boot,
            'campaign_record_sha256': frontier.sha(record_path), 'campaign_start_sha256': frontier.sha(campaign_start),
            'admission_budget_sha256': frontier.sha(admission_budget), 'protocol_sha256': frontier.sha(protocol_path)}


def verify_declaration(root, execution_commit, experiment_id):
    need(bool(experiment_id) and all(c.isalnum() or c in '_.-' for c in experiment_id), 'invalid experiment ID')
    parent = frontier.git(root, 'rev-parse', execution_commit + '^')
    record = f'experiments/records/{experiment_id}.json'
    need(frontier.git(root, 'diff', '--name-only', parent, execution_commit).splitlines() == [record],
         'execution commit must contain only its run declaration')
    doc = json.loads(frontier.git(root, 'show', execution_commit + ':' + record))
    need(doc.get('schema') == 'git-experiment/v1' and doc.get('status') == 'declared' and
         doc.get('id') == experiment_id and doc.get('parent_code_commit') == parent, 'invalid run declaration')
    return parent


def verify_rotation_build(root, rotation_root, generated, original, instrumentation, hot, binaries,
                          attestation_path, compile_commands, original_dependencies, code_commit):
    """Check recorded input/output trees and declared binary-to-build evidence."""
    root = Path(root).resolve(strict=True)
    need(Path(original).resolve(strict=True) == (root / 'VkFFT/vkFFT').resolve(strict=True),
         'original VkFFT must be the committed submodule include tree')
    need(Path(instrumentation).resolve(strict=True) == (root / 'test/plan_selection_study/instrumentation').resolve(strict=True),
         'instrumentation must be from the committed checkout')
    rotation_root = Path(rotation_root).resolve(strict=True)
    path = rotation_root / 'rotation_provenance.json'
    doc = json.loads(path.read_text(encoding='utf8'))
    front = doc.get('frontier_rotation', {})
    need(front.get('schema') == 'frontier-rotation-generated/v1' and front.get('device_validation') == 'not performed' and
         front.get('generator_sha256') == frontier.sha(HERE / 'prepare.py'), 'rotation generator provenance mismatch')
    for name in ('prepare_rotation.py', 'prepare_rotation_v2.py', 'rotation_pool.hpp', 'rotation_cuda_compat.hpp', 'CMakeLists_rotation.txt'):
        need(front['legacy_sources'].get(name) == frontier.sha(EXPANDED / name), 'rotation dependency hash mismatch: ' + name)
    need(front.get('frontier_upstream_commit') == hot['vkfft_commit'], 'rotation VkFFT commit differs from hot evidence')
    need(frontier.sha(Path(generated) / 'expanded_provenance.json') == hot['hot_generated_provenance_sha256'],
         'rotation frontier tree differs from confirmed hot generated provenance')
    prepare.verify_frontier(generated)
    for name, tree in (('frontier_generated', generated), ('original_vkfft', original), ('instrumentation', instrumentation)):
        need(front['inputs'].get(name) == prepare.tree_inventory(tree), 'rotation input tree differs: ' + name)
    for rel, digest in front['output_files'].items():
        target = (rotation_root / rel).resolve(strict=True)
        need(target.is_relative_to(rotation_root) and target.is_file() and frontier.sha(target) == digest,
             'rotation generated output mismatch: ' + rel)
    need(set(front['output_files']) == {'rotating_benchmark.cpp', 'rotation_pool.hpp', 'rotation_cuda_compat.hpp', 'CMakeLists.txt'},
         'rotation generated output inventory incomplete')
    hashes = {name: frontier.sha(path) for name, path in binaries.items()}
    attest = json.loads(Path(attestation_path).read_text(encoding='utf8'))
    need(attest.get('schema') == 'frontier-rotation-build/v1' and attest.get('code_commit') == code_commit and
         attest.get('vkfft_commit') == hot['vkfft_commit'] and
         attest.get('rotation_provenance_sha256') == frontier.sha(path) and attest.get('binaries') == hashes and
         isinstance(attest.get('build_commands'), list) and attest['build_commands'] and
         all(isinstance(x, str) and x.strip() for x in attest['build_commands']), 'rotation build attestation mismatch')
    commands = json.loads(Path(compile_commands).read_text(encoding='utf8'))
    orig = [item['command'] for item in commands if 'vkfft_rotation_original.dir' in item.get('command', '')]
    gen_include, orig_include = str(Path(generated).resolve() / 'include'), str(Path(original).resolve())
    need(len(orig) == 1 and orig_include in orig[0] and gen_include in orig[0] and
         orig[0].index(orig_include) < orig[0].index(gen_include), 'original compile include precedence not proven')
    dep = Path(original_dependencies).read_text(encoding='utf8')
    need(str(Path(original).resolve() / 'vkFFT.h') in dep and str(Path(gen_include) / 'vkFFT.h') not in dep and
         str(Path(gen_include) / 'vkFFT') not in dep, 'original compiler dependencies are not isolated')
    return {'rotation_provenance_sha256': frontier.sha(path), 'rotation_build_attestation_sha256': frontier.sha(attestation_path),
            'compile_commands_sha256': frontier.sha(compile_commands), 'original_dependencies_sha256': frontier.sha(original_dependencies),
            'binaries': hashes, 'limitation': 'attested build association and dependency evidence, not independent binary reproduction'}


class DeadlineLedger:
    """Check the absolute stage/campaign deadline before admitting every child."""
    def __init__(self, ledger, deadline, clock=time.monotonic):
        self.ledger, self.deadline, self.clock = ledger, deadline, clock

    def start(self, identity, expected):
        if self.clock() + expected >= self.deadline:
            self.ledger.event('denied', identity=identity, reason='absolute_campaign_or_stage_deadline')
            return False
        return self.ledger.start(identity, expected)

    def finish(self, status):
        return self.ledger.finish(status)


def lock_inheriting_popen(guarded_popen, lock_fd, binaries):
    """Keep the campaign flock until an already-started GPU child exits."""
    executables = {str(Path(path).resolve(strict=True)) for path in binaries}
    def launch(command, *args, **kwargs):
        if isinstance(command, (tuple, list)) and command and str(Path(command[0]).resolve()) in executables:
            kwargs['pass_fds'] = tuple(sorted(set(kwargs.get('pass_fds', ())) | {lock_fd}))
        return guarded_popen(command, *args, **kwargs)
    return launch


def numerical_failure(row):
    result = row.get('result') or {}
    return (result.get('status') == 'correctness_failed' or
            any((result.get(k) or {}).get('passed') is False for k in ('rotation_initial_validation', 'rotation_final_validation')))


def uncovered(row, gpu, case, plan):
    """Recognize only a named limitation bound to the intended plan and device."""
    try:
        result, work = row['result'], row['workload']
        phases = {'rotation_unsupported_pool_or_graph_size': 'initialize',
                  'rotation_unsupported_device_memory': 'initialize',
                  'rotation graph kernel argument inspection unavailable': 'graph_build'}
        reason = result.get('error')
        original = row['candidate_id'] == 'original'
        expected_id = 'original' if original else case['candidate_id']
        expected_env = {} if original else rotation.validation.request_env(case['request'])
        expected_signature = case['hot_original_signature' if original else 'hot_candidate_signature']
        if (reason not in phases or result.get('failure_phase') != phases[reason] or
            row.get('status') != 'failed' or row.get('returncode') != 1 or
            row.get('guard_clean') is not True or row.get('guard_gpu_uuid') != gpu or
            row['candidate_id'] != expected_id or row.get('direct_environment') != expected_env or
            result.get('status') != 'error' or result['gpu']['uuid'] != gpu or
            result['gpu']['cuda_visible_devices'] != gpu or result.get('header_instrumentation') is not (not original) or
            result.get('requested_config') != {'makeForwardPlanOnly': 1} or
            (work['n'], work['batch'], work['seed'], work['pattern'], work['graph_repeats']) !=
            (case['n'], case['batch'], 42, 'random', plan['graph_repeats']) or
            (result['n'], result['batch'], result['seed'], result['pattern'], result['graph_repeats']) !=
            (case['n'], case['batch'], 42, 'random', plan['graph_repeats']) or
            search.source_signature(result, original=original) != expected_signature):
            return None
        return reason
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def run_session(args, ledger, contract, proof):
    need(os.environ.get('VKFFT_STUDY_GUARD_UUID') == args.gpu and os.environ.get('VKFFT_STUDY_GUARD_ID'), 'live guard required')
    session = uuid.uuid4().hex
    def emit(row):
        frontier.append(args.output, dict(row, session_id=session, guard_id=os.environ['VKFFT_STUDY_GUARD_ID'], contract=contract))
    emit(dict(record_type='rotation_session_start', gpu=args.gpu,
              scope='separate multi-address measurement; not cold-cache, miss-rate, or end-to-end evidence'))

    def one(case, plan, role, phase, round_no=0, order=0, timing=False):
        cid = 'original' if role == 'original' else case['candidate_id']
        work = dict(n=case['n'], batch=case['batch'], candidate_id=cid, phase=phase, round=round_no)
        if not ledger.start(work, args.timeout + 30):
            raise frontier.BudgetStop()
        args.exe = str(args.rotation_original if role == 'original' else args.rotation_expanded)
        args.exe_sha256 = contract['rotation_build']['binaries']['original' if role == 'original' else 'expanded']
        args.session_id = session
        w = dict(n=case['n'], batch=case['batch'], seed=42, config={}, pattern='random',
                 graph_repeats=plan['graph_repeats'], warmup=5, warmup_ms=1000,
                 repeats=31 if timing else 1, study_mode='execute')
        env = {} if role == 'original' else rotation.validation.request_env(case['request'])
        row = suite.run_one(args, w, (cid, env), phase, round_no, order)
        emit(row)
        ledger.finish(row['status'])
        if numerical_failure(row):
            emit(dict(record_type='rotation_numerical_stop', n=case['n'], batch=case['batch'], candidate_id=case['candidate_id'], phase=phase))
            raise frontier.NumericalFailure('rotation numeric failure; no further device work')
        return row

    measured_count, uncovered_count = 0, 0
    for case in proof['cases']:
        key = {'n': case['n'], 'batch': case['batch'], 'candidate_id': case['candidate_id'], 'request': case['request']}
        emit(dict(record_type='rotation_case_start', **key, eligibility_status=case['status']))
        plan = case['rotation']
        if plan is None:
            emit(dict(record_type='rotation_disposition', **key, status='uncovered_pool_or_graph_limit'))
            uncovered_count += 1
            continue
        plan = dict(plan)
        legacy_case = dict(key, original_signature=case['hot_original_signature'], candidate_signature=case['hot_candidate_signature'])
        while True:
            row = one(case, plan, 'original', 'rotation_calibration')
            reason = uncovered(row, args.gpu, case, plan)
            if reason:
                emit(dict(record_type='rotation_disposition', **key, status='uncovered', reason=reason))
                break
            result = rotation.check(row, args.gpu, legacy_case, plan, original=True, timing=False)
            if min(result['raw_interval_ms_samples']) >= 2:
                emit(dict(record_type='rotation_calibration_complete', **key, rotation=plan,
                          guard_run_id=row['guard_run_id'], raw_interval_ms=min(result['raw_interval_ms_samples'])))
                break
            if plan['graph_repeats'] >= 2048:
                emit(dict(record_type='rotation_disposition', **key, status='uncovered_calibration_interval'))
                break
            plan['graph_repeats'] *= 2
        else:
            raise AssertionError('unreachable')
        if reason or min(result['raw_interval_ms_samples']) < 2:
            uncovered_count += 1
            continue
        ratios = []
        for round_no in range(10):
            ordered = search.seeded_order(20261031, proof['plan_sha256'], case['n'], case['batch'], case['candidate_id'], 'rotation', round_no)
            pair = {}
            for index, role in enumerate(ordered):
                row = one(case, plan, role, 'rotation_timing', round_no, index, True)
                reason = uncovered(row, args.gpu, case, plan)
                if reason:
                    emit(dict(record_type='rotation_disposition', **key, status='uncovered', reason=reason, round=round_no))
                    break
                result = rotation.check(row, args.gpu, legacy_case, plan, original=role == 'original', timing=True)
                pair[role] = (statistics.median(result['exec_ms_samples']), row['guard_run_id'])
            if reason:
                break
            ratio = pair['candidate'][0] / pair['original'][0]
            need(math.isfinite(ratio) and ratio > 0, 'invalid rotation timing ratio')
            ratios.append(ratio)
            emit(dict(record_type='rotation_pair', **key, round=round_no, order=ordered, rotation=plan,
                      original_guard_run_id=pair['original'][1], candidate_guard_run_id=pair['candidate'][1],
                      original_median_ms=pair['original'][0], candidate_median_ms=pair['candidate'][0], ratio=ratio))
        if not reason:
            emit(dict(record_type='rotation_disposition', **key, status='complete', rotation=plan,
                      paired_ratios=ratios, median_ratio=statistics.median(ratios)))
            measured_count += 1
        else:
            uncovered_count += 1
    status = ('no_eligible_winner' if not proof['cases'] else
              'complete_with_uncovered' if uncovered_count else 'complete')
    emit(dict(record_type='rotation_session_complete', status=status, case_count=len(proof['cases']),
              measured_count=measured_count, uncovered_count=uncovered_count))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'hot-original', 'hot-expanded', 'rotation-original', 'rotation-expanded',
                 'rotation-root', 'generated-root', 'original-include-root', 'instrumentation-root',
                 'build-attestation', 'compile-commands', 'original-dependencies', 'campaign-record',
                 'campaign-start', 'admission-budget', 'output', 'budget-ledger', 'log', 'incidents', 'lock'):
        p.add_argument('--' + name, type=Path, required=True)
    for side in ('screen', 'first', 'second'):
        for name in ('rows', 'guard', 'incidents', 'budget'):
            p.add_argument('--' + side + '-' + name, type=Path, required=True)
    p.add_argument('--gpu', required=True); p.add_argument('--confirmation-commit', required=True)
    p.add_argument('--seconds', type=float, required=True); p.add_argument('--timeout', type=int, default=600)
    p.add_argument('--idle-seconds', type=float, default=30)
    args = p.parse_args(argv)
    try:
        parsed = uuid.UUID(args.gpu.removeprefix('GPU-'))
        valid_gpu = args.gpu == 'GPU-' + str(parsed)
    except ValueError:
        valid_gpu = False
    if (os.name != 'posix' or not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal') or not valid_gpu or
        not math.isfinite(args.seconds) or not math.isfinite(args.idle_seconds) or args.seconds <= args.idle_seconds or
        args.timeout <= 0 or not 0 <= args.idle_seconds <= 3600):
        p.error('Linux pidfd, one canonical GPU UUID, and valid budget/timeout required')
    expected_lock = Path('/run/lock') / ('vkfft-study-' + args.gpu + '.lock')
    if args.lock.resolve() != expected_lock.resolve():
        p.error('lock must be the shared campaign GPU lock')
    if len({getattr(args, name).resolve() for name in ('output', 'budget_ledger', 'log', 'incidents', 'lock')}) != 5:
        p.error('evidence and lock paths must be distinct')
    root = Path(frontier.git(HERE, 'rev-parse', '--show-toplevel'))
    source_files = [HERE / name for name in ('runner.py', 'eligibility.py', 'prepare.py')]
    source_files += [EXPANDED / name for name in ('rotation_controller.py', 'rotation_pool.hpp', 'prepare_rotation.py',
                                                  'prepare_rotation_v2.py', 'rotation_cuda_compat.hpp', 'CMakeLists_rotation.txt',
                                                  'audit_guard.py', 'audit_pairs.py')]
    source_files += [STUDY / 'frontier_confirmation' / name for name in ('evidence.py', 'audit.py')]
    contract = frontier.provenance(root, source_files, [args.compile_commands, args.original_dependencies])
    need(os.environ.get('EXPERIMENT_CODE_COMMIT') == contract['execution_commit'], 'run through tools/run_experiment.py')
    code_commit = verify_declaration(root, contract['execution_commit'], os.environ.get('EXPERIMENT_ID'))
    paths = {side: {name: getattr(args, side + '_' + name) for name in ('rows', 'guard', 'incidents', 'budget')}
             for side in ('screen', 'first', 'second')}
    hot_binaries = {'original': args.hot_original, 'expanded': args.hot_expanded}
    proof = eligibility.verify(args.plan, paths['screen'], paths['first'], paths['second'], hot_binaries, args.gpu, args.confirmation_commit)
    need(proof['vkfft_commit'] == contract['vkfft_commit'], 'hot and current VkFFT commits differ')
    rot_binaries = {'original': args.rotation_original, 'expanded': args.rotation_expanded}
    need(args.rotation_original.resolve(strict=True) != args.rotation_expanded.resolve(strict=True), 'rotation binaries must differ')
    build = verify_rotation_build(root, args.rotation_root, args.generated_root, args.original_include_root,
                                  args.instrumentation_root, proof, rot_binaries, args.build_attestation,
                                  args.compile_commands, args.original_dependencies, code_commit)
    camp = campaign(root, args.campaign_record, args.campaign_start, args.admission_budget, args.gpu)
    deadline = min(time.monotonic() + args.seconds, camp['deadline'])
    contract.update(schema='frontier-rotation-session/v1', code_commit=code_commit, gpu=args.gpu,
                    eligibility=proof, rotation_build=build, campaign=camp, deadline_monotonic=deadline,
                    timeout=args.timeout, stage_seconds=args.seconds)
    for path in (args.output, args.budget_ledger, args.log, args.incidents, args.lock):
        path.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(EXPANDED))
    import device_lock
    with device_lock.DeviceLock(args.lock) as lock:
        need(not any(path.exists() for path in (args.output, args.budget_ledger, args.log, args.incidents)),
             'rotation requires fresh evidence; automatic resume disabled')
        remaining = deadline - time.monotonic()
        need(remaining > args.idle_seconds, 'rotation or global campaign deadline exhausted')
        ledger = DeadlineLedger(frontier.Ledger(args.budget_ledger, contract, remaining), deadline)
        for path in (args.output, args.log, args.incidents): path.touch(exist_ok=False)
        guard = guard_common.Guard(argparse.Namespace(gpu=args.gpu, log=args.log, incidents=args.incidents, idle_seconds=args.idle_seconds))
        old_popen, old_run = subprocess.Popen, suite.run_one
        keys = ('CUDA_VISIBLE_DEVICES', 'VKFFT_STUDY_GUARD_ID', 'VKFFT_STUDY_GUARD_UUID', 'VKFFT_STUDY_GUARD_LOG')
        saved = {key: os.environ.get(key) for key in keys}
        try:
            os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, VKFFT_STUDY_GUARD_ID=guard.guard_id,
                              VKFFT_STUDY_GUARD_UUID=args.gpu, VKFFT_STUDY_GUARD_LOG=str(args.log.resolve()))
            guard.emit('guard_start', script=str(Path(__file__).resolve()), idle_seconds=args.idle_seconds,
                       memory_limit_mib=guard_common.MEMORY_LIMIT_MIB, monitor_interval_seconds=1,
                       parent_pid=os.getpid(), own_signal_method='pidfd_with_rechecked_PPID_exe_starttime')
            guard.idle_check()
            subprocess.Popen = lock_inheriting_popen(guard.popen(old_popen), lock.fd, rot_binaries.values())
            suite.run_one = guard.wrapped(old_run); guard.start()
            try:
                run_session(args, ledger, contract, proof)
            except frontier.BudgetStop:
                frontier.append(args.output, dict(record_type='rotation_budget_exhausted', contract=contract, guard_id=guard.guard_id))
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
            for key, value in saved.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value


if __name__ == '__main__':
    main()
