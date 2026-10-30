#!/usr/bin/env python3
"""Paired, guarded describe-only probe/control collection; no execution timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import compare_snapshots

FIELDS = ('VKFFT_DIRECT_RADIX', 'VKFFT_DIRECT_BATCH', 'VKFFT_DIRECT_BLOCK')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_one(args, workload, candidate, phase, round_no, order):
    name, _ = candidate
    command = [str(Path(args.exe).resolve()), '--n', str(workload['n']), '--batch', str(workload['batch']),
               '--study-mode', 'describe', '--placement', 'outofplace']
    env = os.environ.copy()
    for key in FIELDS:
        env.pop(key, None)
    row = {'phase': phase, 'workload_id': f"n{workload['n']}_b{workload['batch']}", 'workload': workload,
           'candidate_id': name, 'round': round_no, 'order': order, 'command': command,
           'exe_sha256': args.exe_sha256, 'direct_environment': {}}
    try:
        if digest(args.exe) != args.exe_sha256:
            raise RuntimeError('executable changed after freeze')
        proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=args.timeout, check=False)
        row.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError('expected exactly one JSON record')
        row['result'] = json.loads(lines[0])
        if not isinstance(row['result'], dict) or proc.returncode != 0:
            raise ValueError('harness failed or returned a non-object')
        row['status'] = 'ok'
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        row.update(status='failed', error=f'{type(error).__name__}: {error}')
        if isinstance(error, subprocess.TimeoutExpired):
            row.update(stdout=str(error.stdout or ''), stderr=str(error.stderr or ''))
    return row


def collect(args):
    if not os.environ.get('VKFFT_STUDY_GUARD_ID') or os.environ.get('VKFFT_STUDY_GUARD_UUID') != args.gpu:
        raise ValueError('must run under describe_gpu_guard with matching GPU')
    cases = []
    for item in args.case:
        n, batch = map(int, item.split(':'))
        if n < 2 or n > 8192 or n & (n - 1) or not 1 <= batch <= 65536:
            raise ValueError(f'unsupported case: {item}')
        cases.append({'n': n, 'batch': batch})
    if len({(c['n'], c['batch']) for c in cases}) != len(cases):
        raise ValueError('duplicate case')
    args.output.mkdir(parents=True, exist_ok=True)
    executables = {name: Path(path).resolve(strict=True) for name, path in [('probe', args.probe), ('baseline', args.baseline)]}
    hashes = {name: digest(path) for name, path in executables.items()}
    results = {'probe': [], 'baseline': []}
    with (args.output / 'runs.jsonl').open('x', encoding='utf8') as log:
        for case in cases:
            for order, name in enumerate(('probe', 'baseline')):
                a = SimpleNamespace(exe=executables[name], exe_sha256=hashes[name], timeout=args.timeout)
                row = run_one(a, case, (name, {}), 'describe', 0, order)
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                log.flush(); os.fsync(log.fileno())
                if row.get('status') != 'ok' or row.get('guard_clean') is not True:
                    raise ValueError('describe run failed or guard did not attest clean; see runs.jsonl')
                result = row['result']
                if (result.get('n'), result.get('batch')) != (case['n'], case['batch']) or result.get('gpu', {}).get('uuid') != args.gpu:
                    raise ValueError('workload/GPU identity mismatch')
                results[name].append(result)
    comparison = compare_snapshots.compare(results['probe'])
    comparison['baseline_comparison'] = compare_snapshots.compare_control(results['probe'], results['baseline'])
    for name in ('probe', 'baseline'):
        with (args.output / (name + '.jsonl')).open('x', encoding='utf8') as out:
            for result in results[name]:
                out.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
    with (args.output / 'comparison.json').open('x', encoding='utf8') as out:
        json.dump(comparison, out, ensure_ascii=False, indent=2); out.write('\n')
    return comparison


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', required=True)
    p.add_argument('--probe', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--case', required=True, action='append', metavar='N:BATCH')
    p.add_argument('--timeout', type=int, default=300, choices=range(1, 3601), metavar='SECONDS')
    return p


if __name__ == '__main__':
    parser().error('use describe_gpu_guard.py; unguarded GPU execution is disabled')
