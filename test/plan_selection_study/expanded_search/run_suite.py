"""Guard-wrapped single benchmark subprocess with explicit environment."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

FIELDS = ('VKFFT_DIRECT_RADIX', 'VKFFT_DIRECT_BATCH', 'VKFFT_DIRECT_BLOCK',
          'VKFFT_EXPANDED_RADIX', 'VKFFT_EXPANDED_REGS', 'VKFFT_EXPANDED_BATCH')


def run_one(args, workload, candidate, phase, round_no, order):
    name, override = candidate
    mode = workload.get('study_mode', 'execute')
    if mode == 'describe':
        command = [str(Path(args.exe).resolve()), '--n', str(workload['n']), '--batch', str(workload['batch']),
                   '--study-mode', 'describe', '--placement', 'outofplace']
    elif mode == 'execute':
        command = [str(Path(args.exe).resolve()), '--n', str(workload['n']), '--batch', str(workload['batch']),
                   '--seed', str(workload['seed']), '--timing-mode', 'graph', '--placement', 'outofplace',
                   '--graph-repeats', str(workload['graph_repeats']), '--warmup', str(workload['warmup']),
                   '--warmup-ms', str(workload['warmup_ms']), '--repeats', str(workload['repeats']),
                   '--pattern', workload['pattern'], '--study-mode', 'execute']
    else:
        raise ValueError('invalid study mode')
    env = os.environ.copy()
    for key in FIELDS:
        env.pop(key, None)
    if set(override)-set(FIELDS):
        raise ValueError('unrecognized override variable')
    env.update(override)
    start = time.monotonic()
    row = dict(record_type='execution', session_id=args.session_id, invocation_id=uuid.uuid4().hex,
               phase=phase, workload_id=f"n{workload['n']}_b{workload['batch']}", workload=workload,
               candidate_id=name, round=round_no, order=order, command=command, direct_environment=override,
               executable_sha256=args.exe_sha256, validation_errors=[], started_monotonic=start,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        if hashlib.sha256(Path(args.exe).read_bytes()).hexdigest() != args.exe_sha256:
            raise RuntimeError('frozen executable hash changed')
        proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=args.timeout, check=False)
        row.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        parsed = [json.loads(x) for x in proc.stdout.splitlines() if x.strip()]
        if len(parsed) != 1 or not isinstance(parsed[0], dict):
            raise ValueError('expected exactly one JSON object')
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
