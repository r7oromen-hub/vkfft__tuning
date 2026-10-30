"""The only harness subprocess entry, wrapped by direct_gpu_guard.py."""
import datetime, hashlib, json, os, subprocess, time
from pathlib import Path

FIELDS = ('VKFFT_DIRECT_RADIX', 'VKFFT_DIRECT_BATCH', 'VKFFT_DIRECT_BLOCK')
def run_one(args, workload, candidate, phase, round_no, order):
    name, override = candidate
    command = [str(Path(args.exe).resolve()), '--n', str(workload['n']), '--batch', str(workload['batch']),
               '--seed', str(workload['seed']), '--timing-mode', 'graph', '--placement', 'outofplace',
               '--graph-repeats', str(workload['graph_repeats']), '--warmup', str(workload['warmup']), '--warmup-ms', str(workload['warmup_ms']),
               '--repeats', str(workload['repeats']), '--pattern', workload['pattern'], '--study-mode', 'execute']
    for key, value in workload.get('config', {}).items():
        command.extend(['--' + key.replace('_','-'), str(value)])
    env = os.environ.copy()
    for key in FIELDS: env.pop(key, None)
    env.update(override)
    started = time.monotonic()
    row = dict(phase=phase, workload_id=f"n{workload['n']}_b{workload['batch']}", workload=workload, candidate_id=name, round=round_no, order=order,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), validation_errors=[],
               command=command, direct_environment=override)
    try:
        if hashlib.sha256(Path(args.exe).read_bytes()).hexdigest() != args.exe_sha256:
            raise RuntimeError('executable changed after freeze')
        proc = subprocess.run(command, env=env, text=True, capture_output=True, timeout=args.timeout, check=False)
        row.update(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        parsed=[json.loads(x) for x in proc.stdout.splitlines() if x.strip()]
        if len(parsed)!=1 or not isinstance(parsed[0],dict): raise ValueError('expected exactly one JSON record')
        row['result']=parsed[0]
        row['status']='ok' if proc.returncode==0 else 'failed'
        if proc.returncode: row['validation_errors'].append(f'harness_exit:{proc.returncode}')
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:
        row.update(status='failed', error=f'{type(error).__name__}: {error}')
        row['validation_errors'].append(row['error'])
        if isinstance(error, subprocess.TimeoutExpired):
            row.update(stdout=str(error.stdout or ''), stderr=str(error.stderr or ''))
    row['wall_seconds']=time.monotonic()-started
    row['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    return row
