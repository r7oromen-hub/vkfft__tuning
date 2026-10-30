#!/usr/bin/env python3
"""Guarded, no-override VkFFT default structure and numerical probe."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import run_suite as suite

LENGTHS = (256, 1024, 8192)
BATCHES = (1, 16, 4096)

def validate(row, args, workload):
    if row.get('status') != 'ok' or row.get('guard_clean') is not True:
        raise ValueError('process failed or was not guarded')
    if row.get('direct_environment') != {} or workload.get('config') != {}:
        raise ValueError('default must have no injection or tuning overrides')
    r = row.get('result', {})
    if r.get('status') != 'ok' or r.get('gpu', {}).get('uuid') != args.gpu:
        raise ValueError('benchmark status or GPU UUID mismatch')
    if 'RTX 4090' not in r.get('gpu', {}).get('name', ''):
        raise ValueError('expected an RTX 4090')
    if (r.get('n'), r.get('batch'), r.get('seed')) != (workload['n'], workload['batch'], workload['seed']):
        raise ValueError('workload mismatch')
    if r.get('requested_config') != {'makeForwardPlanOnly': 1}:
        raise ValueError('non-default tuning request')
    if r.get('validation_output_initialization') != '0xff_bytes_fp32_nan_before_single_and_final_timed_forward':
        raise ValueError('output poison missing')
    if r.get('input_unchanged') is not True or r.get('single_forward_input_unchanged') is not True:
        raise ValueError('immutable input changed')
    for key in ('single_forward_correctness', 'correctness'):
        c = r.get(key, {})
        if c.get('passed') is not True or c.get('nonfinite_count') != 0 or c.get('failing_batches') != 0 or c.get('checked_elements') != workload['n'] * workload['batch']:
            raise ValueError(key + ' failed or incomplete')
    if r.get('descriptor_validation', {}).get('all_match') is not True:
        raise ValueError('precompile and postinit descriptors mismatch')
    s = r.get('structural_features', {})
    kernels = s.get('kernels', [])
    pre = r.get('precompile_kernels', [])
    if s.get('numAxisUploads') != 1 or len(kernels) != 1 or len(pre) != 1:
        raise ValueError('expected exactly one upload')
    k, p = kernels[0], pre[0]
    desc = p.get('canonical_descriptor', {})
    if not p.get('raw_descriptor') or not desc or p.get('compiler_options_match') is not True:
        raise ValueError('precompile descriptor or compiler options missing')
    if not re.fullmatch(r'[0-9a-f]{64}', p.get('source_sha256', '')) or not isinstance(p.get('source_bytes'), int) or p['source_bytes'] <= 0:
        raise ValueError('source hash or size missing')
    if p.get('compiler_input', {}).get('source_sha256') != p['source_sha256'] or p.get('actual_compiler_options') != p.get('compiler_input', {}).get('options'):
        raise ValueError('source or compiler input mismatch')
    local = k.get('localSize')
    if not isinstance(local, list) or len(local) != 3 or not all(type(x) is int and x > 0 for x in local):
        raise ValueError('invalid actual localSize')
    if desc.get('localSize') != local or desc.get('stageRadix') != k.get('stageRadix'):
        raise ValueError('actual structure differs from precompile')
    if r.get('graph_repeats') != 1 or r.get('graph_kernel_count') != 1 or r.get('graph_event_record_count') != 2:
        raise ValueError('graph shape mismatch')
    samples = r.get('exec_ms_samples', [])
    if len(samples) != 1 or not isinstance(samples[0], (int, float)) or not math.isfinite(samples[0]) or samples[0] <= 0:
        raise ValueError('missing execution sample')
    return r

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--n', required=True, type=int, choices=LENGTHS)
    p.add_argument('--batch', required=True, type=int, choices=BATCHES)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--timeout', type=int, default=300)
    args = p.parse_args()
    if not os.environ.get('VKFFT_STUDY_GUARD_ID') or os.environ.get('VKFFT_STUDY_GUARD_UUID') != args.gpu:
        p.error('must run through default_gpu_guard for the same GPU UUID')
    if not 1 <= args.timeout <= 3600 or not 0 <= args.seed <= 0xffffffff:
        p.error('invalid timeout or seed')
    args.exe_sha256 = hashlib.sha256(Path(args.exe).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workload = dict(n=args.n, batch=args.batch, seed=args.seed, config={}, pattern='random',
                    graph_repeats=1, warmup=0, warmup_ms=0, repeats=1)
    with args.output.open('x', encoding='utf8') as out:
        def emit(row):
            out.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
        row = suite.run_one(args, workload, ('default', {}), 'default_probe', 0, 0)
        try:
            validate(row, args, workload)
        except (ValueError, KeyError, TypeError) as error:
            row['status'] = 'failed'
            row.setdefault('validation_errors', []).append(str(error))
            emit(row)
            raise
        emit(row)
        emit(dict(record_type='controller_complete', workload=workload, exe_sha256=args.exe_sha256,
                  note='structure probe only; single execution sample is not a timing comparison'))

if __name__ == '__main__':
    main()
