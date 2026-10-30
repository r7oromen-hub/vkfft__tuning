#!/usr/bin/env python3
"""Enumerate prospective joint candidates from a same-device control JSON record.

These are requests, not proven valid plans: the generated C++ hook validates
scheduler registers and geometry; compiled resource and numerical checks follow.
"""
import argparse
import hashlib
import json
from pathlib import Path

RADICES = (2, 4, 8, 16, 32)

def decompositions(stages):
    if not stages or any(x not in RADICES for x in stages):
        raise ValueError("baseline must contain only supported power-of-two radices")
    if len(stages) <= 2:
        return [tuple(stages)]
    remaining = sum(x.bit_length()-1 for x in stages[1:-1])
    def compositions(exp):
        if exp == 0:
            yield ()
        for r in RADICES:
            e = r.bit_length()-1
            if e <= exp:
                for tail in compositions(exp-e):
                    yield (r,) + tail
    return [tuple([stages[0], *middle, stages[-1]])
            for middle in compositions(remaining) if len(middle) <= 30]

def candidates(result):
    f = result['structural_features']
    if f['numAxisUploads'] != 1 or len(f['kernels']) != 1:
        raise ValueError("requires one axis upload and one kernel")
    k = f['kernels'][0]
    d = result['precompile_kernels'][0]['raw_descriptor']
    n, batch = result['n'], result['batch']
    shared = result['resolved_config']['sharedMemorySize']
    local, group, swap = k['localSize'], k['groupedBatch'], d['axisSwapped']
    if n < 2 or n & (n-1) or d['fftDim'] != n or d['complexSize'] != 8 or swap not in (0,1) or local[2] != 1 or shared <= 0:
        raise ValueError("baseline outside 1D single-upload FP32 power-of-two scope")
    threads = local[1] if swap else local[0]
    if threads <= 0 or (local[0] if swap else local[1]) != group:
        raise ValueError("baseline groupedBatch/localSize inconsistent")
    seen = set()
    for radix in decompositions(k['stageRadix']):
        for g in range(1, min(batch, 1024) + 1):
            if batch % g or g & (g-1):
                continue
            # This reproduces the restricted splitter's observed swap rule.
            block = [g, threads, 1] if g > 1 and g*n < shared//8 else [threads, g, 1]
            key = (radix, g, tuple(block))
            if key in seen:
                continue
            seen.add(key)
            request = {'radix':list(radix), 'grouped_batch':g, 'block':block}
            digest = hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            active = radix[:-1] if d['registerBoost'] > 1 else radix
            regs = d['registers_per_radix']
            if g * threads > result['resolved_config']['maxThreadsNum'] or g * (n // d['registerBoost']) > shared // 8:
                reason = 'thread_or_shared_memory_precheck'
            elif not active or any(regs[r] <= 0 or regs[r] % r for r in active):
                reason = 'radix_register_table_missing_or_not_divisible'
            elif (min(regs[r] for r in active) != d['min_logical_registers_per_thread'] or
                  max(regs[r] for r in active) != d['logical_registers_per_thread']):
                reason = 'register_envelope_changed'
            else:
                reason = None
            yield {'candidate_id':digest, 'request':request,
                   'status':'rejected_precheck' if reason else 'pending_scheduler_resource_and_numerical_validation',
                   'reason':reason}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-json', required=True, type=Path, help='one same-device benchmark JSON row or result')
    parser.add_argument('--output', required=True, type=Path, help='new JSONL file; never overwritten')
    args = parser.parse_args()
    row = json.loads(args.baseline_json.read_text(encoding='utf8'))
    result = row.get('result', row)
    if result.get('status') != 'ok':
        raise ValueError('baseline must be a successful benchmark result')
    with args.output.open('x', encoding='utf8') as out:
        for candidate in candidates(result):
            out.write(json.dumps(candidate,sort_keys=True) + '\n')

if __name__ == '__main__':
    main()
