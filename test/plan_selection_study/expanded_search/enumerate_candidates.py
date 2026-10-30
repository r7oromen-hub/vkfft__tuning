#!/usr/bin/env python3
"""Enumerate all restricted requests; static acceptance is not device validity."""
import argparse
import hashlib
import json
from pathlib import Path

RADICES = (2, 4, 8, 16, 32)
REGISTERS = (2, 4, 8, 16, 32, 64)


def decompositions(n):
    if n < 2 or n > 8192 or n & (n-1):
        raise ValueError('n must be a power of two in 2..8192')
    def rec(remaining):
        if remaining == 1:
            yield ()
        else:
            for radix in RADICES:
                if remaining % radix == 0:
                    for tail in rec(remaining//radix):
                        yield (radix,)+tail
    return rec(n)


def candidates(n, batch):
    if batch < 1 or batch > 65536:
        raise ValueError('batch outside 1..65536')
    groups = []
    group = 1
    while group <= batch:
        if batch % group == 0:
            groups.append(group)
        group *= 2
    for radix in decompositions(n):
        for regs in REGISTERS:
            reason = ('registers_exceed_fft_length' if regs > n else
                      'radix_exceeds_registers' if max(radix) > regs else None)
            for group in groups:
                request = {'radix': list(radix), 'logical_registers': regs, 'grouped_batch': group}
                key = json.dumps(request, sort_keys=True, separators=(',', ':')).encode()
                yield {'candidate_id': hashlib.sha256(key).hexdigest(), 'request': request,
                       'status': 'rejected_static' if reason else 'pending_scheduler_and_device_validation',
                       'reason': reason, 'planned_fft_threads': n//regs if reason is None else None,
                       'scope': 'power-of-two 1D FP32 C2C forward out-of-place, one upload, boost 1'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,required=True);p.add_argument('--batch',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with a.output.open('x',encoding='utf8') as out:
        for row in candidates(a.n,a.batch): out.write(json.dumps(row,sort_keys=True)+'\n')


if __name__=='__main__': main()
