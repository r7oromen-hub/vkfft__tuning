#!/usr/bin/env python3
"""Freeze an explicit small radix/R/group by LUT/coalescing screening grid."""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frontier_knobs_request_manifest', HERE / 'requests.py')
requests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(requests)

MAX_CANDIDATES = 32


def build(specification):
    expected = {'workload', 'plans', 'lut_values', 'coalesced_values'}
    if not isinstance(specification, dict) or set(specification) != expected:
        raise ValueError('spec requires exactly workload, plans, lut_values and coalesced_values')
    workload = specification['workload']
    if not isinstance(workload, dict) or set(workload) != {'n', 'batch'}:
        raise ValueError('workload requires exactly n and batch')
    n, batch = workload['n'], workload['batch']
    plans, luts, coalesced = (specification[k] for k in ('plans', 'lut_values', 'coalesced_values'))
    if not all(isinstance(x, list) and x for x in (plans, luts, coalesced)):
        raise ValueError('plans and knob lists must be nonempty lists')
    if len(plans) * len(luts) * len(coalesced) > MAX_CANDIDATES:
        raise ValueError('grid exceeds small-screen limit of 32')
    for value in luts:
        requests.validate_knobs(value, 0)
    for value in coalesced:
        requests.validate_knobs(0, value)
    if len(set(luts)) != len(luts) or len(set(coalesced)) != len(coalesced):
        raise ValueError('duplicate knob value')
    rows, seen = [], set()
    for plan, lut, coal in itertools.product(plans, luts, coalesced):
        if not isinstance(plan, dict) or set(plan) != {'radix', 'logical_registers', 'grouped_batch'}:
            raise ValueError('each plan requires exactly radix, logical_registers and grouped_batch')
        req = dict(plan, use_lut=lut, coalesced_memory=coal)
        reason = requests.static_reason(n, batch, req)
        if reason:
            raise ValueError(f'statically rejected plan: {reason}')
        cid = requests.candidate_id(n, batch, req)
        if cid in seen:
            raise ValueError('duplicate candidate')
        seen.add(cid)
        rows.append({'candidate_id': cid, 'request': req})
    return {'schema': 'frontier-knobs-screen/v1', 'workload': workload, 'candidates': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.spec.read_text(encoding='utf8')))
    with args.output.open('x', encoding='utf8', newline='\n') as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()
