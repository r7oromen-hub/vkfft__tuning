#!/usr/bin/env python3
"""Summarize measured admission coverage separately from log integrity."""
import argparse
from collections import Counter
import json
from pathlib import Path

import audit_admission


def coverage(rows, protocol, integrity):
    statuses = {(x['n'], x['batch']): x for x in rows if x.get('record_type') == 'workload_status'}
    expected = {(w['n'], w['batch']) for w in protocol['workloads']}
    observed = []
    errors = []
    for d in rows:
        if d.get('record_type') != 'candidate_disposition' or d.get('status') != 'numerically_valid':
            continue
        n, batch, cid = d['n'], d['batch'], d['candidate_id']
        default = [x for x in rows if x.get('record_type') == 'execution' and x.get('candidate_id') == 'expanded_default' and
                   x.get('workload', {}).get('n') == n and x.get('workload', {}).get('batch') == batch and x.get('returncode') == 0]
        desc = [x for x in rows if x.get('record_type') == 'execution' and x.get('candidate_id') == cid and
                x.get('workload', {}).get('n') == n and x.get('workload', {}).get('batch') == batch and
                x.get('workload', {}).get('study_mode') == 'describe' and x.get('returncode') == 0]
        if len(default) != 1 or len(desc) != 1:
            errors.append(f'n{n}_b{batch}/{cid}: missing or repeated default/describe')
            continue
        try:
            baseline = default[0]['result']['structural_features']['kernels'][0]['stageRadix']
            request = d['request']; seq = request['radix']
            swap = desc[0]['result']['precompile_kernels'][0]['raw_descriptor']['axisSwapped']
            if swap not in (0, 1):
                raise ValueError('invalid precompile axisSwapped')
            observed.append({'n': n, 'batch': batch, 'candidate_id': cid, 'radix': seq,
                             'default_radix': baseline, 'logical_registers': request['logical_registers'],
                             'grouped_batch': request['grouped_batch'], 'axis_swapped': swap,
                             'first_changed': seq[0] != baseline[0], 'last_changed': seq[-1] != baseline[-1],
                             'multiset_changed': Counter(seq) != Counter(baseline)})
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            errors.append(f'n{n}_b{batch}/{cid}: {exc}')
    dimensions = {'fully_disposed_preregistered_workloads': set(statuses) == expected and len(statuses) == len(expected),
                  'first_radix_changed': any(x['first_changed'] for x in observed),
                  'last_radix_changed': any(x['last_changed'] for x in observed),
                  'radix_multiset_changed': any(x['multiset_changed'] for x in observed),
                  'multiple_register_values': len({x['logical_registers'] for x in observed}) >= 2,
                  'both_axis_orientations': {x['axis_swapped'] for x in observed} == {0, 1},
                  'at_least_two_lengths': len({x['n'] for x in observed}) >= 2,
                  'small_and_large_batch': {1, 4096} <= {x['batch'] for x in observed}}
    extras = {'register_values': sorted({x['logical_registers'] for x in observed}),
              'groups': sorted({x['grouped_batch'] for x in observed}),
              'lengths': sorted({x['n'] for x in observed}),
              'batches': sorted({x['batch'] for x in observed}),
              'axis_orientations': sorted({x['axis_swapped'] for x in observed}),
              'tested_R2': any(x['logical_registers'] == 2 for x in observed),
              'tested_R64': any(x['logical_registers'] == 64 for x in observed)}
    return {'status': 'ready_for_review' if integrity == 'pass' and not errors and all(dimensions.values()) else 'coverage_incomplete',
            'integrity_status': integrity, 'required_dimensions': dimensions, 'observed_dimensions': extras,
            'numerically_valid_candidates': observed, 'errors': errors,
            'scope': 'verified admission candidates only; R2/R64 are reported boundaries, not extra hard gates; any coverage recommendation is confined to this measured domain'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=Path, required=True); p.add_argument('--guard', type=Path, required=True)
    p.add_argument('--incidents', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    args = p.parse_args()
    rows = audit_admission.load(args.rows); protocol = json.loads(args.protocol.read_text(encoding='utf8'))
    integrity = audit_admission.audit(rows, audit_admission.load(args.guard), audit_admission.load(args.incidents), protocol)
    result = coverage(rows, protocol, integrity['status'])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'ready_for_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
