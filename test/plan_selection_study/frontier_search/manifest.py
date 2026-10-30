#!/usr/bin/env python3
"""Offline manifest for the existing, restricted expanded-search prototype.

The manifest is a selection plan, never evidence of scheduler or device validity.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'expanded_search'))
import enumerate_candidates as prototype

REGISTERS = (2, 4, 8, 16, 32, 64, 128, 256)


def candidates(n, batch):
    # Retain the prototype's exact workload and ordered radix domain while
    # explicitly extending its logical-register request values.
    if batch < 1 or batch > 65536:
        raise ValueError('batch outside 1..65536')
    groups = []
    group = 1
    while group <= batch:
        if batch % group == 0:
            groups.append(group)
        group *= 2
    for radix in prototype.decompositions(n):
        for regs in REGISTERS:
            reason = ('registers_exceed_fft_length' if regs > n else
                      'radix_exceeds_registers' if max(radix) > regs else None)
            for group in groups:
                request = {'radix': list(radix), 'logical_registers': regs, 'grouped_batch': group}
                yield {'candidate_id': candidate_id(request), 'request': request,
                       'status': 'rejected_static' if reason else 'pending_scheduler_and_device_validation',
                       'reason': reason, 'planned_fft_threads': n // regs if reason is None else None,
                       'scope': 'power-of-two 1D FP32 C2C forward out-of-place, one upload, boost 1'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def candidate_id(request):
    return hashlib.sha256(canonical(request).encode()).hexdigest()


def rank(seed, value):
    return hashlib.sha256(canonical([seed, value]).encode()).hexdigest()


def estimate(row):
    """Relative scheduling weight only, not time or device resource usage."""
    return row['planned_fft_threads'] * row['request']['grouped_batch'] * len(row['request']['radix'])


def neighbor_ids(request, rows_by_id):
    """One adjacent stage swap or one adjacent R/group value within the domain."""
    neighbors = set()
    radix, regs, group = request['radix'], request['logical_registers'], request['grouped_batch']
    groups = sorted({row['request']['grouped_batch'] for row in rows_by_id.values()})
    for values, value, field in ((REGISTERS, regs, 'logical_registers'),
                                 (groups, group, 'grouped_batch')):
        if value not in values:
            continue
        i = values.index(value)
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                changed = dict(request, **{field: values[j]})
                neighbors.add(candidate_id(changed))
    for i in range(len(radix) - 1):
        changed = list(radix)
        changed[i], changed[i + 1] = changed[i + 1], changed[i]
        neighbors.add(candidate_id(dict(request, radix=changed)))
    return {cid for cid in neighbors if cid in rows_by_id and cid != candidate_id(request)}


def build(n, batch, identity, historical, limit, seed=20261031):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError('limit must be a nonnegative integer')
    if not isinstance(historical, list):
        raise ValueError('historical must be a list, possibly empty')
    rows = list(candidates(n, batch))
    by_id = {row['candidate_id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate candidate ID')
    anchors = []
    forced = defaultdict(set)
    # An unknown default is not interchangeable with a fabricated one. The caller
    # must deliberately supply null when it has not yet observed the identity.
    for label, request in [('identity', identity)] + [('historical', item) for item in historical]:
        if request is None and label == 'identity':
            anchors.append({'kind': label, 'status': 'not_provided'})
            continue
        if not isinstance(request, dict):
            raise ValueError(f'{label} request must be an object')
        cid = candidate_id(request)
        row = by_id.get(cid)
        status = ('outside_prototype_domain' if row is None else
                  row['reason'] if row['status'] == 'rejected_static' else 'included')
        anchors.append({'kind': label, 'candidate_id': cid, 'request': request, 'status': status})
        if status != 'included':
            continue
        forced[cid].add(label)
        if label == 'identity':
            for nid in neighbor_ids(request, by_id):
                if by_id[nid]['status'] != 'rejected_static':
                    forced[nid].add('identity_neighbor')
    if len(forced) > limit:
        raise ValueError(f'limit {limit} smaller than {len(forced)} mandatory valid anchors and neighbors')

    for row in rows:
        row['selection'] = 'rejected_static' if row['reason'] else 'not_selected'
        row['selection_reasons'] = []
    selected = []
    for cid in sorted(forced, key=lambda x: rank(seed, x)):
        row = by_id[cid]
        row['selection'] = 'selected'
        row['selection_reasons'] = sorted(forced[cid])
        selected.append(row)

    # Stage count/R/group are explicit strata. Within each stratum, interleave
    # radix multisets before selecting further orderings of the same multiset.
    strata = defaultdict(lambda: defaultdict(list))
    weights = Counter()
    for row in selected:
        req = row['request']
        key = (len(req['radix']), req['logical_registers'], req['grouped_batch'])
        weights[key] += estimate(row)
    for row in rows:
        if row['selection'] != 'not_selected':
            continue
        req = row['request']
        key = (len(req['radix']), req['logical_registers'], req['grouped_batch'])
        strata[key][tuple(sorted(req['radix']))].append(row)
    queues = {}
    for key, multisets in strata.items():
        groups = []
        for multiset, entries in multisets.items():
            entries.sort(key=lambda row: rank(seed, row['candidate_id']))
            groups.append((multiset, deque(entries)))
        groups.sort(key=lambda item: rank(seed, [key, item[0]]))
        queues[key] = deque(groups)
    while queues and len(selected) < limit:
        key = min(queues, key=lambda k: (weights[k], rank(seed, k)))
        multiset, entries = queues[key].popleft()
        row = entries.popleft()
        if entries:
            queues[key].append((multiset, entries))
        if not queues[key]:
            del queues[key]
        row['selection'] = 'selected'
        row['selection_reasons'] = ['stratified_cost_balance']
        selected.append(row)
        weights[key] += estimate(row)
    for index, row in enumerate(selected):
        row['selection_index'] = index
        row['estimated_cost_weight'] = estimate(row)
    reasons = Counter(row['reason'] for row in rows if row['reason'])
    header = {
        'record_type': 'manifest', 'schema': 'frontier-manifest/v1',
        'workload': {'n': n, 'batch': batch}, 'seed': seed, 'limit': limit,
        'source_sha256': {
            'manifest.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'expanded_search/enumerate_candidates.py': hashlib.sha256(Path(prototype.__file__).read_bytes()).hexdigest(),
        },
        'domain': 'frontier_v2_power_of_two_uniform_registers_up_to_256',
        'validity': 'static screening only; all selected requests require scheduler and device validation',
        'cost_weight': 'n / logical_registers * grouped_batch * stage_count; relative proxy only',
        'enumerated_count': len(rows), 'static_rejected_count': sum(reasons.values()),
        'static_rejection_reasons': dict(sorted(reasons.items())),
        'pending_count': len(rows) - sum(reasons.values()), 'selected_count': len(selected),
        'anchors': anchors,
        'coverage_limit': 'all requests in this prototype domain enumerated; selection is not exhaustive',
    }
    return header, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    expected = {'n', 'batch', 'identity', 'historical', 'limit', 'seed'}
    if set(spec) != expected:
        raise ValueError('spec must contain exactly: ' + ', '.join(sorted(expected)))
    header, rows = build(**spec)
    with args.output.open('x', encoding='utf-8', newline='\n') as out:
        for row in [header, *rows]:
            out.write(canonical(row) + '\n')


if __name__ == '__main__':
    main()
