"""Seeded stage/multiset/planned-layout stratification, before timing."""
from collections import defaultdict, deque
import random


def select(records, identity_request, seed=20261030, limit=64):
    if not 0 <= limit <= 64:
        raise ValueError('per-workload selected request cap must be 0..64')
    records = list(records)
    if len({r['candidate_id'] for r in records}) != len(records):
        raise ValueError('duplicate candidate IDs')
    rng = random.Random(seed)
    by_stage = defaultdict(lambda: defaultdict(list))
    for row in records:
        req = row['request']
        if row['status'] not in ('rejected_static', 'pending_scheduler_and_device_validation'):
            raise ValueError('unknown enumeration status')
        row['selection'] = ('rejected_static' if row['status'] == 'rejected_static' else
                            'identity_control' if req == identity_request else 'not_selected')
        if row['selection'] != 'not_selected':
            continue
        radix = req['radix']
        key = (tuple(sorted(radix)), row['planned_fft_threads'], req['grouped_batch'])
        by_stage[len(radix)][key].append(row)
    stages = sorted(by_stage)
    rng.shuffle(stages)
    buckets = {}
    for stage in stages:
        keys = sorted(by_stage[stage]); rng.shuffle(keys)
        for key in keys: rng.shuffle(by_stage[stage][key])
        buckets[stage] = deque((key, deque(by_stage[stage][key])) for key in keys)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for stage in stages:
            if len(selected) >= limit: break
            if not buckets[stage]: continue
            key, group = buckets[stage].popleft()
            row = group.popleft(); row['selection'] = 'selected'
            row['selection_index'] = len(selected); selected.append(row)
            if group: buckets[stage].append((key, group))
    return records, selected
