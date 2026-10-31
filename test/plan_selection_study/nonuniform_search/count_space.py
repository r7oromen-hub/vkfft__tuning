#!/usr/bin/env python3
"""Count static v1 requests without enumerating their register tables.

Counts describe requests, not successful plans, distinct kernels or speedups.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

import candidate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTRACT = ROOT / 'experiments/research/nonuniform_register_admission_contract.json'
N_VALUES = tuple(2 ** exponent for exponent in range(7, 14))
BATCH_VALUES = (1, 16, 256, 4096)


def _check_domain():
    """The product rule relies on the v1 power-of-two register envelope."""
    for name, values in (('radices', candidate.RADICES), ('registers', candidate.REGISTERS)):
        if (not values or len(values) != len(set(values)) or
                any(type(v) is not int or v < 2 or v & (v - 1) for v in values)):
            raise ValueError(name + ' must be distinct powers of two')
    if set(candidate.PROFILES) != {'nonuniform_active_only', 'uniform_compatibility'}:
        raise ValueError('unsupported candidate profile domain')


def count_workload(n, batch):
    """Count profile requests and the exact overlap after removing profile.

    A table slot is shared by every occurrence of the same radix. Choices for
    distinct active radices are independent under this power-of-two contract.
    """
    _check_domain()
    candidate._workload(n, batch)
    by_active = Counter(tuple(sorted(set(order))) for order in candidate._decompositions(n))
    groups = tuple(candidate._groups(batch))
    active_total = active_uniform = compatibility = intersection = 0
    for active, orders in by_active.items():
        choices = {radix: tuple(r for r in candidate.REGISTERS
                                 if r >= radix and r <= n and r % radix == 0 and n % r == 0)
                   for radix in active}
        assignments = math.prod(len(values) for values in choices.values())
        uniform_values = set.intersection(*(set(values) for values in choices.values()))
        # Compatibility fills every supported slot dividing R, even when inactive.
        overlap = sum(set(active) == {radix for radix in candidate.RADICES if value % radix == 0}
                      for value in uniform_values)
        weight = orders * len(groups)
        active_total += assignments * weight
        active_uniform += len(uniform_values) * weight
        compatibility += len(uniform_values) * weight
        intersection += overlap * weight
    nonuniform = active_total - active_uniform
    return {
        'n': n, 'batch': batch, 'ordered_decompositions': sum(by_active.values()),
        'grouped_batch_values': list(groups),
        'active_only_uniform_requests': active_uniform,
        'active_only_truly_nonuniform_requests': nonuniform,
        'active_only_total_requests': active_total,
        'uniform_compatibility_requests': compatibility,
        'profile_distinct_total_requests': active_total + compatibility,
        'profile_ignored_exact_table_intersection': intersection,
        'profile_ignored_unique_requests': active_total + compatibility - intersection,
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_report():
    sources = {'candidate.py': HERE / 'candidate.py',
               'nonuniform_register_admission_contract.json': CONTRACT,
               'count_space.py': Path(__file__).resolve()}
    before = {name: _sha(path) for name, path in sources.items()}
    rows = [count_workload(n, batch) for n in N_VALUES for batch in BATCH_VALUES]
    if before != {name: _sha(path) for name, path in sources.items()}:
        raise ValueError('source changed during counting')
    count_fields = ('ordered_decompositions', 'active_only_uniform_requests',
                    'active_only_truly_nonuniform_requests', 'active_only_total_requests',
                    'uniform_compatibility_requests', 'profile_distinct_total_requests',
                    'profile_ignored_exact_table_intersection', 'profile_ignored_unique_requests')
    return {
        'schema': 'nonuniform-static-space-count/v1',
        'candidate_schema': candidate.SCHEMA,
        'source_sha256': before,
        'radices': list(candidate.RADICES), 'register_values': list(candidate.REGISTERS),
        'workloads': rows,
        'totals_across_workloads': {name: sum(row[name] for row in rows) for name in count_fields},
        'counting': 'Multiply legal register choices per distinct active radix for each ordered decomposition; multiply by eligible grouped-batch values.',
        'overlap_key': 'workload, ordered radix, complete 33-slot register table and grouped batch, omitting profile only',
        'limitations': [
            'Static request contract only: scheduler, GPU resources and numerical correctness are not established.',
            'Different requests and different tables may still generate identical code; no code-level deduplication is claimed.',
            'The counts are not a performance result or an upper bound on attainable speedup.',
            'The research contract file records its historical proposed status; counting uses the current candidate.py request contract.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='new JSON file within EXPERIMENT_ARTIFACT_DIR')
    args = parser.parse_args()
    directory = Path(os.environ['EXPERIMENT_ARTIFACT_DIR']).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError('EXPERIMENT_ARTIFACT_DIR must be an existing directory')
    output = args.output if args.output.is_absolute() else directory / args.output
    output = output.resolve()
    if not output.is_relative_to(directory) or output == directory:
        raise ValueError('output must be inside EXPERIMENT_ARTIFACT_DIR')
    report = make_report()
    with output.open('x', encoding='utf8') as dest:
        json.dump(report, dest, indent=2, sort_keys=True, allow_nan=False)
        dest.write('\n')


if __name__ == '__main__':
    main()
