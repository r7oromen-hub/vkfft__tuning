#!/usr/bin/env python3
"""Partition audited frontier screening points by historical structural scope."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD_DIR = HERE.parent / 'expanded_search'
sys.path.insert(0, str(OLD_DIR))
import audit_old_space
import analyze

GROUPS = ('old_structural_subset', 'outside_old_space', 'not_comparable')
SOURCES = (Path(__file__).resolve(), HERE / 'analyze.py', OLD_DIR / 'audit_old_space.py')


def source_hashes():
    return {str(p): analyze.sha(p) for p in SOURCES}


def valid_layout(value):
    return (isinstance(value, list) and len(value) == 3 and
            all(type(x) is int and x > 0 for x in value))


def partition(coverage, rows, rows_sha256):
    # This also verifies coverage/raw hash binding, identity, case hashes,
    # disposition and paired timing records before any classification.
    audited = analyze.analyze(coverage, rows, rows_sha256, 20261031)
    baselines, identities, segments = {}, {}, {}
    active = None
    for row in rows:
        kind = row.get('record_type')
        if kind == 'execution' and row.get('phase') == 'identity':
            active = None
            if row.get('candidate_id') == 'expanded_default':
                w = row['workload']; key = (w['n'], w['batch'])
                analyze.need(key not in baselines, 'duplicate expanded default baseline')
                baselines[key] = row.get('result')
        elif kind == 'execution' and row.get('phase') == 'calibration':
            active = None
        elif kind == 'workload_status':
            key = (row['n'], row['batch'])
            if row.get('status') == 'identity_passed':
                identities[key] = row.get('request')
        elif kind == 'candidate_start':
            active = analyze.key(row)
            segments[active] = [row]
        elif active is not None:
            segments[active].append(row)

    grouped = defaultdict(lambda: {name: [] for name in GROUPS})
    for case in coverage['cases']:
        if case['status'] != 'screen_complete':
            continue
        key = analyze.key(case); work = key[:2]
        item = analyze.parse_complete(case, segments[key])
        baseline = baselines.get(work)
        layout = item['observed_plan']['local_size']
        if not isinstance(baseline, dict):
            mark = {'membership': 'not_comparable', 'reason': 'missing_expanded_default_baseline'}
        elif not valid_layout(layout):
            mark = {'membership': 'not_comparable', 'reason': 'missing_or_invalid_actual_layout'}
        else:
            mark = audit_old_space.membership(item['request'], baseline, layout)
        analyze.need(mark.get('membership') in GROUPS, 'unknown membership classification')
        identity_request = identities.get(work)
        item['is_identity'] = (item['request'] == identity_request
                               if isinstance(identity_request, dict) else None)
        item['old_space'] = mark
        grouped[work][mark['membership']].append(item)

    workloads = []
    for work in audited['workloads']:
        key = (work['n'], work['batch']); groups = grouped[key]
        report = {'n': key[0], 'batch': key[1], 'default_status': work['default_status'],
                  'selected_count': work['selected_count'], 'groups': {}}
        for name in GROUPS:
            members = sorted(groups[name], key=lambda x: x['candidate_id'])
            nonidentity = [item for item in members if item['is_identity'] is False]
            best = analyze.select_best(members)
            best_nonidentity = analyze.select_best(nonidentity)
            report['groups'][name] = {
                'count': len(members), 'identity_count': sum(x['is_identity'] is True for x in members),
                'identity_unknown_count': sum(x['is_identity'] is None for x in members),
                'best_screened_ratio': best['ratio'] if best else None,
                'best_nonidentity_screened_ratio': best_nonidentity['ratio'] if best_nonidentity else None,
                'candidates': members,
            }
        workloads.append(report)
    return {'schema': 'frontier-old-space-partition/v1', 'gpu': coverage['gpu'],
            'execution_commit': coverage['execution_commit'], 'source_rows_sha256': rows_sha256,
            'coverage_counts': audited['coverage_counts'], 'workloads': workloads,
            'scope': {
                'population': 'completed selected screen points only; rejected, partial and unmeasured requests excluded',
                'membership': 'conservative historical structural subset based on same-session baseline and observed layout; not source, compiler-options or runtime-state equivalence',
                'comparison': 'descriptive partition of one selected sample; neither a matched-budget old versus new search nor a complete old-domain optimum',
                'measurement': 'three-pair screening ratios against the original default; selected minima are not independent confirmation',
                'identity': 'identity is marked separately and included in its group; nonidentity minima exclude identity and unknown identity',
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = source_hashes()
    coverage_bytes = args.coverage.read_bytes()
    coverage_hash = hashlib.sha256(coverage_bytes).hexdigest()
    coverage = json.loads(coverage_bytes.decode('utf8'))
    rows_hash = analyze.sha(args.rows)
    rows = analyze.read_rows(args.rows)
    analyze.need(analyze.sha(args.rows) == rows_hash, 'raw rows changed while reading')
    result = partition(coverage, rows, rows_hash)
    analyze.need(analyze.sha(args.rows) == rows_hash, 'raw rows changed during analysis')
    analyze.need(analyze.sha(args.coverage) == coverage_hash, 'coverage changed during analysis')
    analyze.need(source_hashes() == sources, 'analysis source changed during analysis')
    result['input_sha256'] = {'coverage': coverage_hash, 'rows': rows_hash}
    result['analysis_source_sha256'] = sources
    with args.output.open('x', encoding='utf8') as out:
        json.dump(result, out, sort_keys=True, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({'status': 'complete', 'output': str(args.output)}))


if __name__ == '__main__':
    main()
