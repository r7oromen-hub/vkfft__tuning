#!/usr/bin/env python3
"""Re-audit completed joint runs and summarize their finite-grid interactions."""
import argparse
import collections
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import audit_joint

ROOT = Path(__file__).resolve().parent / 'results' / 'joint_newhost_20261028'
CASES = ('n256_b16', 'n1024_b16', 'n8192_b16', 'n256_b4096')


def load_case(case):
    root = ROOT / ('joint_' + case)
    records, e1 = audit_joint.load(Path(str(root) + '_rows.jsonl'))
    guard, e2 = audit_joint.load(Path(str(root) + '_guard.jsonl'))
    incidents, e3 = audit_joint.load(Path(str(root) + '_incidents.jsonl'))
    if e1 or e2 or e3:
        raise ValueError(f'{case} invalid files: {e1 + e2 + e3}')
    binary = hashlib.sha256((ROOT / 'vkfft_direct_candidate').read_bytes()).hexdigest()
    check = audit_joint.audit(records, guard, incidents, binary)
    if check['status'] != 'complete' or check['errors']:
        raise ValueError(f'{case} failed independent audit: {check["errors"]}')
    values = {(tuple(x['request']['radix']), x['request']['grouped_batch']): x['screen_ratio'] for x in check['screened']}
    base = check['default_structure']
    identity = (tuple(base['radix']), base['grouped_batch'])
    if identity in values:
        raise ValueError('default identity appears in candidate timings')
    values[identity] = 1.0
    manifests = [x for x in records if x.get('record_type') == 'candidate_manifest']
    initializations = [x['result']['initialize_ms'] for x in records
                       if x.get('phase') == 'numerical' and x.get('result', {}).get('initialize_ms') is not None]
    groups = sorted({x[1] for x in values})
    radices = sorted({x[0] for x in values})
    complete = all((r, g) in values for r in radices for g in groups)
    fractions = None
    if complete and len(groups) > 1 and len(radices) > 1:
        y = {(r, g): math.log(values[r, g]) for r in radices for g in groups}
        overall = statistics.mean(y.values())
        avg_r = {r: statistics.mean(y[r, g] for g in groups) for r in radices}
        avg_g = {g: statistics.mean(y[r, g] for r in radices) for g in groups}
        s_r = len(groups) * sum((v - overall) ** 2 for v in avg_r.values())
        s_g = len(radices) * sum((v - overall) ** 2 for v in avg_g.values())
        s_i = sum((y[r, g] - avg_r[r] - avg_g[g] + overall) ** 2 for r in radices for g in groups)
        total = s_r + s_g + s_i
        fractions = {'radix': s_r / total, 'group': s_g / total, 'interaction': s_i / total} if total else None
    comparisons = []
    for a, b in itertools.combinations(radices, 2):
        for low, high in itertools.combinations(groups, 2):
            if not all((r, g) in values for r, g in ((a, low), (b, low), (a, high), (b, high))):
                continue
            ratio_low, ratio_high = values[a, low] / values[b, low], values[a, high] / values[b, high]
            if (ratio_low - 1) * (ratio_high - 1) < 0:
                comparisons.append({'radix_a': list(a), 'radix_b': list(b), 'group_low': low, 'group_high': high,
                                    'a_over_b_low': ratio_low, 'a_over_b_high': ratio_high,
                                    'strength': min(abs(math.log(ratio_low)), abs(math.log(ratio_high)))})
    return {'case': case, 'counts': check['counts'], 'manifest_size': len(manifests),
            'default': base, 'confirmation': check['confirmation'],
            'radices': [list(r) for r in radices], 'groups': groups, 'complete_grid': complete,
            'finite_grid_log_variance_fractions': fractions,
            'best_reversal': max(comparisons, key=lambda x: x['strength']) if comparisons else None,
            'reversal_count': len(comparisons),
            'screen_ratio_min': min(values.values()), 'screen_ratio_max': max(values.values()),
            'initialize_ms': {'median': statistics.median(initializations), 'min': min(initializations),
                              'max': max(initializations), 'observations': len(initializations)},
            'screen_table': [{'radix': list(r), 'group': g, 'candidate_over_default': values[r, g]}
                             for r in radices for g in groups if (r, g) in values]}


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root
    results = [load_case(c) for c in CASES]
    target = ROOT / 'parameter_structure_summary.json'
    target.write_text(json.dumps({'schema': 'finite-parameter-analysis/v1', 'cases': results},
                                 ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for x in results:
        print(x['case'], 'requests', x['manifest_size'], 'status', x['counts'],
              'ratios', round(x['screen_ratio_min'], 6), round(x['screen_ratio_max'], 6),
              'finite-grid variance', x['finite_grid_log_variance_fractions'],
              'best reversal', x['best_reversal'])
    print('saved', target)


if __name__ == '__main__':
    main()
