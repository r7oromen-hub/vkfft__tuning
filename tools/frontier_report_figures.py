#!/usr/bin/env python3
"""Render the audited 28-workload fallback and two-session confirmation figure."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGGREGATE = ROOT / 'experiments/artifacts/frontier-20261031-returned-aggregate-001/aggregate.json'
DEFAULT_KNOBS = ROOT / 'experiments/artifacts/frontier-20261031-returned-knobs-audit-001/audit.json'


def read_input(path):
    data = path.read_bytes()
    return json.loads(data.decode('utf8')), hashlib.sha256(data).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(aggregate, knobs):
    need(aggregate.get('schema') == 'frontier-confirmed-aggregate/v1', 'unexpected aggregate schema')
    need(knobs.get('schema') == 'frontier-knobs-audit/v1' and knobs.get('status') == 'pass',
         'knob audit did not pass')
    need(aggregate.get('gpu') == knobs.get('gpu') and aggregate.get('gpu'), 'GPU mismatch')
    workloads = aggregate.get('workloads')
    need(isinstance(workloads, list) and len(workloads) == 28, 'expected 28 workloads')
    expected = {(n, batch) for n in (128, 256, 512, 1024, 2048, 4096, 8192)
                for batch in (1, 16, 256, 4096)}
    need({(w['n'], w['batch']) for w in workloads} == expected, 'workload matrix mismatch')
    confirmations = []
    for w in workloads:
        fallback = w.get('fallback')
        need(isinstance(fallback, dict) and finite(fallback.get('ratio')) and fallback['ratio'] > 0,
             'missing fallback ratio')
        conf = w.get('confirmation')
        if conf is not None:
            need(finite(conf.get('equal_weight_geometric_mean_ratio')) and
                 conf['equal_weight_geometric_mean_ratio'] > 0, 'invalid two-session estimate')
            for name in ('first', 'second'):
                session = conf[name]
                lo, value, hi = (session[k] for k in ('ci95_low', 'geometric_mean_ratio', 'ci95_high'))
                need(all(finite(x) and x > 0 for x in (lo, value, hi)) and lo <= value <= hi,
                     'invalid session estimate or interval')
            confirmations.append(w)
        practical = w.get('confirmation_status') == 'confirmed_practical'
        need((fallback.get('kind') == 'confirmed_candidate') == practical, 'fallback policy mismatch')
        if not practical:
            need(fallback.get('kind') == 'default' and fallback['ratio'] == 1.0,
                 'invalid default fallback')
        else:
            need(conf is not None and conf.get('both_sessions_practical') is True and
                 math.isclose(fallback['ratio'], conf['equal_weight_geometric_mean_ratio'], rel_tol=1e-12),
                 'confirmed fallback mismatch')
    need(len(confirmations) == 13 and sum(w['confirmation_status'] == 'confirmed_practical'
                                          for w in workloads) == 8,
         'unexpected confirmation coverage')
    candidates = knobs.get('candidates')
    need(isinstance(candidates, list) and len(candidates) == 30 and
         all(c.get('status') == 'screen_complete' for c in candidates), 'incomplete knob screen')
    plans = {(tuple(c['request']['radix']), c['request']['logical_registers'],
              c['request']['grouped_batch']) for c in candidates}
    need(len(plans) == 2 and knobs.get('workload') == {'n': 128, 'batch': 16},
         'unexpected knob screen scope')
    return sorted(workloads, key=lambda w: (w['n'], w['batch'])), sorted(
        confirmations, key=lambda w: (w['n'], w['batch']))


def label(workload):
    return f"{workload['n']} / {workload['batch']}"


def draw(workloads, confirmations, source_hashes, destination):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 11),
                                     gridspec_kw={'width_ratios': [1.05, 1.25]})
    fig.subplots_adjust(left=.09, right=.97, top=.875, bottom=.15, wspace=.43)
    fig.suptitle('Budget-limited FFT search: fallback and independent confirmations',
                 fontsize=17, fontweight='bold', x=.09, ha='left', y=.97)
    fig.text(.09, .935, 'N / batch  |  candidate vs original default  |  practical fallback: 8 / 28 workloads',
             fontsize=10, color='#46546a')

    values = [100 * (1 - w['fallback']['ratio']) for w in workloads]
    colors = ['#167d78' if w['fallback']['kind'] == 'confirmed_candidate' else '#c9d2dc'
              for w in workloads]
    positions = list(range(len(workloads)))
    left.barh(positions, values, color=colors, height=.7)
    left.scatter([0] * sum(v == 0 for v in values),
                 [i for i, v in enumerate(values) if v == 0], s=12, color='#8a99a8', zorder=3)
    for i, value in enumerate(values):
        left.text(value + .18, i, f'{value:.2f}%', va='center', fontsize=7.4, color='#344152')
    left.set_yticks(positions, [label(w) for w in workloads], fontsize=8)
    left.invert_yaxis()
    left.set_xlim(0, max(values) + 2.6)
    left.set_xlabel('Fallback latency reduction vs default (%)')
    left.set_title('28 workloads · default used when not practical', loc='left', fontweight='bold', pad=10)
    left.grid(axis='x', color='#e6ebef', linewidth=.7)
    left.set_axisbelow(True)

    # Draw each independently estimated within-session interval.  There is no
    # combined confidence interval for the two-session geometric mean.
    for i, w in enumerate(confirmations):
        for name, offset, color, marker in (('first', -.13, '#245a9a', 'o'),
                                             ('second', .13, '#d2782c', 's')):
            session = w['confirmation'][name]
            center = 100 * (1 - session['geometric_mean_ratio'])
            lo = 100 * (1 - session['ci95_high'])
            hi = 100 * (1 - session['ci95_low'])
            right.errorbar(center, i + offset, xerr=[[center - lo], [hi - center]],
                           fmt=marker, color=color, markersize=4, capsize=2,
                           elinewidth=1, markeredgewidth=.7, zorder=3)
        combined = 100 * (1 - w['confirmation']['equal_weight_geometric_mean_ratio'])
        right.plot(combined, i, marker='D', color='#273445', markersize=3.5, zorder=4)
        state = 'both pass' if w['confirmation_status'] == 'confirmed_practical' else 'below criterion'
        right.text(.99, i, state, transform=right.get_yaxis_transform(), ha='right', va='center',
                   fontsize=7.5, color='#3e6f65' if state == 'both pass' else '#7a6d5e')
    right.set_yticks(range(len(confirmations)), [label(w) for w in confirmations], fontsize=8)
    right.invert_yaxis()
    right.set_xlim(-.5, max(values) + 5.5)
    right.axvline(0, color='#9aa8b6', linewidth=.8)
    right.axvline(2, color='#8d9aa7', linewidth=.9, linestyle='--')
    right.text(2.08, -.75, '2% mean threshold', fontsize=7.5, color='#667787')
    right.set_xlabel('Session latency reduction vs default (%)')
    right.set_title('13 screen winners · session estimate and 95% CI', loc='left', fontweight='bold', pad=10)
    right.grid(axis='x', color='#e6ebef', linewidth=.7)
    right.set_axisbelow(True)
    right.legend(handles=[Line2D([], [], color='#245a9a', marker='o', linestyle='none', label='Session 1'),
                          Line2D([], [], color='#d2782c', marker='s', linestyle='none', label='Session 2'),
                          Line2D([], [], color='#273445', marker='D', linestyle='none', label='Two-session estimate'),
                          Patch(facecolor='#167d78', label='Confirmed practical fallback')],
                 loc='lower right', fontsize=8, frameon=False)

    fig.text(.09, .085,
             'Fallback uses the confirmed candidate only when both sessions show ≥2% mean reduction and CI upper ratio < 1; otherwise default.\n'
             'CIs are within-session paired intervals; they do not correct screening selection or multiple comparisons. No combined CI is shown.\n'
             'Separate knob audit: 30/30 requests completed screening on two fixed plans at N=128, batch=16; no knob confirmation.',
             fontsize=8.5, color='#46546a', linespacing=1.5)
    description = ('Source SHA-256: ' + '; '.join(f'{k}={v}' for k, v in source_hashes.items()) + '. '
                   'Per-session 95% intervals; no aggregate confidence interval. '
                   'Fallback uses candidates only when both sessions meet the practical criterion.')
    fig.savefig(destination / 'frontier_fallback_confirmation.svg', bbox_inches='tight',
                metadata={'Title': 'Budget-limited FFT search: fallback and confirmations',
                          'Description': description, 'Creator': 'frontier_report_figures.py'})
    fig.savefig(destination / 'frontier_fallback_confirmation.png', dpi=180, bbox_inches='tight',
                metadata={'Title': 'Budget-limited FFT search: fallback and confirmations',
                          'Description': description, 'Software': 'frontier_report_figures.py'})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument('--knobs-audit', type=Path, default=DEFAULT_KNOBS)
    args = parser.parse_args()
    destination = Path(os.environ['EXPERIMENT_ARTIFACT_DIR'])
    need(destination.is_dir(), 'EXPERIMENT_ARTIFACT_DIR must be an existing directory')
    for suffix in ('svg', 'png'):
        need(not (destination / f'frontier_fallback_confirmation.{suffix}').exists(), 'output already exists')
    aggregate, aggregate_sha = read_input(args.aggregate)
    knobs, knobs_sha = read_input(args.knobs_audit)
    workloads, confirmations = validate(aggregate, knobs)
    source_hashes = {'aggregate': aggregate_sha, 'knobs_audit': knobs_sha,
                     'screen_raw_rows': aggregate.get('source_rows_sha256'),
                     'confirmation_audit': aggregate.get('confirmation_report_sha256'),
                     'knob_raw_rows': (knobs.get('evidence_sha256') or {}).get('rows')}
    need(all(isinstance(value, str) and len(value) == 64 and
             all(c in '0123456789abcdef' for c in value) for value in source_hashes.values()),
         'missing or invalid source SHA-256')
    need(read_input(args.aggregate)[1] == aggregate_sha and read_input(args.knobs_audit)[1] == knobs_sha,
         'input changed while reading')
    draw(workloads, confirmations, source_hashes, destination)
    need(read_input(args.aggregate)[1] == aggregate_sha and read_input(args.knobs_audit)[1] == knobs_sha,
         'input changed during figure generation')
    print(json.dumps({'aggregate_sha256': aggregate_sha, 'knobs_audit_sha256': knobs_sha,
                      'outputs': ['frontier_fallback_confirmation.svg', 'frontier_fallback_confirmation.png']}))


if __name__ == '__main__':
    main()
