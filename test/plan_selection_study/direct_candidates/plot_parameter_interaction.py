#!/usr/bin/env python3
"""Render two complete, re-audited candidate grids for offline research."""
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / 'results' / 'joint_newhost_20261028'
data = json.loads((ROOT / 'parameter_structure_summary.json').read_text(encoding='utf8'))
cases = {item['case']: item for item in data['cases']}
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
a = cases['n256_b16']
rads = a['radices']
groups = a['groups']
lookup = {(tuple(x['radix']), x['group']): x['candidate_over_default'] for x in a['screen_table']}
grid = np.array([[lookup[tuple(r), g] for g in groups] for r in rads])
im = axes[0].imshow(grid, aspect='auto', cmap='YlOrRd', vmin=1, vmax=3)
for i in range(len(rads)):
    for j in range(len(groups)):
        axes[0].text(j, i, f'{grid[i,j]:.2f}', ha='center', va='center', fontsize=9)
axes[0].set_xticks(range(len(groups)), groups)
axes[0].set_yticks(range(len(rads)), ['-'.join(map(str, x)) for x in rads])
axes[0].set(xlabel='FFTs per block', ylabel='Radix sequence', title='N=256, batch=16: candidate / default')
fig.colorbar(im, ax=axes[0], label='Latency ratio', shrink=.82)
b = cases['n1024_b16']
lookup = {(tuple(x['radix']), x['group']): x['candidate_over_default'] for x in b['screen_table']}
for r, marker in [((8,4,4,4,2), 'o'), ((8,8,2,4,2), 's')]:
    groups = [4, 8]
    axes[1].plot(groups, [lookup[r,g] for g in groups], marker=marker, linewidth=2,
                 label='-'.join(map(str,r)))
axes[1].set_xticks([4,8])
axes[1].set(xlabel='FFTs per block', ylabel='Latency / default', title='N=1024, batch=16: ordering reverses')
axes[1].legend(title='Radix sequence', fontsize=9)
axes[1].grid(alpha=.2)
fig.suptitle('Screening data: three paired process rounds per candidate\nFixed endpoints, register envelope and FFT-thread dimension', fontsize=11)
fig.savefig(ROOT / 'parameter_interactions.png', dpi=180, bbox_inches='tight')
