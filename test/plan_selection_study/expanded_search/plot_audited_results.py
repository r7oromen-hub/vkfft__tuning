#!/usr/bin/env python3
"""Plot already-audited screening distributions and paired-session intervals."""
import argparse
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--screen-summary',required=True,type=Path)
    p.add_argument('--combined-audit',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
    screen=json.loads(a.screen_summary.read_text(encoding='utf8'));report=json.loads(a.combined_audit.read_text(encoding='utf8'))
    combined=report.get('combined',report)
    if report.get('status')!='pass' or combined.get('status')!='pass':raise ValueError('two-session audit did not pass')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    cases=sorted(screen['cases'],key=lambda x:(x['batch'],x['n']));fig,ax=plt.subplots(figsize=(11,12),layout='constrained')
    for i,c in enumerate(cases):
        rng=random.Random(c['n']*100000+c['batch']);y=[i+rng.uniform(-.19,.19) for _ in c['all_ratios_sorted']]
        ax.scatter(c['all_ratios_sorted'],y,s=13,alpha=.42,color='#2460a8',edgecolors='none')
        ax.scatter([c['median_ratio']],[i],marker='|',s=120,color='#131820',linewidths=2)
    ax.axvline(1,color='#b4322a',linestyle='--',lw=1.2,label='original default')
    ax.set_yticks(range(len(cases)),[f"N={c['n']}, B={c['batch']}  (n={c['completed']})" for c in cases],fontsize=8)
    ax.invert_yaxis();ax.set_xscale('log');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right')
    ax.set(xlabel='candidate / original default elapsed time (log scale; lower is faster)',
           title='Screening: all completed candidates, median of 3 paired rounds\nExploratory points; bars mark within-workload medians')
    fig.savefig(a.output_dir/'screen_distribution.png',dpi=180,bbox_inches='tight');fig.savefig(a.output_dir/'screen_distribution.svg',bbox_inches='tight');plt.close(fig)
    confirmed=[x for x in combined['candidates'] if x.get('status')=='confirmed_under_preregistered_rule']
    if confirmed:
        fig,ax=plt.subplots(figsize=(9,max(3.4,len(confirmed)*.8+1.5)),layout='constrained')
        for i,c in enumerate(confirmed):
            for session,offset,color in (('first',-.13,'#1764a6'),('second',.13,'#b06b12')):
                d=c[session];m=d['geometric_mean_ratio'];ax.errorbar(m,i+offset,xerr=[[m-d['ci95_low']],[d['ci95_high']-m]],
                    fmt='o',markersize=5,capsize=3,color=color,label=session+' session' if i==0 else None)
        ax.axvline(1,color='#b4322a',linestyle='--',lw=1,label='original default')
        ax.axvline(.98,color='#777',linestyle=':',lw=1,label='2% threshold')
        ax.set_yticks(range(len(confirmed)),[f"N={x['n']}, B={x['batch']}" for x in confirmed]);ax.invert_yaxis()
        ax.set(xlabel='candidate / original default, paired geometric ratio',title='Independent sessions: 95% paired-log t intervals\nIntervals do not adjust for selection or multiple comparisons')
        ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8,loc='best')
        fig.savefig(a.output_dir/'confirmed_pair_intervals.png',dpi=180,bbox_inches='tight');fig.savefig(a.output_dir/'confirmed_pair_intervals.svg',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
