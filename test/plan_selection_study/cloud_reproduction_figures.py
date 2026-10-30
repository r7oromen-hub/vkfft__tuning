#!/usr/bin/env python3
"""Render audited old/new VkFFT cohort figures without pooling observations.

Input is the separately audited cloud_report.json and the original cohort's
independent S1/S2 audit JSON files. No GPU is used. The output directory must
not already exist, so an incomplete report cannot leave stale graphs in place.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, NullFormatter, FixedLocator


BLUE = '#25639a'
ORANGE = '#c2712f'
GRAY = '#737b83'
COMPILATION = {'nvrtc_program_create', 'nvrtc_compile', 'nvrtc_program_destroy',
               'binary_extract', 'module_load', 'module_get_function', 'function_attribute_setup'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def configure_font():
    paths = [Path('C:/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/simhei.ttf'),
             Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    for path in paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'svg.fonttype':'none',
                         'figure.facecolor':'white', 'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.facecolor':'white'})


def finite_positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def checked_old(directory):
    directory = Path(directory)
    s1_review = read(directory/'independent_s1_review.json')
    s1_data = read(directory/'independent_s1_audit_data.json')
    s2_review = read(directory/'independent_s2_review.json')
    s2_data = read(directory/'independent_s2_audit_data.json')
    study = 'plan_selection_20261027_r2'
    if s1_review.get('study_id') != study or s2_review.get('study_id') != study or s2_data.get('study_id') != study:
        raise ValueError('old_review_cohort_identity_mismatch')
    if s2_review.get('audit',{}).get('result') != 'passed' or s1_review.get('audit',{}).get('complete_workload_blocks') != 36:
        raise ValueError('old_independent_reviews_not_complete')
    # Where the retained source files are present, require exact audit-source hashes.
    cohort = directory.parent
    for relative, expected in s2_data.get('source_sha256',{}).items():
        source = cohort/relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('old_review_source_hash_mismatch:' + relative)
    return s1_data, s2_data


def save(fig, directory, stem, synthetic=False):
    if synthetic:
        fig.text(.5,.5,'合成数据 · 布局测试',ha='center',va='center',fontsize=32,color='#bd4a48',alpha=.20,rotation=25)
    for ext in ('png','svg'):
        fig.savefig(directory/(stem+'.'+ext),dpi=190,bbox_inches='tight')
    plt.close(fig)


def h1(report, old_s2, out, synthetic):
    tests = [t for t in report.get('directional_tests',[]) if t.get('hypothesis') == 'H1']
    old = {x['workload_id']:x for x in old_s2['workloads']}
    if len(tests)!=3 or not all(t.get('quality_valid') and finite_positive(t.get('speedup')) and
                                t['workload_id'] in old and old[t['workload_id']]['winner']==t['target'] and
                                len(old[t['workload_id']]['paired_ratios'])==3 for t in tests):
        return False
    fig, ax = plt.subplots(figsize=(9.7,5.0))
    fig.subplots_adjust(left=.23,right=.80,top=.82,bottom=.31)
    ax.axvline(1,color='#a4adb5',lw=1)
    ax.axvline(1.05,color='#b9c2ca',lw=1,ls='--')
    for index, t in enumerate(tests):
        y = len(tests)-1-index
        previous = old[t['workload_id']]
        ratios = previous['paired_ratios']
        ax.scatter(ratios,[y+.15]*len(ratios),s=17,color=GRAY,alpha=.55,zorder=3)
        ax.scatter(previous['confirmation_speedup'],y+.15,s=68,marker='D',color=GRAY,zorder=4,
                   label='旧机 S2：3 轮几何均值' if index==0 else None)
        interval = t.get('interval95')
        if interval and len(interval)==2 and all(finite_positive(x) for x in interval):
            ax.plot(interval,[y-.15,y-.15],color=BLUE,lw=2,zorder=3)
            ax.scatter(interval,[y-.15,y-.15],marker='|',color=BLUE,s=55,zorder=4)
        ax.scatter(t['speedup'],y-.15,s=65,color=BLUE,zorder=4,
                   label='新机：10 轮几何均值及描述性 95% 区间' if index==0 else None)
        ax.text(1.02,y+.15,f"{previous['confirmation_speedup']:.3f}×",transform=ax.get_yaxis_transform(),
                ha='left',va='center',fontsize=9,color=GRAY)
        ax.text(1.02,y-.15,f"{t['speedup']:.3f}×  {'通过' if t['clear_gain'] else '未通过'}",
                transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=9,color=BLUE)
    ax.set_yticks(range(len(tests)),[x['workload_id'].replace('_','  ') for x in reversed(tests)])
    ax.set_xscale('log')
    xmin,xmax = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([v for v in (.25,.5,.75,1,1.5,2,3,4,6,8) if xmin <= v <= xmax]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f'{x:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('默认时间 / 冻结候选时间（越大表示候选越快，倍数）')
    ax.grid(axis='x',color='#dce2e7',lw=.7)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.18),ncol=1,frameon=False,fontsize=9)
    fig.suptitle('H1｜同一冻结候选的独立复测',x=.03,y=.98,ha='left',fontsize=16,fontweight='bold')
    fig.text(.03,.89,'旧机 S2 与新机不合并；候选来自旧数据，非对旧机结果的盲选复现',fontsize=9,color='#505b65')
    fig.text(.03,.025,'虚线 1.05× 为点估计门槛；“通过”还要求新机单侧检验经五项 Holm 校正且质量合格。',fontsize=8.5,color='#505b65')
    save(fig,out,'h1_independent_cohorts',synthetic)
    return True


def h2(report, old_s1, out, synthetic):
    specs = {x['workload_id']:x for x in report.get('directional_tests',[]) if x.get('hypothesis')=='H2'}
    low,high = 'n16384_b1','n16384_b2048'
    candidate = 'a128_r4_l1_np1'
    matches = [x for x in old_s1.get('crossovers',[]) if x.get('n')==16384 and x.get('reference')=='default' and x.get('target')==candidate and x.get('single_batch')==1 and x.get('large_batch')==2048]
    if len(matches)!=1 or set(specs)!={low,high} or not all(specs[w].get('quality_valid') and finite_positive(specs[w].get('speedup')) for w in (low,high)):
        return False
    a,b = specs[low],specs[high]
    if (a['reference'],a['target']) != (candidate,'default') or (b['reference'],b['target']) != ('default',candidate):
        raise ValueError('frozen_H2_directions_changed')
    # Plot one orientation on both endpoints; the low-batch test is its reciprocal.
    new_values = [1/a['speedup'],b['speedup']]
    intervals = []
    for t,invert in ((a,True),(b,False)):
        interval = t.get('interval95')
        intervals.append(([1/interval[1],1/interval[0]] if invert else interval) if interval and all(finite_positive(x) for x in interval) else None)
    fig, axes = plt.subplots(1,2,figsize=(9.8,4.7),sharey=True)
    fig.subplots_adjust(left=.12,right=.96,top=.80,bottom=.22,wspace=.14)
    old = matches[0]
    for ax, values, label, color in ((axes[0],[old['single_ratio'],old['large_ratio']],'旧机 S1：单轮筛选',GRAY),
                                     (axes[1],new_values,'新机：10 轮独立复测',BLUE)):
        ax.axhline(1,color='#9aa6af',lw=1)
        for threshold in (1.05,1/1.05): ax.axhline(threshold,color='#c1c9cf',lw=.8,ls='--')
        ax.plot([0,1],values,color=color,lw=1.5,alpha=.75)
        ax.scatter([0,1],values,color=color,s=70,zorder=3)
        if ax is axes[1]:
            for i,interval in enumerate(intervals):
                if interval:
                    ax.plot([i,i],interval,color=color,lw=2)
                    ax.scatter([i,i],interval,color=color,marker='_',s=80)
        for i,v in enumerate(values): ax.annotate(f'{v:.3f}×',(i,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,color=color)
        ax.set_xticks([0,1],['batch 1','batch 2048'])
        ax.set_title(label,fontsize=11)
        ax.grid(axis='y',color='#e1e6ea',lw=.6)
    axes[0].set_yscale('log')
    ymin,ymax = axes[0].get_ylim()
    axes[0].yaxis.set_major_locator(FixedLocator([v for v in (.25,.5,.75,1,1.5,2,3,4,6,8) if ymin <= v <= ymax]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'{x:g}'))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel('默认时间 / 同一候选时间（倍数）')
    fig.suptitle('H2｜同一候选在两个 batch 的相对位置',x=.03,y=.98,ha='left',fontsize=16,fontweight='bold')
    fig.text(.03,.89,'候选 a128_r4_l1_np1；左侧是旧机 S1 筛选，右侧为新机独立复测',fontsize=9,color='#505b65')
    fig.text(.03,.045,'区间为描述性 95% 配对对数区间；虚线为 1.05× 与其倒数。两方向均通过同一五项 Holm 才能确认反转。',fontsize=8.5,color='#505b65')
    fig.text(.03,.01,'旧机该候选未在 S2 两端完整复测，不能把旧机单轮点标为确认结果。',fontsize=8.5,color='#505b65')
    save(fig,out,'h2_same_candidate_endpoints',synthetic)
    return True


def h3_cost(report, out, synthetic):
    cost = report.get('cost')
    if not cost or 'cost' in report.get('stage_errors',{}):
        return False
    eligible = [p for p in cost.get('pairs',[]) if p.get('classification')=='equivalent' and
                finite_positive(p.get('full_wall_ms')) and finite_positive(p.get('describe_known_choice_wall_ms'))]
    summaries = {x['mode']:x for x in cost.get('stage_summary',[])}
    modes = ('full-search-cost','describe-select-cost')
    if not eligible or not all(mode in summaries and summaries[mode].get('processes')==len(eligible) for mode in modes):
        return False
    workload_ids = sorted({p['workload_id'] for p in eligible}, key=lambda x: tuple(int(v[1:]) for v in x.split('_')))
    fig, (left,right) = plt.subplots(1,2,figsize=(12.6,6.6),gridspec_kw={'width_ratios':[1.45,1]})
    fig.subplots_adjust(left=.15,right=.91,top=.82,bottom=.31,wspace=.38)
    offsets = [-.15,0,.15]
    for i,wid in enumerate(workload_ids):
        for j,p in enumerate(sorted((x for x in eligible if x['workload_id']==wid),key=lambda x:x['round'])):
            y = len(workload_ids)-1-i+offsets[j % len(offsets)]
            left.plot([p['describe_known_choice_wall_ms'],p['full_wall_ms']],[y,y],color='#c4ced5',lw=1,zorder=1)
            left.scatter(p['full_wall_ms'],y,s=24,color=ORANGE,zorder=2)
            left.scatter(p['describe_known_choice_wall_ms'],y,s=24,color=BLUE,zorder=3)
    left.scatter([],[],color=ORANGE,label='26 个完整构建')
    left.scatter([],[],color=BLUE,label='26 次描述 + 已知选择的构建')
    left.set_yticks(range(len(workload_ids)),[w.replace('_','  ') for w in reversed(workload_ids)])
    left.set_xscale('log')
    left.set_xlabel('同上下文管线墙钟时间（ms，对数轴）')
    left.set_title('每条线连接一个 workload / 轮次',loc='left',fontsize=10)
    left.grid(axis='x',color='#e1e6ea',lw=.6)
    left.legend(loc='upper left',bbox_to_anchor=(0,-.13),frameon=False,fontsize=9)
    ys = [1,0]
    labels = ['26 个完整构建','26 次描述 + 已知选择']
    for y,mode in zip(ys,modes):
        summary = summaries[mode]
        n = summary['processes']
        wall = summary['pipeline_wall_ms']/n
        traced = summary['trace_exclusive_ms']/n
        compilation = summary['compilation_loading_exclusive_ms']/n
        other = traced-compilation
        if not all(math.isfinite(x) for x in (wall,traced,compilation,other)) or min(wall,traced,compilation,other) < -1e-6:
            plt.close(fig)
            return False
        residual = wall-traced
        # Residual is retained as a signed difference; it is not recast as a stage.
        right.barh(y,wall,height=.32,color='#ecf0f3',edgecolor='#9ba7b0',label='整个管线墙钟' if y==1 else None)
        right.barh(y,compilation,height=.18,color=ORANGE,label='编译 / 加载排他时间' if y==1 else None)
        right.barh(y,other,left=compilation,height=.18,color=BLUE,label='其余已跟踪排他时间' if y==1 else None)
        right.text(1.02,y,f'墙钟 {wall:.1f} ms\n墙钟−跟踪 {residual:+.1f} ms',transform=right.get_yaxis_transform(),
                   va='center',fontsize=8.5)
    right.set_yticks(ys,labels)
    right.set_xlabel('每条管线平均时间（ms）')
    right.set_title('排他跨度与墙钟分别计量',loc='left',fontsize=10)
    right.grid(axis='x',color='#e1e6ea',lw=.6)
    right.legend(loc='upper left',bbox_to_anchor=(0,-.13),frameon=False,fontsize=9)
    fig.suptitle('H3｜新机上的实际构建管线',x=.03,y=.98,ha='left',fontsize=16,fontweight='bold')
    fig.text(.03,.89,f'仅绘协议通过且描述与完整构建等价的配对：{len(eligible)} / {len(cost.get("pairs",[]))}；每点是独立进程。',fontsize=9,color='#505b65')
    fig.text(.03,.045,'已知选择由旧筛选提供；这比较两种候选枚举构建组件，并非当前默认规划器对在线学习器的端到端比较。',fontsize=8.5,color='#505b65')
    fig.text(.03,.015,'阴影条为同上下文管线墙钟（不含外部进程启动和退出），叠加条为排他跨度；残差可能带符号，不称作独立阶段。',fontsize=8.5,color='#505b65')
    save(fig,out,'h3_paired_pipeline_costs',synthetic)
    return True


def h3_amortization(report, out, synthetic):
    cost = report.get('cost')
    if not cost or 'cost' in report.get('stage_errors',{}): return False
    entries = cost.get('amortization',[])
    names = ('single_candidate_initialize_component','describe_policy_initialize_component')
    valid = [x for x in entries if len(x.get('components',[]))==2 and {c.get('id') for c in x['components']}==set(names)]
    if not valid or not any(c.get('break_even_executions') is not None for x in valid for c in x['components']):
        return False
    valid.sort(key=lambda x:tuple(int(v[1:]) for v in x['workload_id'].split('_')))
    fig,ax = plt.subplots(figsize=(10.6,6.2))
    fig.subplots_adjust(left=.23,right=.95,top=.81,bottom=.31)
    offsets = (.14,-.14)
    for index,row in enumerate(valid):
        y = len(valid)-1-index
        by = {c['id']:c for c in row['components']}
        for j,name in enumerate(names):
            item = by[name]
            value = item.get('break_even_executions')
            color = (ORANGE,BLUE)[j]
            if isinstance(value,int) and value >= 0:
                ax.scatter(value,y+offsets[j],s=46,color=color,zorder=3)
                ax.annotate(str(value),(value,y+offsets[j]),xytext=(6,0),textcoords='offset points',va='center',fontsize=8,color=color)
            else:
                ax.text(.01,y+offsets[j],'无有限值 / 无法估计',transform=ax.get_yaxis_transform(),fontsize=8,color=color,va='center')
    ax.scatter([],[],color=ORANGE,label='固定候选 − 默认：初始化差额')
    ax.scatter([],[],color=BLUE,label='26 描述 + 已知选择 − 默认：初始化差额')
    ax.set_yticks(range(len(valid)),[f"{x['workload_id'].replace('_','  ')}  ({x['execution_rounds']} 轮)" for x in reversed(valid)])
    ax.set_xscale('symlog',linthresh=1)
    finite_counts = [c['break_even_executions'] for row in valid for c in row['components']
                     if isinstance(c.get('break_even_executions'),int) and c['break_even_executions']>=0]
    ax.set_xlim(0,max(2,max(finite_counts)*2))
    ax.set_xlabel('名义回本执行次数（对数尺度；0 表示无额外初始化成本）')
    ax.grid(axis='x',color='#e1e6ea',lw=.6)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.16),frameon=False,fontsize=9)
    fig.suptitle('H3｜同一新机的数据：初始化组件摊销',x=.03,y=.98,ha='left',fontsize=16,fontweight='bold')
    fig.text(.03,.89,'仅在候选执行差额为正且组件完整时计算；10 轮或 3 轮按预定 workload 分别使用。',fontsize=9,color='#505b65')
    fig.text(.03,.045,'数值是名义组件诊断，未计搜索基准、打分、回退和其他开销，也不代表线上收益。',fontsize=8.5,color='#505b65')
    fig.text(.03,.015,'无有限值包含执行差额非正和数据不足；两者均不能解释为回本次数为零。',fontsize=8.5,color='#505b65')
    save(fig,out,'h3_initialization_amortization',synthetic)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True,type=Path,help='audited cloud_report.json')
    p.add_argument('--old-report-dir',required=True,type=Path,help='old independent audit report directory')
    p.add_argument('--output-dir',required=True,type=Path,help='new directory for figures')
    p.add_argument('--synthetic-layout-test',action='store_true',help='watermark every figure; never use for scientific results')
    args = p.parse_args()
    report = read(args.report)
    if report.get('schema_version')!='cloud-reproduction-report/v1' or report.get('study_id')!='plan_selection_cloud_reproduction_20261028_r1':
        raise ValueError('cloud_report_identity_mismatch')
    old_s1, old_s2 = checked_old(args.old_report_dir)
    configure_font()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    names = {}
    execution_ok = not any(x in report.get('stage_errors',{}) for x in ('equivalence','semantics','calibration','health','execution'))
    names['h1'] = bool(execution_ok and h1(report,old_s2,args.output_dir,args.synthetic_layout_test))
    names['h2'] = bool(execution_ok and h2(report,old_s1,args.output_dir,args.synthetic_layout_test))
    names['h3_pipeline'] = bool(execution_ok and h3_cost(report,args.output_dir,args.synthetic_layout_test))
    names['h3_amortization'] = bool(execution_ok and h3_amortization(report,args.output_dir,args.synthetic_layout_test))
    print(json.dumps(dict(figures=names,skipped=[k for k,v in names.items() if not v],synthetic=args.synthetic_layout_test),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
