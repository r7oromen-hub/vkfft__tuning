#!/usr/bin/env python3
"""Render the completed old-S1 H4 result; this script never fits a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).parent
ANALYSIS_ID = 'plan_selection_old_screen_h4_analysis_20261028_r1'
SOURCE_ID = 'plan_selection_20261027_r2'
ADMISSION_HASH = '65b8bae1b7b21fd42af172c2b89c1a87510171cf08ba3f57be02d84a8438e921'
TARGETS = ('log_candidate_over_default','log_absolute_latency')
ABLATIONS = ('workload_knobs','workload_structure','workload_knobs_structure')
BASELINES = ('default','training_only_global','frozen_S1_global_diagnostic','screen_oracle_diagnostic')
FEATURE_LABELS = {'workload_knobs':'工作负载 + 请求参数',
                  'workload_structure':'工作负载 + 编译前结构',
                  'workload_knobs_structure':'工作负载 + 请求参数 + 编译前结构'}
TARGET_LABELS = {'log_candidate_over_default':'主要：相对默认延迟',
                 'log_absolute_latency':'次要：绝对延迟'}
BASELINE_LABELS = {'default':'默认计划（主基线）','training_only_global':'训练集全局计划（主基线）',
                   'frozen_S1_global_diagnostic':'全 S1 冻结全局计划（诊断）',
                   'screen_oracle_diagnostic':'S1 事后最优候选（诊断）'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, digits=3):
    return f'{value:.{digits}f}' if isinstance(value,(int,float)) and math.isfinite(value) else '不可得'


def fraction(value):
    return f'{100*value:.1f}%' if isinstance(value,(int,float)) and math.isfinite(value) else '不可得'


def verify(root, admission_path, synthetic=False):
    root = Path(root)
    result_path, provenance_path, marker_path = (root/'learning_check.json',root/'analysis_provenance.json',root/'analysis_started.json')
    if not all(p.is_file() for p in (result_path,provenance_path,marker_path,Path(admission_path))):
        raise ValueError('completed_result_provenance_or_reservation_missing')
    result, provenance, marker, admission = read(result_path),read(provenance_path),read(marker_path),read(admission_path)
    if (result.get('status')!='completed_auxiliary' or result.get('study_id')!=SOURCE_ID or
        provenance.get('analysis_id')!=ANALYSIS_ID or marker.get('analysis_id')!=ANALYSIS_ID or
        provenance.get('admission_sha256')!=digest(admission_path) or marker.get('admission_sha256')!=digest(admission_path)):
        raise ValueError('result_or_provenance_identity_mismatch')
    if not synthetic and digest(admission_path)!=ADMISSION_HASH:
        raise ValueError('admission_hash_mismatch')
    if provenance.get('terminal_disposition',{}).get('sequence_ended') is not True:
        raise ValueError('primary_sequence_not_terminated')
    if not synthetic:
        # This validates frozen fold and candidate coverage; it does not call run_learning.
        from old_screen_learning import validate_result
        validate_result(result,admission)
    outcomes = result.get('outcomes',[])
    if len(outcomes)!=6 or {(x['target'],x['ablation']) for x in outcomes}!={(t,a) for t in TARGETS for a in ABLATIONS}:
        raise ValueError('fixed_six_outcomes_missing')
    if set(result.get('baselines',{}))!=set(BASELINES):
        raise ValueError('fixed_four_baselines_missing')
    if result.get('heldout_validity_used_for_filtering') is not True or result.get('deployment_end_to_end_evaluated') is not False:
        raise ValueError('model_scope_metadata_mismatch')
    # Recalculate aggregates from the retained per-workload observations.
    from learning_check import aggregate
    for group in outcomes+[result['baselines'][key] for key in BASELINES]:
        rows, recorded = group.get('workloads',[]),group.get('metrics',{})
        if len(rows)!=36:
            raise ValueError('workload_coverage_incomplete')
        recomputed = aggregate(rows)
        if set(recorded)!=set(recomputed):
            raise ValueError('aggregate_fields_mismatch')
        for key,value in recomputed.items():
            actual = recorded[key]
            if isinstance(value,(int,float)):
                if not isinstance(actual,(int,float)) or not math.isfinite(actual) or not math.isclose(value,actual,rel_tol=1e-10,abs_tol=1e-12):
                    raise ValueError('aggregate_value_mismatch:'+key)
            elif actual!=value:
                raise ValueError('aggregate_value_mismatch:'+key)
    return result,provenance,admission


def configure_font():
    for path in (Path('C:/Windows/Fonts/msyh.ttc'),Path('C:/Windows/Fonts/simhei.ttf'),
                 Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})


def figure(result, destination, synthetic=False):
    configure_font()
    outputs = {(o['target'],o['ablation']):o for o in result['outcomes']}
    fig,axes = plt.subplots(1,2,figsize=(11.6,5.6),sharey=True)
    fig.subplots_adjust(left=.28,right=.93,top=.78,bottom=.30,wspace=.20)
    colors = {'log_candidate_over_default':'#25639a','log_absolute_latency':'#b46a2d'}
    for ax,target in zip(axes,TARGETS):
        for idx,ablation in enumerate(ABLATIONS):
            metric = outputs[target,ablation]['metrics']
            value = metric.get('equal_workload_geometric_speedup')
            if isinstance(value,(int,float)) and math.isfinite(value) and value>0:
                ax.scatter([value],[2-idx],s=75,color=colors[target],zorder=3)
                ax.annotate(f"{value:.3f}×  ({metric['workloads']}/36)",(value,2-idx),xytext=(8,0),
                            textcoords='offset points',va='center',fontsize=9,color=colors[target])
        ax.axvline(1,color='#7d8892',lw=1,ls='--',label='默认计划')
        baseline = result['baselines']['training_only_global']['metrics']
        b = baseline.get('equal_workload_geometric_speedup')
        if isinstance(b,(int,float)) and math.isfinite(b) and b>0:
            ax.axvline(b,color='#5a8a76',lw=1.4,ls=':',label=f"训练集全局计划 ({baseline['workloads']}/36)")
        ax.set_title(TARGET_LABELS[target],fontsize=11)
        ax.set_xlabel('默认时间 / 所选计划时间（等工作负载几何比）')
        ax.grid(axis='x',color='#e0e5e9',lw=.7)
        ax.margins(x=.26)
    axes[0].set_yticks([2,1,0],[FEATURE_LABELS[a] for a in ABLATIONS])
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.20),frameon=False,fontsize=9)
    fig.suptitle('旧 S1｜固定模型的分组留出评估',x=.03,y=.98,ha='left',fontsize=16,fontweight='bold')
    fig.text(.03,.89,'同一旧 GPU；四折按 exact N 留出；每个策略的可用工作负载数见括号。',fontsize=9,color='#505b65')
    fig.text(.03,.065,'留出候选集已用事后执行、数值与语义结果过滤；图中不是在线合法性预测或端到端收益。',fontsize=8.5,color='#505b65')
    fig.text(.03,.025,'训练集全局计划若有不可用工作负载，其均值与完整覆盖策略不可直接视为同一集合比较。',fontsize=8.5,color='#505b65')
    if synthetic:
        fig.text(.5,.5,'合成数据 · 布局测试',ha='center',va='center',fontsize=30,color='#bd4a48',alpha=.20,rotation=25)
    for ext in ('png','svg'):
        fig.savefig(destination/('old_screen_h4_features.'+ext),dpi=190,bbox_inches='tight')
    plt.close(fig)


def narrative(result,provenance,root,admission_path,synthetic=False):
    lines = ['# 旧 S1：编译前结构特征的条件排序评估', '']
    if synthetic:
        lines += ['**合成数据：仅用于排版测试，不是研究结果。**','']
    lines += [
        '本辅助分析使用旧机器已完成的 S1 筛选数据，按固定规则比较工作负载、请求参数和编译前结构特征。它属于旧队列的离线分析，不包含云机标签，也不代表云机开展了新的 H4 训练。', '',
        f"源研究：`{result['study_id']}`；分析 ID：`{provenance['analysis_id']}`。输入为 36 个完整工作负载、16 个 exact N 组和 {result['training_rows']} 条离线合法候选记录。原始筛选的每个候选只有一次 S1 计时，因此标签包含筛选噪声；数据来自单个旧 GPU 环境。四折按 exact N 分组，不能解释为跨设备或跨质因子家族泛化。", '',
        '留出工作负载上的候选集合也通过事后的执行、数值和语义检查过滤。有效性不作为模型输入，但评估本身以已知合法集合为条件；因此它没有检验编译前合法性预测、在线失败发现或安全回退。', '',
        '## 六个固定评估结果', '',
        '主要目标预测候选相对默认的对数延迟，绝对延迟目标是次要敏感性分析。下列速度比为默认时间 / 所选时间，大于 1 表示所选计划较快；regret 为所选时间 / 该 S1 候选集中最快时间，越接近 1 越好。每个汇总按工作负载等权，不将候选行或图重复样本当成独立泛化单位。', '',
        '| 目标 | 输入消融 | 可用 / 36 | 不可用 | 等权几何速度比 | 几何 regret | 距 S1 最快 ≤2% | 距 S1 最快 ≤5% | 最差回退 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    outcome_map = {(o['target'],o['ablation']):o for o in result['outcomes']}
    for target in TARGETS:
        for ablation in ABLATIONS:
            m = outcome_map[target,ablation]['metrics']
            lines.append(f"| {TARGET_LABELS[target]} | {FEATURE_LABELS[ablation]} | {m['workloads']} | {m['unavailable']} | {number(m.get('equal_workload_geometric_speedup'))} | {number(m.get('geometric_regret'))} | {fraction(m.get('within2pct_fraction'))} | {fraction(m.get('within5pct_fraction'))} | {fraction(m.get('worst_regression'))} |")
    lines += ['', '## 固定基线与覆盖', '',
              '默认计划和仅用训练折选择的全局计划是主要比较基线。全 S1 冻结全局计划与筛选事后最优使用了 S1 标签，只作为乐观诊断；它们不是无泄漏的部署基线。不可用的留出候选不插补为零，覆盖不同的汇总不能直接当成同一工作负载集合比较。', '',
              '| 基线 | 可用 / 36 | 不可用 | 等权几何速度比 | 几何 regret | 距 S1 最快 ≤5% |',
              '|---|---:|---:|---:|---:|---:|']
    for key in BASELINES:
        m = result['baselines'][key]['metrics']
        lines.append(f"| {BASELINE_LABELS[key]} | {m['workloads']} | {m['unavailable']} | {number(m.get('equal_workload_geometric_speedup'))} | {number(m.get('geometric_regret'))} | {fraction(m.get('within5pct_fraction'))} |")
    lines += ['', '## 候选短名单：假设性诊断', '',
              'Top-2 和 Top-4 数值使用每个留出工作负载的 S1 标签，从排序短名单中事后选最快候选。它们不包含实际在线构建、基准、验证或选择成本，不是已经实现的性能。', '',
              '| 目标 | 输入消融 | 可用 / 36 | Top-2 假设 oracle 几何 regret | Top-4 假设 oracle 几何 regret |', '|---|---|---:|---:|---:|']
    for target in TARGETS:
        for ablation in ABLATIONS:
            m = outcome_map[target,ablation]['metrics']
            lines.append(f"| {TARGET_LABELS[target]} | {FEATURE_LABELS[ablation]} | {m['workloads']} | {number(m.get('geometric_top2_shortlist_regret'))} | {number(m.get('geometric_top4_shortlist_regret'))} |")
    terminal = provenance.get('terminal_disposition',{})
    lines += ['', '## 解释与界限', '',
              '结构特征与请求参数的固定消融能够评价这组已合法候选中的条件排序；即使结果有利，也不能由此证明复杂机器学习是必要的。候选请求中有大量结构和源码别名，去重和简单规则是重要替代基线，但本次固定评估没有覆盖所有此类策略。', '',
              '结果未测量训练、推理、描述枚举、合法性验证或回退的端到端成本，也没有证明可以在线只编译一次并避免执行错误。新机器 H1–H3 与这份旧 S1 H4 保持独立，不混合标签、基线或指标。', '',
              f"新机器主序列终止状态：`{terminal.get('disposition','不可得')}`；原因：{terminal.get('reason','不可得')}。若该序列技术停止，缺失的主研究证据仍然缺失，旧 S1 H4 不能替代它。", '',
              '## 图与审计来源', '',
              '- [六组固定消融与主要基线](old_screen_h4_features.png)（[SVG](old_screen_h4_features.svg)）。',
              f"- 逐工作负载选择、折与所有指标：`{root/'learning_check.json'}`；SHA-256 `{digest(root/'learning_check.json')}`。",
              f"- 预拟合来源、运行版本和序列终止记录：`{root/'analysis_provenance.json'}`；SHA-256 `{digest(root/'analysis_provenance.json')}`。",
              f"- 固定准入：`{admission_path}`；SHA-256 `{digest(admission_path)}`。", '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=HERE/'results/old_screen_h4_20261028_r1')
    p.add_argument('--admission',type=Path,default=HERE/'old_screen_h4_admission.json')
    p.add_argument('--output-dir',type=Path,required=True,help='new directory; never overwrite an earlier report')
    p.add_argument('--synthetic-layout-test',action='store_true',help='watermark charts and report; not scientific results')
    args = p.parse_args()
    result,provenance,_ = verify(args.root,args.admission,args.synthetic_layout_test)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    figure(result,args.output_dir,args.synthetic_layout_test)
    (args.output_dir/'旧S1_H4研究报告.md').write_text(narrative(result,provenance,args.root,args.admission,args.synthetic_layout_test),encoding='utf-8')
    print(json.dumps(dict(output_dir=str(args.output_dir),synthetic=args.synthetic_layout_test),ensure_ascii=False))


if __name__=='__main__':
    main()
