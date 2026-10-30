#!/usr/bin/env python3
"""Render a Chinese research report from separately audited cohort summaries."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, places=3):
    return f'{value:.{places}f}' if isinstance(value,(int,float)) and math.isfinite(value) else '未获得'


def interval(value):
    return f'[{number(value[0])}, {number(value[1])}]' if isinstance(value,list) and len(value)==2 else '不可推断'


def fmt_p(value):
    return f'{value:.3g}' if isinstance(value,(int,float)) and math.isfinite(value) else '未获得'


def render(cloud, old_review, old_confirmation, paths, figure_dir, report_dir=None):
    if cloud.get('schema_version')!='cloud-reproduction-report/v1':
        raise ValueError('unexpected_cloud_report_schema')
    if old_review.get('study_id')!='plan_selection_20261027_r2' or old_review.get('audit',{}).get('result')!='passed':
        raise ValueError('old_cohort_review_not_verified')
    tests = cloud['directional_tests']
    if len(tests)!=5 or len({t['test_id'] for t in tests})!=5:
        raise ValueError('expected_fixed_five_test_family')
    stage_names = {'equivalence':'原头文件与插桩等价','semantics':'四类输入语义','calibration':'新机器 R 校准',
                   'health':'健康负对照','execution':'独立执行','cost':'构建成本'}
    recovered = bool(cloud.get('recovery_id'))
    if recovered:
        stage_names = {'health':'恢复前新健康负对照','execution':'恢复执行（新记录）','cost':'构建成本'}
    lines = ['# VkFFT 计划选择：独立云机复测', '',
        '本研究考察在编译之前观察候选执行结构，是否有机会为具体工作负载选择比默认计划更合适的结构，并厘清枚举描述本身的成本。旧机器完成了完整筛选和独立确认；新机器使用旧筛选冻结的候选身份，重新校准并独立测量。新队列是根据旧数据选择的定向复测，不能视为对原始全部工作负载的无偏重抽样。', '',
        '## 数据状态与边界', '',
        f"新队列 ID：`{cloud['study_id']}`；设备 UUID：`{cloud['gpu_uuid']}`。以下阶段只使用此队列自己的进程数据、种子、校准、守护日志和成本；旧机器数据在下文单独列示，从不与新机器配对、汇总或共同计算 p 值。", '',
        '| 阶段 | 原始行数 | 通过阶段审计记录数 | 守护隔离次数 | 状态 |', '|---|---:|---:|---:|---|']
    for stage, label in stage_names.items():
        errors = cloud['stage_errors'].get(stage,[])
        status = '完成并通过审计' if not errors else '未完成或不可用：'+'；'.join(errors)
        raw = cloud.get('stage_raw_line_counts',cloud['stage_record_counts']).get(stage,0)
        accepted = cloud.get('stage_accepted_counts',{}).get(stage,cloud['stage_record_counts'].get(stage,0) if not errors else 0)
        quarantined = cloud.get('stage_quarantined_guard_run_counts',{}).get(stage,0)
        lines.append(f"| {label} | {raw} | {accepted} | {quarantined} | {status} |")
    if recovered:
        preflight = cloud.get('original_preflight_record_counts',{})
        lines += ['', f"E1 一次性恢复：原等价 / 语义 / 校准 / 健康前置记录数分别为 {preflight.get('equivalence',0)} / {preflight.get('semantics',0)} / {preflight.get('calibration',0)} / {preflight.get('health',0)}；原执行状态仍为 stopped。事故前 {cloud.get('old_preserved_count',0)} 条记录按闭合区间审计获准，恢复会话新增 {cloud['stage_record_counts'].get('execution',0)} 条；显式选集包含 {cloud.get('selected_record_count',0)} 条、{cloud.get('selected_block_count',0)} 个完整块。原受损块的两条孤儿记录与隔离进程均未进入分析。恢复健康检查不能证明事故期间的不可观测窗口干净。两个守护会话处于不同时间，时间与系统状态变化仍是结果的限制；每个配对块均在单一会话内完整测量。"]
    lines += ['', '旧研究 S1 为 36 个工作负载 × 26 个请求；S2 对冻结选择进行了三轮独立确认。旧 S3 中断，旧 S4 与独立 H2 后续复测未执行；新队列不替代或重标这些旧阶段。旧 S2 的 36 个固定工作负载中，冻结选择的等权几何比为 '
              +number(old_review['H1']['fixed_S1_winner_S2_gmean_speedup'])+'，中位比为 '+number(old_review['H1']['fixed_S1_winner_S2_median_speedup'])+
              '；它们描述固定矩阵，不能推断一般 FFT 工作负载的收益比例。', '',
              '## H1：固定候选的独立执行', '',
              '比值为默认执行时间 / 指定候选执行时间，大于 1 表示候选更快。新机器以独立进程中位数为配对单位，三个 H1 方向和两个 H2 方向共同构成预先固定的五项 Holm 家族；检验零假设为平均对数收益不超过 2%，明确收益同时要求点估计至少 5% 和校正后 p < 0.05。旧 S2 只有三轮，以下列出其描述性范围，不将其当作相同的推断证据。', '',
              '| 工作负载 | 固定候选 | 旧机 S2：几何比（3 轮范围） | 新机：几何比及 95% 区间 | 新机 Holm p | 判定 |', '|---|---|---:|---:|---:|---|']
    for test in tests:
        if test['hypothesis']!='H1': continue
        old = old_confirmation.get(test['workload_id'],{})
        primary = old.get('primary') if old.get('winner')==test['target'] else None
        vals = primary.get('ratios',[]) if primary else []
        old_text = f"{number(primary.get('speedup'))}（{number(min(vals))}–{number(max(vals))}）" if primary and vals else '未匹配'
        new_text = number(test.get('speedup'))+' '+interval(test.get('interval95'))
        result = '明确收益' if test.get('clear_gain') else ('未达明确收益' if test.get('inferential') else '不可推断')
        lines.append(f"| `{test['workload_id']}` | `{test['target']}` | {old_text} | {new_text} | {fmt_p(test.get('holm_p'))} | {result} |")
    lines += ['', '点估计、区间和家族判定均须结合负对照及完整阶段审计；零方差或缺失方向保留为不可推断，家族计算使用 p=1 占位，而不是报告人为确定的显著性。`n20480_b16` 的旧筛选赢家是默认值，其选择比按定义为 1，仅用于观察和负对照。', '',
              '## H2：同一候选在不同批量下', '',
              '固定比较 `a128_r4_l1_np1` 与 `default`，两个端点均为 N=16384。小批量的预定方向为候选 / 默认，大批量为默认 / 候选，均是“参考时间 / 目标时间”；只有两个相反方向都在同一五项 Holm 家族达到明确收益，才可声称该端点对的方向反转。原旧机 S2 在这两个端点没有完整的同候选确认覆盖，因此不从旧机补造该配对。', '',
              '| 工作负载 | 参考 → 目标 | 新机比值 | 95% 区间 | Holm p | 判定 |', '|---|---|---:|---:|---:|---|']
    for t in tests:
        if t['hypothesis']!='H2': continue
        result = '明确收益' if t.get('clear_gain') else ('未达明确收益' if t.get('inferential') else '不可推断')
        lines.append(f"| `{t['workload_id']}` | `{t['reference']}` → `{t['target']}` | {number(t.get('speedup'))} | {interval(t.get('interval95'))} | {fmt_p(t.get('holm_p'))} | {result} |")
    lines += ['', '方向反转判定：'+('两个方向均满足预定标准。' if cloud.get('crossover_confirmed') else '尚不能按预定标准确认；部分或全部方向可能未达标准或数据缺失。'), '',
              '请求字段并不等同于实际结构。下表展示新机器执行记录中各 H2 端点的代表性结构；多轮出现不同指纹时保留不同指纹数。线程形状、radix、共享内存、代码与工作集可能同时改变，未采集硬件计数器，不能将差异归因于单个字段。', '',
              '| 工作负载 | 候选 | numAxisUploads（FFT 分解段数） | axisSplit | 不同核心 / 源类数 |', '|---|---|---:|---|---:|']
    for s in cloud.get('structures',[]):
        if s['workload_id'] not in ('n16384_b1','n16384_b2048') or s['candidate_id'] not in ('default','a128_r4_l1_np1'): continue
        core = s['representative_core_descriptor']
        lines.append(f"| `{s['workload_id']}` | `{s['candidate_id']}` | {core.get('numAxisUploads','未获得')} | `{core.get('axisSplit')}` | {len(s['core_fingerprints'])} / {len(s['source_compiler_fingerprints'])} |")
    lines += ['', '## H3：描述、完整构建与成本摊销', '',
              '当前 VkFFT 默认路径构建一个计划。这里实际比较的是同一上下文内完整构建 26 个请求，与描述 26 个请求后构建一个已由旧筛选指定的候选；这是已知选择下的构建组成比较，不是当前默认规划器与在线学习器的端到端比较，也不包含在线评分、验证、回退或搜索基准成本。一次所选计划可能含多个 upload，不能假定只有一次 NVRTC 编译。', '']
    cost = cloud.get('cost')
    if cost is None:
        lines.append('成本阶段尚未形成可审计的完整 24 对记录；不报告成本、描述可行性或摊销结论。')
    else:
        counts = {'equivalent':sum(p['classification']=='equivalent' for p in cost['pairs']), 'negative':sum(p['classification']=='negative' for p in cost['pairs'])}
        lines += [f"完整固定成本对：{len(cost['pairs'])}；描述与完整构建等价 {counts['equivalent']} 对，科学性不一致 {counts['negative']} 对。描述零编译与源/描述符等价必须同时成立；不一致保留为限制，不能改称成功。", '',
                  '| 模式 | 等价对中的管线数 | 独占 trace 总量 ms | 编译/加载占 trace | 管线 wall 总量 ms |', '|---|---:|---:|---:|---:|']
        for s in cost['stage_summary']:
            frac = s.get('compilation_fraction_of_traced')
            lines.append(f"| `{s['mode']}` | {s['processes']} | {number(s['trace_exclusive_ms'],1)} | {number(frac*100,1)+'%' if frac is not None else '未获得'} | {number(s['pipeline_wall_ms'],1)} |")
        lines += ['', '独占阶段是可加的同步 trace 分区；不同层级的 inclusive 时间不能相加。管线 wall 还包括上下文和未归属部分。以下摊销仅为初始化组成的名义估计，分母只使用同一新机器该工作负载预定全部 3 或 10 轮的默认减赢家执行时间；若分母非正或任一组成缺失，不给出有限回本次数。', '',
                  '| 工作负载 / 赢家 | 独立执行轮数 | 单候选初始化组成回本次数 | 描述策略初始化组成回本次数 |', '|---|---:|---:|---:|']
        for a in cost['amortization']:
            cs = {x['id']:x for x in a['components']}
            def break_text(c):
                return str(c['break_even_executions']) if c.get('break_even_executions') is not None else '无有限估计'
            lines.append(f"| `{a['workload_id']}` / `{a['winner']}` | {a['execution_rounds']} | {break_text(cs['single_candidate_initialize_component'])} | {break_text(cs['describe_policy_initialize_component'])} |")
        lines += ['', '单候选组成是完整管线中所选候选与默认初始化时间之差；描述策略组成是 26 次描述加所选物化的初始化时间之和，减去新执行阶段单次默认初始化的均值。后者跨阶段、非配对且不包含外层上下文、清理、评分、验证与回退，不能写成端到端默认规划加速。']
    lines += ['', '## 对编译前选择的含义', '',
              '旧筛选中 26 个名义请求往往折叠到较少的实际核心与源类；旧 S2 的大幅收益集中在少数工作负载，而许多选择接近默认。若新队列完整通过，H1/H2 可进一步约束这些固定候选在新环境中的表现，H3 可检验真实描述路径及其成本。结构去重和简单规则都是必要的比较基线：性能差距和更便宜的描述并不能证明复杂学习模型是必要的，也不能证明模型已经具备合法性判断或在线端到端收益。本次没有新 H4 训练。', '',
              '## 图与来源', '']
    figure_names = [('h1_independent_cohorts.png','H1：旧、新队列独立结果'),('h2_same_candidate_endpoints.png','H2：同一候选的两个批量端点'),
                    ('h3_paired_pipeline_costs.png','H3：固定成本对与组成'),('h3_initialization_amortization.png','H3：同机初始化组成摊销')]
    for filename,label in figure_names:
        target = figure_dir/filename
        if target.exists():
            link = Path(os.path.relpath(target,report_dir or Path.cwd())).as_posix()
            lines.append(f'- [{label}]({link})')
    if report_dir is not None:
        old_h4 = report_dir.parent/'old_screen_h4_20261028_r1'/'report'/'旧S1_H4研究报告.md'
        if old_h4.is_file():
            link = Path(os.path.relpath(old_h4,report_dir)).as_posix()
            lines.append(f'- [旧 S1 独立 H4 条件排序报告]({link})')
    lines += ['', f"- 新队列机器可读审计：`{paths['cloud']}`，SHA-256 `{digest(paths['cloud'])}`。",
              f"- 旧队列独立 S2 审阅：`{paths['old_review']}`，SHA-256 `{digest(paths['old_review'])}`。",
              f"- 旧队列冻结赢家确认摘要：`{paths['old_confirmation']}`，SHA-256 `{digest(paths['old_confirmation'])}`。",
              f"- 新设计 SHA-256：`{cloud['design_sha256']}`；运行清单 SHA-256：`{cloud['manifest_sha256']}`。", '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud',type=Path,required=True)
    p.add_argument('--old-review',type=Path,required=True)
    p.add_argument('--old-confirmation',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    paths = dict(cloud=args.cloud,old_review=args.old_review,old_confirmation=args.old_confirmation)
    text = render(load(args.cloud),load(args.old_review),load(args.old_confirmation),paths,args.figures,args.output.parent)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text,encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
