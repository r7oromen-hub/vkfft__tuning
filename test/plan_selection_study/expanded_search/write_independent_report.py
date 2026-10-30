#!/usr/bin/env python3
"""Write independent report from already independently audited local evidence."""
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent
D=HERE/'results_20261030'


def load(name):return json.loads((D/name).read_text(encoding='utf8'))


def main():
    screen=load('search_local_distribution.json');audit=load('search_local_independent_audit.json')
    chain=load('confirmation_local_chain.json');budgets=load('budgets_local_replayed.json');combined=load('two_session_local_combined.json')
    plans=load('confirmed_local_structures.json');plan=load('confirmation_plan.json')
    if (audit['status']!='pass' or chain['status']!='pass' or combined['status']!='pass' or
            any(x['status']!='pass' for x in budgets.values())):raise ValueError('independent evidence chain has not passed')
    lines=['# 扩展搜索与双会话：独立离线审计报告','',
      '本地复核确认 3 个工作负载符合预注册条件：同一 GPU 上两个不同会话各完成 10 个随机配对区组，候选/原始默认几何平均耗时比均不大于 0.98，且每个会话的 95% 配对 log-t 区间上界均小于 1。三者均为整套计划的比较；由于 radix、寄存器包络、group 或 block 可能同时改变，不能将收益归因给某一个 radix。',
      '', '研究对象为 RTX 4090、FP32、1D、C2C 正向、out-of-place、单 upload、`registerBoost=1`；CUDA graph 中连续执行同一输入/输出地址的变换，预热后测量，不包括传输、规划和验证。该结果不是冷缓存、端到端或跨设备结论。',
      '', '## 独立证据与成本', '',
      '本地重跑 [search 审计](search_local_independent_audit.json)、[两会话合并结果](two_session_local_combined.json)、[完整门票链](confirmation_local_chain.json) 和 [预算重放](budgets_local_replayed.json)。搜索审计结果与远端审计逐字段相同；从原始配对重新生成的确认计划与冻结计划完全一致；两会话审计与远端逐字段一致。原始与实验二进制分别为 `3ca7ee798d8962b11e501923b86a70eba3d74d019d2e4e77f86aa9b466cae445`、`4f39b012db382a486beeb4420d6349ea0116027d02f332d526a5a12cc59d7e23`。守卫及确认快照无事故、无在途项目；三份账本相对于全局 8 小时预算均无超额。',
      '', '覆盖分母如下；全部“请求”是原型限制内的请求，不是所有 FFT 计划的数学空间。静态支持表示通过原型预检，尚不代表调度或设备支持。', '',
      '| 层次 | 数量 | 解释 |','|---|---:|---|',
      '| 原型枚举 | 1,194,648 | 限 2/4/8/16/32 radix、6 个统一 R、二次幂 group |',
      '| 静态拒绝 | 500,976 | 包含寄存器与 radix 等预检拒绝 |',
      '| 静态通过 | 693,672 | 包括身份控制、选中及未选请求 |',
      '| 默认身份控制 | 28 | 不计入每负载 64 个非默认请求 |',
      '| 预先选中非默认请求 | 1,792 | 28 个负载各 64 个 |',
      '| 选中后未选发 describe | 1,143 | 未测，不能称无效 |',
      '| 实际 describe | 649 | 包含拒绝与预算中断 |',
      '| 调度/资源拒绝 | 255 | 已消耗请求及运行成本，不补选 |',
      '| 完成多模式数值验证并完成筛选 | 393 | 各 5 模式 × 3 seed；3 轮原始默认配对 |',
      '| 已 describe 但未完成处置 | 1 | 搜索预算边界 partial，不计作完整筛选 |',
      '| 静态通过但未被选中 | 691,852 | 未测，不可推断合法性或收益 |','',
      '搜索快照距全局预算起点 12,899.626 秒；首、二确认快照分别距起点 13,870.872 和 14,598.022 秒。这些是累计墙钟，包含此前准入及阶段间工作，不是各阶段独立用时。预算重放逐一核对所有历史守卫执行 ID。',
      '', '## 两次独立会话', '',
      '以下比值均为候选耗时 / 原始默认耗时；每条区间独立按该会话 10 个配对 log 比值计算。区间未修正搜索筛选或多重比较，不应视为跨设备泛化概率。', '',
      '![两次确认会话的配对区间](local_figures/confirmed_pair_intervals.png)', '',
      '| N | batch | 状态 | 第 1 会话比值 [95% CI] | 第 2 会话比值 [95% CI] |','|---:|---:|---|---:|---:|']
    label={'confirmed_under_preregistered_rule':'符合预注册条件','unconfirmed':'未达 2% 门槛','default_faster_control':'默认更优对照'}
    def ci(x):return '—' if x is None else f"{x['geometric_mean_ratio']:.6f} [{x['ci95_low']:.6f}, {x['ci95_high']:.6f}]"
    for row in combined['candidates']:
        lines.append(f"| {row['n']} | {row['batch']} | {label[row['status']]} | {ci(row['first'])} | {ci(row['second'])} |")
    lines += ['', '首会话共 11 点（110 对），其中 2 点是默认更优对照；第二会话复核 9 点（90 对），两会话均无 partial。另 6 点两会话点估计低于 1，但至少一会话不满足预注册的 2% 实用门槛，因此不列为正向确认。',
       '', '## 三个确认计划的实际结构', '',
       '下表 R 是每线程逻辑寄存器包络；`CUDA regs` 是编译后每线程物理寄存器数量，两者含义不同。shared 是动态共享内存字节；这三点的编译后静态共享和 local memory 均为 0。block 为实际 local size。', '',
       '| N / batch | 计划 | radix | R | group | block | CUDA regs | dynamic shared B |','|---|---|---|---:|---:|---|---:|---:|']
    for row in plans['results']:
        if row['status']!='confirmed_under_preregistered_rule':continue
        for key,name in (('original','原始默认'),('candidate','候选')):
            q=row[key];lines.append(f"| {row['n']} / {row['batch']} | {name} | {q['radix']} | {q['logical_registers']} | {q['grouped_batch']} | {q['local_size']} | {q['cuda_registers_per_thread']} | {q['dynamic_shared_bytes']} |")
    lines += ['', '三点均不属于按旧端点与寄存器规则重构的旧结构域；此标记是结构比较，不是完整源码与运行状态等价性证明。三点变化不止一个维度。',
      '', '## 全部分布与默认回退', '',
      '393 个完整筛选点中 15 点的三轮中位耗时比小于 1，涉及 9 个负载；其中 6 点的筛选比值不大于 0.98，涉及 3 个负载。19 个负载的已筛选候选都不快于默认，因此描述性回退保留默认。这里的回退仅是在已测点和默认之间取最小值，并不等于已经实现、计入调优成本的在线策略。',
      '', '![所有完整筛选点的耗时比分布](local_figures/screen_distribution.png)', '',
      '| N | batch | 完整点数 | 最小比值 | 中位比值 | 最大比值 | 默认回退后最小比值 |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(screen['cases'],key=lambda x:(x['batch'],x['n'])):
        lines.append(f"| {r['n']} | {r['batch']} | {r['completed']} | {r['min_ratio']:.6f} | {r['median_ratio']:.6f} | {r['max_ratio']:.6f} | {r['fallback_best_screen_ratio']:.6f} |")
    lines += ['', '所有 393 个原始筛选比值以及每个负载的旧域最优点保存在 [分布明细](search_local_distribution.json)。其中 388 点落在旧结构域外，5 点落在旧结构子集；未进行等候选数、等墙钟预算的旧搜索随机复现，不能据此宣称新搜索优于旧搜索。静态分层会相对过度抽取极端 stage 数或寄存器值，实际线程布局覆盖也不保证均匀。',
      '', '## 未覆盖和后续决策', '',
      '[混合因子和多 upload](../混合因子与多上传未覆盖与后续路径.md) 本轮未实施、未测试；所有结论仅限固定原型域。首次超 L2 rotation 因 CUDA 13 编译接口变化而失败；修复版在独立目录完成 7 个 CUDA 构建和 CPU 回归步骤，但构建后 extension 剩余 2,183.586 秒，触发 `insufficient_extension_budget_for_guard_and_pair` 并退出，未启动 GPU 守卫或执行。按 runner 默认的 600 秒超时计算，预留门槛为 2,520 秒；实际启动命令未作为独立文件冻结，不能仅凭本地归档独立证明该默认值就是启动参数。[独立构建与最终预算审计](rotation_uncovered_local_audit.json)核对了原始失败、两套源码冻结、完整头树、编译依赖、修复版二进制、全部构建日志及停止状态。超 L2 的图、数值和性能仍未覆盖；工作集超过 L2 本身也不证明 cache miss 或冷缓存。',
      '', '三点通过了本轮预注册的同机确认门槛，支持继续检查相邻负载与不同设备，但不是模型泛化或实际调优总成本获益的证明。本轮没有训练模型；若后续评估模型，应先补同预算默认/简单搜索对照、完整规划成本和按长度族及 GPU 架构留出的无泄漏测试。',
      '', '审计脚本、原始日志、清单、图的 SVG/PNG 和逐计划完整结构均保存在本目录；探索性筛选和两次确认独立显示，不以未测点或统计显著性替代实际覆盖。','']
    target=D/'搜索与双会话独立审计报告.md'
    with target.open('x',encoding='utf8',newline='\n') as out:out.write('\n'.join(lines))
    print(target)


if __name__=='__main__':main()
