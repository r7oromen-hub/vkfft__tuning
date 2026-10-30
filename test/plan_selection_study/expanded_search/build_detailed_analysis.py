"""Recompute descriptive aggregates and write the local Chinese experiment report.

No GPU or network is required. Input summaries were independently audited against
archived raw logs; this script does not repeat that raw-log audit.
"""
import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results_20261030'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def aggregate(ratios):
    return {
        'workloads': len(ratios),
        'geometric_mean_speedup': math.exp(-sum(map(math.log, ratios)) / len(ratios)),
        'arithmetic_mean_latency_reduction': 1 - sum(ratios) / len(ratios),
    }


def table(headers, rows):
    def row(values, tag):
        return '<tr>' + ''.join(f'<{tag}>{html.escape(str(v))}</{tag}>' for v in values) + '</tr>'
    return '<div class="scroll"><table><thead>' + row(headers, 'th') + '</thead><tbody>' + ''.join(row(r, 'td') for r in rows) + '</tbody></table></div>'


def link(name, title=None):
    return f'<a href="{html.escape(name, quote=True)}">{html.escape(title or name)}</a>'


def main():
    screening = read('search_local_distribution.json')
    confirmations = read('two_session_local_combined.json')
    assert confirmations['status'] == 'pass'
    cases = screening['cases']
    assert len(cases) == 28 and sum(x['completed'] for x in cases) == 393
    confirmed_rows = []
    all_wins, practical = [], []
    for c in confirmations['candidates']:
        a, b = c['first'], c.get('second')
        combined = math.sqrt(a['geometric_mean_ratio'] * b['geometric_mean_ratio']) if b else None
        if b:
            all_wins.append(combined)
        qualifies = c['status'] == 'confirmed_under_preregistered_rule'
        if qualifies:
            practical.append(combined)
        def interval(s):
            return f"{s['geometric_mean_ratio']:.6f} [{s['ci95_low']:.6f}, {s['ci95_high']:.6f}]" if s else '未复测'
        confirmed_rows.append([c['n'], c['batch'], interval(a), interval(b),
                               f'{100*(1-combined):.3f}%' if combined else '—',
                               '达到实用门槛' if qualifies else ('小幅收益，未达门槛' if b else '默认更优对照')])
    assert len(all_wins) == 9 and len(practical) == 3
    summaries = {
        'screen_all_28': aggregate([x['fallback_best_screen_ratio'] for x in cases]),
        'two_session_9_plus_default_19': aggregate(all_wins + [1.] * 19),
        'practical_3_plus_default_25': aggregate(practical + [1.] * 25),
        'only_9_improved': aggregate(all_wins),
        'only_3_practical': aggregate(practical),
    }
    evidence_names = [
        'search_local_distribution.json', 'two_session_local_combined.json',
        'search_local_independent_audit.json', 'confirmation_local_chain.json',
        'confirmed_local_structures.json', 'expanded_provenance_v2.json',
        'budget_final_snapshot.jsonl', 'rotation_uncovered_local_audit.json',
    ]
    evidence = []
    for name in evidence_names:
        raw = (OUT / name).read_bytes()
        evidence.append({'file': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    result = {'scope': '2026-10-30 round only; equal workload weights; hot-address execution; observed best with default fallback; not global optimum',
              'two_session_aggregation': 'geometric mean of two session geometric latency ratios; descriptive point estimate, no pooled confidence interval',
              'metrics': summaries, 'inputs': evidence}
    (OUT / '详细分析汇总.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    sections = []
    def section(title, body):
        sections.append(f'<section><h2>{title}</h2>{body}</section>')
    section('1. 核心结论与研究目的', '''
<p><strong>本轮 28 个工作负载等权计算，选择已测最佳候选、无收益时保留原始默认，独立复测点估计对应的几何平均加速为 1.14%。若只采用达到预设实用门槛的三个胜者，平均加速为 0.98%。</strong>最大单项加速约 20.0%，对应耗时下降约 16.65%；不能把最大值当作平均值。</p>
<p>研究问题是：VkFFT 默认启发式在 radix 分解、顺序和线程布局的联合选择上，是否存在可重复的执行性能缺口；扩大候选空间后，收益发生在哪些工作负载，搜索需要多少成本。这是为下一阶段结构评分模型建立证据，不是本轮训练或验证学习模型。</p>
<p>已经证明的是：本设备与本计时条件下，至少三个工作负载存在优于原始默认的完整计划。尚未证明的是：默认普遍较差、全空间最优已找到、学习模型比简单搜索更好，或应用端到端可以获得同样收益。本轮平均值只涵盖本轮选中并测量的候选，没有合并历史轮次的已知胜者。</p>''')
    section('2. 环境、工作负载和对照', '''
<p>设备为单张 NVIDIA RTX 4090，固定 GPU UUID 为 <code>GPU-492bdfa1-90bc-e9d3-a759-c2c9ea7f629e</code>，工具链为 CUDA 13.0。上游源码标识为 <code>066a17c17068c0f11c9298d848c2976c71fad1c1</code>，来源见 provenance 文件。所有数值结论限于同一 GPU，未进行多架构复现。</p>
<p>核心矩阵：N ∈ {128, 256, 512, 1024, 2048, 4096, 8192}，batch ∈ {1, 16, 256, 4096}，共 28 项。范围为 FP32、1D C2C forward、out-of-place、单 upload、registerBoost=1。一个计时单位是整个 batch 的一次正变换，不是单个元素或单条 FFT。</p>
<p>保留三种控制：原始头文件默认、实验头文件不设置候选的默认、通过新接口重放默认结构的 identity。最终收益分母始终是原始头文件默认。实验功能位于隔离生成的头文件树内，不改变公开 initializeVkFFT 接口。28 项 identity 在修复后的独立准入中全部通过。</p>
<p>默认及实验二进制 SHA-256 分别为 <code>3ca7ee798d8962b11e501923b86a70eba3d74d019d2e4e77f86aa9b466cae445</code>、<code>4f39b012db382a486beeb4420d6349ea0116027d02f332d526a5a12cc59d7e23</code>。冻结源文件和生成头树用于区分运行版本，不用当前可编辑源码代替历史执行证据。</p>''')
    section('3. 搜索的参数到底是什么', '''
<p><strong>radix 是直接搜索变量。</strong>生成因子属于 {2,4,8,16,32}、乘积等于 N 的有序分解，允许阶段数、因子多重集和排列顺序变化。例如 1024 的 [32,32] 和 [8,4,4,8] 都在原型候选域内；它们并非同一分解的简单排序。</p>
<p>同时搜索统一的每线程逻辑寄存器包络 R ∈ {2,4,8,16,32,64}，要求 R 整除 N，且 R 能被每个阶段 radix 整除；group 是 batch 的二次幂因子。R 不等于编译后的物理寄存器数。FFT 线程维规划目标为 N/R，另一逻辑维由 group 决定；最终 localSize/blockDim 由调度器、轴交换和资源规划一致导出，不是随意覆盖 CUDA launch 参数。</p>
<p>在调度入口注入请求，重建 stage 及 per-radix 寄存器状态，保留后续资源检查与依赖推导，再继续 splitter、LUT、读写策略及代码生成。不能只替换最终 stageRadix 或 blockDim 而沿用旧索引、LUT 和共享内存布局。</p>
<p>结构记录区分代码生成前、代码生成后但编译前、编译后。共享内存及部分读写标志可能在代码生成中确定；CUDA 物理寄存器和编译资源属于编译后信息。将来训练“代码生成前”的评分模型时，不能把后两个时点才可获得的特征偷偷作为免费输入。describe 可避免 NVRTC 编译、模块装载和 kernel launch，但仍执行 host 代码生成，可能建立设备上下文或准备 LUT；它不是已经实现的零副作用纯离线规划 API。</p>''')
    coverage_rows = [
        ['原型规则枚举',1194648,'包含静态无效请求；不是完整 VkFFT 数学计划空间'],
        ['静态拒绝',500976,'原型约束过滤，不代表 GPU 数值失败'],
        ['静态通过',693672,'尚不保证调度可执行'],
        ['identity 控制',28,'独立控制，不占非默认候选槽'],
        ['预选非默认',1792,'每项 64 个，固定清单'],
        ['实际 describe',649,'已进入实际调度路径'],
        ['调度拒绝',255,'识别为 VkFFT 3002 拒绝，不补选'],
        ['验证与筛选完整',393,'纳入性能筛选统计'],
        ['已开始但未完成',1,'预算边界 partial，排除收益统计'],
        ['选中但未开始',1143,'不能归类为无效或较慢'],
        ['静态通过但未选中',691852,'没有设备执行证据'],
    ]
    section('4. 候选选择、覆盖率和搜索难度', '''
<p>每项最多预选 64 个非默认请求，种子为 20261030；按阶段数、radix 多重集、预测 FFT 线程维和 group 分层抽样。调度拒绝或重复请求也消耗槽位，不用后续请求替换。执行顺序为 batch 优先、N 升序，在候选轮次之间遍历全部工作负载；预算到点停止新增任务。</p>''' + table(['层次','数量','含义'], coverage_rows) + f'''
<p>预选请求中完整完成筛选的比例为 {393/1792:.2%}；相对于静态通过请求，仅为 {393/693672:.4%}。实际 describe 中约 {255/649:.2%} 被调度拒绝。三个比例分母不同，不能混用。最终每项完整候选只有 2–24 个，远未达到所有项各完成 64 个的均衡搜索。</p>
<p>这说明搜索难点首先是约束与可行性：数学分解合法不等于 VkFFT 当前调度支持，更不等于生成的 kernel 高效。其次是组合规模与测量成本：分解顺序、统一 R、group 共同增加候选数，每个执行候选还需独立校验和配对计时。</p>
<p>393 个完整点中，388 个不属于按旧端点、寄存器和线程约束重构的旧结构域，5 个属于旧域。新实现确实扩展了结构覆盖；但未完成旧、新空间等候选数或等墙钟预算对比，所以不能据此量化“扩展搜索比旧搜索平均多赚多少”。分层抽样也不保证实际布局均匀，可能相对多抽极端阶段数或 R。</p>''')
    section('5. 正确性门槛和性能测量', '''
<p>每个完整候选验证五种输入：zero、impulse、constant、quarter-tone、random，分别运行 seed 42、53、71，共 15 条验证记录。解析输入以解析结果为参照并与独立 CPU FFT 交叉检查；随机输入使用独立双精度 mixed-radix CPU FFT 以及可精确表示的 batch 缩放关系。随机 batch 不是各自独立随机采样，这是验证覆盖的边界。</p>
<p>检查全部 N×batch 输出，不抽样；输入执行前后按字节比较，要求保持不变；输出先填充 NaN 字节以检测遗漏写入。每个 batch 都必须输出有限值，且 L2(error)/(sqrt(N)×L2(input)) ≤ 10⁻⁵、L∞(error)/L2(input) ≤ 10⁻⁴；零输入要求精确零输出。逐元素 atol/rtol 是额外诊断，不是最终判定规则。阈值属于依据先导实验设定的 FP32 工程门槛，不是理论误差界。</p>
<p>还要求请求与实际 radix、R、group、线程维一致，核对源码/编译描述符、函数线程上限、动态与静态共享内存限制。原始默认、实验默认和 identity 的一致性优先于任何性能结论。首次准入曾因初始化后 axisSwapped 被归一化，导致身份检查误判并停止；修正检查时点后重新构建、重新准入。旧失败日志保留，不能说整个研究从未出现异常。</p>
<p>计时使用 CUDA graph 重复读取同一输入、写入同一输出地址，至少 5 次且至少 1000 ms 预热；每进程 31 个测量样本，原始 graph 区间至少 0.5 ms，再除以 graph 内变换重复数得到每次完整 batch 耗时。配对比较中默认与候选顺序随机化，性能计时期间由单一实验调度器独占卡，不并发其他 GPU 实验或重型远端构建。</p>
<p>筛选每个候选进行 3 轮与原始默认的配对比较，按配对耗时比中位数选择胜者；确认使用新测量数据，每个案例每会话 10 个独立进程配对。首会话 11 项、110 对，包含 9 个筛选胜者和 2 个默认更优对照；第二会话复核 9 个胜者、90 对。核心搜索及确认日志中未发现数值失败、编译后资源拒绝或守卫事故；255 个调度拒绝另行保留。</p>
<p>这些时间不包含 H2D/D2H、CPU 参考、初始化、编译、图构建或验证。称为“热地址/热缓存条件”是为了表达重复同址、没有主动清缓存的协议，不能据此断言所有工作负载都完全驻留 L2。它不是冷缓存、流式工作集或应用端到端性能。</p>''')
    summary_rows = []
    for key,label in [('screen_all_28','28 项：筛选最优 + 默认回退'),('two_session_9_plus_default_19','28 项：9 项双会话结果 + 19 项默认'),('practical_3_plus_default_25','28 项：仅采用 3 项实用胜者'),('only_9_improved','只统计 9 项小幅或明显受益者'),('only_3_practical','只统计 3 项实用胜者')]:
        m = summaries[key]
        summary_rows.append([label, m['workloads'], f"{m['geometric_mean_speedup']:.6f}×", f"{100*(m['geometric_mean_speedup']-1):.3f}%", f"{100*m['arithmetic_mean_latency_reduction']:.3f}%"])
    section('6. 平均收益如何计算', '''
<p>设每项候选/默认耗时比为 rᵢ；回退后的比值 qᵢ=min(1,rᵢ)。全部工作负载等权的几何平均加速比为 exp(−Σln(qᵢ)/28)，平均加速百分比为该值减一。另一指标是算术平均耗时下降 Σ(1−qᵢ)/28，两者不是同一个量。</p>
<p>双会话汇总点估计使用 sqrt(rᵢ,会话1×rᵢ,会话2)，19 个未找到更快候选的工作负载取默认比值 1。这是描述性汇总，不将两会话混成一个新的置信区间，也没有为整体平均构造总体置信区间。</p>''' + table(['口径','项数','几何平均加速比','平均加速','算术平均耗时下降'],summary_rows) + '''
<p>因此用户关心的“平均能提升多少”，主答案是约 1.14%；它对应平均耗时下降约 1.064%。只有三个实用胜者的子集平均加速 9.57%，不能用于代表完整矩阵。28 项中位回退收益为 0：多数工作负载本轮没有找到优于默认的方案。这里等权的是工作负载，不是元素数量、调用次数或实际应用耗时；总时间加权收益需要应用调用分布与绝对时间，不能从这个平均值直接推导。</p>
<p>默认回退是对已测结果做的离线最优选择，尚未实现并评估具有相同效果的在线选择策略。未计搜索成本，也不意味着系统事先知道应该选谁。</p>''')
    section('7. 独立确认：收益分布与统计边界', '''
<p>耗时比小于 1 表示候选更快。每会话将 10 个配对比值取对数，以自由度 9 的 t 区间计算均值区间再取指数。预设实用条件为：两个会话各自的几何平均耗时比 ≤0.98，且各自 95% 区间上界 &lt;1。区间依赖配对区组假设，未作多重比较修正；独立确认降低筛选噪声影响，但不是跨设备或所有可能会话的保证。</p>''' + table(['N','batch','会话 1 比值 [95% CI]','会话 2 比值 [95% CI]','两会话点估计耗时下降','判定'],confirmed_rows) + '''
<p>9/28（32.14%）项有两会话方向一致的小幅或明显收益；3/28（10.71%）达到实用门槛。六项较小收益与三个实用胜者分开报告。两个默认更优对照仅做第一会话，未计入正向收益。</p>
<img src="local_figures/confirmed_pair_intervals.png" alt="两个独立确认会话的配对耗时比和区间">
<p>393 个筛选点中，仅 15 个比默认快（3.82%），其中 6 个达到筛选阶段 2% 耗时下降。多数候选较慢，最差约为默认耗时的 6.69 倍。此分布只属于实际抽样并完成的点，不代表完整合法空间的概率分布。</p>''')
    structures = [
        [128,16,'默认','[8,8,2]',8,8,'[8,16,1]',40,9216],
        [128,16,'胜者','[8,16]',16,1,'[8,1,1]',76,1152],
        [128,4096,'默认','[8,8,2]',8,8,'[8,16,1]',40,9216],
        [128,4096,'胜者','[8,2,8]',16,16,'[16,8,1]',64,17408],
        [1024,256,'默认','[32,32]',32,1,'[32,1,1]',88,8704],
        [1024,256,'胜者','[8,4,4,8]',16,1,'[64,1,1]',56,8704],
    ]
    section('8. radix、寄存器和 blockDim 的耦合分析', table(['N','batch','计划','radix','逻辑 R','group','实际 blockDim','物理寄存器/线程','动态 shared B'],structures) + '''
<p>128/16：从三个阶段变为两个阶段，同时 R 增加、group 减少、block 从 128 个线程变为 8 个线程，共享内存下降而每线程物理寄存器增加，最终耗时下降约 16.65%。说明“更少物理寄存器”或“更大 block”都不是普遍可靠的单调准则。</p>
<p>128/4096：radix 多重集相同但顺序变化，R、group 和线程维方向也变化；block 总线程数仍为 128，物理寄存器与共享内存均增加，仍获得约 2.46% 耗时下降。只比较 block 总线程数会遗漏布局差异。</p>
<p>1024/256：从两个 radix-32 阶段变为四个较小 radix 阶段，group 保持 1，R 降低、FFT 线程数翻倍，物理寄存器从 88 降为 56，动态共享内存不变，耗时下降约 6.49%。更少 stage 并不总是更快。</p>
<p><strong>这些是完整计划变化的收益，不是单变量因果实验。</strong>可能机制包括指令量、并行粒度、线程利用率、同步、shuffle、访存以及占用率变化，但本轮没有充分的硬件计数器与固定其他因素的成组消融，不能定量归因。“分解选择与布局耦合”已在可行性和资源推导上明确；统计意义上的交互效应大小尚未估计。</p>
<p>回答“哪些参数影响大”：当前最大的实测改善来自联合变化；尚不能可靠排出 radix 与 group 各自的重要性名次。回答“哪些参数搜索难”：有序分解与一致布局的联合域很大，且存在大量静态通过、实际调度拒绝；但也尚无等预算消融，不能断言某一维独自贡献大部分搜索成本。</p>''')
    rows = [[x['n'],x['batch'],x['completed'],f"{x['min_ratio']:.6f}",f"{x['median_ratio']:.6f}",f"{x['max_ratio']:.6f}",f"{x['fallback_best_screen_ratio']:.6f}"] for x in cases]
    section('9. 完整 28 项筛选结果', '''<p>以下全部是筛选阶段的 3 轮配对比值统计，不混写为确认结果。较小数值更好；回退值为候选最小比值与 1 取小。各项候选数不同，特别是大 batch、大 N 覆盖较少，未找到收益不能证明默认最优。</p>''' + table(['N','batch','完整候选数','最小比值','中位比值','最大比值','默认回退值'],rows) + '<img src="local_figures/screen_distribution.png" alt="完整筛选候选的性能分布">')
    section('10. 成本、失败与未完成范围', table(['阶段','墙钟使用（秒）','上限（秒）'],[['准入',958.3,5400],['核心搜索',12055.0,12600],['独立确认',1914.3,7200],['扩展',1520.4,3600],['合计',16448.0,28800]]) + '''
<p>总设备预留墙钟约 4 小时 34 分 08 秒，包含构建、验证、监控等待和计时，不等于 GPU kernel 忙碌时间。8 小时是上限，不是实际已消耗时间；分阶段未用额度没有转移。搜索因保守的在途任务时间预留而提前停止新增候选，不能把未达到 8 小时理解为搜索已完成。</p>
<p>本轮验证协议重于实际调优的理想最小流程，设备预留成本也不能直接作为每个在线请求的搜索开销。不过执行收益尚未抵扣调优成本是事实。若额外调优成本为 C、每次调用节省 Δt，则至少需要约 C/Δt 次复用才能摊销；本轮没有给出按应用复用次数的盈亏平衡实测。</p>
<p>超 L2 轮换缓冲区实验首次因 CUDA 13 graph API 参数变化编译失败；隔离修复版完成七步构建及 CPU 检查，但剩余阶段预算低于固定启动预留门槛，未启动 GPU 轮换执行。故不存在可用于支持大工作集或冷缓存收益的测量。启动命令也未单独冻结，审计对默认超时推导保留限制。</p>
<p>未覆盖：混合因子长度、多 upload、其他精度与类型、registerBoost 其他值、非统一 R、更广泛布局、其他 GPU、真实应用调用分布、旧新搜索公平成本对比。没有训练评分模型，没有模型泛化实验，也没有本轮独立在线随机搜索基线。当前报告不以文献中的他库结果替代本库实测。</p>''')
    section('11. 对研究动机和下一阶段的判断', '''
<p>可以据此提出的研究动机是：在保证生成与调度依赖一致的前提下，radix 分解和线程布局联合搜索能修正部分默认启发式选择；但可行域复杂、好候选稀疏、实测成本高，因此需要研究更有效的候选生成与排序。现有数据足以支持继续研究，不足以宣称“引入学习模型就能获得显著平均收益”。</p>
<ol><li>先补覆盖：合并并重新确认历史已知胜者，完成本轮未测候选或重新分配固定预算，优先补只有 2–6 个完整点的工作负载。已有赢家未必被本轮抽样覆盖，因此扩大前应先建立跨轮次候选登记表。</li>
<li>做成组消融：固定 group 与 R 比较 radix；固定 radix 比较 R/group；最后比较联合选择。记录那些无法固定派生资源的情况，不把计划变化伪装为单变量作用。</li>
<li>建立同预算对照：原始默认、简单随机/分层搜索、局部搜索，分别用相同候选数与相同墙钟比较，补搜索成本—收益曲线。离线重排必须标为离线重放。</li>
<li>复核适用性：对三个实用胜者补超过 L2 的轮换工作集、相邻 N/batch 与其他 GPU，观察收益是否保持。没有实际运行前，不从热地址结果外推。</li>
<li>最后评估模型：优先考虑合法性过滤、候选排名和是否值得调优的选择；按长度族及设备留出测试，严格区分前代码生成、编译前和编译后特征成本，报告相对简单搜索的收益及额外训练/推理成本。</li></ol>
<p>若补充后平均收益仍约 1%，研究价值更可能来自减少搜索/编译成本、识别少数值得调优的工作负载，而非声称大幅普遍加速。若候选生成仍明显受限，应先修正生成器，不能用目前的负面工作负载证明启发式全局最优。</p>''')
    source_rows = [
        ['实验协议','../study_protocol.json','范围、种子、候选上限、确认门槛及预算'],
        ['候选生成说明','../GENERATOR.md','注入位置、布局约束、结构捕获时点'],
        ['数值与结构审计','../validation.py','校验规则、计时门槛和资源检查'],
        ['确认统计公式','../audit_pairs.py','10 对 log-t 区间及实用判据'],
        ['完整搜索证据归档','search_evidence_immutable.tar','原始搜索日志、清单及冻结证据，约 1.06 GB'],
        ['搜索本地提取目录','search_immutable_local/','归档展开后的本地数据'],
        ['第 1 会话原始日志','confirmation_first_d0a3389193ab406c97c149c0752597f3.jsonl','110 个配对对应的执行记录及结构/数值证据'],
        ['第 2 会话原始日志','confirmation_second_d3e5f3c437f24030830f8cb865247b9a.jsonl','90 个配对对应的执行记录及结构/数值证据'],
        ['完整筛选汇总','search_local_distribution.json','393 点比值与 28 项描述统计'],
        ['双会话汇总','two_session_local_combined.json','逐项、逐会话、逐轮比值及区间'],
        ['独立搜索审计','search_local_independent_audit.json','完整性、处置、来源一致性'],
        ['确认门票链','confirmation_local_chain.json','从筛选结果到确认计划的关联'],
        ['计划资源明细','confirmed_local_structures.json','默认及胜者的实际结构、资源'],
        ['最终预算账本','budget_final_snapshot.jsonl','事件与阶段预算原始记录'],
        ['轮换实验审计','rotation_uncovered_local_audit.json','构建结果与未执行范围'],
        ['本报告聚合输出','详细分析汇总.json','平均值全精度结果和输入 SHA-256'],
        ['本报告生成脚本','../build_detailed_analysis.py','无需 GPU 的可复现计算与 HTML 生成'],
    ]
    sources_html = '<ul>' + ''.join(f'<li>{link(path,title)}：{html.escape(desc)}</li>' for title,path,desc in source_rows) + '</ul>'
    section('12. 数据来源、复现方式与证据边界', '''
<p>本报告以本地保存的 2026-10-30 轮次数据为准。此前独立审计从原始日志重算搜索分布、确认计划和两会话结果；本次报告再从这些审计后的 JSON 全精度重算平均值，并非重新执行 GPU 测试。统计表不从图片读数，也不使用四舍五入后的表格值计算平均。</p>''' + sources_html + '''
<p>在仓库目录运行下列命令可重建本报告与汇总 JSON，不需要远程主机：</p>
<pre>F:\\anaconda\\python.exe test/plan_selection_study/expanded_search/build_detailed_analysis.py</pre>
<p>这一步验证汇总层的数量关系与计算；它不取代原始日志审计，也不重新验证 GPU kernel。关闭远程主机不影响本地离线分析，新增性能证据需要重新提供可用 GPU。下面列出本次聚合所读取或作为证据索引引用的关键文件哈希，用于检测后续文件变化。</p>''' + table(['文件','字节数','SHA-256'],[[e['file'],e['bytes'],e['sha256']] for e in evidence]))
    doc = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VkFFT 调优空间与默认启发式收益详细分析</title>
<style>body{max-width:1200px;margin:36px auto;padding:0 24px;color:#1d2734;background:#fff;font:16px/1.85 system-ui,"Microsoft YaHei",sans-serif}h1{font-size:30px;line-height:1.4}h2{font-size:23px;margin-top:40px;border-bottom:1px solid #ccd6e0;padding-bottom:8px}p{margin:14px 0}table{border-collapse:collapse;font-size:13px;width:100%;line-height:1.5}th,td{border:1px solid #d6dfe8;padding:8px;text-align:left}th{background:#eef3f8}tr:nth-child(even){background:#fafbfd}.scroll{overflow-x:auto}code{overflow-wrap:anywhere;font-size:13px}pre{background:#eef3f8;padding:16px;overflow:auto}img{max-width:100%;height:auto}a{color:#0756a3}li{margin:8px 0}.meta{color:#526173}@media print{body{max-width:none;font-size:11pt}h2{break-after:avoid}tr,img{break-inside:avoid}.scroll{overflow:visible}table{font-size:8pt}}</style>
<h1>VkFFT 调优空间与默认启发式收益详细分析</h1><p class="meta">实验轮次：2026-10-30｜单张 RTX 4090｜本地证据分析｜不含新增 GPU 实验</p>''' + ''.join(sections) + '</html>'
    target = OUT / 'VkFFT调优空间与收益详细分析.html'
    target.write_text(doc, encoding='utf-8')
    print(target)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
