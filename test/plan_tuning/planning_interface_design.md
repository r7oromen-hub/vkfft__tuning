# VkFFT 计划描述与延迟编译接口审查

审查对象为本地 `VkFFT/` 的 1.3.4 源码，提交 `066a17c17068c0f11c9298d848c2976c71fad1c1`，重点是当前 CUDA、FP32、1D、2/3/5-smooth、C2C forward、独立输入/输出实验。以下接口改造均为**未来方案，尚未实现**；本文没有修改 VkFFT、测量程序或运行/分析脚本。

## 当前实验是否需要改 initializeVkFFT

**不需要。** 当前任务是参数搜索与离线关系验证：为每个候选设置已有的 `aimThreads`、`registerBoost`、`useLUT`，完整调用 `initializeVkFFT`，再读取实际计划结构和测量执行时间。现有测量程序的对应位置是 [参数赋值](vkfft_plan_benchmark.cpp#L417)、[完整初始化](vkfft_plan_benchmark.cpp#L421)、[初始化后结构提取](vkfft_plan_benchmark.cpp#L430)。这能研究“请求参数是否改变实际结构”“结构与运行时间的关系”，无需改变原库。

但当前没有实现“仅规划、不编译”。即使分析只使用逻辑结构特征，采集它们的当前程序也已经完成了编译；离线模型结果不能直接证明在线系统节省了规划/编译时间。`initialize_ms` 是完整初始化时间，应用层 CUDA graph 的构造另记 `graph_build_ms`。

三个容易混淆的开关：

| 现有开关 | 实际语义与证据 | 能否直接实现 plan-only |
|---|---|---|
| `makeForwardPlanOnly=1` | 只跳过逆向计划分支；正向分支仍调用 `VkFFTPlanAxis`，见 `InitializeApp.h:1484、1543、1573` | 不能；这里的 only 是只要 forward |
| `keepShaderCode=1` | 在 `Plan_FFT.h:778` 保留生成源码，仍经过 `773` 的编译；`RunApp.h:59` 还会在执行时打印源码 | 不能 |
| `loadApplicationFromString` | CUDA 编译函数 `CompileKernel.h:302` 可读取已编译二进制，之后仍在 `442` 加载模块 | 是已有二进制复用路径，不是提取未编译候选结构 |

## 实际调用链与阶段依赖

下表路径均相对于源码根目录；链接后的行号对应本次审查版本。

| 阶段 | 实际调用点 | 形成的信息与副作用 |
|---|---|---|
| 配置归一化与设备查询 | [`vkFFT_InitializeApp.h:1478`](../../VkFFT/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_InitializeApp.h#L1478) 调 `setConfigurationVkFFT`，定义在 `428` | CUDA 分支 `546–621` 查询架构、线程/共享内存/warp 等限制；`1229–1246` 解析 aim/LUT；`1325` 接收 registerBoost；不是一个纯配置 memcpy |
| host 计划对象与调度 | 同文件 `1544` 分配正向 `VkFFTPlan`，`1548` 调 `VkFFTScheduler` | scheduler 选择分解、passes、stage 顺序和逻辑寄存器；可能改变配置，如 Bluestein 分支 |
| 每个 axis/upload 的准备 | 同文件 `1572–1573` 调 [`VkFFTPlanAxis`](../../VkFFT/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h#L33) | **规划细化、资源准备、代码生成、编译混在同一个函数内** |
| 早期资源准备 | `vkFFT_Plan_FFT.h:231、236、244` | 调 `VkFFTGenerateRaderFFTKernel`、`VkFFT_AllocateLUT`、`VkFFT_AllocateRaderUintLUT`，发生在 launch layout 细化及主 kernel 编译之前 |
| launch/layout 细化 | `vkFFT_Plan_FFT.h:440–470` | 处理 buffer 更新/角色，`452` 调 `VkFFTSplitAxisBlock`，`468–470` 写入实际 `localSize` |
| 源码生成 | `vkFFT_Plan_FFT.h:751–768` | API 参数初始化，分配 `code0`，调用 `shaderGen_FFT`；生成过程还计算 shared layout 与读写寄存器策略 |
| NVRTC 与模块加载 | `vkFFT_Plan_FFT.h:773` 调 [`VkFFT_CompileKernel`](../../VkFFT/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_CompileKernel.h#L26) | CUDA 分支 `317` 创建 NVRTC program，`359` 编译，`389/414` 等取得 CUBIN，`442` 加载模块，`453` 取得 `CUfunction` |
| 编译后 kernel 配置 | `vkFFT_CompileKernel.h:463–464` | 如果需要，对已加载函数设置动态共享内存上限 |
| 本 axis 收尾 | `vkFFT_Plan_FFT.h:778–790` | 释放源码（默认）、释放 API 临时容器，并将 `axisBlock` 为 dispatch 再次规范化；不能假定中途字段等于函数返回后字段 |
| 整个 app 后处理 | `vkFFT_InitializeApp.h:1603、1633、1713–1723` | 有需要时分配临时设备缓冲；Bluestein 还生成 phase vectors；序列化分支在 `1726` 之后 |

源码主链可以简写为：

```text
initializeVkFFT
  setConfigurationVkFFT                 # 硬件限制/请求归一化；部分 host/API 资源
  calloc(VkFFTPlan)
  VkFFTScheduler                        # 全轴的粗分解与逻辑寄存器/stages
  for each axis/upload:
    VkFFTPlanAxis
      Rader/LUT preparation             # 当前有资源副作用，某些算法会递归 FFT
      buffer roles/strides + SplitAxisBlock + localSize
      initParametersAPI + shaderGen_FFT # 包含 shared layout 等细化
      VkFFT_CompileKernel                # NVRTC + module load + function handle
      source/API cleanup + dispatch axisBlock normalization
  temp-buffer allocation
  optional Bluestein phase preparation
  optional compiled-binary serialization
```

注意实际初始化是“先调度各轴，再逐个 upload 准备并立即编译”，并非先完成整个 app 的所有最终描述，再统一编译。因此，不能在某一个 upload 的编译前 callback 中立刻决定整个候选的得分：后面的 upload 描述可能还没有形成。

## 哪些特征能在编译前获得

“可在编译前得到”也有层次：有些在 Scheduler 后就有，有些必须先做 layout 细化，另一些当前只有运行 host codegen 后才最终确定。它们都不同于“无需初始化即可从用户参数直接读取”。

| 属性 | 最早可靠阶段/证据 | 使用约束 |
|---|---|---|
| N、请求 batch、精度、placement、strides、请求 knobs、设备硬件限制 | 原始配置及 `setConfigurationVkFFT` | 设备共享内存容量不是该 kernel 的实际占用 |
| `numAxisUploads`、`axisSplit`、每个 upload 的 `fftDim` | [`Scheduler.h:2651、2968、3207–3208`](../../VkFFT/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h#L2968) | 要保留所有 uploads，不只第一个 |
| `numStages`、`stageRadix`、逻辑 `registers_per_thread`、`min_registers_per_thread`、`usedLocRegs`、实际 registerBoost | `Scheduler.h:3200、3227–3232、3268–3270` | 是生成器的逻辑存储策略，不是 CUDA 物理寄存器数 |
| `groupedBatch`、launch `localSize`、`axisBlock` | [`AxisBlockSplitter.h:26`](../../VkFFT/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h#L26)，`Plan_FFT.h:452、468–470` | `axisBlock` 在 `786–790` 还会交换回 dispatch 语义；应分别保存实际 launch shape 与规范化 dispatch 字段 |
| LUT 使用策略、reorder、strides/offset、buffer 角色 | 配置与 `VkFFTPlanAxis` 的进一步细化；kernel `LUT` 在 `647` 赋值 | LUT 内容/设备地址不应成为模型结构特征；layout 元数据可独立描述 |
| 实际申请的 dynamic shared bytes：`usedSharedMemory` | host codegen：[`shaderGen_FFT:97`](../../VkFFT/vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel2/vkFFT_FFT.h#L97) → [`appendKernelStart:86`](../../VkFFT/vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelStartEnd.h#L86) → [`appendSharedMemoryVkFFT:83–105`](../../VkFFT/vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_SharedMemory.h#L83) | 已在 NVRTC 前确定；但只运行 Scheduler 不会得到它。`sharedMemSize` 是预算字段，不能冒充实际 bytes |
| `readToRegisters`、`writeFromRegisters` | `shaderGen_FFT:99–100` 调 [`setReadToRegisters`/`setWriteFromRegisters`](../../VkFFT/vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_ReadWrite.h#L33) | 是 host codegen 的派生策略；需要复用同一计算或保留该阶段 |
| 源码/源码 hash、源码长度 | `shaderGen_FFT` 成功返回后、`VkFFT_CompileKernel` 前 | 无需 NVRTC，但要付源码生成成本；当前会有每 kernel `maxCodeLength` 的缓冲分配 |
| CUDA 实际寄存器、静态 shared bytes、local memory bytes、函数线程上限 | 模块编译加载并得到 `CUfunction` 后，测量程序 [`cuFuncGetAttribute`](vkfft_plan_benchmark.cpp#L204) | 不能输入“编译前选择”模型；当前已与结构特征分开保存为 `compiled_diagnostics` |
| 执行时间、正确性、graph 节点/构造时间 | 可执行 kernel、资源和应用 graph 建立之后 | 是测量标签/诊断，不是未编译候选的已知结构 |

额外的数据一致性陷阱：`VkFFTPlanAxis:55–60` 会将 1D batch 折入 `actualFFTSizePerAxis[][1]`，同时把 `configuration.numberBatches` 改成 1，并另存 `actualNumBatches`。因此最终 `resolved_config.numberBatches=1` 不代表原请求只有一个 batch，也不能把这个归一化后的结构当原始配置直接再次初始化。新接口必须保留独立、不可变的原请求及逻辑 batch。

请求 knob 也不是强制结构：`registerBoost4Step` 在 `InitializeApp.h:1461` 被设为 1；Scheduler 在多 pass 情形 `2594–2601` 切换 boost 逻辑，在 `2617–2619` 还会因非 2 次幂及 `registerBoostNonPow2` 限制退回。候选应以最终描述为准；不能只按请求参数推断生成结构唯一性。

## 最小侵入路线：先做可靠的“免编译描述”原型

建议保留公开 `initializeVkFFT` 的原有语义，不向现有成功返回的 `VkFFTApplication` 塞入“部分可执行”状态。先新增单独的描述入口；初版只支持本次实验的 smooth 1D CUDA C2C forward 范围，其他情况明确返回不支持。

范围检查必须是显式 allowlist，不能仅判断 N 的素因子。本原型限定 FP32、单 stream、连续布局、自然输出和当前实验的独立输入/输出语义；仅接受明确列出的 N/batch、三项 tuning knobs 及必需配置。卷积、R2C、DCT/DST、额外坐标/多 kernel、callback、非默认 padding/offset、自带临时缓冲以及 application binary 导入/导出等未覆盖设置应拒绝。Scheduler 返回后、进入任何 `VkFFTPlanAxis` 前，再检查所有 uploads 的 Rader/Bluestein 标志，避免到资源准备或递归生成已经发生后才判定不支持。

1. 复用 `setConfigurationVkFFT` 和 Scheduler 的真实分支，不另写一份近似调度公式。每个候选使用自己的 draft 状态，记录原始配置、归一化配置及硬件属性版本。原请求应转换为拥有自身存储的 canonical request：例如 buffer-size 数组需复制数值，不能只复制含借用指针的 `VkFFTConfiguration`；实际 buffer/device/stream bindings 及其生命周期另行管理。`InitializeApp.h:1053、1087、1114` 附近的借用/分配行为需要区分。
2. 将 `VkFFTPlanAxis` 的实现抽出一个内部 build mode；旧签名继续委托完整模式，保持现有调用者行为。描述模式执行现有 layout 与 host codegen，但绕开 `Plan_FFT.h:773` 的 `VkFFT_CompileKernel`。若需要源码 hash/长度，须在 `shaderGen_FFT` 返回后、源码释放前逐 upload 保存这些值；源码在 `778–781` 按明确所有权释放，不能用 `keepShaderCode` 充当这个模式。
3. 所有 axis/uploads 完成并执行相同 dispatch 字段规范化后，导出**只含值与数组**的 `PlanDescription`，然后释放 draft；在 app 尾部的临时设备 buffer 分配、Bluestein phase 执行和二进制序列化之前结束。对 `VkFFTApplication`、`PfContainer` 和 LUT 的指针不能做浅拷贝当持久化描述符。这个 MVP 的描述是评分和一致性核验用的快照，尚不是能独立恢复 executable 的完整 IR。
4. 模型为全部候选描述打分。首版对 winner 使用保留的**原始请求**再次调用原始 `initializeVkFFT`，完整构建唯一被选中的候选；随后比较实际描述与评分描述一致性。比较须采用带版本的语义字段 allowlist，排除设备地址、模块句柄和 `performBufferSetUpdate` 等资源绑定瞬态字段；例如 app 尾部实际分配 temp buffer 后会在 `InitializeApp.h:1685` 改变更新标志。评分字段相等只是必要检查，原型验收还应比较相同请求下的逐 kernel 生成源码 hash、编译选项及逻辑 binding/layout 元数据，不能把较少特征的 hash 相等称作完整 kernel 等价证明。这样会对 winner 重算调度/代码生成，但避免把半成品 app 的内部指针和清理状态带入线上执行。
5. winner 编译失败或描述不一致时，记录失败并按明确策略尝试下一候选或库默认；任何 fallback 都计入端到端时间，不能默默忽略。

这条路线的优势是改动边界局限在新增描述入口及 PlanAxis 的内部编译门控，真正编译和执行仍复用原路径。它实现的是“非 winner 不调用主 kernel 编译器”；不是已经实现的代码，也不是立即得到无 GPU 资源副作用的纯 CPU 规划器。

**初版仍有已知成本**：直接复用当前 PlanAxis 时，LUT=1 分支在主编译前就会创建 LUT 内容、分配/上传设备资源。FP32 路径的证据是 [`ManageLUT.h:975–977、1199–1206`](../../VkFFT/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_ManageLUT.h#L975)。`setConfigurationVkFFT:1087` 有 host 分配，多 stream 时 `630–636` 还创建 events。必须分别计量 draft/layout/codegen/LUT 成本，不能把上述初版称作“零资源纯规划”。本实验单 stream 可避开多 stream event 分支，但没有自动消除 LUT 成本。

原型验证时还应在所有编译入口设置计数/拒绝钩子，证明描述候选期间实际 NVRTC 调用数为零。只观察主 `CompileKernel` 被跳过不够：Rader/Bluestein 的辅助路径可能另行初始化和执行 FFT，见下节。因此初版范围检查必须既验证输入 smooth，也验证 Scheduler 的结果未启用 Rader/Bluestein。

零 NVRTC 计数也不等同于没有物化可执行代码：已有 binary 路径可以绕过 NVRTC 后仍加载模块。因此验收应同时观测整个调用链的编译、模块加载和 kernel launch，要求描述期间三者均为零；LUT 的分配/H2D 则作为 MVP 允许但必须计量、清理的独立副作用。该原型仍依赖真实 CUDA 设备/上下文，不能对外承诺离线、无设备可用。`UpdateBuffers.h:678–692` 的 plan-stage 分支允许缺少部分 bindings，但这本身不证明所有描述字段都与有 bindings 时一致；在专门验证之前，不能靠伪造设备地址或省略实际 buffer-size 元数据来宣称纯 host 描述。

## 下一层：将结构准备、源码生成和资源物化真正分开

在原型验证后，再把当前交织逻辑分成下面的接口；名称仅作设计示例，并非已有 API：

```cpp
DeviceCaps queryDeviceCaps(device);             // 可复用的硬件快照
PlanDescription describePlan(request, caps);    // 无 kernel 编译/加载/执行
FeatureVector extractFeatures(description);    // 数值数组；带 schema 版本
ExecutablePlan materializePlan(description, bindings); // 仅选中候选
```

这里的 `materializePlan(description, bindings)` 属于下一层目标：需要先定义足以重放所有 codegen/resource 决策的完整计划 IR。MVP 更准确的接口是 `initializeSelected(originalRequest, bindings, expectedDescription)`，内部仍调用原始初始化并核验快照。不能把当前模型用的少量数值特征直接当作上面 `materializePlan` 的完整输入。

具体拆分位置：

| 边界 | 改造建议 | 必须保持的依赖 |
|---|---|---|
| `setConfigurationVkFFT` | 把硬件查询、配置归一化与资源创建分离；明确哪些值来自 request，哪些来自设备 | 不能为了无设备规划伪造 caps；同一候选须使用与执行设备一致的限制 |
| `VkFFTScheduler` → `VkFFTPlanAxis` | 保留 Scheduler 原算法，提取 axis/upload 特化、strides 和 block splitter 为确定性的描述准备 | batch 折叠、全轴状态及实际 registerBoost 都是描述的一部分 |
| `VkFFT_AllocateLUT` | 先计算 LUT 是否存在、大小、stage offsets、共享关系；内容计算、device allocate/H2D 到 materialize | 当前 LUT 函数把 offset 元数据和分配混在一起；不能直接全跳过再使用未初始化元数据 |
| `VkFFTCheckUpdateBufferSet` / `VkFFTUpdateBufferSet`（`Plan_FFT.h:440–445`） | 描述阶段使用 input/output/temp 等逻辑角色，物化时才绑定真实地址/句柄 | buffer 角色、stride、offset、临时缓冲需求影响 kernel；无需把真实设备地址喂给模型 |
| `appendSharedMemoryVkFFT` 与 read/write register 策略 | 抽出纯 layout/策略计算，字符串生成复用结果；避免另一套近似计算 | shared stride/bank padding 决定实际 shared bytes；需逐字段对照现有 codegen |
| `shaderGen_FFT` → `VkFFT_CompileKernel`（`768→773`） | 显式保留 codegen 输出和编译输入；只有选中候选进入编译/模块加载 | 同一份描述必须生成一致代码、函数名、push constants 布局、架构选项 |
| `initializeVkFFT` 尾部 | 将 temp buffer、设备 LUT、辅助 kernel、模块资源统一归物化阶段，最后绑定/提交 | 不能在 graph 捕获后改变任何这些资源的地址或生命周期 |

如果第一版模型需要现有 `dynamic_shared_bytes`、read/write flags，可暂时保留 host codegen 作为描述准备的一部分，并把这部分时间计入候选枚举。只有将这些派生决策抽成独立纯函数后，才能同时避免为所有候选生成源码。编译避免与代码生成避免是两个不同的优化目标。

## 风险与必须验证的条件

- **Rader/Bluestein 不能机械套用 smooth 路径。** `Plan_FFT.h:231` 的 Rader 生成函数定义于 [`RecursiveFFTGenerators.h:996`](../../VkFFT/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_RecursiveFFTGenerators.h#L996)，其 `1146` 会递归 `initializeVkFFT`、`1169` 分配设备内存、`1222` 等执行 forward。Bluestein phase 函数在同文件 `35`，有 `56` 的设备分配和 `298` 的递归初始化。必须把这些依赖纳入描述图或暂时拒绝；也不能把 R2C、DCT/DST、多维场景宣布为已支持。
- **描述时点不能错。** Scheduler 后的结构是粗描述；codegen 后、编译前才包括当前模型的全部派生字段。`axisBlock` 的末尾交换与 `localSize` 的实际 launch 语义须分别保存。只截断 `initializeVkFFT:1548` 或只在 NVRTC 调用前打日志都不足以得到可重放的完整计划接口。
- **非执行状态必须明确。** 当前 `VkFFTAppend` 在 [`RunApp.h:104–105`](../../VkFFT/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h#L104) 主要检查计划指针，不代表每个 `CUfunction` 已加载。新增 draft 不应冒充成功初始化的可执行 app；否则可能用空 kernel 句柄启动。
- **所有权和失败清理。** 当前 `Plan_FFT.h:778–785` 与 [`InitAPIParameters.h:550、571`](../../VkFFT/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_InitAPIParameters.h#L550) 分散释放源码与 API 容器；[`deleteAxis:62–68`](../../VkFFT/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_DeletePlan.h#L62) 处理 LUT/模块。延迟编译不能简单移动一行编译调用并沿用未审计的清理时序；应有 draft/codegen/executable 各自的析构、状态标志和部分失败测试。
- **快照与复算一致性。** original request、设备 caps、库提交、backend、precision、布局/offset、feature schema 必须进入描述缓存键；原请求中的 buffer 容量也不能遗漏，因为 `InitializeApp.h:1177–1222` 会据此选择寻址宽度。模型与 executable 缓存还需各自声明工具链/设备兼容域，记录 NVRTC/driver 版本、目标架构及实际编译选项，不能仅以 N 和三项 knobs 命中。当前 `VkFFTApplication` 有可变配置、借用的 buffer 指针及共享 LUT，不是可序列化 IR。选中后原路径重算需要一致性断言；模型评分基于的描述和真正执行的结构必须对应。
- **训练泄漏。** 实际 CUDA registers、local memory 和编译是否失败，都在编译后才知道。只能作为标签/诊断；若纳入在线评分输入，就不能再声称“仅编译选中结构”。编译前可用字段应有显式 allowlist，而不是把完整结构体自动展开。
- **收益计算要完整。** 新流程需测 `设备/配置准备 + 全候选描述/代码生成/资源准备 + 模型推理 + winner 物化/编译 + graph 构造 + fallback`，并与完整搜索及直接默认配置各自比较。重复执行次数和缓存/编译缓存状态决定是否摊销；当前执行加速比本身不能推出初始化或总耗时节省。

未来改造的验收至少包括：描述阶段零编译/模块加载/kernel launch 验证；19 组候选的编译前/原初始化后语义描述逐字段一致，并核验源码与编译输入；单/多 upload、单/大 batch、LUT 两种路径；原数值与不可变输入测试全部保留；默认 API 行为回归；成功/失败/拒绝候选及反复 describe/destroy 的资源清理；以及单独报告端到端成本。以上测试和 plan-only API **均未在当前实验中实现或测量**。
