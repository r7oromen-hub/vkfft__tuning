# 扩展候选生成器与接口

该实验生成独立头文件树和新 harness，不修改原始 VkFFT 或公开初始化接口。默认无环境变量时走原始 Scheduler；请求仅支持 CUDA、1D、FP32 C2C 正变换、out-of-place、单 upload、`registerBoost=1`，二次幂 N=2..8192，batch 为 1..65536。仅靠静态条件或 CPU 测试不能证明 kernel 合法或正确。

请求是 `{"radix":[8,4,8],"logical_registers":8,"grouped_batch":4}`，分别设置：

```text
VKFFT_EXPANDED_RADIX=8,4,8
VKFFT_EXPANDED_REGS=8
VKFFT_EXPANDED_BATCH=4
```

三个变量须同时提供；禁止 `VKFFT_DIRECT_*`。radix 限于 2、4、8、16、32，乘积必须为 N；逻辑寄存器 R 限于 2、4、8、16、32、64，必须整除 N 且被每个 stage radix 整除；group 是 batch 的二次幂因子。不接受用户指定 localSize。枚举器 `candidates(n,batch)` 包含全部上述有序分解、R 和 group 组合，保留 `rejected_static` 原因及待后续验证项；`planned_fft_threads=N/R` 是规划目标，不是测量的 block 布局或合法性保证。

生成的 Scheduler 在原资源规划入口从候选重建 stage 多重集及完整 power-of-two per-radix 表（对每个整除 R 的 2/4/8/16/32 赋 R、其他清零），初始化 min/max；原后续线程限制检查仍运行。仅跳过原 optimizer 对多重集的自动合并，保留它的 `usedLocRegs` 和非二次幂状态推导；排序按请求，原 `VkFFTMinMaxRegisterCheck` 后检查整张表和 R 未被更改。splitter 根据这些数据导出 FFT 线程维，并在检查设备线程、工作组、共享内存、batch 整除等约束后注入 group，然后继续默认 swap、LUT 和 codegen。实际初始化后还校验 stage、表、boost、group、线程维和 block；execute 查询编译函数与设备资源上限，再执行完整 batch 数值及输入不变验证。要求设备守卫，不应并行干扰其他实验。

布局检查分两个时点：codegen 后、编译前以当时的 `axisSwapped` 和 `localSize` 检查实际 FFT/group 维；初始化结束后以规范化的 dispatch `axisBlock` 检查 FFT/group 维。VkFFT 在 dispatch 前交换 axisBlock 并把 `axisSwapped` 复位为 0，因此不再用复位后的标志解释保留的 localSize。第一次准入中 N=128、batch=4096 的旧断言因此误报；旧失败记录应保留，新版本仍须重新经过完整身份核验。

`--study-mode describe --placement outofplace` 对请求运行真实 Scheduler、splitter、LUT 准备和 host 代码生成，不编译或执行，输出实际结构及源码 hash；该路径仍可能创建设备上下文和分配/上传 LUT，必须纳入守卫与预算。`--study-mode execute --placement outofplace` 增加编译、资源检查和数值执行。前置参数拒绝在 `failure_phase=arguments`；计划/资源拒绝为 `status=error` 和 `failure_phase=initialize`。成功的候选记录 `expanded_requested_radix`、`expanded_requested_regs`、`expanded_requested_batch`，execute 还记录 `expanded_resource_validation`。它们与实际 `structural_features`、`precompile_kernels` 一起审计。

每个 kernel 的 `precodegen_raw_descriptor` 在 host codegen 前捕获；`raw_descriptor`、源码哈希和编译输入在 codegen 后捕获。训练“代码生成前”模型不得使用后者、代码生成改变的值或编译后的物理资源；前一快照也不是零设备副作用的离线规划 API。

初始准入先对原始头文件默认、新实验二进制默认以及用新请求重放默认结构逐项比较：完整结构、完整表、源码及编译输入/实际选项、编译资源，以及独立的全输出数值。若默认结构不能由均一 R 表示，记录 `unsupported_identity` 并保留原始/实验默认对照，不搜索该 workload；若可以表示但 replay 不一致，停止调查，不以例外规避。其余候选须通过实际资源、zero/impulse/constant/quarter-tone/random 等完整输出和输入保持检查才可进入计时。任何采样前选择、预算、准入、失败和保留原始证据由独立控制器处理。

## 构建

```bash
python3 test/plan_selection_study/expanded_search/prepare_headers.py \
  --source "$PWD/VkFFT/vkFFT" --output "$PWD/expanded_generated/include"
cmake -S test/plan_selection_study/expanded_search -B expanded_build \
  -DEXPANDED_GENERATED_ROOT="$PWD/expanded_generated" -DCUDAToolkit_ROOT=/path/to/cuda
cmake --build expanded_build --parallel 2
```

生成目录必须为空；保留 `expanded_provenance.json` 和基础 instrumentation provenance。仅运行 CPU 检查：使用独立构建目录并配置 `-DEXPANDED_CPU_ONLY=ON`，随后运行 `ctest --test-dir <目录> --output-on-failure`。本地 CPU 单测覆盖解析、状态重建、请求顺序、生成钩子、枚举范围与重复，但不验证真实 VkFFT/CUDA 编译或 GPU 数值。远端实际执行必须等待独立审查与受控准入。
