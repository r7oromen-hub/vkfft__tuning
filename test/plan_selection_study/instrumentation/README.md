# 独立初始化剖析与 describe 原型

本目录复制旧测量程序及独立 CPU 参考，以新可执行文件和生成的独立头文件树工作。原 `VkFFT/`、旧程序及历史结果均不修改。原始与覆盖后头文件哈希由生成器写入 `build/generated/header_provenance.json`；本目录来源见 `source_provenance.json`。

```bash
cmake -S test/plan_selection_study/instrumentation -B build_instrumented \
  -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
  -DVKFFT_INCLUDE_ROOT="$PWD/VkFFT/vkFFT"
cmake --build build_instrumented --parallel 2
ctest --test-dir build_instrumented --output-on-failure
```

构建产物为 `vkfft_instrumented_benchmark` 与 `vkfft_uninstrumented_control`。后者使用原始头文件，支持相同扩展参数的 execute 验证，明确不提供编译前快照/编译计数。GPU 运行必须由本轮 `gpu_guard.py` 和新 runner 调度；本文命令仅展示 harness 参数，不绕过保护直接执行 GPU。

## 模式与输出

- `--study-mode execute`（默认）：兼容原 harness 的数值与 graph 测量协议。增加 `--register-boost-nonpow2`、`--coalesced-memory`（0/16/32/64/128）、`--use-lut-4step`、`--swap-to-2stage`、`--swap-to-3stage`。`--disable-reorder-four-step` 只接受 0，保持自然输出顺序。
- `--study-mode describe`：仅限 smooth CUDA FP32 1D C2C forward、独立输入/输出。真实运行配置归一化、Scheduler、layout 与 host codegen，保留 LUT 计算/设备分配/H2D；在辅助 Rader/Bluestein 准备之前拒绝超出范围的计划。绕开 CompileKernel，并在 app 尾部 temp-buffer 物化前返回；不执行 forward。编译、模块加载、kernel launch 计数必须为 0。
- `--study-mode full-search-cost --candidate-file FILE.tsv --init-rounds 1`：在一个新进程/primary context 中按 TSV 顺序完整初始化并清理所有候选。
- `--study-mode describe-select-cost --candidate-file FILE.tsv --selected-id ID --init-rounds 1`：同一 context 中描述并清理全部候选，再用已冻结的原请求完整物化指定候选。指定赢家是已知输入，没有实现或计量模型推理；只能解释为 oracle-informed 组件成本。

cost 模式须 `--placement outofplace`。两条 pipeline 各自使用新进程/context，不能将两者标为共享同一 context 的一次执行。外部 runner 控制 pair、AB/BA 和候选 TSV 乱序；`--candidate-order forward` 保持该顺序。TSV 首行为列名，第一列必须是 `candidate_id`，其余列为 `aim_threads, register_boost, use_lut, register_boost_nonpow2, coalesced_memory, use_lut_4step, swap_to_2stage, swap_to_3stage`；可附 `disable_reorder_four_step=0`。0 表示使用对应默认值。重复 ID、未知列或越界值会拒绝。

execute/describe 各输出一条候选 JSON。cost 模式输出一条 `record_type=context_profile_summary`，含 `n,batch,gpu,status,mode,candidate_order_ids,selected_id,apps`。`apps` 各项包含候选/round/pipeline/role、NVRTC 首次/后续标记与 `result`；每个 result 保留完整阶段 trace 和 cleanup。所有候选复用同一组外部输入/输出及 stream，并验证当前 CUDA context 身份。每个 app 仍调用 cuInit/cudaSetDevice/cudaFree(nullptr) 和 GPU metadata 查询，这些重复 host 工作计入 pipeline wall，不叫纯规划成本。

## 时间如何相加

`init_trace.spans` 为嵌套区间，具有 `id,parent_id,stage,axis,upload,start_ns,end_ns,inclusive_ms,exclusive_ms`。exclusive 是 inclusive 减去**直系子区间** inclusive；相加 exclusive 可避免父子重计，不应将所有 inclusive 求和。覆盖 setConfig/device query、Scheduler、PlanAxis、LUT/Rader/resource、layout/split、host codegen、NVRTC、binary extraction、module load/getFunction/attribute 和独立 cleanup。

`initialize_ms` 是外部测量的初始化及 stream 同步墙钟；`initialize` span 是函数本体，同步另有 span。源码快照/hash 开销有独立 span，仍包含在被剖析初始化内。`init_trace.overhead` 每进程测量空 scope 平均与 clock-pair 中位开销，只作为观察值，不自动扣除。

所有 trace 记录在 GPU 执行前暂停；graph capture/replay 的 GPU event 区间不记录 CPU span。shared buffer/stream setup/cleanup 和 primary context bootstrap 在 cost summary 分开报告；`context_pipeline_wall_ms` 包含真实原型的候选、元数据与清理工作。`total_process_wall_ms` 覆盖 C++ main 到结果装配，排除 OS loader/进程退出 CUDA context teardown；外部 runner 另记录真正 subprocess wall。

## 描述与编译信息的边界

`precompile_kernels` 在 CompileKernel 入口（describe 模式则对应跳过点）捕获逐 upload 值描述、实际源码 SHA-256、长度及编译输入。`raw_descriptor` 保存该时点原状；`canonical_descriptor` 显式应用随后 axisBlock 的 dispatch 交换语义，同时保留实际 launch localSize。`descriptor_validation` 逐项比较初始化返回后的 canonical 字段。该一致性不等于跨模式已验证：S0 还必须按同请求对照 full 与 describe 的 canonical/source/compiler-input。

CUDA 编译属性另存 `compiled_diagnostics`。`compiler_input.options` 是受限 FP32 路径推导的完整 options，绑定原 CompileKernel 源码 SHA；full 模式在实际 NVRTC 调用前捕获 `actual_compiler_options` 并断言数组一致。describe 模式不会调用 NVRTC，因此 actual 字段为 null，需由对应 full 对照验收。缓存字段仅记录环境，不推断 driver/NVRTC 命中状态，不自动清 cache。

## S0 验收清单

在 guard 下对默认及扩展候选验证 N=1024 B1、49152 B3、81920 B3 和大长度；zero/impulse/constant/quarter-tone 与随机输入均按 criterion_v2 和输入逐位不变检查。对 full/describe 比较所有 upload 的 canonical、source hash、compiler input options；检查 describe 的三个计数为零、full NVRTC/module 次数等于 uploads、descriptor_validation 全部匹配。以新 uninstrumented control 做相同请求/seed/R 的成对随机顺序 execute 对照，检查数值、结构与编译资源一致；执行延迟只检查协议一致性，初始化差异包含 trace/hash 的真实开销。

原型未经过这些 GPU 验收前，不宣称支持全部矩阵或获得端到端搜索收益。错误、编译失败、范围拒绝和不一致均应保留，不能静默删除。
