# 超 L2 缓冲池轮换原型

此变体从冻结的 `expanded_benchmark.cpp` 生成独立程序，不改动搜索或准入可执行文件。它构建两个二进制：使用扩展头的 `vkfft_rotation_expanded` 和优先使用原始 VkFFT 头的 `vkfft_rotation_original`。两者须在同一 workload、GPU、seed、graph 重复数和轮换池参数下随机配对；原始程序拒绝所有 `VKFFT_EXPANDED_*`，扩展程序接受已准入的正常请求。对照时还须与对应热缓存程序核对 GPU 结构、源码和编译选项，保证核心计划一致。

每个设备输入/输出分别单独分配，同一 slot 使用一对缓冲区；每个 slot 的输入由相同随机基准乘以不同的确定性因子生成，且程序检查各 slot 的代表值在 FP32 中互异。设单缓冲区为 `bytes`，显卡报告 L2 为 `L2`：选择最小二次幂 `K≥2`，要求 `K*bytes > L2`，所以输入池和输出池各自都大于 L2、合计严格大于两倍 L2。总池上限为 1280 MiB，并检查设备空闲显存；要求 graph 重复数 R 为二次幂、`K≤R≤2048` 且 `R%K=0`。不能满足时明确报 `rotation_unsupported_pool_or_graph_size` 或显存不足；不要把未覆盖工作负载当作失败或支持。

程序通过稳定数组元素地址向 VkFFT 传递独立 `VkFFTLaunchParams.inputBuffer` 和 `buffer`，逐节点 capture slot `i % K`。它从真实 CUDA graph 每个 kernel 节点读取参数，验证首两个参数分别是正确的一对设备地址，再检查各地址次数、节点依赖单链和完整轮换顺序；若驱动不提供可读的 `kernelParams`、图节点含非 kernel/event 或任何验证失败则拒绝。原始数组分配还检查设备地址唯一、不重叠。此检查须在真实设备上通过；CPU 测试并不验证 CUDA graph API。

全部输入 H2D 与初始输出 NaN 填充在图捕获/事件计时之外。首次 graph 回放后，逐 slot 完整校验输入逐位不变和全部输出；暖机之后采集的计时样本之间不插入 H2D、poison 或 D2H。所有计时完成后单独 poison 全部输出，重放一次相同 graph，再次逐 slot 校验输入及全部输出。`rotation_first_graph_interval_ms` 是 H2D/poison 后首次回放，`rotation_final_validation_graph_interval_ms` 是计时后的独立验证回放；它们不并入 `exec_ms_samples`。独立验证回放证明该次完整回放正确，不能证明每个计时回放都被逐次读回。

输出包括 `rotation_pool_slots`、`rotation_input_pool_bytes`、`rotation_output_pool_bytes`、`rotation_pool_total_bytes`、`rotation_device_l2_bytes`、`rotation_graph_kernel_arguments_and_order_verified`、`rotation_distinct_input_representatives_verified`、`rotation_initial_validation` 和 `rotation_final_validation`。每项验证含全部 slot 的输入保持、完整输出检查及误差。池规模超过 L2 只说明轮换工作集大于容量，并不证明每次访问都 miss，首轮也不应称为冷缓存。要保留采样间干扰和 graph 节点影响的局限。

## 构建

以下操作须在设备预留和阶段预算已准许时由实验代理执行；先保存并验证冻结 expanded 源/头及原始头的哈希。`ROTATION_DIR` 必须是独立空目录，不能位于冻结生成目录内：

```bash
python3 test/plan_selection_study/expanded_search/prepare_rotation.py \
  --expanded-generated-root "$EXPANDED_GENERATED_ROOT" --output "$ROTATION_DIR"
cmake -S "$ROTATION_DIR" -B "$ROTATION_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DROTATION_GENERATED_ROOT="$ROTATION_DIR" \
  -DEXPANDED_GENERATED_ROOT="$EXPANDED_GENERATED_ROOT" \
  -DVKFFT_INCLUDE_ROOT="$PWD/VkFFT/vkFFT" \
  -DINSTRUMENTATION_ROOT="$PWD/test/plan_selection_study/instrumentation" \
  -DCUDAToolkit_ROOT=/path/to/cuda -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$ROTATION_BUILD" --parallel 2
```

保留 `rotation_provenance.json`、编译命令和依赖，独立检查原始头路径先于生成头且依赖不含生成的 VkFFT 头。每个二进制只能通过设备守卫运行，指定 `--study-mode execute --placement outofplace --timing-mode graph --pattern random --graph-repeats R` 及相同 N、batch、seed、暖机和样本数；扩展候选另设三个 `VKFFT_EXPANDED_*`。预算须覆盖编译、上下文、图构建、全池初始化、全池两次校验、守卫空闲和计时，不能只计 graph event 区间。

本地 CPU 测试：`cmake -S test/plan_selection_study/expanded_search/rotation_cpu -B <新目录>`，再构建并执行 `ctest --test-dir <新目录> --output-on-failure`。目前这些测试只覆盖池尺寸、slot 序列及生成结构；在实际 CUDA 编译和独立校验通过前，超 L2 条件标为未验证。

## 受预算控制的执行

`rotation_runner.py` 是唯一计划中的设备入口：它要求预算状态已进入 extension、独占设备锁以及 `make_two_session_gate.py` 产生的 `expanded-two-session-gate/v1`。其输入包括确认 plan、研究预算 protocol、first gate、两次确认的原始 rows/guard/incidents/不可变 budget snapshot、两份独立 audit、两会话组合 audit 及 two-session gate；这些文件和两份热缓存二进制必须与 gate 的 SHA-256 逐项匹配。控制器重新运行两个确认审计并复算两会话结果，只对 `confirmed_under_preregistered_rule` 且与 plan 的完整请求和签名一致的候选执行。若没有候选，账本记录未覆盖并且不构建新二进制；预算不足、pool 不支持、图校验或数值失败均不形成性能结论。

运行示意（所有尖括号都应替换成实际文件）：

```bash
python3 test/plan_selection_study/expanded_search/rotation_runner.py \
  --state <预算状态目录> --results <结果目录> --expanded-generated-root <已确认构建的生成目录> \
  --plan <确认plan> --protocol <冻结预算protocol> --gate <two-session-gate> \
  --first-gate <first-gate> --first-audit <first-audit> --second-audit <second-audit> --combined-audit <two-session-audit> \
  --first-rows <first-rows> --first-guard <first-guard> --first-incidents <first-incidents> --first-budget <first-budget-snapshot> \
  --second-rows <second-rows> --second-guard <second-guard> --second-incidents <second-incidents> --second-budget <second-budget-snapshot>
```

runner 对证据核验、源/头文件树哈希、生成、构建、CPU 检查及每个配对分别办理预算许可并记账；在相同锁下调用专用 pidfd guard。它保留构建 stdout/stderr、源树和可执行文件哈希、原始头的编译依赖，以及执行/guard/incident JSONL。控制器先用原始头默认校准 graph 原始区间至少 2 ms，再对每个候选进行 10 轮确定性种子随机顺序配对；每侧 31 个样本、实际暖机至少 5 次且 1 秒、各原始区间至少 0.5 ms。每次进程都核验与热缓存确认使用的计划/源码/编译资源签名一致，并检查两次全池验证及 graph 节点实参、序列。单次 rotation 实验是补充机制证据，不替代热缓存两会话确认。
