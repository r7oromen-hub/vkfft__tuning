# VkFFT 计划调优实验

本目录为独立 CUDA/NVRTC 基准与分析脚本，原始 VkFFT 库及头文件保持不变。上游为 VkFFT 1.3.4，提交 `066a17c17068c0f11c9298d848c2976c71fad1c1`。主实验测试 24 个仅含 2/3/5 因子的长度，每个长度包含单 batch 与约 256 MiB 输入的大 batch，共 48 个工作负载；每个测试库默认参数及 18 组 `aimThreads={64,128,256}`、`registerBoost={1,2,4}`、`useLUT={-1,1}` 组合。请求参数与库实际采用参数分别记录。

运行状态：按用户要求停止修复和追加实验，阶段结论见 [plan_selection_findings_zh.md](plan_selection_findings_zh.md)。原始初筛 912 项均通过数值检查，但末段出现共享 GPU 竞争，部分等效配置的耗时发生数倍变化。原始 `full_outofplace_graph_v2` 和 `screen_complete_snapshot` 保留为原始记录与事故诊断，不能直接用其最大加速比或模型指标作为有效收益结论。已按独立遥测窗口标记 61 条初筛记录，详见 [repair_decision_v1.json](results/anomaly_review/repair_decision_v1.json)；未执行修复，三轮确认未完成。停止点见 [RUN_STATUS.json](RUN_STATUS.json)。

命名迁移已完成：本地目录为 `test/plan_tuning`；远端根目录为 `/home/ygb/vkfft_plan_tuning_20261027`，源码为 `test/plan_tuning/vkfft_plan_benchmark.cpp`，测量程序为 `build/vkfft_plan_benchmark`。CMake 项目与目标也已采用 `vkfft_plan_benchmark`。远端主实验在 116 条记录的检查点停止后迁移，测量程序仅重命名，未重新编译，迁移前后 SHA-256 相同。迁移工具保留原 manifest，并在核验程序一致后仅更新活动 manifest 的 executable 路径；原始测量记录不改写，历史记录及 [source_manifest.json](source_manifest.json) 中的旧路径、源码名和校验值保留为溯源信息。运行状态见 [RUN_STATUS.json](RUN_STATUS.json)。以下新构建/复现命令使用独立构建与输出目录，不覆盖已有测量。

## 构建与验证

以下 Linux 命令均从源码根目录执行。需要 CMake、支持 C++17 的 g++、CUDA Toolkit；本基准不需要用 nvcc 编译。FFTW 存在时会额外构建大长度参考验证。

远端历史 `build/` 的 CMake 缓存包含迁移前路径，继续保留测量程序即可；未来重新构建使用新的 `build_plan_tuning/`，不复用该缓存。

```bash
cmake -S test/plan_tuning -B build_plan_tuning -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda-12.6
cmake --build build_plan_tuning --parallel 2
ctest --test-dir build_plan_tuning --output-on-failure
```

GPU 由调用者明确选择；程序仅使用可见设备编号 0。此次物理 GPU1 为 RTX 4090，CUDA 12.6。选择此次设备后执行独立的解析输入复现验证：

```bash
export CUDA_VISIBLE_DEVICES=GPU-fb86b36f-0a9f-6cbd-a36b-03b9073f4786
python3 test/plan_tuning/run_probes.py --harness ./build_plan_tuning/vkfft_plan_benchmark --output-dir ./results/analytic_probes_reproduction
```

解析验证使用 zero、impulse、constant、quarter-tone 四种输入，长度 256、49152、81920、1048576，batch 为 1 与 3，默认及 `a256_r4_l1` 两种配置，共 64 项。采用独立输入/输出、graph R=4，检查单次与重复执行的数值结果及输入逐位不变；这些验证不参与性能统计。该脚本要求新的输出目录。

CPU 参考包含直接 DFT 对照、解析输入交叉验证及错误输出的负测试；可选 FFTW 对照覆盖上述四个长度。Python 分析测试可在安装分析依赖后运行：

```bash
python3 -m unittest discover -s test/plan_tuning -p test_python.py
```

## 主实验与断点续跑

```bash
python3 test/plan_tuning/run_suite.py \
  --exe ./build_plan_tuning/vkfft_plan_benchmark --output ./results/reproduction_plan_tuning \
  --preset full --phase all --placement outofplace --timing-mode graph \
  --graph-repeats 0 --graph-target-ms 0.5 --graph-max-repeats 2048 \
  --max-input-mib 256 --seed 20261027 \
  --warmup-ms 1000 --warmup 5 --repeats 31 \
  --confirm-warmup 5 --confirm-repeats 101 --confirm-rounds 3 \
  --source-commit 066a17c17068c0f11c9298d848c2976c71fad1c1
```

`--graph-repeats 0` 启用按工作负载校准：先运行默认配置 R=1，再验证并调整 R，使默认配置的 graph 原始区间尽量达到 0.5 ms，上限 2048。每个工作负载确定的 R 随后固定用于所有候选。目标时长只针对校准时的默认配置，后续候选或重复测量不保证仍达到目标。

筛选顺序随机化；每项有 31 个采样。选中的候选与默认配置及可用负对照另做 3 轮独立确认，每轮 101 个采样，并使用独立的确定性输入种子。预热同时满足至少 5 次执行及 1000 ms 墙钟时长。筛选中的最佳值存在选择偏差，结论应优先使用独立确认结果。

中断后使用**完全相同的命令、程序、环境与输出目录**续跑，成功记录会跳过。逐条 JSONL 写入并落盘，末尾不完整记录可恢复；`manifest.json` 或已校准 R 不一致会拒绝续跑。已有失败项默认保留，处理原因后加 `--retry-failures` 重试。也可用相同参数分开执行 `--phase screen`、`--phase confirm`；新协议或新程序应使用新的输出目录。

## 计时口径与数值门槛

主实验是 FP32 复数、未归一化正向 FFT，`VkFFTAppend(...,-1,...)`，仅构造正向计划。输入与输出/工作缓冲分离；每个采样前在同一 stream 上恢复输入并同步，恢复不计入事件时间。graph 内串行执行 R 次读取同一不可变输入的 forward，前后为显式 external event 节点；核数量必须等于 `R * numAxisUploads`。记录整个区间 `raw_interval_ms_samples`，除以 R 得到 `exec_ms_samples`，单位是一次完整 batch forward 的毫秒数。

这是重复读取不可变输入、覆盖独立输出的 graph 执行与缓存状态；不进行 cache flush。短 FFT 的 CPU 提交间隙及单次事件量化问题由 graph 和重复区间缓解。输入传输、graph 构造、预热和数值检查均在计时区间外。保留的 `inplace` / `direct` 模式仅用于单独诊断，不能混合汇总为主协议结果。

`initialize_ms` 包含完整 `initializeVkFFT` 及随后 stream 同步，即规划、代码生成、编译和计划资源准备；它不是“纯编译时间”。默认配置也完整初始化编译，再额外构造应用层 graph；这部分另记 `graph_build_ms`，未并入初始化或执行采样。CUDA 上下文启动及 CPU 参考生成也不在 `initialize_ms` 内。

数值验收统一采用 **criterion_v2**：每个 batch 必须全部有限，且 `L2(error)/(sqrt(N)*L2(input)) <= 1e-5`、`Linf(error)/L2(input) <= 1e-4`；零输入要求输出严格为零。两者由 Parseval 分别等价于输出相对 L2 误差和按参考 RMS 归一化的最大绝对误差。阈值是在 pilot 诊断后统一确定的 FP32 工程标准，具有尺度不变性，不是理论误差界。原逐点绝对/相对门槛保留为 `pointwise_passed` 与有界失败详情，不再作为主验收标准。

随机输入使用可精确表示的 dyadic 复数与 batch 缩放/DC 偏移，独立 double 混合基 CPU FFT 加线性关系覆盖所有输出元素。解析模式先交叉检查 CPU FFT，再用已知解析值验证 GPU。独立输入还会在单次 forward 及 R 次重复后逐位检查未改变。

## 原始数据与分析

远端主输出为 `results/full_outofplace_graph_v2/`；下载镜像位于本地 `test/plan_tuning/results/`。各目录保持独立：`results/pilot/` 为旧协议诊断，`results/controls_outofplace_graph_v2/` 为 100 ms 预热负对照，`results/controls_hot_1024/` 与 `results/controls_hot_81920/` 为 1000 ms 预热复查，`results/analytic_probes/` 为解析数值验证。旧 pilot 不与主实验合并，也不引用其筛选最优值作为最终收益。

主目录中的 `manifest.json` 记录参数、设备选择与二进制 SHA-256；`calibration.jsonl`、`calibration_verification.jsonl`、`calibration_summary.json`、`graph_repeats.json` 保存 R 的选择及验证；`measurements.jsonl` 保存原始样本、初始化/graph 时间、请求和实际配置、计划结构、独立编译诊断、数值检查及设备版本；`selected.json` 和 `negative_controls.json` 保存确认阶段的选择。历史源码校验见 [source_manifest.json](source_manifest.json)，此次远端测量二进制 SHA-256 为 `09dd69b59b8fe54b4d00f25029a85f058d371ce81155da27a3d084ae1e0b5eaf`；重命名后的 `build/vkfft_plan_benchmark` 仍为该相同文件内容，新构建应使用其自身 manifest 中的新校验值。

GPU 运行端脚本仅需 Python 标准库。分析可在本地完成，需要 NumPy；模型分析和绘图分别使用 scikit-learn 与 Matplotlib：

```bash
python3 -m pip install numpy scikit-learn matplotlib
python3 test/plan_tuning/analyze.py results/reproduction_plan_tuning
```

分析此次远端测量的本地下载镜像则使用：

```bash
python3 test/plan_tuning/analyze.py test/plan_tuning/results/full_outofplace_graph_v2
```

分析默认写入输入目录的 `analysis/`，包含 `summary.json`、`results_zh.md`、候选排名、结构分组、独立确认、负对照、模型表格及图片。`--skip-models` 可跳过模型分析。结构相同表示记录的描述符相同，不证明生成代码完全相同；实际 CUDA 寄存器等编译后诊断与结构特征分开保存。

本实验测量参数、计划结构及执行表现之间的关系。结构特征当前来自已完成初始化的计划；没有实现或测量 plan-only、跳过编译、在线预测端到端开销，因此不声称节省规划/编译成本或获得 plan-only 收益。现阶段证据及限制已写入 [plan_selection_findings_zh.md](plan_selection_findings_zh.md)。更详细的字段与实现说明见 [harness_notes.md](harness_notes.md)。当前实验无需修改 `initializeVkFFT`；未来延迟编译的源码边界与设计见 [planning_interface_design.md](planning_interface_design.md)。
