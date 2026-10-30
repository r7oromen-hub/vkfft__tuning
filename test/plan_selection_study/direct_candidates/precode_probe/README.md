# 代码生成前后的结构观察探针

本探针复制头文件到隔离构建目录，在 `shaderGen_FFT` 调用前记录一次结构，并在原有的 `VkFFT_CompileKernel` 前记录第二次。它用于研究两次记录之间的变化，不是脱离 GPU 的规划 API，也没有将规划与代码生成拆开。原始 `VkFFT/` 不做修改。

## 源码边界

在当前版本的 `vkFFT_Plan_FFT.h` 中，`VkFFT_AllocateLUT` 在第 236 行，axis block 拆分和 local size 设置分别在第 452、468–470 行，`shaderGen_FFT` 在第 768 行，编译在第 773 行。也就是说，本探针的前一快照已经获得所记录的 radix、block、local size 等当时的值，但它发生在 LUT 准备之后；启用 LUT 时可能已经分配和上传设备内存。当前 benchmark 的 describe 跳过编译和执行，仍运行 host 代码生成，范围限于 instrumentation README 中的 CUDA FP32 1D C2C forward。

不能把前一快照的全部值当作最终描述：`vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_SharedMemory.h` 的第 84、99、171 行在代码生成过程中计算 `usedSharedMemory`；`vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_ReadWrite.h` 第 39 行起选择 `readToRegisters` / `writeFromRegisters`。因此快照中的 `dynamic_shared_bytes`、这些读写标志可能依赖代码生成，其前值不保证是最终值。源码长度和哈希只有源码生成后可计算；编译后的寄存器、静态共享内存、function resource limits 等只能在编译及加载后查询。`registers_per_thread` 是规划中的逻辑值，不等于编译器分配的物理寄存器数。

当前代码逐轴、逐 upload 执行规划与代码生成/编译；若要实现“所有候选先规划，选中后才生成源码和编译”，需要另行抽取并冻结完整依赖，推迟或重构 LUT 等副作用，并明确处理代码生成导出的字段。现有快照不能证明这种重构已完成，也不能保证适用于其他维度、后端或变换类型。样本内前后相同的字段仅为观察，不是所有配置的保证。

## 构建与核对

在仓库根目录、空生成目录下执行：

```bash
python3 test/plan_selection_study/direct_candidates/precode_probe/prepare_probe.py \
  --source "$PWD/VkFFT/vkFFT" --output "$PWD/precode_generated/include"
cmake -S test/plan_selection_study/direct_candidates/precode_probe -B precode_build \
  -DPRECODE_GENERATED_ROOT="$PWD/precode_generated" -DCUDAToolkit_ROOT=/path/to/cuda
cmake --build precode_build --parallel 2
python3 test/plan_selection_study/direct_candidates/precode_probe/test_probe.py
```

只验证生成的 trace 代码及其快照行为时，可将上面的 configure 参数改为 `-DPRECODE_CPU_ONLY=ON`，再执行 `ctest --test-dir precode_build --output-on-failure`。此测试使用合成结构，不代表真实 VkFFT 或 CUDA 编译已经通过。Windows MinGW 若无法运行编译器，需将对应工具链的 `bin` 放在进程 `PATH` 前面。

生成器拒绝覆盖已有的非空输出目录。`precode_probe_hashes.json` 记录实验输出哈希；同时保留常规生成器的 provenance。探针和同源码生成的 direct 基线须在同一 GPU、同一软件栈，以相同参数运行 `--study-mode describe --placement outofplace`；清空直接候选环境变量，因为当前 describe 不支持注入直接候选。GPU 任务应与其他实验协调并通过该环境的 guard 调度。保存各自逐行 JSON 输出，并执行：

`describe_gpu_guard.py` 使用现有 `joint_gpu_guard.py` 的 `Guard` 类、PID 身份检查和竞争监控，只接受本目录的 `describe_controller.py`。实例命令如下，GPU UUID、文件路径须替换为实际值，且运行前应与其他实验协调：

```bash
python3 test/plan_selection_study/direct_candidates/precode_probe/describe_gpu_guard.py \
  --gpu GPU-UUID --log /path/guard.jsonl --incidents /path/incidents.jsonl \
  --probe /path/vkfft_precode_probe --baseline /path/vkfft_direct_candidate \
  --output /path/new-output-dir --case 256:16 --case 1024:16 --case 8192:16
```

每次单独注册、检查并监控子进程。`runs.jsonl` 保存带守卫标记的原始 stdout/stderr；配对成功后写入 `probe.jsonl`、`baseline.jsonl`、`comparison.json`。只有进程返回 0 且 guard 日志末尾为成功的 `guard_complete` 才接受整个批次；失败或竞争发生时，即使输出文件已存在，也须隔离，不可视为通过。运行前不要复用输出目录。控制器的模拟测试为 `python3 test/plan_selection_study/direct_candidates/precode_probe/test_describe_controller.py`；模拟测试不调用 GPU。

重新离线审核保存的逐行 JSON：

```bash
python3 test/plan_selection_study/direct_candidates/precode_probe/compare_snapshots.py \
  probe.jsonl --control baseline.jsonl --output comparison.json
```

比较器要求两组记录顺序相同、请求及 GPU/CUDA 版本一致、均成功且 describe 的 NVRTC/module/launch 计数为零；逐 upload 比较源码哈希、长度、编译输入及原始/规范结构。它逐字段报告探针中 pre-codegen 与 precompile 的变化。输出只证明所测请求的 describe 结果匹配，不证明编译后或数值执行等价。尚未运行远端时不能把这些检查写成通过。
