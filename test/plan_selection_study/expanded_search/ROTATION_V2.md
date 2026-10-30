# CUDA 13 graph 依赖查询兼容修复

首次 rotation 的 CUDA 13.0 构建失败，原因是生成代码以旧的三参数形式调用 `cudaGraphNodeGetDependencies`，而该工具链头文件声明了四参数形式 `(node, dependencies, edgeData, count)`。失败发生在编译期，没有进入 rotation GPU 执行。旧生成器、旧源码清单和失败构建记录均保留；本修复生成独立的新源码树和独立构建目录。

`prepare_rotation_v2.py` 先在新的空目录调用旧生成器，随后仅替换两处依赖查询为 `rotation_graph_dependencies` 并加入 `rotation_cuda_compat.hpp`。包装器以 `CUDART_VERSION` 选择 CUDA 13 及以上的四参数调用，CUDA 12 调用原有三参数形式；对四参数形式传递空 `edgeData`，若查询会丢失非默认 edge 信息则保留 CUDA 错误并停止，不将图认定为已验证。图节点实参、依赖单链、轮换顺序、数值及预算约束均不变。CUDA 12 的[旧形式](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__GRAPH.html)和 CUDA 13 的[四参数形式](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-runtime-api/group__CUDART__GRAPH.html)由 NVIDIA runtime 文档说明。

```bash
python3 test/plan_selection_study/expanded_search/prepare_rotation_v2.py \
  --expanded-generated-root "$EXPANDED_GENERATED_ROOT" --output "$ROTATION_V2_DIR"
cmake -S "$ROTATION_V2_DIR" -B "$ROTATION_V2_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DROTATION_GENERATED_ROOT="$ROTATION_V2_DIR" \
  -DEXPANDED_GENERATED_ROOT="$EXPANDED_GENERATED_ROOT" \
  -DVKFFT_INCLUDE_ROOT="$PWD/VkFFT/vkFFT" \
  -DINSTRUMENTATION_ROOT="$PWD/test/plan_selection_study/instrumentation" \
  -DCUDAToolkit_ROOT=/path/to/cuda -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$ROTATION_V2_BUILD" --parallel 2
```

本机 CPU 回归：以 `rotation_compat_cpu` 为 CMake source，构建后执行 CTest。两个 C++ mock 分别以 `CUDART_VERSION=12060`、`13000` 声明对应签名并验证参数和错误返回传播；Python 测试确认新树两个调用均替换、旧树字节未变、v2 provenance 记录生成代码及两个版本的生成器哈希。这些测试不替代 CUDA 13 真实编译，也不证明图参数可查询或 GPU 数值正确。真实重建、guard 和设备执行须由实验代理使用相同 extension 预算的独立 v2 runner 调度；失败仍记录未覆盖。

`rotation_runner_v2.py` 保留旧 runner 的两会话 gate、同一设备锁和持久 extension 预算，额外要求 `--v1-failure-record`、`--v1-generated-source`、`--v1-provenance`。失败记录须是原预算账本中 `rotation_build` 的失败日志原路径；runner 复核错误涉及本次 CUDA13 调用、日志退出码与账本一致、原生成源码有两处旧调用且哈希与 v1 provenance 一致，随后在 `rotation_repair_source_freeze` 记录三者哈希。新的生成、构建和 CPU 目录分别为 `rotation_generated_repair_v2`、`rotation_build_repair_v2` 和 `rotation_cpu_build_repair_v2`，生成和构建日志另存 `rotation_repair_build_*`，不会覆盖旧目录。

该 runner 保持七个预算化构建步骤，其中独立 `rotation_compat_cpu` 项目同时运行原池/原生成测试与新增 CUDA 12/13 mock 和 v2 生成测试。完成后记录 `rotation_repair_build_complete` 和独立 guard launch/exit checkpoint，并继续使用同一两会话 gate、成对控制器与其原有未覆盖处理。若剩余预算不足、失败证据无法与账本关联、真实构建失败或图参数不可验证，不能启动或声称成功。
