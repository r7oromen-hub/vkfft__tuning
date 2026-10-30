# Original VkFFT headers versus joint default

Incremental package for the Linux `joint_source` tree. It builds a second executable from **the same generated joint harness source**, including its output poison and graph protocol, with untouched VkFFT headers first in the include search path and `VKFFT_STUDY_UNINSTRUMENTED=1`. It does not rebuild or replace the joint executable. Each three-way round compares original default, joint default, and an explicitly specified winner in the joint binary. The runner clears inherited direct-injection variables for every process and adds the requested values only for the winner. All three request `makeForwardPlanOnly=1` and no tuning flags.

The prepared joint source archive has the original `VkFFT/vkFFT` tree plus the generated harness from its prior joint build. Extract this incremental package at that root, then run from the root. The verifier checks all 42 original header files and key source inputs against the recorded joint manifest, plus the exact Linux-generated harness. It requires the existing build directory to be named `build_direct_candidates`.

```sh
python test/plan_selection_study/original_compare/verify_inputs.py
cmake -S test/plan_selection_study/original_compare -B build_original_compare \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DJOINT_BUILD="$PWD/build_direct_candidates" \
  -DVKFFT_INCLUDE_ROOT="$PWD/VkFFT/vkFFT" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build_original_compare --parallel 2 --verbose
sha256sum build_original_compare/vkfft_original_control build_direct_candidates/vkfft_direct_candidate
```

Before running, inspect `build_original_compare/compile_commands.json` and the generated dependency file under `build_original_compare/CMakeFiles/vkfft_original_control.dir/`: `VkFFT/vkFFT/vkFFT.h` and all transitive VkFFT headers must resolve under the original source tree, not `build_direct_candidates/generated/include/vkFFT`. That generated include directory is still necessary for `direct_candidate.hpp` and `vkfft_trace_source_metadata.hpp`. The direct helper is dormant when the runner clears its variables. The fresh CMake target uses C++17, release `-O3`, CUDA runtime/driver/NVRTC and the same Toolkit path as the recorded joint build (GNU 11.4.0, CUDA Toolkit 12.8); verify those versions against the target host's actual joint build. The hash verifier is specific to this joint bundle and fails closed on any differing source or generated harness. Preserve both executable hashes and compiler/dependency evidence with the results.

Read the `calibration_complete.graph_repeats` of the corresponding joint workload and use that same R below. Replace placeholders with the intended UUID and a new output basename. Guard requires Linux `/proc` and pidfd support; it selects the GPU, performs a 30-second idle check and monitors for foreign processes and total GPU memory over 2048 MiB. Use a separate guarded invocation for each N/batch pair.

```sh
GPU_UUID=GPU-REPLACE-WITH-FULL-UUID
N=256
BATCH=4096
R=512
OUT=results/original_compare_n256_b4096_01
python test/plan_selection_study/original_compare/original_compare_guard.py \
  --gpu "$GPU_UUID" --log "${OUT}.guard.jsonl" --incidents "${OUT}.incidents.jsonl" \
  --script test/plan_selection_study/original_compare/original_compare_controller.py -- \
  --joint-exe "$PWD/build_direct_candidates/vkfft_direct_candidate" \
  --joint-sha256 1d0e664fb21309ea6595f7aca72f0dfc812e0b99ae87adfc830d213b90904d2e \
  --original-exe "$PWD/build_original_compare/vkfft_original_control" \
  --gpu "$GPU_UUID" --n "$N" --batch "$BATCH" --graph-repeats "$R" \
  --winner-radix 16,16 --winner-group 1 --winner-block 16,1,1 \
  --output "${OUT}.jsonl" --rounds 10 --seed 42
```

The example uses the measured N=256, batch=4096 winner, its joint-calibrated R=512, and its recorded joint executable SHA-256. Use a winner verified for the exact target workload, not the example for another N/batch. The controller requires five identical pattern/seed numerical checks for all three paths, the same out-of-place poisoned-output protocol and full-array CPU checks, then performs randomized three-way execution measurements. Each timing uses 1 s warmup and 31 samples; every raw graph interval must be at least 0.5 ms. It compares complete actual `structural_features` and compiled resource attributes of the defaults on every round, and compares joint precompile canonical descriptor/source SHA-256/length with original post-initialization retained-source metadata. Winner shape and resource limits must match its explicit request; each variant's structure, source fingerprint and compiled resources must remain stable through the run. For every triple it records the original and winner structures, resource diagnostics and source hashes side by side. The output contains separate per-round joint-default/original-default and winner/original-default median ratios plus a completion record. Require clean completion in both output and guard log. An unguarded, structurally mismatched, invalid or contaminated result cannot produce completion. For an independent drift check, repeat on a separate clean run and assess pairwise ratios and GPU telemetry; this is not a formal estimate of all instrumentation effects.

Original headers provide no precompile snapshot, observed NVRTC options or initialization trace. The original output's actual compiler-options field is null; its expected option, inferred from the verified original compiler source, is checked against the joint observed option. The original harness sets `keepShaderCode=1` during initialization to retain and hash the generated source, then frees it and sets the flag to zero before execution. Joint uses `keepShaderCode=0`. Their source-retention and instrumentation work differ, so **do not compare initialization times**, and do not claim that original NVRTC options or precompile state were directly observed. CPU mocks: `python test/plan_selection_study/original_compare/test_original_compare.py -v`; they do not validate GPU, compiler, nvidia-smi or Linux pidfd operation.
