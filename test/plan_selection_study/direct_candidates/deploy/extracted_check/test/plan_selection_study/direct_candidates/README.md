# Direct candidate experiment (unvalidated prototype)

This creates a separate header tree and binary; upstream and the existing instrumentation sources/results are unchanged. It tests 1D, CUDA, FP32 C2C forward, out-of-place, power-of-two lengths, one upload, no Rader/Bluestein. The input and output layouts are the existing benchmark's contiguous layout. Exactly one benchmark candidate is allowed per process. Existing `aimThreads`, `registerBoost`, `useLUT`, and `registerBoostNonPow2` flags may be used, but compare runs with exactly the same requested flags. Other tuning flags are refused for injected candidates.

A requested radix sequence must be a **permutation of the scheduler's actual sequence**, with the first and last stage held fixed. The scheduler's register calculations and special first/last stage remain intact; this does not explore alternate factorizations. The hook runs after scheduling and before LUT generation and codegen. The batch-per-block injection replaces only the batch dimension in the axis splitter before the original groupedBatch/axisSwapped/localSize processing; it leaves the FFT thread dimension unchanged. It checks batch divisibility, dimensions, thread and shared-memory bounds, then requires exact final launch localSize. After initialization, the benchmark checks the actual stage sequence and localSize again, as well as compiled-function shared memory and thread limits, before any launch. The actual `stageRadix`, raw `localSize`, `axisBlock`, resource validation, descriptors, source hashes, and numerical validation are in the existing JSON record. A rejected candidate is an error, not a timing observation.

All three environment variables must be present together for injection:

- `VKFFT_DIRECT_RADIX`: comma-separated full actual sequence, e.g. `32,8,32,2`.
- `VKFFT_DIRECT_BATCH`: positive number of FFTs grouped per block, e.g. `1`.
- `VKFFT_DIRECT_BLOCK`: exact final CUDA launch block dimensions, comma-separated, e.g. `128,1,1`; derive this from the target device and never assume the example is valid.

When all are absent, the binary is the control path. The variables must be isolated per process and cleared for control runs. Identity replay uses the baseline's own stageRadix, groupedBatch and raw localSize, followed by exact comparisons of precompile canonical descriptors, source hashes, and compiler options. A rejected identity, mismatch, failed correctness, unsafe resource use, GPU error or failed guard invalidates the corresponding comparison. Validate every nonidentity candidate with the benchmark's independent CPU reference, input-unchanged check and varied patterns before using timing. Warmups, paired/randomized order and the existing GPU guard apply; this directory does not start or schedule GPU work.

The old data contains N=16384, `--aim-threads 128 --register-boost 4 --use-lut 1 --register-boost-nonpow2 1`, with `[32,32,8,2]`, and N=32768 with `[32,32,8,4]`. If the target's same-config control produces these exact single-upload stages, test respectively `[32,8,32,2]` and `[32,8,32,4]` with batch-per-block 1 and the observed exact block. They are historical observations, not guarantees for another GPU. These sequences do not demonstrate alternate radix factorizations; many other lengths have no nontrivial permitted order. Test group values 1, 2, 4 separately on smaller transforms and batches divisible by each group, retaining the exact control radix and applying all checks. A requested group may be infeasible; e.g. large transforms may exhaust shared memory at group 2.

Build on an authorized CUDA machine:

```sh
cmake -S test/plan_selection_study/direct_candidates -B build_direct_candidates \
  -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DVKFFT_INCLUDE_ROOT="$PWD/VkFFT/vkFFT"
cmake --build build_direct_candidates --parallel 2
ctest --test-dir build_direct_candidates --output-on-failure
```

The build needs this directory, `test/plan_selection_study/instrumentation` (including `prepare_headers.py`, `vkfft_instrumented_benchmark.cpp`, `trace.hpp`, `cpu_reference.hpp`), and the original `VkFFT/vkFFT` include tree. Generated `direct_candidate_provenance.json` records header hashes. A CPU-only build uses `-DDIRECT_CPU_ONLY=ON`; it verifies parser, permutation, strict anchors and source isolation, but does not compile the CUDA benchmark. This prototype has not been compiled against CUDA or run on a GPU in this session, so no performance or numerical correctness claim is made.

On this Windows host the MinGW compiler and its runtime live under `C:\msys64\ucrt64\bin`; an unrelated earlier DLL on `PATH` caused test exit `0xc0000139`. For a reproducible CPU-only build, use a temporary PowerShell process environment (this does not change the user or system `PATH`):

```powershell
$env:PATH = 'C:\msys64\ucrt64\bin;' + $env:PATH
cmake -S test/plan_selection_study/direct_candidates -B test/plan_selection_study/direct_candidates/cpu_build -DDIRECT_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=Release -G 'MinGW Makefiles'
cmake --build test/plan_selection_study/direct_candidates/cpu_build
ctest --test-dir test/plan_selection_study/direct_candidates/cpu_build --output-on-failure
```

The CPU test target now uses static GCC runtime linking on Windows with GNU to avoid this DLL conflict after rebuilding. Other compilers and the CUDA benchmark do not inherit that option.
