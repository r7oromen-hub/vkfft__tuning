# Direct candidate experiment (unvalidated prototype)

This creates a separate header tree and binary; upstream and the existing instrumentation sources/results are unchanged. It tests 1D, CUDA, FP32 C2C forward, out-of-place, power-of-two lengths, one upload, no Rader/Bluestein. The input and output layouts are the existing benchmark's contiguous layout. Exactly one benchmark candidate is allowed per process. Existing `aimThreads`, `registerBoost`, `useLUT`, and `registerBoostNonPow2` flags may be used, but compare runs with exactly the same requested flags. Other tuning flags are refused for injected candidates.

The new joint-search hook permits alternate interior factorizations with radix 2, 4, 8, 16 or 32 while retaining the baseline first and last stages and total product. Each ordinary stage must have a nonzero, radix-divisible entry in the actual scheduler register table; recalculated min/max registers must equal the scheduler values. When registerBoost exceeds one, the unchanged trailing switch stage is excluded from the min/max calculation, as in the scheduler. The FFT-thread dimension and scheduler-derived register envelope remain fixed. This is a restricted search, not all valid VkFFT decompositions or block dimensions. The hook runs after scheduling and before LUT generation and codegen. The batch-per-block injection replaces only the batch dimension in the axis splitter before the original groupedBatch/axisSwapped/localSize processing; it leaves the FFT thread dimension unchanged. It checks batch divisibility, dimensions, thread and shared-memory bounds, then requires exact final launch localSize. After initialization, the benchmark checks the actual stage sequence and localSize again, as well as compiled-function shared memory and thread limits, before any launch. The actual `stageRadix`, raw `localSize`, `axisBlock`, resource validation, descriptors, source hashes, and numerical validation are in the existing JSON record. A rejected candidate is an error, not a timing observation.

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

The build needs this directory, `test/plan_selection_study/instrumentation` (including `prepare_headers.py`, `vkfft_instrumented_benchmark.cpp`, `trace.hpp`, `cpu_reference.hpp`), and the original `VkFFT/vkFFT` include tree. Generated `direct_candidate_provenance.json` records header hashes. A CPU-only build uses `-DDIRECT_CPU_ONLY=ON`; it verifies parser, refactor preconditions, enumeration, strict anchors and source isolation, but does not compile the CUDA benchmark. This prototype has not been compiled against CUDA or run on a GPU in this session, so no performance or numerical correctness claim is made.

On this Windows host the MinGW compiler and its runtime live under `C:\msys64\ucrt64\bin`; an unrelated earlier DLL on `PATH` caused test exit `0xc0000139`. For a reproducible CPU-only build, use a temporary PowerShell process environment (this does not change the user or system `PATH`):

```powershell
$env:PATH = 'C:\msys64\ucrt64\bin;' + $env:PATH
cmake -S test/plan_selection_study/direct_candidates -B test/plan_selection_study/direct_candidates/cpu_build -DDIRECT_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=Release -G 'MinGW Makefiles'
cmake --build test/plan_selection_study/direct_candidates/cpu_build
ctest --test-dir test/plan_selection_study/direct_candidates/cpu_build --output-on-failure
```

The CPU test target now uses static GCC runtime linking on Windows with GNU to avoid this DLL conflict after rebuilding. Other compilers and the CUDA benchmark do not inherit that option.

## Joint search protocol

`controller/joint_gpu_guard.py` is the Linux fail-closed GPU guard for `controller/joint_controller.py`. The controller uses the exact same instrumented binary for its **requested default** control (`makeForwardPlanOnly=1`, no tuning options) and candidates. It records identity replay, calibrated graph repetitions, three paired default/default timing controls, every candidate request and precheck rejection, all prechecked requests by default, or a seeded subset when `--limit` is nonzero, numerical patterns, actual descriptors/source hashes, duplicate dispositions, paired screening, and independent paired confirmation of the screen winner. A timing ratio below one means the candidate is faster; screening picks the winner, so only the separate confirmation is an unbiased subsequent measurement, and neither establishes a global optimum or significance. Raw rows and guard records must be retained and inspected; no failure is a speed result. Explicit unsupported-length initialization and compiled-resource-limit rejections are recorded as rejected without timing; unexpected initialization, compilation, numerical or execution errors abort the experiment.

For each same-device default baseline with power-of-two length 2 through 8192, the enumerated scope is the ordered compositions of its interior radix exponent with {2,4,8,16,32}, fixed first/last stages and at most 32 stages, crossed with power-of-two divisors of batch up to 1024. FFT-thread dimension is inherited from the baseline; the block is predicted from its geometry. Candidates outside scheduler register envelope, thread limit or shared-memory bound are marked rejected before selection; others remain prospective until hook, compiler resources and numerical validation pass. Only a seeded subset is tested when a nonzero cap applies, and its complete manifest records the omitted requests. Actual plans are deduplicated by canonical descriptor, generated source hash, compiler input and options. It is not full joint enumeration of every possible CUDA block or VkFFT plan.

Run one workload per guarded invocation on a Linux CUDA machine after obtaining authorization; choose unique paths and the actual GPU UUID:

```sh
python test/plan_selection_study/direct_candidates/controller/joint_gpu_guard.py \
  --gpu GPU-UUID --log joint_guard.jsonl --incidents joint_incidents.jsonl \
  --script test/plan_selection_study/direct_candidates/controller/joint_controller.py -- \
  --exe build_direct_candidates/vkfft_direct_candidate --gpu GPU-UUID \
  --output joint_n1024_b16.jsonl --n 1024 --batch 16 --limit 0
```

The output is created exclusively, and this invocation requires a clean 30-second guard observation by default. The controller halts on an unexpected process or correctness failure. The prototype has only been tested on CPU; it has not yet established CUDA compilation, numerical correctness or performance for newly factored plans.
