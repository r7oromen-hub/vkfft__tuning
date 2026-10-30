# CUDA benchmark harness

The package is `test/plan_tuning`, with source `vkfft_plan_benchmark.cpp` and
CMake target/executable `vkfft_plan_benchmark`. The measured remote checkout is
now `/home/ygb/vkfft_plan_tuning_20261027`; its preserved measured executable is
`build/vkfft_plan_benchmark`. The run was checkpointed before migration, and the
executable was renamed without rebuilding. Its verified SHA-256 remains
`09dd69b59b8fe54b4d00f25029a85f058d371ce81155da27a3d084ae1e0b5eaf`.
Original source bundles and measurement records retain historical paths for
provenance. The active manifest's executable path was migrated only after hash
verification, with its original manifest preserved. Historical `build/` CMake
cache paths are stale; use a fresh build directory for future builds.

Build on Linux with CUDA Toolkit and CMake:

```sh
bash test/plan_tuning/build.sh -DCUDAToolkit_ROOT=/usr/local/cuda
CUDA_VISIBLE_DEVICES=GPU-SELECTED-UUID test/plan_tuning/build/vkfft_plan_benchmark --n 4096 --batch 256 --warmup 3 --repeats 15 --seed 42
```

The caller selects one approved GPU using `CUDA_VISIBLE_DEVICES`; the executable
uses only visible ordinal 0. To build the CPU self-test without CUDA, configure
with `-DPLAN_TUNING_CPU_ONLY=ON`. CUDA code is C++17 host code compiled with g++
and linked to cudart, CUDA driver, and NVRTC; nvcc is not required.

Tune a single candidate using `--aim-threads`, `--register-boost`, and `--use-lut`
(`-1` off, `0` automatic, `1` on). Omitting all tuning flags requests library
defaults. `--grid` emits the default and the 18 combinations of aimThreads
64/128/256, registerBoost 1/2/4, useLUT -1/1. JSON includes requested and resolved
configuration separately because the library may change or ignore a request.

Each stdout line is one JSON candidate result; VkFFT diagnostics go to stderr.
Exit 0 means all emitted candidates passed, 1 means at least one runtime or
correctness failure, and 2 means argument parsing failed. Errors include phase,
error text, and VkFFT status code if available. A correctness-failed candidate
retains timings for diagnosis but must not be selected as a valid winner.

`initialize_ms` is wall time for the complete `initializeVkFFT` call and a
following stream synchronization, including planning, code generation,
compilation, and plan resource setup. CUDA context
setup and CPU reference generation are outside this interval. It is not a
compiler-only measurement. Repeated processes may use CUDA's driver cache.

Execution uses an unnormalized complex FP32 in-place forward transform (`-1`),
with `makeForwardPlanOnly=1` by default. Each warmup and event sample first restores the
original deterministic input by H2D and synchronizes that same nonblocking stream.
Events bracket one full-batch forward on the VkFFT-configured stream. Resets,
transfers, warmups, CPU reference generation, and correctness comparison are
outside the event interval. The reset induces an input cache state; no cache
flush is claimed. This protocol avoids overflow from repeated unnormalized FFTs.
Warmup continues until both `--warmup` (default 3) transforms and
`--warmup-ms` (default 100) milliseconds of warmup wall time have elapsed.
The actual warmup iteration count and elapsed wall time are recorded. Every
warmup iteration includes restore, forward execution, and synchronization.
One event interval includes device idle gaps caused by CPU kernel submission,
which matters for very short FFTs. No shader source is printed or retained.

`--timing-mode direct` preserves this direct enqueue protocol (the default).
`--timing-mode graph` captures an external start-event record, the same
`VkFFTAppend` forward sequence, and an external stop-event record on the configured
stream. It instantiates that CUDA graph once, warms it up, then replays it once
per sample after the same outside-timing input restore. The timing events are
explicit graph nodes through `cudaEventRecordExternal`; their device timestamps
measure forward graph latency without CPU enqueue gaps between the kernels.
The stream is synchronized after every replay before querying elapsed time.
`graph_build_ms`, node count, kernel count, and event-record count are reported
separately; graph capture/instantiation are excluded from `initialize_ms` and
execution samples. The plan, device buffers, stream, and events remain alive with
stable addresses for every replay. Graph executable and graph are destroyed
before the plan or events. This measures the graph execution regime; compare it
separately from direct enqueue latency. CUDA Toolkit 12.x is the tested target.

For the main repeated-graph protocol, pass `--placement outofplace --timing-mode
graph --graph-repeats R`. Dedicated immutable input uses `isInputFormatted=1`,
`inputBuffer`, and `inputBufferSize`; `buffer` is the separate output/work buffer.
The graph contains R serialized forwards between its two external event nodes.
`raw_interval_ms_samples`/`raw_interval_ms_median` record the whole graph interval;
`exec_ms_samples`/`exec_ms_median` divide it by R and therefore report milliseconds
per full-batch forward. R defaults to 1 and is bounded to 1..2048. R>1 requires
out-of-place graph execution, which avoids transforming earlier FFT results.
The runner can calibrate one fixed R per workload, then use it for all candidates.
The input is restored before each graph replay; all R forwards within that replay
reread the same immutable input and overwrite the separate output. This repeated
access cache regime must be stated alongside results. Before graph construction,
one direct forward validates full-output correctness and bitwise input integrity;
the final timed graph's output and input integrity are validated again. JSON
records `single_forward_correctness`, `single_forward_input_unchanged`, and
`input_unchanged`. Primary input/output buffers require twice the workload bytes,
plus VkFFT's optional temporary storage. `warmup_actual` counts graph replays;
`warmup_forward_count` counts forwards including the R repetition factor.

The CPU reference independently implements negative-sign double-precision mixed
radix 2/3/5 Cooley-Tukey FFT. Only positive 2/3/5-smooth lengths are accepted.
The base input is complex pseudo-random dyadic noise generated by splitmix64;
each batch applies an exact dyadic scale and DC offset. These values are exactly
FP32-representable. By linearity, one CPU base FFT provides an independent
reference for every element of every batch. All output elements are checked,
including nonfinite detection. Acceptance uses `criterion_v2`: every batch must
be finite and have `L2(error)/(sqrt(N)*L2(input)) <= 1e-5` and
`Linf(error)/L2(input) <= 1e-4`. Zero input requires exactly zero output. Input
norms are computed from the exact deterministic FP32 input. By Parseval these
equal the relative output L2 error and maximum absolute error normalized by
reference RMS respectively. The engineering thresholds are scale invariant and
were chosen after pilot diagnostics; they are not theoretical error bounds.
Worst-batch metrics, batch indices, and failing-batch counts are reported.
The original pointwise rule `error <= 0.002 + 0.00005 * abs(reference)` remains
under `pointwise_passed` with its violation counts/details; `--atol` and `--rtol`
affect only those diagnostic checks. JSON also reports maximum absolute error, maximum
pointwise relative error (denominator floor 1e-30), global L2 relative error,
and maximum absolute error normalized by maximum reference magnitude. Near-zero
bins can make the pointwise relative maximum large without violating the
absolute-plus-relative diagnostic rule. `test_reference` validates the independent
FFT, exact FP32 input representation, and batch linearity against direct DFTs.
For diagnosis, `first_violations` contains at most the first five failures and
`worst_violations` at most the five largest error-to-allowance ratios. Each entry
records batch, frequency bin, flat index, expected and actual complex values,
reference magnitude, absolute error, allowed error, and their ratio. These
bounded diagnostics do not change the acceptance rule.

`--pattern random` is the main experiment input. Diagnostic patterns `zero`,
`impulse`, `constant`, and `quarter-tone` use exact dyadic batch scaling without
the random pattern's DC offsets. Their references are known analytically: zero;
a constant spectrum from impulse `(1,0.5)` at index zero; only DC from constant
input `(1,-0.5)`; and only bin N/4 from the repeating complex quarter-tone
`1,i,-1,-i`. Quarter-tone requires N divisible by four. An independent CPU FFT
first cross-checks the analytic reference, then GPU comparisons use the analytic
values. CPU tests include all patterns and negative checks for zero output on
nonzero input, permuted output, nonfinite output, zero-input leakage, and an
isolated error that passes global L2 but fails the peak criterion.

Structural features distinguish actual CUDA launch `localSize` from planner
`axisBlock` and include logical register counts, dynamic shared bytes, upload
counts, axis splits, stage radices, and upload execution order. Actual compiler
registers, static shared bytes, and local-memory bytes live in a separate
`compiled_diagnostics` array; these must not be used as precompilation model
features because they require a compiled CUDA kernel.
The source does not modify upstream VkFFT headers.
