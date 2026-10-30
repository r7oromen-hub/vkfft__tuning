# Guarded radix acceptance controller

This controller is an experimental guard copy and has only CPU simulation coverage. Build the CUDA binary as in the parent README. From the extracted archive root, after setting `GPU_UUID` to the full intended GPU UUID:

```sh
python test/plan_selection_study/direct_candidates/controller/direct_gpu_guard.py \
  --gpu "$GPU_UUID" --log results/direct_guard.jsonl --incidents results/direct_incidents.jsonl \
  --idle-seconds 30 --script test/plan_selection_study/direct_candidates/controller/direct_controller.py -- \
  --exe "$PWD/build_direct_candidates/vkfft_direct_candidate" --gpu "$GPU_UUID" \
  --case n16384 --output results/direct_n16384.jsonl --rounds 10 --graph-repeats 256
```

For the second case use `--case n32768` and distinct output and guard/incident paths. Output is created exclusively; an existing file is never overwritten. The GPU is selected by the guard using the UUID. The controller's only subprocess entry is the adjacent `run_suite.py`, wrapped by the copied guard. It saves complete process stdout/stderr, command, request, guard identifiers and process status. Baseline and identity checks precede correctness on five patterns, and paired randomized timing only begins if all pass. Historical stage sequences must match the actual baseline exactly; infeasible/incorrect candidates halt the run. Any raw interval under 0.5 ms halts timing. All binaries use out-of-place immutable input; the final graph output and the standalone validation output are poisoned with FP32 NaNs before execution, outside event timing. Graph interval covers 256 forwards and reported forward latency divides raw interval by 256.

CPU-only simulation: `python test/plan_selection_study/direct_candidates/controller/test_controller.py -v`. This does not test nvidia-smi, pidfds, CUDA, or GPU correctness. Block/group-size scans are not implemented by this controller.
