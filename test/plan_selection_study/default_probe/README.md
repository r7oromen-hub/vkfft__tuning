# Guarded VkFFT default structure probe

This is an incremental package for the existing `direct_candidates_guarded_v3_20261028.tar.gz` source tree and its CUDA-built `vkfft_direct_candidate`. Extract it at that tree's root. It does not change the existing controller. The probe executes nine independent workloads: N 256, 1024, 8192 crossed with batch 1, 16, 4096. It requests out-of-place FP32 C2C forward and omits every tuning flag; the runner also clears all three inherited `VKFFT_DIRECT_*` environment variables. The hardware's own VkFFT defaults are recorded, not assumed from historical runs.

Build the v3 binary using the existing direct-candidate README. Verify the intended full GPU UUID and obtain a quiet device before running. This command uses the guard's default 30-second idle precheck and its one-second monitoring, foreign-process and 2048 MiB protections. It creates new exclusive output files and stops on the first failure. Use a fresh results directory on each run:

```sh
GPU_UUID=GPU-REPLACE-WITH-FULL-UUID
OUT=results/default_probe_01
mkdir -p "$OUT"
for N in 256 1024 8192; do
  for B in 1 16 4096; do
    python test/plan_selection_study/default_probe/default_gpu_guard.py \
      --gpu "$GPU_UUID" --log "$OUT/n${N}_b${B}.guard.jsonl" \
      --incidents "$OUT/n${N}_b${B}.incidents.jsonl" \
      --script test/plan_selection_study/default_probe/default_controller.py -- \
      --exe "$PWD/build_direct_candidates/vkfft_direct_candidate" --gpu "$GPU_UUID" \
      --n "$N" --batch "$B" --output "$OUT/n${N}_b${B}.jsonl" || exit 1
  done
done
```

Each row includes the complete original benchmark JSON and stdout/stderr, precompile raw and canonical descriptors, SHA-256 and byte length of the generated source, compiler input and actual options, actual post-initialization `localSize`, all-output independent CPU reference checks and input integrity, executable hash, and guard IDs. The existing instrumentation records the source hash and size, **not the source text**. The probe requires one upload, matching descriptors and compiler options, valid source hash, matching actual and precompile stages/localSize, complete numerical checks, the RTX 4090 name and exact requested UUID. Failures produce a failed row or guard incident and no completion row. Validate that all nine controller completion rows and all nine clean guard completion records exist before using the records. The single R=1 sample is only for functional validation; it is not a performance comparison or calibrated timing result.

CPU mocks: `python test/plan_selection_study/default_probe/test_default_probe.py -v`. They do not test CUDA, GPU correctness, or Linux nvidia-smi/pidfd behavior. The existing guard itself remains separately tested in its original location. No GPU execution or remote connection was made while preparing this package.
