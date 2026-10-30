# Fixed-radix block-group controller increment

Requires the separately deployed `direct_candidates_guarded_v3_20261028.tar.gz` with SHA-256 `e94eac5fd7679ff2ea539ccd5e38d14fda5190252d17458f1397467a7c17266f`. Extract this increment at the same archive root, without overwriting existing files. The source, build and binary remain the v3 version.

For one isolated group, with `GPU_UUID` set to the intended full UUID:

```sh
python test/plan_selection_study/direct_candidates/controller/direct_block_gpu_guard.py \
 --gpu "$GPU_UUID" --log results/block_n256_g2_guard.jsonl --incidents results/block_n256_g2_incidents.jsonl \
 --idle-seconds 30 --script test/plan_selection_study/direct_candidates/controller/direct_block_controller.py -- \
 --exe "$PWD/build_direct_candidates/vkfft_direct_candidate" --gpu "$GPU_UUID" \
 --case n256 --batch 16 --group 2 --output results/block_n256_g2.jsonl --rounds 10 --graph-repeats 256
```

Use `--case n256` or `n1024` and `--group 1`, `2`, or `4`, with distinct paths and separate guarded invocations. The group must divide batch. The control uses the same aimThreads=128, registerBoost=4, useLUT=1, registerBoostNonPow2=1 configuration for every candidate and control in a run.

The controller observes the baseline's raw `axisSwapped`, `localSize`, `groupedBatch`, FFT length and resolved shared memory. For the restricted single-upload power-of-two path it predicts the exact launch shape from the splitter's swap condition. This prediction is not proof of feasibility or correct layout: the generated hook rejects a launch block mismatch before codegen; the controller independently verifies the actual radix, grouped batch, raw local size and unchanged FFT thread dimension after initialization. Device/function limits, identity replay, NaN-poisoned correctness for five patterns, unchanged input, and 31 valid timing samples with raw intervals at least 0.5 ms are required. Any rejection stops that group's run. A `controller_complete` record is admissible only when the guard also reports a clean `guard_complete` and no incident/abort.

CPU simulation: `python test/plan_selection_study/direct_candidates/controller/test_block_controller.py -v`. It checks prediction on both swap orientations and a mocked end-to-end phase sequence; it does not validate the actual target-device layout or GPU execution.
