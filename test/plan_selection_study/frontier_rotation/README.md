# Separate multi-address rotation for confirmed frontier winners

This is an offline-developed extension for a separate working-set experiment. It reuses the existing CUDA 12/13-compatible rotation generator, pool and numerical checks. It has **not** passed GPU admission. It never replaces or merges with the hot repeated-address confirmation results, and it does not measure cache-miss rates or prove a cold cache.

`eligibility.verify` re-audits the raw screen and both confirmation sessions and applies the existing two-session audit. Only `role=winner` and `positive_confirmation=true` cases can enter rotation. The GPU, hot binaries, generated-source provenance, stage/plan/source/resource signatures and device-reported L2 must match the audited hot evidence. No positive winner means no measured rotation case. Negative controls and unconfirmed improvements are not admitted.

For one FP32 complex input buffer of `N * batch * 8` bytes, the pool uses the smallest power-of-two slot count K >= 2 with `K * buffer_bytes > actual_L2_bytes`. Inputs and outputs each occupy K buffers, so the recorded total is `2*K*buffer_bytes`. K and the power-of-two graph repetition count must not exceed 2048, the repetition count must be divisible by K, the combined pool is capped at 1280 MiB, and available device memory must suffice. If these conditions cannot be met, the case is recorded uncovered. In particular, small-batch cases can require more than 2048 slots and remain unsupported by this prototype.

Each slot has a distinct input/output pair and distinguishable deterministic input. The generated program checks the actual graph's kernel arguments, slot sequence and visit counts. It validates every input byte and every output element for every slot after an initial replay and again after a separate post-timing poisoned-output replay. Timed replays themselves are not individually read back. `rotation_initial_validation` and `rotation_final_validation` retain the full per-slot checks, while the first and final validation intervals are separate from timing samples.

## Build from frozen hot inputs

Generate a new, empty rotation tree from the exact frontier generated tree used by the hot confirmation. The generator verifies v2 provenance, captures full input-tree and output hashes, and creates no changes to those inputs. Set absolute paths in the following examples:

```sh
python3 test/plan_selection_study/frontier_rotation/prepare.py \
  --expanded-generated-root "$HOT_GENERATED" --output "$ROTATION_GENERATED"
cmake -S "$ROTATION_GENERATED" -B "$ROTATION_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DROTATION_GENERATED_ROOT="$ROTATION_GENERATED" -DEXPANDED_GENERATED_ROOT="$HOT_GENERATED" \
  -DVKFFT_INCLUDE_ROOT="$PWD/VkFFT/vkFFT" -DINSTRUMENTATION_ROOT="$PWD/test/plan_selection_study/instrumentation" \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$ROTATION_BUILD" --parallel 2
```

Keep the build logs, `compile_commands.json`, and the original target's compiler `.o.d` dependency file. The runner requires original include precedence and rejects dependencies on generated VkFFT headers. Build time and all other work within the campaign consume the original global wall-clock deadline; building is not a GPU admission. After the source commit and build, create the attestation before `tools/run_experiment.py` creates its declaration commit:

```sh
python3 test/plan_selection_study/frontier_rotation/make_build_attestation.py \
  --code-commit "$(git rev-parse HEAD)" --vkfft-commit "$(git -C VkFFT/vkFFT rev-parse HEAD)" \
  --rotation-root "$ROTATION_GENERATED" --original "$ROTATION_BUILD/vkfft_rotation_original" \
  --expanded "$ROTATION_BUILD/vkfft_rotation_expanded" \
  --build-command 'record the actual generation, configuration and build commands' --output "$ROTATION_BUILD/build-attestation.json"
```

The attestation records a declared source/binary association; it is not an independent rebuild or proof of a binary's origin. Preserve the build logs and inspect dependencies. The original target must use the untouched VkFFT checkout; instrumentation must be the committed instrumentation directory. The run's declaration commit is checked to be a single experiment record whose parent equals the attested code commit.

## Guarded execution

Run `runner.py` only through `tools/run_experiment.py` after committing this directory and running formal CPU checks. It requires:

- `--plan`, `--confirmation-commit`, `--gpu`, and the hot `--hot-original`/`--hot-expanded` binaries;
- all raw evidence via `--screen-rows/guard/incidents/budget`, `--first-rows/guard/incidents/budget`, and `--second-rows/guard/incidents/budget`;
- `--rotation-original`, `--rotation-expanded`, `--rotation-root`, `--generated-root`, `--original-include-root`, `--instrumentation-root`, `--build-attestation`, `--compile-commands`, and `--original-dependencies`;
- the committed original `--campaign-record`, its recorded `--campaign-start` and `--admission-budget`, and fresh `--output`, `--budget-ledger`, `--log` and `--incidents` paths;
- `--lock /run/lock/vkfft-study-<GPU UUID>.lock`, `--seconds` for the stage limit, and optionally `--timeout` and `--idle-seconds`.

Use `python3 test/plan_selection_study/frontier_rotation/runner.py --help` for exact flags. The runner verifies the recorded campaign artifact hashes, GPU, original protocol and boot, and uses the earlier of the stage limit and original global deadline. Its shared lock is inherited by each guarded GPU child. Every new child is admitted against the absolute deadline; an in-flight child may finish after it. Evidence paths are exclusive and automatic resume is disabled. The GPU guard uses periodic sampling and cannot rule out interference between samples.

For each eligible case it calibrates the original rotation graph, then runs ten randomly ordered original/candidate process pairs. Every process is checked against its hot signature, actual L2, pool, graph, full-slot correctness and compiled resources before the next process starts. Numerical failure stops all further work. Unsupported pool, memory or graph-inspection conditions are recorded as uncovered. A `complete_with_uncovered` session is not a complete measurement of all confirmed winners.

## Independent offline audit

Use `audit.py` after the guard has closed. Supply the same `--plan`, `--confirmation-commit`, `--gpu`, hot binaries and all twelve screen/first/second evidence paths as for the runner, plus rotation binaries and the rotation `--rows`, `--guard`, `--incidents`, `--budget` and a fresh `--output`. Supply `--rotation-experiment-id` and the same `--rotation-root`, `--generated-root`, `--original-include-root`, `--instrumentation-root`, `--build-attestation`, `--compile-commands`, `--original-dependencies`, `--campaign-record`, `--campaign-start` and `--admission-budget` used at launch. The audit rechecks the frozen declaration/source, generated trees and build association, campaign artifacts and boot, absolute deadline and per-work timeout reserve; it also re-audits the hot evidence and rotation guard and budget chain, parses each raw stdout, reruns each complete pool/plan check and recomputes all ten pairs and their within-session interval. It reports measured and uncovered cases separately. A passing audit means evidence integrity for this session; it does not assert a cold-cache effect or correct for selection and multiple comparisons. Partial sessions fail the complete-session audit and require separate review.

```sh
python3 -m unittest discover -s test/plan_selection_study/frontier_rotation -p 'test_*.py' -v
```

These are CPU development tests and simulated evidence. They do not compile the real CUDA binary, inspect an actual CUDA graph, or establish GPU correctness or speed.
