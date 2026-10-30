# Plan selection study: evidence and interface map

This document is the initial interface map. The executed design and subsequent
clarifications are recorded in `research_design.json`, `research_freeze.json`,
and the numbered `research_addendum_*.json` files. See `RUN_STATUS.json` for
execution status and `research_questions.md` for the current research questions.
Root owns every remote GPU launch. No old measurement is reused as a new timing
label; `test/plan_tuning` and its incomplete repair draft remain historical.

## Questions and evidence

| Research question | Required observations | Primary outputs | Boundary on interpretation |
|---|---|---|---|
| How close is the default heuristic to the best tested plan? | Per-workload default, every screened candidate, screen-selected winner, a single screen-selected global fixed configuration, independent paired process rounds, explicit default/control pairs | Distribution and quantiles of default/best-tested screen ratios; independent-confirmation ratios and intervals; no-improvement/regression counts; representative raw points | The screen oracle is selection-biased and restricted to the tested space. Independent confirmation evaluates the frozen choices. Failure to detect improvement is not equivalence or global optimality. |
| Does the preferred plan depend on batch regime and actual structure? | Requested and resolved knobs; canonical precompile descriptors; ordered uploads, stage radix, local size and axis blocks; generated-source hashes; same candidate pair at both batch regimes | Requested-to-resolved-to-structure/source multiplicities; core structure transition table; paired candidate crossing plots; within-structure time distributions | Identical recorded descriptors are not automatically identical generated code. Distinguish core changes from LUT-only changes. A crossing hypothesis requires both directions and independent measurements. |
| Where does planning/search time go, and when could search be amortized? | Fresh-process wall time; initialization outer wall; nested trace spans with exclusive times; planning/layout/codegen/NVRTC/binary extraction/module load/resource costs; graph construction; evaluated device intervals; cache environment | Non-overlapping cost breakdown; per-candidate and cumulative search-cost tables; duplicate-code compilation budget; conditional break-even curves | Nested inclusive spans cannot be added. Graph construction, warmup, correctness and process overhead are separate. Fresh process does not mean cold driver cache. No actual plan-only path, compiler avoidance or online policy is inferred from instrumentation alone. |
| Are the comparisons temporally credible? | Mandatory guard record IDs, UUID, foreign-compute/memory checks, before/during/after boundaries, incident quarantine, exact negative controls, sample sequences | Per-block stability/control summary, guard coverage, retained failure/incident ledger | Numerical correctness and timing validity are independent. A guard incident aborts the owned run; no silent retries or minimum-of-repeats acceptance. |

## Reusable mechanics

The new orchestrator can retain the old process-call contract:

`run_one(args, workload, candidate, phase, round_no, order) -> row`

Here `args.exe` identifies the exact instrumented binary, `workload` contains
`n`, `batch`, and frozen graph repetition count, and `candidate` is an ID/config
pair. All GPU-capable subprocesses, including calibration and cost probes, must
pass through that one entry point. `gpu_guard.py` already recognizes a target
named `run_suite.py`, imports it from the target directory, wraps `run_one`, and
quarantines an incident before the orchestrator can append a successful row.

Reusable implementation ideas are process isolation, SHA256 input seeds shared
within a paired block, fresh seeds for independent phases, append-only JSONL,
strict manifest identity, crash-tail handling, CUDA graph measurement using
immutable out-of-place input, numerical criterion_v2, and default/control
matching. These mechanics will be copied or refactored into this new directory;
the old files and result directories will not be edited to become the new study.

The harness interface remains compatible with `--n`, `--batch`, the declared knob
flags, `--warmup`, `--warmup-ms`, `--repeats`, `--seed`, `--timing-mode graph`,
`--placement outofplace`, and `--graph-repeats R`. Additional knobs are pending
Astra's decision. The new binary is `vkfft_instrumented_benchmark`; harness-owned
sources and generated instrumentation live under `instrumentation/`.

## Changes required in orchestration

- Replace the old hard-coded 48-by-19 grid with an explicit frozen study manifest.
  Candidate order, workload blocks, sample count, process rounds, fixed global
  baseline selection and confirmation rules must be data-independent code paths.
- Keep screen selection and independent confirmation separate. Do not treat 101
  event samples inside one process as 101 independent process observations.
- Prefer short, balanced candidate blocks and nearby default/control anchors over
  a single globally shuffled hour-long candidate list; exact blocking is pending
  Astra. Record block, round and within-block position explicitly.
- Freeze graph R per workload after declared calibration/verification. R is
  measurement metadata, not a candidate knob or learning feature.
- Require guard coverage for every accepted process. Preserve quarantine and
  failure rows separately, along with source-row hashes and actual timestamps.
- Preserve actual planning/source descriptors and trace spans in full. Do not
  replace them with only a lossy fingerprint or model-ready numeric vector.
- Record observed cache environment and version/compiler/architecture metadata.
  Do not label an unobserved driver-cache hit or miss.
- Keep research-workflow wall time, measured plan-build budget, graph setup and
  device evaluation budget as distinct search-cost definitions. Any amortization
  arithmetic must state which definition it uses and use independently confirmed
  execution gains, including undefined/infinite break-even when no gain exists.

## Record contract additions

Retain command, executable/source hashes, GPU UUID, exact requested/resolved
configuration, seed, process UTC start/end, process wall time, raw graph intervals,
per-FFT normalized samples, graph counts, immutable-input checks and both
single-forward/final numerical checks.

Add `study_id`, `block_id`, `process_round`, `position_in_block`, `measurement_role`
and guard linkage. Harness-provided `init_trace` spans contain ID, parent ID,
stage, axis/upload, start/end nanoseconds and inclusive/exclusive milliseconds.
Raw and canonical precompile descriptors, generated-source hashes and compiled
resource attributes remain separate fields. Verify trace nesting and exclusive
time accounting; report outer initialization minus traced coverage as residual
when appropriate rather than inventing attribution.

The learning model is not a primary deliverable in this study. If Astra includes
one, freeze its target, feature sets and grouped splits before observing new
outcomes, and use only guard-clean data. The previous contaminated screen/model
results remain a separate pilot record.
