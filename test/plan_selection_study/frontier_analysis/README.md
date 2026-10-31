# Descriptive frontier screen analysis

After a single guarded screen has ended and `frontier_confirmation/coverage_audit.py` has produced a passing report, run:

```sh
python3 test/plan_selection_study/frontier_analysis/analyze.py \
  --coverage /path/to/coverage.json --rows /path/to/rows.jsonl \
  --seed 20261031 --output /path/to/screen_analysis.json
```

The output path must not already exist. The tool requires the passing audit and its unique SHA-256 binding to the supplied raw rows, checks the selected-candidate partition, and reconstructs each complete candidate's three paired ratios from the raw samples. It rejects missing, duplicate or mismatched evidence. Preserve both input files and the output hashes; a `status: pass` report is accepted as an audit input, not cryptographically authenticated by this script.

`overall` and each workload's `outcomes` use **complete selected candidates** as the denominator, including an identity candidate if it was actually screened. The default reference at ratio 1 is not added to that denominator. `best_screened` is the best measured candidate; `best_with_default` includes a default fallback only when default evidence exists. A workload not reached before the budget boundary has no observed fallback in the report.

The R≤64 and R128/256 sections report selected, complete, rejected, partial and not-started counts plus completion divided by selected. An independent integer-composition count also supplies the static-supported request denominator for each workload represented in the report. Complete divided by that denominator describes coverage of this finite prototype's requests; static support does not prove that a plan is device-valid, and it is not the full VkFFT space. The coverage report does not identify the register values of individual unselected requests. R≤64 here is the lower-register subdomain of the current uniform-R prototype; it is not the earlier fixed-endpoint search space.

Matched groups contain only completed candidates from the same workload. They compare ordered radix at fixed R and group, and R/group layouts at fixed ordered radix; two additional views hold one of R or group fixed. Each group lists its members, observed layout and available physical-resource fields, and its slowest/fastest ratio. These are differences between complete plans. Their radix, register envelope, layout, generated source or compiled resources can change together; the report does not attribute an effect to an individual parameter.

The replay orders each workload's already completed candidates by a deterministic hash of the seed, workload and candidate ID. Its initial ratio 1 is a default fallback, and each later point is the best measured ratio in the reordered prefix. This is an **offline reorder**, not the original execution sequence, an online tuner, a cost model or an equal-time budget comparison. It excludes rejected, partial and unmeasured requests.

All values are descriptive three-pair screening results on the recorded GPU and workload. They are not independent confirmation, cross-device results, cold-cache measurements, or proof of a VkFFT-wide optimum. No graph, model or training is produced.

## Fixed 28-workload aggregation

`aggregate_confirmed.py` left-joins a passing screen coverage report and its raw rows onto the fixed 7 sizes (128 through 8192) × 4 batches (1, 16, 256, 4096). A screen-only report is available without confirmation arguments. To include confirmation, supply the complete two-session audit report **and** the frozen plan, original screen guard/incidents/budget logs, both sessions' raw evidence directories, and the two binaries:

```sh
python3 test/plan_selection_study/frontier_analysis/aggregate_confirmed.py \
  --coverage /path/coverage.json --rows /path/screen/rows.jsonl \
  --confirmation-audit /path/confirmation/two_session_audit.json \
  --plan /path/confirmation/plan.json \
  --screen-guard /path/screen/guard.jsonl --screen-incidents /path/screen/incidents.jsonl \
  --screen-budget /path/screen/budget.jsonl \
  --first-dir /path/confirmation/first --second-dir /path/confirmation/second \
  --original /path/original --expanded /path/expanded --output /path/aggregate.json
```

Each session directory must contain `rows.jsonl`, `guard.jsonl`, `incidents.jsonl`, and `budget.jsonl`. The tool hashes and re-audits the raw confirmation logs, plan, screen source and binaries, checks them against the supplied confirmation audit, and checks the coverage-to-screen rows hash, GPU and execution commit. It records the input audit hashes; the caller must still obtain the passing coverage and confirmation reports from a trusted, formally registered experiment. Hashes detect mismatched content, but do not authenticate who produced a report. No GPU is used by this analysis.

The original candidate-level fractions use completed selected candidates (including measured identity); separate nondefault-candidate fractions exclude requests identical to the observed default. Workload counts use 28 as denominator, and a workload is screened only after at least one nondefault candidate completes: an identity-only workload remains uncovered. Uncovered, unconfirmed and confirmation below the practical threshold remain separate states. The per-workload confirmation estimate is the geometric mean of the two session ratios, giving equal weight to their log ratios; it never selects the better session. Both session estimates and their intervals remain visible. These estimates do not correct selection or multiple-comparison bias.

The screen mean is present only when all 28 workloads have both observed default evidence and at least one complete nondefault candidate; for each workload its best compares those nondefault candidates with the default ratio 1, excluding identity timing noise. The fallback policy uses a candidate only when both confirmation sessions meet the practical criterion, and otherwise uses an observed default. Its all-28 mean is absent if any workload lacks an observed default. Thus fallback is a deployment-policy estimate, not evidence that an untested workload has zero tuning potential. The arithmetic mean latency reduction is `mean(1-ratio)`; geometric mean speedup is `exp(-mean(log(ratio)))`. Both weight the 28 workloads equally. Neither is a global optimum or a cross-device claim.

## Screen ledger wall-clock costs

After a screen finishes and its coverage audit passes, run:

```sh
python3 test/plan_selection_study/frontier_analysis/search_costs.py \
  --coverage /path/coverage.json --rows /path/search/rows.jsonl \
  --budget /path/search/budget.jsonl --output /path/search_costs.json
```

The tool binds both raw files to distinct hashes in the passing coverage report, verifies the GPU and execution commit, and checks the ledger boot, contract, monotonic event order, work identities, terminal denial and in-flight state against the raw executions. The coverage report is a trusted audit input, not a cryptographic signature; preserve its hash and use a report from a formally registered run. It does not re-run the full coverage audit or authenticate who wrote a report.

`interval` covers ledger start to its final logged finish or budget denial. It partitions that span into paired work intervals and all intervening gaps. `phase_seconds` separates identity, calibration, describe, validation, timing and post-rejection. Candidate records report their own work and, only where a unique matching `screen_pair` proves it, the paired original-default timing; unpaired originals remain in `unattributed`. A unique post-rejection control is assigned to its rejected candidate. Work intervals include the guard, process launch, any JIT/compilation, validation, output and scheduling: they are **not** pure compilation or GPU kernel time.

The candidate `observed_span` runs from its first to last attributable completed work interval, including any intervening gaps; it omits time before the first work and after the last work, and is not the exact candidate-start-to-disposition time. A candidate not started before the budget boundary has unknown costs, not a measured zero. Intervals may exclude binary build, manifest and other preparation, device-lock wait, initial/final guard activity, log tail, cleanup, and independent confirmation. The ledger span is therefore neither whole-experiment wall time nor the entire GPU budget. Separate formally recorded phase timings would be needed to cover those stages. No online search gain is inferred from these accounting records.

```sh
python3 -m unittest discover -s test/plan_selection_study/frontier_analysis -p 'test_*.py' -v
```
