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

```sh
python3 -m unittest discover -s test/plan_selection_study/frontier_analysis -p 'test_*.py' -v
```
