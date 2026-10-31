# Frontier v2 research prototype

This directory creates a separate experimental header tree and a complete, auditable **request inventory** with bounded selection. The generator does not run a GPU, modify upstream VkFFT, test numerical correctness, or establish that a selected request can initialize or execute. Freeze the input spec, generated sources and manifest before collecting timings; retain every failed request without replacement according to a separately approved execution protocol.

The exhaustively enumerated domain extends the earlier prototype: 1D power-of-two N=2..8192, ordered radix decompositions from 2/4/8/16/32, uniform logical R=2/4/8/16/32/64/128/256, power-of-two grouped-batch divisors, one upload and registerBoost=1. It inventories static rejections as well as pending requests. It does not claim all possible VkFFT plans, actual layout coverage, or a device-valid configuration. The actual harness also restricts the workload to CUDA FP32 C2C forward out-of-place. Each selected candidate still requires identity, scheduler, resource and full numerical validation before timing; see `../expanded_search/GENERATOR.md`.

R=128/256 are implemented only in this isolated v2 generator. Its parser, complete 33-entry table seeding and consistency checks use the same divisibility and actual-layout checks as the old harness. Implementation alone does not establish correctness: GPU evidence belongs to specific requests, binaries, devices and recorded admission runs, not the whole wide-R domain. VkFFT's table is indexed by radix 0..32, not R; the wide R is a value in each supported radix entry. Its subsequent scheduler can change that value, in which case the consistency check rejects it. Non-power-of-two grouping, boost>1, and R>256 need further injection contracts and derivation checks. Mixed-prime lengths need per-radix register tables and interaction with optimizer/Rader/Bluestein; multi-upload plans need complete pass splits, layouts and buffers. See `../expanded_search/混合因子与多上传未覆盖与后续路径.md`. Those broader families have not been enumerated or validated here, and the prototype caps must not be mistaken for fundamental VkFFT limits.

Provide an observed identity request and historical requests for the **same workload**, using the existing prototype request format. No winner is assumed from another workload; verify evidence and supply it explicitly. If identity has not been observed, specify `null` and the manifest records that fact. Requests outside the domain or rejected statically appear in the anchor report with their reason, not as selected. Valid identity and historical anchors are mandatory, together with statically accepted immediate identity neighbors: one adjacent stage swap or one step in the R/group value lists. A limit too small for these mandatory requests fails rather than silently dropping them. Unsupported identity requires separate admission before execution; the presence of sampled rows is not authorization to run them.

Remaining slots are deterministically stratified by stage count, R and group, with radix multisets interleaved within a stratum. Selection favors strata with the lowest accumulated `N/R * group * stage_count` weight. This is a reproducible relative proxy for distributing selection effort, not a latency estimate or actual GPU resource measure. The manifest retains all rows, their static disposition, selected index, anchor reasons, counts and rejection reasons. Counts include only the stated prototype domain. A workload with fewer pending requests than the limit selects all of them.

Example input (`identity` and `historical` below are illustrative, not measured winners):

```json
{"n": 32, "batch": 4, "identity": {"radix": [4, 2, 4], "logical_registers": 4, "grouped_batch": 2}, "historical": [], "limit": 20, "seed": 20261031}
```

```text
python test/plan_selection_study/frontier_search/manifest.py --spec spec.json --output manifest.jsonl
python -m unittest discover -s test/plan_selection_study/frontier_search -p "test_*.py"
```

Generate a separate header tree and build on a CUDA host, or build CPU tests only:

```text
python test/plan_selection_study/frontier_search/prepare_headers.py --source VkFFT/vkFFT --output frontier_generated/include
cmake -S test/plan_selection_study/frontier_search -B frontier_build -DFRONTIER_GENERATED_ROOT=frontier_generated
cmake --build frontier_build
cmake -S test/plan_selection_study/frontier_search -B frontier_cpu_build -DFRONTIER_CPU_ONLY=ON
cmake --build frontier_cpu_build
ctest --test-dir frontier_cpu_build --output-on-failure
```

The generated binary keeps the old `VKFFT_EXPANDED_RADIX`, `VKFFT_EXPANDED_REGS`, and `VKFFT_EXPANDED_BATCH` request interface and CLI. v1 outputs and v2 outputs must not be mixed: retain v2 provenance, binary hash and the requested R. The actual new GPU UUID must be bound to a fresh original/default/replay identity, guard, resource and independent numerical admission; older GPU evidence does not establish v2 validity. Named injection rejection reasons are diagnostic only; the first appears both as `expanded_rejection_reason` in the result JSON and as a `VKFFT_FRONTIER_REJECTION:` JSON record on stderr. A generic upstream failure has a null reason and is not evidence of a specific cause. Early argument failures occur outside this injection path and keep the original argument-error format.

The build also produces `vkfft_frontier_original` from the same benchmark source with uninstrumented original VkFFT headers first in its include path. Generated headers supply only helper metadata and candidate/trace declarations; this target does not require a direct-candidate build. Override `FRONTIER_ORIGINAL_INCLUDE_ROOT` if the original checkout is elsewhere. Run the original target without request environment variables and verify its retained source and baseline identity before comparison.

Output creation is exclusive; an existing file is never overwritten. The first JSONL record summarizes the inventory and anchors and records hashes of both the selector and the reused enumerator; it is followed by one row per request. The selected index defines the per-workload order only; it does not authorize a GPU scheduling or budget policy.

For a new, formally registered phase, `tools/frontier_search_phase.py --reserve-seconds SECONDS` controls the remaining campaign wall-clock reserve; the default is 7200 seconds. Reduce it only after required later phases have completed or the campaign budget has been explicitly rearranged. The reserve is recorded in `phase_summary.json` and reapplied after manifest generation. It does not reset the campaign clock, change the same-boot global deadline or eight-hour cap, or raise the phase limit; an already running phase is unaffected.
