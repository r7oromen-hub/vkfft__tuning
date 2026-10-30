# Optional counter collection readiness (2026-10-28)

This is preparation only. No profiler process or GPU workload was launched by
this audit. The eight preregistered profiler processes remain unperformed.
The frozen research design is unchanged.

The existing independent `vkfft_profile_benchmark` target can be built without
rebuilding or replacing either accepted main executable. Its isolated generated
header tree must be the accepted instrumented header tree. Record its SHA256,
the profile source SHA256 and header provenance alongside any eventual report.

Source inspection confirms that its argument validation requires execute mode,
out-of-place placement, direct dispatch, graph-repeats=1, repeats=1 and at least
1000 ms of warmup. It restores the input before `cudaProfilerStart`, calls one
complete forward, synchronizes the stream, then calls `cudaProfilerStop`.
Initialization and correctness work are outside that range. Any profiler result
must show the expected upload count and successful correctness/input checks.

The common harness arguments for each process are:

```text
--study-mode execute --placement outofplace --timing-mode direct
--graph-repeats 1 --repeats 1 --warmup 3 --warmup-ms 1000 --pattern random
```

Only these workload/candidate pairs are permitted (two candidates per row):

| N | Batch | Candidates |
|---|---|---|
| 16384 | 1 | default; a128_r4_l1_np1 |
| 16384 | 2048 | default; a128_r4_l1_np1 |
| 655360 | 1 | default; a128_r4_l1_np1_s3_65536 |
| 655360 | 51 | default; a128_r4_l1_np1_s3_65536 |

Default omits tuning flags. The first tuned candidate adds `--aim-threads 128
--register-boost 4 --use-lut 1 --register-boost-nonpow2 1`; the second also adds
`--swap-to-3stage 65536`. Seeds and candidate requests must be retained.

## Remaining execution gate

The current primary `gpu_guard.py` accepts only the Python process's exact
direct-child executable. It cannot safely authorize a profiler wrapper and its
harness descendant. Do not run ncu through that guard and do not broaden the
primary guard's executable allowlist to work around this restriction.

Before optional collection, an independent profile guard must register the real
ncu ELF (not its shell launcher) and the real profile harness executable. Require
the live root Python -> registered ncu ELF -> registered harness PPID chain,
with exact executable paths and stable PID/start-ticks at every observation.
Any unknown GPU PID, unreadable identity, changed ancestry or reused PID must
abort collection and quarantine output. Termination may address only registered,
revalidated own identities using pidfds. Keep the primary guard's idle/memory
and contamination checks. A wrapper/grandchild topology that differs from the
verified chain is a failure, not permission to accept arbitrary descendants.

Resolve installed ncu options and exact supported metric names before data:
DRAM read/write bytes, L2 read/write bytes, achieved occupancy, launch dimensions
and compiled launch resources. Retain `--version`, metric discovery output,
real ELF identity, exact argv, environment, replay/cache settings, collection
passes and exported raw report. Use `--profile-from-start off` and
`--clock-control none`; freeze explicit replay and cache-control settings after
checking installed help. Sum traffic across every upload in the one forward;
do not sum or average occupancy without stating the weighting convention.

Profiler and replay times never enter primary latency estimators. If the exact
chain/range gate cannot be verified within this fixed optional effort, leave
counter collection unperformed and continue H1/H3 with limited mechanism claims.

## Offline reuse audit

`python -m unittest test_gpu_guard test_runner_s0 test_trace_audit` passed all 22
GPU-free tests on 2026-10-28. No S0 failure requiring a main harness or primary
guard edit was found. Remote GPU health and semantics remain root-scheduled
gates; this offline result makes no claim about their outcome.
