# Proposed nonuniform register requests

This isolated prototype implements the versioned request and static constraints in
`experiments/research/nonuniform_register_admission_contract.json`. It does not inject a
scheduler into the existing frontier, build or execute a kernel, or establish device
validity. A separate header generator below proposes scheduler injection into a generated
tree. Static acceptance always reports pending scheduler, resource and numerical validation.
Device identity, guarded numerical admission and measured actual layouts remain essential
before any performance result.

An exact request has `schema`, `workload` (`n`, `batch`), ordered `radix`, a 33-integer
`register_table`, explicit `profile`, and `grouped_batch`. Slot `r` is the logical register
value shared by every stage with radix `r`. `nonuniform_active_only` requires zero in every
inactive slot; `uniform_compatibility` requires the existing frontier's complete uniform
table, including populated unused supported radix entries. The profiles are distinct even
when their table contents coincide. The SHA-256 identity covers canonical JSON of the full
request including version, workload, exact order, profile, all slots and group.

`validate_request(request)` returns active `min_registers`, `max_registers`, and the
planned `n / min_registers` FFT thread dimension. That prediction is not an observed
block. `candidate_record(request)` snapshots and hashes a valid request.
`enumerate_candidates(n, batch, profiles=...)` returns a lazy iterator over the complete
static request domain for the explicitly selected profiles; no GPU validity or uniform
sampling is implied. Very large domains must be consumed incrementally.

`request_environment(request)` produces the seven `VKFFT_NONUNIFORM_*` strings consumed
by the separate harness. It validates before serialization, including the full workload;
the harness rechecks that `n` and batch match its own command-line workload. Run each
candidate in a clean environment. Any legacy `VKFFT_EXPANDED_*` or `VKFFT_DIRECT_*`
candidate variable conflicts with the new interface and is rejected. An untouched original
control must run with all nonuniform candidate variables absent; it rejects requests.

## Generated planning and build

`prepare_headers.py` composes the existing isolated expanded generator without changing
its files or VkFFT. It replaces the generated scalar request header with this directory's
table-aware version. The generated scheduler seeds the full table and active min/max at
the original register-selection entry, suppresses automatic radix merges and the min-R
first-stage reorder for this exact-order request path, then verifies that later scheduler
steps have not changed table, stage order or envelope. The splitter derives the FFT
dimension from `N/minR`; its group and pre-normalization swap checks remain active. The
generated plan checks full table, exact stage order and local geometry before codegen and
again before compilation. The generated harness rechecks the actual normalized axis block
and compiled-function/device resource limits before any candidate launch. It records the
version, workload, exact radix, full table, profile and group as requested strings; the
instrumented descriptors separately record observed tables and layouts. Its default path
with no candidate variables remains the generated instrumented default.

The generator uses strict replacement anchors and refuses an existing nonempty include
tree or pre-existing generated sibling artifacts. It emits generated source hashes and
the research contract hash in `expanded_provenance.json`; this is an association, not an
independently reproduced build or device validation. Its output is disposable and must
be outside the upstream include tree. Example commands for a later authorized CPU-only
check and build (not evidence that these have been run):

```text
cmake -S test/plan_selection_study/nonuniform_search -B <cpu-build> -DNONUNIFORM_CPU_ONLY=ON
cmake --build <cpu-build>
ctest --test-dir <cpu-build> --output-on-failure
```

For a separately authorized CUDA build, generate with
`python test/plan_selection_study/nonuniform_search/prepare_headers.py --source VkFFT/vkFFT --output <new-root>/include`,
then configure this directory with `-DNONUNIFORM_GENERATED_ROOT=<new-root>` and a CUDA
toolkit. CMake provides separate generated-planner and untouched-original targets.
Compilation and describe mode do not prove numerical correctness. The existing frontier
controllers understand scalar R and cannot safely be reused unchanged for this request
schema; a new guarded controller must validate the full request, observed descriptors,
identity, numerical results and preserved evidence before any timed GPU experiment.

`validation.check` and `check_describe` perform table-aware checks on individual guarded
execution records, including exact request telemetry and pre-codegen, post-codegen and
normalized post-initialization structure. Execute checks the full numerical result and
compiled resources; describe explicitly does not invent compiled observations.
`identity.representable` extracts only an exact supported profile from an observed
instrumented default; `compare_three` compares already-validated original, instrumented
default and replay results. For the unmodified original, actual NVRTC options were not
captured: only the restricted path's expected options and compiler-header hash can be
compared with the instrumented binaries' actual options. These functions do not audit a
guard ledger, bind a binary to a full campaign, or establish 15-case coverage: a separate
guarded controller and independent audit must enforce those requirements. The synthetic
tests are CPU logic checks and are not device evidence.
