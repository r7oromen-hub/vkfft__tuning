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
schema. `admission.py` is a separate correctness-only controller; it does not select or
time candidates for performance and its output requires independent evidence audit.

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

## Proposed guarded correctness admission

`admission.py` takes a committed explicit JSON protocol and manifest, built binaries and
build evidence, and new output paths. It does not generate or randomly select candidates.
The manifest is `{"schema":"nonuniform-admission-manifest/v1","candidates":[...]}`;
each entry contains exactly `candidate_id` and the full versioned `request`. IDs are
recomputed, duplicates rejected, and at most 32 candidates are allowed. Although the
static request model supports N=2, this fixed admission matrix rejects N<4: the harness
does not define quarter-tone for N=2. This is a protocol limitation, not a GPU rejection.

The protocol schema is `nonuniform-admission-protocol/v1` and its exact fields are
`schema`, `gpu_uuid`, `hostname`, `boot_id`, `machine_id`, `seconds`, `idle_seconds`,
`timeout`, `graph_repeats`, `patterns`, and `seeds`. The last two must be the full
`["zero","impulse","constant","quarter-tone","random"]` and `[42,53,71]` matrix.
The caller must also supply `--campaign-deadline-monotonic` as an absolute monotonic
deadline on the same boot recorded in that protocol. The effective deadline is the earlier
of this external campaign deadline and the stage ledger deadline. The stage ledger begins
after build, preflight and device-lock acquisition; those earlier costs therefore must be
charged by the external campaign budget, not ignored or inferred from this stage. No prior
campaign deadline is fabricated by the controller. It checks time before each launch and
preserves a final no-launch denial if the reserve is exhausted. The numeric deadline is
bound to the committed invocation and same-boot protocol, but this controller does not
independently authenticate an external campaign ledger: a later audit must verify its origin
and account for prior build and preflight occupancy.

Before guard startup, the controller verifies clean committed sources, an experiment
declaration whose protocol hash matches, host/boot/GPU identity, both binary hashes,
generated provenance and the complete generated include-tree inventory against the
committed upstream copy and declared generated transformations. The required
`nonuniform-admission-build/v1` attestation records `code_commit`, `vkfft_commit`,
`generated_provenance_sha256`, binary hashes under `binaries`, a nonempty list of
`build_commands`, `tree_sha256` for `original`, `generated`, and tracked `instrumentation`,
plus `compile_commands_sha256` and `original_dependencies_sha256`. The corresponding
compile database must show the original include tree preceding generated includes, and
the original target's object dependency file must show the untouched VkFFT header rather
than the generated VkFFT headers. `tree_inventory` and `tracked_instrumentation` expose
the canonical inventories for preparing such an attestation. The build association and
recorded dependencies are not an independent reproduction of the binaries.

The controller uses the shared UUID-named flock, passes its descriptor to child processes,
and uses the existing pidfd guard with per-process before/after samples. Output, guard,
incident and budget files are exclusively created; no automatic resume is supported.
Each candidate receives describe, original/experimental default and representable
identity checks, then 15 full-output validations. Exact command, sanitized override
environment, stdout, result, identity, source/plan evidence and budget records are kept.
Unknown errors, numeric failures and scheduler error 3002 stop the session: the current
nonuniform generator lacks the evidence needed to classify those as safely recoverable
local rejections. A candidate completing all checks is `checks_passed_pending_audit`, not
an independently audited admission. Budget denial records partial and unstarted requests;
it is not a successful complete matrix. Guard completion alone does not replace a later
independent audit of guard/ledger linkage, stdout versus result, binary provenance and
complete validation coverage. No live GPU evidence has been collected for this controller.
