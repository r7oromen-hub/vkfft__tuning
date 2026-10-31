# Proposed nonuniform register requests

This isolated CPU module implements the versioned request and static constraints in
`experiments/research/nonuniform_register_admission_contract.json`. It does not inject a
scheduler, build or execute a kernel, change the existing frontier, or establish device
validity. Static acceptance always reports pending scheduler, resource and numerical
validation. Full implementation still requires the planning, layout, code-generation,
identity and device-admission closure described in the contract.

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
