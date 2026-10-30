# Guarded frontier screening

This is a Linux-only runner for the v2 uniform-R, boost-1 frontier. It must be committed before use. It refuses a dirty tracked root or VkFFT submodule, verifies the generated v2 provenance and hashes each binary, manifest and supplied build/source evidence. A binary hash and generated-source hashes are records, not independent proof that the binary was built from those sources: retain build commands, compiler version, dependency paths and compile logs as `--artifact-file` evidence.

Example after building the original and frontier executables and producing a `frontier-manifest/v1` manifest (paths are illustrative):

```sh
python3 test/plan_selection_study/frontier_execution/runner.py \
  --gpu GPU-3559ec6b-6278-8f93-0706-5f8a85c4911b \
  --original /work/build-original/vkfft_original_control \
  --expanded /work/build-frontier/vkfft_expanded_candidate \
  --manifest /work/frontier/n256_b4096.jsonl \
  --source-file test/plan_selection_study/frontier_search/prepare_headers.py \
  --source-file test/plan_selection_study/frontier_search/candidate.hpp \
  --generated-provenance /work/generated/expanded_provenance.json \
  --artifact-file /work/build-frontier/compile_commands.json \
  --output /work/frontier/rows.jsonl --budget-ledger /work/frontier/budget.jsonl \
  --log /work/frontier/guard.jsonl --incidents /work/frontier/incidents.jsonl \
  --lock /work/frontier/gpu.lock --seconds 14400 --timeout 300
```

The generated provenance must be alongside `include/` and `expanded_benchmark.cpp`, as produced by the v2 generator. The source-file arguments must identify committed files in the root or VkFFT repository. Add other build evidence using repeated `--artifact-file`; all hashes become part of the immutable run contract. Multiple `--manifest` options preserve their given order. Manifest selection indices must be contiguous from zero; historical anchors and any desired exhaustive subspace must already be selected by the manifest.

For each workload and every resumed session, the runner first checks original default, expanded default and default identity replay. A nonrepresentable default is reported and its candidates are skipped. It then calibrates graph repetitions and runs each selected candidate through guarded describe, 5 patterns × 3 seeds of full-output numerical validation, and three paired timings with alternating order and randomized first side. Existing `validation.py` verifies workload, guard, full outputs, input immutability, graph, descriptor and compiled resources; unexpected errors stop the run. Rejections and numerical failures are separate from timing results and followed by an original control. Candidate and original source/structure signatures must stay stable.

The persisted budget begins before the guard idle check and includes elapsed time while stopped. The same boot, contract, paths, GPU, budget and input hashes are required on resume. A candidate interrupted before disposition is repeated in full; only candidates in sessions with a successful final `guard_complete` are skipped. An unresolved in-flight ledger entry, changed boot, missing output/guard evidence, corruption, altered source or changed binary requires an offline audit and a new explicitly managed run; do not edit the ledger to pretend an interrupted item completed. An in-flight call is allowed to finish even if its upper bound is exceeded, so a budget can overrun. GPU guard sampling cannot exclude interference between samples.

Results are screening measurements on the specified GPU and do not establish cross-device performance or independent confirmation. The original binary does not expose actual NVRTC options; its identity comparison uses the documented expected-options evidence. This runner has CPU mock tests but has not itself been admitted on a GPU.

```sh
python3 -m unittest discover -s test/plan_selection_study/frontier_execution -p 'test_*.py' -v
```
