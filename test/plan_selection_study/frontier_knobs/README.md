# Joint LUT and coalesced-memory prototype

This independent prototype extends `frontier_search` requests with `--use-lut` and `--coalesced-memory`. It has not passed GPU admission and does not establish correctness or performance. The boost remains 1; the single-upload FP32 1D power-of-two C2C forward, out-of-place restrictions and all v2 table, stage, layout and compiled-resource checks remain. It does not implement or authorize a GPU runner.

The generator reuses v2's isolated generated headers and rejection instrumentation. It changes only the generated benchmark's candidate-argument guard to allow these two knobs, and adds explicit requested-versus-resolved checks. The upstream planner still prepares LUT and split/layout state; v2's requested group is imposed during splitting. Actual descriptors, layout, resources and complete outputs must be checked for each combination. The generator records hashes of its source, parent generator, and generated benchmark before and after the change; the v2 provenance retains the hashes of generated files and underlying source.

`use_lut` has values `-1` (off), `0` (automatic), and `1` (on). `coalesced_memory` has values `0` (automatic) and `16`, `32`, `64`, `128` bytes. Automatic is a request to use the device default; it is distinct from an explicit value even if the resolved value matches. Capture both requested and resolved settings. `requested_config` omits automatic zero fields, matching the benchmark's serialization; explicit fields must equal their requested values. Check the actual LUT descriptor and size separately, including the valid case of zero LUT entries for a single stage.

`requests.py` defines a new `frontier-knobs-request/v1` identity over the canonical JSON of schema, workload and complete request. It is intentionally distinct from v2 IDs. `enumerate_requests(n,batch)` streams all v2 ordered radix/R/group requests crossed with all 15 knob combinations, preserving static rejection reasons; `inventory_summary(n,batch)` counts that complete domain without materializing all rows. Passing static screening does not mean the scheduler or GPU accepts the request. The request adapter returns an environment for radix/R/group plus a separate list of CLI arguments for the knobs. An execution controller must preserve both, as well as the exact command and environment, and must not silently replace failed selected candidates.

```python
import importlib.util
from pathlib import Path
path = Path("test/plan_selection_study/frontier_knobs/requests.py")
spec = importlib.util.spec_from_file_location("frontier_knobs_requests", path)
requests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(requests)
request = {"radix": [16, 16], "logical_registers": 16, "grouped_batch": 4,
           "use_lut": 1, "coalesced_memory": 64}
candidate_id = requests.candidate_id(256, 16, request)
invocation = requests.invocation(256, 16, request)
controls = requests.comparison_contract(256, 16, request)
```

For each workload and knob pair, run original default, experimental default, and replay of the experimental default's representable identity with the same knobs and compare their full actual configuration, structure, descriptors, source and compiled resource evidence. Validate each under the device guard with complete numerical checks. An unsupported identity is an explicit disposition, not a pass. The primary performance denominator remains original VkFFT with **no knob overrides**, run separately; same-knob original is an identity/control comparison, not the primary denominator. Independent sessions and budget policies require a separately frozen execution protocol.

`validation.check_execute` and `check_describe` first check the joint knob contract and then retain the existing v2 guard, numerical, actual layout, descriptor and resource validations. `check_matched_identity` requires three guarded execute rows for the same workload and compares original versus experimental default and experimental default versus identity replay. `check_performance_baseline` requires original no-knob default. These functions validate supplied evidence; they do not run workloads or establish guard cleanliness themselves.

```text
python test/plan_selection_study/frontier_knobs/prepare_headers.py --source VkFFT/vkFFT --output knobs_generated/include
cmake -S test/plan_selection_study/frontier_search -B knobs_build -DFRONTIER_GENERATED_ROOT=/absolute/path/to/knobs_generated
cmake --build knobs_build
python -m unittest discover -s test/plan_selection_study/frontier_knobs -p "test_*.py"
```

Use the `frontier_search` CMake targets for the generated benchmark and original-header target, with separate paths and hashes for each build. The generated original target must run without the radix/R/group environment overrides. Run formal CPU validation and all GPU admission only after freezing the new source and execution contract; old v2 evidence does not validate these new combinations.
