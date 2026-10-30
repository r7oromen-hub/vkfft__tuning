#!/usr/bin/env python3
"""Independent analytic GPU correctness probes; excluded from timing analysis.

Uses only Python's standard library and the caller-selected visible GPU.
Runs 4 patterns x 4 lengths x 2 batch counts x 2 configurations = 64 probes.
"""
import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


PATTERNS = ("zero", "impulse", "constant", "quarter-tone")
LENGTHS = (256, 49152, 81920, 1048576)
BATCHES = (1, 3)
CANDIDATES = (("default", ()),
              ("a256_r4_l1", ("--aim-threads", "256", "--register-boost", "4", "--use-lut", "1")))


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(result, n, batch, pattern):
    errors = []
    for field, expected in (("n", n), ("batch", batch), ("pattern", pattern),
                            ("placement", "outofplace"), ("timing_mode", "graph"),
                            ("graph_repeats", 4), ("status", "ok")):
        if result.get(field) != expected:
            errors.append(f"{field}: expected {expected!r}, received {result.get(field)!r}")
    for field in ("single_forward_input_unchanged", "input_unchanged"):
        if result.get(field) is not True:
            errors.append(f"{field} must be true")
    if result.get("graph_event_record_count") != 2:
        errors.append("graph must contain exactly two external event records")
    kernels = result.get("graph_kernel_count")
    expected_kernels = result.get("graph_expected_kernel_count")
    if not isinstance(kernels, int) or kernels <= 0 or kernels != expected_kernels:
        errors.append("graph kernel count does not match positive expected count")
    analytic_error = result.get("reference_cpu_analytic_l2_relative_error")
    if not finite_number(analytic_error) or analytic_error > 1e-11:
        errors.append("independent CPU FFT/analytic reference crosscheck failed")
    for field in ("single_forward_correctness", "correctness"):
        check = result.get(field, {})
        if not isinstance(check, dict):
            errors.append(f"{field} is missing or malformed")
            continue
        for key, expected in (("passed", True), ("criterion", "criterion_v2"),
                              ("failing_batches", 0), ("nonfinite_count", 0),
                              ("checked_elements", n * batch)):
            value = check.get(key)
            if value != expected or (key == "passed" and value is not True):
                errors.append(f"{field}.{key}: expected {expected!r}, received {value!r}")
        for metric, threshold in (("worst_batch_l2_scaled_error", 1e-5),
                                  ("worst_batch_linf_input_scaled_error", 1e-4)):
            value = check.get(metric)
            if not finite_number(value) or not 0 <= value <= threshold:
                errors.append(f"{field}.{metric} violates criterion_v2")
        if pattern == "zero" and check.get("max_absolute_error") != 0:
            errors.append(f"{field}: zero input must have exactly zero output")
    return errors


def parse_result(stdout):
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected one JSON result line, received {len(lines)}")
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    result = json.loads(lines[0], parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("harness result is not a JSON object")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not args.harness.is_file():
        parser.error("--harness must identify an existing executable")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    # The caller owns GPU selection. This script never enumerates or picks GPUs.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or "," in visible:
        parser.error("caller must set CUDA_VISIBLE_DEVICES to exactly one approved GPU")
    harness = args.harness.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "analytic_probes.jsonl"
    summary_path = args.output_dir / "analytic_probes_summary.json"
    if output.exists() or summary_path.exists():
        parser.error("output files already exist; use a fresh --output-dir")
    started = utc()
    failures = []
    passed = 0
    pattern_metrics = {pattern: dict(worst_l2=0.0, worst_linf=0.0, passed=0, failed=0)
                       for pattern in PATTERNS}
    with output.open("w", encoding="utf-8") as file:
        for index, (pattern, n, batch, (candidate, knobs)) in enumerate(
                itertools.product(PATTERNS, LENGTHS, BATCHES, CANDIDATES), 1):
            probe_id = f"{pattern}_n{n}_b{batch}_{candidate}"
            command = [str(harness), "--n", str(n), "--batch", str(batch),
                       "--pattern", pattern, "--placement", "outofplace", "--timing-mode", "graph",
                       "--graph-repeats", "4", "--warmup-ms", "0", "--warmup", "1",
                       "--repeats", "2", "--seed", "42", *knobs]
            record = dict(probe_id=probe_id, probe_index=index, started_at=utc(),
                          pattern=pattern, n=n, batch=batch, candidate_id=candidate,
                          purpose="analytic_correctness_validation_excluded_from_timing_study", command=command)
            begin = time.monotonic()
            errors = []
            try:
                process = subprocess.run(command, capture_output=True, text=True,
                                         timeout=args.timeout, check=False)
                record["returncode"] = process.returncode
                result = parse_result(process.stdout)
                record["result"] = result
                errors.extend(validate(result, n, batch, pattern))
                if process.returncode != 0:
                    errors.append(f"harness returned exit code {process.returncode}")
                if process.stderr:
                    record["stderr_tail"] = process.stderr[-20000:]
                for field in ("single_forward_correctness", "correctness"):
                    check = result.get(field, {})
                    for key, metric in (("worst_l2", "worst_batch_l2_scaled_error"),
                                        ("worst_linf", "worst_batch_linf_input_scaled_error")):
                        value = check.get(metric) if isinstance(check, dict) else None
                        if finite_number(value):
                            pattern_metrics[pattern][key] = max(pattern_metrics[pattern][key], value)
            except (OSError, subprocess.TimeoutExpired, ValueError) as error:
                errors.append(str(error))
            record["wall_seconds"] = time.monotonic() - begin
            record["validation_errors"] = errors
            record["validation_passed"] = not errors
            file.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
            if errors:
                failures.append(dict(probe_id=probe_id, errors=errors))
                pattern_metrics[pattern]["failed"] += 1
            else:
                passed += 1
                pattern_metrics[pattern]["passed"] += 1
            print(f"[{index}/64] {probe_id}: {'FAILED' if errors else 'passed'}", flush=True)
    summary = dict(schema_version=1, started_at=started, finished_at=utc(), total=64,
                   passed=passed, failed=len(failures), all_passed=passed == 64,
                   purpose="analytic_correctness_validation_excluded_from_timing_study",
                   harness=str(harness), harness_sha256=sha256(harness),
                   cuda_visible_devices=visible, criterion="criterion_v2",
                   placement="outofplace", timing_mode="graph", graph_repeats=4,
                   warmup_ms=0, warmup=1, repeats=2, seed=42,
                   lengths=LENGTHS, batches=BATCHES, patterns=PATTERNS,
                   candidates=[candidate for candidate, _ in CANDIDATES],
                   pattern_metrics=pattern_metrics, failures=failures, records_file=str(output.resolve()))
    summary_path.write_text(json.dumps(summary, allow_nan=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"Analytic probes: {passed}/64 passed; summary: {summary_path}")
    return 0 if passed == 64 else 1


if __name__ == "__main__":
    sys.exit(main())
