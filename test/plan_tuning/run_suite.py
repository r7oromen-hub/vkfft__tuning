#!/usr/bin/env python3
"""Reproducible, resumable VkFFT candidate screen and independent confirmation.

Each subprocess initializes a fresh plan. Device timings therefore describe the
harness's restored-input cache regime, while initialize_ms includes compilation.
No model or execution savings from avoiding plan compilation are measured here.
"""
import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import math
import subprocess
import sys
import time


SCHEMA_VERSION = 1
GRID = [dict(aim_threads=a, register_boost=r, use_lut=l)
        for a, r, l in itertools.product((64, 128, 256), (1, 2, 4), (-1, 1))]
CANDIDATES = [("default", {})] + [
    (f"a{c['aim_threads']}_r{c['register_boost']}_l{c['use_lut']}", c) for c in GRID]


def workload_id(w):
    return f"n{w['n']}_b{w['batch']}"


def workloads(preset, max_input_mib):
    if preset == "smoke":
        raw = [(256, 1), (768, 1), (1280, 1), (16384, 16), (49152, 16), (81920, 16)]
    else:
        lengths = ([256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576]
                   + [768, 3072, 12288, 49152, 196608, 786432]
                   + [1280, 5120, 20480, 81920, 327680, 655360])
        raw = [(n, b) for n in sorted(lengths)
               for b in (1, max(1, int(max_input_mib * 1024 ** 2) // (8 * n)))]
    return [dict(n=n, batch=b, family=("pow2" if n & (n - 1) == 0 else ("mixed235" if n % 5 == 0 else "mixed23")),
                 input_bytes=n * b * 8)
            for n, b in raw if n * b * 8 <= max_input_mib * 1024 ** 2]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_json(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, allow_nan=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_records(path):
    rows = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A truncated trailing write after interruption is recoverable.
                    if line_no != len(lines):
                        raise ValueError(f"malformed non-trailing JSONL line {line_no}")
                    print(f"Warning: ignoring incomplete trailing JSONL line {line_no}", file=sys.stderr)
    return rows


def repair_trailing_write(path):
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    tail = data.rsplit(b"\n", 1)[-1]
    try:
        json.loads(tail)
    except (json.JSONDecodeError, UnicodeDecodeError):
        path.write_bytes(data[:len(data) - len(tail)])
        print("Recovered truncated final JSONL write", file=sys.stderr)
    else:
        with path.open("ab") as f:
            f.write(b"\n")


def key(row):
    return (row["phase"], row["workload_id"], row["candidate_id"], row["round"])


def negative_control(rows):
    baseline = next((r for r in rows if r["candidate_id"] == "default"), None)
    if baseline is None:
        return None
    base = baseline["result"]
    matches = [r["candidate_id"] for r in rows if r["candidate_id"] != "default"
               and bool(base.get("resolved_config")) and bool(base.get("structural_features"))
               and r["result"].get("resolved_config") == base["resolved_config"]
               and r["result"].get("structural_features") == base["structural_features"]]
    preferred = "a128_r1_l-1"
    return preferred if preferred in matches else (sorted(matches)[0] if matches else None)


def result_ok(result):
    import math
    samples = result.get("exec_ms_samples", [])
    return (result.get("status") == "ok" and result.get("correctness", {}).get("passed") is True
            and len(samples) > 0 and all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                                         for x in samples)
            and isinstance(result.get("exec_ms_median"), (int, float))
            and math.isfinite(result["exec_ms_median"]) and result["exec_ms_median"] > 0)


def parse_result(stdout):
    for line in reversed(stdout.splitlines()):
        try:
            result = json.loads(line)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    raise ValueError("harness produced no JSON object")


def run_one(args, workload, candidate, phase, round_no, order):
    cid, config = candidate
    seed_text = f"{args.seed}:{phase}:{workload_id(workload)}:{round_no}"
    # Same input for each candidate in a phase/round, independent confirmation input.
    input_seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], "little")
    command = [str(Path(args.exe).resolve()), "--n", str(workload["n"]),
               "--batch", str(workload["batch"]), "--seed", str(input_seed),
               "--warmup", str(args.warmup if phase == "screen" or phase.startswith("calibrat") else args.confirm_warmup),
               "--repeats", str(args.repeats if phase == "screen" or phase.startswith("calibrat") else args.confirm_repeats),
               "--timing-mode", args.timing_mode, "--warmup-ms", str(args.warmup_ms),
               "--placement", args.placement, "--graph-repeats", str(workload.get("graph_repeats", 1))]
    for field, value in config.items():
        command.extend(["--" + field.replace("_", "-"), str(value)])
    row = dict(schema_version=SCHEMA_VERSION, phase=phase, round=round_no,
               workload_id=workload_id(workload), workload=workload,
               candidate_id=cid, config=config, input_seed=input_seed, order=order,
               started_utc=utc(), command=command)
    start = time.perf_counter()
    try:
        p = subprocess.run(command, capture_output=True, text=True,
                           timeout=args.timeout, check=False)
        row.update(returncode=p.returncode, stderr=p.stderr[-16000:])
        result = parse_result(p.stdout)
        row["result"] = result
        if result.get("n") != workload["n"] or result.get("batch") != workload["batch"]:
            raise ValueError("harness workload does not match requested N/batch")
        if result.get("timing_mode") != args.timing_mode:
            raise ValueError("harness timing mode does not match requested protocol")
        if result.get("placement") != args.placement or result.get("graph_repeats") != workload.get("graph_repeats", 1):
            raise ValueError("harness placement/graph repeats do not match requested protocol")
        row["status"] = "ok" if p.returncode == 0 and result_ok(result) else "failed"
        if row["status"] != "ok":
            row["error"] = result.get("error", "nonzero exit, correctness failure, or invalid timing")
    except subprocess.TimeoutExpired as exc:
        row.update(status="timeout", error=f"exceeded {args.timeout}s", stderr=str(exc.stderr or "")[-16000:])
    except (OSError, ValueError) as exc:
        row.update(status="failed", error=str(exc))
    row.update(wall_seconds=time.perf_counter() - start, ended_utc=utc())
    return row


def add_protocol_args(parser):
    parser.add_argument("--timing-mode", choices=("direct", "graph"), default="graph")
    parser.add_argument("--placement", choices=("inplace", "outofplace"), default="outofplace")
    parser.add_argument("--warmup-ms", type=float, default=100)
    parser.add_argument("--graph-repeats", type=int, default=0,
                        help="0 calibrates default R1 once/workload; fixed positive R overrides calibration")
    parser.add_argument("--graph-target-ms", type=float, default=.5)
    parser.add_argument("--graph-max-repeats", type=int, default=2048)


def protocol_manifest(args):
    if (args.graph_repeats < 0 or args.graph_target_ms <= 0 or args.graph_max_repeats < 1
            or (args.graph_repeats > 1 and (args.placement != "outofplace" or args.timing_mode != "graph"))):
        raise ValueError("invalid graph repeat settings; R>1 requires graph+outofplace")
    return dict(timing_mode=args.timing_mode, placement=args.placement, warmup_ms=args.warmup_ms,
                graph_repeats_requested=args.graph_repeats, graph_target_ms=args.graph_target_ms,
                graph_max_repeats=args.graph_max_repeats,
                graph_repeat_policy=("default_R1_calibration_then_up_to_3_final_R_probes_capped"
                                     if args.graph_repeats == 0 and args.timing_mode == "graph" and args.placement == "outofplace"
                                     else ("fixed" if args.graph_repeats else "single_forward_protocol")))


def prepare_protocol(args, ws, output):
    """Calibrate default once per workload; freeze R across every candidate/round."""
    fixed = args.graph_repeats
    if args.timing_mode != "graph" or args.placement != "outofplace":
        fixed = max(1, fixed)
    if fixed:
        return [dict(w, graph_repeats=fixed) for w in ws]
    calibration_path = output / "calibration.jsonl"
    repair_trailing_write(calibration_path)
    existing = {r["workload_id"]:r for r in read_records(calibration_path)}
    verification_path = output / "calibration_verification.jsonl"
    repair_trailing_write(verification_path)
    verified = {(r["workload_id"],r["round"]):r for r in read_records(verification_path)}
    chosen = {}
    diagnostics = {}
    for w in ws:
        wid = workload_id(w)
        if args.dry_run:
            chosen[wid] = None
            continue
        row = existing.get(wid)
        if row is None or (row["status"] != "ok" and args.retry_failures):
            row = run_one(args, dict(w, graph_repeats=1), ("default", {}), "calibrate", 0, len(existing))
            append_json(calibration_path, row)
            existing[wid] = row
            print(f"calibration {wid} {row['status']} {row.get('result',{}).get('exec_ms_median','')}",flush=True)
        if row["status"] != "ok":
            raise RuntimeError(f"default calibration failed for {wid}; inspect calibration.jsonl; fix protocol or use --retry-failures")
        repeats = min(args.graph_max_repeats, max(1, math.ceil(args.graph_target_ms / row["result"]["exec_ms_median"])))
        for probe in range(3):
            probe_key = (wid,probe)
            verification = verified.get(probe_key)
            if verification is not None and verification["workload"].get("graph_repeats") != repeats:
                raise RuntimeError(f"verification repeat choice changed during resume for {wid}")
            if verification is None or (verification["status"] != "ok" and args.retry_failures):
                verification = run_one(args, dict(w,graph_repeats=repeats),("default",{}),
                                       "calibrate_verify",probe,len(verified))
                append_json(verification_path,verification)
                verified[probe_key] = verification
                print(f"calibration verify {wid} R={repeats} probe={probe} {verification['status']} "
                      f"raw_ms={verification.get('result',{}).get('raw_interval_ms_median','')}",flush=True)
            if verification["status"] != "ok":
                raise RuntimeError(f"default final-R verification failed for {wid}; inspect calibration_verification.jsonl")
            raw_ms = verification["result"].get("raw_interval_ms_median")
            if not isinstance(raw_ms,(int,float)) or not math.isfinite(raw_ms) or raw_ms <= 0:
                raise RuntimeError(f"missing/invalid raw graph interval in final-R verification for {wid}")
            reached = raw_ms >= args.graph_target_ms
            if reached or repeats == args.graph_max_repeats or probe == 2:
                diagnostics[wid] = dict(graph_repeats=repeats,verified_default_raw_interval_ms=raw_ms,
                                       target_ms=args.graph_target_ms,target_met=reached,
                                       capped=repeats==args.graph_max_repeats,probes=probe+1,
                                       interpretation="Verified default only; target is not guaranteed for faster candidates or subsequent measurements.")
                break
            repeats = min(args.graph_max_repeats,max(repeats+1,math.ceil(repeats*args.graph_target_ms/raw_ms)))
        chosen[wid] = repeats
    chosen_path = output / "graph_repeats.json"
    if not args.dry_run:
        if chosen_path.exists() and json.loads(chosen_path.read_text(encoding="utf-8")) != chosen:
            raise RuntimeError("calibrated repeat choices changed during resume")
        chosen_path.write_text(json.dumps(chosen,indent=2)+"\n",encoding="utf-8")
        (output / "calibration_summary.json").write_text(json.dumps(diagnostics,indent=2)+"\n",encoding="utf-8")
    return [dict(w, graph_repeats=chosen[workload_id(w)]) for w in ws]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    p.add_argument("--workloads", type=Path, help="JSON list of {n,batch,family?} overriding preset")
    p.add_argument("--max-input-mib", type=float, default=256,
                   help="Input allocation cap; harness temporary/compile memory is additional")
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--warmup", type=int, default=5)
    add_protocol_args(p)
    p.add_argument("--repeats", type=int, default=31)
    p.add_argument("--confirm-warmup", type=int, default=5)
    p.add_argument("--confirm-repeats", type=int, default=101)
    p.add_argument("--confirm-rounds", type=int, default=3)
    p.add_argument("--phase", choices=("all", "screen", "confirm"), default="all")
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--retry-failures", action="store_true")
    p.add_argument("--limit-workloads", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--source-commit", default="unknown")
    args = p.parse_args()
    if args.workloads:
        ws = json.loads(args.workloads.read_text(encoding="utf-8"))
        ws = [dict(w, input_bytes=w["n"] * w["batch"] * 8,
                   family=w.get("family", "custom")) for w in ws
              if w["n"] * w["batch"] * 8 <= args.max_input_mib * 1024 ** 2]
    else:
        ws = workloads(args.preset, args.max_input_mib)
    if args.limit_workloads:
        ws = ws[:args.limit_workloads]
    if not ws or len({workload_id(w) for w in ws}) != len(ws):
        p.error("workloads must be nonempty and unique")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = dict(schema_version=SCHEMA_VERSION, workloads=ws, candidates=CANDIDATES,
                    seed=args.seed, source_commit=args.source_commit,
                    warmup=args.warmup, repeats=args.repeats,
                    **protocol_manifest(args),
                    confirm_warmup=args.confirm_warmup, confirm_repeats=args.confirm_repeats,
                    confirm_rounds=args.confirm_rounds,
                    input_regime=("host-to-device restore before each graph sample, R immutable-input out-of-place forwards, restore outside events"
                                  if args.placement == "outofplace" and args.timing_mode == "graph"
                                  else "host-to-device restore before each single timed forward, restore outside events"),
                    initialize_includes_compile=True,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    executable=str(Path(args.exe).resolve()))
    if Path(args.exe).is_file():
        manifest["executable_sha256"] = hashlib.sha256(Path(args.exe).read_bytes()).hexdigest()
    manifest = json.loads(json.dumps(manifest))
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            p.error("resume manifest differs; use a new output directory or identical measurement settings")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    ws = prepare_protocol(args, ws, args.output)
    results_path = args.output / "measurements.jsonl"
    repair_trailing_write(results_path)
    records = {key(row): row for row in read_records(results_path)}
    order = len(records)

    def execute(w, candidate, phase, round_no):
        nonlocal order
        k = (phase, workload_id(w), candidate[0], round_no)
        if k in records and (records[k]["status"] == "ok" or not args.retry_failures):
            return
        if args.dry_run:
            print(json.dumps(dict(phase=phase, workload=w, candidate=candidate, round=round_no)))
            return
        row = run_one(args, w, candidate, phase, round_no, order)
        order += 1
        append_json(results_path, row)
        records[k] = row
        print(f"{order:5d} {phase:7s} {workload_id(w):20s} {candidate[0]:16s} {row['status']} "
              f"{row.get('result', {}).get('exec_ms_median', '')}", flush=True)

    rng = random.Random(args.seed)
    jobs = [(w, c) for w in ws for c in CANDIDATES]
    rng.shuffle(jobs)
    if args.phase in ("all", "screen"):
        for w, c in jobs:
            execute(w, c, "screen", 0)
    if args.phase in ("all", "confirm"):
        selected = {}
        controls = {}
        by_id = dict(CANDIDATES)
        for w in ws:
            valid = [r for r in records.values() if r["phase"] == "screen"
                     and r["workload_id"] == workload_id(w) and r["status"] == "ok"]
            if not valid:
                continue
            winner = min(valid, key=lambda r: r["result"]["exec_ms_median"])["candidate_id"]
            selected[workload_id(w)] = winner
            controls[workload_id(w)] = negative_control(valid)
        (args.output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        (args.output / "negative_controls.json").write_text(json.dumps(controls, indent=2) + "\n", encoding="utf-8")
        for round_no in range(args.confirm_rounds):
            order_ws = list(ws)
            rng.shuffle(order_ws)
            for w in order_ws:
                winner = selected.get(workload_id(w))
                if winner is None:
                    continue
                candidate_ids = list(dict.fromkeys(["default", winner] + ([controls[workload_id(w)]] if controls.get(workload_id(w)) else [])))
                pair = [(cid, by_id[cid]) for cid in candidate_ids]
                rng.shuffle(pair)
                for c in pair:
                    execute(w, c, "confirm", round_no)
    print(f"Saved {results_path}", flush=True)


if __name__ == "__main__":
    main()
