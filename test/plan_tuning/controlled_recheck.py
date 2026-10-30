#!/usr/bin/env python3
"""Independent interleaved process pairs for timing controls or fixed comparisons.

This stdlib-only runner performs no winner selection. Use matching resolved
configuration and planner descriptors for negative-control interpretation.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics

from run_suite import (CANDIDATES, append_json, key, read_records,
                       repair_trailing_write, run_one, workload_id,
                       add_protocol_args, protocol_manifest, prepare_protocol)


def summarize(records, output):
    pairs = []
    by_key = {key(r):r for r in records}
    for row in records:
        if row["candidate_id"] == "default":
            continue
        baseline = by_key.get((row["phase"], row["workload_id"], "default", row["round"]))
        if baseline is None or baseline["status"] != "ok" or row["status"] != "ok":
            continue
        a, b = baseline["result"], row["result"]
        pairs.append(dict(workload_id=row["workload_id"], n=row["workload"]["n"], batch=row["workload"]["batch"],
                          candidate_id=row["candidate_id"], round=row["round"],
                          default_ms=a["exec_ms_median"], candidate_ms=b["exec_ms_median"],
                          graph_repeats=b.get("graph_repeats",1),
                          default_raw_interval_ms=a.get("raw_interval_ms_median"),
                          candidate_raw_interval_ms=b.get("raw_interval_ms_median"),
                          default_over_candidate=a["exec_ms_median"] / b["exec_ms_median"],
                          descriptor_equal=bool(a.get("structural_features")) and a.get("structural_features") == b.get("structural_features"),
                          resolved_equal=bool(a.get("resolved_config")) and a.get("resolved_config") == b.get("resolved_config"),
                          default_order=baseline["order"], candidate_order=row["order"]))
    if pairs:
        with (output / "control_pairs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    aggregates = []
    for wid, cid in sorted({(r["workload_id"], r["candidate_id"]) for r in pairs}):
        subset = [r for r in pairs if (r["workload_id"],r["candidate_id"]) == (wid,cid)]
        ratios = [r["default_over_candidate"] for r in subset]
        logs = [math.log(x) for x in ratios]
        aggregates.append(dict(workload_id=wid, candidate_id=cid, pairs=len(ratios),
                               ratio_gmean=math.exp(statistics.mean(logs)),
                               ratio_min=min(ratios), ratio_max=max(ratios),
                               ratio_median=statistics.median(ratios),
                               max_absolute_deviation_from_one=max(abs(x-1) for x in ratios),
                               sample_std_log_ratio=statistics.stdev(logs) if len(logs)>1 else None,
                               all_descriptors_equal=all(r["descriptor_equal"] for r in subset),
                               all_resolved_equal=all(r["resolved_equal"] for r in subset)))
    report = dict(measurements=len(records), successful=sum(r["status"]=="ok" for r in records),
                  failed=sum(r["status"]!="ok" for r in records), paired_measurements=len(pairs),
                  workloads=aggregates,
                  interpretation="Fixed comparisons without winner selection. Matching resolved+recorded descriptors are negative controls for noise/drift, not proven identical generated code. Ranges are descriptive.")
    (output / "control_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workloads", required=True, type=Path, help="JSON list of {n,batch,family?}")
    p.add_argument("--candidates", default="default,a128_r1_l-1", help="Comma-separated suite IDs; must include default")
    p.add_argument("--rounds", type=int, default=10)
    add_protocol_args(p)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=101)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--max-input-mib", type=float, default=256)
    p.add_argument("--source-commit", default="unknown")
    p.add_argument("--retry-failures", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    candidates = list(dict.fromkeys(args.candidates.split(",")))
    configs = dict(CANDIDATES)
    if "default" not in candidates or any(cid not in configs for cid in candidates):
        p.error("candidate IDs must be in suite and include default")
    ws = json.loads(args.workloads.read_text(encoding="utf-8"))
    ws = [dict(w, input_bytes=w["n"]*w["batch"]*8, family=w.get("family","custom")) for w in ws]
    if (not ws or args.rounds<1 or len({workload_id(w) for w in ws}) != len(ws)
            or any(w["n"]<1 or w["batch"]<1 or w["input_bytes"]>args.max_input_mib*1024**2 for w in ws)):
        p.error("invalid, duplicate or over-cap workload, or nonpositive round count")
    args.confirm_warmup, args.confirm_repeats = args.warmup, args.repeats
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, kind="controlled_recheck", workloads=ws,
                    candidates=[(cid, configs[cid]) for cid in candidates], rounds=args.rounds,
                    warmup=args.warmup, **protocol_manifest(args),
                    repeats=args.repeats, seed=args.seed, source_commit=args.source_commit,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    input_regime=("host-to-device restore before each graph sample, R immutable-input out-of-place forwards, restore outside events"
                                  if args.placement == "outofplace" and args.timing_mode == "graph"
                                  else "host-to-device restore before each single timed forward, restore outside events"),
                    executable=str(Path(args.exe).resolve()))
    if Path(args.exe).is_file():
        manifest["executable_sha256"] = hashlib.sha256(Path(args.exe).read_bytes()).hexdigest()
    manifest = json.loads(json.dumps(manifest))
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        p.error("resume manifest differs; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    ws = prepare_protocol(args,ws,args.output)
    measurements = args.output / "measurements.jsonl"
    repair_trailing_write(measurements)
    records = {key(r):r for r in read_records(measurements)}
    rng = random.Random(args.seed)
    order = len(records)
    for round_no in range(args.rounds):
        workload_order = list(ws)
        rng.shuffle(workload_order)
        for w in workload_order:
            candidate_order = list(candidates)
            rng.shuffle(candidate_order)
            for cid in candidate_order:
                k = ("control", workload_id(w), cid, round_no)
                if k in records and (records[k]["status"] == "ok" or not args.retry_failures):
                    continue
                if args.dry_run:
                    print(json.dumps(dict(workload=w,candidate=cid,round=round_no,timing_mode=args.timing_mode)))
                    continue
                row = run_one(args,w,(cid,configs[cid]),"control",round_no,order)
                append_json(measurements,row)
                records[k] = row
                order += 1
                print(f"{order:4d} control {workload_id(w):20s} {cid:16s} round={round_no} {row['status']} {row.get('result',{}).get('exec_ms_median','')}",flush=True)
    report = summarize(list(records.values()),args.output)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
