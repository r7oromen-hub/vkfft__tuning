#!/usr/bin/env python3
"""Read-only, standard-library status snapshot of the remote plan experiment."""
import argparse
from collections import Counter
import datetime as dt
import json
import math
import os
from pathlib import Path


def read_json(path, errors):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        errors.append(dict(file=path.name, error=str(error)[:300]))
        return {}


def read_rows(path, errors):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return [], 0
    except OSError as error:
        errors.append(dict(file=path.name, error=str(error)[:300]))
        return [], 0
    lines = text.splitlines()
    rows, partial = [], 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL row is not an object")
            rows.append(row)
        except ValueError as error:
            if index == len(lines) - 1 and not text.endswith("\n"):
                partial += 1
            else:
                errors.append(dict(file=path.name, line=index + 1, error=str(error)[:300]))
    # A later retry supersedes the same phase/workload/candidate/round.
    latest = {}
    for index, row in enumerate(rows):
        key = tuple(row.get(field) for field in ("phase", "workload_id", "candidate_id", "round"))
        if key == (None, None, None, None):
            key = ("unkeyed_row", index)
        latest[key] = row
    return list(latest.values()), partial


def mean_wall(rows):
    values = [row.get("wall_seconds") for row in rows]
    values = [value for value in values if isinstance(value, (int, float))
              and not isinstance(value, bool) and math.isfinite(value) and value >= 0]
    return sum(values) / len(values) if values else None


def process_status(path):
    result = dict(pid=None, alive=None)
    try:
        pid = int(path.read_text(encoding="ascii").strip())
        if pid <= 0:
            raise ValueError("PID must be positive")
        result["pid"] = pid
        # Python's Windows os.kill behavior differs from POSIX. A remote PID
        # also has no meaning in a downloaded local mirror.
        if os.name != "posix":
            result["note"] = "process_probe_requires_the_remote_POSIX_host"
            return result
        try:
            os.kill(pid, 0)
            result["alive"] = True
        except ProcessLookupError:
            result["alive"] = False
        except PermissionError:
            result.update(alive=True, note="process_exists_but_permission_denied")
    except FileNotFoundError:
        result["note"] = "PID_file_missing"
    except (OSError, ValueError) as error:
        result["error"] = str(error)[:300]
    return result


def snapshot(root, folder=None):
    folder = folder if folder is not None else root / "results" / "full_outofplace_graph_v2"
    errors = []
    manifest = read_json(folder / "manifest.json", errors)
    if not isinstance(manifest, dict):
        errors.append(dict(file="manifest.json", error="manifest is not an object"))
        manifest = {}
    workloads, candidates = manifest.get("workloads", []), manifest.get("candidates", [])
    planned = len(workloads) * len(candidates) if isinstance(workloads, list) and isinstance(candidates, list) and workloads and candidates else 912
    calibration, partial_a = read_rows(folder / "calibration.jsonl", errors)
    verification, partial_b = read_rows(folder / "calibration_verification.jsonl", errors)
    measurements, partial_c = read_rows(folder / "measurements.jsonl", errors)
    screen = [row for row in measurements if row.get("phase") == "screen"]
    confirm = [row for row in measurements if row.get("phase") == "confirm"]
    selected = read_json(folder / "selected.json", errors)
    controls = read_json(folder / "negative_controls.json", errors)
    rounds = manifest.get("confirm_rounds")
    workload_ids = [f"n{workload['n']}_b{workload['batch']}" for workload in workloads
                    if isinstance(workload, dict) and "n" in workload and "batch" in workload]
    planning_ready = (bool(workload_ids) and isinstance(selected, dict) and isinstance(controls, dict)
                      and isinstance(rounds, int) and not isinstance(rounds, bool) and rounds >= 0
                      and all(selected.get(wid) and wid in controls for wid in workload_ids))
    expected_confirm = None
    if planning_ready:
        expected_confirm = {(wid, cid, round_no)
                            for wid in workload_ids
                            for cid in {"default", selected[wid], controls[wid]} if cid
                            for round_no in range(rounds)}
    confirm_by_key = {(row.get("workload_id"), row.get("candidate_id"), row.get("round")): row for row in confirm}
    expected_confirm_rows = ([confirm_by_key[key] for key in expected_confirm if key in confirm_by_key]
                             if expected_confirm is not None else confirm)
    confirm_planned = len(expected_confirm) if expected_confirm is not None else None
    confirm_done = len(expected_confirm_rows)
    confirm_remaining = max(0, confirm_planned - confirm_done) if confirm_planned is not None else None
    all_rows = calibration + verification + measurements
    failures = [row for row in all_rows if row.get("status") != "ok"]
    screen_mean = mean_wall(screen)
    remaining = max(0, planned - len(screen))
    eta = remaining * screen_mean if screen_mean is not None else (0 if remaining == 0 else None)
    confirm_mean = mean_wall(confirm)
    confirm_eta_mean = confirm_mean if confirm_mean is not None else screen_mean
    confirm_eta = (confirm_remaining * confirm_eta_mean
                   if confirm_remaining is not None and confirm_eta_mean is not None
                   else (0 if confirm_remaining == 0 else None))
    total_eta = eta + confirm_eta if eta is not None and confirm_eta is not None else None
    # Completion is based on expected record coverage and statuses, never on
    # signal-zero process existence (which can remain true for a zombie PID).
    candidate_ids = [candidate[0] for candidate in candidates
                     if isinstance(candidate, (list, tuple)) and candidate]
    expected_screen = {(wid, cid, 0) for wid in workload_ids for cid in candidate_ids}
    screen_by_key = {(row.get("workload_id"), row.get("candidate_id"), row.get("round")): row for row in screen}
    screen_complete = bool(expected_screen) and expected_screen <= screen_by_key.keys()
    confirmations_complete = expected_confirm is not None and expected_confirm <= confirm_by_key.keys()
    records_complete = screen_complete and confirmations_complete
    records_ok = (records_complete and all(screen_by_key[key].get("status") == "ok" for key in expected_screen)
                  and all(confirm_by_key[key].get("status") == "ok" for key in expected_confirm))
    now = dt.datetime.now(dt.timezone.utc)
    result = dict(timestamp_utc=now.isoformat(), root=str(root.resolve()),
                  runner=process_status(root / "results" / "full.pid"),
                  calibration=dict(done=len(calibration), statuses=dict(Counter(row.get("status", "unknown") for row in calibration))),
                  calibration_verification=dict(done=len(verification), statuses=dict(Counter(row.get("status", "unknown") for row in verification))),
                  screen=dict(done=len(screen), planned=planned, progress=f"{len(screen)}/{planned}",
                              successful=sum(row.get("status") == "ok" for row in screen), remaining=remaining),
                  confirmations_count=len(confirm), measurement_count=len(measurements),
                  confirmations=dict(done=confirm_done, planned=confirm_planned,
                                     progress=f"{confirm_done}/{confirm_planned}" if confirm_planned is not None else None,
                                     remaining=confirm_remaining,
                                     successful=sum(row.get("status") == "ok" for row in expected_confirm_rows),
                                     rounds=rounds, planning_ready=planning_ready,
                                     per_round=confirm_planned // rounds if confirm_planned is not None and rounds else 0),
                  expected_records_complete=records_complete, all_expected_records_ok=records_ok,
                  completion_status=("complete" if records_ok else "records_complete_with_failures" if records_complete else "in_progress"),
                  runner_probe_interpretation="signal_zero_only_checks_process_existence_including_possible_zombies;_completion_uses_expected_records_and_statuses",
                  statuses=dict(Counter(row.get("status", "unknown") for row in all_rows)),
                  failure_count=len(failures),
                  first_failures=[{key: row.get(key) for key in ("phase", "workload_id", "candidate_id", "round", "status")}
                                  | {"error": str(row.get("error", ""))[:400]} for row in failures[:5]],
                  mean_wall_seconds=mean_wall(measurements), mean_screen_wall_seconds=screen_mean,
                  mean_confirmation_wall_seconds=confirm_mean,
                  remaining_screen_eta_seconds=eta,
                  remaining_screen_eta_utc=(now + dt.timedelta(seconds=eta)).isoformat() if eta is not None else None,
                  remaining_confirmation_eta_seconds=confirm_eta,
                  remaining_confirmation_eta_utc=(now + dt.timedelta(seconds=confirm_eta)).isoformat() if confirm_eta is not None else None,
                  confirmation_eta_basis=("approximate_observed_confirmation_mean_process_wall_time" if confirm_mean is not None
                                          else "approximate_screen_mean_fallback_no_confirmation_wall_samples"),
                  remaining_total_eta_seconds=total_eta,
                  eta_basis="approximate_observed_mean_process_wall_time_times_remaining_screen_records_excludes_confirmation",
                  ignored_incomplete_trailing_rows=partial_a + partial_b + partial_c,
                  read_error_count=len(errors), first_read_errors=errors[:5])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="remote experiment root (default: current directory)")
    args = parser.parse_args()
    print(json.dumps(snapshot(args.root), ensure_ascii=False, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
