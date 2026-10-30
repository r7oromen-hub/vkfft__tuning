"""CPU-only tests for the independent cloud reproduction protocol.

No harness executable or GPU is invoked.  Synthetic results exercise the
controller's audit and state-machine boundaries, not GPU correctness.
"""
import argparse
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import cloud_reproduction as cloud
import cloud_reproduction_report as report
from study_common import canonical, input_seed, read_jsonl, sha, write_json
from test_runner_s0 import synthetic_result


HERE = Path(__file__).parent

WINNERS = {
    "n1024_b1": "default",
    "n16384_b1": "a128_r1_lm1_c16",
    "n16384_b2048": "a128_r2_l1_np1",
    "n49152_b1": "a128_r1_lm1",
    "n49152_b682": "default",
    "n655360_b1": "a128_r4_lm1_np1_s2_8192",
    "n655360_b51": "a128_r4_l1_np1_c64",
    "n1048576_b32": "a128_r1_lm1_c64",
    "n12288_b1": "a128_r4_lm1_np1_s2_8192",
    "n20480_b16": "default",
}
KEY = {"n12288_b1", "n16384_b1", "n16384_b2048", "n655360_b1", "n20480_b16"}
COST = ["n1024_b1", "n16384_b1", "n16384_b2048", "n49152_b1", "n49152_b682",
        "n655360_b1", "n655360_b51", "n1048576_b32"]


def design_fixture():
    original = json.loads((HERE / "cloud_reproduction_design.json").read_text(encoding="utf-8"))
    original["study_id"] = "synthetic-cloud-only"
    original["scope"]["device_uuid"] = "GPU-synthetic-cloud"
    original["measurement"]["seed_namespace"] = "synthetic-cloud-only"
    return original


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.design = design_fixture()

    def test_exact_execution_budget_and_candidate_identity(self):
        self.assertTrue(cloud.validate_design(self.design))
        schedule = cloud.build_schedule(self.design)
        self.assertEqual(sum(len(b["candidate_ids"]) for b in schedule), 199)
        self.assertEqual(len(schedule), 65)
        self.assertEqual(len({(b["workload_id"], b["round"]) for b in schedule}), 65)
        for block in schedule:
            wid, k, ids = block["workload_id"], block["round"], block["candidate_ids"]
            expected = {"default", "default_replica", WINNERS[wid]}
            if wid in ("n16384_b1", "n16384_b2048"):
                expected.add("a128_r4_l1_np1")
            self.assertEqual(set(ids), expected)
            self.assertEqual(len(ids), len(expected))
            self.assertEqual(ids[0], "default" if k % 2 == 0 else "default_replica")
            self.assertEqual(ids[-1], "default_replica" if k % 2 == 0 else "default")
        self.assertEqual(schedule, cloud.build_schedule(self.design))
        for wid in WINNERS:
            self.assertEqual({b["round"] for b in schedule if b["workload_id"] == wid},
                             set(range(10 if wid in KEY else 3)))

    def test_actual_design_matches_independent_freeze(self):
        path = HERE/"cloud_reproduction_design.json"
        frozen = json.loads((HERE/"cloud_reproduction_freeze.json").read_text(encoding="utf-8"))
        actual = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(frozen["design_sha256"], sha(path.read_bytes()))
        self.assertEqual(frozen["study_id"], actual["study_id"])
        self.assertTrue(cloud.validate_design(actual))
        self.assertEqual(actual["scope"]["device_uuid"], "GPU-8e235702-e0a9-1aed-0bea-13addf71e86a")
        self.assertEqual(actual["execution"]["frozen_winners"], WINNERS)
        self.assertEqual(len(actual["execution"]["directional_tests"]), 5)
        self.assertEqual(actual["execution"]["family_size"], 5)

    def test_invalid_budgets_and_requests_are_rejected(self):
        for mutate in (
            lambda d: d["execution"]["workloads"][1].update(rounds=9),
            lambda d: d["cost_profile"]["workloads"].pop(),
            lambda d: d["preflight"]["semantic_probes"]["groups"][0]["candidates"].pop(),
            lambda d: d["preflight"]["equivalence"]["cases"][0]["candidates"].pop(),
            lambda d: d["health"]["workloads"].pop(),
            lambda d: d["execution"]["frozen_winners"].update(n12288_b1="unlisted_candidate"),
            lambda d: d["execution"]["workloads"][1]["candidates"].append("unlisted_candidate"),
            lambda d: d["candidates"][0].update(disable_reorder_four_step=1),
        ):
            changed = copy.deepcopy(self.design)
            mutate(changed)
            with self.subTest(changed=changed), self.assertRaises((cloud.Stop, ValueError)):
                cloud.validate_design(changed)

    def test_scientific_negative_is_not_a_protocol_failure(self):
        self.assertEqual(cloud.cost_classification(dict(ok=True, errors=[])), "equivalent")
        self.assertEqual(cloud.cost_classification(dict(ok=False, errors=["describe_full_inequivalent:default"])), "negative")
        self.assertEqual(cloud.cost_classification(dict(ok=False, errors=["selected_materialization_changed_plan_or_source"])), "negative")
        self.assertEqual(cloud.cost_classification(dict(ok=False, errors=["describe_full_inequivalent:default", "describe:app_failed:0:default"])), "fatal")
        self.assertEqual(cloud.cost_classification(dict(ok=False, errors=["describe:describe_side_effect_counter_nonzero:0"])), "fatal")
        self.assertEqual(cloud.cost_classification(dict(ok=False, errors=[])), "fatal")


def fake_execution(args, workload, candidate, phase, round_no, order):
    """Return structurally valid numerical data without running a process."""
    cid, config = candidate
    seed = input_seed(args.design_data, workload.get("seed_phase", phase), workload["id"], round_no)
    result = synthetic_result(workload, cid, config, seed)
    result["gpu"]["uuid"] = args.design_data["scope"]["device_uuid"]
    if workload.get("binary_role") == "plain":
        result["postinit_source_kernels"] = [dict(
            axis=k["axis"], upload=k["upload"], canonical_descriptor=k["canonical_descriptor"],
            source_sha256=k["source_sha256"], expected_compiler_input=k["compiler_input"],
            actual_compiler_options=None, snapshot_stage="post_initialize_retained_source")
            for k in result["precompile_kernels"]]
        result["precompile_kernels"] = []
        result["header_instrumentation"] = False
    return dict(schema_version="plan-selection-process/v1", study_id=args.design_data["study_id"],
                phase=phase, workload_id=workload["id"], workload=workload, candidate_id=cid,
                config=config, round=round_no, order=order, block_id=workload.get("block_id"),
                block_attempt=workload.get("block_attempt", 0),
                position_in_block=workload.get("position_in_block"), input_seed=seed,
                status="ok", returncode=0, validation_errors=[], result=result,
                guard_id="synthetic-guard", guard_clean=True,
                guard_gpu_uuid=args.design_data["scope"]["device_uuid"],
                guard_log=os.environ.get("VKFFT_STUDY_GUARD_LOG", "synthetic.jsonl"),
                guard_run_id=f"{phase}:{workload['id']}:{cid}:{round_no}:{order}")


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE, prefix="cloud_cpu_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.design = design_fixture()
        self.design_path = self.folder/"design.json"
        self.freeze_path = self.folder/"freeze.json"
        self.exe = self.folder/"instrumented"
        self.plain = self.folder/"plain"
        self.provenance = self.folder/"build_provenance.json"
        self.exe.write_bytes(b"CPU-only instrumented placeholder")
        self.plain.write_bytes(b"CPU-only plain placeholder")
        write_json(self.provenance, dict(state="complete", upstream_commit=self.design["source"]["vkfft_commit"],
                   cuda_root="/usr/local/cuda-12.8", actual_cuda_versions=dict(nvrtc=[12, 8], runtime=12080),
                   original_header_files=["synthetic"], generated_header_provenance=dict(sha256="a"*64),
                   harness_source_sha256="b"*64,
                   artifacts={"vkfft_instrumented_benchmark":dict(sha256=sha(self.exe.read_bytes())),
                              "vkfft_uninstrumented_control":dict(sha256=sha(self.plain.read_bytes()))},
                   commands=[dict(returncode=0, output=self.design["scope"]["device_uuid"]+" 610.57.04")]))
        write_json(self.design_path, self.design)
        write_json(self.freeze_path, dict(study_id=self.design["study_id"], design_sha256=sha(self.design_path.read_bytes())))
        self.guard_log = self.folder/"cloud-output"/"guards"/"guard.jsonl"
        self.guard_log.parent.mkdir(parents=True)
        self.guard_log.write_text(json.dumps(dict(event="guard_start", guard_id="synthetic-guard", idle_seconds=30,
                                                 script="cloud_reproduction.py", script_args=["--stage", "equivalence"]))+"\n", encoding="utf-8")
        self.env = dict(VKFFT_STUDY_GUARD_ID="synthetic-guard", VKFFT_STUDY_GUARD_UUID=self.design["scope"]["device_uuid"],
                        VKFFT_STUDY_GUARD_LOG=str(self.guard_log))

    def args(self, stage):
        return argparse.Namespace(design=self.design_path, freeze=self.freeze_path, output=self.folder/"cloud-output",
                                  exe=str(self.exe), plain_exe=str(self.plain), build_provenance=self.provenance, stage=stage, timeout=180)

    def test_design_freeze_guard_and_binary_manifest_are_enforced(self):
        args = self.args("health")
        with patch.dict(os.environ, {key: "" for key in self.env}):
            with self.assertRaisesRegex(cloud.Stop, "GPU_guard_required"):
                cloud.CloudController(args)
        self.design_path.write_bytes(self.design_path.read_bytes()+b" ")
        with patch.dict(os.environ, self.env):
            with self.assertRaisesRegex(cloud.Stop, "design_freeze_mismatch"):
                cloud.CloudController(args)
        self.design_path.write_bytes(self.design_path.read_bytes()[:-1])
        with patch.dict(os.environ, self.env):
            cloud.CloudController(args)
            self.exe.write_bytes(b"modified executable")
            with self.assertRaisesRegex(cloud.Stop, "build_provenance_binary_mismatch"):
                cloud.CloudController(self.args("semantics"))

    def test_timeout_and_guard_history_prevent_unapproved_retry(self):
        argv = ["cloud_reproduction.py", "--design", "unused", "--freeze", "unused", "--output", "unused",
                "--exe", "unused", "--plain-exe", "unused", "--build-provenance", "unused",
                "--stage", "equivalence", "--timeout", "181"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cloud.main()
        old = self.guard_log.parent/"previous.jsonl"
        old.write_text(json.dumps(dict(event="guard_start", guard_id="prior-guard", idle_seconds=30,
                                       script="cloud_reproduction.py", script_args=["--stage", "health"]))+"\n")
        with patch.dict(os.environ, self.env):
            with self.assertRaisesRegex(cloud.Stop, "prior_guard_stage_attempt_forbidden:health"):
                cloud.CloudController(self.args("health"))

    def test_health_recalibrates_and_runs_exact_30_controls(self):
        args = self.args("calibration")
        with (patch.dict(os.environ, self.env), patch.object(cloud, "run_one", side_effect=fake_execution) as calls,
              patch.object(cloud, "audit_row", return_value=dict(ok=True, errors=[])), contextlib.redirect_stdout(io.StringIO())):
            study = cloud.CloudController(args)
            study.calibration()
            self.assertEqual(calls.call_count, 11)
            health = cloud.CloudController(self.args("health"))
            health.health()
            self.assertEqual(calls.call_count, 41)
        rows = read_jsonl(args.output/"calibration"/"measurements.jsonl")
        self.assertEqual(sum(r["phase"] == "calibration" for r in rows), 11)
        self.assertEqual(len(read_jsonl(args.output/"health"/"measurements.jsonl")), 30)
        repeats = json.loads((args.output/"graph_repeats.json").read_text(encoding="utf-8"))
        self.assertEqual(set(repeats), {w["id"] for w in self.design["calibration"]["workloads"]})
        self.assertTrue(all(v["graph_repeats"] == 1 and v["probes"] == 1 for v in repeats.values()))
        self.assertTrue(json.loads((args.output/"health"/"gate.json").read_text())["passed"])

    def test_preexisting_calibration_is_rejected_before_any_process(self):
        args = self.args("calibration")
        write_json(args.output/"graph_repeats.json", {"n1024_b1": {"graph_repeats": 999}})
        with (patch.dict(os.environ, self.env), patch.object(cloud, "run_one", side_effect=fake_execution) as calls,
              patch.object(cloud, "audit_row", return_value=dict(ok=True, errors=[]))):
            study = cloud.CloudController(args)
            with self.assertRaisesRegex(cloud.Stop, "preexisting_calibration_forbidden"):
                study.calibration()
            self.assertEqual(calls.call_count, 0)
        self.assertEqual(json.loads((args.output/"graph_repeats.json").read_text())["n1024_b1"]["graph_repeats"], 999)

    def test_calibration_caps_and_retains_last_measured_probe(self):
        args = self.args("calibration")
        def tiny_interval(*arguments):
            row = fake_execution(*arguments)
            row["result"]["raw_interval_ms_median"] = 0.00001
            return row
        with (patch.dict(os.environ, self.env), patch.object(cloud, "run_one", side_effect=tiny_interval) as calls,
              patch.object(cloud, "audit_row", return_value=dict(ok=True, errors=[])), contextlib.redirect_stdout(io.StringIO())):
            study = cloud.CloudController(args)
            frozen = study.calibrate([study.workloads["n1024_b1"]])["n1024_b1"]
            rows = read_jsonl(args.output/"calibration"/"measurements.jsonl")
            self.assertEqual(calls.call_count, 2)
            self.assertEqual([r["workload"]["graph_repeats"] for r in rows], [1, 2048])
            self.assertEqual(frozen["graph_repeats"], 2048)
            self.assertTrue(frozen["capped"])
            self.assertFalse(frozen["target_met"])
            self.assertEqual(frozen["source_record_sha256"], rows[-1]["record_sha256"])

    def test_successful_equivalence_and_semantics_have_exact_counts(self):
        with (patch.dict(os.environ, self.env), patch.object(cloud, "run_one", side_effect=fake_execution) as calls,
              patch.object(cloud, "audit_row", return_value=dict(ok=True, errors=[])), contextlib.redirect_stdout(io.StringIO())):
            eq = cloud.CloudController(self.args("equivalence"))
            eq.equivalence()
            self.assertEqual(calls.call_count, 8)
            sem = cloud.CloudController(self.args("semantics"))
            sem.semantics()
            self.assertEqual(calls.call_count, 36)
        eq_rows = read_jsonl(eq.measurements)
        sem_rows = read_jsonl(sem.measurements)
        self.assertEqual({r["workload"]["binary_role"] for r in eq_rows}, {"plain", "instrumented"})
        self.assertTrue(all(r["workload"]["graph_repeats"] == 16 for r in eq_rows+sem_rows))
        self.assertEqual({r["workload"]["pattern"] for r in sem_rows}, set(self.design["preflight"]["semantic_probes"]["patterns"]))

    def test_prerequisites_reaudit_session_identity_and_complete_coverage(self):
        with (patch.dict(os.environ, self.env), patch.object(cloud, "run_one", side_effect=fake_execution),
              patch.object(cloud, "audit_row", return_value=dict(ok=True, errors=[])),
              patch.object(cloud, "completed_session", return_value=True), contextlib.redirect_stdout(io.StringIO())):
            eq = cloud.CloudController(self.args("equivalence"))
            eq.equivalence()
            nxt = cloud.CloudController(self.args("semantics"))
            nxt.require("equivalence")
            rows = read_jsonl(eq.measurements)
            ledger = eq.measurements.read_bytes()
            gate_path = eq.stage_path/"gate.json"
            gate_bytes = gate_path.read_bytes()
            with patch.object(cloud, "completed_session", return_value=False):
                with self.assertRaisesRegex(cloud.Stop, "prerequisite_guard_incomplete"):
                    nxt.require("equivalence")
            # Even a count changed in the gate cannot legitimize a missing process.
            eq.measurements.write_bytes(b"".join(canonical(row)+b"\n" for row in rows[:-1]))
            gate = json.loads(gate_bytes)
            gate["record_count"] -= 1
            write_json(gate_path, gate)
            with self.assertRaisesRegex(cloud.Stop, "prerequisite_coverage_invalid"):
                nxt.require("equivalence")
            eq.measurements.write_bytes(ledger)
            gate_path.write_bytes(gate_bytes)
            # A rehashed row from another cohort still cannot be admitted.
            rows[0]["study_id"] = "old-cohort"
            rows[0].pop("record_sha256")
            rows[0]["record_sha256"] = sha(canonical(rows[0]))
            eq.measurements.write_bytes(b"".join(canonical(row)+b"\n" for row in rows))
            with self.assertRaisesRegex(cloud.Stop, "prerequisite_row_provenance_failed"):
                nxt.require("equivalence")

    def test_calibration_source_reaudited_before_health(self):
        with (patch.dict(os.environ, self.env), patch.object(cloud, "run_one", side_effect=fake_execution),
              patch.object(cloud, "audit_row", return_value=dict(ok=True, errors=[])),
              patch.object(cloud, "completed_session", return_value=True), contextlib.redirect_stdout(io.StringIO())):
            cal = cloud.CloudController(self.args("calibration"))
            cal.calibration()
            health = cloud.CloudController(self.args("health"))
            health.require("calibration")
            path = cal.output/"graph_repeats.json"
            frozen = json.loads(path.read_text())
            frozen["n1024_b1"]["graph_repeats"] = 128
            write_json(path, frozen)
            with self.assertRaisesRegex(cloud.Stop, "calibration_source_invalid"):
                health.require("calibration")

    def test_numerical_failure_is_persisted_and_stage_is_not_retried(self):
        args = self.args("semantics")
        def failed(*arguments):
            row = fake_execution(*arguments)
            row.update(status="failed", validation_errors=["numerical_failure:correctness"])
            return row
        with (patch.dict(os.environ, self.env), patch.object(cloud, "run_one", side_effect=failed) as calls,
              patch.object(cloud, "audit_row", return_value=dict(ok=True, errors=[])), contextlib.redirect_stdout(io.StringIO())):
            study = cloud.CloudController(args)
            with self.assertRaisesRegex(cloud.Stop, "process_failed"):
                study.semantics()
            rows = read_jsonl(args.output/"semantics"/"measurements.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["validation_errors"], ["numerical_failure:correctness"])
            before = (args.output/"semantics"/"measurements.jsonl").read_bytes()
            with self.assertRaisesRegex(cloud.Stop, "stage_already_attempted_no_automatic_retry"):
                cloud.CloudController(args)
            self.assertEqual(calls.call_count, 1)
            self.assertEqual((args.output/"semantics"/"measurements.jsonl").read_bytes(), before)

    def test_failed_negative_control_quarantines_entire_block(self):
        args = self.args("execution")
        def drifted(*arguments):
            row = fake_execution(*arguments)
            if row["candidate_id"] == "default_replica":
                row["result"]["exec_ms_median"] *= 1.1
                row["result"]["exec_ms_samples"] = [x*1.1 for x in row["result"]["exec_ms_samples"]]
            return row
        with (patch.dict(os.environ, self.env), patch.object(cloud, "run_one", side_effect=drifted),
              patch.object(cloud, "audit_row", return_value=dict(ok=True, errors=[])), contextlib.redirect_stdout(io.StringIO())):
            study = cloud.CloudController(args)
            w = study.workloads["n16384_b1"]
            with self.assertRaisesRegex(cloud.Stop, "whole_block_quality_failed"):
                study.block(w, "execution", 0, ["default", WINNERS[w["id"]], "default_replica"], 1, 101)
            rows = read_jsonl(args.output/"execution"/"measurements.jsonl")
            block = json.loads(next((args.output/"execution"/"blocks").glob("*.json")).read_text())
            self.assertEqual(len(rows), 3)
            self.assertEqual(block["record_hashes"], [r["record_sha256"] for r in rows])
            self.assertFalse(block["quality"]["ok"])
            self.assertIn("default_replica_ratio_exceeds_5pct", block["quality"]["errors"])

    def test_full_execution_populates_all_65_complete_blocks(self):
        args = self.args("execution")
        repeats = {w["id"]: dict(graph_repeats=1) for w in self.design["calibration"]["workloads"]}
        write_json(args.output/"graph_repeats.json", repeats)
        with (patch.dict(os.environ, self.env), patch.object(cloud, "run_one", side_effect=fake_execution) as calls,
              patch.object(cloud, "audit_row", return_value=dict(ok=True, errors=[])), contextlib.redirect_stdout(io.StringIO())):
            study = cloud.CloudController(args)
            study.execution()
            self.assertEqual(calls.call_count, 199)
        rows = read_jsonl(study.measurements)
        self.assertEqual(len(rows), 199)
        blocks = [json.loads(path.read_text()) for path in (study.stage_path/"blocks").glob("*.json")]
        self.assertEqual(len(blocks), 65)
        self.assertTrue(all(b["quality"]["ok"] for b in blocks))
        self.assertEqual({h for b in blocks for h in b["record_hashes"]}, {r["record_sha256"] for r in rows})
        self.assertTrue(all(r["study_id"] == self.design["study_id"] for r in rows))

    def test_describe_inequivalence_is_recorded_without_reselection(self):
        args = self.args("cost")
        negative = dict(ok=False, errors=["describe_full_inequivalent:default"], comparisons=[])
        with (patch.dict(os.environ, self.env), patch.object(cloud, "run_one", side_effect=fake_execution) as calls,
              patch.object(cloud, "audit_row", return_value=dict(ok=True, errors=[])),
              patch.object(cloud, "audit_pair", return_value=negative), contextlib.redirect_stdout(io.StringIO())):
            study = cloud.CloudController(args)
            study.cost()
            self.assertEqual(calls.call_count, 48)
        gate = json.loads((args.output/"cost"/"gate.json").read_text())
        self.assertEqual(gate["describe_feasibility"], "negative")
        self.assertEqual(len(gate["pairs"]), 24)
        self.assertTrue(all(p["selected_id"] == WINNERS[p["workload_id"]] for p in gate["pairs"]))
        self.assertEqual(len(list((args.output/"cost"/"pairs").glob("*.json"))), 24)
        for path in (args.output/"cost"/"candidate_orders").glob("*.tsv"):
            ids = [line.split("\t")[0] for line in path.read_text().splitlines()[1:]]
            self.assertEqual(len(ids), 26)
            self.assertEqual(set(ids), {c["id"] for c in self.design["candidates"]})
        for pair in gate["pairs"]:
            expected = ["full-search-cost", "describe-select-cost"]
            if pair["round"] % 2:
                expected.reverse()
            self.assertEqual(pair["candidate_ids"], expected)

    def test_protocol_damage_stops_after_pair_and_keeps_both_rows(self):
        args = self.args("cost")
        fatal = dict(ok=False, errors=["describe:describe_side_effect_counter_nonzero:0"], comparisons=[])
        with (patch.dict(os.environ, self.env), patch.object(cloud, "run_one", side_effect=fake_execution) as calls,
              patch.object(cloud, "audit_row", return_value=dict(ok=True, errors=[])),
              patch.object(cloud, "audit_pair", return_value=fatal), contextlib.redirect_stdout(io.StringIO())):
            study = cloud.CloudController(args)
            with self.assertRaisesRegex(cloud.Stop, "cost_protocol_failed"):
                study.cost()
            self.assertEqual(calls.call_count, 2)
            rows = read_jsonl(study.measurements)
            pair = json.loads(next((study.stage_path/"pairs").glob("*.json")).read_text())
            self.assertEqual(pair["classification"], "fatal")
            self.assertEqual(pair["record_hashes"], [r["record_sha256"] for r in rows])
            with self.assertRaisesRegex(cloud.Stop, "stage_already_attempted_no_automatic_retry"):
                cloud.CloudController(args)
            self.assertEqual(calls.call_count, 2)


class AnalysisTests(unittest.TestCase):
    @patch.object(report,'audit_execution_record',return_value=[])
    def test_report_rejects_old_cohort_and_missing_guard_quiet_period(self, _mock_protocol):
        d = design_fixture()
        with tempfile.TemporaryDirectory(dir=HERE, prefix="cloud_report_cpu_") as tmp:
            root = Path(tmp)
            guard = root/"guards"/"synthetic.jsonl"
            guard.parent.mkdir()
            uuid = d["scope"]["device_uuid"]
            events = [dict(event="guard_start", guard_id="g", idle_seconds=30),
                      dict(event="idle_complete", guard_id="g"),
                      dict(event="run_one_start", guard_id="g", active_run=dict(guard_run_id="r")),
                      *[dict(event="sample", guard_id="g", active_run=dict(guard_run_id="r"),
                             boundary=b, ok=True, gpu_uuid=uuid, gpu=dict(memory_used_mib=100)) for b in ("before", "after")],
                      dict(event="run_one_end", guard_id="g", guard_run_id="r", clean=True),
                      dict(event="guard_complete", guard_id="g", success=True, clean=True, script_exit_code=0)]
            guard.write_bytes(b"".join(canonical(e)+b"\n" for e in events))
            row = dict(study_id=d["study_id"], manifest_sha256="current-manifest", status="ok", guard_clean=True,
                       guard_gpu_uuid=uuid, guard_id="g", guard_log=str(guard), guard_run_id="r",
                       phase="execution", workload_id="n1024_b1", round=0,
                       input_seed=input_seed(d,"execution","n1024_b1",0))
            row["record_sha256"] = sha(canonical(row))
            base = root/"execution"
            write_json(base/"gate.json", dict(passed=True, record_count=1, guard_sessions=[dict(guard_id="g", guard_log=str(guard))]))
            (base/"measurements.jsonl").write_bytes(canonical(row)+b"\n")
            self.assertEqual(report.stage_rows(root, "execution", d, "current-manifest")[1], [])
            row['input_seed'] += 1
            row.pop('record_sha256')
            row['record_sha256'] = sha(canonical(row))
            (base/'measurements.jsonl').write_bytes(canonical(row)+b'\n')
            self.assertIn('row_frozen_seed_mismatch',report.stage_rows(root,'execution',d,'current-manifest')[1])
            row['input_seed'] -= 1
            row.pop('record_sha256')
            row['record_sha256'] = sha(canonical(row))
            (base/'measurements.jsonl').write_bytes(canonical(row)+b'\n')
            (base/'gate.json').unlink()
            retained, errors = report.stage_rows(root,'execution',d,'current-manifest')
            self.assertEqual(len(retained),1)
            self.assertIn('stage_gate_missing',errors)
            write_json(base/'gate.json',dict(passed=True,record_count=1,guard_sessions=[dict(guard_id='g',guard_log=str(guard))]))
            row["study_id"] = "old-cohort"
            row.pop("record_sha256")
            row["record_sha256"] = sha(canonical(row))
            (base/"measurements.jsonl").write_bytes(canonical(row)+b"\n")
            self.assertIn("row_identity_status_or_hash_failed", report.stage_rows(root, "execution", d, "current-manifest")[1])
            guard.write_bytes(b"".join(canonical(e)+b"\n" for e in events if e["event"] != "idle_complete"))
            self.assertIn("guard_start_or_quiet_invalid", report.stage_rows(root, "execution", d, "current-manifest")[1])

    def test_report_reaudits_frozen_execution_protocol(self):
        d = design_fixture()
        with tempfile.TemporaryDirectory(dir=HERE,prefix='cloud_report_protocol_') as tmp:
            root = Path(tmp)
            workload = dict(next(w for w in d['workloads'] if w['id']=='n1024_b1'),graph_repeats=8,samples=101)
            write_json(root/'graph_repeats.json',{'n1024_b1':{'graph_repeats':8}})
            seed = input_seed(d,'execution','n1024_b1',0)
            row = dict(workload_id='n1024_b1',workload=workload,candidate_id='default',config={},phase='execution',round=0,
                       input_seed=seed,result=synthetic_result(workload,'default',{},seed))
            row['result']['pattern'] = 'random'
            self.assertEqual(report.audit_execution_record(root,'execution',d,row,[]),[])
            row['result']['correctness']['passed'] = False
            self.assertIn('numerical_failure:correctness',report.audit_execution_record(root,'execution',d,row,[]))

    def test_student_tail_and_holm_family(self):
        # Construct logs with t=2.262157... against the predeclared 2% null.
        offsets = [-4.5, -3.5, -2.5, -1.5, -.5, .5, 1.5, 2.5, 3.5, 4.5]
        scale = .001
        stderr = math.sqrt(sum((x*scale)**2 for x in offsets)/9)/math.sqrt(10)
        mean = math.log(1.02) + 2.2621571627409915*stderr
        value = report.headline_stat([math.exp(mean+x*scale) for x in offsets])
        self.assertAlmostEqual(value["p"], .025, places=10)
        self.assertAlmostEqual(value["t_statistic"], 2.2621571627409915, places=10)
        tests = [dict(p=p, quality_valid=True, inferential=True, speedup=1.1) for p in (.01, .015, .04, .1, .5)]
        result = report.holm(tests)
        for row, expected in zip(result, (.05, .06, .12, .2, .5)):
            self.assertAlmostEqual(row["holm_p"], expected)
        self.assertTrue(all(not r["clear_gain"] for r in result))  # strict p < .05

    def test_zero_variance_and_incomplete_pairs_remain_noninferential(self):
        constant = report.headline_stat([1.1]*10)
        self.assertAlmostEqual(constant["speedup"], 1.1)
        self.assertFalse(constant["inferential"])
        self.assertEqual(constant["p"], 1.)
        self.assertIsNone(constant["interval95"])
        missing = report.headline_stat([1.2]*9)
        self.assertFalse(missing["inferential"])
        self.assertIsNone(missing["speedup"])
        ratios = [math.exp(.10+x) for x in (-.01, -.008, -.006, -.004, -.002, .002, .004, .006, .008, .01)]
        finite = report.headline_stat(ratios)
        self.assertTrue(finite["inferential"])
        self.assertAlmostEqual(finite["speedup"], math.exp(.10))
        self.assertLess(finite["interval95"][0], finite["speedup"])
        self.assertGreater(finite["interval95"][1], finite["speedup"])

    def test_pairing_requires_exact_round_and_seed(self):
        rows = []
        for k in range(10):
            for cid, value in (("default", 2.), ("candidate", 1.)):
                rows.append(dict(workload_id="synthetic", round=k, candidate_id=cid, input_seed=k,
                                 result=dict(exec_ms_median=value), record_sha256=f"{cid}-{k}"))
        result = report.paired(rows, "synthetic", "default", "candidate", 10)
        self.assertEqual(result["ratios"], [2.]*10)
        self.assertEqual(result["speedup"], 2.)
        rows[-1]["input_seed"] = 100
        self.assertIsNone(report.paired(rows, "synthetic", "default", "candidate", 10))
        rows[-1]["input_seed"] = 9
        rows.append(copy.deepcopy(rows[-1]))
        self.assertIsNone(report.paired(rows, "synthetic", "default", "candidate", 10))

    def test_same_cohort_amortization_and_default_winner(self):
        d = design_fixture()
        ids = [c["id"] for c in d["candidates"]]
        cost_rows, execution_rows = [], []
        with tempfile.TemporaryDirectory(dir=HERE, prefix="cloud_cost_cpu_") as tmp:
            root = Path(tmp)
            for wid in COST:
                winner = WINNERS[wid]
                rounds = next(x["rounds"] for x in d["execution"]["workloads"] if x["workload_id"] == wid)
                for k in range(rounds):
                    execution_rows.append(dict(workload_id=wid, round=k, candidate_id="default",
                                               result=dict(exec_ms_median=5., initialize_ms=4.)))
                    if winner != "default":
                        execution_rows.append(dict(workload_id=wid, round=k, candidate_id=winner,
                                                   result=dict(exec_ms_median=4., initialize_ms=6.)))
                for k in range(3):
                    order = sorted(ids)
                    random.Random(input_seed(d,"cost/candidate_order",wid,k)).shuffle(order)
                    groups = []
                    for mode in ("full-search-cost", "describe-select-cost"):
                        apps = []
                        for cid in order:
                            value = (10. if cid == "default" else 12.) if mode == "full-search-cost" else .5
                            apps.append(dict(candidate_id=cid, role="full" if mode == "full-search-cost" else "describe",
                                             sequence_index=len(apps), result=dict(initialize_ms=value, init_trace=dict(counters={}))))
                        if mode == "describe-select-cost":
                            apps.append(dict(candidate_id=winner, role="materialize_selected", sequence_index=26,
                                             result=dict(initialize_ms=2., init_trace=dict(counters={}))))
                        digest = f"{wid}-{k}-{mode}"
                        summary = dict(candidate_order_ids=order, apps=apps, context_pipeline_wall_ms=100.)
                        row = dict(workload_id=wid, round=k, candidate_id=mode, result=summary, record_sha256=digest)
                        cost_rows.append(row)
                        groups.append(row)
                    modes = ["full-search-cost","describe-select-cost"] if k%2==0 else ["describe-select-cost","full-search-cost"]
                    group_by_mode = {r['candidate_id']:r for r in groups}
                    write_json(root/"cost"/"pairs"/f"{wid}_r{k}.json",
                               dict(classification="equivalent",candidate_ids=modes,record_hashes=[group_by_mode[x]['record_sha256'] for x in modes]))
            with (patch.object(report, "audit_pair", return_value=dict(ok=True, errors=[])),
                  patch.object(report, "plan_fingerprints", return_value=dict(core="core", source_compiler="source")),
                  patch.object(report, "audit_trace", return_value=dict(ok=True, root_inclusive_ms=0., stage_totals={}))):
                out = report.analyze_cost(root, d, cost_rows, execution_rows)
                cost_rows[0]['result']['candidate_order_ids'] = list(reversed(cost_rows[0]['result']['candidate_order_ids']))
                with self.assertRaisesRegex(ValueError,'invalid_cost_candidate_coverage'):
                    report.analyze_cost(root,d,cost_rows,execution_rows)
            self.assertEqual(len(out["pairs"]), 24)
            self.assertEqual(len(out["pipelines"]), 48)
            by = {r["workload_id"]:r for r in out["amortization"]}
            accelerated = by["n12288_b1"] if "n12288_b1" in by else by["n16384_b1"]
            self.assertEqual(accelerated["components"][0]["execution_delta_ms"], 1.)
            self.assertEqual(accelerated["components"][0]["break_even_executions"], 2)
            self.assertEqual(accelerated["components"][1]["break_even_executions"], 11)
            for component in by["n1024_b1"]["components"]:
                self.assertEqual(component["execution_delta_ms"], 0.)
                self.assertIsNone(component["break_even_executions"])


if __name__ == "__main__":
    unittest.main()
