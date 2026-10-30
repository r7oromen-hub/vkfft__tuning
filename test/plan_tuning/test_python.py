"""Synthetic unit/integration checks; generated timings are never saved as results."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import analyze
import controlled_recheck
import run_suite


class OrchestrationTests(unittest.TestCase):
    def test_workload_scope_and_cap(self):
        ws = run_suite.workloads("full", 256)
        self.assertEqual(len(ws), 48)
        self.assertEqual(len({w["n"] for w in ws}), 24)
        self.assertEqual({w["family"] for w in ws}, {"pow2", "mixed23", "mixed235"})
        self.assertTrue(all(w["input_bytes"] <= 256 * 1024**2 for w in ws))
        self.assertEqual(len(run_suite.CANDIDATES), 19)

    def test_trailing_crash_recovery(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_test_", dir=Path(__file__).parent) as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_bytes(b'{"a":1}\n{"a":')
            run_suite.repair_trailing_write(path)
            run_suite.append_json(path, {"a":2})
            self.assertEqual(run_suite.read_records(path), [{"a":1}, {"a":2}])

    def test_correctness_and_timing_gate(self):
        obj = dict(status="ok", exec_ms_samples=[1.,2.], exec_ms_median=1.5, correctness={"passed":True})
        self.assertTrue(run_suite.result_ok(obj))
        obj["correctness"]["passed"] = False
        self.assertFalse(run_suite.result_ok(obj))
        obj["correctness"]["passed"] = True
        obj["exec_ms_samples"] = [float("nan")]
        self.assertFalse(run_suite.result_ok(obj))

    def test_negative_control_matches_both_descriptors_and_resolved(self):
        base = dict(candidate_id="default", result=dict(structural_features={"axisSplit":[256]},
                                                       resolved_config={"aimThreads":128}))
        same = dict(candidate_id="a128_r1_l-1", result=dict(base["result"]))
        different = dict(candidate_id="a64_r1_l-1", result=dict(structural_features={"axisSplit":[256]},
                                                              resolved_config={"aimThreads":64}))
        self.assertEqual(run_suite.negative_control([base,same,different]), "a128_r1_l-1")
        self.assertIsNone(run_suite.negative_control([base,different]))

    def test_suite_resume_and_independent_input(self):
        calls = []

        def mock_run(args, workload, candidate, phase, round_no, order):
            calls.append((phase, candidate[0], round_no))
            return dict(phase=phase, workload_id=run_suite.workload_id(workload),
                        workload=workload, candidate_id=candidate[0], config=candidate[1],
                        round=round_no, status="ok", result=dict(exec_ms_median=(2. if candidate[0] == "default" else 1.)))

        with tempfile.TemporaryDirectory(prefix="synthetic_test_", dir=Path(__file__).parent) as temp:
            argv = ["run_suite.py", "--exe", "absent-test-only", "--output", temp,
                    "--limit-workloads", "1", "--confirm-rounds", "2", "--graph-repeats", "1"]
            with patch("sys.argv", argv), patch.object(run_suite, "run_one", mock_run), contextlib.redirect_stdout(io.StringIO()):
                run_suite.main()
                original = list(calls)
                run_suite.main()
            self.assertEqual(calls, original)
            self.assertEqual(sum(phase == "screen" for phase, _, _ in calls), 19)
            self.assertEqual(sum(phase == "confirm" for phase, _, _ in calls), 4)

    def test_run_one_seed_and_workload_validation(self):
        args = SimpleNamespace(exe="fake", seed=42, warmup=1, repeats=2,
                               confirm_warmup=1, confirm_repeats=2, timeout=5, timing_mode="graph", warmup_ms=100,
                               placement="outofplace")
        w = {"n":256, "batch":1}
        obj = dict(status="ok", n=256, batch=1, exec_ms_samples=[1.,2.], exec_ms_median=1.5,
                   correctness={"passed":True}, timing_mode="graph",placement="outofplace",graph_repeats=1)
        process = SimpleNamespace(returncode=0, stdout=json.dumps(obj), stderr="")
        with patch.object(run_suite.subprocess, "run", return_value=process):
            screen1 = run_suite.run_one(args, w, run_suite.CANDIDATES[0], "screen", 0, 0)
            screen2 = run_suite.run_one(args, w, run_suite.CANDIDATES[1], "screen", 0, 1)
            confirmation = run_suite.run_one(args, w, run_suite.CANDIDATES[0], "confirm", 0, 2)
        self.assertEqual(screen1["input_seed"], screen2["input_seed"])
        self.assertNotEqual(screen1["input_seed"], confirmation["input_seed"])
        self.assertEqual(screen1["status"], "ok")
        obj["batch"] = 2
        with patch.object(run_suite.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(obj), stderr="")):
            self.assertEqual(run_suite.run_one(args, w, run_suite.CANDIDATES[0], "screen", 0, 0)["status"], "failed")

    def test_control_pairs_and_resume(self):
        calls = []

        def mock_run(args, workload, candidate, phase, round_no, order):
            calls.append((candidate[0],round_no))
            return dict(phase=phase,workload_id=run_suite.workload_id(workload),workload=workload,
                        candidate_id=candidate[0],config=candidate[1],round=round_no,order=order,status="ok",
                        result=dict(exec_ms_median=1.,structural_features={"axisSplit":[256]},resolved_config={"aimThreads":128}))

        with tempfile.TemporaryDirectory(prefix="synthetic_test_", dir=Path(__file__).parent) as temp:
            folder = Path(temp)
            (folder / "workloads.json").write_text('[{"n":256,"batch":1}]',encoding="utf-8")
            argv = ["controlled_recheck.py","--exe","absent-test-only","--output",str(folder / "results"),
                    "--workloads",str(folder / "workloads.json"),"--rounds","3","--graph-repeats","1"]
            with patch("sys.argv",argv),patch.object(controlled_recheck,"run_one",mock_run),contextlib.redirect_stdout(io.StringIO()):
                controlled_recheck.main()
                controlled_recheck.main()
            summary = json.loads((folder / "results" / "control_summary.json").read_text())
            self.assertEqual(len(calls),6)
            self.assertEqual(summary["paired_measurements"],3)
            self.assertTrue(summary["workloads"][0]["all_descriptors_equal"])
            self.assertTrue(summary["workloads"][0]["all_resolved_equal"])
            self.assertEqual(summary["workloads"][0]["ratio_gmean"],1.)

    def test_graph_calibration_capped_and_frozen_on_resume(self):
        args = SimpleNamespace(graph_repeats=0,timing_mode="graph",placement="outofplace",dry_run=False,
                               retry_failures=False,graph_target_ms=.5,graph_max_repeats=2048)
        ws = [dict(n=256,batch=1),dict(n=16384,batch=1)]
        calls = []

        def mock_calibration(args, w, candidate, phase, round_no, order):
            calls.append((phase,w["n"],w["graph_repeats"]))
            if phase == "calibrate":
                self.assertEqual(w["graph_repeats"],1)
            return dict(workload_id=run_suite.workload_id(w),workload=w,round=round_no,status="ok",
                        result=dict(exec_ms_median=.0001 if w["n"]==256 else .2,
                                    raw_interval_ms_median=.2 if w["n"]==256 else w["graph_repeats"]*.1))

        with tempfile.TemporaryDirectory(prefix="synthetic_test_", dir=Path(__file__).parent) as temp:
            with patch.object(run_suite,"run_one",mock_calibration),contextlib.redirect_stdout(io.StringIO()):
                first = run_suite.prepare_protocol(args,ws,Path(temp))
                second = run_suite.prepare_protocol(args,ws,Path(temp))
            self.assertEqual(first,second)
            self.assertEqual([w["graph_repeats"] for w in first],[2048,5])
            self.assertEqual(len(calls),5)
            diagnostics = json.loads((Path(temp) / "calibration_summary.json").read_text())
            self.assertFalse(diagnostics["n256_b1"]["target_met"])
            self.assertTrue(diagnostics["n256_b1"]["capped"])
            self.assertTrue(diagnostics["n16384_b1"]["target_met"])


class AnalysisTests(unittest.TestCase):
    def test_no_compiler_feature_leakage(self):
        structure = {"axisSplit":[256], "kernels":[{"stageRadix":[4,4,4,4], "cuda_num_regs":32,
                     "cuda_static_shared_bytes":1024, "localSize":[64,1,1]}], "cuda_max_threads_per_block":1024}
        flat = analyze.precompile_features(structure)
        self.assertTrue(flat)
        self.assertFalse(any("cuda_" in k for k in flat))
        self.assertIn("kernels.0.stageRadix.0", flat)

    def test_geomean_and_bootstrap(self):
        self.assertAlmostEqual(analyze.geomean([.5,2]), 1.)
        self.assertEqual(analyze.bootstrap_geomean([2.,2.], np.random.default_rng(1), 20), [2.,2.])

    def test_v2_validation_separates_single_final_costs_and_intervals(self):
        check=dict(passed=True,criterion="criterion_v2",checked_elements=256,nonfinite_count=0,
                   failing_batches=0,pointwise_passed=True,failed_elements=0,
                   worst_batch_l2_scaled_error=2e-7,worst_batch_linf_input_scaled_error=1e-6,
                   l2_scaled_error_threshold=1e-5,linf_input_scaled_error_threshold=1e-4)
        final=dict(check,pointwise_passed=False,failed_elements=3)
        row=dict(phase="screen",workload_id="n256_b1",workload={"n":256,"batch":1},candidate_id="default",round=0,
                 order=0,status="ok",wall_seconds=2.4,started_utc="2026-01-01T00:00:00+00:00",
                 ended_utc="2026-01-01T00:00:02.400000+00:00",result=dict(
                     single_forward_correctness=check,correctness=final,single_forward_input_unchanged=True,
                     input_unchanged=True,exec_ms_samples=[1.,1.,1.,1.1,1.1,1.1],exec_ms_median=1.05,
                     raw_interval_ms_samples=[.4,.6],raw_interval_ms_median=.5,initialize_ms=5.,graph_build_ms=2.,graph_repeats=1))
        with tempfile.TemporaryDirectory(prefix="synthetic_test_",dir=Path(__file__).parent) as temp:
            result=analyze.validation_analysis([row],[],{"graph_target_ms":.5},Path(temp))
            numerical=result["numerical"]["measurements"]
            self.assertEqual(numerical["single_forward"]["passed"],1)
            self.assertEqual(numerical["single_forward"]["pointwise_diagnostic_failed_checks"],0)
            self.assertEqual(numerical["final_graph_or_forward"]["passed"],1)
            self.assertEqual(numerical["final_graph_or_forward"]["pointwise_diagnostic_failed_elements"],3)
            self.assertEqual(result["timing_intervals"]["process_medians_below_target"],0)
            self.assertEqual(result["timing_intervals"]["sample_fraction_below_target"],.5)
            self.assertEqual(result["measured_costs"]["screen"]["subprocess_wall_seconds"],2.4)
            self.assertEqual(result["measured_costs"]["screen"]["initialize_including_compile_ms"]["median"],5.)
            self.assertEqual(result["measured_costs"]["screen"]["graph_build_ms"]["median"],2.)
            self.assertEqual(result["within_process_drift"]["processes_over_5pct_change"],1)

    def test_analysis_end_to_end_temporary_synthetic(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_test_", dir=Path(__file__).parent) as temp:
            folder = Path(temp)
            rows = []
            for n in (256,768,1280):
                w = dict(n=n,batch=1,family="synthetic",input_bytes=n*8)
                wid = run_suite.workload_id(w)
                for i, (cid, config) in enumerate(run_suite.CANDIDATES):
                    rows.append(dict(phase="screen", workload_id=wid, workload=w, candidate_id=cid,
                                     config=config, round=0, status="ok", result=dict(exec_ms_median=2. if i == 0 else 1.+i/100,
                                     initialize_ms=5., gpu={"name":"SYNTHETIC TEST ONLY"},
                                     structural_features={"axisSplit":[n], "cuda_num_regs":i})))
                winner = run_suite.CANDIDATES[1]
                for round_no in range(3):
                    for cid,config in [run_suite.CANDIDATES[0],winner]:
                        rows.append(dict(phase="confirm", workload_id=wid, workload=w, candidate_id=cid,
                                         config=config, round=round_no, status="ok",
                                         result=dict(exec_ms_median=2. if cid == "default" else 1.)))
            for row in rows:
                run_suite.append_json(folder / "measurements.jsonl", row)
            argv = ["analyze.py", str(folder)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                analyze.main()
            summary = json.loads((folder / "analysis" / "summary.json").read_text())
            self.assertEqual(summary["complete_screen_workloads"], 3)
            self.assertAlmostEqual(summary["confirmation_speedup_gmean"], 2.)
            self.assertEqual(summary["confirmation_regressions"], 0)
            self.assertTrue((folder / "analysis" / "confirmed_speedup.png").exists())
            self.assertTrue((folder / "analysis" / "model_ablation.png").exists())
            self.assertTrue((folder / "analysis" / "results_zh.md").exists())
            self.assertEqual(summary["models"]["exact_n"]["target"],"log(t_candidate/t_default)")
            self.assertIn("combined",summary["models"]["exact_n"]["results"])
            self.assertIn("sensitivity",summary["models_absolute_sensitivity"]["exact_n"]["target"])
            quality_argv=["analyze.py",str(folder),"--output",str(folder / "quality"),"--skip-models","--quality-only"]
            with patch("sys.argv",quality_argv),contextlib.redirect_stdout(io.StringIO()):
                analyze.main()
            quality=json.loads((folder / "quality" / "quality_summary.json").read_text())
            self.assertEqual(quality["models"],"skipped_quality_only")
            self.assertNotIn("screen_oracle_speedup_gmean",quality)
            self.assertFalse((folder / "quality" / "model_ablation.png").exists())


if __name__ == "__main__":
    unittest.main()
