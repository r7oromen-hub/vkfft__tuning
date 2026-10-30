import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import controlled_recheck as cr
from study_common import load_design,input_seed,sha,write_json
from test_runner_s0 import synthetic_result


HERE=Path(__file__).parent


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE,prefix="followup_synthetic_")
        self.folder=Path(self.temp.name)
        source=load_design(HERE/"research_design.json")
        self.design=dict(study_id="synthetic_independent_H2",scope=source["scope"],measurement=dict(source["measurement"],seed_namespace="synthetic_H2/new_seed",screen_samples=101),
            workloads=[w for w in source["workloads"] if w["id"] in ("n16384_b1","n16384_b2048")],
            candidates=[c for c in source["candidates"] if c["id"] in ("default","default_replica","a128_r4_l1_np1")],rounds=10,
            directional_tests=[dict(workload_id="n16384_b1",reference="a128_r4_l1_np1",target="default"),
                               dict(workload_id="n16384_b2048",reference="default",target="a128_r4_l1_np1")],
            health=dict(enabled=True,workloads=[w for w in source["workloads"] if w["id"] in source["guard"]["health_gate"]["workloads"]],paired_process_rounds=5),
            budgets=dict(health_processes_max=30,recheck_processes=60,total_processes_max=90,retries=0))
        self.design_path=self.folder/"design.json"
        self.freeze_path=self.folder/"freeze.json"
        self.exe=self.folder/"synthetic_executable"
        self.exe.write_bytes(b"synthetic never executed")
        self.primary=self.folder/"primary"
        primary_design_sha=sha((HERE/"research_design.json").read_bytes())
        write_json(self.primary/"manifest.json",dict(study_id=source["study_id"],design_sha256=primary_design_sha,gpu_uuid=source["scope"]["device_uuid"],executable_sha256=sha(self.exe.read_bytes())))
        calibrations=[]
        repeats={}
        for w in source["workloads"]:
            row=dict(study_id=source["study_id"],phase="calibration",workload_id=w["id"],candidate_id="default",status="ok",result=dict(graph_repeats=16))
            row["record_sha256"]=sha(cr.canonical(row));calibrations.append(row)
            repeats[w["id"]]=dict(graph_repeats=16,source_record_sha256=row["record_sha256"])
        write_json(self.primary/"graph_repeats.json",repeats)
        for row in calibrations: cr.append_json(self.primary/"screen/measurements.jsonl",row)
        write_json(self.primary/"screen_selection.json",dict(synthetic=True))
        write_json(self.primary/"headline_selection.json",dict(synthetic=True))
        for stage in ("screen","confirmation","cost","headline"):
            write_json(self.primary/stage/"status.json",dict(state="complete"))
            write_json(self.primary/stage/"gate.json",dict(passed=True,guard_sessions=[dict(guard_id="synthetic_primary",guard_log="synthetic.jsonl")]))
            if stage!="screen": (self.primary/stage/"measurements.jsonl").write_bytes(b"")
        write_json(self.primary/"cost/pair_audits.json",[])
        (self.primary/"research_design.json").write_bytes((HERE/"research_design.json").read_bytes())
        self.env=dict(VKFFT_STUDY_GUARD_ID="synthetic_followup",VKFFT_STUDY_GUARD_UUID=source["scope"]["device_uuid"],VKFFT_STUDY_GUARD_LOG=str(self.folder/"guard.jsonl"))
        Path(self.env["VKFFT_STUDY_GUARD_LOG"]).write_text(json.dumps(dict(event="guard_start",guard_id="synthetic_followup"))+"\n")
        self.design["scope"]["executable_sha256"]=sha(self.exe.read_bytes())
        self.design["selection_origin"]=dict(primary_design_sha256=primary_design_sha,
            screen_selection_sha256=sha((self.primary/"screen_selection.json").read_bytes()),screen_measurements_sha256=sha((self.primary/"screen/measurements.jsonl").read_bytes()))
        self.design["measurement"]["repeats_source_sha256"]=sha((self.primary/"graph_repeats.json").read_bytes())
        self.design["measurement"]["repeats_frozen"]={w["id"]:16 for w in self.design["workloads"]+self.design["health"]["workloads"]}
        write_json(self.design_path,self.design)
        write_json(self.freeze_path,dict(study_id=self.design["study_id"],design_sha256=sha(self.design_path.read_bytes())))
        write_json(self.design_path.with_name("h2_recheck_activation_C1.json"),dict(study_id=self.design["study_id"],design_sha256=sha(self.design_path.read_bytes())))
        # Original-study nested coverage/trace validators are separately checked below;
        # these fixtures focus on the independent 90-process controller state machine.
        self.original_cost_validator=cr.Recheck.verify_primary_cost
        for method in ("verify_primary_execution","verify_primary_cost"):
            mocked=patch.object(cr.Recheck,method,return_value=None)
            mocked.start();self.addCleanup(mocked.stop)
        sessions=patch.object(cr,"cost_session_allowed",return_value=True)
        sessions.start();self.addCleanup(sessions.stop)

    def tearDown(self): self.temp.cleanup()

    def args(self,stage):
        return argparse.Namespace(design=self.design_path,freeze=self.freeze_path,output=self.folder/"new_followup",primary_output=self.primary,
                                  exe=str(self.exe),stage=stage,timeout=180)

    def reserve(self,stage):
        args=self.args(stage)
        args.reservation_id="synthetic_reservation"
        write_json(args.output/stage/"attempt_reservation.json",dict(reservation_id=args.reservation_id,guard_log=str(Path(self.env["VKFFT_STUDY_GUARD_LOG"]).resolve())))
        write_json(args.output/stage/"status.json",dict(state="reserved"))
        return args

    def mocked_run(self,args,w,candidate,phase,k,order):
        cid,config=candidate
        seed=input_seed(args.design_data,phase,w["id"],k)
        result=synthetic_result(w,cid,config,seed)
        if phase=="h2_recheck":
            speed=1.25+.001*k
            if w["batch"]==1: ms=1/speed if cid in ("default","default_replica") else 1.
            else: ms=1. if cid in ("default","default_replica") else 1/speed
            result.update(exec_ms_median=ms,exec_ms_samples=[ms]*101,raw_interval_ms_median=ms*16)
        return dict(study_id=self.design["study_id"],phase=phase,workload_id=w["id"],workload=w,candidate_id=cid,config=config,round=k,
                    order=order,position_in_block=w["position_in_block"],status="ok",returncode=0,validation_errors=[],result=result,input_seed=seed,
                    guard_id="synthetic_followup",guard_log=self.env["VKFFT_STUDY_GUARD_LOG"],guard_clean=True,
                    guard_gpu_uuid=self.env["VKFFT_STUDY_GUARD_UUID"],guard_run_id=f"{phase}:{w['id']}:{cid}:{k}")

    def test_exact90_fresh_processes_separate_namespace_opposite_family_and_no_reruns(self):
        primary_bytes=(self.primary/"manifest.json").read_bytes()
        with (patch.dict(os.environ,self.env),patch.object(cr,"run_one",side_effect=self.mocked_run) as calls,
              patch.object(cr,"audit_row",return_value=dict(ok=True,errors=[])),patch.object(cr,"completed_session",return_value=True),
              patch.object(cr,"resolve_log",return_value=self.folder/"unused"),contextlib.redirect_stdout(io.StringIO())):
            health=cr.Recheck(self.reserve("health"));health.health()
            recheck_args=self.reserve("recheck")
            recheck=cr.Recheck(recheck_args);recheck.recheck()
            cr.Recheck(recheck_args).recheck()
            self.assertEqual(calls.call_count,90)
            self.assertEqual(len(health.rows),30)
            self.assertEqual(len(recheck.rows),60)
            self.assertEqual({r["round"] for r in recheck.rows},set(range(10)))
            self.assertTrue(all(r["workload"]["samples"]==101 and r["workload"]["graph_repeats"]==16 for r in recheck.rows))
            self.assertEqual(recheck.rows[0]["candidate_id"],"default")
            self.assertEqual(recheck.rows[6]["candidate_id"],"default_replica")
            cr.analyze(self.args("analyze"))
            summary=json.loads((recheck.output/"analysis/summary.json").read_text())
            self.assertTrue(summary["rank_reversal_established"])
            copied=copy.deepcopy(recheck.rows[0]);copied["study_id"]="plan_selection_20261027_r2"
            manifest=json.loads((recheck.output/"manifest.json").read_text())
            with self.assertRaisesRegex(cr.RecheckStop,"study_phase_or_controller_mismatch"):
                cr.verify_followup_row(self.design,manifest,copied,"h2_recheck")
            # Each pair remains within 5%, but the forged successful health gate
            # cannot conceal a 4% aggregate discrepancy across all five pairs.
            for row in health.rows:
                if row["workload_id"]=="n1024_b1" and row["candidate_id"]=="default":
                    row["result"]["exec_ms_median"]*=1.04
                    row["result"]["exec_ms_samples"]=[x*1.04 for x in row["result"]["exec_ms_samples"]]
                row.pop("record_sha256");row["record_sha256"]=sha(cr.canonical(row))
            health.path.write_bytes(b"")
            for row in health.rows: cr.append_json(health.path,row)
            with self.assertRaisesRegex(cr.RecheckStop,"recomputed_geometric_mean_failed"):
                cr.verify_health(self.design,health.output,manifest)
            cr.analyze(self.args("analyze"))
        summary=json.loads((recheck.output/"analysis/summary.json").read_text())
        self.assertEqual(summary["family_size"],2)
        self.assertFalse(summary["rank_reversal_established"])
        self.assertTrue(all(t["p"]==1 and t["interval95"] is None for t in summary["tests"]))
        self.assertEqual((self.primary/"manifest.json").read_bytes(),primary_bytes)

    def test_primary_must_finish_and_any_candidate_failure_stops_without_retry(self):
        write_json(self.primary/"headline/status.json",dict(state="running"))
        with patch.dict(os.environ,self.env),patch.object(cr,"completed_session",return_value=True),patch.object(cr,"resolve_log",return_value=self.folder/"unused"):
            with self.assertRaisesRegex(cr.RecheckStop,"primary_sequence_not_finished"):
                cr.Recheck(self.args("health"))
        write_json(self.primary/"headline/status.json",dict(state="complete"))
        def failed(*args):
            row=self.mocked_run(*args);row.update(status="failed",validation_errors=["synthetic_numeric_failure"]);return row
        with (patch.dict(os.environ,self.env),patch.object(cr,"run_one",side_effect=failed) as calls,
              patch.object(cr,"audit_row",return_value=dict(ok=True,errors=[])),patch.object(cr,"completed_session",return_value=True),
              patch.object(cr,"resolve_log",return_value=self.folder/"unused"),contextlib.redirect_stdout(io.StringIO())):
            reserved=self.reserve("health")
            study=cr.Recheck(reserved)
            with self.assertRaisesRegex(cr.RecheckStop,"any_execution_or_protocol_failure"):
                study.health()
            self.assertEqual(len(study.rows),1)
            with self.assertRaisesRegex(cr.RecheckStop,"not_resumable_no_retry_budget"):
                cr.Recheck(reserved).health()
            self.assertEqual(calls.call_count,1)

    def test_frozen_source_R_primary_guards_and_disjoint_tree_rejections(self):
        with (patch.dict(os.environ,self.env),patch.object(cr,"audit_row",return_value=dict(ok=True,errors=[])),
              patch.object(cr,"completed_session",return_value=True),patch.object(cr,"resolve_log",return_value=self.folder/"unused")):
            graph=self.primary/"graph_repeats.json"
            original=graph.read_bytes();graph.write_bytes(original+b" ")
            with self.assertRaisesRegex(cr.RecheckStop,"frozen_R_source_file_hash_changed"): cr.Recheck(self.args("health"))
            graph.write_bytes(original)
            selection=self.primary/"screen_selection.json"
            original=selection.read_bytes();selection.write_bytes(original+b" ")
            with self.assertRaisesRegex(cr.RecheckStop,"posthoc_S1_selection_source_changed"): cr.Recheck(self.args("health"))
            selection.write_bytes(original)
            write_json(self.primary/"screen/gate.json",dict(passed=True,guard_sessions=[]))
            with self.assertRaisesRegex(cr.RecheckStop,"guard_sessions_empty"): cr.Recheck(self.args("health"))
            nested=self.args("health");nested.output=self.primary/"forbidden_nested_followup"
            with self.assertRaisesRegex(cr.RecheckStop,"disjoint_output_directory"): cr.Recheck(nested)
            self.assertFalse(nested.output.exists())
            ancestor=self.args("health");ancestor.output=self.folder
            with self.assertRaisesRegex(cr.RecheckStop,"disjoint_output_directory"): cr.Recheck(ancestor)

    def test_zero_row_quiet_start_refusal_is_recorded_stopped_and_cannot_retry(self):
        args=self.args("launch-health")
        with (patch.object(cr,"audit_row",return_value=dict(ok=True,errors=[])),patch.object(cr,"completed_session",return_value=True),
              patch.object(cr,"resolve_log",return_value=self.folder/"unused"),patch("gpu_guard.main",return_value=3) as guard):
            self.assertEqual(cr.launch(args),3)
            status=json.loads((args.output/"health/status.json").read_text())
            self.assertEqual(status["state"],"stopped")
            self.assertEqual(status["recorded_processes"],0)
            with self.assertRaisesRegex(cr.RecheckStop,"already_reserved_or_used_no_retry"):
                cr.launch(args)
            self.assertEqual(guard.call_count,1)

    def test_C1_cost_exception_is_complete_exact48_and_only_equivalence_negative(self):
        source=load_design(HERE/"research_design.json")
        rows=[]
        for wid in source["cost_profile"]["workloads"]:
            for k in range(3):
                order=sorted(c["id"] for c in source["candidates"])
                random.Random(input_seed(source,"cost/candidate_order",wid,k)).shuffle(order)
                for mode in ("full-search-cost","describe-select-cost"):
                    rows.append(dict(study_id=source["study_id"],phase="cost",workload_id=wid,round=k,candidate_id=mode,status="ok",returncode=0,
                        validation_errors=[],guard_id="synthetic_primary",guard_log="synthetic.jsonl",result=dict(candidate_order_ids=order)))
        def save_rows():
            (self.primary/"cost/measurements.jsonl").write_bytes(b"")
            for row in rows:
                row.pop("record_sha256",None);row["record_sha256"]=sha(cr.canonical(row));cr.append_json(self.primary/"cost/measurements.jsonl",row)
        save_rows()
        write_json(self.primary/"screen_selection.json",dict(workloads={wid:dict(winner="default") for wid in source["cost_profile"]["workloads"]}))
        write_json(self.primary/"cost/status.json",dict(state="complete_with_limitations",exit_reason="H3_unresolved_or_limited"))
        def save_pairs(quality):
            write_json(self.primary/"cost/pair_audits.json",[dict(workload_id=wid,round=k,**quality) for wid in source["cost_profile"]["workloads"] for k in range(3)])
        quality=dict(ok=False,errors=["describe_full_inequivalent:default"],comparisons=[],interpretation="synthetic")
        save_pairs(quality)
        obj=object.__new__(cr.Recheck)
        main=json.loads((self.primary/"manifest.json").read_text())
        with (patch.object(cr,"audit_row",return_value=dict(ok=True,errors=[])),patch.object(cr,"completed_session",return_value=True),
              patch.object(cr,"resolve_log",return_value=self.folder/"unused"),patch.object(cr,"audit_pair",side_effect=lambda *args:quality)):
            self.original_cost_validator(obj,self.primary,main)
            quality.update(ok=True,errors=[]);save_pairs(quality)
            with self.assertRaisesRegex(cr.RecheckStop,"requires_actual_equivalence_negative"):
                self.original_cost_validator(obj,self.primary,main)
            quality.update(ok=False,errors=["describe_full_inequivalent:default"]);save_pairs(quality)
            quality["errors"]=["describe:app_failed:0:default"];save_pairs(quality)
            with self.assertRaisesRegex(cr.RecheckStop,"nonpermitted_audit_failure"):
                self.original_cost_validator(obj,self.primary,main)
            quality["errors"]=["describe_full_inequivalent:default"];save_pairs(quality)
            rows[0]["result"]["candidate_order_ids"]=list(reversed(rows[0]["result"]["candidate_order_ids"]))
            save_rows()
            with self.assertRaisesRegex(cr.RecheckStop,"candidate_order_changed"):
                self.original_cost_validator(obj,self.primary,main)
            rows[0]["status"]="failed";save_rows()
            with self.assertRaisesRegex(cr.RecheckStop,"pipeline_execution_or_protocol_failed"):
                self.original_cost_validator(obj,self.primary,main)


if __name__=="__main__": unittest.main()
