"""CPU integration using retained initializer snapshots; no executable is invoked."""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cost_recovery as recovery
import run_suite
from study_common import input_seed,load_design,read_jsonl,sha,write_json


HERE=Path(__file__).parent
SOURCE=HERE/"results/resume_20261028_attempt2"


@unittest.skipUnless((SOURCE/"cost/measurements.jsonl").exists(),"retained CPU evidence fixture not present")
class CostRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE,prefix="A6_cpu_fixture_")
        self.root=Path(self.temp.name)
        for name in ("manifest.json","research_design.json","research_freeze.json","screen_selection.json","graph_repeats.json",
                     "cost/measurements.jsonl","cost/pair_audits.json","cost/status.json","guards/cost_01.jsonl","guards/cost_01_incidents.jsonl"):
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/name,target)
        shutil.copytree(SOURCE/"cost/candidate_orders",self.root/"cost/candidate_orders")
        self.prefix=(self.root/"cost/measurements.jsonl").read_bytes()
        self.original=read_jsonl(self.root/"cost/measurements.jsonl")[:21]
        self.design=load_design(HERE/"research_design.json")
        self.manifest=json.loads((self.root/"manifest.json").read_text())
        self.guard="synthetic_A6_single_recovery_guard"
        self.log=self.root/"guards/cost_02.jsonl"
        self.log.write_text(json.dumps(dict(event="guard_start",guard_id=self.guard,script_args=["--stage","cost"]))+"\n")
        self.env=dict(VKFFT_STUDY_GUARD_ID=self.guard,VKFFT_STUDY_GUARD_UUID=self.design["scope"]["device_uuid"],VKFFT_STUDY_GUARD_LOG=str(self.log))
        self.args=argparse.Namespace(design=HERE/"research_design.json",exe="synthetic_not_executed",output=self.root,stage="cost",timeout=180,
            import_calibration_from=None,candidate_fault_diagnostics=None,candidate_fault_decision=None,cost_recovery_admission=HERE/"research_addendum_A6.json")
        self.real_guard=recovery.audit_row
        self.source_sha=sha((SOURCE/"cost/measurements.jsonl").read_bytes())

    def tearDown(self):
        self.assertEqual(sha((SOURCE/"cost/measurements.jsonl").read_bytes()),self.source_sha)
        self.temp.cleanup()

    def audit(self,row,uuid,root=None):
        if row.get("guard_id")==self.guard: return dict(ok=True,errors=[])
        return self.real_guard(row,uuid,root)

    def mocked_run(self,args,w,candidate,phase,k,order):
        mode=candidate[0]
        source=next(r for r in self.original if r["workload_id"]==w["id"] and r["round"]==0 and r["candidate_id"]==mode)
        result=copy.deepcopy(source["result"])
        table=Path(w["candidate_file"]).read_text().splitlines()
        ids=[line.split("\t")[0] for line in table[1:]]
        apps={a["candidate_id"]:a for a in result["apps"] if a["role"] in ("full","describe")}
        selected=[a for a in result["apps"] if a["role"]=="materialize_selected"]
        result["apps"]=[apps[c] for c in ids]+selected
        for i,a in enumerate(result["apps"]): a["sequence_index"]=i
        result.update(candidate_order_ids=ids,context_id=f"synthetic:{w['id']}:{k}:{mode}")
        return dict(schema_version="plan-selection-process/v1",study_id=args.design_data["study_id"],phase=phase,workload_id=w["id"],workload=w,
            candidate_id=mode,config={},round=k,order=order,block_id=w["block_id"],block_attempt=w["block_attempt"],input_seed=input_seed(self.design,phase,w["id"],k),
            status="ok",validation_errors=[],returncode=0,result=result,guard_clean=True,guard_id=self.guard,guard_gpu_uuid=self.env["VKFFT_STUDY_GUARD_UUID"],
            guard_log=str(self.log),guard_run_id=f"synthetic:{w['id']}:{k}:{mode}")

    def test_append49_select48_preserve20_replace_entire_pair_and_stop_second_admission(self):
        def frozen(study): study.args.expected_executable_sha256=self.manifest["executable_sha256"]
        with (patch.dict(os.environ,self.env),patch.object(run_suite.Study,"freeze_manifest",frozen),
              patch.object(run_suite.Study,"require_main",return_value=None),patch.object(run_suite.Study,"require",return_value=None),
              patch.object(run_suite,"run_one",side_effect=self.mocked_run) as calls,patch.object(run_suite,"audit_row",side_effect=self.audit),
              patch.object(recovery,"audit_row",side_effect=self.audit),contextlib.redirect_stdout(io.StringIO())):
            study=run_suite.Study(self.args)
            study.cost()
            self.assertEqual(calls.call_count,28)
            self.assertEqual(len(study.rows),49)
            self.assertTrue(study.measurements.read_bytes().startswith(self.prefix))
            rows,selected,admission=recovery.selected_cost_rows(self.root,require_complete=True)
            self.assertEqual(len(rows),48)
            self.assertEqual(len(selected["pairs"]),24)
            self.assertNotIn(recovery.ORPHAN,{r["record_sha256"] for r in rows})
            replacement=[r for r in rows if (r["workload_id"],r["round"])==recovery.AFFECTED]
            self.assertEqual([r["candidate_id"] for r in replacement],recovery.modes(1))
            self.assertTrue(all(r["block_attempt"]==1 for r in replacement))
            self.assertEqual({r["record_sha256"] for r in rows if r["guard_id"]==recovery.OLD_GUARD},set(admission["preserved_record_hashes"]))
            self.assertTrue(all(r["input_seed"]==2698601450 for r in replacement))
            self.assertEqual(selected["physical_processes_estimate"],50)
            changed=self.root/"guards/cost_03.jsonl"
            changed.write_text(json.dumps(dict(event="guard_start",guard_id="forbidden_second_guard",script_args=["--stage","cost"]))+"\n")
            with self.assertRaisesRegex(ValueError,"only_one_new_cost_guard"):
                recovery.admit_recovery(HERE/"research_addendum_A6.json",self.args.design,self.root,self.manifest,"forbidden_second_guard")


if __name__=="__main__": unittest.main()
