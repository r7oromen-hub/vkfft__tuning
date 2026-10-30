import copy
import unittest

from trace_audit import audit_trace,plan_fingerprints


def trace_fixture():
    spans=[dict(id=0,parent_id=-1,stage="initialize",start_ns=0,end_ns=10_000_000,inclusive_ms=10.,exclusive_ms=4.),
           dict(id=1,parent_id=0,stage="codegen",start_ns=1_000_000,end_ns=3_000_000,inclusive_ms=2.,exclusive_ms=2.),
           dict(id=2,parent_id=0,stage="nvrtc_compile",start_ns=4_000_000,end_ns=8_000_000,inclusive_ms=4.,exclusive_ms=4.),
           dict(id=3,parent_id=-1,stage="cleanup",start_ns=20_000_000,end_ns=21_000_000,inclusive_ms=1.,exclusive_ms=1.)]
    return dict(spans=spans,stage_totals={s["stage"]:dict(calls=1,inclusive_ms=s["inclusive_ms"],exclusive_ms=s["exclusive_ms"]) for s in spans},
                counters=dict(nvrtc_compile=1,module_load=0,driver_kernel_launch=0))


class TraceAuditTests(unittest.TestCase):
    def test_nested_exclusive_partition_includes_separate_cleanup(self):
        result=audit_trace(trace_fixture())
        self.assertTrue(result["ok"],result["errors"])
        self.assertEqual(result["exclusive_sum_ms"],11.)
        self.assertEqual(sum(s["inclusive_ms"] for s in trace_fixture()["spans"]),17.)

    def test_bad_child_overlap_and_reported_totals_are_rejected(self):
        trace=trace_fixture()
        trace["spans"][2]["start_ns"]=2_000_000
        result=audit_trace(trace)
        self.assertFalse(result["ok"])
        self.assertTrue(any("overlapping" in error for error in result["errors"]))

    def test_parent_cycles_are_rejected(self):
        trace=trace_fixture()
        trace["spans"][0]["parent_id"]=1
        trace["spans"][1]["start_ns"]=0
        trace["spans"][1]["end_ns"]=10_000_000
        result=audit_trace(trace)
        self.assertFalse(result["ok"])
        self.assertTrue(any("parent_cycle" in error for error in result["errors"]))

    def test_core_source_and_canonical_fingerprints_are_separate(self):
        result=dict(structural_features=dict(numAxisUploads=1,axisSplit=[256]),precompile_kernels=[
            dict(axis=0,upload=0,canonical_descriptor=dict(stageRadix=[4,4,4,4],localSize=[64,1,1],axisBlock=[64,1,1,256],registerBoost=1,LUT=0),
                 source_sha256="a"*64,compiler_input=dict(architecture="sm_89",options=["--gpu-architecture=sm_89"]))])
        original=plan_fingerprints(result)
        changed=copy.deepcopy(result)
        changed["precompile_kernels"][0]["canonical_descriptor"]["LUT"]=1
        changed["precompile_kernels"][0]["source_sha256"]="b"*64
        alternate=plan_fingerprints(changed)
        self.assertEqual(original["core"],alternate["core"])
        self.assertNotEqual(original["canonical"],alternate["canonical"])
        self.assertNotEqual(original["source_compiler"],alternate["source_compiler"])


if __name__=="__main__":
    unittest.main()
