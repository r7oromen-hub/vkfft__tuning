import unittest
from unittest.mock import patch
from candidate_policy import semantic_session_admission


class CandidatePolicyTests(unittest.TestCase):
    def test_only_exact_reviewed_controller_abort_is_admissible(self):
        fault=dict(record_sha256="fault",guard_id="old",guard_log="old.jsonl")
        error="StudyStop: fatal_execution_protocol_or_baseline_failure:n16384_b1:a128_r4_l1_np1_s2_8192"
        reason="guard_or_target_exception:StudyStop:fatal_execution_protocol_or_baseline_failure:n16384_b1:a128_r4_l1_np1_s2_8192"
        events=[dict(guard_id="old",event="guard_abort",error=error),
                dict(guard_id="old",event="incident",reasons=[reason],active_run=None),
                dict(guard_id="old",event="sample",ok=True,foreign_compute_pids=[],violation_reasons=[],monitor_error=None)]
        with (patch("candidate_policy.resolve_log",return_value="old.jsonl"),patch("candidate_policy.completed_session",return_value=False),
              patch("candidate_policy.read_guard",return_value=events)):
            semantic_session_admission([fault],"unused","fault")
            events[-1]["foreign_compute_pids"]=[123]
            with self.assertRaisesRegex(ValueError,"unclean_sample"):
                semantic_session_admission([fault],"unused","fault")
            events[-1]["foreign_compute_pids"]=[]
            events[1]["reasons"]=["foreign_gpu_process"]
            with self.assertRaisesRegex(ValueError,"real_or_unreviewed_guard_incident"):
                semantic_session_admission([fault],"unused","fault")


if __name__=="__main__": unittest.main()
