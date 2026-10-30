"""GPU-free targeted tests of fail-closed isolation and signal ownership."""
import argparse
from pathlib import Path
import signal
import sys
import unittest
from unittest.mock import Mock, patch

import gpu_guard


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.patch = patch.object(gpu_guard, "append_json", side_effect=lambda path, row: self.rows.append((str(path), row)))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.args = argparse.Namespace(gpu="GPU-test", log=Path("guard.jsonl"),
                                      incidents=Path("incidents.jsonl"), idle_seconds=0)
        self.gpu = dict(uuid="GPU-test", memory_used_mib=138, utilization_gpu_pct=0,
                        utilization_memory_pct=0, sm_clock_mhz=210, temperature_c=30)

    def guard(self, query=None):
        guard = gpu_guard.Guard(self.args, query=query or Mock(return_value=(self.gpu, [])))
        guard.terminate_own_harnesses = Mock()
        return guard

    def call(self, guard, original):
        args = argparse.Namespace(exe=sys.executable)
        return guard.wrapped(original)(args, dict(n=256, batch=1), ("default", {}), "screen", 0, 0)

    def test_clean_boundaries_return_original_row(self):
        guard = self.guard()
        original = Mock(return_value=dict(status="ok"))
        row = self.call(guard, original)
        self.assertEqual(row["status"], "ok")
        self.assertEqual([row["boundary"] for _, row in self.rows if row["event"] == "sample"], ["before", "after"])
        self.assertFalse(guard.abort.is_set())

    def test_foreign_pid_fails_before_start(self):
        guard = self.guard(Mock(return_value=(self.gpu, [dict(pid=999, own_harness=False)])))
        original = Mock()
        with self.assertRaises(gpu_guard.GuardAbort):
            self.call(guard, original)
        original.assert_not_called()
        self.assertTrue(guard.abort.is_set())
        self.assertTrue(any(row["event"] == "incident" for _, row in self.rows))

    def test_memory_limit_is_inclusive(self):
        guard = self.guard()
        self.gpu["memory_used_mib"] = 2048
        self.assertTrue(guard.sample("before")["ok"])
        self.gpu["memory_used_mib"] = 2049
        self.assertFalse(guard.sample("after")["ok"])
        self.assertTrue(guard.abort.is_set())

    def test_monitor_failure_fails_closed(self):
        guard = self.guard(Mock(side_effect=RuntimeError("nvidia-smi unavailable")))
        with self.assertRaises(gpu_guard.GuardAbort):
            guard.idle_check()
        self.assertTrue(guard.abort.is_set())
        self.assertEqual(next(row for _, row in self.rows if row["event"] == "sample")["violation_reasons"], ["monitor_query_failed"])

    def test_during_pollution_quarantines_even_ok_row(self):
        guard = self.guard()
        original_row = dict(status="ok", marker="must_not_reach_main_runner")
        def original(*args):
            guard.trip(["foreign_compute_process"])
            return original_row
        with self.assertRaises(gpu_guard.GuardAbort):
            self.call(guard, original)
        quarantine = [row for path, row in self.rows if path == "incidents.jsonl" and row["event"] == "quarantined_run"]
        self.assertEqual(len(quarantine), 1)
        self.assertEqual(quarantine[0]["quarantined_row"], original_row)
        self.assertFalse(any(row["event"] == "run_one_end" for _, row in self.rows))

    def test_after_boundary_pollution_quarantines(self):
        guard = self.guard(Mock(side_effect=[(self.gpu, []), (self.gpu, [dict(pid=999, own_harness=False)])]))
        with self.assertRaises(gpu_guard.GuardAbort):
            self.call(guard, Mock(return_value=dict(status="ok")))
        self.assertTrue(any(row["event"] == "quarantined_run" for _, row in self.rows))

    def test_dead_monitor_is_not_ignored(self):
        guard = self.guard()
        guard.thread = Mock()
        guard.thread.is_alive.return_value = False
        with self.assertRaises(gpu_guard.GuardAbort):
            guard.check()

    def test_signal_only_verified_direct_child_harness(self):
        guard = gpu_guard.Guard(self.args, query=Mock())
        guard.expected_exe = "/owned/harness"
        identities = {
            10: dict(pid=10, ppid=guard.parent_pid, exe="/owned/harness", start_ticks=5),
            11: dict(pid=11, ppid=555, exe="/owned/harness", start_ticks=6),
            12: dict(pid=12, ppid=guard.parent_pid, exe="/external/program", start_ticks=7)}
        with patch.object(Path, "iterdir", return_value=[Path("/proc/10"), Path("/proc/11"), Path("/proc/12")]), \
             patch.object(gpu_guard, "process_identity", side_effect=lambda pid: identities[pid]), \
             patch.object(gpu_guard.os, "pidfd_open", return_value=100, create=True) as opener, \
             patch.object(gpu_guard.signal, "pidfd_send_signal", create=True) as sender, \
             patch.object(gpu_guard.os, "close") as close, \
             patch.object(gpu_guard.os, "kill") as kill:
            guard.terminate_own_harnesses()
            opener.assert_called_once_with(10, 0)
            sender.assert_called_once_with(100, signal.SIGTERM, None, 0)
            close.assert_called_once_with(100)
            kill.assert_not_called()

    def test_pid_reuse_cannot_receive_signal(self):
        guard = gpu_guard.Guard(self.args, query=Mock())
        guard.expected_exe = "/owned/harness"
        first = dict(pid=10, ppid=guard.parent_pid, exe="/owned/harness", start_ticks=5)
        second = dict(first, start_ticks=6)
        with patch.object(Path, "iterdir", return_value=[Path("/proc/10")]), \
             patch.object(gpu_guard, "process_identity", side_effect=[first, second]), \
             patch.object(gpu_guard.os, "pidfd_open", return_value=100, create=True), \
             patch.object(gpu_guard.signal, "pidfd_send_signal", create=True) as sender, \
             patch.object(gpu_guard.os, "close"), \
             patch.object(gpu_guard.os, "kill") as kill:
            guard.terminate_own_harnesses()
            sender.assert_not_called()
            kill.assert_not_called()

    def test_log_failure_still_stops_own_child(self):
        guard = self.guard()
        guard.emit = Mock(side_effect=OSError("disk full"))
        with self.assertRaises(OSError):
            guard.trip(["monitor_failed"])
        self.assertTrue(guard.abort.is_set())
        guard.terminate_own_harnesses.assert_called_once()

    def test_gpuquery_records_identity_and_clock(self):
        device = {field: "0" for field in gpu_guard.GPU_FIELDS}
        device.update(uuid="GPU-test", **{"pci.bus_id": "0000:65:00.0", "memory.used": "138", "clocks.current.sm": "210"})
        app = dict(gpu_uuid="GPU-test", pid="42", process_name="harness", used_gpu_memory="100")
        identity = dict(pid=42, ppid=10, exe="/owned/harness", start_ticks=1)
        with patch.object(gpu_guard, "query_smi", side_effect=[[device], [app]]), \
             patch.object(gpu_guard, "process_identity", return_value=identity):
            gpu, apps = gpu_guard.GPUQuery("GPU-test")("/owned/harness", 10)
        self.assertEqual(gpu["sm_clock_mhz"], 210)
        self.assertTrue(apps[0]["own_harness"])

    def test_known_reaped_child_is_not_confused_with_unknown_foreign(self):
        device = {field: "0" for field in gpu_guard.GPU_FIELDS}
        device.update(uuid="GPU-test", **{"pci.bus_id": "0000:65:00.0"})
        app = dict(gpu_uuid="GPU-test", pid="42", process_name="harness", used_gpu_memory="100")
        previous = dict(pid=42, ppid=10, exe="/owned/harness", start_ticks=1)
        with patch.object(gpu_guard, "query_smi", side_effect=[[device], [app]]), \
             patch.object(gpu_guard, "process_identity", return_value=None):
            _, apps = gpu_guard.GPUQuery("GPU-test", lambda pid: previous)(None, 10)
        self.assertTrue(apps[0]["own_harness"])
        self.assertEqual(apps[0]["identity_state"], "registered_exited_own")

    def test_live_reused_pid_never_uses_exited_registry(self):
        device = {field: "0" for field in gpu_guard.GPU_FIELDS}
        device.update(uuid="GPU-test", **{"pci.bus_id": "0000:65:00.0"})
        app = dict(gpu_uuid="GPU-test", pid="42", process_name="external", used_gpu_memory="100")
        live = dict(pid=42, ppid=999, exe="/external", start_ticks=2)
        registry = Mock()
        with patch.object(gpu_guard, "query_smi", side_effect=[[device], [app]]), \
             patch.object(gpu_guard, "process_identity", return_value=live):
            _, apps = gpu_guard.GPUQuery("GPU-test", registry)("/owned/harness", 10)
        registry.assert_not_called()
        self.assertFalse(apps[0]["own_harness"])

    def test_registered_zombie_is_own_while_popen_wait_lock_busy(self):
        guard = self.guard()
        previous = dict(pid=42, ppid=guard.parent_pid, exe='/owned/harness', start_ticks=123)
        process = Mock()
        process.poll.return_value = None
        guard.active = dict(guard_run_id='current')
        guard.children[42] = dict(identity=previous, process=process,guard_run_id='current')
        terminal = dict(pid=42, ppid=guard.parent_pid, state='Z', start_ticks=123)
        with patch.object(gpu_guard, 'process_basic_identity', return_value=terminal):
            self.assertEqual(guard.exited_owned(42), previous)
        process.poll.assert_not_called()

    def test_missing_exe_live_or_reused_terminal_pid_is_not_own(self):
        guard = self.guard()
        previous = dict(pid=42, ppid=guard.parent_pid, exe='/owned/harness', start_ticks=123)
        process = Mock()
        process.poll.return_value = 0
        guard.active = dict(guard_run_id='current')
        guard.children[42] = dict(identity=previous, process=process,guard_run_id='current')
        for basic in (dict(pid=42, ppid=guard.parent_pid, state='R', start_ticks=123),
                      dict(pid=42, ppid=guard.parent_pid, state='Z', start_ticks=124),
                      dict(pid=42, ppid=999, state='Z', start_ticks=123)):
            with self.subTest(basic=basic), patch.object(gpu_guard, 'process_basic_identity', return_value=basic), patch.object(gpu_guard.time,'sleep'):
                self.assertIsNone(guard.exited_owned(42))
        process.poll.assert_not_called()

    def test_unknown_or_unreadable_existing_pid_never_gets_exit_allowance(self):
        guard = self.guard()
        self.assertIsNone(guard.exited_owned(999))
        process = Mock()
        process.poll.return_value = 0
        guard.active = dict(guard_run_id='current')
        guard.children[42] = dict(identity=dict(pid=42,ppid=guard.parent_pid,exe='/owned/harness',start_ticks=123),process=process,guard_run_id='current')
        with patch.object(gpu_guard,'process_basic_identity',return_value=None), patch.object(Path,'exists',return_value=True), patch.object(gpu_guard.time,'sleep'):
            self.assertIsNone(guard.exited_owned(42))

    def test_previous_run_registry_cannot_authorize_stale_pid(self):
        guard = self.guard()
        guard.active = dict(guard_run_id='new')
        previous = dict(pid=42,ppid=guard.parent_pid,exe='/owned/harness',start_ticks=123)
        guard.children[42] = dict(identity=previous,process=Mock(),guard_run_id='old')
        with patch.object(gpu_guard,'process_basic_identity') as basic:
            self.assertIsNone(guard.exited_owned(42))
        basic.assert_not_called()

    def test_current_missing_proc_busy_wait_lock_then_reaped_within_bound(self):
        guard = self.guard()
        guard.active = dict(guard_run_id='current')
        previous = dict(pid=42,ppid=guard.parent_pid,exe='/owned/harness',start_ticks=123)
        process = Mock()
        process.poll.side_effect = [None,0]
        guard.children[42] = dict(identity=previous,process=process,guard_run_id='current')
        with patch.object(gpu_guard,'process_basic_identity',return_value=None), \
             patch.object(Path,'exists',return_value=False), patch.object(gpu_guard.time,'sleep') as sleep:
            self.assertEqual(guard.exited_owned(42),previous)
        sleep.assert_called_once()
        self.assertLessEqual(sleep.call_args.args[0],.01)

    def test_retained_pidfd_proves_exact_child_exit_after_old_100ms_window(self):
        guard = self.guard()
        guard.active = dict(guard_run_id='current')
        previous = dict(pid=42,ppid=guard.parent_pid,exe='/owned/harness',start_ticks=123)
        process = Mock()
        process.poll.return_value = None  # main thread owns waitpid lock throughout
        guard.children[42] = dict(identity=previous,process=process,guard_run_id='current',pidfd=201)
        tick = [0.]
        def advance(seconds): tick[0] += seconds
        with patch.object(gpu_guard,'process_basic_identity',return_value=None), \
             patch.object(Path,'exists',return_value=False), \
             patch.object(gpu_guard,'pidfd_has_exited',side_effect=lambda fd:tick[0]>=.3), \
             patch.object(gpu_guard.time,'monotonic',side_effect=lambda:tick[0]), \
             patch.object(gpu_guard.time,'sleep',side_effect=advance):
            self.assertEqual(guard.exited_owned(42),previous)
        self.assertGreaterEqual(tick[0],.3)
        self.assertLessEqual(tick[0],gpu_guard.OWN_EXIT_PROOF_SECONDS)
        proof = next(r for _,r in self.rows if r['event']=='own_exit_identity_recheck')
        self.assertTrue(proof['proof'])
        self.assertTrue(proof['pidfd_retained'])

    def test_ready_old_pidfd_cannot_hide_reused_pid_and_no_wait(self):
        guard = self.guard()
        guard.active = dict(guard_run_id='current')
        old = dict(pid=42,ppid=guard.parent_pid,exe='/owned/harness',start_ticks=123)
        guard.children[42] = dict(identity=old,process=Mock(),guard_run_id='current',pidfd=201)
        new = dict(pid=42,ppid=guard.parent_pid,state='Z',start_ticks=124)
        with patch.object(gpu_guard,'process_basic_identity',return_value=new), \
             patch.object(gpu_guard,'pidfd_has_exited',return_value=True) as fdproof, \
             patch.object(gpu_guard.time,'sleep') as sleep:
            self.assertIsNone(guard.exited_owned(42))
        fdproof.assert_not_called()
        sleep.assert_not_called()

    def test_retained_pidfd_without_terminal_proof_still_fails_closed(self):
        guard = self.guard()
        guard.active = dict(guard_run_id='current')
        old = dict(pid=42,ppid=guard.parent_pid,exe='/owned/harness',start_ticks=123)
        process = Mock();process.poll.return_value = None
        guard.children[42] = dict(identity=old,process=process,guard_run_id='current',pidfd=201)
        tick = [0.]
        def advance(seconds): tick[0] += seconds
        with patch.object(gpu_guard,'process_basic_identity',return_value=None), \
             patch.object(Path,'exists',return_value=False), patch.object(gpu_guard,'pidfd_has_exited',return_value=False), \
             patch.object(gpu_guard.time,'monotonic',side_effect=lambda:tick[0]), patch.object(gpu_guard.time,'sleep',side_effect=advance):
            self.assertIsNone(guard.exited_owned(42))
        self.assertLessEqual(tick[0],gpu_guard.OWN_EXIT_PROOF_SECONDS)
        self.assertFalse(next(r for _,r in self.rows if r['event']=='own_exit_identity_recheck')['proof'])

    def test_registration_pins_and_rechecks_identity_then_closes_only_current_handle(self):
        guard = self.guard()
        candidate = str(Path(sys.executable).resolve())
        guard.expected_exe = candidate;guard.active = dict(guard_run_id='current')
        process = Mock(pid=42)
        old = dict(pid=42,ppid=guard.parent_pid,exe=candidate,start_ticks=123)
        with patch.object(gpu_guard.os,'pidfd_open',return_value=201,create=True) as opener, \
             patch.object(gpu_guard,'process_identity',return_value=old) as identity, \
             patch.object(gpu_guard.os,'close') as close:
            self.assertIs(guard.popen(Mock(return_value=process))([candidate]),process)
            self.assertEqual(identity.call_count,2)
            opener.assert_called_once_with(42,0)
            guard.close_run_handles('other')
            close.assert_not_called()
            guard.close_run_handles('current');guard.close_run_handles('current')
            close.assert_called_once_with(201)

    def test_post_spawn_pidfd_failure_trips_and_cannot_be_candidate_failure(self):
        guard = self.guard()
        candidate = str(Path(sys.executable).resolve())
        guard.expected_exe = candidate;guard.active = dict(guard_run_id='current')
        with patch.object(gpu_guard.os,'pidfd_open',side_effect=OSError('no descriptors'),create=True):
            with self.assertRaises(gpu_guard.GuardAbort):
                guard.popen(Mock(return_value=Mock(pid=42)))([candidate])
        self.assertTrue(guard.abort.is_set())
        guard.terminate_own_harnesses.assert_called_once()

    def test_registration_identity_change_closes_unregistered_pidfd_and_trips(self):
        guard = self.guard()
        candidate = str(Path(sys.executable).resolve())
        guard.expected_exe = candidate;guard.active = dict(guard_run_id='current')
        first = dict(pid=42,ppid=guard.parent_pid,exe=candidate,start_ticks=123)
        with patch.object(gpu_guard.os,'pidfd_open',return_value=201,create=True), \
             patch.object(gpu_guard,'process_identity',side_effect=[first,dict(first,start_ticks=124)]), \
             patch.object(gpu_guard.os,'close') as close:
            with self.assertRaises(gpu_guard.GuardAbort):
                guard.popen(Mock(return_value=Mock(pid=42)))([candidate])
            close.assert_called_once_with(201)
        self.assertTrue(guard.abort.is_set())
        self.assertNotIn(42,guard.children)


if __name__ == "__main__":
    unittest.main()
