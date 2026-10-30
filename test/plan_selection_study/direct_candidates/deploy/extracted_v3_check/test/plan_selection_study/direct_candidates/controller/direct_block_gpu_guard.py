#!/usr/bin/env python3
"""Fail-closed GPU competition guard around the unchanged experiment runners.

Linux /proc and nvidia-smi only; no third-party Python dependencies. Only a
verified direct child harness may receive SIGTERM. External PIDs are never sent
signals. A contaminated run_one result is quarantined, never returned to runner.
"""
import argparse
import csv
import datetime as dt
import importlib
import io
import json
import math
import os
from pathlib import Path
import runpy
import select
import signal
import subprocess
import sys
import threading
import time
import uuid


MEMORY_LIMIT_MIB = 2048
OWN_EXIT_PROOF_SECONDS = 1.0
GPU_FIELDS = ("uuid", "pci.bus_id", "memory.used", "utilization.gpu",
              "utilization.memory", "clocks.current.sm", "temperature.gpu")
APP_FIELDS = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")


class GuardAbort(RuntimeError):
    pass


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def numeric(value, field):
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"Unavailable nvidia-smi {field}: {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Invalid nvidia-smi {field}: {value!r}")
    return number


def query_smi(gpu_uuid, fields, kind):
    process = subprocess.run(
        ["nvidia-smi", f"--id={gpu_uuid}", f"--query-{kind}=" + ",".join(fields),
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
        timeout=2, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"nvidia-smi {kind} failed ({process.returncode}): {process.stderr.strip()[:600]}")
    rows = []
    for row in csv.reader(io.StringIO(process.stdout)):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != len(fields):
            raise ValueError(f"Unexpected nvidia-smi {kind} row with {len(row)} columns")
        rows.append(dict(zip(fields, (value.strip() for value in row))))
    return rows


def process_identity(pid):
    """Read PID start time as well as PPID/exe to protect against PID reuse."""
    try:
        proc = Path("/proc") / str(pid)
        fields = (proc / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return dict(pid=pid, ppid=int(fields[1]), start_ticks=int(fields[19]),
                    exe=str((proc / "exe").resolve(strict=True)))
    except (OSError, ValueError, IndexError):
        return None


def process_basic_identity(pid):
    """Read stat independently of exe, which disappears while a child is a zombie."""
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text(encoding='utf-8').rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]))
    except (OSError, ValueError, IndexError):
        return None


def pidfd_has_exited(pidfd):
    """Kernel terminal notification on the handle retained at child registration."""
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    events = poller.poll(0)
    if any(mask & (select.POLLERR | select.POLLNVAL) for _, mask in events):
        raise OSError('registered pidfd polling failed')
    return any(mask & select.POLLIN for _, mask in events)


def owned_identity(identity, expected_exe, parent_pid):
    return bool(identity and expected_exe and identity["ppid"] == parent_pid
                and identity["exe"] == expected_exe)


class GPUQuery:
    def __init__(self, gpu_uuid, exited_owned=None):
        self.gpu_uuid = gpu_uuid
        self.exited_owned = exited_owned

    def __call__(self, expected_exe, parent_pid):
        devices = query_smi(self.gpu_uuid, GPU_FIELDS, "gpu")
        if len(devices) != 1 or devices[0]["uuid"] != self.gpu_uuid:
            raise RuntimeError("nvidia-smi did not identify exactly the requested GPU UUID")
        device = devices[0]
        gpu = dict(uuid=device["uuid"], pci_bus_id=device["pci.bus_id"],
                   memory_used_mib=numeric(device["memory.used"], "memory.used"),
                   utilization_gpu_pct=numeric(device["utilization.gpu"], "utilization.gpu"),
                   utilization_memory_pct=numeric(device["utilization.memory"], "utilization.memory"),
                   sm_clock_mhz=numeric(device["clocks.current.sm"], "clocks.current.sm"),
                   temperature_c=numeric(device["temperature.gpu"], "temperature.gpu"))
        # NVIDIA's GPU utilization is kernel-active time, not SM occupancy.
        gpu["sm_activity_pct"] = gpu["utilization_gpu_pct"]
        gpu["sm_activity_metric"] = "nvidia-smi utilization.gpu; kernel-active time, not occupancy"
        apps = []
        for app in query_smi(self.gpu_uuid, APP_FIELDS, "compute-apps"):
            if app["gpu_uuid"] != self.gpu_uuid:
                continue
            pid = int(app["pid"])
            if pid <= 0:
                raise ValueError("nvidia-smi returned a nonpositive compute PID")
            identity = process_identity(pid)
            exited = self.exited_owned(pid) if identity is None and self.exited_owned else None
            if exited:
                identity = exited
            apps.append(dict(gpu_uuid=app["gpu_uuid"], pid=pid,
                             process_name=app["process_name"],
                             used_memory_mib=numeric(app["used_gpu_memory"], "used_gpu_memory"),
                             ppid=identity["ppid"] if identity else None,
                             exe=identity["exe"] if identity else None,
                             start_ticks=identity["start_ticks"] if identity else None,
                             identity_state="registered_exited_own" if exited else "live_proc" if identity else "unknown",
                             own_harness=bool(exited) or owned_identity(identity, expected_exe, parent_pid)))
        return gpu, apps


class Guard:
    def __init__(self, args, query=None):
        self.args = args
        self.guard_id = uuid.uuid4().hex
        self.query = query or GPUQuery(args.gpu, self.exited_owned)
        self.parent_pid = os.getpid()
        self.lock = threading.RLock()
        self.sample_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.abort = threading.Event()
        self.shutdown = threading.Event()
        self.thread = None
        self.active = None
        self.expected_exe = None
        self.first_incident = None
        self.sample_count = 0
        self.children = {}

    def exited_owned(self, pid):
        with self.lock:
            registered = self.children.get(pid)
            run_id = (self.active or {}).get('guard_run_id')
            if not registered or not run_id or registered.get('guard_run_id') != run_id:
                return None
            previous = registered['identity']
            # Popen.poll can report None while another thread owns its waitpid
            # lock, or exe can vanish just before stat reaches a terminal state.
            # Only a current-run registered identity gets this bounded recheck.
            begin = time.monotonic()
            deadline = begin + OWN_EXIT_PROOF_SECONDS
            observations = []
            reason = 'unresolved_exit_identity'
            proven = False
            for attempt in range(101):
                basic = process_basic_identity(pid)
                exists = (Path('/proc') / str(pid)).exists()
                if basic is not None and (previous.get('start_ticks') is None or basic['ppid'] != previous['ppid']
                        or basic['ppid'] != self.parent_pid or basic['start_ticks'] != previous['start_ticks']):
                    observations.append(dict(basic=basic,proc_exists=exists,pidfd_exited=None,
                                             elapsed_ms=(time.monotonic()-begin)*1000))
                    reason = 'conflicting_live_or_reused_identity'
                    break  # reject a new identity before consulting the OLD pidfd
                terminal_fd = pidfd_has_exited(registered['pidfd']) if registered.get('pidfd') is not None else False
                observations.append(dict(basic=basic,proc_exists=exists,pidfd_exited=terminal_fd,
                                         elapsed_ms=(time.monotonic()-begin)*1000))
                if basic is not None:
                    if basic['state'] in ('Z', 'X') or terminal_fd:
                        proven = True
                        reason = 'exact_registered_terminal_stat_or_pidfd'
                        break
                else:
                    poll_result = registered['process'].poll()
                    observations[-1]['popen_poll_return'] = poll_result
                    if (poll_result is not None or terminal_fd) and not exists:
                        proven = True
                        reason = 'registered_child_terminal_and_proc_absent'
                        break
                remaining = deadline - time.monotonic()
                if attempt == 100 or remaining <= 0:
                    break
                time.sleep(min(.01, remaining))
            self.emit('own_exit_identity_recheck',identity=previous,active_run=self.active,
                      proof=proven,reason=reason,pidfd_retained=registered.get('pidfd') is not None,
                      max_wait_seconds=OWN_EXIT_PROOF_SECONDS,observations=observations)
            if proven:
                return previous
        return None

    def popen(self, original):
        def create(command, *args, **kwargs):
            values = command if isinstance(command, (list, tuple)) else []
            candidate = str(Path(values[0]).resolve()) if values else None
            with self.lock:
                own = bool(self.active and candidate == self.expected_exe)
                if own:
                    self.check()
                    if kwargs.get("shell") or (kwargs.get("executable") and str(Path(kwargs["executable"]).resolve()) != candidate):
                        raise GuardAbort("Harness launch must directly exec the registered executable")
                    retained_pidfd, registered = None, False
                    try:
                        process = original(command, *args, **kwargs)
                        retained_pidfd = os.pidfd_open(process.pid, 0)
                        identity = process_identity(process.pid)
                        if not owned_identity(identity, candidate, self.parent_pid):
                            raise GuardAbort('Failed to verify direct child harness after spawn')
                        # Revalidate after pinning the kernel process identity.
                        checked = process_identity(process.pid)
                        if checked != identity:
                            raise GuardAbort('Own harness identity changed during registration')
                        self.children[process.pid] = dict(identity=identity, process=process,pidfd=retained_pidfd,
                                                         guard_run_id=self.active.get('guard_run_id'))
                        registered = True
                        self.emit("own_harness_registered", identity=identity, active_run=self.active,pidfd_retained=True)
                        return process
                    except BaseException as error:
                        if retained_pidfd is not None and not registered:
                            try:
                                os.close(retained_pidfd)
                            except OSError:
                                pass
                        # Never let run_one downgrade a post-spawn registration
                        # failure to a candidate-invalid row and continue GPU work.
                        try:
                            self.trip([f'own_harness_spawn_or_registration_failed:{type(error).__name__}:{error}'])
                        finally:
                            raise GuardAbort('Own harness spawn/registration failed; stage stopped') from error
            return original(command, *args, **kwargs)
        return create

    def close_run_handles(self, run_id):
        with self.lock:
            for registered in self.children.values():
                if registered.get('guard_run_id') == run_id and registered.get('pidfd') is not None:
                    os.close(registered['pidfd'])
                    registered['pidfd'] = None

    def emit(self, event, incident=False, **fields):
        row = dict(event=event, utc=utc(), guard_id=self.guard_id,
                   gpu_uuid=self.args.gpu, **fields)
        with self.write_lock:
            append_json(self.args.log, row)
            if incident:
                append_json(self.args.incidents, row)
        return row

    def terminate_own_harnesses(self):
        with self.lock:
            expected = self.expected_exe
        if not expected:
            return
        try:
            pids = [int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit()]
        except OSError as error:
            self.emit("termination_check_error", error=str(error))
            return
        for pid in pids:
            identity = process_identity(pid)
            if not owned_identity(identity, expected, self.parent_pid):
                continue
            pidfd = None
            try:
                # A kernel PID handle eliminates the PID-reuse race between a
                # /proc identity check and delivery. Never signal a raw PID.
                pidfd = os.pidfd_open(pid, 0)
                checked = process_identity(pid)
                if checked != identity or not owned_identity(checked, expected, self.parent_pid):
                    continue
                signal.pidfd_send_signal(pidfd, signal.SIGTERM, None, 0)
                self.emit("own_harness_sigterm", identity=checked, method="pidfd")
            except ProcessLookupError:
                pass
            except OSError as error:
                self.emit("termination_error", identity=identity, error=str(error))
            finally:
                if pidfd is not None:
                    os.close(pidfd)

    def trip(self, reasons, sample=None):
        with self.lock:
            first = not self.abort.is_set()
            self.abort.set()
            if first:
                self.first_incident = dict(reasons=reasons, sample=sample, active_run=self.active)
        if first:
            try:
                self.emit("incident", incident=True, clean=False, **self.first_incident)
            finally:
                # A failed/full log must not prevent stopping our own child.
                self.terminate_own_harnesses()
        else:
            self.terminate_own_harnesses()

    def sample(self, boundary):
        with self.sample_lock:
            with self.lock:
                active = dict(self.active) if self.active else None
                expected = self.expected_exe if active else None
            begin = time.monotonic()
            gpu, apps, monitor_error = None, [], None
            reasons = []
            try:
                gpu, apps = self.query(expected, self.parent_pid)
                foreign = [app["pid"] for app in apps if not app["own_harness"]]
                if foreign:
                    reasons.append("foreign_compute_process")
                if gpu["memory_used_mib"] > MEMORY_LIMIT_MIB:
                    reasons.append("total_gpu_memory_above_2048MiB")
                if time.monotonic() - begin > 2:
                    reasons.append("monitor_sample_took_over_2_seconds")
            except Exception as error:
                foreign = []
                monitor_error = f"{type(error).__name__}: {error}"
                reasons.append("monitor_query_failed")
            row = self.emit("sample", boundary=boundary, gpu=gpu, compute_apps=apps,
                            foreign_compute_pids=foreign, violation_reasons=reasons,
                            monitor_error=monitor_error, ok=not reasons,
                            active_run=active, query_wall_seconds=time.monotonic() - begin)
            self.sample_count += 1
            if reasons:
                self.trip(reasons, row)
            elif self.abort.is_set():
                # Cover the narrow race in which a child is spawned immediately
                # after contamination was detected but before run_one returns.
                self.terminate_own_harnesses()
            return row

    def check(self):
        if self.abort.is_set():
            raise GuardAbort("GPU monitoring detected contamination or failed")
        if self.thread is not None and not self.thread.is_alive() and not self.shutdown.is_set():
            self.trip(["monitor_thread_stopped"])
            raise GuardAbort("GPU monitor thread stopped")

    def idle_check(self):
        begin = time.monotonic()
        while True:
            self.sample("idle")
            self.check()
            elapsed = time.monotonic() - begin
            if elapsed >= self.args.idle_seconds:
                break
            self.shutdown.wait(min(1, self.args.idle_seconds - elapsed))
        self.emit("idle_complete", clean=True, observed_idle_seconds=time.monotonic() - begin)

    def monitor_loop(self):
        next_sample = time.monotonic()
        try:
            while not self.shutdown.is_set():
                self.sample("during" if self.active else "between")
                next_sample += 1
                self.shutdown.wait(max(0, next_sample - time.monotonic()))
        except BaseException as error:
            # No silent thread failure may allow unmonitored benchmark work.
            self.trip([f"monitor_thread_exception:{type(error).__name__}:{error}"])

    def start(self):
        self.thread = threading.Thread(target=self.monitor_loop, name="gpu-guard-monitor", daemon=True)
        self.thread.start()

    def wrapped(self, original):
        def run_one(args, workload, candidate, phase, round_no, order):
            self.check()
            with self.lock:
                self.expected_exe = str(Path(args.exe).resolve(strict=True))
                self.active = dict(phase=phase, workload_id=f"n{workload['n']}_b{workload['batch']}",
                                   candidate_id=candidate[0], round=round_no, order=order,
                                   expected_exe=self.expected_exe,
                                   guard_run_id=f"{self.guard_id}:{uuid.uuid4().hex}")
                run_id = self.active["guard_run_id"]
            self.emit("run_one_start", active_run=self.active)
            before = self.sample("before")
            self.check()
            row = None
            try:
                row = original(args, workload, candidate, phase, round_no, order)
                after = self.sample("after")
                self.check()
                row.update(guard_id=self.guard_id, guard_gpu_uuid=self.args.gpu,
                           guard_run_id=run_id, guard_log=str(self.args.log.resolve()),
                           guard_before_sample_utc=before["utc"], guard_after_sample_utc=after["utc"],
                           guard_clean=True)
            except BaseException as error:
                if not self.abort.is_set():
                    self.trip([f"run_one_exception:{type(error).__name__}:{error}"])
                if self.abort.is_set():
                    self.emit("quarantined_run", incident=True, clean=False,
                              active_run=self.active, quarantined_row=row,
                              first_incident=self.first_incident)
                raise
            finally:
                with self.lock:
                    self.close_run_handles(run_id)
                    self.active = None
                    self.expected_exe = None
            self.emit("run_one_end", clean=True, status=row.get("status"),
                      guard_run_id=run_id, workload_id=row.get("workload_id"), candidate_id=row.get("candidate_id"))
            return row
        return run_one

    def close(self):
        self.shutdown.set()
        if self.thread:
            self.thread.join(timeout=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--incidents", required=True, type=Path)
    parser.add_argument("--idle-seconds", type=float, default=30)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.name != "posix" or not Path("/proc/self/stat").is_file():
        parser.error("gpu_guard requires the remote Linux /proc host")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        parser.error("gpu_guard requires Linux pidfd signaling to prevent PID-reuse races")
    if not args.gpu.startswith("GPU-") or "," in args.gpu:
        parser.error("--gpu must be one full GPU UUID")
    if not math.isfinite(args.idle_seconds) or not 0 <= args.idle_seconds <= 3600:
        parser.error("--idle-seconds must be finite and between 0 and 3600")
    script = args.script.resolve(strict=True)
    if script.name != "direct_block_controller.py":
        parser.error("--script must be direct_block_controller.py")
    if args.log.resolve() == args.incidents.resolve():
        parser.error("--log and --incidents must be different files")
    for path in (args.log, args.incidents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    original_argv = sys.argv
    original_popen = subprocess.Popen
    original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    guard = Guard(args)
    markers = dict(VKFFT_STUDY_GUARD_ID=guard.guard_id, VKFFT_STUDY_GUARD_UUID=args.gpu,
                   VKFFT_STUDY_GUARD_LOG=str(args.log.resolve()))
    original_markers = {key: os.environ.get(key) for key in markers}
    exit_code = 0
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ.update(markers)
        guard.emit("guard_start", script=str(script), idle_seconds=args.idle_seconds,
                   memory_limit_mib=MEMORY_LIMIT_MIB, monitor_interval_seconds=1,
                   parent_pid=os.getpid(), script_args=args.script_args,
                   own_signal_method="pidfd_with_rechecked_PPID_exe_starttime")
        guard.idle_check()
        subprocess.Popen = guard.popen(original_popen)
        sys.path.insert(0, str(script.parent))
        suite = importlib.import_module("run_suite")
        if Path(suite.__file__).resolve() != script.parent / "run_suite.py":
            raise GuardAbort("Imported run_suite does not match the target script directory")
        original_run_one = suite.run_one
        suite.run_one = guard.wrapped(original_run_one)
        target = runpy.run_path(str(script), run_name="__gpu_guard_target__")
        target_main = target.get("main")
        if not callable(target_main):
            raise GuardAbort("Target script has no callable main")
        # run_suite executed through runpy defines its own globals. Replace that
        # run_one too; imported control.prepare_protocol uses suite.run_one.
        target_main.__globals__["run_one"] = suite.run_one
        child_args = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
        sys.argv = [str(script), *child_args]
        guard.start()
        try:
            target_main()
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        guard.sample("final")
        guard.check()
        guard.close()
        guard.check()
        if guard.thread is not None and guard.thread.is_alive():
            raise GuardAbort("GPU monitor did not stop within its bounded shutdown timeout")
        guard.emit("guard_complete", success=True, clean=True, script_exit_code=0,
                   sample_count=guard.sample_count)
    except BaseException as error:
        if not guard.abort.is_set():
            guard.trip([f"guard_or_target_exception:{type(error).__name__}:{error}"])
        guard.close()
        guard.emit("guard_abort", incident=True, success=False, clean=False,
                   error=f"{type(error).__name__}: {error}", sample_count=guard.sample_count)
        print(f"GPU guard stopped its own experiment: {error}", file=sys.stderr)
        exit_code = 3
    finally:
        guard.close()
        sys.argv = original_argv
        subprocess.Popen = original_popen
        if original_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_visible
        for key, value in original_markers.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
