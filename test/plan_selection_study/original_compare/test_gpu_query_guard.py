"""CPU-only checks of single-call query integration and the existing guard boundary."""
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import original_compare_guard as guard


UUID = "GPU-test"
XML = """<?xml version="1.0"?><nvidia_smi_log><attached_gpus>2</attached_gpus>
<gpu id="00000000:00:03.0"><uuid>GPU-test</uuid>
<pci><pci_bus_id>00000000:00:03.0</pci_bus_id></pci>
<fb_memory_usage><used>100 MiB</used></fb_memory_usage>
<utilization><gpu_util>1 %</gpu_util><memory_util>2 %</memory_util></utilization>
<clocks><sm_clock>210 MHz</sm_clock></clocks>
<temperature><gpu_temp>33 C</gpu_temp></temperature>
<processes>{processes}</processes></gpu></nvidia_smi_log>"""


def process(pid, kind):
    return (f'<process_info><pid>{pid}</pid><type>{kind}</type>'
            f'<process_name>/binary/{pid}</process_name><used_memory>2 MiB</used_memory>'
            '</process_info>')


class Args:
    gpu = UUID
    log = Path('unused')
    incidents = Path('unused')


class QueryTests(unittest.TestCase):
    def test_single_invocation_and_foreign_process(self):
        output = XML.format(processes=process(17, 'C') + process(18, 'G') + process(19, 'C+G'))
        result = subprocess.CompletedProcess([], 0, output, '')
        identity = {17: dict(ppid=55, exe='/own', start_ticks=42),
                    19: dict(ppid=88, exe='/foreign', start_ticks=43)}
        with patch.object(guard.subprocess, 'run', return_value=result) as run, \
                patch.object(guard, 'process_identity', side_effect=identity.get):
            gpu, apps = guard.GPUQuery(UUID)('/own', 55)
        run.assert_called_once_with(['nvidia-smi', '-i', UUID, '-q', '-x'],
                                    capture_output=True, text=True, timeout=2, check=False)
        self.assertEqual(gpu['query_source'], 'nvidia-smi -i UUID -q -x')
        self.assertEqual([(a['pid'], a['own_harness']) for a in apps], [(17, True), (19, False)])
        self.assertEqual(apps[1]['start_ticks'], 43)

    def test_registered_exit_identity_and_unknown_remain_fail_closed(self):
        output = XML.format(processes=process(17, 'C') + process(19, 'C'))
        exited = dict(ppid=55, exe='/own', start_ticks=42)
        with patch.object(guard.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, output, '')), \
                patch.object(guard, 'process_identity', return_value=None):
            _, apps = guard.GPUQuery(UUID, lambda pid: exited if pid == 17 else None)('/own', 55)
        self.assertEqual([(a['identity_state'], a['own_harness']) for a in apps],
                         [('registered_exited_own', True), ('unknown', False)])

    def test_guard_still_aborts_on_foreign_timeout_and_query_error(self):
        def run_sample(query, clock, boundary='between', active=None):
            g = guard.Guard(Args(), query=query)
            g.emit = lambda event, **kw: kw
            trips = []
            g.trip = lambda reasons, row: trips.extend(reasons)
            g.active = active
            with patch.object(guard.time, 'monotonic', side_effect=clock):
                row = g.sample(boundary)
            return row, trips

        row, trips = run_sample(lambda *_: ({'memory_used_mib': 1},
                                          [{'pid': 19, 'own_harness': False}]), [0, 1])
        self.assertEqual(row['foreign_compute_pids'], [19])
        self.assertIn('foreign_compute_process', trips)
        for boundary, active in [('before', {'guard_run_id':'a'}), ('during', {'guard_run_id':'a'}),
                                 ('after', {'guard_run_id':'a'}), ('final', None),
                                 ('between', {'guard_run_id':'a'})]:
            row, trips = run_sample(lambda *_: ({'memory_used_mib': 1}, []), [0, 2.001], boundary, active)
            self.assertFalse(row['ok'])
            self.assertFalse(row['idle_monitor_delay'])
            self.assertIn('monitor_sample_took_over_2_seconds', trips)
        def failed(*_):
            raise subprocess.TimeoutExpired('nvidia-smi', 2)
        row, trips = run_sample(failed, [0, 2.001])
        self.assertFalse(row['ok'])
        self.assertEqual(trips, ['monitor_query_failed'])

    def test_idle_delay_only_for_successful_empty_uncontaminated_samples(self):
        for boundary in ('idle', 'between'):
            g = guard.Guard(Args(), query=lambda *_: ({'memory_used_mib': 1}, []))
            g.emit = lambda event, **kw: kw
            g.trip = Mock()
            with patch.object(guard.time, 'monotonic', side_effect=[0, 2.05]):
                row = g.sample(boundary)
            self.assertTrue(row['idle_monitor_delay'])
            self.assertFalse(row['ok'])
            self.assertEqual(row['violation_reasons'], [])
            g.trip.assert_not_called()
        for gpu, apps in [({'memory_used_mib': 2049}, []),
                          ({'memory_used_mib': 1}, [{'pid': 9, 'own_harness': False}])]:
            g = guard.Guard(Args(), query=lambda *_, gpu=gpu, apps=apps: (gpu, apps))
            g.emit = lambda event, **kw: kw
            g.trip = Mock()
            with patch.object(guard.time, 'monotonic', side_effect=[0, 2.05]):
                row = g.sample('between')
            self.assertFalse(row['idle_monitor_delay'])
            g.trip.assert_called_once()

    def test_idle_sampling_and_run_start_are_serialized(self):
        entered, release, ran = threading.Event(), threading.Event(), threading.Event()
        def query(*_):
            entered.set()
            self.assertTrue(release.wait(5))
            return {'memory_used_mib': 1}, []
        g = guard.Guard(Args(), query=query)
        events = []
        g.emit = lambda event, **kw: events.append((event, kw)) or {'utc': 'x', **kw}
        sampler = threading.Thread(target=g.sample, args=('between',))
        class RunArgs: exe = sys.executable
        def original(*_):
            ran.set()
            return {'status': 'ok'}
        errors = []
        def run():
            try:
                g.wrapped(original)(RunArgs(), {'n':128, 'batch':1}, ('x', {}), 'x', 0, 0)
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=run)
        sampler.start()
        self.assertTrue(entered.wait(5))
        worker.start()
        self.assertFalse(ran.wait(.1))
        self.assertIsNone(g.active)
        release.set()
        sampler.join(5)
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(ran.is_set())
        self.assertEqual(events[0][0], 'sample')
        self.assertEqual(events[1][0], 'run_one_start')

    def test_stale_before_never_starts_subprocess(self):
        g = guard.Guard(Args())
        g.active = {'guard_run_id': 'a'}
        g.expected_exe = str(Path(sys.executable).resolve())
        g.before_ready = {'run_id': 'a', 'completed': 0}
        g.trip = Mock()
        original = Mock()
        with patch.object(guard.time, 'monotonic', return_value=2.001):
            with self.assertRaises(guard.GuardAbort):
                g.popen(original)([sys.executable])
        original.assert_not_called()
        g.trip.assert_called_once()


if __name__ == '__main__':
    unittest.main()
