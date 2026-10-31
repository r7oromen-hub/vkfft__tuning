"""CPU-only checks of single-call query integration and the existing guard boundary."""
import subprocess
import unittest
from unittest.mock import patch

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
        def run_sample(query, clock):
            g = guard.Guard(Args(), query=query)
            g.emit = lambda event, **kw: kw
            trips = []
            g.trip = lambda reasons, row: trips.extend(reasons)
            with patch.object(guard.time, 'monotonic', side_effect=clock):
                row = g.sample('between')
            return row, trips

        row, trips = run_sample(lambda *_: ({'memory_used_mib': 1},
                                          [{'pid': 19, 'own_harness': False}]), [0, 1, 1])
        self.assertEqual(row['foreign_compute_pids'], [19])
        self.assertIn('foreign_compute_process', trips)
        row, trips = run_sample(lambda *_: ({'memory_used_mib': 1}, []), [0, 2.001, 2.001])
        self.assertFalse(row['ok'])
        self.assertIn('monitor_sample_took_over_2_seconds', trips)
        def failed(*_):
            raise subprocess.TimeoutExpired('nvidia-smi', 2)
        row, trips = run_sample(failed, [0, 2.001])
        self.assertFalse(row['ok'])
        self.assertEqual(trips, ['monitor_query_failed'])


if __name__ == '__main__':
    unittest.main()
