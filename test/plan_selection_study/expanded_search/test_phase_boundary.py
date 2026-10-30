import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

if sys.platform=='win32':sys.modules.setdefault('fcntl',types.SimpleNamespace())
import start_confirmation_phase as phase


class Lock:
    def __enter__(self):return self
    def __exit__(self,*args):return None


class FakeBudget:
    def __init__(self,ledger):self.ledger=ledger;self.state={'phase':'search','stopped':False,'inflight':None};self.fail_once=False
    def append(self,row):
        with self.ledger.open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
    def advance(self,reason):
        if self.fail_once:self.fail_once=False;raise RuntimeError('simulated crash')
        self.append({'event':'phase_advance','previous_phase':'search'});self.state['phase']='confirmation';return True
    def checkpoint(self,kind,**details):self.append({'event':'checkpoint','kind':kind,**details})


class PhaseTest(unittest.TestCase):
    def test_verified_boundary_and_crash_after_snapshot(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            p=Path(temp);rows=p/'rows';guard=p/'guard';inc=p/'inc';snap=p/'snapshot';ledger=p/'ledger'
            rows.write_text('\n'.join(map(json.dumps,[{'record_type':'search_protocol','session_id':'s'},
                 {'record_type':'execution','session_id':'s','guard_id':'g'},{'record_type':'search_complete','status':'budget_exhausted'}]))+'\n')
            guard.write_text('\n'.join(map(json.dumps,[{'event':'guard_start','guard_id':'g','gpu_uuid':phase.runner.GPU,
                 'script':'/x/search_controller.py','script_args':['--output',str(rows)]},
                 {'event':'guard_complete','guard_id':'g','gpu_uuid':phase.runner.GPU,'success':True,'clean':True,'script_exit_code':0}]))+'\n')
            inc.write_text('')
            argv=['--output',str(rows),'--log',str(guard),'--incidents',str(inc)]
            ledger.write_text('\n'.join(map(json.dumps,[{'event':'checkpoint','kind':'search_guard_launch','session_id':'launch','argv':argv},
                 {'event':'checkpoint','kind':'search_guard_exit','session_id':'launch','exit_code':0}]))+'\n')
            b=FakeBudget(ledger);b.fail_once=True
            args=['tool','--state',str(p),'--rows',str(rows),'--guard',str(guard),'--incidents',str(inc),'--budget-snapshot',str(snap)]
            with patch.object(sys,'argv',args),patch.object(phase.device_lock,'DeviceLock',return_value=Lock()),patch.object(phase.budget,'Budget',return_value=b):
                with self.assertRaisesRegex(RuntimeError,'simulated crash'):phase.main()
                self.assertTrue(snap.exists());phase.main();self.assertEqual(b.state['phase'],'confirmation')
                phase.main()  # existing exact snapshot and checkpoint are idempotent


if __name__=='__main__':unittest.main()
