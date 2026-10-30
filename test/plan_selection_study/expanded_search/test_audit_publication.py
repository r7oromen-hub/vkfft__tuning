import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

if sys.platform=='win32':sys.modules.setdefault('fcntl',types.SimpleNamespace())
import audit_runner


class Lock:
    fd=0
    def __enter__(self):return self
    def __exit__(self,*args):return None


class FakeBudget:
    def __init__(self,fail_finish=False):self.state={'phase':'confirmation','stopped':False,'inflight':None};self.fail_finish=fail_finish
    def may_start(self,*args,**kwargs):return True
    def finish_work(self,*args,**kwargs):
        if self.fail_finish:raise RuntimeError('ledger write failed')
    def checkpoint(self,*args,**kwargs):pass


class AuditPublicationTest(unittest.TestCase):
    def run_case(self,fail_finish):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp);gate=root/'final.json';audit=root/'audit.json'
            def fake_subprocess(cmd,**kwargs):
                def opt(name):return Path(cmd[cmd.index(name)+1])
                opt('--audit').write_text('{"status":"pass"}\n',encoding='utf8')
                opt('--gate').write_text(json.dumps({'status':'pass','audit_sha256':audit_runner.runner.file_sha(opt('--audit'))}),encoding='utf8')
                return types.SimpleNamespace(returncode=0,stdout='',stderr='')
            def fake_link(src,dst):
                with Path(dst).open('xb') as out:out.write(Path(src).read_bytes())
            args=['audit_runner','--kind','screen','--state',str(root),'--rows',str(root/'rows'),
                  '--guard',str(root/'guard'),'--incidents',str(root/'incidents'),'--budget-snapshot',str(root/'budget'),
                  '--protocol',str(root/'protocol'),'--audit',str(audit),'--gate',str(gate),'--manifests',str(root/'manifests'),
                  '--previous-rows',str(root/'prior')]
            with patch.object(sys,'argv',args),patch.object(audit_runner.device_lock,'DeviceLock',return_value=Lock()), \
                 patch.object(audit_runner.budget,'Budget',return_value=FakeBudget(fail_finish)), \
                 patch.object(audit_runner.subprocess,'run',side_effect=fake_subprocess),patch.object(audit_runner.os,'link',side_effect=fake_link):
                if fail_finish:
                    with self.assertRaisesRegex(RuntimeError,'ledger write failed'):audit_runner.main()
                    self.assertFalse(gate.exists())
                else:
                    self.assertEqual(audit_runner.main(),0);self.assertTrue(gate.exists())
                self.assertEqual(len(list(root.glob('final.json.pending-*'))),1)

    def test_successful_audit_publishes(self):self.run_case(False)
    def test_ledger_failure_never_publishes(self):self.run_case(True)


if __name__=='__main__':unittest.main()
