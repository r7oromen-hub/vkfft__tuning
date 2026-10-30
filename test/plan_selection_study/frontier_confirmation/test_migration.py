import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT=Path(__file__).resolve().parents[3]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
driver=load('confirmation_migration_driver',ROOT/'tools/frontier_confirmation_phase.py')
verify=load('admission_migration_verify',ROOT/'tools/frontier_verify_admission.py')


class MigrationTests(unittest.TestCase):
    def fixture(self,base):
        protocol=base/'protocol.json';gate=base/'gate.json'
        protocol.write_text('{}',encoding='utf8');gate.write_text('{}',encoding='utf8')
        source=base/'source';(source/'search').mkdir(parents=True)
        contract={'gpu':'GPU-new','execution_commit':'a'*40,'binaries':{'original':'x','expanded':'y'},
                  'vkfft_commit':'b'*40,'generated_provenance':{'sha256':'z'},
                  'source_files':{str(protocol):driver.sha(protocol),str(gate):driver.sha(gate)}}
        for name in ('guard','incidents','budget'):(source/'search'/f'{name}.jsonl').write_text('',encoding='utf8')
        (source/'search'/'rows.jsonl').write_text(json.dumps({'record_type':'session_start','contract':contract})+'\n',encoding='utf8')
        budget={'event':'start','state':{'contract':contract,'boot':'boot','start':120,'last':120,'inflight':None}}
        (source/'search'/'budget.jsonl').write_text(json.dumps(budget)+'\n',encoding='utf8')
        ctx={'gpu':'GPU-new','protocol':protocol,'gate':gate,'admission_run':'new-admission',
             'admission_contract':copy.deepcopy(contract),'budget_state':{'boot':'boot','start':100},
             'campaign_start':{'start_monotonic':100},'config':{'total_reserved_wall_seconds':1000}}
        record={'execution_commit':'a'*40}
        return source,ctx,record,contract

    def test_screen_source_binds_gpu_commit_gate_and_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);source,ctx,record,contract=self.fixture(root)
            with mock.patch.object(driver.campaign,'load_completed_run',return_value=(record,source,root/'record')), \
                 mock.patch.object(driver.campaign,'verify_artifacts') as verified:
                paths,rows=driver.screen_source(root,ctx,'new-screen')
                self.assertEqual(rows[0]['contract']['gpu'],'GPU-new')
                verified.assert_called_once();self.assertEqual(set(paths),{'rows','guard','incidents','budget'})
                def mismatch(field,value):
                    altered=copy.deepcopy(contract);altered[field]=value
                    paths['rows'].write_text(json.dumps({'contract':altered})+'\n',encoding='utf8')
                    with self.assertRaises(ValueError):driver.screen_source(root,ctx,'new-screen')
                mismatch('gpu','GPU-old');mismatch('execution_commit','c'*40)
                mismatch('binaries',{'original':'old','expanded':'y'})
                mismatch('generated_provenance',{'sha256':'old'})
                mismatch('source_files',{str(ctx['protocol']):driver.sha(ctx['protocol'])})

    def test_admission_can_be_source_without_a_future_gate_reference(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);source,ctx,record,contract=self.fixture(root)
            contract['source_files'].pop(str(ctx['gate']))
            (source/'admission.jsonl').write_text(json.dumps({'contract':contract})+'\n',encoding='utf8')
            for name in ('guard','incidents','budget'):(source/f'{name}.jsonl').write_text('',encoding='utf8')
            budget={'event':'start','state':{'contract':contract,'boot':'boot','start':100,'last':100,'inflight':None}}
            (source/'budget.jsonl').write_text(json.dumps(budget)+'\n',encoding='utf8')
            with mock.patch.object(driver.campaign,'load_completed_run',return_value=(record,source,root/'record')), \
                 mock.patch.object(driver.campaign,'verify_artifacts'):
                paths,_=driver.screen_source(root,ctx,'new-admission')
            self.assertEqual(paths['rows'],source/'admission.jsonl')

    def test_verify_wrapper_uses_protocol_gpu_and_admission_count(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            base=Path(tmp);output=base/'output';output.mkdir()
            (base/'one.manifest.jsonl').write_text('',encoding='utf8')
            config={'gpu_uuid':'GPU-new','admission_workloads':[{'requests':[1,2,3]}]}
            record={'execution_commit':'a'*40,'artifacts':[]};called=[]
            def run(cmd,**kwargs):called.append(cmd);return SimpleNamespace(returncode=0,stdout='',stderr='')
            with mock.patch.dict(os.environ,{'EXPERIMENT_ARTIFACT_DIR':str(output)}), \
                 mock.patch.object(verify.campaign,'load_protocol',return_value=(base/'p',config)), \
                 mock.patch.object(verify.campaign,'load_completed_run',return_value=(record,base,base/'record')), \
                 mock.patch.object(verify.campaign,'verify_artifacts'), \
                 mock.patch.object(verify.subprocess,'run',side_effect=run):
                self.assertEqual(verify.main(['--protocol',str(base/'p'),'--admission-run','new']),0)
            command=called[-1]
            self.assertEqual(command[command.index('--gpu')+1],'GPU-new')
            self.assertEqual(command[command.index('--expected-count')+1],'3')
            self.assertEqual(command[command.index('--execution-commit')+1],record['execution_commit'])


if __name__=='__main__':unittest.main()
