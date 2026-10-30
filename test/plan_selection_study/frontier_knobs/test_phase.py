import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import frontier_knobs_phase as phase


class PhaseTests(unittest.TestCase):
    def source_fixture(self, root):
        protocol=root/'protocol.json';protocol.write_text('{}',encoding='utf8')
        gate=root/'gate.json';gate.write_text('{}',encoding='utf8')
        admission=root/'experiments/artifacts/new-admission';(admission/'build').mkdir(parents=True)
        for name in ('vkfft_frontier_original','vkfft_frontier_v2'):
            (admission/'build'/name).write_text(name,encoding='utf8')
        source=root/'experiments/artifacts/new-search';(source/'search').mkdir(parents=True)
        recpath=root/'record.json';recpath.write_text('{}',encoding='utf8')
        expected={name:phase.sha(admission/'build'/filename) for name,filename in
                  (('original','vkfft_frontier_original'),('expanded','vkfft_frontier_v2'))}
        contract={'protocol':'frontier-execution/v1','gpu':'GPU-new','execution_commit':'a'*40,
                  'binaries':expected,'source_files':{str(protocol.resolve()):phase.sha(protocol),str(gate.resolve()):phase.sha(gate)},
                  'vkfft_commit':'c'*40,'generated_provenance':{'sha256':'generated'}}
        work={'phase':'identity','workload_id':'n128_b1','candidate_id':'original','round':0,'order':0,
              'guard_run_id':'g:r'}
        rows=[{'record_type':'session_start','contract':contract},
              dict(record_type='execution',contract=contract,guard_id='g',guard_gpu_uuid='GPU-new',guard_clean=True,
                   command=['/fake/original'],workload={'n':128,'batch':1},status='ok',**work)]
        active=dict(work)
        def sample(boundary):return {'event':'sample','boundary':boundary,'gpu_uuid':'GPU-new','guard_id':'g',
                                      'gpu':{'uuid':'GPU-new'},'active_run':active,'ok':True,
                                      'violation_reasons':[],'foreign_compute_pids':[],'monitor_error':None}
        guard=[{'event':'guard_start','gpu_uuid':'GPU-new','guard_id':'g',
                'own_signal_method':'pidfd_with_rechecked_PPID_exe_starttime'},
               {'event':'run_one_start','gpu_uuid':'GPU-new','guard_id':'g','active_run':active},
               sample('before'),{'event':'own_harness_registered','gpu_uuid':'GPU-new','guard_id':'g',
                                 'active_run':active,'pidfd_retained':True,'identity':{'exe':'/fake/original'}},
               sample('after'),{'event':'run_one_end','gpu_uuid':'GPU-new','guard_id':'g','guard_run_id':'g:r'},
               {'event':'guard_complete','gpu_uuid':'GPU-new','guard_id':'g','clean':True,'success':True}]
        identity={'n':128,'batch':1,'candidate_id':'original','phase':'identity','round':0}
        def state(last,inflight):return {'contract':contract,'boot':'boot','start':110,'last':last,'inflight':inflight}
        budget=[{'event':'start','state':state(110,None)},
                {'event':'work_start','state':state(111,identity)},
                {'event':'work_finish','identity':identity,'status':'ok','state':state(112,None)}]
        paths={name:source/'search'/(name+'.jsonl') for name in ('rows','guard','incidents','budget')}
        items={'rows':rows,'guard':guard,'incidents':[],'budget':budget}
        for name,path in paths.items():path.write_text(''.join(json.dumps(x)+'\n' for x in items[name]),encoding='utf8')
        record={'id':'new-search','execution_commit':'a'*40,'artifacts':[]}
        def refresh():
            record['artifacts']=[{'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':phase.sha(path)}
                                 for path in paths.values()]
        refresh()
        campaign={'protocol':protocol,'gate':gate,'gpu':'GPU-new','admission_dir':admission,
                  'budget_state':{'boot':'boot','start':100},
                  'admission_contract':{'vkfft_commit':'c'*40,'generated_provenance':{'sha256':'generated'}}}
        audit_src=ROOT/'test/plan_selection_study/expanded_search/audit_guard.py'
        audit_dest=root/'test/plan_selection_study/expanded_search/audit_guard.py'
        audit_dest.parent.mkdir(parents=True);shutil.copyfile(audit_src,audit_dest)
        return campaign,source,recpath,record,paths,items,refresh

    def test_source_is_bound_to_device_gate_binaries_and_guard(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);campaign,source,recpath,record,paths,items,refresh=self.source_fixture(root)
            with mock.patch.object(phase.campaign_inputs,'load_completed_run',return_value=(record,source,recpath)):
                result=phase.verify_source(root,'new-search',campaign)
                self.assertEqual(result['guard_audit']['status'],'pass')
                self.assertEqual(result['guard_audit']['counts']['executions'],1)
                for change in ('gpu','gate','binaries','generated','vkfft'):
                    original=copy.deepcopy(items['rows']);bad=copy.deepcopy(original)
                    for row in bad:
                        c=row['contract']
                        if change=='gpu':c['gpu']='GPU-old'
                        elif change=='gate':c['source_files'][str(campaign['gate'].resolve())]='0'*64
                        elif change=='binaries':c['binaries']['original']='0'*64
                        elif change=='generated':c['generated_provenance']['sha256']='0'*64
                        else:c['vkfft_commit']='d'*40
                    paths['rows'].write_text(''.join(json.dumps(x)+'\n' for x in bad),encoding='utf8');refresh()
                    with self.subTest(change=change),self.assertRaises(ValueError):phase.verify_source(root,'new-search',campaign)
                    paths['rows'].write_text(''.join(json.dumps(x)+'\n' for x in original),encoding='utf8');refresh()
                paths['guard'].write_text('tampered',encoding='utf8')
                with self.assertRaises(ValueError):phase.verify_source(root,'new-search',campaign)

    def test_source_rejects_foreign_guard_and_boot(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);campaign,source,recpath,record,paths,items,refresh=self.source_fixture(root)
            with mock.patch.object(phase.campaign_inputs,'load_completed_run',return_value=(record,source,recpath)):
                wrong=copy.deepcopy(items['guard']);wrong[0]['gpu_uuid']='GPU-old'
                paths['guard'].write_text(''.join(json.dumps(x)+'\n' for x in wrong),encoding='utf8');refresh()
                with self.assertRaisesRegex(ValueError,'guard'):phase.verify_source(root,'new-search',campaign)
                paths['guard'].write_text(''.join(json.dumps(x)+'\n' for x in items['guard']),encoding='utf8');refresh()
                wrong=copy.deepcopy(items['budget']);wrong[0]['state']['boot']='old-boot'
                paths['budget'].write_text(''.join(json.dumps(x)+'\n' for x in wrong),encoding='utf8');refresh()
                with self.assertRaisesRegex(ValueError,'budget'):phase.verify_source(root,'new-search',campaign)
                wrong=copy.deepcopy(items['budget']);wrong[-1]['identity']['candidate_id']='unexpected'
                paths['budget'].write_text(''.join(json.dumps(x)+'\n' for x in wrong),encoding='utf8');refresh()
                with self.assertRaisesRegex(ValueError,'budget work identity'):phase.verify_source(root,'new-search',campaign)

    def test_main_propagates_selected_gpu_cuda_and_source(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);output=root/'output';output.mkdir()
            protocol=root/'protocol';gate=root/'gate';protocol.write_text('{}');gate.write_text('{}')
            campaign={'gpu':'GPU-fb86b36f-0a9f-6cbd-a36b-03b9073f4786','cuda_root':'/usr/local/cuda-12.6',
                      'protocol':protocol,'gate':gate,'admission_run':'new-admission','execution_commit':'a'*40}
            calls=[]
            def run(cmd,**kwargs):
                calls.append((cmd,kwargs));return SimpleNamespace(returncode=0,stdout='',stderr='')
            class Lock:
                def __init__(self,*a):pass
                def __enter__(self):return self
                def __exit__(self,*a):pass
            env={'EXPERIMENT_ARTIFACT_DIR':str(output),'EXPERIMENT_CODE_COMMIT':'b'*40,'PATH':'/bin','LD_LIBRARY_PATH':''}
            with mock.patch.dict(os.environ,env,clear=True), \
                 mock.patch.object(phase.campaign_inputs,'load_campaign',return_value=campaign) as load, \
                 mock.patch.object(phase.campaign_inputs,'validate_clock',return_value=10000), \
                 mock.patch.object(phase,'verify_source',return_value={'run':'new-search'}) as source, \
                 mock.patch.object(phase.time,'monotonic',return_value=100), \
                 mock.patch.object(phase.subprocess,'run',side_effect=run), \
                 mock.patch.object(phase.subprocess,'check_output',return_value='a'*40), \
                 mock.patch.dict(sys.modules,{'device_lock':SimpleNamespace(DeviceLock=Lock)}):
                self.assertEqual(phase.main(['--protocol','new-protocol','--admission-run','new-admission',
                                             '--gate','new-gate','--source-run','new-search']),0)
            load.assert_called_once_with(phase.ROOT,'new-protocol','new-admission','new-gate')
            source.assert_called_once_with(phase.ROOT,'new-search',campaign)
            self.assertEqual(calls[0][0][2],campaign['gpu'])
            cmake=next(cmd for cmd,kw in calls if cmd[0]=='cmake' and '-S' in cmd)
            self.assertIn('-DCUDAToolkit_ROOT=/usr/local/cuda-12.6',cmake)
            execution=next((cmd,kw) for cmd,kw in calls if str(cmd[1]).endswith('frontier_knobs\\runner.py') or str(cmd[1]).endswith('frontier_knobs/runner.py'))
            self.assertEqual(execution[0][execution[0].index('--gpu')+1],campaign['gpu'])
            self.assertTrue(execution[1]['env']['PATH'].startswith('/usr/local/cuda-12.6/bin:'))


if __name__=='__main__':unittest.main()
