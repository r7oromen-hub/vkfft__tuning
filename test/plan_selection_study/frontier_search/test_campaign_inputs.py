import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import frontier_campaign_inputs as inputs


class CampaignInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.run_id='frontier-new-admission-001';self.commit='a'*40;self.parent='b'*40
        self.gpu='GPU-fb86b36f-0a9f-6cbd-a36b-03b9073f4786'
        self.protocol=self.root/'experiments/protocol.json';self.protocol.parent.mkdir(parents=True)
        self.cfg={'schema':'frontier-campaign/v1','gpu_uuid':self.gpu,'cuda_root':'/usr/local/cuda-12.6',
                  'total_reserved_wall_seconds':1000,'admission_workloads':[{'n':128,'batch':16,'requests':[{'x':1}]}]}
        self.protocol.write_text(json.dumps(self.cfg),encoding='utf8')
        self.art=self.root/'experiments/artifacts'/self.run_id;self.art.mkdir(parents=True)
        def put(rel,content):
            path=self.art/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf8');return path
        original=put('build/vkfft_frontier_original','original');expanded=put('build/vkfft_frontier_v2','expanded')
        commands=put('build/compile_commands.json','[]'); generated=put('generated/expanded_provenance.json','{}')
        manifest=put('n128_b16.manifest.jsonl','{}\n')
        contract={'gpu':self.gpu,'execution_commit':self.commit,
                  'binaries':{'original':inputs.sha(original),'expanded':inputs.sha(expanded)},
                  'generated_provenance':{'sha256':inputs.sha(generated)},
                  'source_files':{'/remote/repo/experiments/protocol.json':inputs.sha(self.protocol)},
                  'build_artifacts':{'/remote/repo/build/compile_commands.json':inputs.sha(commands)},
                  'manifests':{'/remote/repo/n128_b16.manifest.jsonl':inputs.sha(manifest)},'vkfft_commit':'c'*40}
        self.contract=contract
        put('admission.jsonl',json.dumps({'record_type':'session_start','contract':contract})+'\n')
        put('guard.jsonl','{}\n');put('incidents.jsonl','')
        state={'contract':contract,'boot':'boot','start':110,'last':120,'inflight':None}
        put('budget.jsonl',json.dumps({'event':'start','state':state})+'\n')
        put('campaign_start.json',json.dumps({'start_monotonic':100,'total_reserved_wall_seconds':1000,
                                              'gpu':self.gpu,'execution_commit':self.commit}))
        self.record_path=self.root/'experiments/records'/f'{self.run_id}.json';self.record_path.parent.mkdir(parents=True)
        self.record={'schema':'git-experiment/v1','id':self.run_id,'status':'completed','returncode':0,
                     'execution_commit':self.commit,'protocol':{'path':'experiments/protocol.json',
                       'bytes':self.protocol.stat().st_size,'sha256':inputs.sha(self.protocol)},
                     'artifacts':[{'path':p.relative_to(self.root).as_posix(),'bytes':p.stat().st_size,'sha256':inputs.sha(p)}
                                  for p in self.art.rglob('*') if p.is_file()]}
        self.record_path.write_text(json.dumps(self.record),encoding='utf8')
        names=('admission.jsonl','guard.jsonl','incidents.jsonl','budget.jsonl','n128_b16.manifest.jsonl')
        self.gate_path=self.root/'experiments/evidence/gate.json';self.gate_path.parent.mkdir(parents=True)
        self.gate={'schema':'frontier-admission-gate/v1','status':'pass','gpu':self.gpu,
                   'execution_commit':self.commit,'candidates':1,'workloads':1,'binary_hashes_rechecked':True,
                   'guard_audit':{'status':'pass'},'evidence_sha256':{str(self.art/n):inputs.sha(self.art/n) for n in names}}
        self.gate_path.write_text(json.dumps(self.gate),encoding='utf8')
        self.patch=mock.patch.object(inputs,'git',side_effect=self.git);self.patch.start();self.addCleanup(self.patch.stop)

    def git(self,root,*args):
        if args[0]=='ls-files':return args[-1]
        if args[0]=='merge-base':return ''
        if args[0]=='rev-parse':return 'c'*40 if args[1]=='HEAD:VkFFT' else self.parent
        if args[0]=='diff':return '' if args[1]=='--quiet' else self.record_path.relative_to(self.root).as_posix()
        if args[0]=='show':return json.dumps({'schema':'git-experiment/v1','status':'declared','id':self.run_id,
                                                'parent_code_commit':self.parent,'protocol':self.record['protocol']})
        raise AssertionError(args)

    def load(self):return inputs.load_campaign(self.root,self.protocol,self.run_id,self.gate_path)

    def test_new_gpu_cuda_and_commit_bound(self):
        campaign=self.load()
        self.assertEqual((campaign['gpu'],campaign['cuda_root'],campaign['execution_commit']),
                         (self.gpu,'/usr/local/cuda-12.6',self.commit))
        self.assertEqual(inputs.validate_clock(campaign,now=150,boot='boot'),1100)
        with self.assertRaises(ValueError):inputs.validate_clock(campaign,now=150,boot='other')
        with self.assertRaises(ValueError):inputs.validate_clock(campaign,now=1100,boot='boot')

    def test_reject_wrong_gpu_commit_and_modified_binary(self):
        for field,value in (('gpu','GPU-3559ec6b-6278-8f93-0706-5f8a85c4911b'),('execution_commit','c'*40)):
            gate=copy.deepcopy(self.gate);gate[field]=value;self.gate_path.write_text(json.dumps(gate),encoding='utf8')
            with self.assertRaises(ValueError):self.load()
        self.gate_path.write_text(json.dumps(self.gate),encoding='utf8')
        (self.art/'build/vkfft_frontier_original').write_text('changed',encoding='utf8')
        with self.assertRaisesRegex(ValueError,'artifact'):self.load()

    def test_reject_bad_ids_paths_duplicate_gate_and_artifact_entries(self):
        for value in ('../other','a/b','..'):
            with self.assertRaises(ValueError):inputs.load_completed_run(self.root,value,self.protocol)
        with self.assertRaises(ValueError):inputs.load_protocol(self.root,ROOT/'README.md')
        gate=copy.deepcopy(self.gate);gate['evidence_sha256']['C:/unrelated/admission.jsonl']='f'*64
        self.gate_path.write_text(json.dumps(gate),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'ambiguous'):self.load()
        record=copy.deepcopy(self.record);record['artifacts'].append(copy.deepcopy(record['artifacts'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):inputs.verify_artifacts(self.root,record,[self.art/'budget.jsonl'])

    def test_reject_wrong_record_protocol_or_execution_declaration(self):
        record=copy.deepcopy(self.record);record['protocol']['sha256']='0'*64
        self.record_path.write_text(json.dumps(record),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'protocol'):self.load()
        self.record_path.write_text(json.dumps(self.record),encoding='utf8')
        with mock.patch.object(inputs,'git',side_effect=lambda root,*a: 'unrelated' if a[:2]==('diff','--name-only') else self.git(root,*a)):
            with self.assertRaisesRegex(ValueError,'declaration'):self.load()

    def test_uncommitted_tracked_input_is_rejected(self):
        def dirty(root,*args):
            if args[:2]==('diff','--quiet'):raise subprocess.CalledProcessError(1,'git diff')
            return self.git(root,*args)
        with mock.patch.object(inputs,'git',side_effect=dirty):
            with self.assertRaises(subprocess.CalledProcessError):inputs.load_protocol(self.root,self.protocol)


if __name__=='__main__':unittest.main()
