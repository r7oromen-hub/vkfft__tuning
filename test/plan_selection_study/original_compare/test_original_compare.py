import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import original_compare_controller as c
import original_compare_guard as g
import run_suite

HERE = Path(__file__).resolve().parent

def mock(args,w,candidate,phase,round_no,order):
    variant,env=candidate; h='a'*64; options=['--gpu-architecture=sm_89']; desc={'localSize':[32,1,1]}
    kernel={'localSize':[32,1,1],'stageRadix':[8,8,4],'groupedBatch':1}
    if variant=='winner':
        kernel={'localSize':[16,1,1],'stageRadix':[16,16],'groupedBatch':1}
        desc={'localSize':[16,1,1]};h='b'*64
    correctness=dict(passed=True,nonfinite_count=0,failing_batches=0,checked_elements=w['n']*w['batch'])
    result=dict(status='ok',gpu={'uuid':args.gpu},header_instrumentation=(variant!='original'),n=w['n'],batch=w['batch'],seed=w['seed'],pattern=w['pattern'],
      requested_config={'makeForwardPlanOnly':1},validation_output_initialization='0xff_bytes_fp32_nan_before_single_and_final_timed_forward',
      input_unchanged=True,single_forward_input_unchanged=True,correctness=correctness,single_forward_correctness=correctness,
      structural_features={'numAxisUploads':1,'kernels':[kernel]},compiled_diagnostics=[{'cuda_num_regs':8}],
      resolved_config={'keepShaderCode':1 if variant=='original' else 0,'useLUT':0},graph_repeats=w['graph_repeats'],graph_kernel_count=w['graph_repeats'],graph_event_record_count=2,
      exec_ms_samples=[0.002]*w['repeats'],raw_interval_ms_samples=[1.0]*w['repeats'])
    if variant!='original':
        result.update(descriptor_validation={'all_match':True},precompile_kernels=[dict(canonical_descriptor=desc,source_sha256=h,source_bytes=42,compiler_options_match=True,actual_compiler_options=options,compiler_input={'original_compile_header_sha256':h})])
        if variant=='winner': result['direct_resource_validation']={'dynamic_shared_bytes':1,'static_shared_bytes':1,'function_dynamic_limit_bytes':10,'device_shared_limit_bytes':10,'threads':16,'function_thread_limit':1024}
    else:
        result.update(init_trace={'available':False},postinit_source_kernels=[dict(canonical_descriptor=desc,source_sha256=h,source_bytes=42,actual_compiler_options=None,expected_compiler_input={'options':options,'original_compile_header_sha256':h})])
    return dict(status='ok',guard_clean=True,direct_environment=env,result=result,workload=w)

class Tests(unittest.TestCase):
    def test_pair_workflow_and_freeze(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            exe1=Path(tmp)/'joint';exe2=Path(tmp)/'original';exe1.write_text('a');exe2.write_text('b');out=Path(tmp)/'rows.jsonl'
            calls=[]
            def observed(args,w,candidate,phase,round_no,order):
                self.assertEqual(c.freeze(args.exe),args.exe_sha256); calls.append((candidate[0],phase,w['pattern']))
                return mock(args,w,candidate,phase,round_no,order)
            argv=['compare','--joint-exe',str(exe1),'--joint-sha256',c.freeze(exe1),'--original-exe',str(exe2),'--gpu','GPU-test','--output',str(out),'--n','256','--batch','16','--graph-repeats','512','--rounds','2','--winner-radix','16,16','--winner-group','1','--winner-block','16,1,1']
            with patch.object(sys,'argv',argv),patch.dict(os.environ,{'VKFFT_STUDY_GUARD_ID':'x','VKFFT_STUDY_GUARD_UUID':'GPU-test'}),patch.object(c.suite,'run_one',observed): c.main()
            rows=[json.loads(x) for x in out.read_text().splitlines()]
            self.assertEqual(len(calls),21);self.assertEqual(rows[-1]['record_type'],'controller_complete')
            self.assertEqual(len(rows[-1]['joint_over_original_median_ratios']),2)
            self.assertEqual(len(rows[-1]['winner_over_original_median_ratios']),2)

    def test_reject_inequivalent_outputs(self):
        class Args: gpu='GPU-test';graph_repeats=512;winner_env={}
        w=dict(n=256,batch=16,seed=42,pattern='random',graph_repeats=512,repeats=1)
        a=mock(Args(),w,('joint',{}),'numeric',0,0)['result'];b=mock(Args(),w,('original',{}),'numeric',0,1)['result']
        c.compare(a,b)
        mutations=[lambda x:x['structural_features']['kernels'][0].update(localSize=[64,1,1]),
                   lambda x:x['postinit_source_kernels'][0].update(source_sha256='bad'),
                   lambda x:x['compiled_diagnostics'][0].update(cuda_num_regs=9),
                   lambda x:x['postinit_source_kernels'][0].update(actual_compiler_options=[])]
        for mutation in mutations:
            bad=copy.deepcopy(b);mutation(bad)
            with self.assertRaises(ValueError): c.compare(a,bad)

    def test_guard_and_timing_are_fail_closed(self):
        class Args: gpu='GPU-test';graph_repeats=512
        w=dict(n=256,batch=16,seed=42,pattern='random',graph_repeats=512,repeats=31)
        row=mock(Args(),w,('joint',{}),'timing',0,0)
        for change in (lambda x:x.update(guard_clean=False),lambda x:x['result'].update(header_instrumentation=False),lambda x:x['result'].update(raw_interval_ms_samples=[0.49]*31),lambda x:x['result']['correctness'].update(checked_elements=1)):
            bad=copy.deepcopy(row);change(bad)
            with self.assertRaises(ValueError):c.validate(bad,Args(),w,'joint',31)

    def test_guard_wraps_current_binary(self):
        class Args: gpu='GPU-test';log=Path('unused');incidents=Path('unused')
        guard=g.Guard(Args(),query=lambda *a:None);guard.emit=lambda *a,**kw:None;guard.sample=lambda b:{'utc':'x'}
        class Target:exe=sys.executable
        result=guard.wrapped(lambda *a:{'status':'ok'})(Target(),{'n':256,'batch':1},('joint',{}),'timing',0,0)
        self.assertTrue(result['guard_clean']);self.assertIsNone(guard.active)

    def test_runner_clears_direct_fields(self):
        class Args:exe=sys.executable;exe_sha256=c.freeze(sys.executable);timeout=2
        class Process:returncode=0;stdout='{}\n';stderr=''
        def fake(command,**kw):
            self.assertTrue(all(k not in kw['env'] for k in run_suite.FIELDS))
            return Process()
        w=dict(n=256,batch=1,seed=42,pattern='random',config={},graph_repeats=1,warmup=0,warmup_ms=0,repeats=1)
        with patch.dict(os.environ,{k:'bad' for k in run_suite.FIELDS}),patch.object(run_suite.subprocess,'run',fake):
            self.assertEqual(run_suite.run_one(Args(),w,('original',{}),'x',0,0)['status'],'ok')

    def test_winner_shape_and_limits(self):
        class Args:
            gpu='GPU-test';graph_repeats=512;winner_radix=[16,16];winner_group=1;winner_block=[16,1,1]
            winner_env={'VKFFT_DIRECT_RADIX':'16,16','VKFFT_DIRECT_BATCH':'1','VKFFT_DIRECT_BLOCK':'16,1,1'}
        w=dict(n=256,batch=4096,seed=42,pattern='random',graph_repeats=512,repeats=1)
        row=mock(Args(),w,('winner',Args.winner_env),'numerical',0,0)
        c.validate(row,Args(),w,'winner',1)
        for change in (lambda x:x['result']['structural_features']['kernels'][0].update(groupedBatch=2),
                       lambda x:x['result']['direct_resource_validation'].update(dynamic_shared_bytes=100),
                       lambda x:x.update(direct_environment={})):
            bad=copy.deepcopy(row);change(bad)
            with self.assertRaises(ValueError):c.validate(bad,Args(),w,'winner',1)

if __name__=='__main__':unittest.main()
