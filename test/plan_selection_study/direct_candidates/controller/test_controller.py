import contextlib, importlib.util, json, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import direct_controller as c
spec=importlib.util.spec_from_file_location('direct_gpu_guard',HERE/'direct_gpu_guard.py')
g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

class Tests(unittest.TestCase):
 def test_timing_requires_raw_interval(self):
  c.validate_timing({'exec_ms_samples':[0.002]*31,'raw_interval_ms_samples':[0.512]*31})
  with self.assertRaises(ValueError): c.validate_timing({'exec_ms_samples':[0.002]*31,'raw_interval_ms_samples':[0.49]*31})
  with self.assertRaises(ValueError): c.validate_timing({'exec_ms_samples':[0.002]*31,'raw_interval_ms_samples':[float('nan')]*31})

 def test_guard_wraps_exact_entry(self):
  class Args: gpu='GPU-test'; log=Path('unused'); incidents=Path('unused')
  guard=g.Guard(Args(),query=lambda *a: None)
  guard.emit=lambda *a,**kw: None
  guard.sample=lambda b: {'utc':'sample'}
  seen=[]
  def original(args,w,candidate,phase,round_no,order):
   seen.append((guard.active['candidate_id'],guard.expected_exe,phase,round_no,order))
   return {'status':'ok'}
  class Target: exe=sys.executable
  row=guard.wrapped(original)(Target(),{'n':16,'batch':1},('candidate',{}),'numeric',2,1)
  self.assertEqual(seen[0][0],'candidate'); self.assertTrue(row['guard_clean']); self.assertTrue(row['guard_run_id'])
  self.assertIsNone(guard.active)

 def test_full_sequence_and_identity_gate(self):
  with tempfile.TemporaryDirectory(dir=HERE) as td:
   output=Path(td)/'rows.jsonl'; calls=[]
   def mock(args,w,candidate,phase,round_no,order):
    name,env=candidate; calls.append((phase,name,w['pattern']))
    stage=[int(x) for x in env['VKFFT_DIRECT_RADIX'].split(',')] if env else [32,32,8,2]
    kernel={'stageRadix':stage,'localSize':[128,1,1],'groupedBatch':1}
    correctness={'passed':True,'nonfinite_count':0,'failing_batches':0}
    source='candidate' if name=='candidate' else 'baseline'
    result={'status':'ok','gpu':{'uuid':'GPU-test'},'validation_output_initialization':'0xff_bytes_fp32_nan_before_single_and_final_timed_forward',
       'n':w['n'],'batch':w['batch'],'seed':w['seed'],
       'requested_config':{'makeForwardPlanOnly':1,'aimThreads':128,'registerBoost':4,'useLUT':1,'registerBoostNonPow2':1},
       'input_unchanged':True,'single_forward_input_unchanged':True,'correctness':correctness,'single_forward_correctness':correctness,
       'descriptor_validation':{'all_match':True},'structural_features':{'numAxisUploads':1,'kernels':[kernel]},
       'precompile_kernels':[{'canonical_descriptor':{'name':source},'source_sha256':source,'compiler_input':{},'actual_compiler_options':[], 'compiler_options_match':True}],
       'graph_repeats':w['graph_repeats'],'graph_kernel_count':w['graph_repeats'],'graph_event_record_count':2,
       'direct_resource_validation':{'dynamic_shared_bytes':0,'static_shared_bytes':0,'function_dynamic_limit_bytes':1024,'device_shared_limit_bytes':1024,'threads':128,'function_thread_limit':1024},
       'exec_ms_samples':[1.0]*w['repeats'],'raw_interval_ms_samples':[1.0]*w['repeats']}
    return {'status':'ok','guard_clean':True,'result':result,'phase':phase,'workload':w,'direct_environment':env}
   argv=['direct_controller.py','--exe',sys.executable,'--gpu','GPU-test','--case','n16384','--output',str(output),'--rounds','2']
   with patch.object(sys,'argv',argv), patch.dict(os.environ,{'VKFFT_STUDY_GUARD_ID':'test','VKFFT_STUDY_GUARD_UUID':'GPU-test'}), patch.object(c.suite,'run_one',mock): c.main()
   self.assertEqual([x[0] for x in calls[:2]],['baseline','identity'])
   self.assertEqual(sum(x[0]=='numerical' for x in calls),10)
   self.assertEqual(sum(x[0]=='timing' for x in calls),4)
   self.assertEqual(json.loads(output.read_text().splitlines()[-1])['record_type'],'controller_complete')

if __name__=='__main__': unittest.main()
