import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import direct_block_controller as b

class Tests(unittest.TestCase):
 def baseline(self,swapped=0):
  local=[4,64,1] if swapped else [64,4,1]
  return {'n':256,'batch':16,'resolved_config':{'sharedMemorySize':49152},
    'structural_features':{'numAxisUploads':1,'kernels':[{'localSize':local,'groupedBatch':4}]},
    'precompile_kernels':[{'raw_descriptor':{'axisSwapped':swapped,'fftDim':256,'complexSize':8}}]}
 def test_predict_block(self):
  for swapped in (0,1):
   base=self.baseline(swapped)
   self.assertEqual(b.predict_block(base,1),([64,1,1],64))
   self.assertEqual(b.predict_block(base,2),([2,64,1],64))
   self.assertEqual(b.predict_block(base,4),([4,64,1],64))
  base=self.baseline();base['resolved_config']['sharedMemorySize']=1024
  self.assertEqual(b.predict_block(base,4),([64,4,1],64))
  base=self.baseline();base['structural_features']['kernels'][0]['groupedBatch']=3
  with self.assertRaises(ValueError): b.predict_block(base,2)
 def test_guarded_flow_and_group(self):
  with tempfile.TemporaryDirectory(dir=HERE) as td:
   output=Path(td)/'result.jsonl'; calls=[]
   def fake(args,w,candidate,phase,round_no,order):
    name,env=candidate; calls.append((phase,name))
    group=int(env.get('VKFFT_DIRECT_BATCH','1')); swapped=int(group>1); block=[group,64,1] if swapped else [64,1,1]
    rad=[4,4,4,4]; check={'passed':True,'nonfinite_count':0,'failing_batches':0}
    k={'stageRadix':rad,'localSize':block,'groupedBatch':group}
    pre={'raw_descriptor':{'axisSwapped':swapped,'fftDim':256,'complexSize':8},'canonical_descriptor':{'same':True},'source_sha256':'identity','compiler_input':{},'actual_compiler_options':[],'compiler_options_match':True}
    r={'n':w['n'],'batch':w['batch'],'seed':w['seed'],'requested_config':{'makeForwardPlanOnly':1,'aimThreads':128,'registerBoost':4,'useLUT':1,'registerBoostNonPow2':1},
       'gpu':{'uuid':'GPU-test'},'status':'ok','validation_output_initialization':'0xff_bytes_fp32_nan_before_single_and_final_timed_forward',
       'single_forward_input_unchanged':True,'input_unchanged':True,'single_forward_correctness':check,'correctness':check,
       'resolved_config':{'sharedMemorySize':49152},'descriptor_validation':{'all_match':True},
       'structural_features':{'numAxisUploads':1,'kernels':[k]},'precompile_kernels':[pre],
       'graph_repeats':w['graph_repeats'],'graph_kernel_count':w['graph_repeats'],'graph_event_record_count':2,
       'direct_resource_validation':{'dynamic_shared_bytes':0,'static_shared_bytes':0,'function_dynamic_limit_bytes':1024,'device_shared_limit_bytes':1024,'threads':64*group,'function_thread_limit':1024},
       'exec_ms_samples':[1.0]*w['repeats'],'raw_interval_ms_samples':[1.0]*w['repeats']}
    return {'status':'ok','guard_clean':True,'result':r,'phase':phase,'workload':w,'direct_environment':env}
   argv=['block','--exe',sys.executable,'--gpu','GPU-test','--case','n256','--group','4','--output',str(output),'--rounds','1']
   with patch.object(sys,'argv',argv),patch.dict(os.environ,{'VKFFT_STUDY_GUARD_ID':'fake','VKFFT_STUDY_GUARD_UUID':'GPU-test'}),patch.object(b.suite,'run_one',fake): b.main()
   rows=[json.loads(x) for x in output.read_text().splitlines()]
   self.assertEqual(rows[-1]['candidate_block'],[4,64,1]);self.assertEqual(rows[-1]['fft_threads'],64)
   self.assertEqual(sum(x[0]=='numerical' for x in calls),10);self.assertEqual(sum(x[0]=='timing' for x in calls),2)

if __name__=='__main__': unittest.main()
