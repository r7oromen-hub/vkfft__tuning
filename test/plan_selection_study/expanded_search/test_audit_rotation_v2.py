import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_rotation_v2 as a


class RepairAuditTest(unittest.TestCase):
    def test_failure_and_exact_compatibility_transformation(self):
        source_root=Path(__file__).parent
        expected=a.literal_base_sources();oldhash={name:'old-'+name for name in expected}
        frozen=dict(oldhash,**{name:a.base.sha(source_root/name) for name in a.EXTRA})
        with tempfile.TemporaryDirectory(dir=source_root) as temp:
            root=Path(temp);repair=root/'repair';repair.mkdir()
            original='#include <vkFFT.h>\ncudaGraphNodeGetDependencies(nodes[i],nullptr,&count_dep);\ncudaGraphNodeGetDependencies(nodes[i],&predecessor,&count_dep);\n'
            oldsrc=root/'old.cpp';oldsrc.write_bytes(original.encode('utf8'));oldsha=a.base.sha(oldsrc)
            newtext=original.replace('#include <vkFFT.h>','#include <vkFFT.h>\n#include "rotation_cuda_compat.hpp"').replace(
                'cudaGraphNodeGetDependencies(nodes[i]','rotation_graph_dependencies(nodes[i]')
            (repair/'rotating_benchmark.cpp').write_bytes(newtext.encode('utf8'))
            (repair/'rotation_cuda_compat.hpp').write_bytes((source_root/'rotation_cuda_compat.hpp').read_bytes())
            oldprov=root/'old.json';oldprov.write_text(json.dumps({'rotating_benchmark.cpp':oldsha}),encoding='utf8')
            failure=root/'failed.json';failure.write_text(json.dumps({'argv':['cmake','--build','old'], 'returncode':1,
                'stderr':'cudaGraphNodeGetDependencies cudaGraphEdgeData'}),encoding='utf8')
            prior={'failure_record':a.base.sha(failure),'generated_source':oldsha,'provenance':a.base.sha(oldprov)}
            prov={'previous_rotating_benchmark_sha256':oldsha,'rotation_cuda_compat.hpp':frozen['rotation_cuda_compat.hpp'],
                  'prepare_rotation_v1_sha256':frozen['prepare_rotation.py'],'prepare_rotation_v2_sha256':frozen['prepare_rotation_v2.py']}
            (repair/'rotation_provenance.json').write_text(json.dumps(prov),encoding='utf8')
            old={'event':'checkpoint','kind':'rotation_source_freeze','source_hashes':oldhash,'input_tree_sha256':{'x':'y'},
                 'gate_sha256':'gate','confirmed_hot_sha256':'hot'}
            new={'event':'checkpoint','kind':'rotation_repair_source_freeze','source_hashes':frozen,'input_tree_sha256':{'x':'y'},
                 'gate_sha256':'gate','confirmed_hot_sha256':'hot','v1_failure_evidence_sha256':prior}
            build={'event':'checkpoint','kind':'rotation_repair_build_complete','source_hashes':frozen,'input_tree_sha256':{'x':'y'}}
            oldsteps=[{'event':'work_finished','kind':'rotation_build','status':'ok','exit_code':0} for _ in range(2)]
            oldsteps.append({'event':'work_finished','kind':'rotation_build','status':'failed','exit_code':1,
                             'identity':{'argv':['cmake','--build','old']}})
            newsteps=[{'event':'work_finished','kind':'rotation_build','status':'ok','exit_code':0} for _ in range(7)]
            ledger=[old,*oldsteps,new,*newsteps,build]
            args=(ledger,{},root/'a',root/'b',repair,root/'generated',root/'commands',root/'deps',
                  '/original','/generated',source_root,root/'generated-tree',root/'original-tree',root/'instrumentation',failure,oldsrc,oldprov)
            with patch.object(a.base,'audit_build',return_value={'status':'pass','errors':[]}):
                ok=a.repair_audit_build(*args);self.assertEqual(ok['status'],'pass',ok['errors'])
                (repair/'rotating_benchmark.cpp').write_bytes((newtext+'tamper').encode('utf8'))
                self.assertEqual(a.repair_audit_build(*args)['status'],'fail')


if __name__=='__main__':unittest.main()
