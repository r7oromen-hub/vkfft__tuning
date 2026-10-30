import hashlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import prepare_headers
import prepare_rotation
import prepare_rotation_v2
if os.name=='nt' and 'fcntl' not in sys.modules:sys.modules['fcntl']=types.ModuleType('fcntl')
import rotation_runner_v2

HERE=Path(__file__).resolve().parent


class RotationV2Test(unittest.TestCase):
    def test_archived_cuda13_failure_and_budget_ledger_binding(self):
        folder=HERE/'results_20261030'
        failure=folder/'rotation_build_v1_2.json';source=folder/'rotating_benchmark_v1.cpp';provenance=folder/'rotation_provenance_v1.json'
        if not all(path.is_file() for path in (failure,source,provenance)):
            self.skipTest('downloaded v1 CUDA13 failure evidence not present')
        rows=[{'event':'work_finished','kind':'rotation_build','status':'failed','result_path':str(failure),'exit_code':2}]
        before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (failure,source,provenance)}
        result=rotation_runner_v2.verify_v1_failure(rows,failure,source,provenance)
        self.assertEqual(result['generated_source'],before[source])
        self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        for invalid in ([],[dict(rows[0],exit_code=1)],[dict(rows[0],status='ok')],rows+rows):
            with self.assertRaises(ValueError):rotation_runner_v2.verify_v1_failure(invalid,failure,source,provenance)

    def test_new_tree_only_and_two_dependency_calls_wrapped(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root=Path(temp);generated=root/'expanded';old=root/'rotation_v1';new=root/'rotation_v2'
            prepare_headers.prepare(HERE.parents[2]/'VkFFT/vkFFT',generated/'include')
            prepare_rotation.prepare(generated,old)
            old_before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in old.iterdir() if p.is_file()}
            provenance=prepare_rotation_v2.prepare(generated,new)
            self.assertEqual(old_before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in old.iterdir() if p.is_file()})
            code=(new/'rotating_benchmark.cpp').read_text(encoding='utf8')
            self.assertEqual(code.count('rotation_graph_dependencies(nodes[i]'),2)
            self.assertNotIn('cudaGraphNodeGetDependencies(nodes[i]',code)
            self.assertLess(code.index('#include <vkFFT.h>'),code.index('#include "rotation_cuda_compat.hpp"'))
            self.assertIn('rotation_graph_kernel_arguments_and_order_verified',code)
            self.assertEqual(provenance['previous_rotating_benchmark_sha256'],old_before['rotating_benchmark.cpp'])
            self.assertEqual(provenance['rotating_benchmark.cpp'],hashlib.sha256((new/'rotating_benchmark.cpp').read_bytes()).hexdigest())
            with self.assertRaises(ValueError):prepare_rotation_v2.prepare(generated,old)


if __name__=='__main__':unittest.main()
