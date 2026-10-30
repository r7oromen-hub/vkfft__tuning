#!/usr/bin/env python3
"""Create a new rotation tree with CUDA 12/13 graph dependency compatibility.

The original generator and prior generated/build trees are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import prepare_rotation as v1

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(generated, output):
    output=Path(output).resolve()
    provenance=v1.prepare(generated,output)
    path=output/'rotating_benchmark.cpp'
    text=path.read_text(encoding='utf8')
    text=v1.once(text,'#include <vkFFT.h>','#include <vkFFT.h>\n#include "rotation_cuda_compat.hpp"')
    text=v1.once(text,'cudaGraphNodeGetDependencies(nodes[i],nullptr,&count_dep)',
                 'rotation_graph_dependencies(nodes[i],nullptr,&count_dep)')
    text=v1.once(text,'cudaGraphNodeGetDependencies(nodes[i],&predecessor,&count_dep)',
                 'rotation_graph_dependencies(nodes[i],&predecessor,&count_dep)')
    path.write_text(text,encoding='utf8')
    (output/'rotation_cuda_compat.hpp').write_bytes((HERE/'rotation_cuda_compat.hpp').read_bytes())
    provenance['previous_rotating_benchmark_sha256']=provenance['rotating_benchmark.cpp']
    provenance['rotating_benchmark.cpp']=sha(path)
    provenance['rotation_cuda_compat.hpp']=sha(output/'rotation_cuda_compat.hpp')
    provenance['prepare_rotation_v2_sha256']=sha(HERE/'prepare_rotation_v2.py')
    provenance['prepare_rotation_v1_sha256']=sha(HERE/'prepare_rotation.py')
    provenance['api_compatibility']='CUDART_VERSION >= 13000 uses four arguments; older uses three; null edgeData rejects lossy queries'
    (output/'rotation_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf8')
    return provenance


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expanded-generated-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();print(json.dumps(prepare(a.expanded_generated_root,a.output),indent=2))
