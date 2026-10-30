#!/usr/bin/env python3
"""Audit isolated CUDA-13 rotation repair while retaining the v1 failure.

All budget and execution checks use the unmodified ledger. The compatibility
build is mapped to the frozen v1 build auditor only after separately checking
the old failure, both freezes, every repair step and all added source hashes.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import audit_rotation as base


EXTRA={'rotation_runner_v2.py','prepare_rotation_v2.py','rotation_cuda_compat.hpp',
       'test_rotation_cuda_compat.cpp','test_rotation_v2.py','rotation_compat_cpu/CMakeLists.txt'}


def literal_base_sources():
    module=ast.parse(Path(base.__file__).read_text(encoding='utf8'))
    item=next(n for n in ast.walk(module) if isinstance(n,ast.Assign) and
              any(isinstance(t,ast.Name) and t.id=='expected_sources' for t in n.targets))
    return ast.literal_eval(item.value)


def repair_audit_build(ledger,header,original_binary,expanded_binary,rotation_source,generated_benchmark,
                       compile_commands,original_dependencies,original_include_prefix,generated_include_prefix,
                       frozen_source_root,expanded_generated_root,original_header_root,instrumentation_root,
                       failure_record,old_source,old_provenance):
    errors=[]
    old_freezes=[x for x in ledger if x.get('event')=='checkpoint' and x.get('kind')=='rotation_source_freeze']
    new_freezes=[x for x in ledger if x.get('event')=='checkpoint' and x.get('kind')=='rotation_repair_source_freeze']
    builds=[x for x in ledger if x.get('event')=='checkpoint' and x.get('kind')=='rotation_repair_build_complete']
    old_builds=[x for x in ledger if x.get('event')=='checkpoint' and x.get('kind')=='rotation_build_complete']
    if len(old_freezes)!=1 or len(new_freezes)!=1 or len(builds)!=1 or old_builds:
        return {'status':'fail','errors':['v1/v2 source or build checkpoints are missing, duplicated, or overwritten']}
    old,new,build=old_freezes[0],new_freezes[0],builds[0]
    oldpos,newpos,buildpos=ledger.index(old),ledger.index(new),ledger.index(build)
    if not oldpos<newpos<buildpos:errors.append('repair checkpoint order invalid')
    oldsteps=[x for x in ledger[oldpos:newpos] if x.get('event')=='work_finished' and x.get('kind')=='rotation_build']
    steps=[x for x in ledger[newpos:buildpos] if x.get('event')=='work_finished' and x.get('kind')=='rotation_build']
    if len(oldsteps)!=3 or [x.get('status') for x in oldsteps]!=['ok','ok','failed']:
        errors.append('v1 build failure and preceding steps are not preserved')
    if len(steps)!=7 or not all(x.get('status')=='ok' and x.get('exit_code')==0 for x in steps):
        errors.append('all seven v2 build and combined CPU regression steps must succeed')
    failure=base.load(failure_record);oldprov=base.load(old_provenance);oldtext=Path(old_source).read_text(encoding='utf8')
    oldhashes={'failure_record':base.sha(failure_record),'generated_source':base.sha(old_source),'provenance':base.sha(old_provenance)}
    if (not oldsteps or oldsteps[-1].get('exit_code')!=failure.get('returncode') or failure.get('returncode')==0 or
            oldsteps[-1].get('identity',{}).get('argv')!=failure.get('argv') or
            'cudaGraphNodeGetDependencies' not in failure.get('stderr','') or 'cudaGraphEdgeData' not in failure.get('stderr','') or
            oldprov.get('rotating_benchmark.cpp')!=oldhashes['generated_source'] or
            oldtext.count('cudaGraphNodeGetDependencies(nodes[i]')!=2 or new.get('v1_failure_evidence_sha256')!=oldhashes):
        errors.append('original CUDA13 build error or source provenance not bound to repair')
    expected=literal_base_sources();newhashes=new.get('source_hashes',{})
    if set(newhashes)!=expected|EXTRA:errors.append('repair source inventory missing or unexpected file')
    for name in EXTRA:
        if newhashes.get(name)!=base.sha(Path(frozen_source_root)/name):errors.append('repair frozen source mismatch: '+name)
    if (old.get('input_tree_sha256')!=new.get('input_tree_sha256') or
            any(old.get('source_hashes',{}).get(k)!=newhashes.get(k) for k in expected) or
            any(old.get(k)!=new.get(k) for k in ('gate_sha256','confirmed_hot_sha256'))):
        errors.append('repair does not preserve v1 input trees, original frozen sources and gate')
    provenance=base.load(Path(rotation_source)/'rotation_provenance.json');text=(Path(rotation_source)/'rotating_benchmark.cpp').read_text(encoding='utf8')
    substitutions=[('#include <vkFFT.h>\n#include "rotation_cuda_compat.hpp"','#include <vkFFT.h>'),
                   ('rotation_graph_dependencies(nodes[i],nullptr,&count_dep)','cudaGraphNodeGetDependencies(nodes[i],nullptr,&count_dep)'),
                   ('rotation_graph_dependencies(nodes[i],&predecessor,&count_dep)','cudaGraphNodeGetDependencies(nodes[i],&predecessor,&count_dep)')]
    for newer,older in substitutions:
        if text.count(newer)!=1:errors.append('v2 generated source differs from exact expected compatibility replacement')
        text=text.replace(newer,older,1)
    if (hashlib.sha256(text.encode('utf8')).hexdigest()!=oldhashes['generated_source'] or
            provenance.get('previous_rotating_benchmark_sha256')!=oldhashes['generated_source'] or
            provenance.get('rotation_cuda_compat.hpp')!=base.sha(Path(rotation_source)/'rotation_cuda_compat.hpp') or
            provenance.get('rotation_cuda_compat.hpp')!=newhashes.get('rotation_cuda_compat.hpp') or
            provenance.get('prepare_rotation_v1_sha256')!=newhashes.get('prepare_rotation.py') or
            provenance.get('prepare_rotation_v2_sha256')!=newhashes.get('prepare_rotation_v2.py')):
        errors.append('repair is not the recorded, limited source transformation')
    # Reuse the frozen v1 auditor for binary, full base source inventory, trees,
    # provenance, include precedence and dependency isolation. Only repair
    # checkpoints and seven already independently checked build rows are mapped.
    clean_new=dict(new,kind='rotation_source_freeze',source_hashes={k:newhashes.get(k) for k in expected})
    clean_build=dict(build,kind='rotation_build_complete',source_hashes={k:build.get('source_hashes',{}).get(k) for k in expected})
    mapped=[clean_new]+steps[:7]+[clean_build]
    common=base.audit_build(mapped,header,original_binary,expanded_binary,rotation_source,generated_benchmark,
                            compile_commands,original_dependencies,original_include_prefix,generated_include_prefix,
                            frozen_source_root,expanded_generated_root,original_header_root,instrumentation_root)
    errors.extend(common['errors'])
    return {'status':'fail' if errors else 'pass','errors':errors,'v1_failure_evidence_sha256':oldhashes,
            'v2_source_freeze_sha256':newhashes,'common_build_audit':common,
            'scope':'v1 failure retained, repair-source transformation and separate build checked; full ledger independently reconciled'}


def main():
    p=argparse.ArgumentParser(add_help=False)
    for name in ('v1-failure-record','v1-generated-source','v1-provenance'):p.add_argument('--'+name,required=True,type=Path)
    args,remaining=p.parse_known_args()
    common=base.audit_build
    def proxy(*positional):
        try:
            base.audit_build=common
            return repair_audit_build(*positional,args.v1_failure_record,args.v1_generated_source,args.v1_provenance)
        finally:base.audit_build=proxy
    base.audit_build=proxy
    try:return base.main(remaining)
    finally:base.audit_build=common


if __name__=='__main__':raise SystemExit(main())
