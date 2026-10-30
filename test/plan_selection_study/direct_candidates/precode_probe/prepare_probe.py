#!/usr/bin/env python3
"""Generate an isolated diagnostic comparing pre-codegen and precompile fields.

This is an observation hook, not a dry-run API. It still generates and compiles
code normally, and must never be used as a production plan representation.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIRECT = HERE.parent
sys.path.insert(0, str(DIRECT))
import prepare_direct_headers as direct


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one anchor, got {text.count(old)}: {old[:100]!r}')
    return text.replace(old, new, 1)


def prepare(source, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty')
    direct.prepare(Path(source), output)
    plan = output / 'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h'
    contents = plan.read_text(encoding='utf8')
    anchor = '\t\tresFFT = vkfft_trace::invoke("shader_codegen",'
    contents = replace_once(contents, anchor,
        '\t\tvkfft_trace::capture_precodegen(app, axis);\n' + anchor)
    plan.write_text(contents, encoding='utf8')

    trace = (DIRECT.parent / 'instrumentation/trace.hpp').read_text(encoding='utf8')
    trace = replace_once(trace, 'inline std::vector<Kernel> kernels;',
        'inline std::vector<Kernel> kernels;\ninline std::map<std::pair<int,int>,std::string> precodegen;')
    trace = replace_once(trace, 'spans.clear(); stack.clear(); kernels.clear();',
        'spans.clear(); stack.clear(); kernels.clear(); precodegen.clear();')
    anchor = 'template<class App,class Axis> void capture_precompile(App* app,Axis* axis) {'
    trace = replace_once(trace, anchor, '''template<class App,class Axis> void capture_precodegen(App* app,Axis* axis) {
    auto key=std::make_pair(axis->specializationConstants.axis_id,axis->specializationConstants.axis_upload_id);
    if (precodegen.count(key)) throw std::runtime_error("duplicate pre-codegen snapshot");
    precodegen[key]=descriptor(app,axis,false);
}
''' + anchor)
    anchor = 'j.add("raw_descriptor",k.raw);'
    trace = replace_once(trace, anchor, '''auto key=std::make_pair(k.axis,k.upload);auto pre=precodegen.find(key);
        if(pre==precodegen.end()) throw std::runtime_error("missing pre-codegen snapshot");
        j.add("precodegen_raw_descriptor",pre->second);''' + anchor)
    (output / 'trace.hpp').write_text(trace, encoding='utf8')
    files = [plan, output / 'trace.hpp', output / 'direct_candidate.hpp', output.parent / 'direct_benchmark.cpp']
    metadata = {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    (output.parent / 'precode_probe_hashes.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
