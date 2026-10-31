#!/usr/bin/env python3
"""Generate an isolated nonuniform planner and harness from untouched VkFFT headers.

This composes the existing expanded generator, then changes only generated files.
It does not compile, execute or admit any candidate.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPANDED_GENERATOR = HERE.parent / 'expanded_search' / 'prepare_headers.py'
spec = importlib.util.spec_from_file_location('nonuniform_expanded_prepare', EXPANDED_GENERATOR)
expanded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expanded)


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected exactly one generated anchor, found {text.count(old)}: {old[:100]!r}')
    return text.replace(old, new, 1)


def transform_splitter(text):
    # The existing scalar R was both min and max.  Nonuniform FFT threads use
    # the active minimum R; the scheduler's clamp must still match exactly.
    text = once(text,
                'axis->axisBlock[0] != (pfUINT)(axis->specializationConstants.fftDim.data.i / request.regs)',
                'axis->axisBlock[0] != (pfUINT)(axis->specializationConstants.fftDim.data.i / request.min_regs)')
    return once(text,
                'request.batch > (long long)FFTPlan->actualFFTSizePerAxis[0][1] ||',
                'request.workload_batch != (long long)FFTPlan->actualFFTSizePerAxis[0][1] ||\n'
                '                    request.batch > (long long)FFTPlan->actualFFTSizePerAxis[0][1] ||')


def transform_plan(text):
    text = once(text,
                '!expanded_candidate::layout_matches((int)s.fftDim.data.i, request.regs, request.batch, (int)s.axisSwapped,',
                '!expanded_candidate::layout_matches((int)s.fftDim.data.i, request.min_regs, request.batch, (int)s.axisSwapped,')
    # The expanded wrapper's existing check is after shader generation. Also
    # fail closed before the pre-codegen snapshot and shader generator.
    anchor = '\t\tvkfft_trace::capture_precodegen(app, axis);'
    check = '''        if (expanded_candidate::enabled()) {
            expanded_candidate::Request request; auto& s = axis->specializationConstants;
            int stages[32];
            if (!expanded_candidate::request(request) || s.numStages < 1 || s.numStages > 32) {
                deleteVkFFT(app); return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
            }
            for (int i=0; i<s.numStages; ++i) stages[i]=s.stageRadix[i];
            if (!expanded_candidate::reorder(stages, s.numStages) ||
                !std::equal(stages, stages+s.numStages, s.stageRadix) ||
                !expanded_candidate::consistent((int)s.fftDim.data.i, s.min_registers_per_thread,
                                                s.registers_per_thread, s.registers_per_thread_per_radix, s.registerBoost) ||
                axis->groupedBatch != (pfUINT)request.batch ||
                !expanded_candidate::layout_matches((int)s.fftDim.data.i, request.min_regs, request.batch, (int)s.axisSwapped,
                    s.localSize[0].data.i, s.localSize[1].data.i, s.localSize[2].data.i)) {
                deleteVkFFT(app); return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
            }
        }
'''
    return once(text, anchor, check + anchor)


def transform_benchmark(text):
    old = '''        result.str("expanded_requested_radix", std::getenv("VKFFT_EXPANDED_RADIX"));
        result.str("expanded_requested_regs", std::getenv("VKFFT_EXPANDED_REGS"));
        result.str("expanded_requested_batch", std::getenv("VKFFT_EXPANDED_BATCH"));'''
    new = '''        result.str("nonuniform_request_version", std::getenv("VKFFT_NONUNIFORM_VERSION"));
        result.str("nonuniform_requested_n", std::getenv("VKFFT_NONUNIFORM_N"));
        result.str("nonuniform_requested_workload_batch", std::getenv("VKFFT_NONUNIFORM_WORKLOAD_BATCH"));
        result.str("nonuniform_requested_radix", std::getenv("VKFFT_NONUNIFORM_RADIX"));
        result.str("nonuniform_requested_register_table", std::getenv("VKFFT_NONUNIFORM_TABLE"));
        result.str("nonuniform_requested_profile", std::getenv("VKFFT_NONUNIFORM_PROFILE"));
        result.str("nonuniform_requested_group", std::getenv("VKFFT_NONUNIFORM_GROUP"));'''
    text = once(text, old, new)
    text = once(text,
                '!expanded_candidate::request(request) || !expanded_candidate::valid(request, o.n) ||',
                '!expanded_candidate::request(request) || !expanded_candidate::valid(request, o.n, o.batch) ||')
    anchor = '        if (expanded_candidate::direct_enabled()) throw std::runtime_error("VKFFT_DIRECT_* is unsupported in expanded harness");'
    text = once(text, anchor, '''#ifdef VKFFT_STUDY_UNINSTRUMENTED
        if (expanded_candidate::enabled()) throw std::runtime_error("nonuniform requests require the generated planner binary");
#endif
        if (expanded_candidate::direct_enabled()) throw std::runtime_error("conflicting legacy candidate environment");''')
    text = once(text, 'axis.axisBlock[0] != (pfUINT)(o.n/request.regs)',
                'axis.axisBlock[0] != (pfUINT)(o.n/request.min_regs)')
    text = once(text,
                'int stat=0, dynlimit=0, devlimit=0, threadlimit=0;',
                'int stat=0, dynlimit=0, devlimit=0, threadlimit=0, physical_regs=0, local_bytes=0;')
    anchor = '            driver_check(cuFuncGetAttribute(&stat, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, axis.VkFFTKernel), "expanded static shared query");'
    text = once(text, anchor, '''            driver_check(cuFuncGetAttribute(&physical_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, axis.VkFFTKernel), "nonuniform physical register query");
            driver_check(cuFuncGetAttribute(&local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, axis.VkFFTKernel), "nonuniform local memory query");
''' + anchor)
    text = once(text,
                'if (dyn < 0 || stat < 0 || dyn > dynlimit || dyn + stat > devlimit || threads > threadlimit)',
                'if (physical_regs < 0 || local_bytes < 0 || dyn < 0 || stat < 0 || dyn > dynlimit || dyn + stat > devlimit || threads > threadlimit)')
    anchor = '            Object limits; limits.integer("dynamic_shared_bytes",dyn); limits.integer("static_shared_bytes",stat);'
    text = once(text, anchor, '''            Object limits; limits.integer("cuda_registers_per_thread",physical_regs); limits.integer("cuda_local_bytes",local_bytes);
            limits.integer("dynamic_shared_bytes",dyn); limits.integer("static_shared_bytes",stat);''')
    return text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty')
    # The delegated generator writes these siblings. Refuse overwriting any of
    # them; the base generator independently checks upstream/output isolation.
    for filename in ('expanded_benchmark.cpp', 'expanded_provenance.json', 'header_provenance.json'):
        if (output.parent / filename).exists():
            raise ValueError(f'output parent already contains {filename}')
    expanded.prepare(source, output)
    splitter = output / 'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h'
    plan = output / 'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h'
    benchmark = output.parent / 'expanded_benchmark.cpp'
    for path, transform in ((splitter, transform_splitter), (plan, transform_plan),
                            (benchmark, transform_benchmark)):
        original = path.read_text(encoding='utf8')
        path.write_text(transform(original), encoding='utf8')
    (output / 'candidate.hpp').write_bytes((HERE / 'candidate.hpp').read_bytes())
    provenance_path = output.parent / 'expanded_provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf8'))
    for rel in provenance['expanded']:
        provenance['expanded'][rel] = sha(output.parent / rel)
    provenance['nonuniform_v1'] = {
        'scope': 'proposed exact-order per-radix logical register table; single upload, boost 1',
        'status': 'generated only; no compile, scheduler, resource, numerical or device admission',
        'generator_sha256': sha(Path(__file__)),
        'candidate_sha256': sha(HERE / 'candidate.hpp'),
        'expanded_generator_sha256': sha(EXPANDED_GENERATOR),
        'research_contract_sha256': sha(HERE.parents[2] / 'experiments/research/nonuniform_register_admission_contract.json'),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf8')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == '__main__':
    main()
