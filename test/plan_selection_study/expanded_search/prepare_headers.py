#!/usr/bin/env python3
"""Create a separate header tree and harness for expanded single-upload requests."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
INSTRUMENTATION = HERE.parent / 'instrumentation'
spec = importlib.util.spec_from_file_location('vkfft_base_prepare_headers', INSTRUMENTATION/'prepare_headers.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one anchor, found {text.count(old)}: {old[:100]!r}')
    return text.replace(old, new, 1)


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty')
    provenance = base.prepare(source, output)
    scheduler = output/'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h'
    splitter = output/'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h'
    plan = output/'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h'
    text = scheduler.read_text(encoding='utf8')
    text = '#include "candidate.hpp"\n' + text
    text = once(text, '\t//optimize used radix kernels\n', '\t//optimize used radix kernels\n\tif (!expanded_candidate::enabled()) {\n')
    anchor = '\tfor (int i = 2; i < 33; i++) {\n\t\tint usedLocRegs = 0;'
    text = once(text, anchor, '\t}\n' + anchor)
    old = '\t\tres = VkFFTGetRegistersPerThread(app, (int)locAxisSplit[k], extraSharedMemoryForPow2, max_rhs / locAxisSplit[k], axes[k].specializationConstants.numRaderPrimes, loc_multipliers, registers_per_thread_per_radix, &registers_per_thread, &min_registers_per_thread, &isGoodSequence);'
    new = '''        if (expanded_candidate::enabled()) {
            if (axis_id != 0 || numPasses != 1 || k != 0 || registerBoost != 1 ||
                app->configuration.FFTdim != 1 || app->useBluesteinFFT[axis_id] ||
                axes[k].specializationConstants.numRaderPrimes || app->configuration.performR2C ||
                app->configuration.performDCT || app->configuration.performDST ||
                app->configuration.performConvolution || app->configuration.doublePrecision ||
                app->configuration.halfPrecision || !app->configuration.isInputFormatted ||
                locAxisSplit[k] != app->configuration.size[0] ||
                !expanded_candidate::seed(locAxisSplit[k], loc_multipliers, registers_per_thread_per_radix,
                                          registers_per_thread, min_registers_per_thread))
                return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
            isGoodSequence = 1;
        } else {
''' + old + '''
        }'''
    text = once(text, old, new)
    anchor = '\t\t//add more registers for Rader FFT if needed'
    text = once(text, anchor, '''        if (expanded_candidate::enabled() &&
            !expanded_candidate::reorder(axes[k].specializationConstants.stageRadix,
                                         axes[k].specializationConstants.numStages))
            return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;

''' + anchor)
    anchor = '\t\tif (switchRegisterBoost > 0) {'
    text = once(text, anchor, '''        if (expanded_candidate::enabled() &&
            !expanded_candidate::consistent((int)locAxisSplit[k], min_registers_per_thread, registers_per_thread,
                                            axes[k].specializationConstants.registers_per_thread_per_radix, registerBoost))
            return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
''' + anchor)
    text = once(text, '\t\t\tif (min_registers_per_thread != registers_per_thread) {',
                '\t\t\tif (!expanded_candidate::enabled() && min_registers_per_thread != registers_per_thread) {')
    scheduler.write_text(text, encoding='utf8')
    text = splitter.read_text(encoding='utf8')
    anchor = '\t\t\twhile ((axis->axisBlock[1] * (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.registerBoost)) > maxSequenceLengthSharedMemory) axis->axisBlock[1] /= 2;\n\t\t\taxis->groupedBatch = axis->axisBlock[1];'
    inject = '''
            if (expanded_candidate::enabled()) {
                expanded_candidate::Request request;
                if (!expanded_candidate::request(request) || axis_id != 0 || axis_upload_id != 0 ||
                    FFTPlan->numAxisUploads[0] != 1 || axis->specializationConstants.registerBoost != 1 ||
                    request.batch > (long long)FFTPlan->actualFFTSizePerAxis[0][1] ||
                    FFTPlan->actualFFTSizePerAxis[0][1] % request.batch != 0 ||
                    request.batch > (long long)app->configuration.maxComputeWorkGroupSize[1] ||
                    axis->axisBlock[0] != (pfUINT)(axis->specializationConstants.fftDim.data.i / request.regs) ||
                    request.batch > (long long)(maxThreadNum / axis->axisBlock[0]) ||
                    request.batch > (long long)(maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i))
                    return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
                axis->axisBlock[1] = (pfUINT)request.batch;
            }
'''
    text = once(text, anchor, anchor.split('\n')[0] + '\n' + inject + '\t\t\taxis->groupedBatch = axis->axisBlock[1];')
    splitter.write_text(text, encoding='utf8')
    text = plan.read_text(encoding='utf8')
    old = '\t\tvkfft_trace::invoke("axis_block_split", app, int(axis_id), int(axis_upload_id), [&]() { return VkFFTSplitAxisBlock(app, FFTPlan, axis, axis_id, axis_upload_id, allowedSharedMemory, allowedSharedMemoryPow2); });'
    new = '\t\tresFFT = vkfft_trace::invoke("axis_block_split", app, int(axis_id), int(axis_upload_id), [&]() { return VkFFTSplitAxisBlock(app, FFTPlan, axis, axis_id, axis_upload_id, allowedSharedMemory, allowedSharedMemoryPow2); });\n\t\tif (resFFT != VKFFT_SUCCESS) return resFFT;'
    text = once(text, old, new)
    anchor = '\t\taxis->specializationConstants.localSize[2].data.i = axis->axisBlock[2];'
    text = once(text, anchor, anchor + '''
        if (expanded_candidate::enabled() &&
            (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0] ||
             axis->axisBlock[1] > app->configuration.maxComputeWorkGroupSize[1] ||
             axis->axisBlock[2] > app->configuration.maxComputeWorkGroupSize[2]))
            return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
''')
    anchor = '\t\tresFFT = vkfft_trace::invoke("shader_codegen",'
    text = once(text, anchor, '\t\tvkfft_trace::capture_precodegen(app, axis);\n' + anchor)
    anchor = '\t\tif (vkfft_trace::describe_only) { vkfft_trace::capture_precompile(app, axis); resFFT = VKFFT_SUCCESS; }'
    text = once(text, anchor, '''        if (expanded_candidate::enabled()) {
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
                !expanded_candidate::layout_matches((int)s.fftDim.data.i, request.regs, request.batch, (int)s.axisSwapped,
                    s.localSize[0].data.i, s.localSize[1].data.i, s.localSize[2].data.i)) {
                deleteVkFFT(app); return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
            }
        }
''' + anchor)
    plan.write_text(text, encoding='utf8')
    (output/'candidate.hpp').write_bytes((HERE/'candidate.hpp').read_bytes())
    trace = (INSTRUMENTATION/'trace.hpp').read_text(encoding='utf8')
    trace = once(trace, 'inline std::vector<Kernel> kernels;',
                 'inline std::vector<Kernel> kernels;\ninline std::map<std::pair<int,int>,std::string> precodegen;')
    trace = once(trace, 'spans.clear(); stack.clear(); kernels.clear();',
                 'spans.clear(); stack.clear(); kernels.clear(); precodegen.clear();')
    anchor = 'template<class App,class Axis> void capture_precompile(App* app,Axis* axis) {'
    trace = once(trace, anchor, '''template<class App,class Axis> void capture_precodegen(App* app,Axis* axis) {
    auto key=std::make_pair(axis->specializationConstants.axis_id,axis->specializationConstants.axis_upload_id);
    if (precodegen.count(key)) throw std::runtime_error("duplicate pre-codegen snapshot");
    precodegen[key]=descriptor(app,axis,false);
}
''' + anchor)
    anchor = 'j.add("raw_descriptor",k.raw);'
    trace = once(trace, anchor, '''auto key=std::make_pair(k.axis,k.upload);auto pre=precodegen.find(key);
        if(pre==precodegen.end()) throw std::runtime_error("missing pre-codegen snapshot");
        j.add("precodegen_raw_descriptor",pre->second);''' + anchor)
    (output/'trace.hpp').write_text(trace,encoding='utf8')
    benchmark = '#include "candidate.hpp"\n' + (INSTRUMENTATION/'vkfft_instrumented_benchmark.cpp').read_text(encoding='utf8')
    anchor = '    result.str("study_mode", o.study_mode); result.str("candidate_id", o.candidate_id);'
    benchmark = once(benchmark, anchor, anchor + '''
    if (expanded_candidate::enabled()) {
        result.str("expanded_requested_radix", std::getenv("VKFFT_EXPANDED_RADIX"));
        result.str("expanded_requested_regs", std::getenv("VKFFT_EXPANDED_REGS"));
        result.str("expanded_requested_batch", std::getenv("VKFFT_EXPANDED_BATCH"));
    }
''')
    anchor = '        Options o = options(argc, argv); bool all_passed = true;'
    benchmark = once(benchmark, anchor, anchor + '''
        if (expanded_candidate::direct_enabled()) throw std::runtime_error("VKFFT_DIRECT_* is unsupported in expanded harness");
        if (expanded_candidate::enabled()) {
            expanded_candidate::Request request;
            if (!expanded_candidate::request(request) || !expanded_candidate::valid(request, o.n) ||
                o.batch % request.batch || (o.study_mode != "execute" && o.study_mode != "describe") || o.placement != "outofplace" || o.grid ||
                o.aim || o.boost || o.lut || o.nonpow2 || o.coalesced || o.lut4 || o.swap2 || o.swap3)
                throw std::runtime_error("expanded candidate requires valid request, execute or describe/outofplace, and default config");
        }
''')
    anchor = '        result.add("structural_features", structure(resources.app, compiled_diagnostics));'
    benchmark = once(benchmark, anchor, anchor + '''
        if (expanded_candidate::enabled()) {
            expanded_candidate::Request request; if (!expanded_candidate::request(request)) throw std::runtime_error("invalid expanded request");
            const auto& plan = *resources.app.localFFTPlan;
            if (plan.numAxisUploads[0] != 1) throw std::runtime_error("expanded upload mismatch");
            const auto& axis = plan.axes[0][0]; const auto& s = axis.specializationConstants;
            int stages[32]; for (int i=0; i<s.numStages && i<32; ++i) stages[i]=s.stageRadix[i];
            if (s.numStages < 1 || s.numStages > 32 || !expanded_candidate::reorder(stages, s.numStages) ||
                !std::equal(stages, stages+s.numStages, s.stageRadix) ||
                !expanded_candidate::consistent((int)o.n, s.min_registers_per_thread, s.registers_per_thread, s.registers_per_thread_per_radix, s.registerBoost) ||
                axis.groupedBatch != (pfUINT)request.batch ||
                axis.axisBlock[0] != (pfUINT)(o.n/request.regs) || axis.axisBlock[1] != (pfUINT)request.batch || axis.axisBlock[2] != 1)
                throw std::runtime_error("expanded post-initialize plan mismatch");
            if (!o.describe) {
            int stat=0, dynlimit=0, devlimit=0, threadlimit=0;
            driver_check(cuFuncGetAttribute(&stat, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, axis.VkFFTKernel), "expanded static shared query");
            driver_check(cuFuncGetAttribute(&dynlimit, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, axis.VkFFTKernel), "expanded dynamic shared query");
            driver_check(cuFuncGetAttribute(&threadlimit, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, axis.VkFFTKernel), "expanded thread limit query");
            driver_check(cuDeviceGetAttribute(&devlimit, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device), "expanded device shared query");
            const auto dyn=s.usedSharedMemory.data.i;
            const auto threads=s.localSize[0].data.i*s.localSize[1].data.i*s.localSize[2].data.i;
            if (dyn < 0 || stat < 0 || dyn > dynlimit || dyn + stat > devlimit || threads > threadlimit)
                throw std::runtime_error("expanded candidate exceeds compiled function or device resource limit");
            Object limits; limits.integer("dynamic_shared_bytes",dyn); limits.integer("static_shared_bytes",stat);
            limits.integer("function_dynamic_limit_bytes",dynlimit); limits.integer("device_shared_limit_bytes",devlimit);
            limits.integer("function_thread_limit",threadlimit); limits.integer("threads",threads);
            result.add("expanded_resource_validation",limits.finish());
            }
        }
''')
    anchor = '            restore(); append(); cuda_check(cudaStreamSynchronize(resources.stream), "synchronize one-forward validation");'
    benchmark = once(benchmark, anchor, '''            restore();
            cuda_check(cudaMemsetAsync(resources.buffer, 0xff, bytes, resources.stream), "poison single-forward output");
            append(); cuda_check(cudaStreamSynchronize(resources.stream), "synchronize one-forward validation");
            result.str("validation_output_initialization", "0xff_bytes_fp32_nan_before_single_and_final_timed_forward");''')
    anchor = '''        for (int i = 0; i < o.repeats; ++i) {
            restore();
            if (o.timing_mode == "graph") {'''
    benchmark = once(benchmark, anchor, '''        for (int i = 0; i < o.repeats; ++i) {
            restore();
            if (o.placement == "outofplace" && i == o.repeats - 1)
                cuda_check(cudaMemsetAsync(resources.buffer, 0xff, bytes, resources.stream), "poison final validation output");
            if (o.timing_mode == "graph") {''')
    (output.parent/'expanded_benchmark.cpp').write_text(benchmark, encoding='utf8')
    changed = (scheduler,splitter,plan,output/'candidate.hpp',output/'trace.hpp',output.parent/'expanded_benchmark.cpp')
    provenance['expanded'] = {str(p.relative_to(output.parent)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in changed}
    (output.parent/'expanded_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path); p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(); prepare(a.source, a.output)
