#!/usr/bin/env python3
"""Generate isolated experimental headers from the existing instrumentation tree."""
import argparse, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'instrumentation'))
import prepare_headers as base


def replace_once(s, old, new):
    if s.count(old) != 1:
        raise RuntimeError(f'expected one anchor, got {s.count(old)}: {old[:100]!r}')
    return s.replace(old, new, 1)


def prepare(source, output):
    provenance = base.prepare(source, output)
    output = Path(output)
    scheduler = output / 'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h'
    split = output / 'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h'
    plan = output / 'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h'
    s = scheduler.read_text(encoding='utf8')
    anchor = '\n\t}\n\treturn VKFFT_SUCCESS;\n}\n#endif'
    inject = '''
        if (direct_candidate::enabled()) {
            if (axis_id != 0 || numPasses != 1 || app->configuration.FFTdim != 1 ||
                app->useBluesteinFFT[axis_id] || axes[k].specializationConstants.useRader ||
                axes[k].specializationConstants.numRaderPrimes || app->configuration.performR2C ||
                app->configuration.performDCT || app->configuration.performDST ||
                app->configuration.performConvolution || app->configuration.doublePrecision ||
                app->configuration.halfPrecision || !app->configuration.isInputFormatted ||
                app->configuration.size[0] < 2 ||
                (app->configuration.size[0] & (app->configuration.size[0] - 1)) ||
                !direct_candidate::reorder(axes[k].specializationConstants.stageRadix,
                                           axes[k].specializationConstants.numStages))
                return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
        }
'''
    s = '#include "direct_candidate.hpp"\n' + replace_once(s, anchor, inject + anchor)
    scheduler.write_text(s, encoding='utf8')
    s = split.read_text(encoding='utf8')
    anchor = '\t\t\twhile ((axis->axisBlock[1] * (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.registerBoost)) > maxSequenceLengthSharedMemory) axis->axisBlock[1] /= 2;\n\t\t\taxis->groupedBatch = axis->axisBlock[1];'
    extra = '''
            if (direct_candidate::enabled()) {
                const long long b = direct_candidate::batch();
                if (b <= 0 || axis_id != 0 || axis_upload_id != 0 || FFTPlan->numAxisUploads[0] != 1 ||
                    b > (long long)FFTPlan->actualFFTSizePerAxis[0][1] ||
                    FFTPlan->actualFFTSizePerAxis[0][1] % b != 0 ||
                    b > (long long)app->configuration.maxComputeWorkGroupSize[1] ||
                    b > (long long)(maxThreadNum / axis->axisBlock[0]) ||
                    b > (long long)(maxSequenceLengthSharedMemory / (axis->specializationConstants.fftDim.data.i / axis->specializationConstants.registerBoost)))
                    return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
                axis->axisBlock[1] = (pfUINT)b;
            }
'''
    s = replace_once(s, anchor, anchor.split('\n')[0] + '\n' + extra + '\t\t\taxis->groupedBatch = axis->axisBlock[1];')
    split.write_text(s, encoding='utf8')
    s = plan.read_text(encoding='utf8')
    old = '\t\tvkfft_trace::invoke("axis_block_split", app, int(axis_id), int(axis_upload_id), [&]() { return VkFFTSplitAxisBlock(app, FFTPlan, axis, axis_id, axis_upload_id, allowedSharedMemory, allowedSharedMemoryPow2); });'
    new = '\t\tresFFT = vkfft_trace::invoke("axis_block_split", app, int(axis_id), int(axis_upload_id), [&]() { return VkFFTSplitAxisBlock(app, FFTPlan, axis, axis_id, axis_upload_id, allowedSharedMemory, allowedSharedMemoryPow2); });\n\t\tif (resFFT != VKFFT_SUCCESS) return resFFT;'
    s = replace_once(s, old, new)
    anchor = '\t\taxis->specializationConstants.localSize[2].data.i = axis->axisBlock[2];'
    s = replace_once(s, anchor, anchor + '''
        if (direct_candidate::enabled() && !direct_candidate::match_block(
            axis->specializationConstants.localSize[0].data.i,
            axis->specializationConstants.localSize[1].data.i,
            axis->specializationConstants.localSize[2].data.i)) return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
        if (direct_candidate::enabled() &&
            (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0] ||
             axis->axisBlock[1] > app->configuration.maxComputeWorkGroupSize[1] ||
             axis->axisBlock[2] > app->configuration.maxComputeWorkGroupSize[2]))
            return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
''')
    plan.write_text(s, encoding='utf8')
    (output / 'direct_candidate.hpp').write_bytes((HERE / 'direct_candidate.hpp').read_bytes())
    provenance['experiment'] = {str(p.relative_to(output)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in (scheduler, split, plan, output / 'direct_candidate.hpp')}
    (output.parent / 'direct_candidate_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf8')
    benchmark = (HERE.parent / 'instrumentation/vkfft_instrumented_benchmark.cpp').read_text(encoding='utf8')
    benchmark = '#include "direct_candidate.hpp"\n' + benchmark
    anchor = '        result.add("structural_features", structure(resources.app, compiled_diagnostics));'
    benchmark = replace_once(benchmark, anchor, anchor + '''
        if (direct_candidate::enabled()) {
            const auto& plan = *resources.app.localFFTPlan;
            if (plan.numAxisUploads[0] != 1) throw std::runtime_error("direct candidate upload mismatch");
            const auto& s = plan.axes[0][0].specializationConstants;
            if (!direct_candidate::match_stages(s.stageRadix, s.numStages) ||
                !direct_candidate::match_block(s.localSize[0].data.i, s.localSize[1].data.i, s.localSize[2].data.i))
                throw std::runtime_error("direct candidate post-initialize mismatch");
            int static_shared = 0, function_dynamic_limit = 0, device_shared_limit = 0, function_thread_limit = 0;
            driver_check(cuFuncGetAttribute(&static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, plan.axes[0][0].VkFFTKernel), "direct static shared query");
            driver_check(cuFuncGetAttribute(&function_dynamic_limit, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, plan.axes[0][0].VkFFTKernel), "direct function dynamic shared limit query");
            driver_check(cuFuncGetAttribute(&function_thread_limit, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, plan.axes[0][0].VkFFTKernel), "direct function thread limit query");
            driver_check(cuDeviceGetAttribute(&device_shared_limit, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device), "direct device shared limit query");
            const auto dynamic_shared = s.usedSharedMemory.data.i;
            const auto threads = s.localSize[0].data.i * s.localSize[1].data.i * s.localSize[2].data.i;
            if (dynamic_shared < 0 || static_shared < 0 || dynamic_shared > function_dynamic_limit ||
                dynamic_shared + static_shared > device_shared_limit || threads > function_thread_limit)
                throw std::runtime_error("direct candidate exceeds compiled function or device resource limit");
            Object limits; limits.integer("dynamic_shared_bytes", dynamic_shared); limits.integer("static_shared_bytes", static_shared);
            limits.integer("function_dynamic_limit_bytes", function_dynamic_limit); limits.integer("device_shared_limit_bytes", device_shared_limit);
            limits.integer("function_thread_limit", function_thread_limit); limits.integer("threads", threads);
            result.add("direct_resource_validation", limits.finish());
        }
''')
    anchor = '    result.str("study_mode", o.study_mode); result.str("candidate_id", o.candidate_id);'
    benchmark = replace_once(benchmark, anchor, anchor + '''
    if (direct_candidate::enabled()) {
        result.str("direct_requested_radix", std::getenv("VKFFT_DIRECT_RADIX"));
        result.str("direct_requested_batch", std::getenv("VKFFT_DIRECT_BATCH"));
        result.str("direct_requested_block", std::getenv("VKFFT_DIRECT_BLOCK"));
    }
''')
    anchor = '        Options o = options(argc, argv); bool all_passed = true;'
    benchmark = replace_once(benchmark, anchor, anchor + '''
        if (direct_candidate::enabled()) {
            std::vector<long long> r, b;
            if (!direct_candidate::numbers(std::getenv("VKFFT_DIRECT_RADIX"), r) ||
                direct_candidate::batch() <= 0 ||
                !direct_candidate::numbers(std::getenv("VKFFT_DIRECT_BLOCK"), b) || b.size() != 3 ||
                o.study_mode != "execute" || o.placement != "outofplace" || o.grid ||
                o.coalesced || o.lut4 || o.swap2 || o.swap3)
                throw std::runtime_error("direct candidate requires all three fields, execute/outofplace and no coalesced/lut4/swap tuning");
        }
''')
    (output.parent / 'direct_benchmark.cpp').write_text(benchmark, encoding='utf8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
