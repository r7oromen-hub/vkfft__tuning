"""Add first-failure diagnostics to an isolated expanded_search header tree."""
from pathlib import Path


HERE = Path(__file__).resolve().parent


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one rejection anchor, found {text.count(old)}: {old[:100]!r}')
    return text.replace(old, new, 1)


def checks(items, *, cleanup=False, indent='            '):
    action = ('expanded_rejection::record("{}"); ' + ('deleteVkFFT(app); ' if cleanup else '') +
              'return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;')
    return '\n'.join(indent + 'if (' + condition + ') { ' + action.format(reason) + ' }'
                     for condition, reason in items)


def instrument(output):
    output = Path(output)
    scheduler = output/'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h'
    splitter = output/'vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h'
    plan = output/'vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h'
    benchmark = output.parent/'expanded_benchmark.cpp'
    text = scheduler.read_text(encoding='utf8')
    old = '''            if (axis_id != 0 || numPasses != 1 || k != 0 || registerBoost != 1 ||
                app->configuration.FFTdim != 1 || app->useBluesteinFFT[axis_id] ||
                axes[k].specializationConstants.numRaderPrimes || app->configuration.performR2C ||
                app->configuration.performDCT || app->configuration.performDST ||
                app->configuration.performConvolution || app->configuration.doublePrecision ||
                app->configuration.halfPrecision || !app->configuration.isInputFormatted ||
                locAxisSplit[k] != app->configuration.size[0] ||
                !expanded_candidate::seed(locAxisSplit[k], loc_multipliers, registers_per_thread_per_radix,
                                          registers_per_thread, min_registers_per_thread))
                return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;'''
    text = once(text, old, checks([
        ('axis_id != 0', 'scheduler_axis'), ('numPasses != 1', 'scheduler_pass_count'),
        ('k != 0', 'scheduler_pass_index'), ('registerBoost != 1', 'scheduler_register_boost'),
        ('app->configuration.FFTdim != 1', 'scheduler_fft_dimension'),
        ('app->useBluesteinFFT[axis_id]', 'scheduler_bluestein'),
        ('axes[k].specializationConstants.numRaderPrimes', 'scheduler_rader'),
        ('app->configuration.performR2C', 'scheduler_r2c'), ('app->configuration.performDCT', 'scheduler_dct'),
        ('app->configuration.performDST', 'scheduler_dst'),
        ('app->configuration.performConvolution', 'scheduler_convolution'),
        ('app->configuration.doublePrecision', 'scheduler_double_precision'),
        ('app->configuration.halfPrecision', 'scheduler_half_precision'),
        ('!app->configuration.isInputFormatted', 'scheduler_input_format'),
        ('locAxisSplit[k] != app->configuration.size[0]', 'scheduler_split_length'),
        ('!expanded_candidate::seed(locAxisSplit[k], loc_multipliers, registers_per_thread_per_radix,\n                                          registers_per_thread, min_registers_per_thread)', 'scheduler_invalid_seed'),
    ]))
    text = once(text, '''        if (expanded_candidate::enabled() &&
            !expanded_candidate::reorder(axes[k].specializationConstants.stageRadix,
                                         axes[k].specializationConstants.numStages))
            return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;''', '''        if (expanded_candidate::enabled() &&
            !expanded_candidate::reorder(axes[k].specializationConstants.stageRadix,
                                         axes[k].specializationConstants.numStages)) {
            expanded_rejection::record("scheduler_stage_multiset"); return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
        }''')
    text = once(text, '''        if (expanded_candidate::enabled() &&
            !expanded_candidate::consistent((int)locAxisSplit[k], min_registers_per_thread, registers_per_thread,
                                            axes[k].specializationConstants.registers_per_thread_per_radix, registerBoost))
            return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;''', '''        if (expanded_candidate::enabled() &&
            !expanded_candidate::consistent((int)locAxisSplit[k], min_registers_per_thread, registers_per_thread,
                                            axes[k].specializationConstants.registers_per_thread_per_radix, registerBoost)) {
            expanded_rejection::record("scheduler_register_table_changed"); return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
        }''')
    scheduler.write_text('#include "rejection.hpp"\n' + text, encoding='utf8')

    text = splitter.read_text(encoding='utf8')
    old = '''                if (!expanded_candidate::request(request) || axis_id != 0 || axis_upload_id != 0 ||
                    FFTPlan->numAxisUploads[0] != 1 || axis->specializationConstants.registerBoost != 1 ||
                    request.batch > (long long)FFTPlan->actualFFTSizePerAxis[0][1] ||
                    FFTPlan->actualFFTSizePerAxis[0][1] % request.batch != 0 ||
                    request.batch > (long long)app->configuration.maxComputeWorkGroupSize[1] ||
                    axis->axisBlock[0] != (pfUINT)(axis->specializationConstants.fftDim.data.i / request.regs) ||
                    request.batch > (long long)(maxThreadNum / axis->axisBlock[0]) ||
                    request.batch > (long long)(maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i))
                    return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;'''
    text = once(text, old, checks([
        ('!expanded_candidate::request(request)', 'splitter_invalid_request'),
        ('axis_id != 0', 'splitter_axis'), ('axis_upload_id != 0', 'splitter_upload'),
        ('FFTPlan->numAxisUploads[0] != 1', 'splitter_upload_count'),
        ('axis->specializationConstants.registerBoost != 1', 'splitter_register_boost'),
        ('request.batch > (long long)FFTPlan->actualFFTSizePerAxis[0][1]', 'splitter_group_exceeds_batch'),
        ('FFTPlan->actualFFTSizePerAxis[0][1] % request.batch != 0', 'splitter_group_not_divisor'),
        ('request.batch > (long long)app->configuration.maxComputeWorkGroupSize[1]', 'splitter_group_axis_limit'),
        ('axis->axisBlock[0] != (pfUINT)(axis->specializationConstants.fftDim.data.i / request.regs)', 'splitter_fft_threads'),
        ('axis->axisBlock[0] == 0 || axis->specializationConstants.fftDim.data.i == 0', 'splitter_zero_dimension'),
        ('request.batch > (long long)(maxThreadNum / axis->axisBlock[0])', 'splitter_thread_limit'),
        ('request.batch > (long long)(maxSequenceLengthSharedMemory / axis->specializationConstants.fftDim.data.i)', 'splitter_shared_memory_limit'),
    ], indent='                '))
    splitter.write_text('#include "rejection.hpp"\n' + text, encoding='utf8')

    text = plan.read_text(encoding='utf8')
    old = '''        if (expanded_candidate::enabled() &&
            (axis->axisBlock[0] > app->configuration.maxComputeWorkGroupSize[0] ||
             axis->axisBlock[1] > app->configuration.maxComputeWorkGroupSize[1] ||
             axis->axisBlock[2] > app->configuration.maxComputeWorkGroupSize[2]))
            return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;'''
    text = once(text, old, '        if (expanded_candidate::enabled()) {\n' + checks([
        (f'axis->axisBlock[{i}] > app->configuration.maxComputeWorkGroupSize[{i}]', f'plan_workgroup_dimension_{i}')
        for i in range(3)]) + '\n        }')
    old = '''            if (!expanded_candidate::request(request) || s.numStages < 1 || s.numStages > 32) {
                deleteVkFFT(app); return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
            }'''
    text = once(text, old, checks([
        ('!expanded_candidate::request(request)', 'plan_invalid_request'),
        ('s.numStages < 1 || s.numStages > 32', 'plan_stage_count'),
    ], cleanup=True))
    old = '''            if (!expanded_candidate::reorder(stages, s.numStages) ||
                !std::equal(stages, stages+s.numStages, s.stageRadix) ||
                !expanded_candidate::consistent((int)s.fftDim.data.i, s.min_registers_per_thread,
                                                s.registers_per_thread, s.registers_per_thread_per_radix, s.registerBoost) ||
                axis->groupedBatch != (pfUINT)request.batch ||
                !expanded_candidate::layout_matches((int)s.fftDim.data.i, request.regs, request.batch, (int)s.axisSwapped,
                    s.localSize[0].data.i, s.localSize[1].data.i, s.localSize[2].data.i)) {
                deleteVkFFT(app); return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
            }'''
    text = once(text, old, checks([
        ('!expanded_candidate::reorder(stages, s.numStages)', 'plan_stage_multiset'),
        ('!std::equal(stages, stages+s.numStages, s.stageRadix)', 'plan_stage_order'),
        ('!expanded_candidate::consistent((int)s.fftDim.data.i, s.min_registers_per_thread,\n                                                s.registers_per_thread, s.registers_per_thread_per_radix, s.registerBoost)', 'plan_register_table_changed'),
        ('axis->groupedBatch != (pfUINT)request.batch', 'plan_grouped_batch'),
        ('!expanded_candidate::layout_matches((int)s.fftDim.data.i, request.regs, request.batch, (int)s.axisSwapped,\n                    s.localSize[0].data.i, s.localSize[1].data.i, s.localSize[2].data.i)', 'plan_local_layout'),
    ], cleanup=True))
    plan.write_text('#include "rejection.hpp"\n' + text, encoding='utf8')

    text = benchmark.read_text(encoding='utf8')
    text = once(text, '    vkfft_trace::reset(o.describe); vkfft_trace::pause();',
                '    expanded_rejection::reset();\n    vkfft_trace::reset(o.describe); vkfft_trace::pause();')
    text = once(text, '    result.integer("vkfft_result_code", vkfft_code);', '''    if (expanded_rejection::reason) result.str("expanded_rejection_reason", expanded_rejection::reason);
    else result.add("expanded_rejection_reason", "null");
    result.integer("vkfft_result_code", vkfft_code);''')
    benchmark.write_text('#include "rejection.hpp"\n' + text, encoding='utf8')
    (output/'rejection.hpp').write_bytes((HERE/'rejection.hpp').read_bytes())
