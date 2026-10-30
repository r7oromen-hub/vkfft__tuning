#!/usr/bin/env python3
"""Copy immutable upstream headers and add bounded tracing/describe hooks.

Writes only --output. Strict anchors fail on an unexpected upstream revision.
The original include tree is never edited. Generated files are disposable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one instrumentation anchor: {old[:100]!r}")
    return text.replace(old, new, 1)


def entry(text, function, lines):
    pattern = rf"(static inline [^\n]*\b{re.escape(function)}\([^\n]*\)\s*\{{)"
    updated, count = re.subn(pattern, lambda match: match[1] + "\n" + lines, text, count=1)
    if count != 1:
        raise RuntimeError(f"Missing function entry {function}")
    return updated


def calls(text, mapping, axis="-1", upload="-1"):
    for function, stage in mapping.items():
        pattern = rf"(?P<prefix>\b(?:resFFT|res|res_t|result|result2)\s*=\s*)(?P<call>{re.escape(function)}\s*\([^;]*?\))\s*;"
        text = re.sub(pattern, lambda match: match["prefix"] +
                      f'vkfft_trace::invoke("{stage}", app, int({axis}), int({upload}), [&]() {{ return ' + match["call"] + "; });", text)
    return text


def transform(relative, text):
    name = Path(relative).name
    if name == "vkFFT_InitializeApp.h":
        text = entry(text, "initializeVkFFT", 'VKFFT_TRACE_SCOPE("initialize", app, -1, -1);')
        text = entry(text, "setConfigurationVkFFT", 'VKFFT_TRACE_SCOPE("set_configuration", app, -1, -1);')
        text = calls(text, {"cuDeviceGetAttribute": "device_attribute_query", "cudaEventCreate": "stream_event_create",
                            "cudaMalloc": "temp_buffer_allocate", "VkFFTGeneratePhaseVectors": "phase_vectors_total"})
        anchor = "\tif (app->configuration.allocateTempBuffer && (app->configuration.tempBuffer == 0)) {"
        text = once(text, anchor, "\tif (vkfft_trace::describe_only) return VKFFT_SUCCESS; // restricted describe: no app-tail materialization\n" + anchor)
    elif name == "vkFFT_Scheduler.h":
        text = entry(text, "VkFFTScheduler", 'VKFFT_TRACE_SCOPE("scheduler", app, axis_id, -1);')
    elif name == "vkFFT_Plan_FFT.h":
        guard = '''VKFFT_TRACE_SCOPE("plan_axis", app, axis_id, axis_upload_id);
if (vkfft_trace::describe_only) {
    if (app->configuration.FFTdim != 1 || inverse || reverseBluesteinMultiUpload ||
        app->configuration.doublePrecision || app->configuration.halfPrecision ||
        app->configuration.performR2C || app->configuration.performDCT || app->configuration.performDST ||
        app->configuration.performConvolution || !app->configuration.isInputFormatted ||
        app->useBluesteinFFT[axis_id]) return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
    for (pfUINT vkfft_u=0; vkfft_u<FFTPlan->numAxisUploads[axis_id]; ++vkfft_u) {
        const auto& vkfft_s = FFTPlan->axes[axis_id][vkfft_u].specializationConstants;
        if (vkfft_s.useRader || vkfft_s.useBluesteinFFT) return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
    }
}
vkfft_trace::Scope vkfft_layout_scope("axis_layout_host_prepare", app, int(axis_id), int(axis_upload_id));'''
        text = entry(text, "VkFFTPlanAxis", guard)
        text = once(text, "\t\tresFFT = initParametersAPI(app, &axis->specializationConstants);",
                    "\t\tvkfft_layout_scope.finish();\n\t\tresFFT = initParametersAPI(app, &axis->specializationConstants);")
        text = calls(text, {"VkFFTGenerateRaderFFTKernel": "rader_prepare", "VkFFT_AllocateRaderUintLUT": "rader_uint_lut_prepare",
                            "VkFFTCheckUpdateBufferSet": "buffer_binding_prepare", "VkFFTUpdateBufferSet": "buffer_binding_prepare",
                            "initMemoryParametersAPI": "memory_parameters_init", "initParametersAPI": "parameters_init",
                            "shaderGen_FFT": "shader_codegen"}, "axis_id", "axis_upload_id")
        original = "\t\tresFFT = VkFFT_CompileKernel(app, axis);"
        replacement = "\t\tif (vkfft_trace::describe_only) { vkfft_trace::capture_precompile(app, axis); resFFT = VKFFT_SUCCESS; }\n\t\telse { resFFT = VkFFT_CompileKernel(app, axis); }"
        text = once(text, original, replacement)
        text = once(text, "\t\tVkFFTSplitAxisBlock(app, FFTPlan, axis, axis_id, axis_upload_id, allowedSharedMemory, allowedSharedMemoryPow2);",
                    '\t\tvkfft_trace::invoke("axis_block_split", app, int(axis_id), int(axis_upload_id), [&]() { return VkFFTSplitAxisBlock(app, FFTPlan, axis, axis_id, axis_upload_id, allowedSharedMemory, allowedSharedMemoryPow2); });')
    elif name == "vkFFT_CompileKernel.h":
        text = entry(text, "VkFFT_CompileKernel", 'VKFFT_TRACE_SCOPE("compile_kernel_total", app, axis->specializationConstants.axis_id, axis->specializationConstants.axis_upload_id);\nvkfft_trace::capture_precompile(app, axis);')
        text = once(text, "\t\tresult = nvrtcCompileProgram(prog,",
                    "\t\tvkfft_trace::capture_actual_options(axis, numOpts, opts);\n\t\tresult = nvrtcCompileProgram(prog,")
        text = calls(text, {"nvrtcCreateProgram": "nvrtc_program_create", "nvrtcCompileProgram": "nvrtc_compile",
                            "nvrtcGetCUBINSize": "binary_extract", "nvrtcGetCUBIN": "binary_extract",
                            "nvrtcGetPTXSize": "binary_extract", "nvrtcGetPTX": "binary_extract",
                            "nvrtcDestroyProgram": "nvrtc_program_destroy", "cuModuleLoadDataEx": "module_load",
                            "cuModuleGetFunction": "module_get_function", "cuFuncSetAttribute": "function_attribute_setup"},
                     "axis->specializationConstants.axis_id", "axis->specializationConstants.axis_upload_id")
    elif name == "vkFFT_ManageLUT.h":
        text = entry(text, "VkFFT_AllocateLUT", 'VKFFT_TRACE_SCOPE("lut_prepare", app, axis->specializationConstants.axis_id, axis->specializationConstants.axis_upload_id);')
        text = calls(text, {"cudaMalloc": "lut_device_allocate", "VkFFT_TransferDataFromCPU": "lut_upload"},
                     "axis->specializationConstants.axis_id", "axis->specializationConstants.axis_upload_id")
    elif name == "vkFFT_DispatchPlan.h":
        text = calls(text, {"cuLaunchKernel": "driver_kernel_launch"}, "axis->specializationConstants.axis_id", "axis->specializationConstants.axis_upload_id")
    return '#include "trace.hpp"\n' + text


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Generated include tree must be outside the upstream include directory")
    if not (source / "vkFFT.h").is_file():
        raise ValueError("--source must be the upstream include directory containing vkFFT.h")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output, dirs_exist_ok=True)
    names = {"vkFFT_InitializeApp.h", "vkFFT_Scheduler.h", "vkFFT_Plan_FFT.h",
             "vkFFT_CompileKernel.h", "vkFFT_ManageLUT.h", "vkFFT_DispatchPlan.h"}
    provenance = dict(upstream_commit="066a17c17068c0f11c9298d848c2976c71fad1c1",
                      source=str(source), output=str(output), files={})
    for path in source.rglob("*.h"):
        if path.name not in names:
            continue
        relative = path.relative_to(source)
        original = path.read_bytes()
        modified = transform(str(relative), original.decode("utf-8")).encode("utf-8")
        (output / relative).write_bytes(modified)
        provenance["files"][str(relative).replace("\\", "/")] = dict(
            original_sha256=hashlib.sha256(original).hexdigest(),
            generated_sha256=hashlib.sha256(modified).hexdigest())
    if len(provenance["files"]) != len(names):
        raise RuntimeError("Did not instrument exactly the expected six upstream headers")
    (output.parent / "header_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    compile_entry = next(value for key, value in provenance["files"].items() if key.endswith("vkFFT_CompileKernel.h"))
    (output / "vkfft_trace_source_metadata.hpp").write_text(
        '#pragma once\n#define VKFFT_ORIGINAL_COMPILE_HEADER_SHA "' + compile_entry["original_sha256"] + '"\n', encoding="utf-8")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.source, args.output)
    print(json.dumps(dict(instrumented_headers=len(result["files"]), output=result["output"])))
