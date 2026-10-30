// CPU-only synthetic shape: tests the generated trace instrumentation, not VkFFT planning.
#include "trace.hpp"
#include <stdexcept>
#include <string>

struct Value { struct { long long i = 0; } data; };
struct Constants {
    int axis_id = 0, axis_upload_id = 0, numAxisUploads = 1, precision = 0, complexSize = 8;
    int inverse = 0, actualInverse = 0, axisSwapped = 0, numStages = 1;
    int registers_per_thread = 4, min_registers_per_thread = 4, usedLocRegs = 0, registerBoost = 1;
    int sharedMemSize = 0, sharedMemSizePow2 = 0, LUT = 0, LUT_4step = 0, reorderFourStep = 0;
    int readToRegisters = 0, writeFromRegisters = 0, stridedSharedLayout = 0;
    int inputNumberByteSize = 4, outputNumberByteSize = 4, performR2C = 0, performDCT = 0, performDST = 0;
    int useRader = 0, useBluesteinFFT = 0;
    int stageRadix[1] = {4}, registers_per_thread_per_radix[33] = {};
    Value fftDim, fft_dim_full, localSize[3], usedSharedMemory, inputOffset, outputOffset, inputStride[3], outputStride[3];
    const char* code0 = "void VkFFT_main() {}";
};
struct Axis {
    Constants specializationConstants;
    unsigned long long axisBlock[4] = {32, 1, 1, 1};
    int groupedBatch = 1, bufferLUTSize = 0;
    const char* VkFFTFunctionName = "VkFFT_main";
};
struct Plan { int axisSplit[1][4] = {}, actualFFTSizePerAxis[1][4] = {}; };
struct Configuration {
    int FFTdim = 1, numberBatches = 1, isInputFormatted = 1, isOutputFormatted = 0;
    int computeCapabilityMajor = 8, computeCapabilityMinor = 0;
};
struct App { Configuration configuration; int actualNumBatches = 1; Plan* localFFTPlan; };

int main() {
    Plan plan;
    App app{}; app.localFFTPlan = &plan;
    Axis axis;
    vkfft_trace::reset(true);
    vkfft_trace::capture_precodegen(&app, &axis);
    bool duplicate_rejected = false;
    try { vkfft_trace::capture_precodegen(&app, &axis); }
    catch (const std::runtime_error&) { duplicate_rejected = true; }
    if (!duplicate_rejected) return 1;
    axis.specializationConstants.usedSharedMemory.data.i = 1024;
    vkfft_trace::capture_precompile(&app, &axis);
    auto snapshot = vkfft_trace::kernels_json();
    auto early = snapshot.find("\"precodegen_raw_descriptor\"");
    auto later = snapshot.find("\"raw_descriptor\"");
    auto early_value = snapshot.find("\"dynamic_shared_bytes\":0", early);
    auto later_value = snapshot.find("\"dynamic_shared_bytes\":1024", later);
    if (early == std::string::npos || later == std::string::npos || early_value >= later || later_value == std::string::npos) return 2;
    vkfft_trace::reset(true);
    if (vkfft_trace::kernels_json() != "[]") return 3;
    vkfft_trace::capture_precodegen(&app, &axis); // reset must also clear precodegen map
    return 0;
}
