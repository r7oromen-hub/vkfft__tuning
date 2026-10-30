// Standalone CUDA/NVRTC VkFFT plan benchmark. See harness_notes.md.
#include "cpu_reference.hpp"
#include <vkFFT.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <io.h>
#define DUP _dup
#define DUP2 _dup2
#define CLOSE _close
#define FILENO _fileno
#else
#include <unistd.h>
#define DUP dup
#define DUP2 dup2
#define CLOSE close
#define FILENO fileno
#endif

namespace {
using Clock = std::chrono::steady_clock;
std::string quote(const std::string& s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << char(c);
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (c < 32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else o << char(c);
    }
    o << '"'; return o.str();
}
std::string number(double x) {
    if (!std::isfinite(x)) return "null";
    std::ostringstream o; o << std::setprecision(17) << x; return o.str();
}
struct Object {
    std::ostringstream s; bool first = true;
    Object() { s << '{'; }
    void add(const std::string& key, const std::string& raw) {
        if (!first) s << ','; first = false; s << quote(key) << ':' << raw;
    }
    template<class T> void integer(const std::string& key, T x) { add(key, std::to_string(x)); }
    void str(const std::string& key, const std::string& value) { add(key, quote(value)); }
    std::string finish() { return s.str() + '}'; }
};
template<class T> std::string array(const T* data, std::size_t n) {
    std::ostringstream o; o << '[';
    for (std::size_t i = 0; i < n; ++i) { if (i) o << ','; o << number(double(data[i])); }
    o << ']'; return o.str();
}
void cuda_check(cudaError_t r, const char* what) {
    if (r != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(r));
}
void driver_check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* msg = nullptr; cuGetErrorString(r, &msg);
        throw std::runtime_error(std::string(what) + ": " + (msg ? msg : "unknown CUDA driver error"));
    }
}
// VkFFT prints some compilation/launch failures with printf. Keep stdout JSONL.
struct DiagnosticsToStderr {
    int saved = -1;
    DiagnosticsToStderr() {
        std::fflush(stdout); saved = DUP(FILENO(stdout));
        if (saved < 0 || DUP2(FILENO(stderr), FILENO(stdout)) < 0)
            throw std::runtime_error("Could not redirect VkFFT diagnostics to stderr");
    }
    ~DiagnosticsToStderr() { std::fflush(stdout); if (saved >= 0) { DUP2(saved, FILENO(stdout)); CLOSE(saved); } }
};
struct Options {
    std::size_t n = 4096, batch = 64;
    int warmup = 3, repeats = 15;
    uint64_t seed = 42;
    int aim = 0, boost = 0, lut = 0;
    bool grid = false;
    double atol = 0.002, rtol = 0.00005;
    double warmup_ms = 100;
    std::string timing_mode = "direct";
    std::string pattern = "random";
    std::string placement = "inplace";
    int graph_repeats = 1;
};
uint64_t parse_unsigned(const std::string& s) {
    if (s.empty() || s[0] == '-') throw std::runtime_error("Expected unsigned integer: " + s);
    std::size_t end = 0; auto value = std::stoull(s, &end);
    if (end != s.size()) throw std::runtime_error("Invalid integer: " + s);
    return value;
}
Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "vkfft_plan_benchmark --n N --batch B [--aim-threads A --register-boost R --use-lut {-1,0,1}]\n"
                         " [--warmup 3 --warmup-ms 100 --repeats 15 --seed 42 --atol .002 --rtol .00005 --grid]\n"
                         " [--timing-mode direct|graph]\n"
                         " [--pattern random|zero|impulse|constant|quarter-tone]\n"
                         " [--placement inplace|outofplace --graph-repeats 1]\n"
                         "Default candidate omits tuning flags. --grid runs default then 18 combinations.\n"
                         "CUDA device is always visible ordinal 0; caller must set CUDA_VISIBLE_DEVICES.\n";
            std::exit(0);
        }
        if (arg == "--grid") { o.grid = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
        std::string value = argv[++i];
        if (arg == "--n") o.n = parse_unsigned(value);
        else if (arg == "--batch") o.batch = parse_unsigned(value);
        else if (arg == "--seed") o.seed = parse_unsigned(value);
        else if (arg == "--warmup") o.warmup = int(parse_unsigned(value));
        else if (arg == "--repeats") o.repeats = int(parse_unsigned(value));
        else if (arg == "--aim-threads") o.aim = int(parse_unsigned(value));
        else if (arg == "--register-boost") o.boost = int(parse_unsigned(value));
        else if (arg == "--use-lut") { std::size_t end = 0; o.lut = std::stoi(value, &end); if (end != value.size()) throw std::runtime_error("Invalid LUT value"); }
        else if (arg == "--atol") o.atol = std::stod(value);
        else if (arg == "--rtol") o.rtol = std::stod(value);
        else if (arg == "--warmup-ms") o.warmup_ms = std::stod(value);
        else if (arg == "--timing-mode") o.timing_mode = value;
        else if (arg == "--pattern") o.pattern = value;
        else if (arg == "--placement") o.placement = value;
        else if (arg == "--graph-repeats") o.graph_repeats = int(parse_unsigned(value));
        else throw std::runtime_error("Unknown argument " + arg);
    }
    if (!motivation::smooth235(o.n) || !o.batch) throw std::runtime_error("N must be positive and 2/3/5-smooth; batch must be positive");
    if (o.repeats < 1 || o.repeats > 10000 || o.warmup < 0 || o.warmup > 10000)
        throw std::runtime_error("repeats must be 1..10000 and warmup 0..10000");
    if (o.aim < 0 || o.boost < 0 || o.lut < -1 || o.lut > 1 ||
        !std::isfinite(o.atol) || !std::isfinite(o.rtol) || o.atol < 0 || o.rtol < 0 ||
        !std::isfinite(o.warmup_ms) || o.warmup_ms < 0 || o.warmup_ms > 60000)
        throw std::runtime_error("Invalid tuning knob or correctness threshold");
    if (o.n > std::numeric_limits<std::size_t>::max() / o.batch / sizeof(float2))
        throw std::runtime_error("Workload byte count overflows size_t");
    if (o.timing_mode != "direct" && o.timing_mode != "graph")
        throw std::runtime_error("timing-mode must be direct or graph");
    if (o.pattern != "random" && o.pattern != "zero" && o.pattern != "impulse" &&
        o.pattern != "constant" && o.pattern != "quarter-tone") throw std::runtime_error("Unknown input pattern");
    if (o.pattern == "quarter-tone" && o.n % 4) throw std::runtime_error("quarter-tone requires N divisible by 4");
    if (o.placement != "inplace" && o.placement != "outofplace") throw std::runtime_error("placement must be inplace or outofplace");
    if (o.graph_repeats < 1 || o.graph_repeats > 2048) throw std::runtime_error("graph-repeats must be 1..2048");
    if (o.graph_repeats > 1 && (o.placement != "outofplace" || o.timing_mode != "graph"))
        throw std::runtime_error("graph-repeats>1 requires outofplace placement and graph timing");
    return o;
}
std::string requested(const Options& o) {
    Object j;
    if (o.aim) j.integer("aimThreads", o.aim);
    if (o.boost) j.integer("registerBoost", o.boost);
    if (o.lut) j.integer("useLUT", o.lut);
    j.integer("makeForwardPlanOnly", 1);
    return j.finish();
}
std::string resolved(const VkFFTConfiguration& c) {
    Object j;
    j.integer("aimThreads", c.aimThreads); j.integer("registerBoost", c.registerBoost);
    j.integer("registerBoostNonPow2", c.registerBoostNonPow2); j.integer("registerBoost4Step", c.registerBoost4Step);
    j.integer("useLUT", c.useLUT); j.integer("useLUT_4step", c.useLUT_4step);
    j.integer("numberBatches", c.numberBatches); j.integer("reorderFourStep", c.reorderFourStep);
    j.integer("sharedMemorySize", c.sharedMemorySize); j.integer("sharedMemorySizeStatic", c.sharedMemorySizeStatic);
    j.integer("maxThreadsNum", c.maxThreadsNum); j.integer("warpSize", c.warpSize);
    j.integer("makeForwardPlanOnly", c.makeForwardPlanOnly); j.integer("num_streams", c.num_streams);
    j.integer("keepShaderCode", c.keepShaderCode);
    j.integer("isInputFormatted", c.isInputFormatted); j.integer("isOutputFormatted", c.isOutputFormatted);
    return j.finish();
}
std::string structure(const VkFFTApplication& app, std::string& compiled_diagnostics) {
    if (!app.localFFTPlan) throw std::runtime_error("Forward plan is missing");
    const auto& plan = *app.localFFTPlan;
    Object j; j.integer("numAxisUploads", plan.numAxisUploads[0]);
    j.add("axisSplit", array(plan.axisSplit[0], 4));
    j.add("actualFFTSizePerAxis", array(plan.actualFFTSizePerAxis[0], app.configuration.FFTdim));
    std::ostringstream kernels, diagnostics; kernels << '['; diagnostics << '[';
    for (std::size_t u = 0; u < plan.numAxisUploads[0]; ++u) {
        if (u) kernels << ',';
        const auto& axis = plan.axes[0][u]; const auto& s = axis.specializationConstants;
        Object k; k.integer("upload", u); k.str("function", axis.VkFFTFunctionName);
        pfINT local[3] = {s.localSize[0].data.i, s.localSize[1].data.i, s.localSize[2].data.i};
        k.add("localSize", array(local, 3)); k.add("axisBlock", array(axis.axisBlock, 4));
        k.integer("groupedBatch", axis.groupedBatch); k.integer("fftDim", s.fftDim.data.i);
        k.integer("numStages", s.numStages); k.add("stageRadix", array(s.stageRadix, std::min(33, s.numStages)));
        k.integer("logical_registers_per_thread", s.registers_per_thread);
        k.integer("min_logical_registers_per_thread", s.min_registers_per_thread);
        k.integer("registerBoost", s.registerBoost); k.integer("usedLocRegs", s.usedLocRegs);
        k.integer("dynamic_shared_bytes", s.usedSharedMemory.data.i);
        k.integer("sharedMemSize", s.sharedMemSize); k.integer("LUT", s.LUT);
        k.integer("reorderFourStep", s.reorderFourStep); k.integer("readToRegisters", s.readToRegisters);
        k.integer("writeFromRegisters", s.writeFromRegisters);
        Object d; d.integer("upload", u);
        for (const auto& attr : std::vector<std::pair<std::string, CUfunction_attribute>>{
            {"cuda_num_regs", CU_FUNC_ATTRIBUTE_NUM_REGS},
            {"cuda_static_shared_bytes", CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES},
            {"cuda_local_bytes", CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES},
            {"cuda_max_threads_per_block", CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK}}) {
            int value = 0; auto rc = cuFuncGetAttribute(&value, attr.second, axis.VkFFTKernel);
            if (rc == CUDA_SUCCESS) d.integer(attr.first, value); else d.add(attr.first, "null");
        }
        if (u) diagnostics << ',';
        diagnostics << d.finish();
        kernels << k.finish();
    }
    diagnostics << ']'; compiled_diagnostics = diagnostics.str();
    kernels << ']'; j.add("kernels", kernels.str());
    // Single-axis forward VkFFT dispatches uploads in descending upload index.
    std::vector<int> dispatch;
    for (int i = int(plan.numAxisUploads[0]) - 1; i >= 0; --i) dispatch.push_back(i);
    j.add("execution_upload_order", array(dispatch.data(), dispatch.size()));
    return j.finish();
}
struct Resources {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    void* buffer = nullptr;
    void* input_buffer = nullptr;
    VkFFTApplication app = {};
    bool initialized = false;
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (initialized) deleteVkFFT(&app);
        if (buffer) cudaFree(buffer);
        if (input_buffer) cudaFree(input_buffer);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (stream) cudaStreamDestroy(stream);
    }
};
std::string gpu_metadata(CUdevice device) {
    cudaDeviceProp prop{}; cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    Object j; j.str("name", prop.name); j.integer("ordinal", 0);
    j.str("compute_capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
    j.integer("multiprocessors", prop.multiProcessorCount); j.integer("total_memory_bytes", prop.totalGlobalMem);
    j.integer("l2_cache_bytes", prop.l2CacheSize);
    std::size_t free_bytes = 0, total_bytes = 0; cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    j.integer("free_memory_bytes", free_bytes);
    char pci[64] = {}; cuda_check(cudaDeviceGetPCIBusId(pci, sizeof(pci), 0), "cudaDeviceGetPCIBusId"); j.str("pci_bus_id", pci);
    CUuuid uuid{}; driver_check(cuDeviceGetUuid(&uuid, device), "cuDeviceGetUuid");
    std::ostringstream hex; hex << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) { if (i == 4 || i == 6 || i == 8 || i == 10) hex << '-'; hex << std::setw(2) << unsigned(static_cast<unsigned char>(uuid.bytes[i])); }
    j.str("uuid", hex.str());
    const char* visible = std::getenv("CUDA_VISIBLE_DEVICES"); j.str("cuda_visible_devices", visible ? visible : "unset");
    return j.finish();
}
std::string correctness(const Options& o, const std::vector<float2>& actual,
                        const std::vector<motivation::Complex>& ref,
                        const std::vector<motivation::Complex>& base, bool& passed) {
    double max_abs = 0, max_rel = 0, sum_error2 = 0, sum_ref2 = 0, max_ref = 0;
    std::size_t nonfinite = 0, failed = 0;
    constexpr double l2_threshold = motivation::l2_error_threshold, linf_threshold = motivation::linf_error_threshold;
    double worst_l2 = 0, worst_linf = 0;
    std::size_t failing_batches = 0, worst_l2_batch = 0, worst_linf_batch = 0, zero_input_batches = 0;
    struct Violation {
        std::size_t batch, bin;
        motivation::Complex expected;
        float2 actual;
        double error, tolerance, ratio;
    };
    constexpr std::size_t details_limit = 5;
    std::vector<Violation> first_violations, worst_violations;
    auto record_violation = [&](std::size_t b, std::size_t k, motivation::Complex expected,
                                float2 value, double error, double tolerance) {
        const double ratio = tolerance > 0 ? error / tolerance : std::numeric_limits<double>::infinity();
        const Violation v{b, k, expected, value, error, tolerance, ratio};
        if (first_violations.size() < details_limit) first_violations.push_back(v);
        worst_violations.push_back(v);
        std::stable_sort(worst_violations.begin(), worst_violations.end(),
                         [](const Violation& a, const Violation& b) { return a.ratio > b.ratio; });
        if (worst_violations.size() > details_limit) worst_violations.resize(details_limit);
    };
    for (std::size_t b = 0; b < o.batch; ++b) {
      double batch_error2 = 0, batch_input2 = 0, batch_max_abs = 0;
      bool batch_finite = true;
      for (std::size_t k = 0; k < o.n; ++k) {
        const auto expected = motivation::reference_value(ref[k], k, b, o.n, o.pattern);
        batch_input2 += std::norm(motivation::input_value(base[k], b, o.pattern));
        const auto value = actual[b * o.n + k];
        const double magnitude = std::abs(expected); max_ref = std::max(max_ref, magnitude);
        sum_ref2 += std::norm(expected);
        const double tolerance = o.atol + o.rtol * magnitude;
        if (!std::isfinite(value.x) || !std::isfinite(value.y)) {
            ++nonfinite; ++failed;
            batch_finite = false;
            record_violation(b, k, expected, value, std::numeric_limits<double>::infinity(), tolerance);
            continue;
        }
        const double error = std::abs(motivation::Complex(value.x, value.y) - expected);
        batch_error2 += error * error; batch_max_abs = std::max(batch_max_abs, error);
        max_abs = std::max(max_abs, error); sum_error2 += error * error;
        max_rel = std::max(max_rel, error / std::max(1e-30, magnitude));
        if (error > tolerance) { ++failed; record_violation(b, k, expected, value, error, tolerance); }
      }
      const bool zero_input = batch_input2 == 0;
      if (zero_input) ++zero_input_batches;
      const auto metrics = motivation::evaluate_normwise(o.n, batch_input2, batch_error2, batch_max_abs, batch_finite);
      const double l2 = metrics.l2, linf = metrics.linf;
      if (l2 > worst_l2) { worst_l2 = l2; worst_l2_batch = b; }
      if (linf > worst_linf) { worst_linf = linf; worst_linf_batch = b; }
      if (!metrics.passed) ++failing_batches;
    }
    passed = failing_batches == 0;
    Object j; j.add("passed", passed ? "true" : "false");
    j.str("criterion", "criterion_v2");
    j.str("criterion_provenance", "pilot_informed_scale_invariant_FP32_engineering_thresholds_not_a_theoretical_error_bound");
    j.str("reference", o.pattern == "random" ? "independent_double_mixed_radix_235_fft_plus_exact_dyadic_batch_linearity" : "analytic_pattern_reference_with_independent_CPU_FFT_crosscheck");
    j.add("pointwise_passed", failed == 0 ? "true" : "false");
    j.integer("failing_batches", failing_batches); j.integer("zero_input_batches", zero_input_batches);
    j.add("worst_batch_l2_scaled_error", number(worst_l2)); j.integer("worst_l2_batch", worst_l2_batch);
    j.add("worst_batch_linf_input_scaled_error", number(worst_linf)); j.integer("worst_linf_batch", worst_linf_batch);
    j.add("l2_scaled_error_threshold", number(l2_threshold)); j.add("linf_input_scaled_error_threshold", number(linf_threshold));
    j.integer("checked_elements", actual.size()); j.integer("nonfinite_count", nonfinite); j.integer("failed_elements", failed);
    j.add("max_absolute_error", number(max_abs)); j.add("max_relative_error", number(max_rel));
    j.add("l2_relative_error", nonfinite ? "null" : number(std::sqrt(sum_error2 / std::max(1e-300, sum_ref2))));
    j.add("normalized_max_absolute_error", number(max_abs / std::max(1e-30, max_ref)));
    j.add("atol", number(o.atol)); j.add("rtol", number(o.rtol));
    auto encode_violations = [&](const std::vector<Violation>& violations) {
        std::ostringstream output; output << '[';
        for (std::size_t i = 0; i < violations.size(); ++i) {
            if (i) output << ',';
            const auto& v = violations[i]; Object d;
            d.integer("batch", v.batch); d.integer("bin", v.bin); d.integer("flat_index", v.batch * o.n + v.bin);
            d.add("expected_real", number(v.expected.real())); d.add("expected_imag", number(v.expected.imag()));
            d.add("actual_real", number(v.actual.x)); d.add("actual_imag", number(v.actual.y));
            d.add("reference_magnitude", number(std::abs(v.expected))); d.add("absolute_error", number(v.error));
            d.add("allowed_error", number(v.tolerance)); d.add("error_to_allowance_ratio", number(v.ratio));
            output << d.finish();
        }
        output << ']'; return output.str();
    };
    j.integer("violation_details_limit", details_limit);
    j.add("first_violations", encode_violations(first_violations));
    j.add("worst_violations", encode_violations(worst_violations));
    j.str("pointwise_diagnostic_rule", "all_complex_abs_errors <= atol + rtol * abs(reference), and no nonfinite outputs");
    j.str("pass_rule", "every_batch_finite_and_L2(error)/(sqrt(N)*L2(input))<=1e-5_and_Linf(error)/L2(input)<=1e-4;_zero_input_requires_exact_zero_output");
    return j.finish();
}
std::string run(const Options& o, bool& passed) {
    Object result; result.integer("schema_version", 1); result.integer("n", o.n); result.integer("batch", o.batch);
    result.add("requested_config", requested(o)); result.integer("seed", o.seed);
    result.integer("warmup", o.warmup); result.integer("repeats", o.repeats);
    result.add("warmup_ms_requested", number(o.warmup_ms));
    result.str("backend", "CUDA_NVRTC"); result.integer("vkfft_version", VkFFTGetVersion());
    result.str("upstream_commit", "066a17c"); result.str("precision", "complex_fp32"); result.integer("forward_direction", -1);
    result.str("timing_scope", o.placement == "outofplace" ?
               "full_batch_forward_immutable_input_separate_output_same_stream_excluding_input_restore" :
               "one_full_batch_in_place_forward_same_stream_excluding_input_restore");
    result.str("placement", o.placement); result.integer("graph_repeats", o.graph_repeats);
    result.str("exec_ms_samples_unit", "milliseconds_per_full_batch_forward_raw_interval_divided_by_graph_repeats");
    result.str("timing_mode", o.timing_mode);
    result.str("timing_metric", o.timing_mode == "graph" ?
               "cuda_graph_forward_device_latency_between_external_event_nodes" :
               "device_timeline_forward_latency_including_possible_inter_kernel_submission_gaps");
    result.str("input_cache_regime", o.placement == "outofplace" && o.timing_mode == "graph" ?
               "H2D_restore_before_each_sample_repeated_graph_reads_immutable_input_without_cache_flush" :
               "H2D_restore_before_every_forward_no_cache_flush_reset_induced_cache_state");
    result.str("pattern", o.pattern);
    result.str("input", o.pattern == "random" ? "splitmix64_dyadic_complex_noise_with_exact_batch_scale_and_DC_offset" : "analytic_pattern_with_exact_dyadic_batch_scale_no_DC_offset");
    std::string phase = "setup";
    double initialize_ms = std::numeric_limits<double>::quiet_NaN();
    int vkfft_code = 0;
    std::vector<float> samples;
    std::vector<float> raw_samples;
    passed = false;
    try {
        DiagnosticsToStderr redirect;
        driver_check(cuInit(0), "cuInit"); cuda_check(cudaSetDevice(0), "cudaSetDevice(0)");
        cuda_check(cudaFree(nullptr), "initialize CUDA runtime primary context");
        CUdevice device{}; driver_check(cuDeviceGet(&device, 0), "cuDeviceGet(0)");
        result.add("gpu", gpu_metadata(device));
        int runtime = 0, driver = 0, rtc_major = 0, rtc_minor = 0;
        cuda_check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion"); cuda_check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        nvrtcVersion(&rtc_major, &rtc_minor);
        result.integer("cuda_runtime_version", runtime); result.integer("cuda_driver_version", driver);
        result.str("nvrtc_version", std::to_string(rtc_major) + "." + std::to_string(rtc_minor));
        const auto base = motivation::base_input(o.n, o.seed, o.pattern);
        auto ref = motivation::reference_fft(base);
        if (o.pattern != "random") {
            auto analytic = motivation::analytic_reference(o.n, o.pattern);
            double error2 = 0, norm2 = 0;
            for (std::size_t k = 0; k < o.n; ++k) { error2 += std::norm(ref[k] - analytic[k]); norm2 += std::norm(analytic[k]); }
            const double relative = norm2 > 0 ? std::sqrt(error2 / norm2) : std::sqrt(error2);
            result.add("reference_cpu_analytic_l2_relative_error", number(relative));
            if (!std::isfinite(relative) || relative > 1e-11) throw std::runtime_error("CPU FFT disagrees with analytic input-pattern reference");
            ref = std::move(analytic);
        }
        const std::size_t count = o.n * o.batch, bytes = count * sizeof(float2);
        result.integer("buffer_bytes", bytes);
        std::vector<float2> host(count);
        for (std::size_t b = 0; b < o.batch; ++b) for (std::size_t k = 0; k < o.n; ++k) {
            const auto v = motivation::input_value(base[k], b, o.pattern);
            host[b * o.n + k] = make_float2(float(v.real()), float(v.imag()));
        }
        Resources resources;
        cuda_check(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        cuda_check(cudaEventCreate(&resources.start), "cudaEventCreate(start)"); cuda_check(cudaEventCreate(&resources.stop), "cudaEventCreate(stop)");
        cuda_check(cudaMalloc(&resources.buffer, bytes), "cudaMalloc");
        if (o.placement == "outofplace") cuda_check(cudaMalloc(&resources.input_buffer, bytes), "cudaMalloc immutable input");
        result.integer("allocated_primary_buffers_bytes", bytes * (o.placement == "outofplace" ? 2 : 1));
        pfUINT buffer_bytes = bytes;
        VkFFTConfiguration config{}; config.FFTdim = 1; config.size[0] = o.n; config.numberBatches = o.batch;
        config.device = &device; config.stream = &resources.stream; config.num_streams = 1;
        config.buffer = &resources.buffer; config.bufferSize = &buffer_bytes; config.makeForwardPlanOnly = 1;
        if (o.placement == "outofplace") {
            config.isInputFormatted = 1; config.inputBuffer = &resources.input_buffer; config.inputBufferSize = &buffer_bytes;
        }
        config.aimThreads = o.aim; config.registerBoost = o.boost; config.useLUT = o.lut;
        config.keepShaderCode = 0; config.printMemoryLayout = 0;
        phase = "initialize";
        const auto begin = Clock::now();
        const VkFFTResult init_result = initializeVkFFT(&resources.app, config);
        resources.initialized = init_result == VKFFT_SUCCESS;
        const cudaError_t init_sync = resources.initialized ? cudaStreamSynchronize(resources.stream) : cudaSuccess;
        initialize_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        vkfft_code = int(init_result);
        if (init_result != VKFFT_SUCCESS) throw std::runtime_error("initializeVkFFT failed with code " + std::to_string(vkfft_code));
        cuda_check(init_sync, "synchronize initialization setup");
        std::string compiled_diagnostics;
        result.add("resolved_config", resolved(resources.app.configuration));
        result.add("structural_features", structure(resources.app, compiled_diagnostics));
        result.add("compiled_diagnostics", compiled_diagnostics);
        VkFFTLaunchParams launch{}; launch.buffer = &resources.buffer;
        if (o.placement == "outofplace") launch.inputBuffer = &resources.input_buffer;
        auto restore = [&]() {
            void* input = o.placement == "outofplace" ? resources.input_buffer : resources.buffer;
            cuda_check(cudaMemcpyAsync(input, host.data(), bytes, cudaMemcpyHostToDevice, resources.stream), "restore input H2D");
            // Wait for reset completion so the event interval cannot contain H2D.
            cuda_check(cudaStreamSynchronize(resources.stream), "synchronize input restore");
        };
        auto append = [&]() {
            auto r = VkFFTAppend(&resources.app, -1, &launch); vkfft_code = int(r);
            if (r != VKFFT_SUCCESS) throw std::runtime_error("VkFFTAppend failed with code " + std::to_string(vkfft_code));
        };
        auto input_unchanged = [&]() {
            std::vector<float2> check(count);
            cuda_check(cudaMemcpyAsync(check.data(), resources.input_buffer, bytes, cudaMemcpyDeviceToHost, resources.stream), "read immutable input");
            cuda_check(cudaStreamSynchronize(resources.stream), "synchronize immutable input readback");
            return std::memcmp(check.data(), host.data(), bytes) == 0;
        };
        if (o.placement == "outofplace") {
            phase = "single_forward_validation";
            restore(); append(); cuda_check(cudaStreamSynchronize(resources.stream), "synchronize one-forward validation");
            const bool unchanged = input_unchanged(); result.add("single_forward_input_unchanged", unchanged ? "true" : "false");
            std::vector<float2> one_output(count);
            cuda_check(cudaMemcpyAsync(one_output.data(), resources.buffer, bytes, cudaMemcpyDeviceToHost, resources.stream), "read one-forward validation output");
            cuda_check(cudaStreamSynchronize(resources.stream), "synchronize one-forward output readback");
            bool one_passed = false; result.add("single_forward_correctness", correctness(o, one_output, ref, base, one_passed));
            if (!unchanged || !one_passed) throw std::runtime_error("Single out-of-place forward failed immutable-input or numerical validation");
        }
        if (o.timing_mode == "graph") {
            phase = "graph_build";
            const auto graph_begin = Clock::now();
            cuda_check(cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeThreadLocal), "begin graph capture");
            try {
                // External records produce explicit graph event-record nodes,
                // rather than only capture-internal synchronization dependencies.
                cuda_check(cudaEventRecordWithFlags(resources.start, resources.stream, cudaEventRecordExternal), "capture external start event");
                for (int i = 0; i < o.graph_repeats; ++i) append();
                cuda_check(cudaEventRecordWithFlags(resources.stop, resources.stream, cudaEventRecordExternal), "capture external stop event");
                cuda_check(cudaStreamEndCapture(resources.stream, &resources.graph), "end graph capture");
            } catch (...) {
                cudaGraph_t abandoned = nullptr;
                cudaStreamEndCapture(resources.stream, &abandoned);
                if (abandoned) cudaGraphDestroy(abandoned);
                throw;
            }
            cuda_check(cudaGraphInstantiateWithFlags(&resources.graph_exec, resources.graph, 0), "instantiate graph");
            const double graph_build_ms = std::chrono::duration<double, std::milli>(Clock::now() - graph_begin).count();
            std::size_t node_count = 0;
            cuda_check(cudaGraphGetNodes(resources.graph, nullptr, &node_count), "query graph node count");
            std::vector<cudaGraphNode_t> nodes(node_count);
            cuda_check(cudaGraphGetNodes(resources.graph, nodes.data(), &node_count), "query graph nodes");
            std::size_t kernel_count = 0, event_record_count = 0;
            for (auto node : nodes) {
                cudaGraphNodeType type{}; cuda_check(cudaGraphNodeGetType(node, &type), "query graph node type");
                if (type == cudaGraphNodeTypeKernel) ++kernel_count;
                if (type == cudaGraphNodeTypeEventRecord) ++event_record_count;
            }
            result.add("graph_build_ms", number(graph_build_ms));
            result.integer("graph_node_count", node_count); result.integer("graph_kernel_count", kernel_count);
            result.integer("graph_event_record_count", event_record_count);
            const std::size_t expected_kernel_count = std::size_t(o.graph_repeats) * resources.app.localFFTPlan->numAxisUploads[0];
            result.integer("graph_expected_kernel_count", expected_kernel_count);
            if (kernel_count != expected_kernel_count || event_record_count != 2)
                throw std::runtime_error("Captured graph kernel count must equal graph_repeats * numAxisUploads and contain exactly two external event records");
        }
        auto execute = [&]() {
            if (o.timing_mode == "graph") cuda_check(cudaGraphLaunch(resources.graph_exec, resources.stream), "launch graph");
            else append();
        };
        phase = "warmup";
        const auto warmup_begin = Clock::now();
        int warmup_actual = 0;
        double warmup_wall_ms = 0;
        while (warmup_actual < o.warmup || warmup_wall_ms < o.warmup_ms) {
            restore(); execute(); cuda_check(cudaStreamSynchronize(resources.stream), "synchronize warmup");
            ++warmup_actual;
            warmup_wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - warmup_begin).count();
        }
        result.integer("warmup_actual", warmup_actual); result.add("warmup_wall_ms", number(warmup_wall_ms));
        result.integer("warmup_forward_count", uint64_t(warmup_actual) * uint64_t(o.graph_repeats));
        phase = "execution";
        for (int i = 0; i < o.repeats; ++i) {
            restore();
            if (o.timing_mode == "graph") {
                execute(); cuda_check(cudaStreamSynchronize(resources.stream), "synchronize graph forward");
            } else {
                cuda_check(cudaEventRecord(resources.start, resources.stream), "record start");
                append(); cuda_check(cudaEventRecord(resources.stop, resources.stream), "record stop");
                cuda_check(cudaEventSynchronize(resources.stop), "synchronize stop");
            }
            float ms = 0; cuda_check(cudaEventElapsedTime(&ms, resources.start, resources.stop), "cudaEventElapsedTime");
            if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("Nonpositive/nonfinite event duration");
            raw_samples.push_back(ms); samples.push_back(ms / float(o.graph_repeats));
        }
        phase = "correctness";
        bool unchanged = true;
        if (o.placement == "outofplace") {
            unchanged = input_unchanged(); result.add("input_unchanged", unchanged ? "true" : "false");
        }
        // The final timed forward consumed the same restored input as every sample.
        cuda_check(cudaMemcpyAsync(host.data(), resources.buffer, bytes, cudaMemcpyDeviceToHost, resources.stream), "read output D2H");
        cuda_check(cudaStreamSynchronize(resources.stream), "synchronize output D2H");
        result.add("correctness", correctness(o, host, ref, base, passed));
        passed = passed && unchanged;
        result.str("status", passed ? "ok" : "correctness_failed");
    } catch (const std::exception& error) {
        result.str("status", "error"); result.str("failure_phase", phase); result.str("error", error.what());
    }
    result.integer("vkfft_result_code", vkfft_code);
    result.add("initialize_ms", number(initialize_ms)); result.add("exec_ms_samples", array(samples.data(), samples.size()));
    result.add("raw_interval_ms_samples", array(raw_samples.data(), raw_samples.size()));
    if (!samples.empty()) {
        auto sorted = samples; std::sort(sorted.begin(), sorted.end());
        const auto n = sorted.size(); const double median = n % 2 ? sorted[n / 2] : (double(sorted[n / 2 - 1]) + sorted[n / 2]) / 2;
        result.add("exec_ms_median", number(median));
        auto raw_sorted = raw_samples; std::sort(raw_sorted.begin(), raw_sorted.end());
        const double raw_median = n % 2 ? raw_sorted[n / 2] : (double(raw_sorted[n / 2 - 1]) + raw_sorted[n / 2]) / 2;
        result.add("raw_interval_ms_median", number(raw_median));
    } else { result.add("exec_ms_median", "null"); result.add("raw_interval_ms_median", "null"); }
    return result.finish();
}
}
int main(int argc, char** argv) {
    try {
        Options o = options(argc, argv); bool all_passed = true;
        auto emit = [&](const Options& candidate) { bool passed = false; std::string json = run(candidate, passed); std::cout << json << std::endl; all_passed &= passed; };
        if (!o.grid) emit(o);
        else {
            auto candidate = o; candidate.aim = candidate.boost = candidate.lut = 0; emit(candidate);
            for (int aim : {64, 128, 256}) for (int boost : {1, 2, 4}) for (int lut : {-1, 1}) {
                candidate.aim = aim; candidate.boost = boost; candidate.lut = lut; emit(candidate);
            }
        }
        return all_passed ? 0 : 1;
    } catch (const std::exception& error) {
        Object j; j.str("status", "error"); j.str("failure_phase", "arguments"); j.str("error", error.what());
        std::cout << j.finish() << std::endl; return 2;
    }
}
