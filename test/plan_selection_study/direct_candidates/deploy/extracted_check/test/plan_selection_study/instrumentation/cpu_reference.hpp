#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace motivation {
using Complex = std::complex<double>;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double l2_error_threshold = 1e-5;
constexpr double linf_error_threshold = 1e-4;
struct NormwiseEvaluation { double l2, linf; bool passed; };
inline NormwiseEvaluation evaluate_normwise(std::size_t n, double input_norm2,
                                           double error_norm2, double max_abs_error, bool finite) {
    double l2 = 0, linf = 0;
    if (!finite || (input_norm2 == 0 && (max_abs_error != 0 || error_norm2 != 0)))
        l2 = linf = std::numeric_limits<double>::infinity();
    else if (input_norm2 > 0) {
        l2 = std::sqrt(error_norm2) / (std::sqrt(double(n)) * std::sqrt(input_norm2));
        linf = max_abs_error / std::sqrt(input_norm2);
    }
    return {l2, linf, finite && l2 <= l2_error_threshold && linf <= linf_error_threshold};
}

inline bool smooth235(std::size_t n) {
    if (!n) return false;
    for (auto r : {2u, 3u, 5u}) while (n % r == 0) n /= r;
    return n == 1;
}

// Independent decimation-in-time Cooley-Tukey reference. Negative-sign,
// unnormalized forward FFT, with no dependency on VkFFT or its generated code.
inline void fft_recursive(const Complex* input, std::size_t stride,
                          Complex* output, std::size_t n) {
    if (n == 1) { output[0] = input[0]; return; }
    std::size_t radix = n % 2 == 0 ? 2 : (n % 3 == 0 ? 3 : 5);
    const std::size_t m = n / radix;
    std::vector<Complex> sub(n);
    for (std::size_t j = 0; j < radix; ++j)
        fft_recursive(input + j * stride, stride * radix, sub.data() + j * m, m);
    for (std::size_t k = 0; k < n; ++k) {
        const Complex w = std::polar(1.0, -2.0 * pi * double(k) / double(n));
        Complex phase(1, 0), sum(0, 0);
        for (std::size_t j = 0; j < radix; ++j) {
            sum += sub[j * m + k % m] * phase;
            phase *= w;
        }
        output[k] = sum;
    }
}
inline std::vector<Complex> reference_fft(const std::vector<Complex>& input) {
    if (!smooth235(input.size())) throw std::runtime_error("N must contain only factors 2, 3, and 5");
    std::vector<Complex> output(input.size());
    fft_recursive(input.data(), 1, output.data(), input.size());
    return output;
}
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
inline std::vector<Complex> base_input(std::size_t n, uint64_t seed, const std::string& pattern = "random") {
    std::vector<Complex> input(n);
    if (pattern == "zero") return input;
    if (pattern == "constant") { std::fill(input.begin(), input.end(), Complex(1, -0.5)); return input; }
    if (pattern == "impulse") { input[0] = Complex(1, 0.5); return input; }
    if (pattern == "quarter-tone") {
        if (n % 4) throw std::runtime_error("quarter-tone pattern requires N divisible by 4");
        const Complex values[] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (std::size_t i = 0; i < n; ++i) input[i] = values[i % 4];
        return input;
    }
    if (pattern != "random") throw std::runtime_error("Unknown input pattern");
    for (std::size_t i = 0; i < n; ++i) {
        const auto x = splitmix64(seed + i);
        input[i] = Complex((int(x & 2047) - 1024) / 1024.0,
                           (int((x >> 16) & 2047) - 1024) / 1024.0);
    }
    return input;
}
inline double batch_scale(std::size_t b) { return double(1 + b % 7) / 8.0; }
inline Complex batch_offset(std::size_t b) {
    return Complex((int(b % 11) - 5) / 32.0, (int(b % 13) - 6) / 32.0);
}
// The base and batch transforms are dyadic with <= 17 significant bits, so
// these inputs are exactly representable in FP32. FFT linearity therefore
// gives an independent full-batch reference without rounding ambiguities.
inline Complex input_value(const Complex& base, std::size_t b, const std::string& pattern = "random") {
    return batch_scale(b) * base + (pattern == "random" ? batch_offset(b) : Complex(0, 0));
}
inline Complex reference_value(const Complex& base_fft, std::size_t k,
                               std::size_t b, std::size_t n, const std::string& pattern = "random") {
    return batch_scale(b) * base_fft + (k == 0 && pattern == "random" ? double(n) * batch_offset(b) : Complex(0, 0));
}
inline std::vector<Complex> analytic_reference(std::size_t n, const std::string& pattern) {
    std::vector<Complex> result(n);
    if (pattern == "zero") return result;
    if (pattern == "constant") result[0] = double(n) * Complex(1, -0.5);
    else if (pattern == "impulse") std::fill(result.begin(), result.end(), Complex(1, 0.5));
    else if (pattern == "quarter-tone" && n % 4 == 0) result[n / 4] = Complex(double(n), 0);
    else throw std::runtime_error("Pattern has no supported analytic reference");
    return result;
}
}
