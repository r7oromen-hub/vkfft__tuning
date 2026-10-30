#include "cpu_reference.hpp"
#include <algorithm>
#include <iostream>

int main() {
    try {
        double max_error = 0;
        for (std::size_t n : {1, 2, 3, 5, 6, 8, 9, 10, 12, 15, 16, 25, 30, 64, 75, 125}) {
            auto base = motivation::base_input(n, 42);
            const auto fft = motivation::reference_fft(base);
            for (std::size_t b = 0; b < 20; ++b) {
                for (std::size_t k = 0; k < n; ++k) {
                    motivation::Complex expected(0, 0);
                    for (std::size_t j = 0; j < n; ++j) {
                        auto v = motivation::input_value(base[j], b);
                        if (double(float(v.real())) != v.real() || double(float(v.imag())) != v.imag())
                            throw std::runtime_error("Input is not exactly representable in FP32");
                        expected += v * std::polar(1.0, -2 * motivation::pi * double(j * k) / double(n));
                    }
                    max_error = std::max(max_error, std::abs(expected - motivation::reference_value(fft[k], k, b, n)));
                }
            }
        }
        if (max_error > 1e-9 || motivation::smooth235(7) || motivation::smooth235(0))
            throw std::runtime_error("CPU FFT reference self-test failed");
        for (std::size_t n : {4, 12, 64, 100, 1280}) for (const std::string pattern : {"zero", "impulse", "constant", "quarter-tone"}) {
            const auto input = motivation::base_input(n, 42, pattern);
            const auto cpu = motivation::reference_fft(input), analytic = motivation::analytic_reference(n, pattern);
            double input2 = 0, error2 = 0, peak = 0;
            for (std::size_t k = 0; k < n; ++k) {
                input2 += std::norm(input[k]);
                const double e = std::abs(cpu[k] - analytic[k]); error2 += e * e; peak = std::max(peak, e);
            }
            if (!motivation::evaluate_normwise(n, input2, error2, peak, true).passed)
                throw std::runtime_error("Analytic pattern reference crosscheck failed");
        }
        using motivation::evaluate_normwise;
        if (!evaluate_normwise(1024, 0, 0, 0, true).passed || evaluate_normwise(1024, 0, 1e-30, 1e-15, true).passed ||
            evaluate_normwise(1024, 1, 0, 0, false).passed)
            throw std::runtime_error("Zero/nonfinite criterion negative test failed");
        // Nonzero impulse input has constant output of magnitude one.
        if (evaluate_normwise(1024, 1, 1024, 1, true).passed)
            throw std::runtime_error("All-zero output was accepted for nonzero input");
        // A quarter-tone moved to the wrong bin has two errors of magnitude N.
        if (evaluate_normwise(1024, 1024, 2.0 * 1024 * 1024, 1024, true).passed)
            throw std::runtime_error("Permuted output was accepted");
        // One 1e-3*reference-RMS bin error passes L2 for large N but must fail peak.
        const auto isolated = evaluate_normwise(1048576, 1, 1e-6, 1e-3, true);
        if (isolated.l2 > motivation::l2_error_threshold || isolated.passed || isolated.linf <= motivation::linf_error_threshold)
            throw std::runtime_error("Isolated-bin peak criterion negative test failed");
        std::cout << "CPU mixed-radix FFT vs direct DFT: passed; max absolute error=" << max_error << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
