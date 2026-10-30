#include "cpu_reference.hpp"
#include <fftw3.h>
#include <algorithm>
#include <iostream>

int main() {
    try {
        for (std::size_t n : {256, 49152, 81920, 1048576}) {
            const auto input = motivation::base_input(n, 42);
            const auto reference = motivation::reference_fft(input);
            auto* in = fftw_alloc_complex(n);
            auto* out = fftw_alloc_complex(n);
            if (!in || !out) throw std::runtime_error("FFTW allocation failed");
            for (std::size_t i = 0; i < n; ++i) { in[i][0] = input[i].real(); in[i][1] = input[i].imag(); }
            auto plan = fftw_plan_dft_1d(int(n), in, out, FFTW_FORWARD, FFTW_ESTIMATE);
            if (!plan) throw std::runtime_error("FFTW plan failed");
            fftw_execute(plan);
            double error2 = 0, reference2 = 0, max_error = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const motivation::Complex expected(out[i][0], out[i][1]);
                const double error = std::abs(reference[i] - expected);
                error2 += error * error; reference2 += std::norm(expected);
                max_error = std::max(max_error, error);
            }
            fftw_destroy_plan(plan); fftw_free(in); fftw_free(out);
            const double relative = std::sqrt(error2 / reference2);
            std::cout << "CPU reference vs FFTW: N=" << n << " max_abs=" << max_error << " l2_relative=" << relative << '\n';
            if (!std::isfinite(relative) || relative > 1e-11) throw std::runtime_error("Large FFT reference mismatch");
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
