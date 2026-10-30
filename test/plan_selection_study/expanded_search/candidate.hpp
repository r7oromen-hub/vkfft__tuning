#pragma once
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <vector>

// Isolated experimental requests. These checks are necessary, not a proof that
// a kernel is numerically or resource valid on a device.
namespace expanded_candidate {
inline bool enabled() {
    return std::getenv("VKFFT_EXPANDED_RADIX") || std::getenv("VKFFT_EXPANDED_REGS") || std::getenv("VKFFT_EXPANDED_BATCH");
}
inline bool direct_enabled() {
    return std::getenv("VKFFT_DIRECT_RADIX") || std::getenv("VKFFT_DIRECT_BATCH") || std::getenv("VKFFT_DIRECT_BLOCK");
}
inline bool numbers(const char* text, std::vector<int>& result) {
    if (!text || !*text) return false;
    while (*text) {
        if (*text < '0' || *text > '9' || result.size() >= 32) return false;
        char* end = nullptr; errno = 0;
        long v = std::strtol(text, &end, 10);
        if (errno || end == text || v <= 0 || v > INT_MAX) return false;
        result.push_back(int(v)); text = end;
        if (!*text) return true;
        if (*text++ != ',' || !*text) return false;
    }
    return false;
}
struct Request { std::vector<int> radix; int regs = 0, batch = 0; };
inline bool request(Request& r) {
    std::vector<int> regs, batch;
    if (!numbers(std::getenv("VKFFT_EXPANDED_RADIX"), r.radix) ||
        !numbers(std::getenv("VKFFT_EXPANDED_REGS"), regs) || regs.size() != 1 ||
        !numbers(std::getenv("VKFFT_EXPANDED_BATCH"), batch) || batch.size() != 1) return false;
    r.regs = regs[0]; r.batch = batch[0];
    return r.regs >= 2 && r.regs <= 64 && !(r.regs & (r.regs - 1)) && !(r.batch & (r.batch - 1));
}
inline bool valid(const Request& r, long long n) {
    if (n < 2 || n > 8192 || (n & (n - 1)) || r.radix.empty() || r.radix.size() > 32 ||
        r.regs < 2 || r.regs > 64 || (r.regs & (r.regs - 1)) || n % r.regs ||
        r.batch < 1 || (r.batch & (r.batch - 1))) return false;
    long long product = 1;
    for (int radix : r.radix) {
        if ((radix != 2 && radix != 4 && radix != 8 && radix != 16 && radix != 32) ||
            r.regs < radix || r.regs % radix || product > n / radix) return false;
        product *= radix;
    }
    return product == n;
}
inline bool seed(long long n, int* multipliers, int* per_radix, int& max_regs, int& min_regs) {
    Request r; if (!request(r) || !valid(r, n)) return false;
    for (int i = 0; i < 33; ++i) multipliers[i] = per_radix[i] = 0;
    // Match the power-of-two table construction in VkFFTGetRegistersPerThread:
    // retain entries for every supported radix dividing R, including unused ones.
    for (int radix : {2, 4, 8, 16, 32}) if (r.regs % radix == 0) per_radix[radix] = r.regs;
    for (int radix : r.radix) ++multipliers[radix];
    min_regs = max_regs = r.regs;
    return true;
}
inline bool reorder(int* stages, int count) {
    Request r; if (!request(r) || count != int(r.radix.size())) return false;
    int expected[33] = {}, actual[33] = {};
    for (int radix : r.radix) { if (radix < 2 || radix > 32) return false; ++expected[radix]; }
    for (int i = 0; i < count; ++i) { if (stages[i] < 2 || stages[i] > 32) return false; ++actual[stages[i]]; }
    for (int i = 2; i < 33; ++i) if (expected[i] != actual[i]) return false;
    for (int i = 0; i < count; ++i) stages[i] = r.radix[i];
    return true;
}
inline bool consistent(int n, int min_regs, int max_regs, const int* per_radix, int boost) {
    Request r; if (!request(r) || !valid(r, n) || boost != 1 || min_regs != r.regs || max_regs != r.regs) return false;
    for (int radix = 0; radix < 33; ++radix) {
        const bool supported = radix == 2 || radix == 4 || radix == 8 || radix == 16 || radix == 32;
        const int expected = supported && r.regs % radix == 0 ? r.regs : 0;
        if (per_radix[radix] != expected) return false;
    }
    return true;
}
// Evaluate before VkFFT resets axisSwapped while normalizing dispatch geometry.
inline bool layout_matches(int n, int regs, int batch, int axis_swapped,
                           long long local0, long long local1, long long local2) {
    if (n <= 0 || regs <= 0 || n % regs || batch <= 0 || (axis_swapped != 0 && axis_swapped != 1)) return false;
    return (axis_swapped ? local1 : local0) == n / regs &&
           (axis_swapped ? local0 : local1) == batch && local2 == 1;
}
}
