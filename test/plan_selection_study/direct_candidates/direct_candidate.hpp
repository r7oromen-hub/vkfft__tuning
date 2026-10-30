#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <vector>
namespace direct_candidate {
inline bool enabled() { return std::getenv("VKFFT_DIRECT_RADIX") || std::getenv("VKFFT_DIRECT_BATCH") || std::getenv("VKFFT_DIRECT_BLOCK"); }
inline bool numbers(const char* s, std::vector<long long>& out) {
    if (!s || !*s) return false;
    while (*s) {
        if (*s < '0' || *s > '9') return false;
        char* end = nullptr; errno = 0; long long n = std::strtoll(s, &end, 10);
        if (errno || end == s || n <= 0 || out.size() >= 32) return false;
        out.push_back(n); s = end;
        if (!*s) return true;
        if (*s++ != ',' || !*s) return false;
    }
    return false;
}
// Re-factor only the interior. Endpoints are held fixed because they affect
// direct I/O and the registerBoost switch. Validate the scheduler's register
// envelope again for the requested stages before changing any state.
inline bool refactor(int* stages, int& count, const int* per_radix,
                     int min_regs, int max_regs, int register_boost) {
    std::vector<long long> wanted;
    if (count < 1 || count > 32 || !numbers(std::getenv("VKFFT_DIRECT_RADIX"), wanted) || wanted.size() > 32) return false;
    if (wanted.front() != stages[0] || wanted.back() != stages[count - 1]) return false;
    if ((count == 1 || wanted.size() == 1) && (count != 1 || wanted.size() != 1)) return false;
    auto exponent = [](long long x) -> int {
        switch (x) { case 2: return 1; case 4: return 2; case 8: return 3; case 16: return 4; case 32: return 5; default: return -1; }
    };
    int old_exp = 0, new_exp = 0, actual_min = 10000000, actual_max = 0;
    for (int i = 0; i < count; ++i) {
        const int e = exponent(stages[i]); if (e < 0) return false; old_exp += e;
    }
    for (size_t i = 0; i < wanted.size(); ++i) {
        const int e = exponent(wanted[i]); if (e < 0) return false;
        const int r = int(wanted[i]); new_exp += e;
        // VkFFT computes the min/max before appending the boost switch stage.
        if (register_boost > 1 && i + 1 == wanted.size()) continue;
        if (per_radix[r] <= 0 || per_radix[r] % r != 0) return false;
        actual_min = std::min(actual_min, per_radix[r]);
        actual_max = std::max(actual_max, per_radix[r]);
    }
    if (old_exp != new_exp || actual_min != min_regs || actual_max != max_regs) return false;
    for (size_t i = 0; i < wanted.size(); ++i) stages[i] = int(wanted[i]);
    count = int(wanted.size());
    return true;
}
inline long long batch() {
    std::vector<long long> v;
    return numbers(std::getenv("VKFFT_DIRECT_BATCH"), v) && v.size() == 1 ? v[0] : 0;
}
inline bool match_block(long long x, long long y, long long z) {
    std::vector<long long> v;
    return numbers(std::getenv("VKFFT_DIRECT_BLOCK"), v) && v.size() == 3 && v[0] == x && v[1] == y && v[2] == z;
}
inline bool match_stages(const int* stages, int count) {
    std::vector<long long> v;
    if (count < 1 || count > 32 || !numbers(std::getenv("VKFFT_DIRECT_RADIX"), v) || int(v.size()) != count) return false;
    for (int i = 0; i < count; ++i) if (v[i] != stages[i]) return false;
    return true;
}
}
