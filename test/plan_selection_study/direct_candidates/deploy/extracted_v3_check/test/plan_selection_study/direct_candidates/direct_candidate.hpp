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
inline bool reorder(int* stages, int count) {
    std::vector<long long> wanted;
    if (count < 1 || count > 32 || !numbers(std::getenv("VKFFT_DIRECT_RADIX"), wanted) || int(wanted.size()) != count) return false;
    if (wanted.front() != stages[0] || wanted.back() != stages[count-1]) return false;
    std::vector<long long> old(stages, stages + count), sorted = wanted;
    std::sort(old.begin(), old.end()); std::sort(sorted.begin(), sorted.end());
    if (old != sorted) return false;
    for (int i = 0; i < count; ++i) stages[i] = int(wanted[i]);
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
