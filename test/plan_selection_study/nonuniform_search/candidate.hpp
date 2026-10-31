#pragma once
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// This is a prospective request contract, not scheduler or device admission.
namespace nonuniform_candidate {
struct Request {
    int n = 0;
    int workload_batch = 0;
    int group = 0;
    // Compatibility alias for the isolated expanded harness: batch is G,
    // while workload_batch is the workload's full batch count.
    int batch = 0;
    int min_regs = 0;
    int max_regs = 0;
    std::vector<int> radix;
    std::array<int, 33> table{};
    std::string profile;
};

inline bool enabled() {
    return std::getenv("VKFFT_NONUNIFORM_VERSION") || std::getenv("VKFFT_NONUNIFORM_N") ||
           std::getenv("VKFFT_NONUNIFORM_WORKLOAD_BATCH") || std::getenv("VKFFT_NONUNIFORM_RADIX") ||
           std::getenv("VKFFT_NONUNIFORM_TABLE") || std::getenv("VKFFT_NONUNIFORM_PROFILE") ||
           std::getenv("VKFFT_NONUNIFORM_GROUP");
}
inline bool conflicting() {
    return std::getenv("VKFFT_EXPANDED_RADIX") || std::getenv("VKFFT_EXPANDED_REGS") ||
           std::getenv("VKFFT_EXPANDED_BATCH") || std::getenv("VKFFT_DIRECT_RADIX") ||
           std::getenv("VKFFT_DIRECT_BATCH") || std::getenv("VKFFT_DIRECT_BLOCK");
}
// The reused harness calls this before checking enabled().
inline bool direct_enabled() { return conflicting(); }

inline bool numbers(const char* text, std::vector<int>& result, std::size_t limit, bool allow_zero) {
    if (!text || !*text || !result.empty()) return false;
    while (*text) {
        if (*text < '0' || *text > '9' || result.size() >= limit) return false;
        char* end = nullptr;
        errno = 0;
        const long value = std::strtol(text, &end, 10);
        if (errno || end == text || value < 0 || value > INT_MAX || (!allow_zero && value == 0)) return false;
        result.push_back(static_cast<int>(value));
        text = end;
        if (!*text) return true;
        if (*text++ != ',' || !*text) return false;
    }
    return false;
}
inline bool scalar(const char* text, int& value) {
    std::vector<int> parsed;
    if (!numbers(text, parsed, 1, false) || parsed.size() != 1) return false;
    value = parsed[0];
    return true;
}
inline bool supported_radix(int value) {
    return value == 2 || value == 4 || value == 8 || value == 16 || value == 32;
}
inline bool power_of_two(int value) { return value > 0 && !(value & (value - 1)); }

inline bool valid(const Request& r, long long n, long long batch) {
    if (n != r.n || batch != r.workload_batch || n < 2 || n > 8192 ||
        (n & (n - 1)) || batch < 1 || batch > 65536 || r.radix.empty() ||
        r.radix.size() > 32 || !power_of_two(r.group) || r.batch != r.group ||
        batch % r.group) return false;
    bool active[33] = {};
    long long product = 1;
    int min_regs = INT_MAX, max_regs = 0;
    for (const int radix : r.radix) {
        if (!supported_radix(radix) || product > n / radix) return false;
        product *= radix;
        active[radix] = true;
        const int value = r.table[radix];
        if (!power_of_two(value) || value < radix || value > 256 || value > n ||
            value % radix || n % value) return false;
        min_regs = std::min(min_regs, value);
        max_regs = std::max(max_regs, value);
    }
    if (product != n || min_regs != r.min_regs || max_regs != r.max_regs || n % min_regs) return false;
    for (int radix : r.radix) if (max_regs % r.table[radix]) return false;
    if (r.profile == "nonuniform_active_only") {
        for (int i = 0; i < 33; ++i) if (!active[i] && r.table[i] != 0) return false;
    } else if (r.profile == "uniform_compatibility") {
        if (min_regs != max_regs) return false;
        for (int i = 0; i < 33; ++i) {
            const int expected = supported_radix(i) && max_regs % i == 0 ? max_regs : 0;
            if (r.table[i] != expected) return false;
        }
    } else return false;
    return true;
}
inline bool valid(const Request& r, long long n) { return valid(r, n, r.workload_batch); }

inline bool request(Request& result) {
    const char* version = std::getenv("VKFFT_NONUNIFORM_VERSION");
    const char* profile = std::getenv("VKFFT_NONUNIFORM_PROFILE");
    if (conflicting() || !version || std::strcmp(version, "nonuniform-register-request/v1") != 0 ||
        !profile || !*profile) return false;
    Request parsed;
    std::vector<int> table;
    if (!scalar(std::getenv("VKFFT_NONUNIFORM_N"), parsed.n) ||
        !scalar(std::getenv("VKFFT_NONUNIFORM_WORKLOAD_BATCH"), parsed.workload_batch) ||
        !scalar(std::getenv("VKFFT_NONUNIFORM_GROUP"), parsed.group) ||
        !numbers(std::getenv("VKFFT_NONUNIFORM_RADIX"), parsed.radix, 32, false) ||
        !numbers(std::getenv("VKFFT_NONUNIFORM_TABLE"), table, 33, true) || table.size() != 33)
        return false;
    parsed.batch = parsed.group;
    parsed.profile = profile;
    std::copy(table.begin(), table.end(), parsed.table.begin());
    parsed.min_regs = INT_MAX;
    for (const int radix : parsed.radix) {
        if (!supported_radix(radix)) return false;
        parsed.min_regs = std::min(parsed.min_regs, parsed.table[radix]);
        parsed.max_regs = std::max(parsed.max_regs, parsed.table[radix]);
    }
    if (!valid(parsed, parsed.n, parsed.workload_batch)) return false;
    result = parsed;
    return true;
}

inline bool seed(long long n, int* multipliers, int* per_radix, int& max_regs, int& min_regs) {
    Request r;
    if (!multipliers || !per_radix || !request(r) || !valid(r, n, r.workload_batch)) return false;
    for (int i = 0; i < 33; ++i) {
        multipliers[i] = 0;
        per_radix[i] = r.table[i];
    }
    for (const int radix : r.radix) ++multipliers[radix];
    min_regs = r.min_regs;
    max_regs = r.max_regs;
    return true;
}
// Called after stage multiset construction but before downstream planning.
inline bool reorder(int* stages, int count) {
    Request r;
    if (!stages || !request(r) || count != static_cast<int>(r.radix.size())) return false;
    int expected[33] = {}, actual[33] = {};
    for (int radix : r.radix) ++expected[radix];
    for (int i = 0; i < count; ++i) {
        if (!supported_radix(stages[i])) return false;
        ++actual[stages[i]];
    }
    for (int i = 0; i < 33; ++i) if (expected[i] != actual[i]) return false;
    std::copy(r.radix.begin(), r.radix.end(), stages);
    return true;
}
inline bool matches_stages(const int* stages, int count) {
    Request r;
    return stages && request(r) && count == static_cast<int>(r.radix.size()) &&
           std::equal(r.radix.begin(), r.radix.end(), stages);
}
inline bool consistent(int n, int min_regs, int max_regs, const int* per_radix, int boost) {
    Request r;
    if (!per_radix || !request(r) || !valid(r, n, r.workload_batch) || boost != 1 ||
        min_regs != r.min_regs || max_regs != r.max_regs) return false;
    return std::equal(r.table.begin(), r.table.end(), per_radix);
}
// Evaluate before VkFFT resets axisSwapped while normalizing dispatch geometry.
inline bool layout_matches(int n, int min_regs, int group, int axis_swapped,
                           long long local0, long long local1, long long local2) {
    if (n <= 0 || min_regs <= 0 || n % min_regs || group <= 0 ||
        (axis_swapped != 0 && axis_swapped != 1)) return false;
    return (axis_swapped ? local1 : local0) == n / min_regs &&
           (axis_swapped ? local0 : local1) == group && local2 == 1;
}
}

// The separately generated harness is adapted from expanded_search.
namespace expanded_candidate = nonuniform_candidate;
