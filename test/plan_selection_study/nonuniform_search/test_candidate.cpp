#include "candidate.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <sstream>
#include <string>

static void env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value) setenv(key, value, 1); else unsetenv(key);
#endif
}

static std::string table_text(const std::array<int, 33>& table) {
    std::ostringstream out;
    for (std::size_t i = 0; i < table.size(); ++i) {
        if (i) out << ',';
        out << table[i];
    }
    return out.str();
}

static void set_table(const std::array<int, 33>& table) {
    const std::string value = table_text(table);
    env("VKFFT_NONUNIFORM_TABLE", value.c_str());
}

static void set_request(const char* radix, const std::array<int, 33>& table,
                        const char* profile = "nonuniform_active_only") {
    env("VKFFT_NONUNIFORM_VERSION", "nonuniform-register-request/v1");
    env("VKFFT_NONUNIFORM_N", "128");
    env("VKFFT_NONUNIFORM_WORKLOAD_BATCH", "16");
    env("VKFFT_NONUNIFORM_RADIX", radix);
    set_table(table);
    env("VKFFT_NONUNIFORM_PROFILE", profile);
    env("VKFFT_NONUNIFORM_GROUP", "2");
}

int main() {
    using namespace nonuniform_candidate;
    for (const char* key : {"VKFFT_NONUNIFORM_VERSION", "VKFFT_NONUNIFORM_N",
                            "VKFFT_NONUNIFORM_WORKLOAD_BATCH", "VKFFT_NONUNIFORM_RADIX",
                            "VKFFT_NONUNIFORM_TABLE", "VKFFT_NONUNIFORM_PROFILE",
                            "VKFFT_NONUNIFORM_GROUP", "VKFFT_EXPANDED_RADIX",
                            "VKFFT_EXPANDED_REGS", "VKFFT_EXPANDED_BATCH",
                            "VKFFT_DIRECT_RADIX", "VKFFT_DIRECT_BATCH", "VKFFT_DIRECT_BLOCK"})
        env(key, nullptr);
    assert(!enabled());

    std::array<int, 33> active{}; active[2] = 4; active[8] = 8;
    set_request("8,8,2", active);
    Request r; assert(enabled() && !conflicting() && request(r));
    assert(valid(r, 128, 16));
    assert(!valid(r, 256, 16) && !valid(r, 128, 8));
    int multipliers[33] = {}, per[33] = {}, hi = 0, lo = 0;
    assert(seed(128, multipliers, per, hi, lo));
    assert(lo == 4 && hi == 8 && multipliers[8] == 2 && multipliers[2] == 1);
    for (int i = 0; i < 33; ++i) assert(per[i] == active[static_cast<std::size_t>(i)]);
    assert(consistent(128, lo, hi, per, 1));
    assert(!consistent(128, 8, 8, per, 1));
    assert(!consistent(128, lo, hi, per, 2));
    per[4] = 4; assert(!consistent(128, lo, hi, per, 1)); per[4] = 0;

    int sequence[] = {2, 8, 8};
    assert(!matches_stages(sequence, 3));
    assert(reorder(sequence, 3));
    assert(matches_stages(sequence, 3));
    assert(sequence[0] == 8 && sequence[1] == 8 && sequence[2] == 2);
    int wrong_multiset[] = {4, 4, 8}; assert(!reorder(wrong_multiset, 3));
    assert(!reorder(sequence, 2));
    assert(layout_matches(128, 4, 2, 0, 32, 2, 1));
    assert(layout_matches(128, 4, 2, 1, 2, 32, 1));
    assert(!layout_matches(128, 4, 2, 0, 16, 2, 1));  // N/maxR is wrong.
    assert(!layout_matches(128, 4, 2, 1, 32, 2, 1));
    assert(!layout_matches(128, 4, 2, 2, 2, 32, 1));

    // Each active radix has one shared table entry, including repeated 8 stages.
    std::array<int, 33> bad = active; bad[8] = 4;
    set_request("8,8,2", bad); Request invalid; assert(!request(invalid));
    bad = active; bad[2] = 0; set_table(bad); assert(!request(invalid));
    bad = active; bad[2] = 6; set_table(bad); assert(!request(invalid));
    bad = active; bad[4] = 4; set_table(bad); assert(!request(invalid));
    set_request("8,8,2", active); env("VKFFT_NONUNIFORM_GROUP", "3"); assert(!request(invalid));
    env("VKFFT_NONUNIFORM_GROUP", "32"); assert(!request(invalid));
    env("VKFFT_NONUNIFORM_GROUP", "2"); env("VKFFT_NONUNIFORM_N", "256"); assert(!request(invalid));
    env("VKFFT_NONUNIFORM_N", "128"); env("VKFFT_NONUNIFORM_RADIX", "8,8,2,"); assert(!request(invalid));
    env("VKFFT_NONUNIFORM_RADIX", "8,8,2"); env("VKFFT_NONUNIFORM_TABLE", "0,0,4"); assert(!request(invalid));
    set_table(active); env("VKFFT_NONUNIFORM_VERSION", "nonuniform-register-request/v2"); assert(!request(invalid));
    env("VKFFT_NONUNIFORM_VERSION", nullptr); assert(!request(invalid));

    // Uniform compatibility preserves the frontier's unused supported entries.
    std::array<int, 33> uniform{};
    for (int radix : {2, 4, 8, 16}) uniform[static_cast<std::size_t>(radix)] = 16;
    set_request("8,16", uniform, "uniform_compatibility");
    assert(request(r) && valid(r, 128, 16));
    assert(seed(128, multipliers, per, hi, lo) && lo == 16 && hi == 16);
    bad = uniform; bad[4] = 0; set_table(bad); assert(!request(invalid));
    set_table(uniform); env("VKFFT_NONUNIFORM_PROFILE", "nonuniform_active_only"); assert(!request(invalid));
    env("VKFFT_NONUNIFORM_PROFILE", "uniform_compatibility");
    env("VKFFT_EXPANDED_REGS", "16"); assert(conflicting() && !request(invalid));
    env("VKFFT_EXPANDED_REGS", nullptr);
    env("VKFFT_DIRECT_RADIX", "8,16"); assert(conflicting() && !request(invalid));
    env("VKFFT_DIRECT_RADIX", nullptr);
    for (int wide : {128, 256}) {
        std::array<int, 33> frontier_table{};
        for (int radix : {2, 4, 8, 16, 32}) frontier_table[static_cast<std::size_t>(radix)] = wide;
        set_request("32,16,16", frontier_table, "uniform_compatibility");
        env("VKFFT_NONUNIFORM_N", "8192");
        assert(request(r) && valid(r, 8192, 16));
        assert(seed(8192, multipliers, per, hi, lo) && hi == wide && lo == wide);
        assert(consistent(8192, lo, hi, per, 1));
    }
    return 0;
}
