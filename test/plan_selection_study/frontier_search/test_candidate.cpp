#include "candidate.hpp"
#include "rejection.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>

void env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value) setenv(key, value, 1); else unsetenv(key);
#endif
}

int main() {
    expanded_rejection::reset();
    assert(!expanded_rejection::reason);
    expanded_rejection::record("first"); expanded_rejection::record("second");
    assert(!std::strcmp(expanded_rejection::reason, "first"));
    expanded_rejection::reset(); assert(!expanded_rejection::reason);
    using namespace expanded_candidate;
    env("VKFFT_EXPANDED_RADIX", "32,16,16");
    env("VKFFT_EXPANDED_BATCH", "4");
    for (const char* value : {"128", "256"}) {
        env("VKFFT_EXPANDED_REGS", value);
        Request r; assert(request(r) && valid(r, 8192));
        int multipliers[33], per[33], hi, lo;
        assert(seed(8192, multipliers, per, hi, lo));
        int expected = std::atoi(value);
        assert(lo == expected && hi == expected);
        for (int i = 0; i < 33; ++i) {
            bool supported = i == 2 || i == 4 || i == 8 || i == 16 || i == 32;
            assert(per[i] == (supported ? expected : 0));
            assert(multipliers[i] == (i == 32 ? 1 : i == 16 ? 2 : 0));
        }
        assert(consistent(8192, lo, hi, per, 1));
        assert(!consistent(8192, lo, hi, per, 2));
        per[2] = 64; assert(!consistent(8192, lo, hi, per, 1));
        assert(layout_matches(8192, expected, 4, 0, 8192 / expected, 4, 1));
        assert(!valid(r, expected / 2));
    }
    env("VKFFT_EXPANDED_REGS", "512"); Request above; assert(!request(above));
    env("VKFFT_EXPANDED_REGS", "128x"); Request malformed; assert(!request(malformed));
    env("VKFFT_EXPANDED_REGS", "128"); env("VKFFT_EXPANDED_BATCH", "3");
    Request bad_group; assert(!request(bad_group));
}
