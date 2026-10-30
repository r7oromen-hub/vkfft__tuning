#include "direct_candidate.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#ifdef _WIN32
static void set(const char* k, const char* v) { _putenv_s(k, v); }
#else
static void set(const char* k, const char* v) { setenv(k, v, 1); }
#endif
int main() {
    int stages[] = {2,4,8,16};
    set("VKFFT_DIRECT_RADIX", "2,8,4,16");
    set("VKFFT_DIRECT_BATCH", "4");
    set("VKFFT_DIRECT_BLOCK", "4,32,1");
    assert(direct_candidate::enabled());
    assert(direct_candidate::reorder(stages, 4));
    assert(direct_candidate::match_stages(stages, 4));
    assert(direct_candidate::batch() == 4);
    assert(direct_candidate::match_block(4, 32, 1));
    assert(!direct_candidate::match_block(32, 4, 1));
    set("VKFFT_DIRECT_RADIX", "8,2,4,16"); assert(!direct_candidate::reorder(stages, 4));
    set("VKFFT_DIRECT_RADIX", "2,8,8,16"); assert(!direct_candidate::reorder(stages, 4));
    set("VKFFT_DIRECT_RADIX", "2,8,4,16,"); assert(!direct_candidate::reorder(stages, 4));
    set("VKFFT_DIRECT_BATCH", "-1"); assert(direct_candidate::batch() == 0);
    set("VKFFT_DIRECT_BLOCK", "4,32,1,0"); assert(!direct_candidate::match_block(4,32,1));
}
