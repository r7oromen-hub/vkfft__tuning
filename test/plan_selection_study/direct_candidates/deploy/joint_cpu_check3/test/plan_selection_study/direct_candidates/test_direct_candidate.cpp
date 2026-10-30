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
    int regs[33] = {}; regs[2] = regs[4] = regs[8] = regs[16] = regs[32] = 32;
    int stages[33] = {32,32,8,2}; int count = 4;
    set("VKFFT_DIRECT_RADIX", "32,8,8,4,2");
    set("VKFFT_DIRECT_BATCH", "4"); set("VKFFT_DIRECT_BLOCK", "4,32,1");
    assert(direct_candidate::enabled());
    assert(direct_candidate::refactor(stages, count, regs, 32, 32, 2));
    assert(count == 5 && direct_candidate::match_stages(stages, count));
    assert(direct_candidate::batch() == 4 && direct_candidate::match_block(4,32,1));
    assert(!direct_candidate::match_block(32,4,1));
    set("VKFFT_DIRECT_RADIX", "32,8,8,8,2"); assert(!direct_candidate::refactor(stages,count,regs,32,32,2));
    set("VKFFT_DIRECT_RADIX", "8,32,8,4,2"); assert(!direct_candidate::refactor(stages,count,regs,32,32,2));
    set("VKFFT_DIRECT_RADIX", "32,8,8,4,4"); assert(!direct_candidate::refactor(stages,count,regs,32,32,2));
    set("VKFFT_DIRECT_RADIX", "32,8,8,4,2,"); assert(!direct_candidate::refactor(stages,count,regs,32,32,2));
    set("VKFFT_DIRECT_RADIX", "32,8,8,4,2"); regs[4] = 0; assert(!direct_candidate::refactor(stages,count,regs,32,32,2));
    regs[4] = 16; assert(!direct_candidate::refactor(stages,count,regs,32,32,2));
    regs[4] = 30; assert(!direct_candidate::refactor(stages,count,regs,32,32,2));
    set("VKFFT_DIRECT_BATCH", "-1"); assert(direct_candidate::batch() == 0);
    set("VKFFT_DIRECT_BLOCK", "4,32,1,0"); assert(!direct_candidate::match_block(4,32,1));
}
