#include "candidate.hpp"
#include <cassert>
#include <cstdlib>

void env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
int main() {
    using namespace expanded_candidate;
    env("VKFFT_EXPANDED_RADIX", nullptr); env("VKFFT_EXPANDED_REGS", nullptr); env("VKFFT_EXPANDED_BATCH", nullptr);
    assert(!enabled());
    env("VKFFT_EXPANDED_RADIX", "8,4,8"); env("VKFFT_EXPANDED_REGS", "8"); env("VKFFT_EXPANDED_BATCH", "4");
    Request r; assert(request(r) && valid(r, 256));
    int multipliers[33], per[33], lo, hi;
    for(int i=0;i<33;++i) multipliers[i]=per[i]=999;
    assert(seed(256,multipliers,per,hi,lo));
    assert(lo==8 && hi==8 && multipliers[8]==2 && multipliers[4]==1 && per[8]==8 && per[4]==8);
    assert(per[2]==8 && per[16]==0 && per[3]==0 && multipliers[2]==0);
    int actual[]={8,8,4}; assert(reorder(actual,3)); assert(actual[0]==8 && actual[1]==4 && actual[2]==8);
    assert(consistent(256,lo,hi,per,1)); assert(!consistent(256,lo,hi,per,2));
    assert(!consistent(256,lo*2,hi*2,per,1));
    per[3]=8; assert(!consistent(256,lo,hi,per,1)); per[3]=0;
    assert(layout_matches(128,8,8,0,16,8,1));
    assert(layout_matches(128,8,8,1,8,16,1));
    assert(!layout_matches(128,8,8,0,8,16,1));
    assert(!layout_matches(128,8,8,1,16,8,1));
    assert(!layout_matches(128,8,8,2,8,16,1));
    env("VKFFT_EXPANDED_REGS", "4"); Request invalid; assert(request(invalid) && !valid(invalid,256));
    env("VKFFT_EXPANDED_REGS", "8x"); Request malformed; assert(!request(malformed));
    env("VKFFT_EXPANDED_REGS", "8"); env("VKFFT_EXPANDED_RADIX", "16,16");
    Request big; assert(request(big) && !valid(big,256));
    env("VKFFT_EXPANDED_RADIX", "16,4,4"); env("VKFFT_EXPANDED_REGS", "16");
    Request diff; assert(request(diff) && valid(diff,256));
    int bad[]={16,8,2}; assert(!reorder(bad,3));
    env("VKFFT_EXPANDED_BATCH", "3"); Request notpow; assert(!request(notpow));
    env("VKFFT_EXPANDED_BATCH", nullptr); Request missing; assert(!request(missing));
    return 0;
}
