#pragma once
#include <cstdio>

// First explicit prototype rejection in this thread, not a diagnosis of an
// arbitrary VkFFT or CUDA error. Strings are static literals owned by code.
namespace expanded_rejection {
inline thread_local const char* reason = nullptr;
inline void reset() { reason = nullptr; }
inline void record(const char* value) {
    if (!reason) {
        reason = value;
        // All callers supply fixed identifier literals, not user input.
        std::fprintf(stderr, "VKFFT_FRONTIER_REJECTION:{\"reason\":\"%s\"}\n", value);
    }
}
}
