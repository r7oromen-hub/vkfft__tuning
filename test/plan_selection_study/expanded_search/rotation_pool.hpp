#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rotation_pool {
struct Plan { std::size_t slots, total_bytes, bytes_per_buffer; };
// K powers of two makes a power-of-two graph repeat count divide K exactly.
// `max_bytes` is a hard pool ceiling, not a claim about free device memory.
inline bool plan(std::size_t bytes, std::size_t l2_bytes, std::size_t max_bytes,
                 std::size_t graph_repeats, Plan& result) {
    if (!bytes || !l2_bytes || !max_bytes || graph_repeats < 2 || graph_repeats > 2048 ||
        (graph_repeats & (graph_repeats-1))) return false;
    std::size_t slots=2;
    while (slots <= graph_repeats && slots <= l2_bytes / bytes) slots *= 2;
    if (slots > graph_repeats || graph_repeats % slots ||
        bytes > std::numeric_limits<std::size_t>::max()/2/slots) return false;
    const auto total=2*slots*bytes;
    if (total > max_bytes) return false;
    result={slots,total,bytes};return true;
}
inline std::size_t slot(std::size_t index, std::size_t slots) { return index % slots; }
// Scales are exactly representable and separated enough to reject a slot swap
// under the experiment's normwise 1e-5 threshold (R <= 2048).
inline double scale(std::size_t index) { return 1.0 + double(index) / 2048.0; }
}
