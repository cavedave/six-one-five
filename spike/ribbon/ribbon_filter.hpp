#pragma once
// Ribbon filter spike — host-side interface stub.
// Full design: ../../615-ribbon-filter-plan.md

#include <cstdint>
#include <vector>

using RibbonKey = std::uint64_t;

struct RibbonFilter {
    std::vector<std::uint64_t> bits;  // packed bit array F
    std::uint64_t num_bits = 0;
    int rank = 48;                    // target false-positive rate 2^-rank
    int word_bits = 64;
};

// Build a ribbon filter over fp64 keys. Throws or returns empty on failure.
// TODO(M4): BuRR-style construction with seed-retry.
// M1 staging structure is xor_filter.hpp (use that for A100 path first).
inline RibbonFilter build_ribbon(const std::vector<RibbonKey>& keys, int rank = 48, int word_bits = 64) {
    (void)keys;
    (void)rank;
    (void)word_bits;
    return {};
}

// Host query: true iff key might be present (never false for inserted keys).
inline bool ribbon_might_contain(const RibbonFilter& f, RibbonKey key) {
    (void)f;
    (void)key;
    return true;   // stub: permissive until M1 lands
}
