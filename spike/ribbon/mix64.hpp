#pragma once
// Shared 64-bit mixer for store keys / shard IDs (host + later device).
// Design: ../../615-a100-ribbon-shard-plan.md §2.1, §9 R7

#include <cstdint>

inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Independent streams from one key (position / coeffs / fingerprint / shard).
inline std::uint64_t mix64_seeded(std::uint64_t x, std::uint64_t seed) {
    return mix64(x + seed);
}
