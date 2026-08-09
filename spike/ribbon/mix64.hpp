#pragma once
// Shared 64-bit mixer for store keys / shard IDs (host + device).
// Design: ../../615-a100-ribbon-shard-plan.md §2.1, §9 R7

#include <cstdint>

#ifdef __CUDACC__
#define MIX64_HD __host__ __device__ __forceinline__
#else
#define MIX64_HD inline
#endif

MIX64_HD std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Independent streams from one key (position / coeffs / fingerprint / shard).
MIX64_HD std::uint64_t mix64_seeded(std::uint64_t x, std::uint64_t seed) {
    return mix64(x + seed);
}

#undef MIX64_HD
