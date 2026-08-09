#pragma once
// Bit-packed xor cell load/store (fingerprint width = r bits per cell).
// Host + device. Pad packed buffer with +8 bytes for unaligned 64-bit reads.

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define XORPACK_HD __host__ __device__ __forceinline__
#else
#define XORPACK_HD inline
#endif

XORPACK_HD std::uint64_t xor_fp_mask(std::uint32_t r) {
    return r >= 64 ? ~0ULL : ((1ULL << r) - 1ULL);
}

// Bytes needed for m_cells fingerprints of width r, plus 8-byte pad.
XORPACK_HD std::size_t xor_packed_bytes(std::uint64_t m_cells, std::uint32_t r) {
    if (m_cells == 0 || r == 0) return 0;
    const std::uint64_t bits = m_cells * (std::uint64_t)r;
    return (std::size_t)((bits + 7ull) / 8ull + 8ull);
}

XORPACK_HD double xor_packed_gb(std::uint64_t m_cells, std::uint32_t r) {
    return (double)xor_packed_bytes(m_cells, r) / 1e9;
}

XORPACK_HD std::uint64_t xor_load_cell(const std::uint8_t* p, std::uint64_t cell_index,
                                       std::uint32_t r) {
    const std::uint64_t bit_index = cell_index * (std::uint64_t)r;
    const std::uint64_t byte_index = bit_index >> 3;
    const unsigned bit_offset = (unsigned)(bit_index & 7u);
    const std::uint64_t mask = xor_fp_mask(r);

#ifdef __CUDA_ARCH__
    std::uint64_t word = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        word |= (std::uint64_t)p[byte_index + (std::uint64_t)k] << (8 * k);
#else
    std::uint64_t word = 0;
    std::memcpy(&word, p + byte_index, 8);
#endif

    if (bit_offset + r <= 64u)
        return (word >> bit_offset) & mask;

    // Spans two 64-bit windows (only when r > 56).
#ifdef __CUDA_ARCH__
    std::uint64_t word2 = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        word2 |= (std::uint64_t)p[byte_index + 8ull + (std::uint64_t)k] << (8 * k);
#else
    std::uint64_t word2 = 0;
    std::memcpy(&word2, p + byte_index + 8, 8);
#endif
    return ((word >> bit_offset) | (word2 << (64u - bit_offset))) & mask;
}

// Host-side pack (also visible to nvcc host pass; not used on device).
// Do not wrap in #ifndef __CUDA_ARCH__ — that breaks .cu includes of build_xor.
#ifdef __CUDACC__
__host__
#endif
inline void xor_store_cell(std::uint8_t* p, std::uint64_t cell_index, std::uint32_t r,
                           std::uint64_t value) {
    value &= xor_fp_mask(r);
    const std::uint64_t bit_index = cell_index * (std::uint64_t)r;
    const std::uint64_t byte_index = bit_index >> 3;
    const unsigned bit_offset = (unsigned)(bit_index & 7u);
    const std::uint64_t mask = xor_fp_mask(r);

    auto load8 = [&](std::size_t off) {
        std::uint64_t w = 0;
#ifdef __CUDA_ARCH__
        for (int k = 0; k < 8; ++k)
            w |= (std::uint64_t)p[off + (std::size_t)k] << (8 * k);
#else
        std::memcpy(&w, p + off, 8);
#endif
        return w;
    };
    auto store8 = [&](std::size_t off, std::uint64_t w) {
#ifdef __CUDA_ARCH__
        for (int k = 0; k < 8; ++k)
            p[off + (std::size_t)k] = (std::uint8_t)((w >> (8 * k)) & 0xffu);
#else
        std::memcpy(p + off, &w, 8);
#endif
    };

    if (bit_offset + r <= 64u) {
        const std::uint64_t wmask = mask << bit_offset;
        std::uint64_t word = load8((std::size_t)byte_index);
        word = (word & ~wmask) | (value << bit_offset);
        store8((std::size_t)byte_index, word);
        return;
    }
    const unsigned low_bits = 64u - bit_offset;
    std::uint64_t word = load8((std::size_t)byte_index);
    const std::uint64_t low_mask = (low_bits >= 64u) ? ~0ULL : ((1ULL << low_bits) - 1ULL);
    word = (word & ~(low_mask << bit_offset)) | ((value & low_mask) << bit_offset);
    store8((std::size_t)byte_index, word);

    std::uint64_t word2 = load8((std::size_t)byte_index + 8);
    const std::uint64_t high = value >> low_bits;
    const std::uint64_t high_bits = r - low_bits;
    const std::uint64_t high_mask = (high_bits >= 64u) ? ~0ULL : ((1ULL << high_bits) - 1ULL);
    word2 = (word2 & ~high_mask) | (high & high_mask);
    store8((std::size_t)byte_index + 8, word2);
}

#undef XORPACK_HD
