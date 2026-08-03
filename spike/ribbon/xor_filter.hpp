#pragma once
// Classic 3-hash xor filter (Graf & Lemire), host builder + query.
// Staging structure before ribbon (615-ribbon-filter-plan.md §6, R1).
// Campaign plan: ../../615-a100-ribbon-shard-plan.md M1.

#include "mix64.hpp"
#include "store_header.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct XorFilter {
    store615::StoreHeader hdr{};
    std::vector<std::uint64_t> cells;  // length = hdr.m_cells; low r bits used

    std::uint64_t fp_mask() const {
        return hdr.r >= 64 ? ~0ULL : ((1ULL << hdr.r) - 1ULL);
    }
};

namespace xor_detail {

inline std::uint64_t fingerprint(std::uint64_t hash, std::uint32_t r) {
    if (r >= 64) return hash;
    return hash & ((1ULL << r) - 1ULL);
}

// Map hash -> three cell indices in [0, 3*block).
inline void hashes(std::uint64_t hash, std::uint32_t block, std::uint32_t out[3]) {
    const std::uint64_t h0 = hash;
    const std::uint64_t h1 = mix64(hash ^ 0x9E3779B97F4A7C15ULL);
    const std::uint64_t h2 = mix64(hash ^ 0xBF58476D1CE4E5B9ULL);
    out[0] = static_cast<std::uint32_t>(h0 % block);
    out[1] = static_cast<std::uint32_t>(h1 % block) + block;
    out[2] = static_cast<std::uint32_t>(h2 % block) + 2u * block;
}

inline std::uint64_t key_hash(std::uint64_t key, std::uint64_t seed) {
    return mix64_seeded(key, seed);
}

struct PeelEntry {
    std::uint64_t hash;
    std::uint32_t index;  // alone cell
};

inline std::size_t capacity_for(std::size_t n) {
    // 1.23 * n, rounded up to multiple of 3; +32 slack like reference impls.
    double need = 32.0 + 1.23 * static_cast<double>(n);
    std::size_t cap = static_cast<std::size_t>(need + 0.999);
    if (cap < 3) cap = 3;
    cap = (cap + 2) / 3 * 3;
    return cap;
}

inline bool construct_with_seed(const std::vector<std::uint64_t>& keys, std::uint32_t r,
                                std::uint64_t seed, XorFilter& out) {
    const std::size_t n = keys.size();
    if (n == 0) {
        out.hdr = {};
        out.hdr.magic = store615::kMagic;
        out.hdr.version = store615::kVersion;
        out.hdr.kind = static_cast<std::uint32_t>(store615::Kind::Xor);
        out.hdr.r = r;
        out.hdr.mix_seed = seed;
        out.hdr.n_keys = 0;
        out.hdr.m_cells = 0;
        out.hdr.shard_count = 1;
        out.cells.clear();
        return true;
    }

    const std::size_t array_len = capacity_for(n);
    const std::uint32_t block = static_cast<std::uint32_t>(array_len / 3);

    std::vector<std::uint64_t> setxor(array_len, 0);
    std::vector<std::uint32_t> counts(array_len, 0);
    std::vector<std::uint64_t> hashes_v(n);

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t h = key_hash(keys[i], seed);
        hashes_v[i] = h;
        std::uint32_t hs[3];
        hashes(h, block, hs);
        for (int t = 0; t < 3; ++t) {
            setxor[hs[t]] ^= h;
            counts[hs[t]] += 1;
        }
    }

    std::vector<PeelEntry> stack;
    stack.reserve(n);
    std::vector<std::uint32_t> queue;
    queue.reserve(array_len);
    for (std::uint32_t i = 0; i < array_len; ++i) {
        if (counts[i] == 1) queue.push_back(i);
    }

    std::size_t qhead = 0;
    while (qhead < queue.size()) {
        const std::uint32_t i = queue[qhead++];
        if (counts[i] != 1) continue;
        const std::uint64_t h = setxor[i];
        std::uint32_t hs[3];
        hashes(h, block, hs);
        stack.push_back(PeelEntry{h, i});
        for (int t = 0; t < 3; ++t) {
            const std::uint32_t j = hs[t];
            setxor[j] ^= h;
            if (counts[j] > 0) {
                counts[j] -= 1;
                if (counts[j] == 1) queue.push_back(j);
            }
        }
    }

    if (stack.size() != n) return false;  // peel stalled — retry seed

    out.cells.assign(array_len, 0);
    const std::uint64_t mask = (r >= 64) ? ~0ULL : ((1ULL << r) - 1ULL);

    for (std::size_t k = stack.size(); k-- > 0;) {
        const PeelEntry& e = stack[k];
        std::uint32_t hs[3];
        hashes(e.hash, block, hs);
        std::uint64_t xorv = fingerprint(e.hash, r);
        for (int t = 0; t < 3; ++t) {
            if (hs[t] != e.index) xorv ^= (out.cells[hs[t]] & mask);
        }
        out.cells[e.index] = xorv & mask;
    }

    out.hdr.magic = store615::kMagic;
    out.hdr.version = store615::kVersion;
    out.hdr.kind = static_cast<std::uint32_t>(store615::Kind::Xor);
    out.hdr.r = r;
    out.hdr.w = 0;
    out.hdr.N = 0;
    out.hdr.n_keys = n;
    out.hdr.m_cells = array_len;
    out.hdr.mix_seed = seed;
    out.hdr.shard_count = 1;
    out.hdr.shard_index = 0;
    out.hdr.shard_mode = static_cast<std::uint32_t>(store615::ShardMode::KeyModS);
    out.hdr.reserved = 0;
    return true;
}

}  // namespace xor_detail

// Dedup keys (stable-unique after sort). Xor construction requires a set.
inline std::vector<std::uint64_t> xor_unique_keys(std::vector<std::uint64_t> keys) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

// Build xor filter. Throws if construction fails after max_seed_tries.
inline XorFilter build_xor(const std::vector<std::uint64_t>& keys_in, int rank = 48,
                           std::uint64_t base_seed = 0x615615615615615ULL,
                           int max_seed_tries = 64) {
    if (rank <= 0 || rank > 64) {
        throw std::invalid_argument("build_xor: rank must be in 1..64");
    }
    const std::vector<std::uint64_t> keys = xor_unique_keys(keys_in);
    XorFilter f;
    for (int try_i = 0; try_i < max_seed_tries; ++try_i) {
        const std::uint64_t seed = base_seed + static_cast<std::uint64_t>(try_i) * 0x9E3779B97F4A7C15ULL;
        if (xor_detail::construct_with_seed(keys, static_cast<std::uint32_t>(rank), seed, f)) {
            return f;
        }
    }
    throw std::runtime_error("build_xor: peel failed after " + std::to_string(max_seed_tries) +
                             " seeds (n=" + std::to_string(keys.size()) + ")");
}

inline bool xor_might_contain(const XorFilter& f, std::uint64_t key) {
    if (f.hdr.n_keys == 0) return false;
    if (f.cells.empty() || f.hdr.m_cells < 3) return false;
    const std::uint32_t block = static_cast<std::uint32_t>(f.hdr.m_cells / 3);
    const std::uint64_t h = xor_detail::key_hash(key, f.hdr.mix_seed);
    std::uint32_t hs[3];
    xor_detail::hashes(h, block, hs);
    const std::uint64_t mask = f.fp_mask();
    const std::uint64_t got =
        (f.cells[hs[0]] ^ f.cells[hs[1]] ^ f.cells[hs[2]]) & mask;
    return got == xor_detail::fingerprint(h, f.hdr.r);
}
