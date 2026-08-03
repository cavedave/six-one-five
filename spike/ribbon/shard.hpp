#pragma once
// Residue sharding helpers (steered probe). Design: 615-a100-ribbon-shard-plan.md §2.4

#include "mix64.hpp"
#include "store_header.hpp"

#include <cstdint>

inline std::uint32_t shard_of_mixed(std::uint64_t x_mixed, std::uint32_t shard_count,
                                    store615::ShardMode mode = store615::ShardMode::KeyModS) {
    if (shard_count <= 1) return 0;
    switch (mode) {
        case store615::ShardMode::KeyModS:
        default:
            return static_cast<std::uint32_t>(x_mixed % shard_count);
    }
}

// Mix then shard — canonical path for raw pair-sum residues.
inline std::uint32_t shard_of_key(std::uint64_t key_raw, std::uint64_t mix_seed,
                                  std::uint32_t shard_count,
                                  store615::ShardMode mode = store615::ShardMode::KeyModS) {
    return shard_of_mixed(mix64_seeded(key_raw, mix_seed), shard_count, mode);
}
