#pragma once
// On-disk / in-memory header for table | xor | ribbon stores.
// Design: ../../615-a100-ribbon-shard-plan.md §4.2

#include <cstdint>

namespace store615 {

constexpr std::uint32_t kMagic = 0x53353136u;  // '615S' little-endian-ish
constexpr std::uint32_t kVersion = 1;

enum class Kind : std::uint32_t {
    Table = 0,
    Xor = 1,
    Ribbon = 2,
};

enum class ShardMode : std::uint32_t {
    KeyModS = 0,
};

struct StoreHeader {
    std::uint32_t magic = kMagic;
    std::uint32_t version = kVersion;
    std::uint32_t kind = static_cast<std::uint32_t>(Kind::Xor);
    std::uint32_t r = 48;            // fingerprint bits
    std::uint32_t w = 0;             // ribbon width (0 if N/A)
    std::uint32_t N = 0;             // pair index cap
    std::uint64_t n_keys = 0;
    std::uint64_t m_cells = 0;       // xor: array length; ribbon: bit length
    std::uint64_t mix_seed = 0;
    std::uint32_t shard_count = 1;
    std::uint32_t shard_index = 0;
    std::uint32_t shard_mode = static_cast<std::uint32_t>(ShardMode::KeyModS);
    std::uint32_t reserved = 0;
};

inline bool header_ok(const StoreHeader& h) {
    return h.magic == kMagic && h.version == kVersion && h.shard_count >= 1 &&
           h.shard_index < h.shard_count && h.r > 0 && h.r <= 64;
}

}  // namespace store615
