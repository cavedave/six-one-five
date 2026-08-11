#pragma once
// Classic 3-hash xor filter (Graf & Lemire), host builder + query.
// Cells are bit-packed to `r` bits each (xor_pack.hpp). Staging before ribbon.
// Design: ../../615-ribbon-filter-plan.md §6, R1.
// On-disk: StoreHeader version 2 + packed bytes (see xor_packed_bytes).

#include "mix64.hpp"
#include "store_header.hpp"
#include "xor_pack.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct XorFilter {
    store615::StoreHeader hdr{};
    std::vector<std::uint8_t> packed;  // bit-packed fingerprints + 8-byte pad

    std::uint64_t fp_mask() const { return xor_fp_mask(hdr.r); }

    std::size_t store_bytes() const { return packed.size(); }

    double store_gb() const { return packed.size() / 1e9; }

    // Unpacked equivalent size (legacy u64 cells) for A/B logs.
    double unpacked_u64_gb() const {
        return hdr.m_cells * 8.0 / 1e9;
    }
};

namespace xor_detail {

inline std::uint64_t fingerprint(std::uint64_t hash, std::uint32_t r) {
    return hash & xor_fp_mask(r);
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

// [MEM-7] Peel stack is ~16n bytes (~62–72 GB at campaign N). Keep it in RAM for
// small builds. For large n: pre-size a scratch file, MAP_SHARED, and
// MADV_DONTNEED older pages during peel so page cache does not pin ~70 GB while
// setxor/counts are still live (that was the MEM-7 v1 bad_alloc at ~200 GiB).
inline constexpr std::size_t kPeelStackDiskThreshold = 50'000'000;  // ~800 MB stack

inline std::string peel_scratch_dir() {
    if (const char* e = std::getenv("XOR_PEEL_SCRATCH")) {
        if (e[0] != '\0') return std::string(e);
    }
    return "/mnt/scratch/iamreddave/615xor";
}

struct FilePeelStack {
    std::string path;
    int fd = -1;
    std::size_t capacity = 0;
    std::size_t count = 0;
    PeelEntry* map = nullptr;
    std::size_t map_bytes = 0;

    void create(const std::string& dir, std::size_t n_max) {
        capacity = n_max;
        map_bytes = n_max * sizeof(PeelEntry);
        path = dir + "/xor_peel_" + std::to_string(static_cast<long long>(::getpid())) + ".stk";
        fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) {
            throw std::runtime_error("MEM-7: cannot create peel stack file " + path +
                                     " (set XOR_PEEL_SCRATCH?)");
        }
        if (::ftruncate(fd, (off_t)map_bytes) != 0) {
            throw std::runtime_error("MEM-7: ftruncate peel stack failed (" +
                                     std::to_string(map_bytes) + " bytes)");
        }
        void* p = ::mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            throw std::runtime_error("MEM-7: mmap peel stack failed");
        }
        map = static_cast<PeelEntry*>(p);
    }

    void push(const PeelEntry& e) {
        if (count >= capacity) {
            throw std::runtime_error("MEM-7: peel stack overflow");
        }
        map[count++] = e;
        // Evict older pages from RSS (Linux). Keep a small hot tail.
#if defined(__linux__)
        constexpr std::size_t kChunk = (64ull << 20) / sizeof(PeelEntry);  // 64 MiB
        constexpr std::size_t kKeep = (256ull << 20) / sizeof(PeelEntry);  // 256 MiB hot
        if (count >= kKeep + kChunk && (count % kChunk) == 0) {
            const std::size_t drop_end = count - kKeep;
            const std::size_t drop_begin = drop_end - kChunk;
            ::madvise(reinterpret_cast<char*>(map) + drop_begin * sizeof(PeelEntry),
                      kChunk * sizeof(PeelEntry), MADV_DONTNEED);
        }
#endif
    }

    // After setxor/counts are freed: fault stack pages back for fingerprinting.
    void willneed_all() {
#if defined(__linux__)
        if (map && count) {
            ::madvise(map, count * sizeof(PeelEntry), MADV_WILLNEED);
        }
#endif
    }

    const PeelEntry& at(std::size_t i) const { return map[i]; }

    void close_unlink() {
        if (map && map_bytes) {
            ::munmap(map, map_bytes);
            map = nullptr;
            map_bytes = 0;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        if (!path.empty()) {
            ::unlink(path.c_str());
            path.clear();
        }
        capacity = 0;
        count = 0;
    }

    ~FilePeelStack() { close_unlink(); }
};

inline std::size_t capacity_for(std::size_t n) {
    // 1.23 * n, rounded up to multiple of 3; +32 slack like reference impls.
    double need = 32.0 + 1.23 * static_cast<double>(n);
    std::size_t cap = static_cast<std::size_t>(need + 0.999);
    if (cap < 3) cap = 3;
    cap = (cap + 2) / 3 * 3;
    return cap;
}

// [MEM-4] No longer used by construct_with_seed (fingerprints are written
// straight into `packed`). Kept for other translation units in the tree.
inline void pack_from_cells(XorFilter& out, const std::vector<std::uint64_t>& cells,
                            std::uint32_t r) {
    const std::size_t nbytes = xor_packed_bytes(cells.size(), r);
    out.packed.assign(nbytes, 0);
    for (std::size_t i = 0; i < cells.size(); ++i)
        xor_store_cell(out.packed.data(), i, r, cells[i]);
}

// [MEM-5]/MEM-6] Keys are only needed for the incidence loop.
// MEM-6: free them *before* allocating the peel stack (~8n bytes, e.g. ~36 GB
// at N=95238). That cuts the peak that OOMs at ~230 GB on a 283 GB host.
// On peel failure keys are already gone — build_xor cannot retry in-place and
// must regenerate the key vector (rare with load factor 1.23).
inline bool construct_with_seed(std::vector<std::uint64_t>& keys, std::uint32_t r,
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
        out.packed.clear();
        return true;
    }

    const std::size_t array_len = capacity_for(n);
    const std::uint32_t block = static_cast<std::uint32_t>(array_len / 3);

    std::vector<std::uint64_t> setxor(array_len, 0);
    // [MEM-3] u16 instead of u32. Incidence counts are Poisson with mean 3/1.23
    // ~ 2.4; 65535 is unreachable for any realistic key set, but guard anyway
    // rather than wrap silently.
    std::vector<std::uint16_t> counts(array_len, 0);
    // [MEM-1] hashes_v deleted — it was write-only.

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t h = key_hash(keys[i], seed);
        std::uint32_t hs[3];
        hashes(h, block, hs);
        for (int t = 0; t < 3; ++t) {
            setxor[hs[t]] ^= h;
            if (counts[hs[t]] == 0xffffu)
                throw std::runtime_error("build_xor: cell incidence count overflowed u16");
            counts[hs[t]] += 1;
        }
    }

    // [MEM-6] Drop keys before stack/queue (~8n). Peel uses only setxor/counts.
    { std::vector<std::uint64_t>().swap(keys); }

    const bool disk_stack = (n >= kPeelStackDiskThreshold);
    std::vector<PeelEntry> ram_stack;
    FilePeelStack file_stack;
    if (disk_stack) {
        const std::string dir = peel_scratch_dir();
        std::fprintf(stderr,
                     "[xor] MEM-7: peel stack mmap+DONTNEED (~%.1f GB) under %s\n",
                     (n * sizeof(PeelEntry)) / 1e9, dir.c_str());
        file_stack.create(dir, n);
    } else {
        ram_stack.reserve(n);
    }

    // Do not reserve full array_len for the queue (~4*m bytes ≈ 22 GB at N=95k).
    std::vector<std::uint32_t> queue;
    queue.reserve(std::min(array_len, n + (n >> 3) + 4096));
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
        const PeelEntry ent{h, i};
        if (disk_stack) file_stack.push(ent);
        else ram_stack.push_back(ent);
        for (int t = 0; t < 3; ++t) {
            const std::uint32_t j = hs[t];
            setxor[j] ^= h;
            if (counts[j] > 0) {
                counts[j] -= 1;
                if (counts[j] == 1) queue.push_back(j);
            }
        }
    }

    const std::size_t stack_n = disk_stack ? file_stack.count : ram_stack.size();
    if (stack_n != n) {
        if (disk_stack) file_stack.close_unlink();
        return false;  // peel stalled; keys already freed (MEM-6)
    }

    // Free the peel scratch we no longer need before allocating the output.
    { std::vector<std::uint64_t>().swap(setxor); }
    { std::vector<std::uint16_t>().swap(counts); }
    { std::vector<std::uint32_t>().swap(queue); }

    if (disk_stack) file_stack.willneed_all();

    // [MEM-4] Assign fingerprints directly into the bit-packed buffer. The old
    // code staged them in a u64 array the same length as the output (8 bytes
    // per cell instead of r/8) and packed afterwards. Reading a neighbour cell
    // through xor_load_cell is exact, so the result is identical.
    const std::uint64_t mask = xor_fp_mask(r);
    out.packed.assign(xor_packed_bytes(array_len, r), 0);

    for (std::size_t k = stack_n; k-- > 0;) {
        const PeelEntry& e = disk_stack ? file_stack.at(k) : ram_stack[k];
        std::uint32_t hs[3];
        hashes(e.hash, block, hs);
        std::uint64_t xorv = fingerprint(e.hash, r);
        for (int t = 0; t < 3; ++t) {
            if (hs[t] != e.index) xorv ^= xor_load_cell(out.packed.data(), hs[t], r);
        }
        xor_store_cell(out.packed.data(), e.index, r, xorv & mask);
    }

    if (disk_stack) file_stack.close_unlink();

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
    out.hdr.reserved = 0;  // packed layout (version >= 2)
    return true;  // [MEM-4] packed was filled in place above
}

}  // namespace xor_detail

// Dedup keys (stable-unique after sort). Xor construction requires a set.
inline std::vector<std::uint64_t> xor_unique_keys(std::vector<std::uint64_t> keys) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

// Build xor filter. Throws if construction fails after max_seed_tries.
// [MEM-2] Takes the key vector BY VALUE and moves it through the dedup step, so
// only one copy is ever live. Call it as build_xor(std::move(keys), r).
// [MEM-6] Keys are released after incidence (before peel stack). A peel stall
// then throws — regenerate keys and call again (see xor_build_pairs outer loop).
inline XorFilter build_xor(std::vector<std::uint64_t> keys_in, int rank = 48,
                           std::uint64_t base_seed = 0x615615615615615ULL,
                           int max_seed_tries = 64) {
    if (rank <= 0 || rank > 64) {
        throw std::invalid_argument("build_xor: rank must be in 1..64");
    }
    std::vector<std::uint64_t> keys = xor_unique_keys(std::move(keys_in));
    const std::size_t n0 = keys.size();
    XorFilter f;
    for (int try_i = 0; try_i < max_seed_tries; ++try_i) {
        if (keys.empty() && n0 > 0) {
            throw std::runtime_error(
                "build_xor: peel stalled after MEM-6 freed keys (n=" + std::to_string(n0) +
                "); regenerate the key vector and call build_xor again");
        }
        const std::uint64_t seed = base_seed + static_cast<std::uint64_t>(try_i) * 0x9E3779B97F4A7C15ULL;
        if (xor_detail::construct_with_seed(keys, static_cast<std::uint32_t>(rank), seed, f)) {
            return f;
        }
        // Peel stalled: keys already freed (MEM-6). Next loop iteration throws.
    }
    throw std::runtime_error("build_xor: peel failed after " + std::to_string(max_seed_tries) +
                             " seeds (n=" + std::to_string(n0) + ")");
}

inline bool xor_might_contain(const XorFilter& f, std::uint64_t key) {
    if (f.hdr.n_keys == 0) return false;
    if (f.packed.empty() || f.hdr.m_cells < 3) return false;
    const std::uint32_t block = static_cast<std::uint32_t>(f.hdr.m_cells / 3);
    const std::uint64_t h = xor_detail::key_hash(key, f.hdr.mix_seed);
    std::uint32_t hs[3];
    xor_detail::hashes(h, block, hs);
    const std::uint64_t got = xor_load_cell(f.packed.data(), hs[0], f.hdr.r) ^
                              xor_load_cell(f.packed.data(), hs[1], f.hdr.r) ^
                              xor_load_cell(f.packed.data(), hs[2], f.hdr.r);
    return got == xor_detail::fingerprint(h, f.hdr.r);
}
