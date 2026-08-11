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
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__)
#include <malloc.h>
#endif

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
// Indices are u64: for large N, m=3*block exceeds 2^32 (N≳83600 for pair keys).
inline void hashes(std::uint64_t hash, std::uint64_t block, std::uint64_t out[3]) {
    const std::uint64_t h0 = hash;
    const std::uint64_t h1 = mix64(hash ^ 0x9E3779B97F4A7C15ULL);
    const std::uint64_t h2 = mix64(hash ^ 0xBF58476D1CE4E5B9ULL);
    out[0] = (h0 % block);
    out[1] = (h1 % block) + block;
    out[2] = (h2 % block) + 2ull * block;
}

inline std::uint64_t key_hash(std::uint64_t key, std::uint64_t seed) {
    return mix64_seeded(key, seed);
}

struct PeelEntry {
    std::uint64_t hash;
    std::uint64_t index;  // alone cell (u64: m can exceed 2^32)
};
static_assert(sizeof(PeelEntry) == 16, "PeelEntry must stay 16 bytes for disk stack");

// [MEM-7c] Peel stack ~16n bytes. For large n: buffered write()+fadvise (never
// mmap the full stack). Fingerprint uses reverse pread chunks so peak stays
// ~setxor+counts during peel, then ~packed only during fingerprint.
inline constexpr std::size_t kPeelStackDiskThreshold = 50'000'000;

inline std::string peel_scratch_dir() {
    if (const char* e = std::getenv("XOR_PEEL_SCRATCH")) {
        if (e[0] != '\0') return std::string(e);
    }
    return "/mnt/scratch/iamreddave/615xor";
}

struct FilePeelStack {
    std::string path;
    int fd = -1;
    std::size_t count = 0;
    off_t written_bytes = 0;
    std::vector<PeelEntry> wbuf;
    // Reverse-scan cache for fingerprint (mutable — logical const at()).
    mutable std::vector<PeelEntry> rbuf;
    mutable std::size_t rbuf_lo = 0;
    mutable std::size_t rbuf_hi = 0;
    static constexpr std::size_t kWbufCap = (1ull << 20) / sizeof(PeelEntry);  // ~1 MiB

    void create(const std::string& dir, std::size_t /*n_max*/) {
        path = dir + "/xor_peel_" + std::to_string(static_cast<long long>(::getpid())) + ".stk";
        fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) {
            throw std::runtime_error("MEM-7: cannot create peel stack file " + path +
                                     " (set XOR_PEEL_SCRATCH?)");
        }
        wbuf.reserve(kWbufCap);
    }

    void flush_wbuf() {
        if (wbuf.empty()) return;
        const std::size_t nbytes = wbuf.size() * sizeof(PeelEntry);
        const char* p = reinterpret_cast<const char*>(wbuf.data());
        std::size_t left = nbytes;
        while (left) {
            const ssize_t n = ::write(fd, p, left);
            if (n <= 0) {
                throw std::runtime_error("MEM-7: short write to peel stack " + path);
            }
            p += n;
            left -= (std::size_t)n;
        }
        written_bytes += (off_t)nbytes;
        wbuf.clear();
#if defined(__linux__)
        ::posix_fadvise(fd, 0, written_bytes, POSIX_FADV_DONTNEED);
#endif
    }

    void push(const PeelEntry& e) {
        wbuf.push_back(e);
        ++count;
        if (wbuf.size() >= kWbufCap) flush_wbuf();
    }

    void prepare_fingerprint() {
        flush_wbuf();
        rbuf.assign(kWbufCap, PeelEntry{});
        rbuf_lo = rbuf_hi = 0;
#if defined(__linux__)
        ::posix_fadvise(fd, 0, written_bytes, POSIX_FADV_DONTNEED);
#endif
    }

    const PeelEntry& at(std::size_t i) const {
        if (i < rbuf_lo || i >= rbuf_hi) {
            // Load a chunk ending just past i (reverse scan).
            const std::size_t end = i + 1;
            const std::size_t begin = (end > kWbufCap) ? (end - kWbufCap) : 0;
            const std::size_t nent = end - begin;
            const off_t off = (off_t)(begin * sizeof(PeelEntry));
            const std::size_t nbytes = nent * sizeof(PeelEntry);
            char* p = reinterpret_cast<char*>(rbuf.data());
            std::size_t got = 0;
            while (got < nbytes) {
                const ssize_t n = ::pread(fd, p + got, nbytes - got, off + (off_t)got);
                if (n <= 0) {
                    throw std::runtime_error("MEM-7: short pread peel stack");
                }
                got += (std::size_t)n;
            }
            rbuf_lo = begin;
            rbuf_hi = end;
        }
        return rbuf[i - rbuf_lo];
    }

    void close_unlink() {
        wbuf.clear();
        wbuf.shrink_to_fit();
        rbuf.clear();
        rbuf.shrink_to_fit();
        rbuf_lo = rbuf_hi = 0;
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        if (!path.empty()) {
            ::unlink(path.c_str());
            path.clear();
        }
        count = 0;
        written_bytes = 0;
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

inline void log_vm_rss(const char* tag) {
#if defined(__linux__)
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (std::fgets(line, sizeof line, f)) {
        long kb = 0;
        if (std::sscanf(line, "VmRSS: %ld", &kb) == 1) {
            std::fprintf(stderr, "[xor] RSS %-20s %6.1f GB\n", tag, kb / (1024.0 * 1024.0));
            break;
        }
    }
    std::fclose(f);
#else
    (void)tag;
#endif
}

inline void release_heap_to_os() {
#if defined(__linux__)
    ::malloc_trim(0);
#endif
}

// Spill key vector to scratch, free RAM, return path. Caller deletes file.
inline std::string spill_keys_to_scratch(std::vector<std::uint64_t>& keys) {
    const std::string dir = peel_scratch_dir();
    const std::string path =
        dir + "/xor_keys_" + std::to_string(static_cast<long long>(::getpid())) + ".bin";
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("MEM-8: cannot write key spill " + path +
                                 " (set XOR_PEEL_SCRATCH?)");
    }
    const std::size_t n = keys.size();
    if (n && std::fwrite(keys.data(), sizeof(std::uint64_t), n, fp) != n) {
        std::fclose(fp);
        ::unlink(path.c_str());
        throw std::runtime_error("MEM-8: short write key spill");
    }
    std::fclose(fp);
    { std::vector<std::uint64_t>().swap(keys); }
    release_heap_to_os();
    std::fprintf(stderr, "[xor] MEM-8: spilled %zu keys (%.1f GB) to %s\n", n,
                 (n * sizeof(std::uint64_t)) / 1e9, path.c_str());
    return path;
}

// [MEM-5/6/8] Keys only for incidence. For large n: spill keys to scratch (MEM-8)
// before allocating setxor/counts so those never overlap the ~36 GB key buffer.
// MEM-6: keys gone before peel stack. MEM-7c: peel stack write+fadvise + pread.
// On peel failure keys are already gone — build_xor regenerates (rare).
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
    const std::uint64_t block = static_cast<std::uint64_t>(array_len / 3);
    if (array_len > (std::size_t)UINT32_MAX) {
        std::fprintf(stderr,
                     "[xor] IDX-64: m_cells=%zu (>2^32) — using u64 cell indices\n",
                     array_len);
    }
    const bool spill_keys = (n >= kPeelStackDiskThreshold);

    log_vm_rss("after-unique-keys");

    std::string key_path;
    if (spill_keys) {
        key_path = spill_keys_to_scratch(keys);
        log_vm_rss("after-key-spill");
    }

    std::vector<std::uint64_t> setxor(array_len, 0);
    // [MEM-3] u16 instead of u32.
    std::vector<std::uint16_t> counts(array_len, 0);
    log_vm_rss("after-setxor-counts");

    if (spill_keys) {
        std::FILE* fp = std::fopen(key_path.c_str(), "rb");
        if (!fp) {
            throw std::runtime_error("MEM-8: cannot re-read key spill " + key_path);
        }
        constexpr std::size_t kBuf = 1 << 20;  // 1M keys ~8 MB
        std::vector<std::uint64_t> buf(kBuf);
        std::size_t seen = 0;
        while (seen < n) {
            const std::size_t want = std::min(kBuf, n - seen);
            if (std::fread(buf.data(), sizeof(std::uint64_t), want, fp) != want) {
                std::fclose(fp);
                throw std::runtime_error("MEM-8: short read key spill");
            }
            for (std::size_t i = 0; i < want; ++i) {
                const std::uint64_t h = key_hash(buf[i], seed);
                std::uint64_t hs[3];
                hashes(h, block, hs);
                for (int t = 0; t < 3; ++t) {
                    setxor[hs[t]] ^= h;
                    if (counts[hs[t]] == 0xffffu)
                        throw std::runtime_error("build_xor: cell incidence count overflowed u16");
                    counts[hs[t]] += 1;
                }
            }
            seen += want;
        }
        std::fclose(fp);
        ::unlink(key_path.c_str());
        key_path.clear();
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t h = key_hash(keys[i], seed);
            std::uint64_t hs[3];
            hashes(h, block, hs);
            for (int t = 0; t < 3; ++t) {
                setxor[hs[t]] ^= h;
                if (counts[hs[t]] == 0xffffu)
                    throw std::runtime_error("build_xor: cell incidence count overflowed u16");
                counts[hs[t]] += 1;
            }
        }
        { std::vector<std::uint64_t>().swap(keys); }
        release_heap_to_os();
    }

    log_vm_rss("after-incidence");

    const bool disk_stack = (n >= kPeelStackDiskThreshold);
    std::vector<PeelEntry> ram_stack;
    FilePeelStack file_stack;
    if (disk_stack) {
        const std::string dir = peel_scratch_dir();
        std::fprintf(stderr,
                     "[xor] MEM-7: peel stack write+fadvise (~%.1f GB) under %s\n",
                     (n * sizeof(PeelEntry)) / 1e9, dir.c_str());
        file_stack.create(dir, n);
    } else {
        ram_stack.reserve(n);
    }

    // Small starting queue; grows as needed (avoid a 20 GB reserve).
    // u64 indices: a uint32 loop here infinite-loops when array_len > 2^32.
    std::vector<std::uint64_t> queue;
    queue.reserve(1 << 20);
    for (std::size_t i = 0; i < array_len; ++i) {
        if (counts[i] == 1) queue.push_back(static_cast<std::uint64_t>(i));
    }

    std::size_t qhead = 0;
    while (qhead < queue.size()) {
        const std::uint64_t i = queue[qhead++];
        if (counts[i] != 1) continue;
        const std::uint64_t h = setxor[i];
        std::uint64_t hs[3];
        hashes(h, block, hs);
        const PeelEntry ent{h, i};
        if (disk_stack) file_stack.push(ent);
        else ram_stack.push_back(ent);
        for (int t = 0; t < 3; ++t) {
            const std::uint64_t j = hs[t];
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
        return false;
    }

    { std::vector<std::uint64_t>().swap(setxor); }
    { std::vector<std::uint16_t>().swap(counts); }
    { std::vector<std::uint64_t>().swap(queue); }
    release_heap_to_os();
    log_vm_rss("after-peel-frees");

    if (disk_stack) file_stack.prepare_fingerprint();
    log_vm_rss("after-stack-prepare");

    const std::uint64_t mask = xor_fp_mask(r);
    out.packed.assign(xor_packed_bytes(array_len, r), 0);
    log_vm_rss("after-packed-alloc");

    for (std::size_t k = stack_n; k-- > 0;) {
        const PeelEntry& e = disk_stack ? file_stack.at(k) : ram_stack[k];
        std::uint64_t hs[3];
        hashes(e.hash, block, hs);
        std::uint64_t xorv = fingerprint(e.hash, r);
        for (int t = 0; t < 3; ++t) {
            if (hs[t] != e.index) xorv ^= xor_load_cell(out.packed.data(), hs[t], r);
        }
        xor_store_cell(out.packed.data(), e.index, r, xorv & mask);
    }

    if (disk_stack) file_stack.close_unlink();
    log_vm_rss("done-construct");

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
    const std::uint64_t block = f.hdr.m_cells / 3;
    const std::uint64_t h = xor_detail::key_hash(key, f.hdr.mix_seed);
    std::uint64_t hs[3];
    xor_detail::hashes(h, block, hs);
    const std::uint64_t got = xor_load_cell(f.packed.data(), hs[0], f.hdr.r) ^
                              xor_load_cell(f.packed.data(), hs[1], f.hdr.r) ^
                              xor_load_cell(f.packed.data(), hs[2], f.hdr.r);
    return got == xor_detail::fingerprint(h, f.hdr.r);
}
