#pragma once
// Shared host xor pair-store helpers for fourcore_*_v4 (post-i128 + xor).
// Device query lives in each .cu (Hit shapes differ).
// Keys: (i^6+j^6) mod 2^64 for 1<=i<=j<=N — same as solve_516_v4.

#include "spike/ribbon/xor_filter.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using fc_xor_u64 = std::uint64_t;
using fc_xor_u32 = std::uint32_t;
using fc_xor_u128 = unsigned __int128;

struct PairRecover {
  int N = 0;
  std::vector<fc_xor_u64> pw6;
  std::unordered_map<fc_xor_u64, std::vector<fc_xor_u32>> by6;

  void build(int n) {
    N = n;
    pw6.assign((size_t)n + 1, 0);
#pragma omp parallel for schedule(static)
    for (int x = 1; x <= n; ++x) {
      fc_xor_u64 x2 = (fc_xor_u64)x * (fc_xor_u64)x;
      pw6[(size_t)x] = x2 * x2 * x2;
    }
    by6.clear();
    by6.reserve((size_t)n * 2);
    for (int x = 1; x <= n; ++x) by6[pw6[(size_t)x]].push_back((fc_xor_u32)x);
  }

  void recover(fc_xor_u64 fp, std::vector<std::pair<fc_xor_u32, fc_xor_u32>>& out) const {
    out.clear();
    for (int i = 1; i <= N; ++i) {
      const fc_xor_u64 need = fp - pw6[(size_t)i];
      auto it = by6.find(need);
      if (it == by6.end()) continue;
      for (fc_xor_u32 j : it->second)
        if ((fc_xor_u32)i <= j) out.emplace_back((fc_xor_u32)i, j);
    }
  }
};

// Host FPR smoke test for a loaded packed xor (should be ~0 hits in 200k random keys at r=48).
inline int xor_fpr_smoke(const XorFilter& f, int trials = 200000) {
  if (f.packed.empty() || f.hdr.m_cells < 3 || f.hdr.r == 0) return -1;
  const fc_xor_u32 block = (fc_xor_u32)(f.hdr.m_cells / 3);
  int hits = 0;
  fc_xor_u64 x = 0x123456789abcdef0ULL;
  for (int t = 0; t < trials; ++t) {
    x = x * 0x9E3779B97F4A7C15ULL + 1;
    if (xor_might_contain(f, (std::uint64_t)x)) ++hits;
  }
  std::fprintf(stderr, "[xor] FPR smoke: %d/%d random keys hit (expect ~0 at r=%u)\n", hits, trials,
               f.hdr.r);
  return hits;
}

inline XorFilter xor_build_pairs(int N, int r) {
  const double pairs_est = (double)N * (N + 1) / 2.0;
  fprintf(stderr, "[xor] building keys N=%d pairs=%.3e r=%d ...\n", N, pairs_est, r);
  std::vector<fc_xor_u64> pw6((size_t)N + 1);
#pragma omp parallel for schedule(static)
  for (int x = 1; x <= N; ++x) {
    fc_xor_u64 x2 = (fc_xor_u64)x * (fc_xor_u64)x;
    pw6[(size_t)x] = x2 * x2 * x2;
  }

  auto make_keys = [&]() {
    std::vector<std::uint64_t> keys;
    keys.reserve((size_t)pairs_est + 16);
    for (int i = 1; i <= N; ++i)
      for (int j = i; j <= N; ++j)
        keys.push_back((std::uint64_t)pw6[(size_t)i] + (std::uint64_t)pw6[(size_t)j]);
    return keys;
  };

  // [MEM-6] Peel stall frees keys inside build_xor; regenerate pair keys and
  // bump the seed base (rare with capacity 1.23).
  XorFilter f;
  const auto t0 = std::chrono::steady_clock::now();
  constexpr int kOuter = 4;
  for (int outer = 0; outer < kOuter; ++outer) {
    try {
      const std::uint64_t base =
          0x615615615615615ULL + (std::uint64_t)outer * 0x9E3779B97F4A7C15ULL;
      f = build_xor(make_keys(), r, base);
      break;
    } catch (const std::runtime_error& e) {
      const char* msg = e.what();
      const bool mem6 = msg && std::strstr(msg, "MEM-6") != nullptr;
      if (!mem6 || outer + 1 >= kOuter) throw;
      fprintf(stderr, "[xor] %s — regenerating keys (outer %d/%d)\n", msg, outer + 1, kOuter);
    }
  }
  f.hdr.N = (std::uint32_t)N;
  const double ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
          .count();
  const double gb = f.store_gb();
  const double u64_gb = f.unpacked_u64_gb();
  fprintf(stderr,
          "[xor] N=%d keys=%llu cells=%llu packed=%.2f GB (u64-equiv=%.2f GB) r=%d "
          "seed=0x%llx build=%.0f ms\n",
          N, (unsigned long long)f.hdr.n_keys, (unsigned long long)f.hdr.m_cells, gb, u64_gb, r,
          (unsigned long long)f.hdr.mix_seed, ms);
  if (gb > 20.0)
    fprintf(stderr, "[xor] warning: store %.2f GB > 20 GB — tight on 24 GB cards (leave headroom)\n",
            gb);
  return f;
}

// Soft campaign cap: packed xor ~1.23 * pairs * r bits. N=120k r=48 ≈ 53 GB.
inline constexpr int kXorNSoftMax = 120000;

// Persist packed xor (StoreHeader v2 + packed bytes). Same layout as xor_build_save /
// solve_516_v4 --save-table.
inline bool xor_save_file(const char* path, const XorFilter& f) {
  FILE* fp = std::fopen(path, "wb");
  if (!fp) {
    std::fprintf(stderr, "[xor] cannot open %s for write\n", path);
    return false;
  }
  if (std::fwrite(&f.hdr, sizeof(f.hdr), 1, fp) != 1 ||
      (!f.packed.empty() &&
       std::fwrite(f.packed.data(), 1, f.packed.size(), fp) != f.packed.size())) {
    std::fprintf(stderr, "[xor] short write %s\n", path);
    std::fclose(fp);
    return false;
  }
  std::fclose(fp);
  std::fprintf(stderr, "[xor] saved %s (%.2f GB packed, N=%u, r=%u)\n", path, f.store_gb(),
               f.hdr.N, f.hdr.r);
  return true;
}

// Load packed xor. Requires file N == expected_N (pass expected_N<=0 to accept any N).
// Returns false on missing/bad file or N mismatch (caller may rebuild).
inline bool xor_load_file(const char* path, XorFilter& f, int expected_N) {
  FILE* fp = std::fopen(path, "rb");
  if (!fp) {
    std::fprintf(stderr, "[xor] cannot open %s for read\n", path);
    return false;
  }
  if (std::fread(&f.hdr, sizeof(f.hdr), 1, fp) != 1 || !store615::header_ok(f.hdr) ||
      f.hdr.kind != (std::uint32_t)store615::Kind::Xor) {
    std::fclose(fp);
    std::fprintf(stderr, "[xor] bad file %s\n", path);
    return false;
  }
  if (f.hdr.version != store615::kVersion) {
    std::fclose(fp);
    std::fprintf(stderr, "[xor] unsupported store version %u (need %u packed)\n", f.hdr.version,
                 store615::kVersion);
    return false;
  }
  if (expected_N > 0 && (int)f.hdr.N != expected_N) {
    std::fclose(fp);
    std::fprintf(stderr, "[xor] N mismatch: file N=%u, need %d\n", f.hdr.N, expected_N);
    return false;
  }
  const size_t want = xor_packed_bytes(f.hdr.m_cells, f.hdr.r);
  f.packed.assign(want, 0);
  if (want && std::fread(f.packed.data(), 1, want, fp) != want) {
    std::fclose(fp);
    std::fprintf(stderr, "[xor] short read %s\n", path);
    return false;
  }
  std::fclose(fp);
  std::fprintf(stderr, "[xor] loaded %s (N=%u, cells=%llu, r=%u, %.2f GB packed)\n", path,
               f.hdr.N, (unsigned long long)f.hdr.m_cells, f.hdr.r, f.store_gb());
  return true;
}
