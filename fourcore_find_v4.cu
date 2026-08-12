// =============================================================================
// fourcore_find_v4.cu — post-i128 find4/find3/find2 with xor pair store
//
// Fork of fourcore_find_v3.cu: same GMP peels / jobs / kernels, but the OA
// Slot table is replaced by a GPU xor filter (solve_516_v4 / spike/ribbon).
// Hits carry fp; host PairRecover fills (a,b) before exact+GMP verify.
//
// N may exceed 65535 (xor has no u16 payload). Soft cap ~120k (~71 GB cells).
//
// Build:
//   make fourcore-find-v4
// Host-only smoke:
//   make fourcore-find-v4-host && ./fourcore_find_v4_host --selftest-host
// =============================================================================

#include "fourcore_classes_v3.hpp"
#include "fourcore_gmp.hpp"
#include "fourcore_job_bin.hpp"
#include "fourcore_xor_store.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef HOST_ONLY
#include <cuda_runtime.h>
#define CU(x) do { cudaError_t _e=(x); if((_e)!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
  exit(1); } } while(0)
#endif

using Clock = std::chrono::steady_clock;
using u16 = std::uint16_t;

// ------------------------------ gate ---------------------------------------
struct GateData {
  u64 g504[8] = {0};
  u64 g247[4] = {0};
  u16 p6504[504];
  u16 p6247[247];
};
static inline void gate_set(u64* b, int x) { b[x >> 6] |= 1ULL << (x & 63); }
static inline bool gate_get(const u64* b, int x) {
  return (b[x >> 6] >> (x & 63)) & 1ULL;
}
static inline u32 mod_pow6_h(u32 x, u32 m) {
  u64 a = x;
  u64 x2 = (a * a) % m;
  u64 x4 = (x2 * x2) % m;
  return (u32)((x2 * x4) % m);
}
static void build_gate(GateData& g) {
  for (int i = 0; i < 504; ++i) {
    const int si = (int)mod_pow6_h((u32)i, 504);
    g.p6504[i] = (u16)si;
    for (int j = i; j < 504; ++j)
      gate_set(g.g504, (si + (int)mod_pow6_h((u32)j, 504)) % 504);
  }
  for (int i = 0; i < 247; ++i) {
    const int si = (int)mod_pow6_h((u32)i, 247);
    g.p6247[i] = (u16)si;
    for (int j = i; j < 247; ++j)
      gate_set(g.g247, (si + (int)mod_pow6_h((u32)j, 247)) % 247);
  }
}
static bool gate_selftest_host(const GateData& g) {
  int c504 = 0, c247 = 0;
  for (int x = 0; x < 504; ++x) c504 += (int)gate_get(g.g504, x);
  for (int x = 0; x < 247; ++x) c247 += (int)gate_get(g.g247, x);
  if (c504 != 27 || c247 != 50) {
    fprintf(stderr, "gate sizes %d/%d want 27/50\n", c504, c247);
    return false;
  }
  printf("[gate] bitmaps 27/504 and 50/247 OK\n");
  return true;
}

// ------------------------------ jobs / table -------------------------------
struct Job {
  int cls = 0;
  u64 B = 0, u = 0;
  u64 free1 = 0, free2 = 0;  // d ; e (cls5)
  u128 T = 0;
  u32 lim = 0;
};

// Hit: a,b filled on host via PairRecover; c/d are peel indices; fp from GPU.
struct Hit { u32 job, a, b, c, d; u64 fp; };

struct Cand {
  u64 q_lo, q_hi;
  u32 lim;
  u32 lo, hi;       // find3/4 window; unused for find2
  u32 job;
  u32 T_r504, T_r247;
  u32 pad;
};

static inline u128 pow6_full(u64 x) {
  u128 a = (u128)x * (u128)x;
  return a * a * a;
}
static u32 iroot6_host(u128 n) {
  double x = ldexp((double)(u64)(n >> 64), 64) + (double)(u64)n;
  if (x < 1.0) return 0;
  double r = pow(x, 1.0 / 6.0);
  u64 a = (u64)r;
  while ((pow6_full(a + 1) <= n) && a < 0xffffffffULL - 2) ++a;
  while (pow6_full(a) > n && a) --a;
  return (u32)a;
}

static bool load_jobs_but(const char* path, std::vector<Job>& jobs) {
  FILE* f = fopen(path, "r");
  if (!f) return false;
  char buf[256];
  size_t bad = 0;
  while (fgets(buf, sizeof buf, f)) {
    int cls;
    unsigned long long B, u, f1, f2, tlo, thi;
    if (sscanf(buf, "%d %llu %llu %llu %llu %llu %llu",
               &cls, &B, &u, &f1, &f2, &tlo, &thi) != 7) {
      ++bad; continue;
    }
    Job J;
    J.cls = cls; J.B = B; J.u = u; J.free1 = f1; J.free2 = f2;
    J.T = join_u128(tlo, thi);
    J.lim = (u32)((B - 1) / 42);
    jobs.push_back(J);
  }
  fclose(f);
  if (bad) fprintf(stderr, "[jobs] skipped %zu bad lines\n", bad);
  return !jobs.empty();
}

static bool load_units_buc(const char* path, std::vector<Job>& units) {
  FILE* f = fopen(path, "r");
  if (!f) return false;
  char buf[256];
  while (fgets(buf, sizeof buf, f)) {
    int cls;
    unsigned long long B, u;
    if (sscanf(buf, "%d %llu %llu", &cls, &B, &u) != 3) continue;
    Job J;
    J.cls = cls; J.B = B; J.u = u; J.lim = (u32)((B - 1) / 42);
    units.push_back(J);
  }
  fclose(f);
  return !units.empty();
}

// Expand one Stage-1 unit into reduced-T jobs (may be huge for cls5).
static void expand_unit(const fc3::RootTables& rt, const Job& U,
                        std::vector<Job>& out, u64& failT,
                        u64 max_out, bool& hit_cap) {
  if (hit_cap) return;
  auto push = [&](Job J) {
    if (max_out && out.size() >= (size_t)max_out) { hit_cap = true; return; }
    out.push_back(J);
  };
  if (U.cls == 1) {
    u128 T = 0;
    if (!compute_T_gmp(U.B, U.u, 42, T)) { ++failT; return; }
    Job J = U; J.free1 = J.free2 = 0; J.T = T; push(J);
    return;
  }
  if (U.cls >= 2 && U.cls <= 4) {
    const int f = fc3::free_factor(U.cls);
    auto fs = fc3::free_d_cls234(rt, (long long)U.B, (long long)U.u, U.cls);
    fc3::for_each_free(fs, [&](long long d) {
      if (hit_cap) return;
      u128 T = 0;
      if (!compute_T_peel1_gmp(U.B, U.u, (u64)f, (u64)d, T)) { ++failT; return; }
      if (T == 0) return;
      Job J = U; J.free1 = (u64)d; J.free2 = 0; J.T = T; push(J);
    });
    return;
  }
  if (U.cls == 5) {
    fc3::FreeTermSpec es, ds;
    fc3::free_de_cls5(rt, (long long)U.B, (long long)U.u, es, ds);
    fc3::for_each_free(es, [&](long long e) {
      if (hit_cap) return;
      fc3::for_each_free(ds, [&](long long d) {
        if (hit_cap) return;
        u128 T = 0;
        if (!compute_T_peel2_gmp(U.B, U.u, (u64)d, (u64)e, T)) { ++failT; return; }
        if (T == 0) return;
        Job J = U; J.free1 = (u64)d; J.free2 = (u64)e; J.T = T; push(J);
      });
    });
  }
}

#ifndef HOST_ONLY
#include "fourcore_find_device.cuh"

// ------------------------------ device aliases (shared header) -------------
__device__ __forceinline__ u64 d_pow6_64(u32 x) { return fc_d_pow6_64(x); }
__device__ __forceinline__ void d_pow6_128(u32 x, u64& h, u64& l) { fc_d_pow6_128(x, h, l); }
__device__ __forceinline__ u32 d_iroot6(u64 rhi, u64 rlo) { return fc_d_iroot6(rhi, rlo); }

__device__ __forceinline__ bool d_xor_might_contain(const uint8_t* packed, u32 block, u32 r,
                                                     u64 seed, u64 key) {
  return fc_d_xor_might_contain(packed, block, r, seed, key);
}

__device__ int d_probe(const uint8_t* packed, u32 block, u32 r, u64 seed, u64 fp, u32 c_ctx,
                       u32 d_ctx, u32 job, Hit* hits, u32* nhit, u32 hit_cap) {
  return fc_d_probe(packed, block, r, seed, fp, c_ctx, d_ctx, job, hits, nhit, hit_cap);
}

struct GateSh { u64 g504[8]; u64 g247[4]; u16 p6504[504]; u16 p6247[247]; };
__device__ __forceinline__ bool d_gate_bit(const u64* b, int x) { return fc_d_gate_bit(b, x); }
__device__ __forceinline__ void gate_load_sh(GateSh& gs, const GateData* gd) {
  FcGateSh& fgs = *reinterpret_cast<FcGateSh*>(&gs);
  fc_gate_load_sh(fgs, gd);
}

// find2: one block per cand; thread 0 probes (or all race — single probe).
__global__ void k_find2(const Cand* __restrict__ cands, int nc,
                        const uint8_t* __restrict__ xor_cells, u32 xor_block, u32 xor_r, u64 xor_seed,
                        const GateData* __restrict__ gd, int use_gate,
                        Hit* __restrict__ hits, u32* __restrict__ nhit, u32 hit_cap,
                        u64* __restrict__ out_calls, u64* __restrict__ out_gated,
                        u64* __restrict__ out_probes) {
  const int ci = blockIdx.x;
  if (ci >= nc) return;
  if (threadIdx.x != 0) return;
  const Cand C = cands[ci];
  atomicAdd((unsigned long long*)out_calls, 1ull);
  if (use_gate) {
    if (!(d_gate_bit(gd->g504, (int)C.T_r504) && d_gate_bit(gd->g247, (int)C.T_r247))) {
      atomicAdd((unsigned long long*)out_gated, 1ull);
      return;
    }
  }
  atomicAdd((unsigned long long*)out_probes, 1ull);
  d_probe(xor_cells, xor_block, xor_r, xor_seed, C.q_lo, C.lim, 0, C.job, hits, nhit, hit_cap);
}

// find3: block per cand; threads stride c3 in [lo,hi]
__global__ void k_find3(const Cand* __restrict__ cands, int nc,
                        const uint8_t* __restrict__ xor_cells, u32 xor_block, u32 xor_r, u64 xor_seed,
                        const GateData* __restrict__ gd, int use_gate,
                        Hit* __restrict__ hits, u32* __restrict__ nhit, u32 hit_cap,
                        u64* __restrict__ out_calls, u64* __restrict__ out_gated,
                        u64* __restrict__ out_probes) {
  const u32 ci = blockIdx.x;
  if ((int)ci >= nc) return;
  const Cand C = cands[ci];
  __shared__ GateSh gs;
  if (use_gate) gate_load_sh(gs, gd);
  u64 lcalls = 0, lgated = 0, lprobes = 0;
  for (u32 c3 = C.lo + threadIdx.x; c3 <= C.hi; c3 += blockDim.x) {
    ++lcalls;
    if (use_gate) {
      u32 r504 = C.T_r504 + 504u - gs.p6504[c3 % 504];
      if (r504 >= 504u) r504 -= 504u;
      u32 r247 = C.T_r247 + 247u - gs.p6247[c3 % 247];
      if (r247 >= 247u) r247 -= 247u;
      if (!(d_gate_bit(gs.g504, (int)r504) && d_gate_bit(gs.g247, (int)r247))) {
        ++lgated; continue;
      }
    }
    ++lprobes;
    d_probe(xor_cells, xor_block, xor_r, xor_seed, C.q_lo - d_pow6_64(c3), c3, 0, C.job, hits, nhit, hit_cap);
  }
  atomicAdd((unsigned long long*)out_calls, (unsigned long long)lcalls);
  atomicAdd((unsigned long long*)out_gated, (unsigned long long)lgated);
  atomicAdd((unsigned long long*)out_probes, (unsigned long long)lprobes);
}

// find4: same structure as fourcore_find4_v2
__global__ void k_find4(const Cand* __restrict__ cands, int nc,
                        const uint8_t* __restrict__ xor_cells, u32 xor_block, u32 xor_r, u64 xor_seed,
                        const GateData* __restrict__ gd, int use_gate,
                        Hit* __restrict__ hits, u32* __restrict__ nhit, u32 hit_cap,
                        u64* __restrict__ out_calls, u64* __restrict__ out_gated,
                        u64* __restrict__ out_probes) {
  const u32 ci = blockIdx.x;
  if ((int)ci >= nc) return;
  const Cand C = cands[ci];
  __shared__ GateSh gs;
  if (use_gate) gate_load_sh(gs, gd);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const u32 base = C.lo + blockIdx.y * 2048u + (u32)warp;
  u64 lcalls = 0, lgated = 0, lprobes = 0;
  for (u32 c4 = base; c4 <= C.hi; c4 += 8) {
    if (c4 < 1 || c4 > C.lim) continue;
    u64 c4h, c4l; d_pow6_128(c4, c4h, c4l);
    const u64 rlo = C.q_lo - c4l;
    const u64 rhi = C.q_hi - c4h - (C.q_lo < c4l ? 1ull : 0ull);
    if ((rhi >> 63) || ((rhi | rlo) == 0)) continue;
    u32 hi3 = d_iroot6(rhi, rlo);
    const u32 capm = C.lim < c4 ? C.lim : c4;
    if (hi3 > capm) hi3 = capm;
    if (hi3 < 1) continue;
    u32 lo3 = (u32)(hi3 * 0.8);
    if (lo3 < 1) lo3 = 1;
    if (lo3 > hi3) continue;
    const u64 tfp = C.q_lo - d_pow6_64(c4);
    u32 r4_504 = 0, r4_247 = 0;
    if (use_gate) {
      r4_504 = gs.p6504[c4 % 504];
      r4_247 = gs.p6247[c4 % 247];
    }
    for (u32 c3 = lo3 + (u32)lane; c3 <= hi3; c3 += 32) {
      ++lcalls;
      if (use_gate) {
        u32 r504 = C.T_r504 + 1008u - r4_504 - gs.p6504[c3 % 504];
        while (r504 >= 504u) r504 -= 504u;
        u32 r247 = C.T_r247 + 494u - r4_247 - gs.p6247[c3 % 247];
        while (r247 >= 247u) r247 -= 247u;
        if (!(d_gate_bit(gs.g504, (int)r504) && d_gate_bit(gs.g247, (int)r247))) {
          ++lgated; continue;
        }
      }
      ++lprobes;
      d_probe(xor_cells, xor_block, xor_r, xor_seed, tfp - d_pow6_64(c3), c3, c4, C.job, hits, nhit, hit_cap);
    }
  }
  atomicAdd((unsigned long long*)out_calls, (unsigned long long)lcalls);
  atomicAdd((unsigned long long*)out_gated, (unsigned long long)lgated);
  atomicAdd((unsigned long long*)out_probes, (unsigned long long)lprobes);
}

struct GpuState {
  uint8_t* d_xor = nullptr;
  u32 xor_block = 0;
  u32 xor_r = 48;
  u64 xor_seed = 0;
  GateData* d_gd = nullptr;
  Hit* d_hits = nullptr;
  u32* d_nh = nullptr;
  u64* d_calls = nullptr;
  u64* d_gated = nullptr;
  u64* d_probes = nullptr;
  Cand* d_cands = nullptr;
  size_t cand_cap = 0;
  u32 hit_cap = 1u << 20;
  bool use_gate = true;
  bool quiet = false;
  u64 tot_calls = 0, tot_gated = 0, tot_probes = 0;
  PairRecover* pair_ix = nullptr;  // host-owned, not freed here
};

static void gpu_init(GpuState& g, const XorFilter& xf, bool use_gate, bool quiet = false) {
  g.xor_block = xf.hdr.m_cells ? (u32)(xf.hdr.m_cells / 3) : 0;
  g.xor_r = xf.hdr.r;
  g.xor_seed = xf.hdr.mix_seed;
  g.use_gate = use_gate;
  g.quiet = quiet;
  g.tot_calls = g.tot_gated = g.tot_probes = 0;
  if (!xf.packed.empty()) {
    CU(cudaMalloc(&g.d_xor, xf.packed.size()));
    CU(cudaMemcpy(g.d_xor, xf.packed.data(), xf.packed.size(), cudaMemcpyHostToDevice));
  }
  fprintf(stderr, "[gpu] xor uploaded (%.2f GB packed, r=%u, block=%u)\n",
          xf.store_gb(), g.xor_r, g.xor_block);
  GateData gd; build_gate(gd);
  CU(cudaMalloc(&g.d_gd, sizeof(gd)));
  CU(cudaMemcpy(g.d_gd, &gd, sizeof(gd), cudaMemcpyHostToDevice));
  CU(cudaMalloc(&g.d_hits, g.hit_cap * sizeof(Hit)));
  CU(cudaMalloc(&g.d_nh, sizeof(u32)));
  CU(cudaMalloc(&g.d_calls, sizeof(u64)));
  CU(cudaMalloc(&g.d_gated, sizeof(u64)));
  CU(cudaMalloc(&g.d_probes, sizeof(u64)));
}

static void gpu_free(GpuState& g) {
  if (g.d_xor) cudaFree(g.d_xor);
  if (g.d_gd) cudaFree(g.d_gd);
  if (g.d_hits) cudaFree(g.d_hits);
  if (g.d_nh) cudaFree(g.d_nh);
  if (g.d_calls) cudaFree(g.d_calls);
  if (g.d_gated) cudaFree(g.d_gated);
  if (g.d_probes) cudaFree(g.d_probes);
  if (g.d_cands) cudaFree(g.d_cands);
}

static void ensure_cands(GpuState& g, size_t n) {
  if (n <= g.cand_cap) return;
  if (g.d_cands) CU(cudaFree(g.d_cands));
  g.cand_cap = n + n / 4 + 64;
  CU(cudaMalloc(&g.d_cands, g.cand_cap * sizeof(Cand)));
}

static Cand make_cand(const Job& J, u32 job_idx, int mode) {
  Cand C{};
  C.q_lo = (u64)J.T;
  C.q_hi = (u64)(J.T >> 64);
  C.lim = J.lim;
  C.job = job_idx;
  C.T_r504 = (u32)(J.T % 504);
  C.T_r247 = (u32)(J.T % 247);
  if (mode == 2) {
    C.lo = C.hi = 0;
  } else if (mode == 3) {
    u32 hi = std::min(iroot6_host(J.T), J.lim);
    u32 lo = 1;
    while (lo <= hi && pow6_full(lo) * 3 < J.T) ++lo;
    if (lo > hi) { C.lo = 1; C.hi = 0; }
    else { C.lo = lo; C.hi = hi; }
  } else {  // find4
    u32 hi = std::min(iroot6_host(J.T), J.lim);
    u64 lo = 1;
    while (lo <= hi && pow6_full(lo) * 4 < J.T) ++lo;
    if (lo > hi) { C.lo = 1; C.hi = 0; }
    else { C.lo = (u32)lo; C.hi = hi; }
  }
  return C;
}

static void report_solution(const Job& J, u32 a, u32 b, u32 c, u32 d) {
  long long terms[5];
  int n = 0;
  if (J.cls == 1) {
    terms[0] = 42LL * a; terms[1] = 42LL * b; terms[2] = 42LL * c;
    terms[3] = 42LL * d; terms[4] = (long long)J.u; n = 5;
    if (!verify_fourcore_gmp(J.B, J.u, 42, a, b, c, d)) return;
  } else if (J.cls >= 2 && J.cls <= 4) {
    const int f = fc3::free_factor(J.cls);
    terms[0] = 42LL * a; terms[1] = 42LL * b; terms[2] = 42LL * c;
    terms[3] = (long long)f * (long long)J.free1; terms[4] = (long long)J.u; n = 5;
    if (!verify_cls234_gmp(J.B, J.u, (u64)f, J.free1, a, b, c)) return;
  } else if (J.cls == 5) {
    terms[0] = 42LL * a; terms[1] = 42LL * b;
    terms[2] = 21LL * (long long)J.free1; terms[3] = 14LL * (long long)J.free2;
    terms[4] = (long long)J.u; n = 5;
    if (!verify_cls5_gmp(J.B, J.u, J.free1, J.free2, a, b)) return;
  } else return;
  std::sort(terms, terms + n);
  for (int i = 0; i < n - 1; ++i) if (terms[i] == terms[i + 1]) return;
  if (terms[n - 1] >= (long long)J.B || terms[0] < 1) return;
  long long g = terms[0];
  for (int i = 1; i < n; ++i) g = std::gcd(g, terms[i]);
  g = std::gcd(g, (long long)J.B);
  printf("SOLUTION cls=%d B=%llu u=%llu a1=%lld a2=%lld a3=%lld a4=%lld a5=%lld gcd=%lld %s\n",
         J.cls, (unsigned long long)J.B, (unsigned long long)J.u,
         terms[0], terms[1], terms[2], terms[3], terms[4], g,
         g == 1 ? "primitive" : "NON-PRIMITIVE");
  fflush(stdout);
  fprintf(stderr, "%llu\tSOLUTION\tcls=%d\n", (unsigned long long)J.B, J.cls);
}

// Run a homogeneous batch (same cls / same kernel). jobs[i] must match mode.
static u64 run_batch(GpuState& g, const std::vector<Job>& jobs, int mode,
                     std::vector<u128>* p6 /*nullable, size N+1*/) {
  if (jobs.empty()) return 0;
  std::vector<Cand> cands;
  cands.reserve(jobs.size());
  std::vector<u32> ytiles;
  for (size_t i = 0; i < jobs.size(); ++i) {
    Cand C = make_cand(jobs[i], (u32)i, mode);
    if (mode != 2 && C.hi < C.lo) continue;
    cands.push_back(C);
    if (mode == 4) {
      u32 span = C.hi - C.lo + 1;
      u32 yt = (span + 2047) / 2048;
      if (yt < 1) yt = 1;
      ytiles.push_back(yt);
    }
  }
  if (cands.empty()) return 0;

  ensure_cands(g, cands.size());
  CU(cudaMemcpy(g.d_cands, cands.data(), cands.size() * sizeof(Cand), cudaMemcpyHostToDevice));
  CU(cudaMemset(g.d_nh, 0, sizeof(u32)));
  CU(cudaMemset(g.d_calls, 0, sizeof(u64)));
  CU(cudaMemset(g.d_gated, 0, sizeof(u64)));
  CU(cudaMemset(g.d_probes, 0, sizeof(u64)));

  if (mode == 2) {
    k_find2<<<(int)cands.size(), 32>>>(
        g.d_cands, (int)cands.size(), g.d_xor, g.xor_block, g.xor_r, g.xor_seed,
        g.d_gd, g.use_gate ? 1 : 0,
        g.d_hits, g.d_nh, g.hit_cap, g.d_calls, g.d_gated, g.d_probes);
  } else if (mode == 3) {
    k_find3<<<(int)cands.size(), 256>>>(
        g.d_cands, (int)cands.size(), g.d_xor, g.xor_block, g.xor_r, g.xor_seed,
        g.d_gd, g.use_gate ? 1 : 0,
        g.d_hits, g.d_nh, g.hit_cap, g.d_calls, g.d_gated, g.d_probes);
  } else {
    for (size_t i = 0; i < cands.size(); ++i) {
      CU(cudaMemcpy(g.d_cands, &cands[i], sizeof(Cand), cudaMemcpyHostToDevice));
      k_find4<<<dim3(1, ytiles[i]), 256>>>(
          g.d_cands, 1, g.d_xor, g.xor_block, g.xor_r, g.xor_seed,
          g.d_gd, g.use_gate ? 1 : 0,
          g.d_hits, g.d_nh, g.hit_cap, g.d_calls, g.d_gated, g.d_probes);
    }
  }
  CU(cudaDeviceSynchronize());

  u32 nh = 0;
  u64 calls = 0, gated = 0, probes = 0;
  CU(cudaMemcpy(&nh, g.d_nh, sizeof(u32), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(&calls, g.d_calls, sizeof(u64), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(&gated, g.d_gated, sizeof(u64), cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(&probes, g.d_probes, sizeof(u64), cudaMemcpyDeviceToHost));
  g.tot_calls += calls;
  g.tot_gated += gated;
  g.tot_probes += probes;
  if (calls && !g.quiet)
    fprintf(stderr, "[gate] batch n=%zu mode=%d calls=%llu gated=%llu probes=%llu gate=%.1f%%\n",
            cands.size(), mode, (unsigned long long)calls, (unsigned long long)gated,
            (unsigned long long)probes, 100.0 * (double)gated / (double)calls);

  if (nh > g.hit_cap) { fprintf(stderr, "!! hit overflow\n"); nh = g.hit_cap; }
  std::vector<Hit> raw(nh);
  if (nh) CU(cudaMemcpy(raw.data(), g.d_hits, nh * sizeof(Hit), cudaMemcpyDeviceToHost));

  // Expand xor maybe-hits → concrete (a,b) with a<=b
  std::vector<Hit> hits;
  hits.reserve(raw.size());
  std::vector<std::pair<u32, u32>> pairs;
  if (!g.pair_ix) { fprintf(stderr, "[fatal] pair_ix unset\n"); exit(1); }
  for (const Hit& H : raw) {
    g.pair_ix->recover(H.fp, pairs);
    for (auto [a, b] : pairs) {
      Hit X = H; X.a = a; X.b = b;
      hits.push_back(X);
    }
  }

  u64 sols = 0;
  for (const Hit& H : hits) {
    if (H.job >= jobs.size()) continue;
    const Job& J = jobs[H.job];
    if (mode == 2) {
      if (!p6 || H.a > J.lim || H.b > J.lim) continue;
      if ((*p6)[H.a] + (*p6)[H.b] != J.T) continue;
      report_solution(J, H.a, H.b, 0, 0);
      ++sols;
    } else if (mode == 3) {
      if (!p6 || H.c < 1 || H.b > H.c) continue;
      if ((*p6)[H.a] + (*p6)[H.b] + (*p6)[H.c] != J.T) continue;
      if (!(H.a <= H.b && H.b <= H.c)) continue;
      report_solution(J, H.a, H.b, H.c, 0);
      ++sols;
    } else {
      if (!p6 || H.c < 1 || H.d < 1 || H.b > H.c || H.c > H.d) continue;
      if ((*p6)[H.a] + (*p6)[H.b] + (*p6)[H.c] + (*p6)[H.d] != J.T) continue;
      if (!(H.a <= H.b && H.b <= H.c && H.c <= H.d)) continue;
      report_solution(J, H.a, H.b, H.c, H.d);
      ++sols;
    }
  }
  return sols;
}

static int mode_for_cls(int cls) {
  if (cls == 1) return 4;
  if (cls >= 2 && cls <= 4) return 3;
  if (cls == 5) return 2;
  return 0;
}

static u64 process_jobs(GpuState& g, std::vector<Job>& jobs, int N,
                        bool stop_first) {
  std::vector<u128> p6(N + 1);
  for (int x = 1; x <= N; ++x) p6[x] = pow6_full((u64)x);
  // group by mode
  u64 sols = 0;
  for (int mode : {4, 3, 2}) {
    std::vector<Job> batch;
    for (const auto& J : jobs)
      if (mode_for_cls(J.cls) == mode) batch.push_back(J);
    if (batch.empty()) continue;
    fprintf(stderr, "[run] mode=find%d jobs=%zu\n", mode, batch.size());
    // chunk find2 heavily
    const size_t CHUNK = (mode == 2) ? 4096 : (mode == 3) ? 256 : 32;
    for (size_t off = 0; off < batch.size(); off += CHUNK) {
      size_t n = std::min(CHUNK, batch.size() - off);
      std::vector<Job> sub(batch.begin() + off, batch.begin() + off + n);
      sols += run_batch(g, sub, mode, &p6);
      if (stop_first && sols) return sols;
      if ((off / CHUNK) % 20 == 0)
        fprintf(stderr, "[progress] mode=%d %zu/%zu sols=%llu\n",
                mode, off + n, batch.size(), (unsigned long long)sols);
    }
  }
  return sols;
}

static Job job_from_bin(const fc_jobbin::JobBinRec& r) {
  Job J;
  J.cls = (int)r.cls;
  J.B = r.B;
  J.u = r.u;
  J.free1 = r.free1;
  J.free2 = r.free2;
  J.T = join_u128(r.T_lo, r.T_hi);
  J.lim = (u32)((r.B > 0 ? r.B - 1 : 0) / 42);
  return J;
}

// Flush a mixed batch (split by find mode). Clears `jobs`.
static u64 flush_job_batch(GpuState& g, std::vector<Job>& jobs,
                           const std::vector<u128>& p6, bool stop_first) {
  u64 sols = 0;
  for (int mode : {4, 3, 2}) {
    std::vector<Job> sub;
    for (const auto& J : jobs)
      if (mode_for_cls(J.cls) == mode) sub.push_back(J);
    if (sub.empty()) continue;
    const size_t CHUNK = (mode == 2) ? 4096 : (mode == 3) ? 256 : 32;
    for (size_t off = 0; off < sub.size(); off += CHUNK) {
      size_t n = std::min(CHUNK, sub.size() - off);
      std::vector<Job> piece(sub.begin() + off, sub.begin() + off + n);
      sols += run_batch(g, piece, mode, &p6);
      if (stop_first && sols) {
        jobs.clear();
        return sols;
      }
    }
  }
  jobs.clear();
  return sols;
}

// Stream .bbj from disk (constant RAM aside from xor store).
static u64 stream_jobs_bbj(GpuState& g, const char* path, int N, bool stop_first,
                           u64& jobs_done) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return 0;
  }
  fc_jobbin::JobBinHeader hdr{};
  if (!fc_jobbin::read_header(f, hdr)) {
    fprintf(stderr, "[bbj] bad header %s\n", path);
    fclose(f);
    return 0;
  }
  fprintf(stderr, "[bbj] stream %s n_jobs=%llu B=[%u,%u]\n", path,
          (unsigned long long)hdr.n_jobs, hdr.min_B, hdr.max_B);

  std::vector<u128> p6(N + 1);
  for (int x = 1; x <= N; ++x) p6[x] = pow6_full((u64)x);

  std::vector<Job> batch;
  batch.reserve(4096);
  u64 sols = 0;
  jobs_done = 0;
  fc_jobbin::JobBinRec rec{};
  while (fc_jobbin::read_rec(f, rec)) {
    Job J = job_from_bin(rec);
    if (J.lim > (u32)N) {
      fprintf(stderr, "[bbj] job lim=%u > N=%d — skip\n", J.lim, N);
      continue;
    }
    batch.push_back(J);
    ++jobs_done;
    if (batch.size() >= 2048) {
      sols += flush_job_batch(g, batch, p6, stop_first);
      if ((jobs_done / 2048) % 20 == 0)
        fprintf(stderr, "[progress] bbj %llu/%llu sols=%llu\n",
                (unsigned long long)jobs_done, (unsigned long long)hdr.n_jobs,
                (unsigned long long)sols);
      if (stop_first && sols) break;
    }
  }
  sols += flush_job_batch(g, batch, p6, stop_first);
  fclose(f);
  return sols;
}

// Expand --units in small batches (no giant jobs vector).
static u64 stream_units_expand(GpuState& g, const fc3::RootTables& rt,
                               std::vector<Job>& units, int N, u64 max_expand,
                               bool stop_first, u64& failT, u64& jobs_done) {
  std::vector<u128> p6(N + 1);
  for (int x = 1; x <= N; ++x) p6[x] = pow6_full((u64)x);

  std::vector<Job> batch;
  batch.reserve(4096);
  u64 sols = 0;
  jobs_done = 0;
  failT = 0;
  bool cap = false;
  auto t0 = Clock::now();

  for (size_t ui = 0; ui < units.size() && !cap; ++ui) {
    std::vector<Job> expanded;
    expand_unit(rt, units[ui], expanded, failT, max_expand ? max_expand - jobs_done : 0, cap);
    for (auto& J : expanded) {
      if (J.lim > (u32)N) continue;
      batch.push_back(J);
      ++jobs_done;
      if (batch.size() >= 2048) {
        sols += flush_job_batch(g, batch, p6, stop_first);
        if (stop_first && sols) return sols;
      }
      if (max_expand && jobs_done >= max_expand) {
        cap = true;
        break;
      }
    }
    if ((ui + 1) % 100 == 0 || ui + 1 == units.size()) {
      const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
      fprintf(stderr, "[stream-units] %zu/%zu jobs=%llu failT=%llu sols=%llu %.1fs\n",
              ui + 1, units.size(), (unsigned long long)jobs_done,
              (unsigned long long)failT, (unsigned long long)sols, sec);
    }
  }
  sols += flush_job_batch(g, batch, p6, stop_first);
  return sols;
}
#endif // !HOST_ONLY

// ------------------------------ selftests ----------------------------------
static int selftest_host() {
  printf("[selftest-host] gate + peel plants + cls5 peel ctx\n");
  GateData gd; build_gate(gd);
  if (!gate_selftest_host(gd)) return 1;
  // find3 plant identity: T=3^6+4^6+5^6, peel reconstruct
  u128 T = pow6_full(3) + pow6_full(4) + pow6_full(5);
  u64 lo, hi; split_u128(T, lo, hi);
  if (join_u128(lo, hi) != T) { printf("FAIL split\n"); return 1; }
  // cls2 peel sample already in hunt_v3; re-check one
  u128 Tp = 0;
  if (!compute_T_peel1_gmp(100139, 73365, 14, 19, Tp)) {
    printf("FAIL peel1 sample\n"); return 1;
  }
  // cls5 cached peel must match one-shot peel2 (past i128 wall too)
  {
    const u64 B = 2354027, u = 17646;  // admissible-ish; u may not be cls5 unit
    // use known selftest pair from hunt_v3 cls5: B=100003 u=17646 d=9 e=44
    const u64 B0 = 100003, u0 = 17646, d0 = 9, e0 = 44;
    u128 T1 = 0, T2 = 0;
    if (!compute_T_peel2_gmp(B0, u0, d0, e0, T1)) {
      printf("FAIL peel2 sample\n"); return 1;
    }
    Cls5PeelCtx ctx;
    ctx.init(B0, u0);
    if (!ctx.ok || !ctx.peel(d0, e0, T2) || T1 != T2) {
      printf("FAIL Cls5PeelCtx mismatch\n"); return 1;
    }
    ctx.init(B, u);  // smoke init past wall
    if (!ctx.ok) { printf("FAIL Cls5PeelCtx past wall\n"); return 1; }
    printf("[selftest-host] Cls5PeelCtx ok (T bits~%d)\n",
           (int)(T1 ? 64 + ((T1 >> 64) ? 64 : 0) : 0));
  }
  printf("[selftest-host] PASS\n");
  return 0;
}

// Host-only / pre-GPU: stream-expand cls5 units and count jobs (no table).
static int stream_cls5_count(const char* units_path, u64 max_expand, size_t progress_every) {
  fc3::RootTables rt;
  std::vector<Job> units;
  if (!load_units_buc(units_path, units)) {
    fprintf(stderr, "failed to load %s\n", units_path);
    return 1;
  }
  u64 n5 = 0, jobs = 0, failT = 0, skip_cls = 0;
  auto t0 = Clock::now();
  for (size_t ui = 0; ui < units.size(); ++ui) {
    const Job& U = units[ui];
    if (U.cls != 5) { ++skip_cls; continue; }
    ++n5;
    Cls5PeelCtx ctx;
    ctx.init(U.B, U.u);
    if (!ctx.ok) { ++failT; continue; }
    fc3::FreeTermSpec es, ds;
    fc3::free_de_cls5(rt, (long long)U.B, (long long)U.u, es, ds);
    fc3::for_each_free(es, [&](long long e) {
      if (max_expand && jobs >= max_expand) return;
      fc3::for_each_free(ds, [&](long long d) {
        if (max_expand && jobs >= max_expand) return;
        u128 T = 0;
        if (!ctx.peel((u64)d, (u64)e, T)) { ++failT; return; }
        if (T == 0) return;
        ++jobs;
      });
    });
    if (progress_every && (n5 % progress_every == 0)) {
      const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
      fprintf(stderr, "[count-cls5] units=%llu jobs=%llu failT=%llu rate=%.2e jobs/s\n",
              (unsigned long long)n5, (unsigned long long)jobs,
              (unsigned long long)failT, sec > 0 ? jobs / sec : 0.0);
    }
    if (max_expand && jobs >= max_expand) break;
  }
  const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
  printf("---- count-cls5: units5=%llu skip_other=%llu jobs=%llu failT=%llu "
         "time=%.1fs rate=%.3e jobs/s ----\n",
         (unsigned long long)n5, (unsigned long long)skip_cls,
         (unsigned long long)jobs, (unsigned long long)failT, sec,
         sec > 0 ? jobs / sec : 0.0);
  return 0;
}

#ifndef HOST_ONLY
static int selftest_gpu() {
  printf("[selftest-gpu] tiny xor store + planted find2/find3/find4\n");
  const int N = 40;
  XorFilter xf = xor_build_pairs(N, 48);
  PairRecover pr; pr.build(N);
  GpuState g;
  gpu_init(g, xf, true);
  g.pair_ix = &pr;

  std::vector<Job> jobs;
  // find2 plant
  {
    Job J; J.cls = 5; J.B = 999999; J.u = 1; J.free1 = 1; J.free2 = 1;
    J.T = pow6_full(3) + pow6_full(4); J.lim = N;
    jobs.push_back(J);
  }
  // find3 plant
  {
    Job J; J.cls = 2; J.B = 999998; J.u = 1; J.free1 = 1; J.free2 = 0;
    J.T = pow6_full(3) + pow6_full(4) + pow6_full(5); J.lim = N;
    jobs.push_back(J);
  }
  // find4 plant
  {
    Job J; J.cls = 1; J.B = 999997; J.u = 1; J.free1 = 0; J.free2 = 0;
    J.T = pow6_full(3) + pow6_full(4) + pow6_full(5) + pow6_full(6); J.lim = N;
    jobs.push_back(J);
  }

  // Don't use report_solution GMP verify against fake B — check T hits only.
  std::vector<u128> p6(N + 1);
  for (int x = 1; x <= N; ++x) p6[x] = pow6_full((u64)x);

  u64 hits_exact = 0;
  for (int mode : {2, 3, 4}) {
    std::vector<Job> batch;
    for (auto& J : jobs) if (mode_for_cls(J.cls) == mode) batch.push_back(J);
    if (batch.empty()) continue;
    // custom: count exact T matches without verify_615
    ensure_cands(g, batch.size());
    std::vector<Cand> cands;
    for (size_t i = 0; i < batch.size(); ++i) cands.push_back(make_cand(batch[i], (u32)i, mode));
    CU(cudaMemcpy(g.d_cands, cands.data(), cands.size() * sizeof(Cand), cudaMemcpyHostToDevice));
    CU(cudaMemset(g.d_nh, 0, sizeof(u32)));
    CU(cudaMemset(g.d_calls, 0, sizeof(u64)));
    CU(cudaMemset(g.d_gated, 0, sizeof(u64)));
    CU(cudaMemset(g.d_probes, 0, sizeof(u64)));
    if (mode == 2)
      k_find2<<<(int)cands.size(), 32>>>(g.d_cands, (int)cands.size(), g.d_xor, g.xor_block, g.xor_r, g.xor_seed,
          g.d_gd, 1, g.d_hits, g.d_nh, g.hit_cap, g.d_calls, g.d_gated, g.d_probes);
    else if (mode == 3)
      k_find3<<<(int)cands.size(), 256>>>(g.d_cands, (int)cands.size(), g.d_xor, g.xor_block, g.xor_r, g.xor_seed,
          g.d_gd, 1, g.d_hits, g.d_nh, g.hit_cap, g.d_calls, g.d_gated, g.d_probes);
    else {
      Cand C = cands[0];
      u32 span = C.hi - C.lo + 1;
      u32 yt = (span + 2047) / 2048; if (!yt) yt = 1;
      k_find4<<<dim3(1, yt), 256>>>(g.d_cands, 1, g.d_xor, g.xor_block, g.xor_r, g.xor_seed,
          g.d_gd, 1, g.d_hits, g.d_nh, g.hit_cap, g.d_calls, g.d_gated, g.d_probes);
    }
    CU(cudaDeviceSynchronize());
    u32 nh = 0;
    CU(cudaMemcpy(&nh, g.d_nh, sizeof(u32), cudaMemcpyDeviceToHost));
    std::vector<Hit> raw(nh);
    if (nh) CU(cudaMemcpy(raw.data(), g.d_hits, nh * sizeof(Hit), cudaMemcpyDeviceToHost));
    std::vector<std::pair<u32, u32>> pairs;
    u64 ok = 0;
    for (auto& H : raw) {
      if (H.job >= batch.size()) continue;
      const Job& J = batch[H.job];
      pr.recover(H.fp, pairs);
      for (auto [a, b] : pairs) {
        if (mode == 2 && p6[a] + p6[b] == J.T) ++ok;
        if (mode == 3 && p6[a] + p6[b] + p6[H.c] == J.T) ++ok;
        if (mode == 4 && p6[a] + p6[b] + p6[H.c] + p6[H.d] == J.T) ++ok;
      }
    }
    printf("  find%d plant exact_hits=%llu (raw nh=%u)\n", mode,
           (unsigned long long)ok, nh);
    if (!ok) { printf("FAIL find%d plant\n", mode); gpu_free(g); return 1; }
    hits_exact += ok;
  }
  gpu_free(g);
  printf("[selftest-gpu] PASS (exact=%llu)\n", (unsigned long long)hits_exact);
  return 0;
}
#endif

static void usage() {
  printf(
      "usage: fourcore_find_v4 [--selftest-host] [--selftest-gpu]\n"
      "   or: fourcore_find_v4 --units FILE.buc [options]\n"
      "   or: fourcore_find_v4 --jobs FILE.but|.bbj [options]\n"
      "   or: fourcore_find_v4 --units FILE.buc [options]   (stream-expand; low RAM)\n"
      "   or: fourcore_find_v4 --stream-cls5 --units FILE.buc [options]\n"
      " options: --N n --r BITS (xor rank, default 48) --batch N --max-expand N\n"
      "           --max-table-gb G (ignored; kept for CLI compat) --S ignored\n"
      "          --count-only (stream-cls5: expand/count only, no GPU)\n"
      "          --quiet (no per-batch [gate] lines; stream-cls5 defaults on)\n"
      "          --no-quiet --no-gate --stop-first --device K\n"
      "  .buc: cls B u\n"
      "  .but: ASCII jobs | .bbj: binary jobs (~36 B/job; streamed, not all in RAM)\n"
      "  --units: expand Stage-1 in GPU-sized batches (prefer over giant .but)\n"
      "  --stream-cls5: DEBUG/COUNT only — host expands cls5 (d,e); ~1000x slower\n"
      "                 than fourcore_cls5_gpu_v4 for production searches\n");
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  setvbuf(stderr, nullptr, _IOLBF, 0);

  std::string units_path, jobs_path;
  int N = 0, S_override = 0, device = 0, xor_r = 48;
  double max_table_gb = 80.0;  // CLI compat; unused with xor
  u64 max_expand = 0;
  size_t find2_batch = 4096;
  bool use_gate = true, stop_first = false;
  bool do_host = false, do_gpu = false;
  bool stream_cls5 = false, count_only = false;
  int quiet_flag = -1;  // -1 = default (on for stream-cls5), 0/1 explicit

  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
    };
    if (s == "--selftest-host") do_host = true;
    else if (s == "--selftest-gpu") do_gpu = true;
    else if (s == "--stream-cls5") stream_cls5 = true;
    else if (s == "--count-only") count_only = true;
    else if (s == "--quiet") quiet_flag = 1;
    else if (s == "--no-quiet") quiet_flag = 0;
    else if (s == "--units") units_path = next();
    else if (s == "--jobs") jobs_path = next();
    else if (s == "--N") N = atoi(next().c_str());
    else if (s == "--r") xor_r = atoi(next().c_str());
    else if (s == "--S") S_override = atoi(next().c_str());  // ignored (xor)
    else if (s == "--max-table-gb") max_table_gb = atof(next().c_str());  // ignored
    else if (s == "--max-expand") max_expand = strtoull(next().c_str(), nullptr, 10);
    else if (s == "--batch") find2_batch = (size_t)strtoull(next().c_str(), nullptr, 10);
    else if (s == "--no-gate") use_gate = false;
    else if (s == "--stop-first") stop_first = true;
    else if (s == "--device") device = atoi(next().c_str());
    else if (s == "-h" || s == "--help") { usage(); return 0; }
    else { fprintf(stderr, "unknown %s\n", s.c_str()); usage(); return 1; }
  }
  const bool quiet = (quiet_flag >= 0) ? (quiet_flag != 0) : stream_cls5;
  (void)S_override;
  (void)max_table_gb;

  if (do_host || argc == 1) return selftest_host();

  if (stream_cls5 && count_only) {
    if (units_path.empty()) { usage(); return 1; }
    return stream_cls5_count(units_path.c_str(), max_expand, 1);
  }

#ifdef HOST_ONLY
  if (do_gpu || (stream_cls5 && !count_only)) {
    fprintf(stderr, "HOST_ONLY: use --count-only for stream-cls5, or nvcc build for GPU\n");
    return 1;
  }
  fprintf(stderr, "HOST_ONLY build — use nvcc for GPU runs\n");
  return 0;
#else
  if (do_gpu) {
    CU(cudaSetDevice(device));
    return selftest_gpu();
  }
  if (units_path.empty() && jobs_path.empty()) { usage(); return 1; }

  CU(cudaSetDevice(device));
  fc3::RootTables rt;

  // ---- streaming cls5: expand (d,e) → find2 batches, never store full job list ----
  if (stream_cls5) {
    if (units_path.empty()) { usage(); return 1; }
    std::vector<Job> units;
    if (!load_units_buc(units_path.c_str(), units)) {
      fprintf(stderr, "failed to load %s\n", units_path.c_str());
      return 1;
    }
    u32 need = 0;
    u64 n5 = 0;
    for (auto& U : units) {
      if (U.cls != 5) continue;
      ++n5;
      need = std::max(need, U.lim);
    }
    if (!n5) { fprintf(stderr, "no cls5 units in %s\n", units_path.c_str()); return 1; }
    if (N <= 0) N = (int)need;
    if ((u32)N < need) N = (int)need;
    if (N > kXorNSoftMax) {
      fprintf(stderr, "[fatal] N=%d exceeds soft xor cap %d (~71 GB cells)\n", N, kXorNSoftMax);
      return 1;
    }
    if (xor_r < 8 || xor_r > 64) { fprintf(stderr, "--r must be 8..64\n"); return 1; }
    fprintf(stderr,
            "[stream-cls5] WARNING: debug/count path — for production cls5 use "
            "fourcore_cls5_gpu_v4 (~1000x faster). See README / search_density_and_rates.md\n");
    fprintf(stderr,
            "[stream-cls5] units_file=%s cls5_units=%llu N=%d r=%d batch=%zu max_expand=%llu\n",
            units_path.c_str(), (unsigned long long)n5, N, xor_r, find2_batch,
            (unsigned long long)max_expand);

    XorFilter xf = xor_build_pairs(N, xor_r);
    PairRecover pair_ix;
    pair_ix.build(N);
    GpuState g;
    gpu_init(g, xf, use_gate, quiet);
    g.pair_ix = &pair_ix;
    decltype(xf.packed)().swap(xf.packed);

    std::vector<u128> p6(N + 1);
    for (int x = 1; x <= N; ++x) p6[x] = pow6_full((u64)x);

    std::vector<Job> batch;
    batch.reserve(find2_batch);
    u64 jobs_done = 0, failT = 0, sols = 0, units_done = 0;
    auto t0 = Clock::now();
    bool stop = false;

    auto flush = [&]() {
      if (batch.empty()) return;
      sols += run_batch(g, batch, /*mode=*/2, &p6);
      jobs_done += batch.size();
      batch.clear();
    };

    for (size_t ui = 0; ui < units.size() && !stop; ++ui) {
      const Job& U = units[ui];
      if (U.cls != 5) continue;
      ++units_done;
      Cls5PeelCtx ctx;
      ctx.init(U.B, U.u);
      if (!ctx.ok) { ++failT; continue; }
      fc3::FreeTermSpec es, ds;
      fc3::free_de_cls5(rt, (long long)U.B, (long long)U.u, es, ds);
      fc3::for_each_free(es, [&](long long e) {
        if (stop) return;
        fc3::for_each_free(ds, [&](long long d) {
          if (stop) return;
          if (max_expand && jobs_done + batch.size() >= max_expand) {
            stop = true;
            return;
          }
          u128 T = 0;
          if (!ctx.peel((u64)d, (u64)e, T)) { ++failT; return; }
          if (T == 0) return;
          Job J = U;
          J.free1 = (u64)d;
          J.free2 = (u64)e;
          J.T = T;
          batch.push_back(J);
          if (batch.size() >= find2_batch) flush();
        });
      });
      // Progress every 25 units (quiet-friendly); always on last.
      if (units_done % 25 == 0 || units_done == n5 || stop) {
        const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
        const double gate_pct = g.tot_calls
            ? 100.0 * (double)g.tot_gated / (double)g.tot_calls : 0.0;
        fprintf(stderr,
                "[stream-cls5] units=%llu/%llu jobs=%llu failT=%llu sols=%llu "
                "gate=%.1f%% %.1fs (%.2e jobs/s)\n",
                (unsigned long long)units_done, (unsigned long long)n5,
                (unsigned long long)jobs_done, (unsigned long long)failT,
                (unsigned long long)sols, gate_pct, sec,
                sec > 0 ? jobs_done / sec : 0.0);
      }
      if (stop_first && sols) break;
    }
    flush();
    gpu_free(g);
    const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    printf("---- done stream-cls5: solutions=%llu jobs=%llu units=%llu failT=%llu "
           "time=%.1fs%s ----\n",
           (unsigned long long)sols, (unsigned long long)jobs_done,
           (unsigned long long)units_done, (unsigned long long)failT, sec,
           stop && max_expand ? " (hit --max-expand)" : "");
    return 0;
  }

  // ---- jobs / units (stream .bbj and --units; ASCII .but still load-all) ----
  if (xor_r < 8 || xor_r > 64) { fprintf(stderr, "--r must be 8..64\n"); return 1; }

  u64 failT = 0, jobs_done = 0, sols = 0;

  // Binary jobs: stream from disk
  if (!jobs_path.empty() && fc_jobbin::is_bbj_path(jobs_path)) {
    FILE* hf = fopen(jobs_path.c_str(), "rb");
    if (!hf) { perror(jobs_path.c_str()); return 1; }
    fc_jobbin::JobBinHeader hdr{};
    if (!fc_jobbin::read_header(hf, hdr)) {
      fprintf(stderr, "bad .bbj %s\n", jobs_path.c_str());
      fclose(hf);
      return 1;
    }
    fclose(hf);
    if (!hdr.n_jobs) { fprintf(stderr, "empty .bbj\n"); return 1; }
    u32 need = hdr.max_B ? (hdr.max_B - 1) / 42 : 0;
    if (N <= 0) N = (int)need;
    if ((u32)N < need) N = (int)need;
    if (N > kXorNSoftMax) {
      fprintf(stderr, "[fatal] N=%d exceeds soft xor cap %d\n", N, kXorNSoftMax);
      return 1;
    }
    XorFilter xf = xor_build_pairs(N, xor_r);
    PairRecover pair_ix;
    pair_ix.build(N);
    GpuState g;
    gpu_init(g, xf, use_gate);
    g.pair_ix = &pair_ix;
    decltype(xf.packed)().swap(xf.packed);
    sols = stream_jobs_bbj(g, jobs_path.c_str(), N, stop_first, jobs_done);
    gpu_free(g);
    printf("---- done: solutions=%llu jobs=%llu failT=0 (bbj stream) ----\n",
           (unsigned long long)sols, (unsigned long long)jobs_done);
    return 0;
  }

  // Stage-1 units: stream-expand (low host RAM)
  if (!units_path.empty() && jobs_path.empty()) {
    std::vector<Job> units;
    if (!load_units_buc(units_path.c_str(), units)) {
      fprintf(stderr, "failed to load %s\n", units_path.c_str());
      return 1;
    }
    u32 need = 0;
    for (auto& U : units) need = std::max(need, U.lim);
    if (N <= 0) N = (int)need;
    if ((u32)N < need) N = (int)need;
    if (N > kXorNSoftMax) {
      fprintf(stderr, "[fatal] N=%d exceeds soft xor cap %d\n", N, kXorNSoftMax);
      return 1;
    }
    fprintf(stderr, "[units] %zu from %s — stream-expand (max_expand=%llu) N=%d\n",
            units.size(), units_path.c_str(), (unsigned long long)max_expand, N);
    XorFilter xf = xor_build_pairs(N, xor_r);
    PairRecover pair_ix;
    pair_ix.build(N);
    GpuState g;
    gpu_init(g, xf, use_gate);
    g.pair_ix = &pair_ix;
    decltype(xf.packed)().swap(xf.packed);
    sols = stream_units_expand(g, rt, units, N, max_expand, stop_first, failT, jobs_done);
    gpu_free(g);
    printf("---- done: solutions=%llu jobs=%llu failT=%llu (units stream) ----\n",
           (unsigned long long)sols, (unsigned long long)jobs_done,
           (unsigned long long)failT);
    return 0;
  }

  // ASCII .but: load-all (prefer .bbj or --units for large bands)
  std::vector<Job> jobs;
  if (!jobs_path.empty()) {
    if (!load_jobs_but(jobs_path.c_str(), jobs)) {
      fprintf(stderr, "failed to load %s\n", jobs_path.c_str());
      return 1;
    }
    fprintf(stderr, "[jobs] loaded %zu from %s (ASCII)\n", jobs.size(), jobs_path.c_str());
  } else {
    fprintf(stderr, "no --jobs / --units\n");
    usage();
    return 1;
  }
  if (jobs.empty()) {
    fprintf(stderr, "no jobs to run\n");
    return 1;
  }

  u32 need = 0;
  for (auto& J : jobs) need = std::max(need, J.lim);
  if (N <= 0) N = (int)need;
  if ((u32)N < need) N = (int)need;
  if (N > kXorNSoftMax) {
    fprintf(stderr, "[fatal] N=%d exceeds soft xor cap %d (~71 GB cells)\n", N, kXorNSoftMax);
    return 1;
  }

  XorFilter xf = xor_build_pairs(N, xor_r);
  PairRecover pair_ix;
  pair_ix.build(N);
  GpuState g;
  gpu_init(g, xf, use_gate);
  g.pair_ix = &pair_ix;
  decltype(xf.packed)().swap(xf.packed);

  sols = process_jobs(g, jobs, N, stop_first);
  gpu_free(g);
  printf("---- done: solutions=%llu jobs=%zu failT=%llu ----\n",
         (unsigned long long)sols, jobs.size(), (unsigned long long)failT);
  return 0;
#endif
}
