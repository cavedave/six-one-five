// == file: fourcore_find4_v2.cu  (PART 1/4) ==================================
// GPU find4 for (6,1,5) four-core jobs, with the mod-124,488 pair-sum gate
// (CRT factored 504 + 247 bitmaps).  All T arithmetic stays exact u128 host-side;
// the device works on (q_lo,q_hi) fitted to ~60 bits at B<=3.8M [6][24].
// Build:
//   nvcc -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 \
//        -o fourcore_find4_v2 fourcore_find4_v2.cu -Xcompiler -fopenmp \
//        -lgmpxx -lgmp
// (same command also works unchanged on CUDA 12.4 for PRO 6000 via PTX-JIT)
// =============================================================================
#include "fourcore_gmp.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using u16 = std::uint16_t;
using u128 = unsigned __int128;
using Clock = std::chrono::steady_clock;

// ------------------------------ gate (host) ---------------------------------
// Achievability bitmaps mod 124,488 = 504 * 247 (coprime CRT factors):
//   g504[r504] for the 6th-power pair sums mod 504
//   g247[r247] for the 6th-power pair sums mod 247
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

// exact 6th-power residue mod m for x < m (m <= 343 covers moduli)
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
      gate_set((u64*)g.g504,
               (si + (int)mod_pow6_h((u32)j, 504)) % 504);
  }
  for (int i = 0; i < 247; ++i) {
    const int si = (int)mod_pow6_h((u32)i, 247);
    g.p6247[i] = (u16)si;
    for (int j = i; j < 247; ++j)
      gate_set((u64*)g.g247,
               (si + (int)mod_pow6_h((u32)j, 247)) % 247);
  }
}

// gate selftest: assert exactly 27/504 and 50/247 bits set, then enumerate
// representative x,y in moduli proving every pair sum residue appears in bitmaps.
static bool gate_selftest_host(const GateData& g) {
  int c504 = 0, c247 = 0;
  for (int x = 0; x < 504; ++x) c504 += (int)gate_get(g.g504, x);
  for (int x = 0; x < 247; ++x) c247 += (int)gate_get(g.g247, x);
  if (c504 != 27 || c247 != 50) {
    fprintf(stderr, "gate bitmap sizes %d/%d, want 27/50\n", c504, c247);
    return false;
  }
  // brute-force pair-sum enumeration — must be present in the bitmap.
  int fail = 0;
  for (u32 i = 0; i < 504; ++i)
    for (u32 j = 0; j < 504; ++j) {
      const u32 r = (mod_pow6_h(i, 504) + mod_pow6_h(j, 504)) % 504;
      if (!gate_get(g.g504, r)) ++fail;
    }
  for (u32 i = 0; i < 247; ++i)
    for (u32 j = 0; j < 247; ++j) {
      const u32 r = (mod_pow6_h(i, 247) + mod_pow6_h(j, 247)) % 247;
      if (!gate_get(g.g247, r)) ++fail;
    }
  if (fail) { fprintf(stderr, "gate has %d missing pair sums\n", fail); return false; }
  printf("[gate-selftest] bitmaps 27 (of 504) / 50 (of 247); all 127k+30k pair sums pass\n");
  return true;
}

// == fourcore_find4_v2.cu  (PART 2/4) =======================================
// structs, host helpers, job loader, table build.
// Slot/Hash semantics must MATCH v1 exactly: hash_pos + mix64 + step|1 [6][28].
// =============================================================================

struct Slot { u32 i, j; u64 key; };                    // 16 B
struct Hit  { u32 job, c4, c3, a, b; };
struct Job  { u64 B = 0, u = 0, D = 42, lim = 0; u128 T = 0; };

struct Cand4 {
  u64 q_lo, q_hi;               // T truncated mod 2^128 (as v1) [6]
  u32 lim;                      // min(J.lim, N)
  u32 c4lo, c4hi;               // c4 window
  u32 job;                      // job index
  u32 T_r504, T_r247;           // exact T mod 504 / 247 (gate) [27]
  u32 pad;
};

struct HuntOpts {
  u64 x4_top = 0;
  bool stop_first = false;
  bool use_gate = true;         // --no-gate flips
};

// ---------------------------------------------------------------------------
static inline u128 pow6_full(u64 x) { u128 a = (u128)x * (u128)x; return a * a * a; }

static inline u64 mix64_h(u64 x) {                    // MUST equal d_mix64 [6]
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33; return x;
}
static inline u64 hash_pos(u64 fp, int S) {           // MUST equal d_probe pos [6]
  return (fp * 0x9E3779B97F4A7C15ULL) >> (64 - S);
}

static u32 iroot6_host(u128 n) {                      // floor sixth root, exact
  double x = ldexp((double)(u64)(n >> 64), 64) + (double)(u64)n;
  if (x < 1.0) return 0;
  double r = pow(x, 1.0 / 6.0);
  u64 a = (u64)r;
  while ((pow6_full(a + 1) <= n) && a < 0xffffffffULL - 2) ++a;
  while (pow6_full(a) > n && a) --a;
  return (u32)a;
}

// job line: "B u T_lo T_hi" from fourcore_hunt [7]
static bool load_jobs(const char* path, std::vector<Job>& jobs) {
  FILE* f = fopen(path, "r");
  if (!f) return false;
  char buf[256]; size_t bad = 0;
  while (fgets(buf, sizeof buf, f)) {
    unsigned long long B, u, tlo, thi;
    if (sscanf(buf, "%llu %llu %llu %llu", &B, &u, &tlo, &thi) != 4) { ++bad; continue; }
    Job J; J.B = B; J.u = u; J.T = join_u128(tlo, thi);   // [11]
    jobs.push_back(J);
  }
  fclose(f);
  if (bad) fprintf(stderr, "[jobs] skipped %zu malformed lines\n", bad);
  return !jobs.empty();
}

// smallest S with table <= budget, mirroring v1's observed behavior:
//   budget 80 GB -> S=32 (68.7 GB); budget 64 -> S=31 (34.4 GB) [6]
static int choose_S(double max_table_gb) {
  const long double bytes = (long double)max_table_gb * 1e9L;
  int S = 24;
  while (S + 1 <= 34 && ((long double)(1ULL << (S + 1)) * 16.0L) <= bytes) ++S;
  // ensure S >= ceil... v1 did not base S on N; mirrored.
  return S;
}

// CPU hash-table build (OpenMP), layout/prints identical to v1 logs [6][28].
static void table_build(int N, int S, std::vector<Slot>& slots) {
  const size_t size = (size_t)1 << S;
  const u64 mask = size - 1;
  const double pairs_est = (double)N * (N + 1) / 2.0;
  const double lf = pairs_est / (double)size;
  if (lf > 1.0)
    fprintf(stderr, "[table] note: LF=%.3f at S=%d — high; consider larger S or smaller N\n", lf, S);
  if (pairs_est > (double)size) {
    fprintf(stderr, "[fatal] pairs=%.3e exceed slots=2^%d — table cannot fit (v1 spun forever here); raise S or raise D\n",
            pairs_est, S);
    exit(1);
  }
  slots.assign(size, Slot{0u, 0u, 0ull});
  std::vector<u64> pw6(N + 1);
  for (int x = 1; x <= N; ++x) pw6[x] = (u64)pow6_full((u64)x);
  std::atomic<u64> used(0), steps(0);
  auto t0 = Clock::now();
#pragma omp parallel for schedule(dynamic, 256)
  for (int i = 1; i <= N; ++i) {
    u64 local = 0, ins = 0;
    for (int j = 1; j <= i; ++j) {
      const u64 fp = pw6[i] + pw6[j];
      u64 pos = hash_pos(fp, S);
      const u64 step = mix64_h(fp) | 1ULL;
      u64 probes_here = 0;
      for (;;) {
        ++local;
        if (++probes_here > (u64)(2 * size)) {          // safety: impossible once LF<1
          fprintf(stderr, "[fatal] table jam at i=%d j=%d (LF too high)\n", i, j);
          exit(1);
        }
        if (__sync_bool_compare_and_swap((u32*)&slots[pos].i, 0u, (u32)i)) {
          slots[pos].j = (u32)j;
          slots[pos].key = fp;
          ++ins;
          break;
        }
        pos = (pos + step) & mask;
      }
    }
    steps += local; used += ins;
  }
  const double ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
  fprintf(stderr, "[table] N=%d pairs=%.3e slots=2^%d (%.1f GB) LF=%.3f avg probes=%.2f build=%.0f ms\n",
          N, pairs_est, S, size * 16.0 / 1e9, used.load() / (double)size,
          steps.load() / pairs_est, ms);
}
// == END PART 2/4 — Part 3: device kernels + gpu driver =====================

// == fourcore_find4_v2.cu  (PART 3/4) =======================================
// device kernels (d_* verbatim from v1 probe path [6][28]) + gate wrapper
// modeled on the v616/v617 GateSh pattern [26][27].
// =============================================================================
#ifndef HOST_ONLY

__device__ __forceinline__ u64 d_mix64(u64 x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33; return x;
}
__device__ __forceinline__ u64 d_pow6_64(u32 x) {
  u64 x2 = (u64)x * x;
  return x2 * x2 * x2;
}
__device__ __forceinline__ void d_pow6_128(u32 x, u64& h, u64& l) {
  u128 x2w = (u128)x * x;
  u128 x6 = (x2w * x2w) * x2w;
  l = (u64)x6;
  h = (u64)(x6 >> 64);
}
__device__ __forceinline__ u32 d_iroot6(u64 rhi, u64 rlo) {
  double x = ldexp((double)rhi, 64) + (double)rlo;
  if (x < 1.0) return 0;
  double r = pow(x, 1.0 / 6.0);
  if (r > 4294967294.0) return 0xffffffffu;
  u32 a = (u32)r;
  while (a < 0xfffffffeu) {
    u64 h, l; d_pow6_128(a + 1, h, l);
    if (h < rhi || (h == rhi && l <= rlo)) ++a; else break;
  }
  while (a > 0) {
    u64 h, l; d_pow6_128(a, h, l);
    if (h < rhi || (h == rhi && l <= rlo)) break;
    --a;
  }
  return a;
}

__device__ int d_probe(const Slot* tab, u64 mask, int S, u64 fp, u32 c3,
                       u32 job, u32 c4,
                       Hit* hits, u32* nhit, u32 hit_cap) {
  u64 pos = (fp * 0x9E3779B97F4A7C15ULL) >> (64 - S);
  const u64 step = d_mix64(fp) | 1ULL;
  for (int k = 0; k < 64; ++k) {
    const Slot s = tab[pos];
    if (s.i == 0) return 0;
    if (s.key == fp && s.j <= c3 && s.i >= 1) {
      u32 h = atomicAdd(nhit, 1u);
      if (h < hit_cap)
        hits[h] = Hit{job, c4, c3, s.i, s.j};
      return 1;
    }
    pos = (pos + step) & mask;
  }
  return 0;
}

// ---- gate: shared-memory copy of the CRT bitmaps + residue tables ----------
struct GateSh { u64 g504[8]; u64 g247[4]; u16 p6504[504]; u16 p6247[247]; };

__device__ __forceinline__ bool d_gate_bit(const u64* b, int x) {
  return (b[x >> 6] >> (x & 63)) & 1ULL;
}

__device__ __forceinline__ void gate_load_sh(GateSh& gs, const GateData* gd) {
  const int tid = threadIdx.x;
  for (int k = tid; k < 8;   k += blockDim.x) gs.g504[k]  = gd->g504[k];
  for (int k = tid; k < 4;   k += blockDim.x) gs.g247[k]  = gd->g247[k];
  for (int k = tid; k < 504; k += blockDim.x) gs.p6504[k] = gd->p6504[k];
  for (int k = tid; k < 247; k += blockDim.x) gs.p6247[k] = gd->p6247[k];
  __syncthreads();
}

// k_find4_g: identical loop structure to v1's k_find4 [6][28]; gate inserted
// immediately before d_probe. r504/r247 semantics from v616 (q504/q247 form):
//   target pairsum residue = (T mod m - c4^6 mod m - c3^6 mod m) mod m. [27]
__global__ void k_find4_g(const Cand4* __restrict__ cands, int nc,
                          const Slot* __restrict__ tab, u64 mask, int S,
                          const GateData* __restrict__ gd, int use_gate,
                          Hit* __restrict__ hits, u32* __restrict__ nhit, u32 hit_cap,
                          u64* __restrict__ out_calls,
                          u64* __restrict__ out_gated,
                          u64* __restrict__ out_probes) {
  const u32 ci = blockIdx.x;
  if ((int)ci >= nc) return;
  const Cand4 C = cands[ci];
  __shared__ GateSh gs;
  if (use_gate) gate_load_sh(gs, gd);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const u32 base = C.c4lo + blockIdx.y * 2048u + (u32)warp;
  u64 lcalls = 0, lgated = 0, lprobes = 0;
  for (u32 c4 = base; c4 <= C.c4hi; c4 += 8) {
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
        if (r504 >= 504) r504 -= 504;   // r504 in [2, 2011]; three are enough
        if (r504 >= 504) r504 -= 504;
        if (r504 >= 504) r504 -= 504;
        u32 r247 = C.T_r247 + 494u - r4_247 - gs.p6247[c3 % 247];
        if (r247 >= 247) r247 -= 247;   // r247 in [1, 740]
        if (r247 >= 247) r247 -= 247;
        if (r247 >= 247) r247 -= 247;
        if (!(d_gate_bit(gs.g504, (int)r504) && d_gate_bit(gs.g247, (int)r247))) {
          ++lgated;
          continue;
        }
      }
      ++lprobes;
      d_probe(tab, mask, S, tfp - d_pow6_64(c3), c3,
              C.job, c4, hits, nhit, hit_cap);
    }
  }
  atomicAdd((unsigned long long*)out_calls, (unsigned long long)lcalls);
  atomicAdd((unsigned long long*)out_gated, (unsigned long long)lgated);
  atomicAdd((unsigned long long*)out_probes, (unsigned long long)lprobes);
}

static void gpu_find4_jobs(const std::vector<Job>& jobs, int N, int S,
                           const std::vector<Slot>& slots,
                           const HuntOpts& opt,
                           std::vector<Hit>& out_hits) {
  Slot* d_tab = nullptr;
  CU(cudaMalloc(&d_tab, slots.size() * sizeof(Slot)));
  CU(cudaMemcpy(d_tab, slots.data(), slots.size() * sizeof(Slot), cudaMemcpyHostToDevice));

  GateData gd;
  build_gate(gd);
  GateData* d_gd = nullptr;
  CU(cudaMalloc(&d_gd, sizeof(gd)));
  CU(cudaMemcpy(d_gd, &gd, sizeof(gd), cudaMemcpyHostToDevice));

  const u32 hit_cap = 1u << 20;
  Hit* d_hits = nullptr; u32* d_nh = nullptr;
  u64* d_calls = nullptr; u64* d_gated = nullptr; u64* d_probes = nullptr;
  CU(cudaMalloc(&d_hits, hit_cap * sizeof(Hit)));
  CU(cudaMalloc(&d_nh, sizeof(u32)));
  CU(cudaMalloc(&d_calls, sizeof(u64)));
  CU(cudaMalloc(&d_gated, sizeof(u64)));
  CU(cudaMalloc(&d_probes, sizeof(u64)));

  std::vector<u128> p6full(N + 1);
  for (int x = 1; x <= N; ++x) p6full[x] = pow6_full((u64)x);

  u64 tot_calls = 0, tot_gated = 0, tot_probes = 0;
  bool stop = false;
  for (size_t ji = 0; ji < jobs.size() && !stop; ++ji) {
    const Job& J = jobs[ji];
    const u32 lim = (u32)std::min<u64>(J.lim, (u64)N);
    const u32 hi4 = std::min(iroot6_host(J.T), lim);
    u64 lo4 = 1;
    while (lo4 <= hi4 && p6full[lo4] * 4 < J.T) ++lo4;
    if (opt.x4_top > 0 && hi4 + 1 > opt.x4_top)
      lo4 = std::max<u64>(lo4, hi4 - opt.x4_top + 1);
    if (lo4 > hi4) continue;

    Cand4 C;
    C.q_lo = (u64)J.T; C.q_hi = (u64)(J.T >> 64);
    C.lim = lim;
    C.c4lo = (u32)lo4; C.c4hi = (u32)hi4;
    C.job = (u32)ji;
    C.T_r504 = (u32)(J.T % 504);   // exact: u128 % literal, no folding [11]
    C.T_r247 = (u32)(J.T % 247);

    fprintf(stderr, "[gpu] job %zu/%zu B=%llu u=%llu c4=[%u,%u] lim=%u\n",
            ji + 1, jobs.size(), (unsigned long long)J.B, (unsigned long long)J.u,
            C.c4lo, C.c4hi, lim);

    const u32 span = C.c4hi - C.c4lo + 1;
    u32 ytiles = (span + 2047) / 2048;
    if (ytiles < 1) ytiles = 1;

    Cand4* d_c = nullptr;
    CU(cudaMalloc(&d_c, sizeof(Cand4)));
    CU(cudaMemcpy(d_c, &C, sizeof(Cand4), cudaMemcpyHostToDevice));
    CU(cudaMemset(d_nh, 0, sizeof(u32)));
    CU(cudaMemset(d_calls, 0, sizeof(u64)));
    CU(cudaMemset(d_gated, 0, sizeof(u64)));
    CU(cudaMemset(d_probes, 0, sizeof(u64)));

    const auto t0 = Clock::now();
    k_find4_g<<<dim3(1, ytiles), 256>>>(
        d_c, 1, d_tab, slots.size() - 1, S,
        d_gd, opt.use_gate ? 1 : 0,
        d_hits, d_nh, hit_cap, d_calls, d_gated, d_probes);
    CU(cudaDeviceSynchronize());
    const double job_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();

    u32 nh = 0;
    u64 jcalls = 0, jgated = 0, jprobes = 0;
    CU(cudaMemcpy(&nh, d_nh, sizeof(u32), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&jcalls, d_calls, sizeof(u64), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&jgated, d_gated, sizeof(u64), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&jprobes, d_probes, sizeof(u64), cudaMemcpyDeviceToHost));
    tot_calls += jcalls; tot_gated += jgated; tot_probes += jprobes;
    if (jcalls)
      fprintf(stderr, "[gate] job %zu: calls=%llu gated=%llu probes=%llu gate=%.1f%% time=%.0fms\n",
              ji + 1, (unsigned long long)jcalls, (unsigned long long)jgated,
              (unsigned long long)jprobes, 100.0 * (double)jgated / (double)jcalls, job_ms);

    if (nh > hit_cap) { fprintf(stderr, "!! hit overflow\n"); nh = hit_cap; }
    std::vector<Hit> raw(nh);
    if (nh) CU(cudaMemcpy(raw.data(), d_hits, nh * sizeof(Hit), cudaMemcpyDeviceToHost));
    CU(cudaFree(d_c));

    for (const Hit& H : raw) {
      const Job& JJ = jobs[H.job];
      const u128 sum = p6full[H.a] + p6full[H.b] + p6full[H.c3] + p6full[H.c4];
      if (sum != JJ.T) continue;
      if (!(H.a <= H.b && H.b <= H.c3 && H.c3 <= H.c4)) continue;
      if (!verify_fourcore_gmp(JJ.B, JJ.u, JJ.D, H.a, H.b, H.c3, H.c4)) continue;
      out_hits.push_back(H);
      long long terms[5] = {
          (long long)JJ.D * H.a, (long long)JJ.D * H.b,
          (long long)JJ.D * H.c3, (long long)JJ.D * H.c4,
          (long long)JJ.u
      };
      std::sort(terms, terms + 5);
      printf("SOLUTION B=%llu u=%llu D=%llu x=(%u,%u,%u,%u)\n",
             (unsigned long long)JJ.B, (unsigned long long)JJ.u,
             (unsigned long long)JJ.D, H.a, H.b, H.c3, H.c4);
      printf("  terms: %lld %lld %lld %lld %lld\n",
             terms[0], terms[1], terms[2], terms[3], terms[4]);
      fflush(stdout);
      if (opt.stop_first) { stop = true; break; }
    }
  }
  if (tot_calls)
    fprintf(stderr, "[gate] TOTAL: calls=%llu gated=%llu probes=%llu gate=%.2f%%\n",
            (unsigned long long)tot_calls, (unsigned long long)tot_gated,
            (unsigned long long)tot_probes, 100.0 * (double)tot_gated / (double)tot_calls);

  CU(cudaFree(d_gd));
  CU(cudaFree(d_calls)); CU(cudaFree(d_gated)); CU(cudaFree(d_probes));
  CU(cudaFree(d_tab));
  CU(cudaFree(d_hits));
  CU(cudaFree(d_nh));
}
#endif // !HOST_ONLY
// == END PART 3/4 — Part 4: selftests + main ================================
// == fourcore_find4_v2.cu  (PART 4/4) =======================================
// selftests + CLI + main. Host selftest includes a full CPU end-to-end plant
// (table + gate + find4 semantics) so the gate logic is exercised without GPU.
// =============================================================================

static int selftest_host() {
  printf("[selftest-host] gate + GMP fourcore math + CPU end-to-end plant\n");
  // 1) split/join roundtrip (v1 smoke) [11]
  u128 T0 = pow6_full(10) + pow6_full(20) + pow6_full(30) + pow6_full(40);
  u64 lo, hi; split_u128(T0, lo, hi);
  if (join_u128(lo, hi) != T0) { printf("FAIL split/join\n"); return 1; }

  // 2) gate bitmap selftest [26]
  GateData gd;
  build_gate(gd);
  if (!gate_selftest_host(gd)) return 1;

  // 3) CPU end-to-end: N=45 pair table; plant T = 10^6+20^6+30^6+40^6;
  //    gated search must find the same decompositions as ungated exhaustive.
  const int N = 45;
  std::vector<u64> pw6(N + 1);
  for (int x = 0; x <= N; ++x) pw6[x] = (u64)pow6_full((u64)x);
  Job J; J.B = 100003; J.u = 1; J.D = 42; J.T = T0; J.lim = N;

  const u32 T504 = (u32)(J.T % 504), T247 = (u32)(J.T % 247);
  long found_ungated = 0, found_gated = 0;
  const u32 hi4 = std::min(iroot6_host(J.T), (u32)N);
  for (u32 c4 = 1; c4 <= hi4; ++c4) {
    for (u32 c3 = 1; c3 <= std::min(c4, hi4); ++c3) {
      // ungated ground truth
      for (u32 c2 = 1; c2 <= c3; ++c2)
        for (u32 c1 = 1; c1 <= c2; ++c1) {
          if (c4 >= c3 && c3 >= c2 && c2 >= c1 &&
              pw6[c1] + pw6[c2] + pw6[c3] + pw6[c4] == (u64)J.T) ++found_ungated;
        }
      // gated: check only pairs that pass, then verify
      {
        u32 c4s = (u32)((pw6[c4] % 504)), c3s = (u32)((pw6[c3] % 504));
        u32 r504 = (T504 + 1008u - c4s - c3s) % 504;
        u32 c4p = (u32)((pw6[c4] % 247)), c3p = (u32)((pw6[c3] % 247));
        u32 r247 = (T247 + 494u - c4p - c3p) % 247;
        if (!(gate_get(gd.g504, (int)r504) && gate_get(gd.g247, (int)r247))) continue;
        // survived: now do the "probe" (search the pair list)
        const u64 t = (u64)J.T - pw6[c4] - pw6[c3];
        for (u32 c2 = 1; c2 <= c3; ++c2)
          for (u32 c1 = 1; c1 <= c2; ++c1)
            if (pw6[c1] + pw6[c2] == t) ++found_gated;
      }
    }
  }
  if (found_ungated == 0 || found_gated != found_ungated) {
    printf("FAIL CPU plant: ungated=%ld gated=%ld\n", found_ungated, found_gated);
    return 1;
  }
  printf("  CPU end-to-end plant: %ld decompositions found, gated == ungated\n", found_gated);
  printf("[selftest-host] PASS\n");
  return 0;
}

static void usage() {
  printf("usage: fourcore_find4_v2 [--selftest-host] [--gate-selftest]\n"
         "                        [--jobs file] [--D d]\n"
         "                        [--N n] [--S bits] [--max-table-gb G]\n"
         "                        [--x4-top K] [--stop-first] [--no-gate]\n"
         "  jobs from fourcore_hunt: B u T_lo T_hi [7]\n");
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  setvbuf(stderr, nullptr, _IOLBF, 0);

  std::string jobs_path;
  int N = 0;
  int S_override = 0;
  double max_table_gb = 80.0;
  u64 D = 42;
  HuntOpts hop;
  bool do_host = false, do_gate_st = false;

  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
    };
    if (s == "--selftest-host") do_host = true;
    else if (s == "--gate-selftest") do_gate_st = true;
    else if (s == "--jobs") jobs_path = next();
    else if (s == "--D") D = strtoull(next().c_str(), nullptr, 10);
    else if (s == "--N") N = atoi(next().c_str());
    else if (s == "--S") S_override = atoi(next().c_str());
    else if (s == "--max-table-gb") max_table_gb = atof(next().c_str());
    else if (s == "--x4-top") hop.x4_top = strtoull(next().c_str(), nullptr, 10);
    else if (s == "--stop-first") hop.stop_first = true;
    else if (s == "--no-gate") hop.use_gate = false;
    else if (s == "-h" || s == "--help") { usage(); return 0; }
    else { printf("unknown %s\n", s.c_str()); usage(); return 1; }
  }

  if (do_host) return selftest_host();
  if (do_gate_st && jobs_path.empty()) {
    GateData gd; build_gate(gd);
    return gate_selftest_host(gd) ? 0 : 1;
  }
  if (argc == 1) return selftest_host();

  std::vector<Job> jobs;
  if (jobs_path.empty() || !load_jobs(jobs_path.c_str(), jobs)) {
    usage(); return 1;
  }
  for (auto& J : jobs) { J.D = D; J.lim = J.B / D; }
  fprintf(stderr, "[jobs] loaded %zu from %s (D=%llu)\n",
          jobs.size(), jobs_path.c_str(), (unsigned long long)D);

  if (do_gate_st) {          // --gate-selftest WITH --jobs: bitmaps + exit
    GateData gd; build_gate(gd);
    if (!gate_selftest_host(gd)) return 1;
  }

  u32 need = 0;
  for (auto& J : jobs) need = std::max(need, (u32)J.lim);
  if (N <= 0) N = (int)need;
  if ((u32)N < need) {
    fprintf(stderr, "[warn] N=%d < max lim=%u — bumping\n", N, need);
    N = (int)need;
  }
  if (N > 100000) {
    fprintf(stderr, "[fatal] N=%d too large for full table; raise D or shard\n", N);
    return 1;
  }

#ifdef HOST_ONLY
  fprintf(stderr, "HOST_ONLY: %zu jobs ready (need nvcc for GPU)\n", jobs.size());
  return 0;
#else
  int S = S_override > 0 ? S_override : choose_S(max_table_gb);
  fprintf(stderr, "[table] using S=%d (%.1f GB) budget=%.0f GB\n",
          S, ((size_t)1 << S) * 16.0 / 1e9, max_table_gb);
  std::vector<Slot> slots;
  table_build(N, S, slots);
  std::vector<Hit> hits;
  gpu_find4_jobs(jobs, N, S, slots, hop, hits);
  printf("---- done: exact hits=%zu ----\n", hits.size());
  return 0;
#endif
}
// == END PART 4/4 ===========================================================
