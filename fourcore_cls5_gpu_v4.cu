// =============================================================================
// fourcore_cls5_gpu_v4.cu — PRODUCTION post-i128 cls5 path (GPU peel + xor)
//
// *** THIS IS THE FAST PATH. Do NOT use fourcore_find_* --stream-cls5 for
// *** production cls5 (~1000× slower: host expands (d,e); weeks vs minutes).
//
// 192-bit (d,e) peel on GPU + GMP verify; xor pair store (not OA table).
// N may exceed 65535 (soft cap ~120k).
//
// Build:  make fourcore-cls5-gpu-v4
//   ./fourcore_cls5_gpu_v4 --selftest-host
//   ./fourcore_cls5_gpu_v4 --units FILE.buc --r 48
//   ./fourcore_cls5_gpu_v4 --units FILE.buc --load-table xor_N.bin [--save-table F]
//   ./fourcore_cls5_gpu_v4_host --check-table --N K --load-table F   # no GPU
// =============================================================================

#include "fourcore_classes_v3.hpp"
#include "fourcore_gmp.hpp"
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
#include <utility>
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

static constexpr u64 M42u = 5489031744ULL;
static constexpr u64 M21u = 85766121ULL;

struct U192 { u64 l0 = 0, l1 = 0, l2 = 0; };

static bool mpz_to_u192(const mpz_class& z, U192& o) {
  if (z < 0) return false;
  if (mpz_sizeinbase(z.get_mpz_t(), 2) > 192) return false;
  o = {};
  if (z == 0) return true;
  size_t count = 0;
  mpz_export(&o.l0, &count, -1, sizeof(u64), 0, 0, z.get_mpz_t());
  return count <= 3;
}

static u64 inv216_mod_2_64() {
  u64 a = M21u, x = a;
  for (int i = 0; i < 6; ++i) x = x * (2 - a * x);
  return x;
}

static inline u128 pow6_full(u64 x) {
  u128 a = (u128)x * x;
  return a * a * a;
}

static u64 funnel_fp_host(u64 rh, u64 rl, u64 inv216) {
  return ((rl >> 6) | (rh << 58)) * inv216;
}

// Host mirror of device d_pow6_192 / sub192 (selftest + offline debug).
static void pow6_192_host(u32 x, u64& o2, u64& o1, u64& o0) {
  const u64 x2 = (u64)x * (u64)x;
  const u128 x4 = (u128)x2 * (u128)x2;
  const u64 x4_lo = (u64)x4;
  const u64 x4_hi = (u64)(x4 >> 64);
  const u128 p_lo = (u128)x4_lo * (u128)x2;
  const u128 p_hi = (u128)x4_hi * (u128)x2;
  o0 = (u64)p_lo;
  const u64 mid = (u64)(p_lo >> 64);
  const u64 hi_lo = (u64)p_hi;
  const u64 hi_hi = (u64)(p_hi >> 64);
  const u64 mid2 = mid + hi_lo;
  const u64 c = (mid2 < mid) ? 1ull : 0ull;
  o1 = mid2;
  o2 = hi_hi + c;
}

static bool sub192_host(u64& a2, u64& a1, u64& a0, u64 b2, u64 b1, u64 b0) {
  const u64 r0 = a0 - b0;
  u64 br = (a0 < b0) ? 1ull : 0ull;
  const u64 a1b = a1 - br;
  br = (a1 < br) ? 1ull : 0ull;
  const u64 r1 = a1b - b1;
  br |= (a1b < b1) ? 1ull : 0ull;
  const u64 a2b = a2 - br;
  br = (a2 < br) ? 1ull : 0ull;
  const u64 r2 = a2b - b2;
  br |= (a2b < b2) ? 1ull : 0ull;
  a0 = r0; a1 = r1; a2 = r2;
  return br != 0;
}

// ------------------------------ gate ---------------------------------------
struct GateData {
  u64 g504[8] = {0};
  u64 g247[4] = {0};
  u16 p6504[504];
  u16 p6247[247];
};
static inline void gate_set(u64* b, int x) { b[x >> 6] |= 1ULL << (x & 63); }
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

// ------------------------------ xor store ----------------------------------
struct Hit { u32 unit, a, b, d, e; u64 fp; };

// ------------------------------ units --------------------------------------
struct Unit5 {
  u64 B = 0, u = 0;
  U192 base{};
  u32 lim = 0;
  u32 mod1 = 0, mod2 = 0;
  u32 fmax1 = 0, fmax2 = 0;
  u32 nres1 = 0, nres2 = 0;
  u32 res[24]{};
};

struct BucLine { int cls; u64 B, u; };

static bool load_buc5(const char* path, std::vector<BucLine>& out) {
  FILE* f = fopen(path, "r");
  if (!f) return false;
  char buf[256];
  while (fgets(buf, sizeof buf, f)) {
    int cls; unsigned long long B, u;
    if (sscanf(buf, "%d %llu %llu", &cls, &B, &u) != 3) continue;
    if (cls != 5) continue;
    out.push_back({cls, B, u});
  }
  fclose(f);
  return !out.empty();
}

static bool pack_unit5(const fc3::RootTables& rt, u64 B, u64 u, Unit5& U) {
  U = {};
  U.B = B; U.u = u; U.lim = (u32)((B - 1) / 42);
  mpz_class base = mpz_pow6(B) - mpz_pow6(u);
  if (base <= 0) return false;
  if (!mpz_to_u192(base, U.base)) return false;
  fc3::FreeTermSpec es, ds;
  fc3::free_de_cls5(rt, (long long)B, (long long)u, es, ds);
  U.mod1 = (u32)es.mod; U.fmax1 = (u32)es.fmax;
  U.mod2 = (u32)ds.mod; U.fmax2 = (u32)ds.fmax;
  U.nres1 = (u32)std::min<size_t>(es.residues.size(), 12);
  U.nres2 = (u32)std::min<size_t>(ds.residues.size(), 12);
  for (u32 i = 0; i < U.nres1; ++i) U.res[i] = (u32)es.residues[i];
  for (u32 i = 0; i < U.nres2; ++i) U.res[U.nres1 + i] = (u32)ds.residues[i];
  return U.nres1 > 0 && U.nres2 > 0;
}

static int selftest_host() {
  printf("[selftest-host] cls5 GPU peel (192-bit funnel vs GMP T)\n");
  const u64 inv216 = inv216_mod_2_64();
  fc3::RootTables rt;

  // Spot-check pow6_192 vs GMP for values near the pair-index ceiling.
  for (u32 x : {1u, 42u, 100003u, 2353975u, 2752470u}) {
    u64 o2 = 0, o1 = 0, o0 = 0;
    pow6_192_host(x, o2, o1, o0);
    U192 g{};
    if (!mpz_to_u192(mpz_pow6(x), g) || g.l0 != o0 || g.l1 != o1 || g.l2 != o2) {
      printf("FAIL pow6_192 x=%u host=(%llu,%llu,%llu)\n", x,
             (unsigned long long)o2, (unsigned long long)o1, (unsigned long long)o0);
      return 1;
    }
  }

  auto check_peel = [&](u64 B, u64 u, u64 d, u64 e, const char* tag) -> int {
    u128 T_gmp = 0;
    if (!compute_T_peel2_gmp(B, u, d, e, T_gmp)) {
      printf("FAIL gmp peel %s\n", tag); return 1;
    }
    Unit5 U;
    if (!pack_unit5(rt, B, u, U)) { printf("FAIL pack %s\n", tag); return 1; }
    u64 r2 = U.base.l2, r1 = U.base.l1, r0 = U.base.l0;
    u64 e2, e1, e0, d2, d1, d0;
    pow6_192_host((u32)(14 * e), e2, e1, e0);
    if (sub192_host(r2, r1, r0, e2, e1, e0)) {
      printf("FAIL under-sub e %s\n", tag); return 1;
    }
    pow6_192_host((u32)(21 * d), d2, d1, d0);
    if (sub192_host(r2, r1, r0, d2, d1, d0)) {
      printf("FAIL under-sub d %s\n", tag); return 1;
    }
    const u64 fp = funnel_fp_host(r1, r0, inv216);
    if (fp != (u64)T_gmp) {
      printf("FAIL fp %s funnel=%llu Tlo=%llu\n", tag,
             (unsigned long long)fp, (unsigned long long)(u64)T_gmp);
      return 1;
    }
    return 0;
  };

  if (check_peel(100003, 17646, 9, 44, "pre-wall")) return 1;

  // Past i128 wall: find one divisible (d,e) from the unit's residue classes.
  {
    Unit5 Up;
    if (!pack_unit5(rt, 2353975, 85710, Up)) {
      printf("FAIL pack past-wall unit\n"); return 1;
    }
    bool found = false;
    for (u32 ie = 0; ie < Up.nres1 && !found; ++ie) {
      for (u32 ke = 0; ke < 8 && !found; ++ke) {
        const u64 e = (u64)Up.res[ie] + (u64)ke * Up.mod1;
        if (e == 0 || e > Up.fmax1) continue;
        for (u32 id = 0; id < Up.nres2 && !found; ++id) {
          for (u32 kd = 0; kd < 8 && !found; ++kd) {
            const u64 d = (u64)Up.res[Up.nres1 + id] + (u64)kd * Up.mod2;
            if (d == 0 || d > Up.fmax2) continue;
            u128 Ttmp = 0;
            if (!compute_T_peel2_gmp(2353975, 85710, d, e, Ttmp)) continue;
            if (check_peel(2353975, 85710, d, e, "past-wall")) return 1;
            found = true;
          }
        }
      }
    }
    if (!found) { printf("FAIL no divisible past-wall (d,e)\n"); return 1; }
  }

  mpz_class b6 = mpz_pow6(2752470ULL);
  const long bits = (long)mpz_sizeinbase(b6.get_mpz_t(), 2);
  printf("  pow6_192/funnel ok; past-wall peel ok; B=2752470 B^6 bits=%ld\n", bits);
  if (bits > 192) { printf("FAIL >192 bits\n"); return 1; }
  printf("[selftest-host] PASS\n");
  return 0;
}

#ifndef HOST_ONLY
__device__ __forceinline__ u64 d_funnel_fp(u64 rh, u64 rl, u64 inv216) {
  return ((rl >> 6) | (rh << 58)) * inv216;
}

__device__ __forceinline__ void d_pow6_192(u32 x, u64& o2, u64& o1, u64& o0) {
  const u64 x2 = (u64)x * (u64)x;
  const u64 x4_lo = x2 * x2;
  const u64 x4_hi = __umul64hi(x2, x2);
  o0 = x4_lo * x2;
  u64 mid = __umul64hi(x4_lo, x2);
  const u64 hi_lo = x4_hi * x2;
  const u64 hi_hi = __umul64hi(x4_hi, x2);
  const u64 mid2 = mid + hi_lo;
  const u64 c = (mid2 < mid) ? 1ull : 0ull;
  o1 = mid2;
  o2 = hi_hi + c;
}

__device__ __forceinline__ bool sub192(u64& a2, u64& a1, u64& a0,
                                       u64 b2, u64 b1, u64 b0) {
  const u64 r0 = a0 - b0;
  u64 br = (a0 < b0) ? 1ull : 0ull;
  const u64 a1b = a1 - br;
  br = (a1 < br) ? 1ull : 0ull;
  const u64 r1 = a1b - b1;
  br |= (a1b < b1) ? 1ull : 0ull;
  const u64 a2b = a2 - br;
  br = (a2 < br) ? 1ull : 0ull;
  const u64 r2 = a2b - b2;
  br |= (a2b < b2) ? 1ull : 0ull;
  a0 = r0; a1 = r1; a2 = r2;
  return br != 0;
}

__device__ __forceinline__ u64 mod192(u64 a2, u64 a1, u64 a0, u64 m, u64 two64_mod_m) {
  u64 r = a2 % m;
  r = (u64)((unsigned __int128)r * two64_mod_m % m);
  r = (u64)((unsigned __int128)r * two64_mod_m % m);
  r = (u64)(((unsigned __int128)r + (unsigned __int128)(a1 % m) * two64_mod_m) % m);
  r += a0 % m;
  if (r >= m) r -= m;
  return r;
}

struct GateSh { u64 g504[8]; u64 g247[4]; u16 p6504[504]; u16 p6247[247]; };
__device__ __forceinline__ bool d_gate_bit(const u64* b, int x) {
  return (b[x >> 6] >> (x & 63)) & 1ULL;
}
__device__ __forceinline__ void gate_load_sh(GateSh& gs, const GateData* gd) {
  for (int k = threadIdx.x; k < 8; k += blockDim.x) gs.g504[k] = gd->g504[k];
  for (int k = threadIdx.x; k < 4; k += blockDim.x) gs.g247[k] = gd->g247[k];
  for (int k = threadIdx.x; k < 504; k += blockDim.x) gs.p6504[k] = gd->p6504[k];
  for (int k = threadIdx.x; k < 247; k += blockDim.x) gs.p6247[k] = gd->p6247[k];
  __syncthreads();
}

__device__ __forceinline__ bool d_xor_might_contain(const uint8_t* packed, u32 block,
                                                     u32 r, u64 seed, u64 key) {
  if (block == 0) return false;
  const u64 h = mix64_seeded(key, seed);
  const u64 h1 = mix64(h ^ 0x9E3779B97F4A7C15ULL);
  const u64 h2 = mix64(h ^ 0xBF58476D1CE4E5B9ULL);
  const u32 i0 = (u32)(h % block);
  const u32 i1 = (u32)(h1 % block) + block;
  const u32 i2 = (u32)(h2 % block) + 2u * block;
  const u64 got = xor_load_cell(packed, i0, r) ^ xor_load_cell(packed, i1, r) ^
                  xor_load_cell(packed, i2, r);
  return got == (h & xor_fp_mask(r));
}

__device__ int d_probe(const uint8_t* packed, u32 block, u32 r, u64 seed, u64 fp, u32 lim,
                       u32 unit, u32 d, u32 e,
                       Hit* hits, u32* nhit, u32 hit_cap) {
  (void)lim;  // host filters a,b <= lim after PairRecover
  if (!d_xor_might_contain(packed, block, r, seed, fp)) return 0;
  u32 h = atomicAdd(nhit, 1u);
  if (h < hit_cap) hits[h] = Hit{unit, 0, 0, d, e, fp};
  return 1;
}

struct Unit5D {
  u64 B, u;
  u64 b0, b1, b2;
  u32 lim, mod1, mod2, fmax1, fmax2, nres1, nres2;
  u32 res[24];
};

__global__ void k_cls5_192(const Unit5D* __restrict__ units, int n_units,
                           const uint8_t* __restrict__ xor_cells, u32 xor_block, u32 xor_r, u64 xor_seed,
                           const GateData* __restrict__ gd, int use_gate,
                           u64 inv216, u64 c_m504, u64 c_m247, u64 inv42_247,
                           Hit* __restrict__ hits, u32* __restrict__ nhit, u32 hit_cap,
                           u64* __restrict__ out_calls, u64* __restrict__ out_gated,
                           u64* __restrict__ out_probes) {
  const int ui = blockIdx.x;
  if (ui >= n_units) return;
  const Unit5D U = units[ui];
  const u32 kmax1 = U.fmax1 / U.mod1 + 1;
  const u32 t1 = blockIdx.y;
  if (t1 >= U.nres1 * kmax1) return;
  const u32 e = U.res[t1 % U.nres1] + (t1 / U.nres1) * U.mod1;
  if (e == 0 || e > U.fmax1) return;

  __shared__ u64 s_b0, s_b1, s_b2, s_e0, s_e1, s_e2;
  __shared__ u32 s_res2[12], s_n2, s_mod2, s_fmax2, s_total, s_lim;
  __shared__ GateSh s_gate;
  if (use_gate) gate_load_sh(s_gate, gd);
  if (threadIdx.x == 0) {
    s_b0 = U.b0; s_b1 = U.b1; s_b2 = U.b2;
    d_pow6_192(14u * e, s_e2, s_e1, s_e0);
    s_n2 = U.nres2; s_mod2 = U.mod2; s_fmax2 = U.fmax2; s_lim = U.lim;
    s_total = U.nres2 * (U.fmax2 / U.mod2 + 1);
    for (u32 i = 0; i < U.nres2 && i < 12; ++i) s_res2[i] = U.res[U.nres1 + i];
  }
  __syncthreads();

  u64 lcalls = 0, lgated = 0, lprobes = 0;
  for (u32 t2 = threadIdx.x; t2 < s_total; t2 += blockDim.x) {
    const u32 d = s_res2[t2 % s_n2] + (t2 / s_n2) * s_mod2;
    if (d == 0 || d > s_fmax2) continue;
    ++lcalls;
    u64 r0 = s_b0, r1 = s_b1, r2 = s_b2;
    if (sub192(r2, r1, r0, s_e2, s_e1, s_e0)) continue;
    u64 d2, d1, d0;
    d_pow6_192(21u * d, d2, d1, d0);
    if (sub192(r2, r1, r0, d2, d1, d0)) continue;
    if ((r2 | r1 | r0) == 0) continue;
    if (use_gate) {
      const u64 m504 = 504ULL * M42u;
      const u64 rmod = mod192(r2, r1, r0, m504, c_m504);
      const u32 t504 = (u32)((rmod / M42u) % 504ULL);
      const u64 r247 = mod192(r2, r1, r0, 247ULL, c_m247);
      const u32 t247 = (u32)(r247 * inv42_247 % 247ULL);
      if (!(d_gate_bit(s_gate.g504, (int)t504) && d_gate_bit(s_gate.g247, (int)t247))) {
        ++lgated; continue;
      }
    }
    ++lprobes;
    d_probe(xor_cells, xor_block, xor_r, xor_seed, d_funnel_fp(r1, r0, inv216), s_lim,
            (u32)ui, d, e, hits, nhit, hit_cap);
  }
  atomicAdd((unsigned long long*)out_calls, (unsigned long long)lcalls);
  atomicAdd((unsigned long long*)out_gated, (unsigned long long)lgated);
  atomicAdd((unsigned long long*)out_probes, (unsigned long long)lprobes);
}

static void run_cls5_gpu(const std::vector<Unit5>& units, int N, int xor_r, XorFilter xf,
                         bool use_gate, bool quiet, bool stop_first) {
  size_t free_b = 0, total_b = 0;
  CU(cudaMemGetInfo(&free_b, &total_b));
  const size_t need_bytes = xf.store_bytes();
  fprintf(stderr, "[gpu] free=%.1f / total=%.1f GB; xor needs %.2f GB packed (r=%d)\n",
          free_b / 1e9, total_b / 1e9, need_bytes / 1e9, xor_r);
  if (need_bytes + (size_t)2e9 > free_b) {
    fprintf(stderr,
            "[fatal] not enough free VRAM for xor (need ~%.1f GB + 2 GB slack, "
            "free %.1f GB).\n",
            need_bytes / 1e9, free_b / 1e9);
    exit(1);
  }

  uint8_t* d_xor = nullptr;
  {
    cudaError_t e = cudaMalloc(&d_xor, need_bytes);
    if (e != cudaSuccess) {
      fprintf(stderr, "CUDA cudaMalloc xor (%.1f GB): %s\n",
              need_bytes / 1e9, cudaGetErrorString(e));
      exit(1);
    }
  }
  CU(cudaMemcpy(d_xor, xf.packed.data(), need_bytes, cudaMemcpyHostToDevice));
  const u32 xor_block = xf.hdr.m_cells ? (u32)(xf.hdr.m_cells / 3) : 0;
  const u32 xor_r_u = xf.hdr.r;
  const u64 xor_seed = xf.hdr.mix_seed;
  decltype(xf.packed)().swap(xf.packed);

  PairRecover pair_ix;
  pair_ix.build(N);

  GateData gd; build_gate(gd);
  GateData* d_gd = nullptr;
  CU(cudaMalloc(&d_gd, sizeof(gd)));
  CU(cudaMemcpy(d_gd, &gd, sizeof(gd), cudaMemcpyHostToDevice));

  const u64 inv216 = inv216_mod_2_64();
  const u64 m504 = 504ULL * M42u;
  const u64 c_m504 = (u64)(((u128)1 << 64) % (u128)m504);
  const u64 c_m247 = (u64)(((u128)1 << 64) % (u128)247ULL);
  const u64 inv42_247 = (u64)fc3::mod_inv(fc3::mod_pow6(42, 247), 247);

  const u32 hit_cap = 1u << 20;
  Hit* d_hits = nullptr; u32* d_nh = nullptr;
  u64 *d_calls = nullptr, *d_gated = nullptr, *d_probes = nullptr;
  CU(cudaMalloc(&d_hits, hit_cap * sizeof(Hit)));
  CU(cudaMalloc(&d_nh, sizeof(u32)));
  CU(cudaMalloc(&d_calls, sizeof(u64)));
  CU(cudaMalloc(&d_gated, sizeof(u64)));
  CU(cudaMalloc(&d_probes, sizeof(u64)));

  std::vector<u128> p6(N + 1);
  for (int x = 1; x <= N; ++x) p6[x] = pow6_full((u64)x);

  const int CHUNK = 16;
  u64 tot_calls = 0, tot_gated = 0, tot_probes = 0, sols = 0;
  auto t0 = Clock::now();

  for (size_t off = 0; off < units.size(); off += CHUNK) {
    const int n = (int)std::min((size_t)CHUNK, units.size() - off);
    std::vector<Unit5D> h(n);
    u32 ymax = 1;
    for (int i = 0; i < n; ++i) {
      const Unit5& U = units[off + i];
      h[i] = {};
      h[i].B = U.B; h[i].u = U.u;
      h[i].b0 = U.base.l0; h[i].b1 = U.base.l1; h[i].b2 = U.base.l2;
      h[i].lim = U.lim; h[i].mod1 = U.mod1; h[i].mod2 = U.mod2;
      h[i].fmax1 = U.fmax1; h[i].fmax2 = U.fmax2;
      h[i].nres1 = U.nres1; h[i].nres2 = U.nres2;
      memcpy(h[i].res, U.res, sizeof h[i].res);
      const u32 ye = U.nres1 * (U.fmax1 / U.mod1 + 1);
      if (ye > ymax) ymax = ye;
    }
    Unit5D* d_u = nullptr;
    CU(cudaMalloc(&d_u, n * sizeof(Unit5D)));
    CU(cudaMemcpy(d_u, h.data(), n * sizeof(Unit5D), cudaMemcpyHostToDevice));
    CU(cudaMemset(d_nh, 0, sizeof(u32)));
    CU(cudaMemset(d_calls, 0, sizeof(u64)));
    CU(cudaMemset(d_gated, 0, sizeof(u64)));
    CU(cudaMemset(d_probes, 0, sizeof(u64)));

    k_cls5_192<<<dim3(n, ymax), 256>>>(
        d_u, n, d_xor, xor_block, xor_r_u, xor_seed, d_gd, use_gate ? 1 : 0,
        inv216, c_m504, c_m247, inv42_247,
        d_hits, d_nh, hit_cap, d_calls, d_gated, d_probes);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    u32 nh = 0; u64 calls = 0, gated = 0, probes = 0;
    CU(cudaMemcpy(&nh, d_nh, sizeof(u32), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&calls, d_calls, sizeof(u64), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&gated, d_gated, sizeof(u64), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&probes, d_probes, sizeof(u64), cudaMemcpyDeviceToHost));
    tot_calls += calls; tot_gated += gated; tot_probes += probes;

    if (nh > hit_cap) nh = hit_cap;
    std::vector<Hit> raw(nh);
    if (nh) CU(cudaMemcpy(raw.data(), d_hits, nh * sizeof(Hit), cudaMemcpyDeviceToHost));
    std::vector<std::pair<u32, u32>> pairs;
    for (const Hit& Hr : raw) {
      if (Hr.unit >= (u32)n) continue;
      const Unit5& U = units[off + Hr.unit];
      pair_ix.recover(Hr.fp, pairs);
      for (auto [a, b] : pairs) {
        if (a > U.lim || b > U.lim) continue;
        u128 T = 0;
        if (!compute_T_peel2_gmp(U.B, U.u, Hr.d, Hr.e, T)) continue;
        if (p6[a] + p6[b] != T) continue;
        if (!verify_cls5_gmp(U.B, U.u, Hr.d, Hr.e, a, b)) continue;
        long long terms[5] = {42LL * a, 42LL * b, 21LL * (long long)Hr.d,
                              14LL * (long long)Hr.e, (long long)U.u};
        std::sort(terms, terms + 5);
        bool bad = false;
        for (int i = 0; i < 4; ++i) if (terms[i] == terms[i + 1]) bad = true;
        if (bad || terms[4] >= (long long)U.B) continue;
        long long g = terms[0];
        for (int i = 1; i < 5; ++i) g = std::gcd(g, terms[i]);
        g = std::gcd(g, (long long)U.B);
        printf("SOLUTION cls=5 B=%llu u=%llu a1=%lld a2=%lld a3=%lld a4=%lld a5=%lld gcd=%lld %s\n",
               (unsigned long long)U.B, (unsigned long long)U.u,
               terms[0], terms[1], terms[2], terms[3], terms[4], g,
               g == 1 ? "primitive" : "NON-PRIMITIVE");
        fflush(stdout);
        ++sols;
        if (stop_first) break;
      }
      if (stop_first && sols) break;
    }
    CU(cudaFree(d_u));

    const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    if (!quiet || (off / CHUNK) % 2 == 0 || off + n >= units.size()) {
      const double gate_pct = tot_calls ? 100.0 * (double)tot_gated / (double)tot_calls : 0.0;
      fprintf(stderr,
              "[cls5-gpu] units=%zu/%zu calls=%llu probes=%llu gate=%.1f%% sols=%llu %.1fs\n",
              off + n, units.size(), (unsigned long long)tot_calls,
              (unsigned long long)tot_probes, gate_pct, (unsigned long long)sols, sec);
    }
    if (stop_first && sols) break;
  }

  CU(cudaFree(d_xor)); CU(cudaFree(d_gd));
  CU(cudaFree(d_hits)); CU(cudaFree(d_nh));
  CU(cudaFree(d_calls)); CU(cudaFree(d_gated)); CU(cudaFree(d_probes));
  const double sec = std::chrono::duration<double>(Clock::now() - t0).count();
  printf("---- done cls5-gpu: solutions=%llu units=%zu calls=%llu probes=%llu time=%.1fs ----\n",
         (unsigned long long)sols, units.size(), (unsigned long long)tot_calls,
         (unsigned long long)tot_probes, sec);
}
#endif

// Load or peel packed xor. On --load-table failure, FATAL unless --allow-rebuild
// (OPS-4). Always FPR-smoke; refuse if hits > 16 (OPS-3).
static bool acquire_xor_filter(XorFilter& xf, int N, int& xor_r,
                               const std::string& load_table, const std::string& save_table,
                               bool allow_rebuild) {
  bool loaded = false;
  if (!load_table.empty()) {
    loaded = xor_load_file(load_table.c_str(), xf, N);
    if (!loaded) {
      if (!allow_rebuild) {
        fprintf(stderr,
                "[cls5-gpu] FATAL: --load-table failed (see [xor] lines above). Fix N/path, or "
                "pass --allow-rebuild to peel on the host (needs tens of GB RAM).\n");
        return false;
      }
      fprintf(stderr,
              "[cls5-gpu] --load-table failed; --allow-rebuild: peeling xor on host...\n");
    }
  }
  if (!loaded) xf = xor_build_pairs(N, xor_r);
  else xor_r = (int)xf.hdr.r;
  if (!save_table.empty() && !xor_save_file(save_table.c_str(), xf)) return false;
  const int fpr = xor_fpr_smoke(xf);
  if (fpr < 0 || fpr > 16) {
    fprintf(stderr, "[cls5-gpu] FATAL: xor FPR smoke too high (%d) — refuse to search\n", fpr);
    return false;
  }
  return true;
}

static void usage() {
  printf("usage: fourcore_cls5_gpu_v4 [--selftest-host]\n"
         "   or: fourcore_cls5_gpu_v4 --units FILE.buc [--r BITS]\n"
         "                            [--N K] [--quiet|--no-quiet] [--no-gate]\n"
         "                            [--stop-first] [--device K]\n"
         "                            [--load-table F] [--save-table F]\n"
         "                            [--allow-rebuild]\n"
         "   or: fourcore_cls5_gpu_v4 --check-table --N K --load-table F\n"
         "                            [--r BITS] [--save-table F] [--allow-rebuild]\n"
         "  --load-table F: packed xor from xor_build_save / --save-table.\n"
         "  --save-table F: write packed xor after build or successful load.\n"
         "  --allow-rebuild: if --load-table fails, peel on host (default: FATAL).\n"
         "  --check-table: host-only load/FPR smoke (no GPU; works on _host build).\n");
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  setvbuf(stderr, nullptr, _IOLBF, 0);

  std::string units_path, load_table, save_table;
  double max_table_gb = 80.0;
  int device = 0, S_override = 0, N_override = 0, xor_r = 48;
  bool do_host = false, quiet = true, use_gate = true, stop_first = false;
  bool allow_rebuild = false, check_table = false;

  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
    };
    if (s == "--selftest-host") do_host = true;
    else if (s == "--units") units_path = next();
    else if (s == "--max-table-gb") max_table_gb = atof(next().c_str());  // ignored
    else if (s == "--S") S_override = atoi(next().c_str());  // ignored
    else if (s == "--r") xor_r = atoi(next().c_str());
    else if (s == "--N") N_override = atoi(next().c_str());
    else if (s == "--quiet") quiet = true;
    else if (s == "--no-quiet") quiet = false;
    else if (s == "--no-gate") use_gate = false;
    else if (s == "--stop-first") stop_first = true;
    else if (s == "--device") device = atoi(next().c_str());
    else if (s == "--load-table") load_table = next();
    else if (s == "--save-table") save_table = next();
    else if (s == "--allow-rebuild") allow_rebuild = true;
    else if (s == "--check-table") check_table = true;
    else if (s == "-h" || s == "--help") { usage(); return 0; }
    else { fprintf(stderr, "unknown %s\n", s.c_str()); usage(); return 1; }
  }

  if (do_host || argc == 1) return selftest_host();

  if (xor_r < 8 || xor_r > 64) { fprintf(stderr, "--r must be 8..64\n"); return 1; }

  // Host-only CLI smoke for save/load (Step 3); no CUDA required.
  if (check_table) {
    if (N_override <= 0) {
      fprintf(stderr, "--check-table needs --N\n");
      return 1;
    }
    if (N_override > kXorNSoftMax) {
      fprintf(stderr, "N=%d exceeds soft xor cap %d\n", N_override, kXorNSoftMax);
      return 1;
    }
    XorFilter xf;
    if (!acquire_xor_filter(xf, N_override, xor_r, load_table, save_table, allow_rebuild))
      return 1;
    fprintf(stderr, "[cls5-gpu] --check-table OK N=%d r=%d packed=%.3f GB\n", N_override, xor_r,
            xf.store_gb());
    return 0;
  }

#ifdef HOST_ONLY
  fprintf(stderr, "HOST_ONLY build — GPU run needs: make fourcore-cls5-gpu-v4\n"
                  "  (table smoke: --check-table --N K --load-table F)\n");
  return 1;
#else
  if (units_path.empty()) { usage(); return 1; }
  CU(cudaSetDevice(device));
  size_t free_b = 0, total_b = 0;
  CU(cudaMemGetInfo(&free_b, &total_b));
  fprintf(stderr, "[cls5-gpu] device=%d free=%.1f / total=%.1f GB\n",
          device, free_b / 1e9, total_b / 1e9);

  fc3::RootTables rt;
  std::vector<BucLine> lines;
  if (!load_buc5(units_path.c_str(), lines)) {
    fprintf(stderr, "no cls5 units in %s\n", units_path.c_str());
    return 1;
  }
  std::vector<Unit5> units;
  units.reserve(lines.size());
  u32 need = 0;
  for (auto& L : lines) {
    Unit5 U;
    if (!pack_unit5(rt, L.B, L.u, U)) continue;
    need = std::max(need, U.lim);
    units.push_back(U);
  }
  fprintf(stderr, "[cls5-gpu] packed %zu/%zu units from %s\n",
          units.size(), lines.size(), units_path.c_str());
  if (units.empty()) return 1;
  int N = N_override > 0 ? N_override : (int)need;
  if (N > kXorNSoftMax) {
    fprintf(stderr, "N=%d exceeds soft xor cap %d\n", N, kXorNSoftMax);
    return 1;
  }
  (void)S_override; (void)max_table_gb;
  fprintf(stderr, "[cls5-gpu] N=%d r=%d\n", N, xor_r);
  XorFilter xf;
  if (!acquire_xor_filter(xf, N, xor_r, load_table, save_table, allow_rebuild)) return 1;
  run_cls5_gpu(units, N, xor_r, std::move(xf), use_gate, quiet, stop_first);
  return 0;
#endif
}
