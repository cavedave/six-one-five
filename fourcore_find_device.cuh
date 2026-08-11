#pragma once
// Shared GPU find2/find3 probe path for fourcore_find_v4 and solve_624_v1.
// Host types GateData / Cand / Hit are declared by the including .cu before this
// header when HOST_ONLY is off; device kernels need CUDA.

#include "spike/ribbon/mix64.hpp"
#include "spike/ribbon/xor_pack.hpp"

#include <cstdint>

#ifndef HOST_ONLY

using fc_dev_u64 = std::uint64_t;
using fc_dev_u32 = std::uint32_t;
using fc_dev_u16 = std::uint16_t;
using fc_dev_u128 = unsigned __int128;

// Expect including TU to define:
//   struct GateData { u64 g504[8]; u64 g247[4]; u16 p6504[504]; u16 p6247[247]; };
//   struct Hit { u32 job, a, b, c, d; u64 fp; };
//   struct Cand { u64 q_lo, q_hi; u32 lim, lo, hi, job, T_r504, T_r247, pad; };

__device__ __forceinline__ fc_dev_u64 fc_d_pow6_64(fc_dev_u32 x) {
  fc_dev_u64 x2 = (fc_dev_u64)x * x;
  return x2 * x2 * x2;
}
__device__ __forceinline__ void fc_d_pow6_128(fc_dev_u32 x, fc_dev_u64& h, fc_dev_u64& l) {
  fc_dev_u128 x2w = (fc_dev_u128)x * x;
  fc_dev_u128 x6 = (x2w * x2w) * x2w;
  l = (fc_dev_u64)x6;
  h = (fc_dev_u64)(x6 >> 64);
}
__device__ __forceinline__ fc_dev_u32 fc_d_iroot6(fc_dev_u64 rhi, fc_dev_u64 rlo) {
  double x = ldexp((double)rhi, 64) + (double)rlo;
  if (x < 1.0) return 0;
  double r = pow(x, 1.0 / 6.0);
  if (r > 4294967294.0) return 0xffffffffu;
  fc_dev_u32 a = (fc_dev_u32)r;
  while (a < 0xfffffffeu) {
    fc_dev_u64 h, l;
    fc_d_pow6_128(a + 1, h, l);
    if (h < rhi || (h == rhi && l <= rlo)) ++a;
    else break;
  }
  while (a > 0) {
    fc_dev_u64 h, l;
    fc_d_pow6_128(a, h, l);
    if (h < rhi || (h == rhi && l <= rlo)) break;
    --a;
  }
  return a;
}

__device__ __forceinline__ bool fc_d_xor_might_contain(const uint8_t* packed, fc_dev_u32 block,
                                                       fc_dev_u32 r, fc_dev_u64 seed,
                                                       fc_dev_u64 key) {
  if (block == 0) return false;
  const fc_dev_u64 h = mix64_seeded(key, seed);
  const fc_dev_u64 h1 = mix64(h ^ 0x9E3779B97F4A7C15ULL);
  const fc_dev_u64 h2 = mix64(h ^ 0xBF58476D1CE4E5B9ULL);
  const fc_dev_u32 i0 = (fc_dev_u32)(h % block);
  const fc_dev_u32 i1 = (fc_dev_u32)(h1 % block) + block;
  const fc_dev_u32 i2 = (fc_dev_u32)(h2 % block) + 2u * block;
  const fc_dev_u64 got = xor_load_cell(packed, i0, r) ^ xor_load_cell(packed, i1, r) ^
                         xor_load_cell(packed, i2, r);
  return got == (h & xor_fp_mask(r));
}

// On xor maybe-hit: record fp (a,b=0). Host PairRecover expands pairs with i<=j.
template <typename HitT>
__device__ int fc_d_probe(const uint8_t* packed, fc_dev_u32 block, fc_dev_u32 r, fc_dev_u64 seed,
                          fc_dev_u64 fp, fc_dev_u32 c_ctx, fc_dev_u32 d_ctx, fc_dev_u32 job,
                          HitT* hits, fc_dev_u32* nhit, fc_dev_u32 hit_cap) {
  if (!fc_d_xor_might_contain(packed, block, r, seed, fp)) return 0;
  fc_dev_u32 h = atomicAdd(nhit, 1u);
  if (h < hit_cap) hits[h] = HitT{job, 0, 0, c_ctx, d_ctx, fp};
  return 1;
}

struct FcGateSh {
  fc_dev_u64 g504[8];
  fc_dev_u64 g247[4];
  fc_dev_u16 p6504[504];
  fc_dev_u16 p6247[247];
};
__device__ __forceinline__ bool fc_d_gate_bit(const fc_dev_u64* b, int x) {
  return (b[x >> 6] >> (x & 63)) & 1ULL;
}
template <typename GateDataT>
__device__ __forceinline__ void fc_gate_load_sh(FcGateSh& gs, const GateDataT* gd) {
  const int tid = threadIdx.x;
  for (int k = tid; k < 8; k += blockDim.x) gs.g504[k] = gd->g504[k];
  for (int k = tid; k < 4; k += blockDim.x) gs.g247[k] = gd->g247[k];
  for (int k = tid; k < 504; k += blockDim.x) gs.p6504[k] = gd->p6504[k];
  for (int k = tid; k < 247; k += blockDim.x) gs.p6247[k] = gd->p6247[k];
  __syncthreads();
}

template <typename CandT, typename GateDataT, typename HitT>
__global__ void fc_k_find2(const CandT* __restrict__ cands, int nc,
                           const uint8_t* __restrict__ xor_cells, fc_dev_u32 xor_block,
                           fc_dev_u32 xor_r, fc_dev_u64 xor_seed,
                           const GateDataT* __restrict__ gd, int use_gate,
                           HitT* __restrict__ hits, fc_dev_u32* __restrict__ nhit,
                           fc_dev_u32 hit_cap, fc_dev_u64* __restrict__ out_calls,
                           fc_dev_u64* __restrict__ out_gated,
                           fc_dev_u64* __restrict__ out_probes) {
  const int ci = blockIdx.x;
  if (ci >= nc) return;
  if (threadIdx.x != 0) return;
  const CandT C = cands[ci];
  atomicAdd((unsigned long long*)out_calls, 1ull);
  if (use_gate) {
    if (!(fc_d_gate_bit(gd->g504, (int)C.T_r504) && fc_d_gate_bit(gd->g247, (int)C.T_r247))) {
      atomicAdd((unsigned long long*)out_gated, 1ull);
      return;
    }
  }
  atomicAdd((unsigned long long*)out_probes, 1ull);
  fc_d_probe(xor_cells, xor_block, xor_r, xor_seed, C.q_lo, C.lim, 0, C.job, hits, nhit,
             hit_cap);
}

template <typename CandT, typename GateDataT, typename HitT>
__global__ void fc_k_find3(const CandT* __restrict__ cands, int nc,
                           const uint8_t* __restrict__ xor_cells, fc_dev_u32 xor_block,
                           fc_dev_u32 xor_r, fc_dev_u64 xor_seed,
                           const GateDataT* __restrict__ gd, int use_gate,
                           HitT* __restrict__ hits, fc_dev_u32* __restrict__ nhit,
                           fc_dev_u32 hit_cap, fc_dev_u64* __restrict__ out_calls,
                           fc_dev_u64* __restrict__ out_gated,
                           fc_dev_u64* __restrict__ out_probes) {
  const fc_dev_u32 ci = blockIdx.x;
  if ((int)ci >= nc) return;
  const CandT C = cands[ci];
  __shared__ FcGateSh gs;
  if (use_gate) fc_gate_load_sh(gs, gd);
  fc_dev_u64 lcalls = 0, lgated = 0, lprobes = 0;
  for (fc_dev_u32 c3 = C.lo + threadIdx.x; c3 <= C.hi; c3 += blockDim.x) {
    ++lcalls;
    if (use_gate) {
      fc_dev_u32 r504 = C.T_r504 + 504u - gs.p6504[c3 % 504];
      if (r504 >= 504u) r504 -= 504u;
      fc_dev_u32 r247 = C.T_r247 + 247u - gs.p6247[c3 % 247];
      if (r247 >= 247u) r247 -= 247u;
      if (!(fc_d_gate_bit(gs.g504, (int)r504) && fc_d_gate_bit(gs.g247, (int)r247))) {
        ++lgated;
        continue;
      }
    }
    ++lprobes;
    fc_d_probe(xor_cells, xor_block, xor_r, xor_seed, C.q_lo - fc_d_pow6_64(c3), c3, 0, C.job,
               hits, nhit, hit_cap);
  }
  atomicAdd((unsigned long long*)out_calls, (unsigned long long)lcalls);
  atomicAdd((unsigned long long*)out_gated, (unsigned long long)lgated);
  atomicAdd((unsigned long long*)out_probes, (unsigned long long)lprobes);
}

#endif  // !HOST_ONLY
