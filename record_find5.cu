// record_find5.cu  —  GPU find5 for record-style jobs from record_hunt
//
// GOAL: find ONE large B with seven sixth-powers summing to B^6.
// Completeness is explicitly NOT required — defaults cut the search hard
// (narrow x5-top band, stop at first hit). That may skip valid B.
//
// Job line:  B u v
// Inner:     x1^6+...+x5^6 = T = (B^6 - u^6 - v^6) / 294^6 ,  xi <= B/294
//
// Pair table uses u32 indices (N may exceed 65535).
//
// Build on server:
//   nvcc -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 \
//        -o record_find5 record_find5.cu -Xcompiler -fopenmp
//
// Host-only smoke (Mac):
//   g++ -O3 -std=c++17 -DHOST_ONLY -o record_find5_host record_find5.cu
//   ./record_find5_host --selftest-host
//
// Server:
//   ./runs/record_evening.sh
// =============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
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
#define CU(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)
#endif

using u32  = uint32_t;
using u64  = uint64_t;
using u128 = unsigned __int128;

using Clock = std::chrono::steady_clock;

// ------------------------------------------------------------------ u256 ------
struct u256 { u128 lo = 0, hi = 0; };
static inline int u256_cmp(u256 a, u256 b) {
    if (a.hi != b.hi) return (a.hi < b.hi) ? -1 : 1;
    if (a.lo != b.lo) return (a.lo < b.lo) ? -1 : 1;
    return 0;
}
static inline u256 u256_sub(u256 a, u256 b) {
    u256 r; r.lo = a.lo - b.lo; r.hi = a.hi - b.hi - (a.lo < b.lo); return r;
}
static inline u256 u256_mul_u128(u128 a, u128 b) {
    const u64 a0 = (u64)a, a1 = (u64)(a >> 64);
    const u64 b0 = (u64)b, b1 = (u64)(b >> 64);
    const u128 p00 = (u128)a0 * b0, p01 = (u128)a0 * b1;
    const u128 p10 = (u128)a1 * b0, p11 = (u128)a1 * b1;
    u256 r; r.lo = (u64)p00;
    u128 acc = (p00 >> 64) + (u64)p01 + (u64)p10;
    r.lo |= (acc << 64);
    r.hi = p11 + (p01 >> 64) + (p10 >> 64) + (acc >> 64);
    return r;
}
static inline u256 ipow6_u256(u64 x) {
    const u128 x2 = (u128)x * x;
    return u256_mul_u128(x2 * x2, x2);
}
static inline u128 pow6_u128(u64 x) {
    const u128 x2 = (u128)x * x;
    return (x2 * x2) * x2;
}
static bool u256_div_u64_exact(u256 n, u64 d, u128& q) {
    if (!d) return false;
    u128 lo = 0, hi = (n.hi == 0) ? n.lo : ((u128)1 << 120);
    while (lo < hi) {
        const u128 mid = lo + ((hi - lo + 1) >> 1);
        if (u256_cmp(u256_mul_u128(mid, (u128)d), n) <= 0) lo = mid;
        else hi = mid - 1;
    }
    if (u256_cmp(u256_mul_u128(lo, (u128)d), n) != 0) return false;
    q = lo; return true;
}
static u64 iroot6_u128(u128 n) {
    if (n < 1) return 0;
    u64 lo = 0, hi = 1;
    while (hi < 2500000u && pow6_u128(hi) <= n) {
        lo = hi;
        if (hi > 1200000u) { hi = 2500000u; break; }
        hi <<= 1;
    }
    while (lo + 1 < hi) {
        const u64 mid = lo + ((hi - lo) >> 1);
        if (pow6_u128(mid) <= n) lo = mid; else hi = mid;
    }
    return lo;
}

static constexpr u64 F294 = 294;
static constexpr u64 F294_6 = 645779095649856ULL;

static bool compute_T(u64 B, u64 u, u64 v, u128& T) {
    u256 num = ipow6_u256(B);
    num = u256_sub(num, ipow6_u256(u));
    if (u256_cmp(num, ipow6_u256(v)) < 0) return false;
    num = u256_sub(num, ipow6_u256(v));
    return u256_div_u64_exact(num, F294_6, T);
}

struct Job { u64 B, u, v; u128 T; u32 lim; };

static bool load_jobs(const char* path, std::vector<Job>& out) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    char line[256];
    while (fgets(line, sizeof line, f)) {
        u64 B=0, u=0, v=0;
        if (sscanf(line, "%llu %llu %llu",
                   (unsigned long long*)&B, (unsigned long long*)&u,
                   (unsigned long long*)&v) != 3) continue;
        u128 T = 0;
        if (!compute_T(B, u, v, T) || T < 5) continue;
        Job J; J.B = B; J.u = u; J.v = v; J.T = T;
        J.lim = (u32)(B / F294);
        if (J.lim < 2) continue;
        if (T > pow6_u128(J.lim) * 5) continue;
        out.push_back(J);
    }
    fclose(f);
    fprintf(stderr, "[jobs] loaded %zu from %s\n", out.size(), path);
    return true;
}

// Known record job (regression).
static Job known_job() {
    Job J;
    J.B = 21934895; J.u = 8306212; J.v = 19424433;
    if (!compute_T(J.B, J.u, J.v, J.T)) { fprintf(stderr, "known T fail\n"); exit(1); }
    J.lim = (u32)(J.B / F294);
    return J;
}

// =============================================================================
// Pair table — empty slot marked by i==0
// =============================================================================
struct Slot { u64 key; u32 i; u32 j; };

static constexpr u64 PHI64 = 0x9E3779B97F4A7C15ULL;
static inline u64 mix64(u64 x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}
static inline u64 hash_pos(u64 fp, int S) { return (fp * PHI64) >> (64 - S); }

// Choose table size: prefer LF <= target_lf, but never exceed max_bytes on GPU.
// On a 102 GB card, default ~80 GB leaves headroom for hits/cands/workspace.
static int choose_S(int N, double max_gb = 80.0, double target_lf = 0.75) {
    const double pairs = (double)N * (N + 1) / 2;
    const size_t max_slots = (size_t)((max_gb * 1e9) / 16.0);
    int Smax = 1;
    while (Smax < 40 && ((size_t)1 << (Smax + 1)) <= max_slots) ++Smax;

    int S = 1;
    while (S < Smax && ((size_t)1 << S) * target_lf < pairs) ++S;
    const double lf = pairs / (double)((size_t)1 << S);
    if (lf > 0.92) {
        fprintf(stderr,
                "[table] FATAL: N=%d needs LF=%.3f at max S=%d (%.1f GB cap); "
                "lower --N or raise --max-table-gb\n",
                N, lf, S, max_gb);
        exit(1);
    }
    if (lf > 0.70)
        fprintf(stderr, "[table] note: LF=%.3f at S=%d (capped by %.0f GB budget)\n", lf, S, max_gb);
    return S;
}

// Find-ONE hunt options (shared host/CUDA). Completeness is NOT the goal.
struct HuntOpts {
    u64 x5_top = 3000;      // 0 = full window
    bool stop_first = true;
};

static void table_build(int N, int S, std::vector<Slot>& slots) {
    const size_t size = (size_t)1 << S;
    const double pairs_est = (double)N * (N + 1) / 2;
    if ((double)size * 0.95 < pairs_est) {
        fprintf(stderr, "[table] FATAL LF too high; increase S\n");
        exit(1);
    }
    slots.assign(size, Slot{0, 0, 0});
    std::vector<u64> pw6(N + 1);
#pragma omp parallel for schedule(static)
    for (int x = 1; x <= N; ++x) {
        u64 x2 = (u64)x * x;
        pw6[x] = x2 * x2 * x2;
    }
    const u64 mask = size - 1;
    std::atomic<u64> steps{0};
    const auto t0 = Clock::now();
#pragma omp parallel for schedule(static)
    for (int i = 1; i <= N; ++i) {
        u64 local = 0;
        for (int j = i; j <= N; ++j) {
            const u64 fp = pw6[i] + pw6[j];
            u64 pos = hash_pos(fp, S);
            const u64 step = mix64(fp) | 1ULL;
            for (;;) {
                ++local;
                if (__sync_bool_compare_and_swap((u32*)&slots[pos].i, 0u, (u32)i)) {
                    slots[pos].j = (u32)j;
                    slots[pos].key = fp;   // publish key last so probes seeing i≠0 wait for key
                    break;
                }
                pos = (pos + step) & mask;
            }
        }
        steps += local;
    }
    const double ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    fprintf(stderr, "[table] N=%d pairs=%.3e slots=2^%d (%.1f GB) LF=%.3f avg probes=%.2f build=%.0f ms\n",
            N, pairs_est, S, size * 16.0 / 1e9, pairs_est / size, steps.load() / pairs_est, ms);
}

// =============================================================================
#ifndef HOST_ONLY
// Kernel mirrors solve_617 k_find4: warp-strided c4, lane-strided c3, pair probe.
// Host peels c5 with --x5-top (find-ONE cut; record needs ~1.8k below iroot6(T)).
// =============================================================================
struct Cand4 {
    u64 q_lo, q_hi;
    u32 c5, lim;
    u32 c4lo, c4hi;
    u32 job;
};

struct Hit {
    u32 job, c5, c4, c3, a, b;
};

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
    // x^6 for x < 2^21 fits in u128; split for subtract-with-borrow
    u64 x2 = (u64)x * x;
    u64 x4lo = x2 * x2;          // may wrap — use __umul64hi path for hi
    // For our N<=1e5, x^6 < 2^100 roughly; compute via double for hi estimate
    // Exact: use 32-bit schoolbook on x2 * x4 when x2 fits.
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
    // one-step bump using low64 only (good enough for windowing)
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
__device__ __forceinline__ u32 d_min_c3(u64 rhi, u64 rlo, u32 hi3) {
    // lo such that 3*lo^6 <= rem  (approx hi3 * 3^{-1/6} ≈ 0.8*hi3)
    u32 lo = (u32)(hi3 * 0.8);
    if (lo < 1) lo = 1;
    return lo;
}

__device__ int d_probe(const Slot* tab, u64 mask, int S, u64 fp, u32 c3,
                       u32 job, u32 c5, u32 c4,
                       Hit* hits, u32* nhit, u32 hit_cap) {
    u64 pos = (fp * 0x9E3779B97F4A7C15ULL) >> (64 - S);
    const u64 step = d_mix64(fp) | 1ULL;
    for (int k = 0; k < 64; ++k) {
        const Slot s = tab[pos];
        if (s.i == 0) return 0;
        if (s.key == fp && s.j <= c3 && s.i >= 1) {
            u32 h = atomicAdd(nhit, 1u);
            if (h < hit_cap)
                hits[h] = Hit{job, c5, c4, c3, s.i, s.j};
            return 1;
        }
        pos = (pos + step) & mask;
    }
    return 0;
}

// Grid: (n_cand, ytiles)  ytiles cover c4 window in chunks of 2048*8 warps pattern
__global__ void k_find4(const Cand4* __restrict__ cands, int nc,
                        const Slot* __restrict__ tab, u64 mask, int S,
                        Hit* __restrict__ hits, u32* __restrict__ nhit, u32 hit_cap) {
    const u32 ci = blockIdx.x;
    if ((int)ci >= nc) return;
    const Cand4 C = cands[ci];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const u32 base = C.c4lo + blockIdx.y * 2048u + (u32)warp;
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
        u32 lo3 = d_min_c3(rhi, rlo, hi3);
        if (lo3 > hi3) continue;
        // fingerprint uses low64 of Q4 - c4^6 (wrapping), same as host table keys
        const u64 tfp = C.q_lo - d_pow6_64(c4);
        for (u32 c3 = lo3 + (u32)lane; c3 <= hi3; c3 += 32) {
            d_probe(tab, mask, S, tfp - d_pow6_64(c3), c3,
                    C.job, C.c5, c4, hits, nhit, hit_cap);
        }
    }
}

static void gpu_find5_jobs(const std::vector<Job>& jobs, int N, int S,
                           const std::vector<Slot>& slots,
                           const HuntOpts& opt,
                           std::vector<Hit>& out_hits) {
    Slot* d_tab = nullptr;
    CU(cudaMalloc(&d_tab, slots.size() * sizeof(Slot)));
    CU(cudaMemcpy(d_tab, slots.data(), slots.size() * sizeof(Slot), cudaMemcpyHostToDevice));

    const u32 hit_cap = 1u << 20;
    Hit* d_hits = nullptr; u32* d_nh = nullptr;
    CU(cudaMalloc(&d_hits, hit_cap * sizeof(Hit)));
    CU(cudaMalloc(&d_nh, sizeof(u32)));

    std::vector<u128> p6full(N + 1);
    for (int x = 1; x <= N; ++x) p6full[x] = pow6_u128((u64)x);

    bool stop = false;
    for (size_t ji = 0; ji < jobs.size() && !stop; ++ji) {
        const Job& J = jobs[ji];
        const u32 lim = std::min(J.lim, (u32)N);
        const u64 hi5 = std::min<u64>(iroot6_u128(J.T), lim);
        u64 lo5_full = (u64)(hi5 * 0.764706);
        if (lo5_full < 1) lo5_full = 1;
        while (lo5_full <= hi5 && p6full[lo5_full] * 5 < J.T) ++lo5_full;
        u64 lo5 = lo5_full;
        if (opt.x5_top > 0 && hi5 + 1 > opt.x5_top)
            lo5 = std::max(lo5_full, hi5 - opt.x5_top + 1);

        // Emit c5 descending: hit dominant-max solutions first (find-one posture)
        std::vector<Cand4> cands;
        cands.reserve((size_t)(hi5 - lo5 + 1));
        for (u64 c5 = hi5; c5 >= lo5; --c5) {
            const u128 Q4 = J.T - p6full[c5];
            if (Q4 < 4) { if (c5 == lo5) break; continue; }
            u64 hi4 = std::min<u64>({iroot6_u128(Q4), c5, (u64)lim});
            u64 lo4 = (u64)(hi4 * 0.7937005260);
            if (lo4 < 1) lo4 = 1;
            while (lo4 <= hi4 && p6full[lo4] * 4 < Q4) ++lo4;
            while (lo4 > 1 && p6full[lo4 - 1] * 4 >= Q4) --lo4;
            if (lo4 <= hi4) {
                Cand4 C;
                C.q_lo = (u64)Q4; C.q_hi = (u64)(Q4 >> 64);
                C.c5 = (u32)c5; C.lim = (u32)c5;
                C.c4lo = (u32)lo4; C.c4hi = (u32)hi4;
                C.job = (u32)ji;
                cands.push_back(C);
            }
            if (c5 == lo5) break;
        }
        if (cands.empty()) continue;
        fprintf(stderr, "[gpu] job %zu/%zu B=%llu c5-cands=%zu (x5-top=%llu hi5=%llu lo5=%llu)\n",
                ji + 1, jobs.size(), (unsigned long long)J.B, cands.size(),
                (unsigned long long)opt.x5_top, (unsigned long long)hi5, (unsigned long long)lo5);

        const size_t BATCH = 8192;   // one c5 each; keep grids modest, drain hits often
        for (size_t off = 0; off < cands.size() && !stop; off += BATCH) {
            const size_t n = std::min(BATCH, cands.size() - off);
            // y-tiles for widest c4 window in this batch
            u32 maxspan = 1;
            for (size_t i = 0; i < n; ++i) {
                u32 span = cands[off + i].c4hi - cands[off + i].c4lo + 1;
                if (span > maxspan) maxspan = span;
            }
            u32 ytiles = (maxspan + 2047) / 2048;
            if (ytiles < 1) ytiles = 1;

            Cand4* d_c = nullptr;
            CU(cudaMalloc(&d_c, n * sizeof(Cand4)));
            CU(cudaMemcpy(d_c, cands.data() + off, n * sizeof(Cand4), cudaMemcpyHostToDevice));
            CU(cudaMemset(d_nh, 0, sizeof(u32)));
            k_find4<<<dim3((unsigned)n, ytiles), 256>>>(
                d_c, (int)n, d_tab, slots.size() - 1, S, d_hits, d_nh, hit_cap);
            CU(cudaDeviceSynchronize());
            u32 nh = 0;
            CU(cudaMemcpy(&nh, d_nh, sizeof(u32), cudaMemcpyDeviceToHost));
            if (nh > hit_cap) {
                fprintf(stderr, "!! hit overflow — raise hit_cap\n");
                nh = hit_cap;
            }
            std::vector<Hit> raw(nh);
            if (nh) CU(cudaMemcpy(raw.data(), d_hits, nh * sizeof(Hit), cudaMemcpyDeviceToHost));
            CU(cudaFree(d_c));

            for (const Hit& H : raw) {
                const Job& JJ = jobs[H.job];
                const u128 sum = p6full[H.a] + p6full[H.b] + p6full[H.c3]
                              + p6full[H.c4] + p6full[H.c5];
                if (sum != JJ.T) continue;
                if (!(H.a <= H.b && H.b <= H.c3 && H.c3 <= H.c4 && H.c4 <= H.c5)) continue;
                if (H.c5 > JJ.lim) continue;
                out_hits.push_back(H);
                long long terms[7] = {
                    (long long)F294 * H.a, (long long)F294 * H.b, (long long)F294 * H.c3,
                    (long long)F294 * H.c4, (long long)F294 * H.c5,
                    (long long)JJ.u, (long long)JJ.v
                };
                std::sort(terms, terms + 7);
                printf("SOLUTION B=%llu u=%llu v=%llu x=(%u,%u,%u,%u,%u)\n",
                       (unsigned long long)JJ.B, (unsigned long long)JJ.u,
                       (unsigned long long)JJ.v, H.a, H.b, H.c3, H.c4, H.c5);
                printf("  terms: %lld %lld %lld %lld %lld %lld %lld\n",
                       terms[0], terms[1], terms[2], terms[3], terms[4], terms[5], terms[6]);
                fflush(stdout);
                if (opt.stop_first) { stop = true; break; }
            }
        }
    }
    CU(cudaFree(d_tab)); CU(cudaFree(d_hits)); CU(cudaFree(d_nh));
}
#endif // !HOST_ONLY

// =============================================================================
static int selftest_host() {
    printf("[selftest-host] known T + job load math\n");
    Job K = known_job();
    const int xs[5] = {19555, 30505, 33238, 46958, 65063};
    u128 Tsum = 0;
    for (int x : xs) Tsum += pow6_u128((u64)x);
    if (Tsum != K.T) { printf("T mismatch\n"); return 1; }
    printf("  known T ok  lim=%u  iroot6(T)=%llu\n",
           K.lim, (unsigned long long)iroot6_u128(K.T));

    // write tiny job file and reload
    const char* path = "/tmp/record_find5_known.buv";
    FILE* f = fopen(path, "w");
    fprintf(f, "%llu %llu %llu\n",
            (unsigned long long)K.B, (unsigned long long)K.u, (unsigned long long)K.v);
    fclose(f);
    std::vector<Job> jobs;
    if (!load_jobs(path, jobs) || jobs.size() != 1) { printf("load fail\n"); return 1; }
    if (jobs[0].T != K.T) { printf("reload T fail\n"); return 1; }
    printf("  job I/O ok\n");

    // table S choice for N=1000 (tiny — local only)
    int N = 1000, S = choose_S(N);
    std::vector<Slot> slots;
    table_build(N, S, slots);
    // probe a planted pair
    u64 fp = 0;
    { u64 a=3,b=5; u64 a2=a*a,b2=b*b; fp = a2*a2*a2 + b2*b2*b2; }
    u64 pos = hash_pos(fp, S), step = mix64(fp) | 1, mask = slots.size() - 1;
    bool found = false;
    for (int k = 0; k < 64; ++k) {
        if (slots[pos].i == 3 && slots[pos].j == 5 && slots[pos].key == fp) { found = true; break; }
        if (slots[pos].i == 0) break;
        pos = (pos + step) & mask;
    }
    printf("  tiny table plant (3,5): %s\n", found ? "ok" : "FAIL");
    printf("[selftest-host] %s\n", found ? "PASS" : "FAIL");
    return found ? 0 : 1;
}

static void usage() {
    printf("usage: record_find5 [--selftest-host] [--jobs file] [--N n]\n"
           "                    [--known-first] [--x5-top K] [--no-stop-first]\n"
           "                    [--S bits] [--max-table-gb G]\n"
           "  find-ONE posture (default): x5-top=3000, stop at first exact hit.\n"
           "  table: default max ~80 GB (fits 102 GB GPUs); override with --S / --max-table-gb.\n"
           "  completeness: --x5-top 0 --no-stop-first\n");
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IOLBF, 0);

    std::string jobs_path;
    int N = 78000;
    int S_override = 0;          // 0 = auto
    double max_table_gb = 80.0;  // fits RTX PRO 6000 102 GB with headroom
    bool known_first = false, do_host = false;
    u64 x5_top = 3000;
    bool stop_first = true;
    std::string save_tbl, load_tbl;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (s == "--selftest-host") do_host = true;
        else if (s == "--jobs") jobs_path = next();
        else if (s == "--N") N = atoi(next().c_str());
        else if (s == "--S") S_override = atoi(next().c_str());
        else if (s == "--max-table-gb") max_table_gb = atof(next().c_str());
        else if (s == "--known-first") known_first = true;
        else if (s == "--x5-top") x5_top = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--no-stop-first") stop_first = false;
        else if (s == "--save-table") save_tbl = next();
        else if (s == "--load-table") load_tbl = next();
        else if (s == "-h" || s == "--help") { usage(); return 0; }
        else { printf("unknown %s\n", s.c_str()); usage(); return 1; }
    }

    if (do_host || argc == 1) return selftest_host();

#ifdef HOST_ONLY
    (void)x5_top; (void)stop_first; (void)known_first; (void)save_tbl; (void)load_tbl;
    fprintf(stderr, "HOST_ONLY build: use --selftest-host, or rebuild with nvcc for GPU search\n");
    if (!jobs_path.empty()) {
        std::vector<Job> jobs;
        load_jobs(jobs_path.c_str(), jobs);
        printf("[host] %zu jobs ready (GPU search requires nvcc build)\n", jobs.size());
    }
    return 0;
#else
    HuntOpts hop;
    hop.x5_top = x5_top;
    hop.stop_first = stop_first;

    std::vector<Job> jobs;
    if (known_first) jobs.push_back(known_job());
    if (!jobs_path.empty()) {
        std::vector<Job> more;
        if (!load_jobs(jobs_path.c_str(), more)) return 1;
        jobs.insert(jobs.end(), more.begin(), more.end());
    }
    if (jobs.empty()) { usage(); return 1; }

    fprintf(stderr, "[hunt] find-ONE mode: x5-top=%llu stop-first=%s  (%zu jobs)\n",
            (unsigned long long)hop.x5_top, hop.stop_first ? "yes" : "no", jobs.size());

    u32 need = 0;
    for (auto& J : jobs) need = std::max(need, J.lim);
    if ((u32)N < need) {
        fprintf(stderr, "[warn] N=%d < max lim=%u — bumping N\n", N, need);
        N = (int)need;
    }
    if (N > 100000) {
        fprintf(stderr, "[fatal] N=%d too large for full table on ~100GB; need partitioned later\n", N);
        return 1;
    }

    int S = S_override > 0 ? S_override : choose_S(N, max_table_gb);
    fprintf(stderr, "[table] using S=%d (%.1f GB slots) max-budget=%.0f GB\n",
            S, ((size_t)1 << S) * 16.0 / 1e9, max_table_gb);
    std::vector<Slot> slots;
    table_build(N, S, slots);

    std::vector<Hit> hits;
    gpu_find5_jobs(jobs, N, S, slots, hop, hits);
    printf("---- done: exact hits=%zu ----\n", hits.size());
    return 0;
#endif
}
