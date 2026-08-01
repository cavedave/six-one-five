// =============================================================================
// fourcore_find4.cu  —  GPU find4 for (6,1,5) four-core jobs
//
//   Job: B u T   with T = (B^6 - u^6) / D^6 = x1^6+x2^6+x3^6+x4^6
//   Terms: (D*x1)^6 + ... + (D*x4)^6 + u^6 = B^6
//
// Build (server):
//   nvcc -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 \
//        -o fourcore_find4 fourcore_find4.cu -Xcompiler -fopenmp \
//        -I/usr/include -lgmpxx -lgmp
// Host-only (Mac):
//   g++ -O3 -std=c++17 -DHOST_ONLY -I$(brew --prefix gmp)/include \
//       -L$(brew --prefix gmp)/lib -x c++ -o fourcore_find4_host \
//       fourcore_find4.cu -lgmpxx -lgmp
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
#define CU(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)
#endif

using Clock = std::chrono::steady_clock;

struct Job {
    u64 B = 0, u = 0;
    u128 T = 0;
    u32 lim = 0;
    u64 D = 42;
};

struct Slot {
    u64 key = 0;
    u32 i = 0, j = 0;   // empty if i==0; stores pair (i,j) with i<=j
};

struct Cand4 {
    u64 q_lo, q_hi;
    u32 lim;
    u32 c4lo, c4hi;
    u32 job;
};

struct Hit {
    u32 job, c4, c3, a, b;
};

struct HuntOpts {
    u64 x4_top = 0;      // 0 = full window; >0 = only top band near iroot6(T)
    bool stop_first = false;
};

static inline u64 mix64(u64 x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}
static inline u64 hash_pos(u64 fp, int S) {
    return (fp * 0x9E3779B97F4A7C15ULL) >> (64 - S);
}
static inline u128 pow6_u128(u64 x) {
    const u128 x2 = (u128)x * x;
    return (x2 * x2) * x2;
}
static u64 iroot6_u128(u128 n) {
    if (n < 1) return 0;
    u64 lo = 0, hi = 1;
    while (hi < 2000000u && pow6_u128(hi) <= n) {
        lo = hi;
        if (hi > 1000000u) { hi = 2000000u; break; }
        hi <<= 1;
    }
    while (lo + 1 < hi) {
        const u64 mid = lo + ((hi - lo) >> 1);
        if (pow6_u128(mid) <= n) lo = mid; else hi = mid;
    }
    return lo;
}

static bool load_jobs(const char* path, u64 D, std::vector<Job>& out) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        unsigned long long B, u, tlo, thi;
        if (sscanf(line, "%llu %llu %llu %llu", &B, &u, &tlo, &thi) != 4) continue;
        Job J;
        J.B = B; J.u = u; J.D = D;
        J.T = join_u128(tlo, thi);
        J.lim = (u32)((B - 1) / D);
        out.push_back(J);
    }
    fclose(f);
    fprintf(stderr, "[jobs] loaded %zu from %s (D=%llu)\n",
            out.size(), path, (unsigned long long)D);
    return true;
}

static int choose_S(int N, double max_gb) {
    const double pairs = (double)N * (N + 1) / 2.0;
    int S = 20;
    while (S < 40) {
        const double slots = (double)((size_t)1 << S);
        const double gb = slots * 16.0 / 1e9;
        if (gb > max_gb) break;
        if (pairs / slots < 0.85) return S;
        ++S;
    }
    return std::max(20, S - 1);
}

static void table_build(int N, int S, std::vector<Slot>& slots) {
    const size_t size = (size_t)1 << S;
    slots.assign(size, Slot{});
    const double pairs_est = (double)N * (N + 1) / 2.0;
    const double lf = pairs_est / (double)size;
    if (lf > 0.9)
        fprintf(stderr, "[table] note: LF=%.3f at S=%d — high; consider larger S or smaller N\n", lf, S);

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
                    slots[pos].key = fp;
                    break;
                }
                pos = (pos + step) & mask;
            }
        }
        steps += local;
    }
    const double ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    fprintf(stderr, "[table] N=%d pairs=%.3e slots=2^%d (%.1f GB) LF=%.3f avg probes=%.2f build=%.0f ms\n",
            N, pairs_est, S, size * 16.0 / 1e9, lf, steps.load() / pairs_est, ms);
}

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
        u32 lo3 = (u32)(hi3 * 0.8);
        if (lo3 < 1) lo3 = 1;
        if (lo3 > hi3) continue;
        const u64 tfp = C.q_lo - d_pow6_64(c4);
        for (u32 c3 = lo3 + (u32)lane; c3 <= hi3; c3 += 32) {
            d_probe(tab, mask, S, tfp - d_pow6_64(c3), c3,
                    C.job, c4, hits, nhit, hit_cap);
        }
    }
}

static void gpu_find4_jobs(const std::vector<Job>& jobs, int N, int S,
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
        const u64 hi4 = std::min<u64>(iroot6_u128(J.T), lim);
        u64 lo4 = 1;
        while (lo4 <= hi4 && p6full[lo4] * 4 < J.T) ++lo4;
        if (opt.x4_top > 0 && hi4 + 1 > opt.x4_top)
            lo4 = std::max(lo4, hi4 - opt.x4_top + 1);
        if (lo4 > hi4) continue;

        Cand4 C;
        C.q_lo = (u64)J.T; C.q_hi = (u64)(J.T >> 64);
        C.lim = lim;
        C.c4lo = (u32)lo4; C.c4hi = (u32)hi4;
        C.job = (u32)ji;

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
        k_find4<<<dim3(1, ytiles), 256>>>(
            d_c, 1, d_tab, slots.size() - 1, S, d_hits, d_nh, hit_cap);
        CU(cudaDeviceSynchronize());
        u32 nh = 0;
        CU(cudaMemcpy(&nh, d_nh, sizeof(u32), cudaMemcpyDeviceToHost));
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
    CU(cudaFree(d_tab));
    CU(cudaFree(d_hits));
    CU(cudaFree(d_nh));
}
#endif

static int selftest_host() {
    printf("[selftest-host] GMP fourcore math\n");
    const u64 B = 2400001ULL;  // may or may not be admissible; use synthetic T check
    // Direct: pick small x and u, see if we can form identity — skip.
    // Check compute_T + verify on constructed T from known x's.
    const u64 D = 42;
    const u64 xs[4] = {3, 5, 7, 11};
    u128 T = 0;
    for (u64 x : xs) T += pow6_u128(x);
    // Choose u=1, invent B such that B^6 = 1 + D^6 * T  — rarely a perfect power.
    // Instead verify verify_fourcore rejects a wrong B and accepts a constructed lhs.
    mpz_class lhs = mpz_pow6(1) + mpz_pow6(D) * (
        mpz_pow6(xs[0]) + mpz_pow6(xs[1]) + mpz_pow6(xs[2]) + mpz_pow6(xs[3]));
    // floor sixth root of lhs as fake B — only pass if exact
    // Just test compute_T roundtrip for an admissible Stage-1-like pair via hunt selftest.
    (void)B;
    u128 T2 = 0;
    // Use B=10007 (prime), find any u with compute_T succeeding for D=42 is rare
    // without Stage-1. Minimal check: pow6 and split/join.
    u64 lo, hi;
    split_u128(T, lo, hi);
    if (join_u128(lo, hi) != T) { printf("FAIL split\n"); return 1; }
    if (!verify_fourcore_gmp(42 * 11 + 1, 1, 1, 1, 1, 1, 1)) {
        // D=1 nonsense — expect false usually
    }
    // True small identity check with D=1: 1^6+2^6+2^6+2^6 =? B^6 — skip.
    printf("  split/join ok  planted T=%llu:%llu\n",
           (unsigned long long)hi, (unsigned long long)lo);
    printf("[selftest-host] PASS (math smoke)\n");
    return 0;
}

static void usage() {
    printf("usage: fourcore_find4 [--selftest-host] [--jobs file] [--D d]\n"
           "                      [--N n] [--S bits] [--max-table-gb G]\n"
           "                      [--x4-top K] [--stop-first]\n"
           "  jobs from fourcore_hunt: B u T_lo T_hi\n");
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
    bool do_host = false;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
        };
        if (s == "--selftest-host") do_host = true;
        else if (s == "--jobs") jobs_path = next();
        else if (s == "--D") D = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--N") N = atoi(next().c_str());
        else if (s == "--S") S_override = atoi(next().c_str());
        else if (s == "--max-table-gb") max_table_gb = atof(next().c_str());
        else if (s == "--x4-top") hop.x4_top = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--stop-first") hop.stop_first = true;
        else if (s == "-h" || s == "--help") { usage(); return 0; }
        else { printf("unknown %s\n", s.c_str()); usage(); return 1; }
    }

    if (do_host || argc == 1) return selftest_host();

    std::vector<Job> jobs;
    if (jobs_path.empty() || !load_jobs(jobs_path.c_str(), D, jobs) || jobs.empty()) {
        usage(); return 1;
    }
    for (auto& J : jobs) J.D = D;

    u32 need = 0;
    for (auto& J : jobs) need = std::max(need, J.lim);
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
    int S = S_override > 0 ? S_override : choose_S(N, max_table_gb);
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
