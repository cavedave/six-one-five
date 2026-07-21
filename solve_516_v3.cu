// =============================================================================
// solve_516_v3.cu — GPU-accelerated five-class (6,1,5) solver
//     a1^6 + a2^6 + a3^6 + a4^6 + a5^6 = B^6,  1<=a1<...<a5<B, gcd=1
// =============================================================================
//
// GPU successor to solve_516_v2.cpp. THE MATHEMATICS IS IDENTICAL TO v2: same
// five Meyrignac classes, same master congruences, same seed/unit machinery,
// same completeness guarantees. The GPU only accelerates the decomposition
// stage (find4/find3/find2 over the meet-in-the-middle pair table).
//
// DESIGN (see 615-gpu-runbook.md):
//   * Pair table = cuckoo-style open-addressing hash in VRAM. Slot = 16 bytes:
//       key     = (c_i^6 + c_j^6) mod 2^64   ("fingerprint", fp64)
//       payload = (i<<16) | j                (i <= j; requires N <= 65535,
//                                             i.e. B <= 2.75M — covers the
//                                             whole i128 regime, B <= 2.2M)
//     Probing: double hashing, pos = h1(fp), step = mix64(fp)|1, scan until an
//     empty slot (payload==0). Duplicate fingerprints occupy multiple slots and
//     are ALL reported (scan continues past matches until the first empty).
//   * WHY NO 128-BIT MATH ON THE GPU: a query target is only ever needed
//     mod 2^64. For class 1 the CPU passes Q=(B^6-u^6)/42^6 precomputed
//     (one i128 division per candidate). For classes 2-5 the target is
//     T = R/42^6 with R = B^6-u^6-(free terms)^6; since R is always divisible
//     by 64 (valuation laws) and 42^6 = 64 * 21^6 with 21^6 odd,
//         T mod 2^64 = ((R>>6) mod 2^64) * (21^6)^{-1}  (mod 2^64).
//     So the kernel does: 128-bit subtract (2 instr), funnel shift, one 64-bit
//     multiply. No 128-bit division anywhere. (Identity verified in selftest.)
//   * CORRECTNESS: completeness — every pair (i,j), i<=j<=N, is in the table,
//     and any true pair has matching fp64, so every decomposition v2 would
//     find is reported. Soundness — every fingerprint hit is re-verified on
//     the CPU with exact i128 arithmetic (plus the mod-300 filter); false
//     positives (fp64 collisions) are expected ~1e-5 per campaign.
//   * WINDOW MATH ON GPU: sixth-root bounds are computed with a double-power
//     approximation then EXACT fix-up loops using 128-bit integer compares
//     (saturating multiplies), so no floating-point error can affect results.
//
// WORK DISTRIBUTION (per eligible B at B=2.2M; see 615-search-code-reading-notes):
//   cls5 ~80% of probes (2-D (e,d) grid — the best GPU kernel), cls4 ~7%,
//   cls2 ~4%, cls3 ~2%, cls1 ~0.3%. Campaign 730k..2.2M ~ 1.6e14 probes
//   ~ a day at 2-4e9 probes/s (vs ~1.5-3 months on 4 CPU cores).
//
// Build (on the server, CUDA >= 12.8 for Blackwell sm_120):
//   nvcc -O3 -std=c++20 -arch=native -Xcompiler -fopenmp \
//        -o solve_516_v3 solve_516_v3.cu
// Run:
//   ./solve_516_v3 --selftest                 # host math + GPU plant tests
//   ./solve_516_v3 730000 2200000 all --save-table table_2p2M.bin
//   ./solve_516_v3 730001 750000 all --load-table table_2p2M.bin
//   ./solve_516_v3 300000 400000 all --xcheck # CPU-vs-GPU cross-validation
// Options: --chunk K (B per batch, default 8192), --device K, --hit-cap N,
//          --slots-log2 S, --bench K, --quiet. u band: [u_lo] [u_hi] as in v2.
//
// References: LPS 1967; Resta-Meyrignac 2003; Gerbicz-Meyrignac-Beckert
// arXiv:1108.0462; Bernstein Math. Comp. 70 (2001); euler.free.fr database.
// =============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if __has_include("mod60.hpp")
#include "mod60.hpp"
#define HAVE_MOD60 1
#endif
#if __has_include("quad_sum.hpp")
#include "quad_sum.hpp"
#define HAVE_XCHECK 1
#endif

using u64 = unsigned long long;
using u32 = unsigned int;
using u16 = unsigned short;
using i128 = __int128_t;
using u128 = __uint128_t;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------- constants --
static constexpr long long M2 = 64LL;         // 2^6
static constexpr long long M3 = 729LL;        // 3^6
static constexpr long long M7 = 117649LL;     // 7^6
static constexpr long long M14 = 7529536LL;   // 14^6
static constexpr long long M21 = 85766121LL;  // 21^6  (odd part of 42^6)
static constexpr long long M42 = 5489031744LL;// 42^6 = 64 * 21^6
static constexpr int K = 6;
static constexpr long long B_HARD_MAX = 2200000;   // B^6 < 2^127 (i128 guard)
static constexpr u64 PHI64 = 0x9E3779B97F4A7C15ULL; // fibonacci hash constant

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

// ------------------------------------------------- host small modular tools --
static long long mod_pow6(long long base, long long mod) {
    long long r = 1;
    base %= mod;
    if (base < 0) base += mod;
    for (int e = 0; e < 6; ++e) r = (r * base) % mod;
    return r;
}
static long long egcd(long long a, long long b, long long& x, long long& y) {
    if (b == 0) { x = 1; y = 0; return a; }
    long long x1, y1;
    const long long g = egcd(b, a % b, x1, y1);
    x = y1; y = x1 - y1 * (a / b);
    return g;
}
static long long mod_inv(long long a, long long m) {
    long long x, y;
    egcd(a, m, x, y);
    x %= m; if (x < 0) x += m;
    return x;
}
static long long crt2(long long r1, long long m1, long long r2, long long m2) {
    long long x, y;
    const long long g = egcd(m1, m2, x, y);
    const long long t = ((r2 - r1) / g) * x;
    const long long mod = m1 / g * m2;
    long long ans = r1 + m1 * (t % (m2 / g));
    ans %= mod;
    if (ans < 0) ans += mod;
    return ans;
}

// i128 sixth power (host; B <= 2.2M so x^6 < 2^127 always).
static i128 ipow6(long long x) {
    i128 r = 1, b = x;
    for (int e = 0; e < 6; ++e) r *= b;
    return r;
}
// exact floor sixth root of nonnegative i128.
static long long iroot6_i128(i128 n) {
    if (n <= 0) return 0;
    long long lo = 0, hi = 1;
    while (ipow6(hi) <= n) hi <<= 1;
    while (hi - lo > 1) {
        const long long mid = (lo + hi) >> 1;
        if (ipow6(mid) <= n) lo = mid; else hi = mid;
    }
    return lo;
}

// ------------------------------------------------------------- root tables --
// Identical to v2: for every sixth-power residue r among UNITS mod m, the
// units a with a^6 ≡ r. mod 2^6: 8 residues x 4 roots; mod 3^6: 81 x 6;
// mod 7^6: 16807 x 6 (Teichmuller).
struct RootTables {
    std::vector<std::vector<long long>> r2, r3, r7;
    RootTables() : r2(M2), r3(M3), r7(M7) {
        for (long long a = 1; a < M2; ++a) if (std::gcd(a, M2) == 1) r2[mod_pow6(a, M2)].push_back(a);
        for (long long a = 1; a < M3; ++a) if (std::gcd(a, M3) == 1) r3[mod_pow6(a, M3)].push_back(a);
        for (long long a = 1; a < M7; ++a) if (std::gcd(a, M7) == 1) r7[mod_pow6(a, M7)].push_back(a);
    }
};

static std::vector<long long> seeds_for_B(const RootTables& rt, long long B, int cls) {
    const long long b2 = mod_pow6(B, M2), b3 = mod_pow6(B, M3), b7 = mod_pow6(B, M7);
    std::vector<long long> out;
    if (cls == 1) {          // 4 x 6 x 6 = 144 classes mod 42^6
        for (long long s2 : rt.r2[b2])
            for (long long s3 : rt.r3[b3]) {
                const long long s23 = crt2(s2, M2, s3, M3);
                for (long long s7 : rt.r7[b7]) out.push_back(crt2(s23, M2 * M3, s7, M7));
            }
    } else if (cls == 2) {   // 24 classes mod 14^6
        for (long long s2 : rt.r2[b2])
            for (long long s7 : rt.r7[b7]) out.push_back(crt2(s2, M2, s7, M7));
    } else if (cls == 3) {   // 36 classes mod 21^6
        for (long long s3 : rt.r3[b3])
            for (long long s7 : rt.r7[b7]) out.push_back(crt2(s3, M3, s7, M7));
    } else {                 // classes 4,5: 6 classes mod 7^6
        out = rt.r7[b7];
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}
static long long class_modulus(int cls) {
    switch (cls) {
        case 1: return M42; case 2: return M14; case 3: return M21; default: return M7;
    }
}
static bool unit_ok(long long u, int cls) {
    switch (cls) {
        case 1: return (u & 1) && std::gcd(u, 42LL) == 1;
        case 2: return (u & 1) && (u % 3 == 0) && (u % 9 != 0);
        case 3: return !(u & 1) && std::gcd(u, 21LL) == 1;
        case 4:
        case 5: return (u % 6 == 0) && (u % 7 != 0);
    }
    return false;
}
static std::vector<long long> unit_candidates(const RootTables& rt, long long B, int cls,
                                              double lo_frac, double hi_frac) {
    const long long M = class_modulus(cls);
    const long long lo = std::max(1LL, (long long)(lo_frac * B));
    const long long hi = std::min(B - 1, (long long)(hi_frac * B));
    std::vector<long long> out;
    for (long long s : seeds_for_B(rt, B, cls)) {
        long long u = s % M;
        if (u < lo) u += (lo - u + M - 1) / M * M;
        for (; u <= hi; u += M)
            if (unit_ok(u, cls)) out.push_back(u);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// ------------------------------------------- free-term residue class sets --
// Same valuation filters as v2.
static std::vector<long long> classes_d_cls2(const RootTables& rt, long long B) {
    const long long t = mod_pow6(B, M3) * mod_inv(mod_pow6(14, M3), M3) % M3;
    return rt.r3[t];                                    // 6 classes mod 729
}
static std::vector<long long> classes_d_cls3(const RootTables& rt, long long B, long long u) {
    long long rhs = (mod_pow6(B, M2) - mod_pow6(u, M2)) % M2; if (rhs < 0) rhs += M2;
    const long long t = rhs * mod_inv(mod_pow6(21, M2), M2) % M2;
    return rt.r2[t];                                    // 4 classes mod 64
}
static std::vector<long long> classes_d_cls4(const RootTables& rt, long long B) {
    const long long t2 = mod_pow6(B, M2) * mod_inv(mod_pow6(7, M2), M2) % M2;
    const long long t3 = mod_pow6(B, M3) * mod_inv(mod_pow6(7, M3), M3) % M3;
    std::vector<long long> out;
    for (long long a : rt.r2[t2]) for (long long b : rt.r3[t3]) out.push_back(crt2(a, M2, b, M3));
    std::sort(out.begin(), out.end());
    return out;                                         // 24 classes mod 46656
}
static std::vector<long long> classes_e_cls5(const RootTables& rt, long long B) {
    return classes_d_cls2(rt, B);                       // (14e)^6 ≡ B^6 (3^6)
}
static std::vector<long long> classes_d_cls5(const RootTables& rt, long long B, long long u) {
    return classes_d_cls3(rt, B, u);                    // (21d)^6 ≡ B^6-u^6 (2^6)
}

// -------------------------------------------------- 21^6 inverse mod 2^64 --
// Newton iteration: x <- x(2 - a x) doubles correct bits; start x=a (mod 8).
static u64 inv216_mod_2_64() {
    u64 a = (u64)M21, x = a;
    for (int i = 0; i < 6; ++i) x = x * (2 - a * x);   // wrapping u64 arithmetic
    return x;
}

// =============================================================================
// PAIR TABLE (host build, device query)
// Slot: 16 bytes. payload==0 marks empty. payload=(i<<16)|j, i<=j, N<=65535.
// =============================================================================
struct Slot { u64 key; u32 payload; u32 pad; };

__host__ __device__ __forceinline__ u64 mix64(u64 x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}
__host__ __device__ __forceinline__ u64 hash_pos(u64 fp, int S) {
    return (fp * PHI64) >> (64 - S);
}

// Build slots for all pairs 1<=i<=j<=N, fp = (i^6+j^6) mod 2^64.
// Duplicated fingerprints are stored in multiple slots (query scans to empty).
static void table_build(int N, int S, std::vector<Slot>& slots) {
    const size_t size = (size_t)1 << S;
    const double pairs_est = (double)N * (N + 1) / 2;
    if ((double)size * 0.95 < pairs_est) {   // insert probes need empty slots to terminate
        fprintf(stderr, "[table] FATAL: load factor %.2f (pairs=%.3e slots=2^%d); increase S\n",
                pairs_est / (double)size, pairs_est, S);
        exit(1);
    }
    slots.assign(size, Slot{0, 0, 0});
    std::vector<u64> pw6(N + 1);
#pragma omp parallel for schedule(static)
    for (int x = 1; x <= N; ++x) {
        u64 x2 = (u64)x * x;
        pw6[x] = x2 * x2 * x2;   // x^6 mod 2^64 (wrapping)
    }
    const u64 mask = size - 1;
    std::atomic<u64> steps{0};
    const auto t0 = Clock::now();
#pragma omp parallel for schedule(static)
    for (int i = 1; i <= N; ++i) {
        u64 local = 0;
        for (int j = i; j <= N; ++j) {
            const u64 fp = pw6[i] + pw6[j];
            const u32 pld = ((u32)i << 16) | (u32)j;
            u64 pos = hash_pos(fp, S);
            const u64 step = mix64(fp) | 1ULL;
            for (;;) {
                ++local;
                if (__sync_bool_compare_and_swap(&slots[pos].payload, 0u, pld)) {
                    slots[pos].key = fp;   // after claim; only concurrent inserters
                    break;                 // can mis-read, causing a benign extra dup
                }
                pos = (pos + step) & mask;
            }
        }
        steps += local;
    }
    const double ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    const double P = (double)N * (N + 1) / 2;
    fprintf(stderr, "[table] N=%d pairs=%.3e slots=2^%d (%.1f GB) LF=%.3f avg insert probes=%.2f build=%.0f ms\n",
            N, P, S, size * 16.0 / 1e9, P / size, steps.load() / P, ms);
}

struct TableHeader { char magic[8]; u64 N, S; };
static void table_save(const char* path, const std::vector<Slot>& slots, int N, int S) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[table] cannot open %s for write\n", path); return; }
    TableHeader h; memcpy(h.magic, "516TBL01", 8); h.N = (u64)N; h.S = (u64)S;
    fwrite(&h, sizeof(h), 1, f);
    fwrite(slots.data(), sizeof(Slot), slots.size(), f);
    fclose(f);
    fprintf(stderr, "[table] saved to %s (%.1f GB)\n", path, slots.size() * 16.0 / 1e9);
}
static bool table_load(const char* path, std::vector<Slot>& slots, int N, int& S) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    TableHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1 || memcmp(h.magic, "516TBL01", 8) != 0) {
        fclose(f); fprintf(stderr, "[table] bad file %s\n", path); return false;
    }
    if ((int)h.N != N) {
        fclose(f);
        fprintf(stderr, "[table] N mismatch: file N=%llu, need %d — rebuilding\n", (unsigned long long)h.N, N);
        return false;
    }
    S = (int)h.S;
    slots.assign((size_t)1 << S, Slot{0, 0, 0});
    const size_t want = slots.size() * sizeof(Slot);
    const size_t got = fread(slots.data(), 1, want, f);
    fclose(f);
    if (got != want) { fprintf(stderr, "[table] short read %s\n", path); return false; }
    fprintf(stderr, "[table] loaded %s (N=%d, 2^%d slots)\n", path, N, S);
    return true;
}

// =============================================================================
// DEVICE SIDE
// =============================================================================
// Candidate batch record. q_lo/q_hi carry:
//   cls1: Q = (B^6-u^6)/42^6 (exact, CPU-divided)         — c4lo/c4hi set
//   cls2-5: base = B^6-u^6 (128-bit)                      — free-term fields set
// res[]: free-term residue classes; cls5: e-classes at [0..nres1), d-classes at
// [nres1..nres1+nres2). cls2/3/4: d-classes at [0..nres1).
struct GpuCand {
    u64 q_lo, q_hi;        // 16
    u32 B, u, lim;         // 12
    u32 c4lo, c4hi;        // 8   (cls1 only)
    u32 fmax1, fmax2;      // 8
    u32 mod1, mod2;        // 8
    u32 cls, nres1, nres2, pad; // 16
    u32 res[24];           // 96
};                         // ~164 bytes

struct Hit { u32 cand, a, b, c, d; };  // a,b = pair indices; c,d = context (c3,c4 | c3,d | d,e)

struct Params {
    const Slot* __restrict__ tab;
    u64 mask;
    int S;
    Hit* __restrict__ hits;
    u32* __restrict__ hitc;
    u32 cap;
    u32* __restrict__ overflow;
    u64* __restrict__ probes;
    const GpuCand* __restrict__ cands;
    u32 n_cand;
    u64 inv216;              // (21^6)^{-1} mod 2^64
    u32 factor;              // cls234 free-term factor: 14 / 21 / 7
};

// ------------------------------------------- device 64/128-bit primitives --
// x^6 mod 2^64 (x <= 65535 in all uses).
__device__ __forceinline__ u64 pow6_64(u32 x) {
    const u64 x2 = (u64)x * x;
    return x2 * x2 * x2;
}
// x^6 as exact 128-bit (hi,lo). Valid for x < 2^21.33 (all our inputs < 2.3e6).
__device__ __forceinline__ void pow6_128(u32 x, u64& hi, u64& lo) {
    const u64 x2 = (u64)x * x;                 // < 2^43
    const u64 h4 = __umul64hi(x2, x2), l4 = x2 * x2;   // x^4 < 2^86
    lo = l4 * x2;
    hi = __umul64hi(l4, x2) + h4 * x2;         // x^6 < 2^127
}
// (hi:lo) * m with saturation flag (m small). Returns true if product >= 2^128.
__device__ __forceinline__ bool mul128_small(u64& hi, u64& lo, u64 m) {
    const u64 h1 = __umul64hi(lo, m);
    const u64 h2 = hi * m;                     // wraps if overflowing
    bool of = (hi != 0) && (__umul64hi(hi, m) != 0);
    hi = h1 + h2;
    of |= (hi < h1);
    lo = lo * m;
    return of;
}
// greater-than compare of 128-bit values.
__device__ __forceinline__ bool gt128(u64 ah, u64 al, u64 bh, u64 bl) {
    return (ah > bh) || (ah == bh && al > bl);
}
// Is SCALE * c^6 > R ? (saturating: overflow counts as "greater").
__device__ __forceinline__ bool scaled_gt(u64 rh, u64 rl, u32 c, u64 scale) {
    u64 ch, cl; pow6_128(c, ch, cl);
    if (mul128_small(ch, cl, scale)) return true;
    return gt128(ch, cl, rh, rl);
}
// Largest r with scale*r^6 <= R. Double approx + exact fix-up.
__device__ u32 iroot6_fix(u64 rh, u64 rl, double inv_scale, u64 scale) {
    const double d = (double)rh * 18446744073709551616.0 + (double)rl;
    u32 r = (u32)(pow(d * inv_scale, 1.0 / 6.0)) + 2;
    while (r > 0 && scaled_gt(rh, rl, r, scale)) --r;
    while (!scaled_gt(rh, rl, r + 1, scale)) ++r;
    return r;
}
// Is scale * c^6 < R ? (saturating: overflow counts as "not less").
__device__ __forceinline__ bool scaled_lt(u64 rh, u64 rl, u32 c, u64 scale) {
    u64 ch, cl; pow6_128(c, ch, cl);
    if (mul128_small(ch, cl, scale)) return false;
    return gt128(rh, rl, ch, cl);
}
// Smallest c >= 1 with 3*scale*c^6 >= R, seeded near r_hi * 3^(-1/6).
__device__ u32 min_c_fix(u64 rh, u64 rl, u32 r_hi, u64 scale) {
    u32 c = (u32)(r_hi * 0.8326831149);
    if (c < 1) c = 1;
    while (scaled_lt(rh, rl, c, 3 * scale)) ++c;                 // first c with 3*scale*c^6 >= R
    while (c > 1 && !scaled_lt(rh, rl, c - 1, 3 * scale)) --c;   // trim overshoot
    return c;
}
// fp64 of target R/42^6 given R (128-bit, divisible by 64). = ((R>>6) mod 2^64) * 21^-6.
__device__ __forceinline__ u64 funnel_fp(u64 rh, u64 rl, u64 inv216) {
    return ((rl >> 6) | (rh << 58)) * inv216;
}

// Probe the table for fp; on every key match push a hit (scan continues to the
// first empty slot so duplicate fingerprints are all collected).
// Returns number of slot reads (for the probe-rate counter).
__device__ u64 probe(const Params& P, u64 fp, u32 ci, u32 v3, u32 v4) {
    u64 pos = hash_pos(fp, P.S);
    const u64 step = mix64(fp) | 1ULL;
    u64 reads = 0;
    for (;;) {
        const Slot s = P.tab[pos];
        ++reads;
        if (s.payload == 0) break;
        if (s.key == fp) {
            const u32 idx = atomicAdd(P.hitc, 1u);
            if (idx < P.cap) P.hits[idx] = Hit{ci, s.payload >> 16, s.payload & 0xffffu, v3, v4};
            else atomicExch(P.overflow, 1u);
        }
        pos = (pos + step) & P.mask;
    }
    return reads;
}

// ------------------------------------------------------------- kernels --
// Class 1: Q = c1^6+c2^6+c3^6+c4^6. block per (cand, c4-chunk of 8*256 c4),
// warp per c4, lanes over c3 window.
__global__ void k_cls1(Params P) {
    const u32 ci = blockIdx.x;
    if (ci >= P.n_cand) return;
    const GpuCand C = P.cands[ci];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const u32 base = C.c4lo + blockIdx.y * 2048 + warp;
    u64 cnt = 0;
    for (u32 c4 = base; c4 <= C.c4hi; c4 += 8) {
        u64 c4h, c4l; pow6_128(c4, c4h, c4l);
        const u64 rlo = C.q_lo - c4l;
        const u64 rhi = C.q_hi - c4h - (C.q_lo < c4l ? 1 : 0);   // Q' = Q - c4^6
        if ((rhi >> 63) || ((rhi | rlo) == 0)) continue;
        u32 hi3 = iroot6_fix(rhi, rlo, 1.0, 1);
        const u32 capm = C.lim < c4 ? C.lim : c4;
        if (hi3 > capm) hi3 = capm;
        if (hi3 < 1) continue;
        const u32 lo3 = min_c_fix(rhi, rlo, hi3, 1);
        const u64 tfp = C.q_lo - pow6_64(c4);
        for (u32 c3 = lo3 + lane; c3 <= hi3; c3 += 32)
            cnt += probe(P, tfp - pow6_64(c3), ci, c3, c4);
    }
    if (cnt) atomicAdd(P.probes, cnt);
}

// Classes 2/3/4: find3 target T = (base - (f*d)^6)/42^6, window over c3.
// block per (cand, d-flat-index); thread 0 computes R and the c3 window.
__global__ void k_cls234(Params P) {
    const u32 ci = blockIdx.x;
    if (ci >= P.n_cand) return;
    const GpuCand C = P.cands[ci];
    const u32 kmax = C.fmax1 / C.mod1 + 1;
    const u32 t = blockIdx.y;
    if (t >= C.nres1 * kmax) return;
    const u32 d = C.res[t % C.nres1] + (t / C.nres1) * C.mod1;
    if (d == 0 || d > C.fmax1) return;

    __shared__ u64 s_tfp;
    __shared__ u32 s_lo, s_hi;
    __shared__ int s_active;
    if (threadIdx.x == 0) {
        s_active = 0;
        u64 fh, fl; pow6_128(P.factor * d, fh, fl);              // (f*d)^6
        const u64 rlo = C.q_lo - fl;
        const u64 rhi = C.q_hi - fh - (C.q_lo < fl ? 1 : 0);     // R = base - (fd)^6
        if ((rhi >> 63) == 0 && ((rhi | rlo) != 0)) {
            u32 hi3 = iroot6_fix(rhi, rlo, 1.0 / (double)M42, (u64)M42);
            if (hi3 > C.lim) hi3 = C.lim;
            if (hi3 >= 1) {
                const u32 lo3 = min_c_fix(rhi, rlo, hi3, (u64)M42);
                if (lo3 <= hi3) {
                    s_lo = lo3; s_hi = hi3;
                    s_tfp = funnel_fp(rhi, rlo, P.inv216);
                    s_active = 1;
                }
            }
        }
    }
    __syncthreads();
    if (!s_active) return;
    const u64 tfp = s_tfp;
    const u32 lo = s_lo, hi = s_hi;
    u64 cnt = 0;
    for (u32 c3 = lo + threadIdx.x; c3 <= hi; c3 += blockDim.x)
        cnt += probe(P, tfp - pow6_64(c3), ci, c3, d);
    if (cnt) atomicAdd(P.probes, cnt);
}

// Class 5: find2 target T = (base - (14e)^6 - (21d)^6)/42^6.
// block per (cand, e-flat-index); threads stride the d grid.
__global__ void k_cls5(Params P) {
    const u32 ci = blockIdx.x;
    if (ci >= P.n_cand) return;
    const GpuCand C = P.cands[ci];
    const u32 kmax1 = C.fmax1 / C.mod1 + 1;
    const u32 t1 = blockIdx.y;
    if (t1 >= C.nres1 * kmax1) return;
    const u32 e = C.res[t1 % C.nres1] + (t1 / C.nres1) * C.mod1;
    if (e == 0 || e > C.fmax1) return;

    __shared__ u64 s_Eh, s_El, s_bh, s_bl;
    __shared__ u32 s_res2[24], s_n2, s_mod2, s_fmax2, s_total;
    if (threadIdx.x == 0) {
        pow6_128(14 * e, s_Eh, s_El);
        s_bh = C.q_hi; s_bl = C.q_lo;
        s_n2 = C.nres2; s_mod2 = C.mod2; s_fmax2 = C.fmax2;
        const u32 k2 = C.fmax2 / C.mod2 + 1;
        s_total = C.nres2 * k2;
        for (u32 i = 0; i < C.nres2 && i < 24; ++i) s_res2[i] = C.res[C.nres1 + i];
    }
    __syncthreads();
    u64 cnt = 0;
    for (u32 t2 = threadIdx.x; t2 < s_total; t2 += blockDim.x) {
        const u32 d = s_res2[t2 % s_n2] + (t2 / s_n2) * s_mod2;
        if (d == 0 || d > s_fmax2) continue;
        u64 dh, dl; pow6_128(21 * d, dh, dl);
        u64 rlo = s_bl - s_El;
        u64 rhi = s_bh - s_Eh - (s_bl < s_El ? 1 : 0);            // base - (14e)^6
        const u64 rlo2 = rlo - dl;
        const u64 rhi2 = rhi - dh - (rlo < dl ? 1 : 0);           // R
        if ((rhi2 >> 63) || ((rhi2 | rlo2) == 0)) continue;
        cnt += probe(P, funnel_fp(rhi2, rlo2, P.inv216), ci, d, e);
    }
    if (cnt) atomicAdd(P.probes, cnt);
}

// Plant-test kernel: each thread probes fps[t] and verifies that the expected
// packed payload appears among the matches.
__global__ void k_probe_test(const u64* fps, const u32* expect, int n,
                             const Slot* tab, u64 mask, int S, u32* ok, u32* bad) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const u64 fp = fps[t];
    u64 pos = hash_pos(fp, S);
    const u64 step = mix64(fp) | 1ULL;
    bool found = false;
    for (;;) {
        const Slot s = tab[pos];
        if (s.payload == 0) break;
        if (s.key == fp && s.payload == expect[t]) found = true;
        pos = (pos + step) & mask;
    }
    if (found) atomicAdd(ok, 1u); else atomicAdd(bad, 1u);
}

// =============================================================================
// HOST ORCHESTRATION
// =============================================================================
struct GpuCtx {
    Slot* d_tab = nullptr;
    size_t tab_slots = 0;
    int S = 0;
    Hit* d_hits = nullptr;
    u32 hit_cap = 0;
    u32* d_hitc = nullptr;
    u32* d_overflow = nullptr;
    u64* d_probes = nullptr;
    GpuCand* d_cands = nullptr;
    size_t cand_cap = 0;
};

static void gpu_init(GpuCtx& g, int device, u32 hit_cap) {
    int ndev = 0;
    CU(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { fprintf(stderr, "no CUDA device found\n"); exit(1); }
    if (device >= ndev) { fprintf(stderr, "device %d out of range (%d present)\n", device, ndev); exit(1); }
    CU(cudaSetDevice(device));
    cudaDeviceProp prop;
    CU(cudaGetDeviceProperties(&prop, device));
    fprintf(stderr, "[gpu] %s, %.1f GB, %d SMs, compute %d.%d\n",
            prop.name, prop.totalGlobalMem / 1e9, prop.multiProcessorCount, prop.major, prop.minor);
    size_t free_b = 0, tot_b = 0;
    CU(cudaMemGetInfo(&free_b, &tot_b));
    fprintf(stderr, "[gpu] memory free %.1f GB / total %.1f GB\n", free_b / 1e9, tot_b / 1e9);
    g.hit_cap = hit_cap;
    CU(cudaMalloc(&g.d_hits, (size_t)hit_cap * sizeof(Hit)));
    CU(cudaMalloc(&g.d_hitc, sizeof(u32)));
    CU(cudaMalloc(&g.d_overflow, sizeof(u32)));
    CU(cudaMalloc(&g.d_probes, sizeof(u64)));
}

static void gpu_upload_table(GpuCtx& g, const std::vector<Slot>& slots, int S) {
    g.S = S;
    g.tab_slots = slots.size();
    CU(cudaMalloc(&g.d_tab, slots.size() * sizeof(Slot)));
    const auto t0 = Clock::now();
    CU(cudaMemcpy(g.d_tab, slots.data(), slots.size() * sizeof(Slot), cudaMemcpyHostToDevice));
    const double ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    fprintf(stderr, "[gpu] table uploaded (%.1f GB) in %.0f ms\n", slots.size() * 16.0 / 1e9, ms);
}

// ----------------------------------------------------- candidate generation --
static void gen_class_cands(const RootTables& rt, long long B, int cls, double u_lo, double u_hi,
                            std::vector<GpuCand>& out) {
    const auto units = unit_candidates(rt, B, cls, u_lo, u_hi);
    const i128 B6 = ipow6(B);
    for (long long u : units) {
        GpuCand C{};
        C.B = (u32)B; C.u = (u32)u; C.cls = (u32)cls; C.lim = (u32)((B - 1) / 42);
        const i128 base = B6 - ipow6(u);
        if (base <= 0) continue;
        if (cls == 1) {
            if (base % M42 != 0) continue;                    // guaranteed by seeds
            const i128 Q = base / M42;
            C.q_lo = (u64)Q; C.q_hi = (u64)((u128)Q >> 64);
            const long long hi4 = std::min(iroot6_i128(Q), (long long)C.lim);
            long long lo4 = (long long)(hi4 * 0.7937005260);  // 4^(-1/6)
            if (lo4 < 1) lo4 = 1;
            while (4 * ipow6(lo4) < Q) ++lo4;
            while (lo4 > 1 && 4 * ipow6(lo4 - 1) >= Q) --lo4;
            if (lo4 > hi4) continue;
            C.c4lo = (u32)lo4; C.c4hi = (u32)hi4;
        } else {
            C.q_lo = (u64)base; C.q_hi = (u64)((u128)base >> 64);
            std::vector<long long> r1, r2;
            switch (cls) {
                case 2: r1 = classes_d_cls2(rt, B);    C.mod1 = M3; C.fmax1 = (u32)((B - 1) / 14); break;
                case 3: r1 = classes_d_cls3(rt, B, u); C.mod1 = M2; C.fmax1 = (u32)((B - 1) / 21); break;
                case 4: r1 = classes_d_cls4(rt, B);    C.mod1 = M2 * M3; C.fmax1 = (u32)((B - 1) / 7); break;
                case 5:
                    r1 = classes_e_cls5(rt, B);    C.mod1 = M3; C.fmax1 = (u32)((B - 1) / 14);
                    r2 = classes_d_cls5(rt, B, u);   C.mod2 = M2; C.fmax2 = (u32)((B - 1) / 21);
                    break;
            }
            C.nres1 = (u32)r1.size();
            for (size_t i = 0; i < r1.size() && i < 24; ++i) C.res[i] = (u32)r1[i];
            C.nres2 = (u32)r2.size();
            for (size_t i = 0; i < r2.size() && C.nres1 + i < 24; ++i) C.res[C.nres1 + i] = (u32)r2[i];
        }
        out.push_back(C);
    }
}

// ------------------------------------------------------------ GPU run/drain --
struct RunResult {
    std::vector<Hit> hits;
    u32 raw_count = 0;
    u64 probes = 0;
    bool overflow = false;
    double kernel_ms = 0.0;
};

static RunResult gpu_run(GpuCtx& g, int cls, const std::vector<GpuCand>& v, u64 inv216) {
    RunResult R;
    if (v.empty()) return R;
    if (v.size() > g.cand_cap) {
        if (g.d_cands) CU(cudaFree(g.d_cands));
        g.cand_cap = v.size() + v.size() / 4 + 1024;
        CU(cudaMalloc(&g.d_cands, g.cand_cap * sizeof(GpuCand)));
    }
    CU(cudaMemcpy(g.d_cands, v.data(), v.size() * sizeof(GpuCand), cudaMemcpyHostToDevice));
    CU(cudaMemset(g.d_hitc, 0, sizeof(u32)));
    CU(cudaMemset(g.d_overflow, 0, sizeof(u32)));
    CU(cudaMemset(g.d_probes, 0, sizeof(u64)));

    Params P{};
    P.tab = g.d_tab; P.mask = g.tab_slots - 1; P.S = g.S;
    P.hits = g.d_hits; P.hitc = g.d_hitc; P.cap = g.hit_cap; P.overflow = g.d_overflow;
    P.probes = g.d_probes;
    P.cands = g.d_cands; P.n_cand = (u32)v.size();
    P.inv216 = inv216;
    P.factor = (cls == 2) ? 14 : (cls == 3) ? 21 : (cls == 4) ? 7 : 0;

    u32 ymax = 1;
    if (cls == 1) ymax = 32;                       // 2048-c4 chunks (window <= ~11k)
    else {
        for (const auto& C : v) {
            const u32 f = C.nres1 * (C.fmax1 / C.mod1 + 1);
            if (f > ymax) ymax = f;
        }
    }
    const dim3 grid((u32)v.size(), ymax), block(256);
    const auto t0 = Clock::now();
    if (cls == 1)      k_cls1<<<grid, block>>>(P);
    else if (cls == 5) k_cls5<<<grid, block>>>(P);
    else               k_cls234<<<grid, block>>>(P);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
    R.kernel_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();

    u32 h_ovf = 0;
    CU(cudaMemcpy(&R.raw_count, g.d_hitc, sizeof(u32), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&h_ovf, g.d_overflow, sizeof(u32), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&R.probes, g.d_probes, sizeof(u64), cudaMemcpyDeviceToHost));
    R.overflow = (h_ovf != 0);
    const u32 n = std::min(R.raw_count, g.hit_cap);
    R.hits.resize(n);
    if (n) CU(cudaMemcpy(R.hits.data(), g.d_hits, n * sizeof(Hit), cudaMemcpyDeviceToHost));
    return R;
}

// ------------------------------------------------------------- verification --
// Returns 0 = fingerprint false positive, 1 = exact decomposition but tuple
// rejected (non-distinct / >= B / mod-300), 2 = SOLUTION.
static int verify_hit(const GpuCand& C, const Hit& H, long& solutions,
                      std::set<std::array<long long, 5>>& reported) {
    const int cls = (int)C.cls;
    const u128 a6 = (u128)ipow6(H.a), b6 = (u128)ipow6(H.b);
    const u128 q = ((u128)C.q_hi << 64) | C.q_lo;
    std::array<long long, 5> t;
    bool exact = false;
    switch (cls) {
        case 1: {
            const u128 s = a6 + b6 + (u128)ipow6(H.c) + (u128)ipow6(H.d);   // < 2^98, safe
            exact = (s == q);
            t = {42LL * H.a, 42LL * H.b, 42LL * H.c, 42LL * H.d, (long long)C.u};
            break;
        }
        case 2: case 3: case 4: {
            // exact <=> a^6+b^6+c^6 == (q - (f d)^6)/42^6, checked without
            // ever forming the 42^6-multiple (would overflow u128).
            const long long f = (cls == 2) ? 14 : (cls == 3) ? 21 : 7;
            const u128 fd6 = (u128)ipow6(f * (long long)H.d);
            if (fd6 <= q) {
                const u128 rhs = q - fd6;
                exact = (rhs % (u128)M42 == 0) && (a6 + b6 + (u128)ipow6(H.c) == rhs / (u128)M42);
            }
            t = {42LL * H.a, 42LL * H.b, 42LL * H.c, f * (long long)H.d, (long long)C.u};
            break;
        }
        case 5: {
            const u128 d6 = (u128)ipow6(21LL * H.c), e6 = (u128)ipow6(14LL * H.d);
            if (d6 <= q && e6 <= q - d6) {
                const u128 rhs = q - d6 - e6;
                exact = (rhs % (u128)M42 == 0) && (a6 + b6 == rhs / (u128)M42);
            }
            t = {42LL * H.a, 42LL * H.b, 21LL * H.c, 14LL * H.d, (long long)C.u};
            break;
        }
    }
    if (!exact) return 0;
    std::sort(t.begin(), t.end());
    for (int i = 0; i < 4; ++i) if (t[i] == t[i + 1]) return 1;
    if (t.back() >= (long long)C.B || t.front() < 1) return 1;
#if HAVE_MOD60
    static mod60::SixthPowerFilter300 f300;
    if (!f300.passes((long long)C.B, {t[0], t[1], t[2], t[3], t[4]})) return 1;
#endif
    i128 lhs = ipow6((long long)C.B);       // subtractive: partial sums stay in
    for (long long x : t) {                 // (-B^6, B^6), no i128 overflow
        lhs -= ipow6(x);
        if (lhs < 0) return 1;
    }
    if (lhs != 0) return 1;
    if (reported.insert(t).second) {
        long long g = std::gcd(t[0], t[1]);
        for (int i = 2; i < 5; ++i) g = std::gcd(g, t[i]);
        g = std::gcd(g, (long long)C.B);
        printf("SOLUTION cls=%d B=%u a1=%lld a2=%lld a3=%lld a4=%lld a5=%lld (unit=%u) gcd=%lld %s\n",
               cls, C.B, t[0], t[1], t[2], t[3], t[4], C.u, g, g == 1 ? "primitive" : "NON-PRIMITIVE");
        fflush(stdout);
        fprintf(stderr, "%u\tSOLUTION\tcls=%d\n", C.B, cls);
    }
    ++solutions;
    return 2;
}

// =============================================================================
// SELFTEST — host mirrors of the device math, checked against i128 truth.
// =============================================================================
static u128 h_pow6_128(u32 x) { return (u128)ipow6(x); }   // host mirror reference

static bool selftest_host(const RootTables& rt, u64 inv216) {
    fprintf(stderr, "[selftest] host math\n");
    // (a) seeds satisfy the master congruences (as v2)
    srand(615);
    for (int trial = 0; trial < 200; ++trial) {
        long long B = 1000 + rand() % 2000000;
        while (B % 2 == 0 || B % 3 == 0 || B % 7 == 0) ++B;
        for (int cls = 1; cls <= 5; ++cls)
            for (long long s : seeds_for_B(rt, B, cls)) {
                bool ok = mod_pow6(s, M7) == mod_pow6(B, M7);
                if (cls == 1) ok = ok && mod_pow6(s, M2) == mod_pow6(B, M2) && mod_pow6(s, M3) == mod_pow6(B, M3);
                if (cls == 2) ok = ok && mod_pow6(s, M2) == mod_pow6(B, M2);
                if (cls == 3) ok = ok && mod_pow6(s, M3) == mod_pow6(B, M3);
                if (!ok) { fprintf(stderr, "seed fail cls=%d B=%lld\n", cls, B); return false; }
            }
    }
    // (b) 21^6 inverse
    if ((u64)((u128)M21 * inv216) != 1) { fprintf(stderr, "inv216 wrong\n"); return false; }
    // (c) fingerprint identity: ((R>>6) mod 2^64) * inv216 == (R/42^6) mod 2^64
    for (int t = 0; t < 200000; ++t) {
        const u128 T = ((u128)rand() << 60) ^ ((u128)rand() << 30) ^ (u128)rand();
        const u128 R = T * (u128)M42;
        const u64 rh = (u64)(R >> 64), rl = (u64)R;
        const u64 fp = ((rl >> 6) | (rh << 58)) * inv216;
        if (fp != (u64)T) { fprintf(stderr, "fp identity fail\n"); return false; }
    }
    // (d) pow6_64 wrapping matches i128 low half
    for (int t = 0; t < 100000; ++t) {
        const u32 x = 1 + rand() % 3000000;
        const u64 w = (u64)x * x * x * x * x * x;
        if (w != (u64)h_pow6_128(x)) { fprintf(stderr, "pow6_64 fail\n"); return false; }
    }
    // (e) iroot6_i128 boundaries
    for (long long n = 0; n < 5000; ++n) {
        const long long r = iroot6_i128(n);
        if (ipow6(r) > n || ipow6(r + 1) <= n) { fprintf(stderr, "iroot6 fail n=%lld\n", n); return false; }
    }
    fprintf(stderr, "[selftest] host math OK\n");
    return true;
}

// ------------------------------------------------------- GPU plant tests --
// Build a small table, plant known targets, require the kernels to find them.
static bool selftest_gpu(GpuCtx& g, u64 inv216) {
    fprintf(stderr, "[selftest] GPU plant tests (small table N=4096)\n");
    const int N = 4096, S = 24;   // LF ~0.5: S must satisfy 2^S*0.95 >= N(N+1)/2
    std::vector<Slot> slots;
    table_build(N, S, slots);
    gpu_upload_table(g, slots, S);

    // -- pair plant: 20000 random pairs must all be found by k_probe_test --
    {
        const int n = 20000;
        std::vector<u64> fps(n);
        std::vector<u32> exp(n);
        std::vector<u64> pw6(N + 1);
        for (int x = 1; x <= N; ++x) { const u64 x2 = (u64)x * x; pw6[x] = x2 * x2 * x2; }
        srand(1234);
        for (int t = 0; t < n; ++t) {
            const int i = 1 + rand() % N, j = i + rand() % (N - i + 1);
            fps[t] = pw6[i] + pw6[j];
            exp[t] = ((u32)i << 16) | (u32)j;
        }
        u64 *d_fps; u32 *d_exp, *d_ok, *d_bad;
        CU(cudaMalloc(&d_fps, n * 8)); CU(cudaMalloc(&d_exp, n * 4));
        CU(cudaMalloc(&d_ok, 4)); CU(cudaMalloc(&d_bad, 4));
        CU(cudaMemcpy(d_fps, fps.data(), n * 8, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_exp, exp.data(), n * 4, cudaMemcpyHostToDevice));
        CU(cudaMemset(d_ok, 0, 4)); CU(cudaMemset(d_bad, 0, 4));
        k_probe_test<<<(n + 255) / 256, 256>>>(d_fps, d_exp, n, g.d_tab, g.tab_slots - 1, S, d_ok, d_bad);
        CU(cudaGetLastError()); CU(cudaDeviceSynchronize());
        u32 ok = 0, bad = 0;
        CU(cudaMemcpy(&ok, d_ok, 4, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(&bad, d_bad, 4, cudaMemcpyDeviceToHost));
        cudaFree(d_fps); cudaFree(d_exp); cudaFree(d_ok); cudaFree(d_bad);
        fprintf(stderr, "[plant] pairs: ok=%u bad=%u\n", ok, bad);
        if (bad != 0) return false;
    }

    auto run_and_count_exact = [&](int cls, std::vector<GpuCand>& cands,
                                   const std::vector<u128>& targets) -> int {
        RunResult R = gpu_run(g, cls, cands, inv216);
        std::vector<char> got(cands.size(), 0);
        for (const Hit& H : R.hits) {
            if (H.cand >= cands.size()) continue;
            const GpuCand& C = cands[H.cand];
            const u128 a6 = h_pow6_128(H.a), b6 = h_pow6_128(H.b);
            bool ex = false;
            if (cls == 1) ex = (a6 + b6 + h_pow6_128(H.c) + h_pow6_128(H.d) == targets[H.cand]);
            else if (cls == 5) {
                const u128 base = ((u128)C.q_hi << 64) | C.q_lo;
                ex = ((u128)M42 * (a6 + b6) + h_pow6_128(21 * H.c) + h_pow6_128(14 * H.d) == base);
            } else {
                const u128 base = ((u128)C.q_hi << 64) | C.q_lo;
                const long long f = (cls == 2) ? 14 : (cls == 3) ? 21 : 7;
                ex = ((u128)M42 * (a6 + b6 + h_pow6_128(H.c)) + h_pow6_128(f * (long long)H.d) == base);
            }
            if (ex) got[H.cand] = 1;
        }
        int cnt = 0;
        for (char c : got) cnt += c;
        return cnt;
    };

    srand(777);
    // -- cls1 plant: Q = sum of 4 distinct sixth powers --
    {
        const int T = 512;
        std::vector<GpuCand> cands(T);
        std::vector<u128> targets(T);
        for (int t = 0; t < T; ++t) {
            int cs[4];
            do { for (int k = 0; k < 4; ++k) cs[k] = 1 + rand() % N; }
            while (cs[0] == cs[1] || cs[0] == cs[2] || cs[0] == cs[3] || cs[1] == cs[2] || cs[1] == cs[3] || cs[2] == cs[3]);
            const i128 Q = ipow6(cs[0]) + ipow6(cs[1]) + ipow6(cs[2]) + ipow6(cs[3]);
            GpuCand C{};
            C.cls = 1; C.B = 42 * N + 1; C.u = 1; C.lim = N;
            C.q_lo = (u64)Q; C.q_hi = (u64)((u128)Q >> 64);
            const long long hi4 = std::min(iroot6_i128(Q), (long long)N);
            long long lo4 = (long long)(hi4 * 0.7937005260); if (lo4 < 1) lo4 = 1;
            while (4 * ipow6(lo4) < Q) ++lo4;
            while (lo4 > 1 && 4 * ipow6(lo4 - 1) >= Q) --lo4;
            C.c4lo = (u32)lo4; C.c4hi = (u32)hi4;
            cands[t] = C; targets[t] = (u128)Q;
        }
        const int got = run_and_count_exact(1, cands, targets);
        fprintf(stderr, "[plant] cls1: %d/%d\n", got, T);
        if (got != T) return false;
    }
    // -- cls234 plant (factor 14): base = (14d)^6 + 42^6(i^6+j^6+k^6) --
    {
        const int T = 512;
        std::vector<GpuCand> cands(T);
        std::vector<u128> targets(T);
        for (int t = 0; t < T; ++t) {
            const int d = 1 + rand() % 280;
            int cs[3];
            do { for (int k = 0; k < 3; ++k) cs[k] = 1 + rand() % N; }
            while (cs[0] == cs[1] || cs[0] == cs[2] || cs[1] == cs[2]);
            const u128 base = (u128)ipow6(14LL * d) + (u128)M42 * ((u128)ipow6(cs[0]) + ipow6(cs[1]) + ipow6(cs[2]));
            GpuCand C{};
            C.cls = 2; C.B = 42 * N + 1; C.u = 1; C.lim = N;
            C.q_lo = (u64)base; C.q_hi = (u64)(base >> 64);
            C.mod1 = M3; C.fmax1 = (u32)d; C.nres1 = 1; C.res[0] = (u32)(d % M3);
            cands[t] = C; targets[t] = base;
        }
        const int got = run_and_count_exact(2, cands, targets);
        fprintf(stderr, "[plant] cls234: %d/%d\n", got, T);
        if (got != T) return false;
    }
    // -- cls5 plant: base = (14e)^6 + (21d)^6 + 42^6(i^6+j^6) --
    {
        const int T = 512;
        std::vector<GpuCand> cands(T);
        std::vector<u128> targets(T);
        for (int t = 0; t < T; ++t) {
            const int e = 1 + rand() % 280, d = 1 + rand() % 190;
            int i, j;
            do { i = 1 + rand() % N; j = 1 + rand() % N; } while (i == j);
            const u128 base = (u128)ipow6(14LL * e) + (u128)ipow6(21LL * d)
                            + (u128)M42 * ((u128)ipow6(i) + ipow6(j));
            GpuCand C{};
            C.cls = 5; C.B = 42 * N + 1; C.u = 1; C.lim = N;
            C.q_lo = (u64)base; C.q_hi = (u64)(base >> 64);
            C.mod1 = M3; C.fmax1 = (u32)e; C.nres1 = 1; C.res[0] = (u32)(e % M3);
            C.mod2 = M2; C.fmax2 = (u32)d; C.nres2 = 1; C.res[1] = (u32)(d % M2);
            cands[t] = C; targets[t] = base;
        }
        const int got = run_and_count_exact(5, cands, targets);
        fprintf(stderr, "[plant] cls5: %d/%d\n", got, T);
        if (got != T) return false;
    }
    fprintf(stderr, "[selftest] GPU plant tests OK\n");
    return true;
}

// =============================================================================
// XCHECK — CPU reference finders (v2 logic) vs GPU kernels, same candidates.
// Requires quad_sum.hpp (+k14_common.hpp). Limited to B_max <= 600000 so the
// CPU pair index fits comfortably next to the GPU table.
// =============================================================================
#if HAVE_XCHECK
static std::optional<std::array<int, 2>> xc_find2(const QuadSumIndex& ix, i128 Q, int lim) {
    if (Q <= 0) return std::nullopt;
    const auto [lo, hi] = ix.equal_sum_range(Q);
    for (size_t p = lo; p < hi; ++p) {
        const int i = ix.pairs[p].i, j = ix.pairs[p].j;
        if (i > lim || j > lim) continue;
        if (i < j) return std::array<int, 2>{i, j};
    }
    return std::nullopt;
}
static std::optional<std::array<int, 3>> xc_find3(const QuadSumIndex& ix, i128 Q, int lim) {
    if (Q <= 0) return std::nullopt;
    long long hi = iroot6_i128(Q), lo = iroot6_i128(Q / 3);
    while (3 * ipow6(lo) < Q) ++lo;
    if (hi > lim) hi = lim;
    for (long long c3 = hi; c3 >= lo; --c3) {
        const i128 rem = Q - ipow6(c3);
        if (rem <= 0) continue;
        auto p = xc_find2(ix, rem, (int)std::min((long long)lim, c3));
        if (!p) continue;
        int a = (*p)[0], b = (*p)[1];
        if (a != b && b != (int)c3 && a != (int)c3) return std::array<int, 3>{a, b, (int)c3};
    }
    return std::nullopt;
}
static std::optional<std::array<int, 4>> xc_find4(const QuadSumIndex& ix, i128 Q, int lim) {
    if (Q <= 0) return std::nullopt;
    long long hi = iroot6_i128(Q), lo = iroot6_i128(Q / 4);
    while (4 * ipow6(lo) < Q) ++lo;
    if (hi > lim) hi = lim;
    for (long long c4 = hi; c4 >= lo; --c4) {
        const i128 rem = Q - ipow6(c4);
        if (rem <= 0) continue;
        auto t = xc_find3(ix, rem, (int)std::min((long long)lim, c4));
        if (!t) continue;
        int a = (*t)[0], b = (*t)[1], c = (*t)[2];
        if (a != (int)c4 && b != (int)c4 && c != (int)c4) return std::array<int, 4>{a, b, c, (int)c4};
    }
    return std::nullopt;
}
// CPU reference: does this candidate have an all-distinct decomposition?
static bool xc_cpu_found(const QuadSumIndex& ix, const GpuCand& C) {
    const i128 q = ((i128)(u128)C.q_hi << 64) | (i128)(u128)C.q_lo;
    const int lim = (int)C.lim;
    if (C.cls == 1) return xc_find4(ix, q, lim).has_value();
    if (C.cls == 5) {
        const i128 base = q;
        const u32 k1 = C.fmax1 / C.mod1 + 1;
        for (u32 t1 = 0; t1 < C.nres1 * k1; ++t1) {
            const u32 e = C.res[t1 % C.nres1] + (t1 / C.nres1) * C.mod1;
            if (!e || e > C.fmax1) continue;
            const i128 R1 = base - ipow6(14LL * e);
            if (R1 <= 0) continue;
            const u32 k2 = C.fmax2 / C.mod2 + 1;
            for (u32 t2 = 0; t2 < C.nres2 * k2; ++t2) {
                const u32 d = C.res[C.nres1 + t2 % C.nres2] + (t2 / C.nres2) * C.mod2;
                if (!d || d > C.fmax2) continue;
                const i128 R = R1 - ipow6(21LL * d);
                if (R <= 0 || R % M42 != 0) continue;
                if (xc_find2(ix, R / M42, lim)) return true;
            }
        }
        return false;
    }
    const long long f = (C.cls == 2) ? 14 : (C.cls == 3) ? 21 : 7;
    const i128 base = q;
    const u32 k1 = C.fmax1 / C.mod1 + 1;
    for (u32 t1 = 0; t1 < C.nres1 * k1; ++t1) {
        const u32 d = C.res[t1 % C.nres1] + (t1 / C.nres1) * C.mod1;
        if (!d || d > C.fmax1) continue;
        const i128 R = base - ipow6(f * (long long)d);
        if (R <= 0 || R % M42 != 0) continue;
        if (xc_find3(ix, R / M42, lim)) return true;
    }
    return false;
}
#endif

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char** argv) {
    // ---- argument parsing ----
    std::vector<std::string> pos;
    long long chunk = 8192, bench_chunks = 0;
    int device = 0, slots_log2 = 0;
    u32 hit_cap = 1u << 20;
    std::string save_table, load_table;
    bool opt_selftest = false, opt_xcheck = false, quiet = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") opt_selftest = true;
        else if (a == "--xcheck") opt_xcheck = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--chunk" && i + 1 < argc) chunk = atoll(argv[++i]);
        else if (a == "--device" && i + 1 < argc) device = atoi(argv[++i]);
        else if (a == "--slots-log2" && i + 1 < argc) slots_log2 = atoi(argv[++i]);
        else if (a == "--hit-cap" && i + 1 < argc) hit_cap = (u32)atol(argv[++i]);
        else if (a == "--save-table" && i + 1 < argc) save_table = argv[++i];
        else if (a == "--load-table" && i + 1 < argc) load_table = argv[++i];
        else if (a == "--bench") bench_chunks = (i + 1 < argc && argv[i + 1][0] != '-') ? atoll(argv[++i]) : 4;
        else if (!a.empty() && a[0] == '-') { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
        else pos.push_back(a);
    }

    const RootTables rt;
    const u64 inv216 = inv216_mod_2_64();

    if (opt_selftest) {
        if (!selftest_host(rt, inv216)) { fprintf(stderr, "SELFTEST FAIL (host)\n"); return 1; }
        int ndev = 0;
        cudaGetDeviceCount(&ndev);
        if (ndev < 1) { fprintf(stderr, "[selftest] no GPU — host part only, PASS\n"); return 0; }
        GpuCtx g;
        gpu_init(g, device, 1u << 16);
        if (!selftest_gpu(g, inv216)) { fprintf(stderr, "SELFTEST FAIL (GPU)\n"); return 1; }
        fprintf(stderr, "SELFTEST PASS\n");
        return 0;
    }

    if (pos.size() < 2 && bench_chunks > 0) {
        // bare "--bench [K]": time K chunks over a representative campaign slice
        pos.push_back("1000000");
        pos.push_back("2200000");
    }
    if (pos.size() < 2) {
        fprintf(stderr,
            "usage: %s <B_min> <B_max> [classes] [u_lo] [u_hi] [options]\n"
            "  classes: \"all\" (default) or e.g. \"1,3\";  u band as fractions of B\n"
            "  options: --chunk K --device K --slots-log2 S --hit-cap N\n"
            "           --save-table F --load-table F --bench [K] --xcheck --quiet\n"
            "           %s --selftest\n", argv[0], argv[0]);
        return 1;
    }
    const long long B_min = atoll(pos[0].c_str());
    const long long B_max = atoll(pos[1].c_str());
    std::vector<int> classes{1, 2, 3, 4, 5};
    if (pos.size() >= 3 && pos[2] != "all") {
        classes.clear();
        for (char c : pos[2]) if (c >= '1' && c <= '5') classes.push_back(c - '0');
        if (classes.empty()) classes = {1, 2, 3, 4, 5};
    }
    const double u_lo = pos.size() >= 4 ? atof(pos[3].c_str()) : 0.0;
    const double u_hi = pos.size() >= 5 ? atof(pos[4].c_str()) : 1.0;

    if (B_max > B_HARD_MAX) { fprintf(stderr, "B_max capped at %lld (i128: B^6 < 2^127)\n", B_HARD_MAX); return 1; }
    const int N = (int)(B_max / 42);
    if (N < 4) { fprintf(stderr, "B_max too small\n"); return 1; }
    if (N > 65535) { fprintf(stderr, "N=%d exceeds 32-bit payload packing (B<=2.75M)\n", N); return 1; }

    GpuCtx g;
    gpu_init(g, device, hit_cap);

    // ---- table: load or build, then upload ----
    int S = slots_log2;
    std::vector<Slot> slots;
    bool loaded = false;
    if (!load_table.empty()) loaded = table_load(load_table.c_str(), slots, N, S);
    if (!loaded) {
        if (S == 0) {                       // smallest pow2 with load factor <= 0.6
            const double P = (double)N * (N + 1) / 2;
            S = 20;
            while ((double)((u64)1 << S) * 0.6 < P) ++S;
        }
        const double P = (double)N * (N + 1) / 2;
        if ((double)((u64)1 << S) * 0.95 < P) {   // probes need empty slots to terminate
            fprintf(stderr, "slots 2^%d too small for %.3e pairs — raise --slots-log2\n", S, P);
            return 1;
        }
        table_build(N, S, slots);
        if (!save_table.empty()) table_save(save_table.c_str(), slots, N, S);
    }
    gpu_upload_table(g, slots, S);
    std::vector<Slot>().swap(slots);   // free the host copy

    // ---- xcheck mode: CPU reference vs GPU over the range ----
    if (opt_xcheck) {
#if HAVE_XCHECK
        if (B_max > 600000) { fprintf(stderr, "xcheck limited to B_max <= 600000\n"); return 1; }
        QuadSumIndex ix;
        ix.build(K, N);
        fprintf(stderr, "[xcheck] CPU index built (N=%d); comparing engines over [%lld, %lld]\n", N, B_min, B_max);
        long long mismatch = 0, checked = 0, both_found = 0;
        long solutions = 0;
        std::set<std::array<long long, 5>> reported;
        for (long long B = std::max(B_min, 86LL); B <= B_max; ++B) {
            if (!(B & 1) || B % 3 == 0 || B % 7 == 0) continue;
            for (int cls : classes) {
                std::vector<GpuCand> cands;
                gen_class_cands(rt, B, cls, u_lo, u_hi, cands);
                if (cands.empty()) continue;
                RunResult R = gpu_run(g, cls, cands, inv216);
                std::vector<char> gpu_exact(cands.size(), 0);
                for (const Hit& H : R.hits) {
                    if (H.cand >= cands.size()) continue;
                    long dummy = 0;
                    const int v = verify_hit(cands[H.cand], H, dummy, reported);
                    if (v >= 1) gpu_exact[H.cand] = 1;
                }
                for (size_t k = 0; k < cands.size(); ++k) {
                    ++checked;
                    const bool cpu = xc_cpu_found(ix, cands[k]);
                    if (cpu && !gpu_exact[k]) {
                        ++mismatch;
                        fprintf(stderr, "[xcheck] MISS cls=%d B=%lld u=%u — CPU found, GPU missed!\n",
                                cls, B, cands[k].u);
                    }
                    if (cpu && gpu_exact[k]) ++both_found;
                }
            }
        }
        fprintf(stderr, "[xcheck] candidates=%lld cpu-found=%lld cpu∩gpu=%lld CPU-found-but-GPU-missed=%lld %s\n",
                checked, both_found + mismatch, both_found, mismatch, mismatch == 0 ? "PASS" : "FAIL");
        return mismatch == 0 ? 0 : 1;
#else
        fprintf(stderr, "xcheck needs quad_sum.hpp (and k14_common.hpp) in the include path\n");
        return 1;
#endif
    }

    // ---- normal campaign loop ----
    fprintf(stderr, "solve_516_v3: B in [%lld, %lld], classes ", B_min, B_max);
    for (int c : classes) fprintf(stderr, "%d", c);
    fprintf(stderr, ", unit band [%.4f, %.4f)*B, chunk=%lld\n", u_lo, u_hi, chunk);

    long solutions = 0;
    long long stat_cands[6] = {0}, stat_exact[6] = {0}, stat_false[6] = {0};
    u64 stat_probes[6] = {0};
    double stat_ms[6] = {0};
    std::set<std::array<long long, 5>> reported;

    long long chunks_done = 0;
    for (long long c0 = B_min; c0 <= B_max; c0 += chunk) {
        const long long c1 = std::min(B_max, c0 + chunk - 1);
        std::vector<long long> Bl;
        for (long long B = std::max(c0, 86LL); B <= c1; ++B)
            if ((B & 1) && B % 3 != 0 && B % 7 != 0) Bl.push_back(B);

        std::vector<GpuCand> gc[6];
#pragma omp parallel for schedule(dynamic, 16)
        for (size_t bi = 0; bi < Bl.size(); ++bi) {
            std::vector<GpuCand> loc[6];
            for (int cls : classes) gen_class_cands(rt, Bl[bi], cls, u_lo, u_hi, loc[cls]);
            for (int cls : classes)
                if (!loc[cls].empty()) {
#pragma omp critical
                    gc[cls].insert(gc[cls].end(), loc[cls].begin(), loc[cls].end());
                }
        }

        for (int cls : classes) {
            if (gc[cls].empty()) continue;
            RunResult R = gpu_run(g, cls, gc[cls], inv216);
            stat_cands[cls] += (long long)gc[cls].size();
            stat_probes[cls] += R.probes;
            stat_ms[cls] += R.kernel_ms;
            if (R.overflow)
                fprintf(stderr, "!! hit buffer overflow at B<=%lld — rerun with larger --hit-cap\n", c1);
            for (const Hit& H : R.hits) {
                if (H.cand >= gc[cls].size()) continue;
                const int v = verify_hit(gc[cls][H.cand], H, solutions, reported);
                if (v >= 1) stat_exact[cls]++; else stat_false[cls]++;
            }
        }
        if (!quiet) {
            fprintf(stderr, "B %lld..%lld | cands", c0, c1);
            for (int cls : classes) fprintf(stderr, " c%d=%lld", cls, stat_cands[cls]);
            fprintf(stderr, " | probes");
            for (int cls : classes) fprintf(stderr, " c%d=%.2e", cls, (double)stat_probes[cls]);
            fprintf(stderr, " | sol=%ld\n", solutions);
        }
        if (bench_chunks > 0 && ++chunks_done >= bench_chunks) break;
    }

    fprintf(stderr, "---- summary ----\n");
    for (int cls : classes)
        if (stat_cands[cls])
            fprintf(stderr, "cls%d: cands=%lld probes=%.4e kernel=%.1fs rate=%.2e probes/s exact=%lld fp-false=%lld\n",
                    cls, stat_cands[cls], (double)stat_probes[cls], stat_ms[cls] / 1e3,
                    stat_probes[cls] / std::max(1e-9, stat_ms[cls] / 1e3), stat_exact[cls], stat_false[cls]);
    fprintf(stderr, "total solutions: %ld\n", solutions);
    return 0;
}
