// =============================================================================
// solve_616_v1.cu — GPU-accelerated five-class (6,1,6) solver
//     a1^6 + a2^6 + a3^6 + a4^6 + a5^6 + a6^6 = B^6,  1<=a1<...<a6<B, gcd=1
// =============================================================================
//
// Sibling of solve_516_v3.cu (same table, same kernels, same validation
// philosophy). THE CLASS SYSTEM IS NEW but built from the same parts.
//
// MATHEMATICAL STRUCTURE (completeness argument):
//   For a primitive solution, B must be coprime to 42 (mod-8/9/7 counting:
//   B even => all a_i even; 3|B => all a_i/3; 7|B => all a_i/7 — non-primitive,
//   a d^6-scaling of a smaller primitive one). With gcd(B,42)=1, sixth powers
//   are 0/1 mod 8,9,7, so among the six terms there is EXACTLY ONE odd term,
//   EXACTLY ONE term not divisible by 3 ("~3"), and EXACTLY ONE term not
//   divisible by 7 ("~7"). The three "roles" {odd, ~3, ~7} are distributed
//   over the six terms in one of the FIVE set-partitions, and every term not
//   carrying a role is divisible by 42. Distinguishing the role-carrying
//   terms as the "free" terms u, v(, w) and writing the divisible terms as
//   42*c (seeds c <= B/42) gives five classes:
//
//   cls1  {o37}        u: gcd(u,42)=1, u^6≡B^6 (42^6) [144 classes];
//                      sixth term = 42w (w = any seed); 4 seeds -> find4.
//                      Q = (B^6-u^6)/42^6 - w^6 = c1^6+c2^6+c3^6+c4^6
//   cls2  {o7}|{3}     u: odd, 3|u, u^6≡B^6 (14^6) [24 classes];
//                      v = 14v', (14v')^6≡B^6 (3^6) [6 classes]; 4 seeds.
//                      Q = (B^6-u^6-(14v')^6)/42^6
//   cls3  {37}|{o}     u: even, gcd(u,21)=1, u^6≡B^6 (21^6) [36 classes];
//                      v = 21v', (21v')^6≡B^6-u^6 (2^6) [4 classes]; 4 seeds.
//                      Q = (B^6-u^6-(21v')^6)/42^6
//   cls4  {o3}|{7}     u: odd, ~3, 7|u, u^6≡B^6 (2^6*3^6) [24 classes];
//                      v = 6v', (6v')^6≡B^6 (7^6) [6 classes]; 4 seeds.
//                      Q = (B^6-u^6-(6v')^6)/42^6
//   cls5  {o}|{3}|{7}  u = 14u', (14u')^6≡B^6 (3^6)  [~3 term];
//                      v = 6v',  (6v')^6≡B^6 (7^6)  [~7 term];
//                      kernel grid w = 21w', (21w')^6≡B^6 (2^6) [odd term];
//                      3 seeds -> find3:  T = (B^6-(14u')^6-(6v')^6-(21w')^6)/42^6
//                      = c1^6+c2^6+c3^6   (k_cls234, factor 21)
//
//   The partitions are disjoint by role-placement, so every primitive
//   solution lies in EXACTLY ONE class (up to interchangeable 42-divisible
//   terms, deduplicated at verification). Classes are COMPLETE by the same
//   mod-8/9/7 counting as v2/v3 (Gerbicz-Meyrignac-Beckert arXiv:1108.0462).
//
// KERNEL REUSE: cls1-4 run k_cls1 verbatim (host precomputes the find4
// target Q and the c4 window). cls5 runs k_cls234 verbatim (factor=21; the
// grid term is the odd role-carrier 21w'). 515's cls4/cls5 kernels are gone.
//
// PROBE GATE (mod 124,488 = 8*9*7*13*19): a pair sum i^6+j^6 lands in only
//   3/8, 3/9, 3/7 of residues mod 8,9,7 (sixth powers are {0,1}) and only
//   5/13, 10/19 mod 13,19 (sixth powers are {0,±1} mod 13, {0,1,7,11} mod 19).
//   By CRT a target is achievable mod 124,488 in just 27*50/124,488 = 1.08%
//   of residues. The kernels carry two tiny SHARED bitmaps (504-bit + 247-bit
//   — the CRT-factored form of the 124,488-entry bitmap, same predicate) and
//   test (T - c3^6) before every table probe: 98.9% of probes die for ~20 ALU
//   cycles and never touch the table. c3's residues advance incrementally
//   (stride add + conditional subtract), so the hot loop has NO divisions.
//   Sound by construction (every real pair sum passes); the plant tests and
//   xcheck would fail loudly otherwise. --no-gate disables it (A/B bench).
//   (Design: 615-ribbon-filter-plan.md; projected ~90x probe-stage speedup.)
//
// GpuCand fields (per class):
//   cls1: q=Q (find4 target, = Q0 - w^6), c4lo/hi; u=role term, w=42*w (6th term)
//   cls2: q=Q, c4lo/hi; u=role term, w=14*v'
//   cls3: q=Q, c4lo/hi; u=role term, w=21*v'
//   cls4: q=Q, c4lo/hi; u=role term, w=6*v'
//   cls5: q=base=B^6-(14u')^6-(6v')^6; res/mod1/fmax1 = w'-grid (mod 64);
//         u=14*u' (~3 term), w=6*v' (~7 term); factor=21
//
// COST MODEL (per-B, B in units of 1e6; find4 ~3.2e7 probes/cand, find3
//   window ~7.4e3 probes per grid value — windows scale B^2, cand counts ~B^2,
//   so every class scales B^4):
//     cls1 ~1.3e10  cls2 ~6e10  cls3 ~4e10  cls4 ~2e10  cls5 ~1.1e11 probes/B
//   cls5 dominates (u' x v' anchors ~ B^2/2e8, each with grid ~B/336 x window).
//   TOTAL ~2.3e11 probes/B at 1M (B^4): at the 615-measured ~1.4e10 probes/s
//     [110267, 400k] ~ 2.5-3 h, [110267, 500k] ~ 8 h on the RTX PRO 6000.
// KNOWN BOUND: EulerNet (euler.free.fr/progress.htm) covered (6,1,6) to
//   B = 110,266 by Jan 2000 — new ground starts at 110,267. xcheck overlap
//   target: [100000, 130000] (inside their range: any "solution" there is a bug).
//
// Build (server, CUDA 12.4 stopgap for sm_120):
//   nvcc -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 -o solve_616_v1 solve_616_v1.cu -lineinfo
// Host-only logic check (no GPU needed):
//   g++ -O2 -std=c++17 -DHOST_ONLY -fopenmp -o solve_616_host solve_616_v1.cu && ./solve_616_host
// Run:
//   ./solve_616_v1 --selftest
//   ./solve_616_v1 100000 130000 all --xcheck
//   ./solve_616_v1 110267 400000 all
// =============================================================================

#ifndef HOST_ONLY
#include <cuda_runtime.h>
#endif

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

#ifdef HOST_ONLY
#define HD inline
#else
#define HD __host__ __device__ __forceinline__
#endif

// ---------------------------------------------------------------- constants --
static constexpr long long M2 = 64LL;         // 2^6
static constexpr long long M3 = 729LL;        // 3^6
static constexpr long long M7 = 117649LL;     // 7^6
static constexpr long long M14 = 7529536LL;   // 14^6
static constexpr long long M21 = 85766121LL;  // 21^6  (odd part of 42^6)
static constexpr long long M42 = 5489031744LL;// 42^6 = 64 * 21^6
static constexpr int K = 6;
static constexpr long long B_HARD_MAX = 2353973;   // B^6 < 2^127 (i128 guard)
static constexpr u64 PHI64 = 0x9E3779B97F4A7C15ULL; // fibonacci hash constant

#ifndef HOST_ONLY
#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)
#endif

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

// i128 sixth power (host; B <= 2.35M so x^6 < 2^127 always).
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
// Same as v2/v3: for every sixth-power residue r among UNITS mod m, the units
// a with a^6 ≡ r. mod 2^6: 8 residues x 4 roots; mod 3^6: 81 x 6; mod 7^6:
// 16807 x 6 (Teichmuller).
struct RootTables {
    std::vector<std::vector<long long>> r2, r3, r7;
    RootTables() : r2(M2), r3(M3), r7(M7) {
        for (long long a = 1; a < M2; ++a) if (std::gcd(a, M2) == 1) r2[mod_pow6(a, M2)].push_back(a);
        for (long long a = 1; a < M3; ++a) if (std::gcd(a, M3) == 1) r3[mod_pow6(a, M3)].push_back(a);
        for (long long a = 1; a < M7; ++a) if (std::gcd(a, M7) == 1) r7[mod_pow6(a, M7)].push_back(a);
    }
};

// roots of (scale*x)^6 ≡ B^6 (mod m): x^6 ≡ B^6 * (scale^6)^{-1}.
static std::vector<long long> scaled_roots(const std::vector<std::vector<long long>>& tab,
                                           long long b6, long long scale, long long mod) {
    const long long t = b6 * mod_inv(mod_pow6(scale, mod), mod) % mod;
    return tab[t];
}

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
    } else {                 // cls 3: 36 classes mod 21^6
        for (long long s3 : rt.r3[b3])
            for (long long s7 : rt.r7[b7]) out.push_back(crt2(s3, M3, s7, M7));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// (14d)^6 ≡ B^6 (3^6): 6 classes mod 729. (cls2 v', cls5 u')
static std::vector<long long> classes_d_cls2(const RootTables& rt, long long B) {
    const long long t = mod_pow6(B, M3) * mod_inv(mod_pow6(14, M3), M3) % M3;
    return rt.r3[t];
}
// (21d)^6 ≡ B^6-u^6 (2^6): 4 classes mod 64. (cls3 v')
static std::vector<long long> classes_d_cls3(const RootTables& rt, long long B, long long u) {
    long long rhs = (mod_pow6(B, M2) - mod_pow6(u, M2)) % M2; if (rhs < 0) rhs += M2;
    const long long t = rhs * mod_inv(mod_pow6(21, M2), M2) % M2;
    return rt.r2[t];
}
// u^6 ≡ B^6 (2^6, 3^6): 24 classes mod 46656. (cls4 u)
static std::vector<long long> classes_d_cls4(const RootTables& rt, long long B) {
    const long long t2 = mod_pow6(B, M2) * mod_inv(mod_pow6(7, M2), M2) % M2;
    const long long t3 = mod_pow6(B, M3) * mod_inv(mod_pow6(7, M3), M3) % M3;
    std::vector<long long> out;
    for (long long a : rt.r2[t2]) for (long long b : rt.r3[t3]) out.push_back(crt2(a, M2, b, M3));
    std::sort(out.begin(), out.end());
    return out;
}

// ------------------------------------------------- 616 class glue (host) --
// primary free-term classes and modulus per class
static std::vector<long long> seeds616(const RootTables& rt, long long B, int cls) {
    switch (cls) {
        case 1: return seeds_for_B(rt, B, 1);          // mod 42^6
        case 2: return seeds_for_B(rt, B, 2);          // mod 14^6
        case 3: return seeds_for_B(rt, B, 3);          // mod 21^6
        case 4: return classes_d_cls4(rt, B);          // mod 2^6*3^6 = 46656
        default: return classes_d_cls2(rt, B);         // cls5 u': (14u')^6≡B^6 (3^6)
    }
}
static long long mod616(int cls) {
    switch (cls) {
        case 1: return M42; case 2: return M14; case 3: return M21;
        case 4: return M2 * M3; default: return M3;
    }
}
static bool unit_ok616(long long u, int cls) {
    switch (cls) {
        case 1: return (u & 1) && std::gcd(u, 42LL) == 1;   // auto from seeds (safety)
        case 2: return (u & 1) && (u % 3 == 0);             // odd (auto); ÷3
        case 3: return !(u & 1) && std::gcd(u, 21LL) == 1;  // even; ~3 ~7 (auto)
        case 4: return (u % 7 == 0);                        // odd,~3 (auto); ÷7
        default: return true;                               // cls5 u': ~3 auto
    }
}

// =============================================================================
// PAIR TABLE (host build, device query) — identical to v3.
// Slot: 16 bytes. payload==0 marks empty. payload=(i<<16)|j, i<=j, N<=65535.
// =============================================================================
struct Slot { u64 key; u32 payload; u32 pad; };

HD u64 mix64(u64 x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}
HD u64 hash_pos(u64 fp, int S) {
    return (fp * PHI64) >> (64 - S);
}

// Build slots for all pairs 1<=i<=j<=N, fp = (i^6+j^6) mod 2^64.
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
                    slots[pos].key = fp;
                    break;
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
    TableHeader h; memcpy(h.magic, "616TBL01", 8); h.N = (u64)N; h.S = (u64)S;
    fwrite(&h, sizeof(h), 1, f);
    fwrite(slots.data(), sizeof(Slot), slots.size(), f);
    fclose(f);
    fprintf(stderr, "[table] saved to %s (%.1f GB)\n", path, slots.size() * 16.0 / 1e9);
}
static bool table_load(const char* path, std::vector<Slot>& slots, int N, int& S) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    TableHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1 || memcmp(h.magic, "616TBL01", 8) != 0) {
        fclose(f); fprintf(stderr, "[table] bad file %s\n", path); return false;
    }
    if ((int)h.N != N) {
        fclose(f); fprintf(stderr, "[table] N mismatch: file N=%llu, need %d — rebuilding\n", (unsigned long long)h.N, N);
        return false;
    }
    S = (int)h.S;
    slots.assign((size_t)1 << S, Slot{0, 0, 0});
    const size_t want = slots.size() * sizeof(Slot);
    const size_t got = fread(slots.data(), sizeof(Slot), want, f);
    fclose(f);
    if (got != want) { fprintf(stderr, "[table] short read %s\n", path); return false; }
    fprintf(stderr, "[table] loaded %s (N=%d, 2^%d slots)\n", path, N, S);
    return true;
}

// ------------------------------------------------------- probe gate (host) --
// Pair-sum achievability bitmaps, CRT-factored: x mod 124,488 is achievable
//   <=> g504[x % 504] && g247[x % 247]        (124,488 = 504 * 247, coprime)
struct GateData {
    u64 g504[8] = {0}, g247[4] = {0};   // 504-bit / 247-bit achievability maps
    u16 p6504[504], p6247[247];         // c^6 mod 504 / mod 247 lookup tables
};
static void gate_set(u64* b, int x) { b[x >> 6] |= 1ULL << (x & 63); }
static bool gate_get(const u64* b, int x) { return (b[x >> 6] >> (x & 63)) & 1ULL; }
static void build_gate(GateData& g) {
    for (int i = 0; i < 504; ++i) {
        const int si = (int)mod_pow6(i, 504);
        g.p6504[i] = (u16)si;
        for (int j = i; j < 504; ++j) gate_set(g.g504, (si + (int)mod_pow6(j, 504)) % 504);
    }
    for (int i = 0; i < 247; ++i) {
        const int si = (int)mod_pow6(i, 247);
        g.p6247[i] = (u16)si;
        for (int j = i; j < 247; ++j) gate_set(g.g247, (si + (int)mod_pow6(j, 247)) % 247);
    }
}

// --------------------------------------------------- triple-sum window gate --
// (6,1,6) cls5 only. Necessary condition on the whole find3 target
//   T = R/42^6:  T must be writable as c1^6+c2^6+c3^6, tested ONCE per block
// (not per c3). Sound (weaker than the per-c3 pair gate); zero false negatives.
// Uses CRT factors 504 (=8*9*7) and 247 (=13*19) — the SAME residues the pair
// gate already computes per block, so the triple test is just two extra bitmap
// lookups. (Independent moduli 11 and 25 were measured to be fully dense for
// three sixth powers — 100% pass — so they add no filtering and are omitted.)
struct TriData {
    u64 t504[8] = {0}, t247[4] = {0};
    int n504 = 0, n247 = 0;   // per-modulus pass counts (for the "keeps %" diagnostic)
};
// Fill bitmap b with { (a^6+b^6+c^6) mod m }, computed as p3 = (p2 = sp+sp)+sp
// over the sixth-power residue set sp. Returns the number of set bits.
static int build_tri_set(u64* b, int m) {
    std::vector<char> sp(m, 0), p2(m, 0), p3(m, 0);
    for (int i = 0; i < m; ++i) sp[(int)mod_pow6(i, m)] = 1;
    for (int a = 0; a < m; ++a) if (sp[a]) for (int c = 0; c < m; ++c) if (sp[c]) p2[(a + c) % m] = 1;
    for (int a = 0; a < m; ++a) if (p2[a]) for (int c = 0; c < m; ++c) if (sp[c]) p3[(a + c) % m] = 1;
    int cnt = 0;
    for (int x = 0; x < m; ++x) if (p3[x]) { gate_set(b, x); ++cnt; }
    return cnt;
}
static void build_tri_gate(TriData& t) {
    t.n504 = build_tri_set(t.t504, 504);
    t.n247 = build_tri_set(t.t247, 247);
}

// =============================================================================
// DEVICE SIDE
// =============================================================================
// Candidate batch record. q_lo/q_hi carry:
//   cls1-4: Q (find4 target, exact, CPU-divided)           — c4lo/c4hi set
//   cls5:   base = B^6-(14u')^6-(6v')^6 (128-bit)          — w'-grid fields set
// w: second free term's ACTUAL value (host-side only; kernels ignore it):
//   cls1: 42*wseed, cls2: 14*v', cls3: 21*v', cls4: 6*v', cls5: 6*v'
struct GpuCand {
    u64 q_lo, q_hi;        // 16
    u32 B, u, lim;         // 12
    u32 c4lo, c4hi;        // 8   (cls1-4 only)
    u32 fmax1, fmax2;      // 8   (fmax2 unused in 616)
    u32 mod1, mod2;        // 8   (mod2 unused in 616)
    u32 cls, nres1, nres2, w; // 16 (w replaces v3's pad)
    u32 q504, q247;        // 8   (cls1-4: Q mod 504 / mod 247, for the gate)
    u32 res[24];           // 96
};                         // ~172 bytes

struct Hit { u32 cand, a, b, c, d; };  // a,b = pair indices; c,d = context (c3,c4 | c3,w')

#ifndef HOST_ONLY

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
    u32 factor;              // cls5 grid-term factor: 21
    u32 use_gate;            // 0 disables the probe gate (--no-gate)
    const u64* __restrict__ g504;   // gate bitmaps (8 and 4 words)
    const u64* __restrict__ g247;
    const u16* __restrict__ p6504;  // c^6 mod 504 / mod 247 tables
    const u16* __restrict__ p6247;
    u64* __restrict__ calls;        // probe() invocations
    u64* __restrict__ gated;        // probes killed by the gate
    u64 c_m504;                // 2^64 mod (504*42^6)   (cls5 T-mod reduction)
    u64 c_m247;                // 2^64 mod 247
    u64 inv42_247;             // (42^6)^{-1} mod 247
    // --- cls5 triple-sum window gate (once-per-block; reuses t504/t247 residues) ---
    u32 use_tri_gate;               // 0 disables it (--no-tri-gate)
    const u64* __restrict__ t504;   // triple-sum achievability bitmaps
    const u64* __restrict__ t247;
    u64* __restrict__ tri_skipped;  // blocks killed by the triple-sum gate
    u64* __restrict__ tri_total;    // blocks that reached the gate (valid window)
    u32 defer_gate_load;            // cls5: gate_load only after s_active (not before)
};

// ------------------------------------------- device 64/128-bit primitives --
__device__ __forceinline__ u64 pow6_64(u32 x) {
    const u64 x2 = (u64)x * x;
    return x2 * x2 * x2;
}
__device__ __forceinline__ void pow6_128(u32 x, u64& hi, u64& lo) {
    const u64 x2 = (u64)x * x;
    const u64 h4 = __umul64hi(x2, x2), l4 = x2 * x2;
    lo = l4 * x2;
    hi = __umul64hi(l4, x2) + h4 * x2;
}
__device__ __forceinline__ bool mul128_small(u64& hi, u64& lo, u64 m) {
    const u64 h1 = __umul64hi(lo, m);
    const u64 h2 = hi * m;
    bool of = (hi != 0) && (__umul64hi(hi, m) != 0);
    hi = h1 + h2;
    of |= (hi < h1);
    lo = lo * m;
    return of;
}
__device__ __forceinline__ bool gt128(u64 ah, u64 al, u64 bh, u64 bl) {
    return (ah > bh) || (ah == bh && al > bl);
}
__device__ __forceinline__ bool scaled_gt(u64 rh, u64 rl, u32 c, u64 scale) {
    u64 ch, cl; pow6_128(c, ch, cl);
    if (mul128_small(ch, cl, scale)) return true;
    return gt128(ch, cl, rh, rl);
}
__device__ u32 iroot6_fix(u64 rh, u64 rl, double inv_scale, u64 scale) {
    const double d = (double)rh * 18446744073709551616.0 + (double)rl;
    u32 r = (u32)(pow(d * inv_scale, 1.0 / 6.0)) + 2;
    while (r > 0 && scaled_gt(rh, rl, r, scale)) --r;
    while (!scaled_gt(rh, rl, r + 1, scale)) ++r;
    return r;
}
__device__ __forceinline__ bool scaled_lt(u64 rh, u64 rl, u32 c, u64 scale) {
    u64 ch, cl; pow6_128(c, ch, cl);
    if (mul128_small(ch, cl, scale)) return false;
    return gt128(rh, rl, ch, cl);
}
__device__ u32 min_c_fix(u64 rh, u64 rl, u32 r_hi, u64 scale) {
    u32 c = (u32)(r_hi * 0.8326831149);
    if (c < 1) c = 1;
    while (scaled_lt(rh, rl, c, 3 * scale)) ++c;
    while (c > 1 && !scaled_lt(rh, rl, c - 1, 3 * scale)) --c;
    return c;
}
// fp64 of target R/42^6 given R (128-bit, divisible by 64). = ((R>>6) mod 2^64) * 21^-6.
__device__ __forceinline__ u64 funnel_fp(u64 rh, u64 rl, u64 inv216) {
    return ((rl >> 6) | (rh << 58)) * inv216;
}

// (hi:lo) mod m, with c = 2^64 mod m precomputed (host). Requires m < 2^42.
__device__ u64 mod128_64(u64 hi, u64 lo, u64 m, u64 c) {
    const u64 r = hi % m;
    const u64 ph = __umul64hi(r, c), pl = r * c;   // r*c < 2^84
    u64 t = (ph * c) % m;                          // ph < 2^22 -> ph*c < 2^64
    t += pl % m; if (t >= m) t -= m;
    t += lo % m; if (t >= m) t -= m;
    return t;
}

// ------------------------------------------------------------- probe gate --
// Shared copy of the factored bitmaps + sixth-power tables (~1.6 KB/block).
struct GateSh { u64 g504[8]; u64 g247[4]; u16 p6504[504]; u16 p6247[247]; };
__device__ __forceinline__ void gate_load(GateSh& s, const Params& P) {
    for (int i = threadIdx.x; i < 8;   i += blockDim.x) s.g504[i]  = P.g504[i];
    for (int i = threadIdx.x; i < 4;   i += blockDim.x) s.g247[i]  = P.g247[i];
    for (int i = threadIdx.x; i < 504; i += blockDim.x) s.p6504[i] = P.p6504[i];
    for (int i = threadIdx.x; i < 247; i += blockDim.x) s.p6247[i] = P.p6247[i];
    __syncthreads();
}
// Is (T - c3^6) an achievable pair sum mod 124,488? c3 enters via its
// residues r504/r247 (maintained incrementally by the caller: stride add +
// conditional subtract — no divisions in the hot loop).
__device__ __forceinline__ bool gate_pass(const GateSh& s, u32 t504, u32 t247,
                                          u32 r504, u32 r247) {
    u32 x = t504 + 504u - s.p6504[r504]; if (x >= 504u) x -= 504u;
    if (!((s.g504[x >> 6] >> (x & 63)) & 1ULL)) return false;
    u32 y = t247 + 247u - s.p6247[r247]; if (y >= 247u) y -= 247u;
    return (s.g247[y >> 6] >> (y & 63)) & 1ULL;
}

// Direct achievability test for the triple-sum window gate (no c3 offset —
// the target residue is looked up as-is; sound necessary condition on T).
__device__ __forceinline__ bool tri_get(const u64* __restrict__ b, u32 x) {
    return (b[x >> 6] >> (x & 63)) & 1ULL;
}

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
// find4 (616 cls1-4): Q = c1^6+c2^6+c3^6+c4^6. block per (cand, c4-chunk),
// warp per c4, lanes over c3 window.
__global__ void k_cls1(Params P) {
    const u32 ci = blockIdx.x;
    if (ci >= P.n_cand) return;
    const GpuCand C = P.cands[ci];
    __shared__ GateSh s_gate;
    gate_load(s_gate, P);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const u32 base = C.c4lo + blockIdx.y * 2048 + warp;
    u64 cnt = 0, ncall = 0, skip = 0;
    for (u32 c4 = base; c4 <= C.c4hi; c4 += 8) {
        u64 c4h, c4l; pow6_128(c4, c4h, c4l);
        const u64 rlo = C.q_lo - c4l;
        const u64 rhi = C.q_hi - c4h - (C.q_lo < c4l ? 1 : 0);
        if ((rhi >> 63) || ((rhi | rlo) == 0)) continue;
        u32 hi3 = iroot6_fix(rhi, rlo, 1.0, 1);
        const u32 capm = C.lim < c4 ? C.lim : c4;
        if (hi3 > capm) hi3 = capm;
        if (hi3 < 1) continue;
        const u32 lo3 = min_c_fix(rhi, rlo, hi3, 1);
        const u64 tfp = C.q_lo - pow6_64(c4);
        u32 t4504 = 0, t4247 = 0, r504 = 0, r247 = 0;
        if (P.use_gate) {
            const u32 c4r504 = c4 % 504, c4r247 = c4 % 247;
            t4504 = C.q504 + 504u - s_gate.p6504[c4r504]; if (t4504 >= 504u) t4504 -= 504u;
            t4247 = C.q247 + 247u - s_gate.p6247[c4r247]; if (t4247 >= 247u) t4247 -= 247u;
            r504 = (lo3 + lane) % 504; r247 = (lo3 + lane) % 247;
        }
        for (u32 c3 = lo3 + lane; c3 <= hi3; c3 += 32) {
            if (!P.use_gate || gate_pass(s_gate, t4504, t4247, r504, r247)) {
                ++ncall;
                cnt += probe(P, tfp - pow6_64(c3), ci, c3, c4);
            } else ++skip;
            if (P.use_gate) {
                r504 += 32; if (r504 >= 504u) r504 -= 504u;
                r247 += 32; if (r247 >= 247u) r247 -= 247u;
            }
        }
    }
    if (cnt) atomicAdd(P.probes, cnt);
    if (ncall) atomicAdd(P.calls, ncall);
    if (skip) atomicAdd(P.gated, skip);
}

// find3 (616 cls5): T = (base - (21w')^6)/42^6, window over c3.
// block per (cand, w'-flat-index); thread 0 computes R and the c3 window.
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
    __shared__ u32 s_t504, s_t247;
    __shared__ GateSh s_gate;
    // Baseline (defer off): gate_load on every launched block, then maybe return.
    // Optimized (defer on):  thread 0 decides s_active first; dead blocks never
    // load the ~1.6 KB gate bitmaps into shared memory.
    if (!P.defer_gate_load) gate_load(s_gate, P);
    if (threadIdx.x == 0) {
        s_active = 0;
        u64 fh, fl; pow6_128(P.factor * d, fh, fl);              // (21*w')^6
        const u64 rlo = C.q_lo - fl;
        const u64 rhi = C.q_hi - fh - (C.q_lo < fl ? 1 : 0);     // R = base - (21w')^6
        if ((rhi >> 63) == 0 && ((rhi | rlo) != 0)) {
            u32 hi3 = iroot6_fix(rhi, rlo, 1.0 / (double)M42, (u64)M42);
            if (hi3 > C.lim) hi3 = C.lim;
            if (hi3 >= 1) {
                const u32 lo3 = min_c_fix(rhi, rlo, hi3, (u64)M42);
                if (lo3 <= hi3) {
                    s_lo = lo3; s_hi = hi3;
                    s_tfp = funnel_fp(rhi, rlo, P.inv216);
                    // gate residues of T = R/42^6 (R is divisible by 42^6):
                    //   T mod 504 = ((R mod 504*42^6) / 42^6) mod 504
                    //   T mod 247 = (R mod 247) * (42^6)^{-1} mod 247
                    const u64 r1 = mod128_64(rhi, rlo, 504ULL * (u64)M42, P.c_m504);
                    const u32 my504 = (u32)((r1 / (u64)M42) % 504ULL);
                    s_t504 = my504;
                    const u64 r2 = mod128_64(rhi, rlo, 247ULL, P.c_m247);
                    const u32 my247 = (u32)(r2 * P.inv42_247 % 247ULL);
                    s_t247 = my247;
                    // Triple-sum window gate: T must be a sum of three sixth
                    // powers (mod 504 and 247). One test per block; sound. Reuses
                    // the residues already computed above — no extra reductions.
                    bool tri_ok = true;
                    if (P.use_tri_gate) {
                        tri_ok = tri_get(P.t504, my504) && tri_get(P.t247, my247);
                        atomicAdd(P.tri_total, 1ULL);
                        if (!tri_ok) atomicAdd(P.tri_skipped, 1ULL);
                    }
                    if (tri_ok) s_active = 1;
                }
            }
        }
    }
    __syncthreads();
    if (!s_active) return;
    if (P.defer_gate_load) gate_load(s_gate, P);
    const u64 tfp = s_tfp;
    const u32 lo = s_lo, hi = s_hi;
    const u32 t504 = s_t504, t247 = s_t247;
    u64 cnt = 0, ncall = 0, skip = 0;
    u32 r504 = (lo + threadIdx.x) % 504, r247 = (lo + threadIdx.x) % 247;
    for (u32 c3 = lo + threadIdx.x; c3 <= hi; c3 += blockDim.x) {
        if (!P.use_gate || gate_pass(s_gate, t504, t247, r504, r247)) {
            ++ncall;
            cnt += probe(P, tfp - pow6_64(c3), ci, c3, d);
        } else ++skip;
        if (P.use_gate) {
            r504 += 256; if (r504 >= 504u) r504 -= 504u;
            r247 += 256; while (r247 >= 247u) r247 -= 247u;
        }
    }
    if (cnt) atomicAdd(P.probes, cnt);
    if (ncall) atomicAdd(P.calls, ncall);
    if (skip) atomicAdd(P.gated, skip);
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
    // probe gate
    u64* d_g504 = nullptr;
    u64* d_g247 = nullptr;
    u16* d_p6504 = nullptr;
    u16* d_p6247 = nullptr;
    u64* d_calls = nullptr;
    u64* d_gated = nullptr;
    u64 c_m504 = 0, c_m247 = 0, inv42_247 = 0;
    bool gate_ready = false, gate_on = true;
    // triple-sum window gate (cls5)
    u64* d_t504 = nullptr;
    u64* d_t247 = nullptr;
    u64* d_tri_skipped = nullptr;
    u64* d_tri_total = nullptr;
    bool tri_ready = false, tri_on = true;
    bool defer_gate_load = true;    // cls5: skip gate_load on dead blocks (A/B: --no-defer-gate-load)
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

static void gpu_upload_gate(GpuCtx& g, const GateData& gd) {
    CU(cudaMalloc(&g.d_g504, sizeof(gd.g504)));
    CU(cudaMalloc(&g.d_g247, sizeof(gd.g247)));
    CU(cudaMalloc(&g.d_p6504, sizeof(gd.p6504)));
    CU(cudaMalloc(&g.d_p6247, sizeof(gd.p6247)));
    CU(cudaMemcpy(g.d_g504, gd.g504, sizeof(gd.g504), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(g.d_g247, gd.g247, sizeof(gd.g247), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(g.d_p6504, gd.p6504, sizeof(gd.p6504), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(g.d_p6247, gd.p6247, sizeof(gd.p6247), cudaMemcpyHostToDevice));
    CU(cudaMalloc(&g.d_calls, sizeof(u64)));
    CU(cudaMalloc(&g.d_gated, sizeof(u64)));
    const u64 m504 = 504ULL * (u64)M42;
    g.c_m504 = (u64)(((u128)1 << 64) % (u128)m504);
    g.c_m247 = (u64)(((u128)1 << 64) % (u128)247);
    g.inv42_247 = (u64)mod_inv(mod_pow6(42, 247), 247);
    g.gate_ready = true;
    fprintf(stderr, "[gpu] probe gate uploaded (mod 124,488; keeps 27*50/124488 = %.3f%% of probes)\n",
            100.0 * 27.0 * 50.0 / 124488.0);
}

static void gpu_upload_tri(GpuCtx& g, const TriData& td) {
    CU(cudaMalloc(&g.d_t504, sizeof(td.t504)));
    CU(cudaMalloc(&g.d_t247, sizeof(td.t247)));
    CU(cudaMemcpy(g.d_t504, td.t504, sizeof(td.t504), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(g.d_t247, td.t247, sizeof(td.t247), cudaMemcpyHostToDevice));
    CU(cudaMalloc(&g.d_tri_skipped, sizeof(u64)));
    CU(cudaMalloc(&g.d_tri_total, sizeof(u64)));
    g.tri_ready = true;
    // Combined keep-rate = product of per-modulus densities (CRT-independent).
    const double keep = (td.n504 / 504.0) * (td.n247 / 247.0);
    fprintf(stderr, "[gpu] triple-sum window gate uploaded (cls5): "
            "mod504=%d/504=%.1f%% mod247=%d/247=%.1f%% -> keeps ~%.2f%% of blocks\n",
            td.n504, 100.0 * td.n504 / 504.0, td.n247, 100.0 * td.n247 / 247.0, 100.0 * keep);
}

#endif // HOST_ONLY

// ----------------------------------------------------- candidate generation --
// (host; used by campaign, xcheck, and the HOST_ONLY selftest)
static void gen_class_cands(const RootTables& rt, long long B, int cls, double u_lo, double u_hi,
                            std::vector<GpuCand>& out) {
    const long long lo = std::max(1LL, (long long)(u_lo * B));
    const long long hi = std::min(B - 1, (long long)(u_hi * B));
    const i128 B6 = ipow6(B);
    const u32 lim = (u32)((B - 1) / 42);

    // lift residue classes (mod m) to actual values <= maxv
    const auto lift = [](const std::vector<long long>& res, long long m, long long maxv) {
        std::vector<long long> v;
        for (long long r : res) {
            long long x = r % m;
            if (x == 0) x = m;
            for (; x <= maxv; x += m) v.push_back(x);
        }
        std::sort(v.begin(), v.end());
        return v;
    };
    // find4 c4-window (c4 = largest of the four seeds)
    const auto set_window = [&](GpuCand& C, i128 Q) -> bool {
        const long long hi4 = std::min(iroot6_i128(Q), (long long)C.lim);
        long long lo4 = (long long)(hi4 * 0.7937005260);   // 4^(-1/6)
        if (lo4 < 1) lo4 = 1;
        while (4 * ipow6(lo4) < Q) ++lo4;
        while (lo4 > 1 && 4 * ipow6(lo4 - 1) >= Q) --lo4;
        if (lo4 > hi4) return false;
        C.c4lo = (u32)lo4; C.c4hi = (u32)hi4;
        return true;
    };

    if (cls <= 4) {
        // actual values of the class anchor term u, in the [lo,hi] band
        std::vector<long long> uvals;
        if (cls == 4) {
            // u = 7x: x runs over the 24 classes mod 46656 with (7x)^6≡B^6 (2^6,3^6).
            // (u odd, ~3 from x; 7|u automatic.) seeds616(4) returns x-classes.
            for (long long x : lift(seeds616(rt, B, 4), M2 * M3, (B - 1) / 7)) {
                const long long u = 7 * x;
                if (u >= lo && u <= hi) uvals.push_back(u);
            }
        } else {
            const std::vector<long long> seeds = seeds616(rt, B, cls);
            const long long M = mod616(cls);
            for (long long s : seeds) {
                long long u0 = s % M;
                if (u0 < lo) u0 += (lo - u0 + M - 1) / M * M;
                for (long long u = u0; u <= hi; u += M)
                    if (unit_ok616(u, cls)) uvals.push_back(u);
            }
        }
        std::sort(uvals.begin(), uvals.end());
        uvals.erase(std::unique(uvals.begin(), uvals.end()), uvals.end());
        for (long long u : uvals) {
            const i128 u6 = ipow6(u);
            if (u6 >= B6) continue;
            {
                if (cls == 1) {
                    // {o37}: Q0 = (B^6-u^6)/42^6; the sixth term 42w gets the w-loop
                    const i128 R = B6 - u6;
                    if (R % M42 != 0) continue;              // guaranteed by seeds
                    const i128 Q0 = R / M42;
                    const long long wmax = std::min((long long)lim, iroot6_i128(Q0 - 4));
                    for (long long w = 1; w <= wmax; ++w) {
                        const i128 Q = Q0 - ipow6(w);
                        if (Q < 4) break;
                        GpuCand C{};
                        C.B = (u32)B; C.u = (u32)u; C.cls = 1; C.lim = lim; C.w = (u32)(42 * w);
                        C.q_lo = (u64)Q; C.q_hi = (u64)((u128)Q >> 64);
                        if (set_window(C, Q)) out.push_back(C);
                    }
                } else {
                    long long f; std::vector<long long> vres; long long vmod, vmax;
                    if (cls == 2) {      // {o7}|{3}: v=14v', (14v')^6≡B^6 (3^6)
                        f = 14; vres = classes_d_cls2(rt, B); vmod = M3; vmax = (B - 1) / 14;
                    } else if (cls == 3) { // {37}|{o}: v=21v', (21v')^6≡B^6-u^6 (2^6)
                        f = 21; vres = classes_d_cls3(rt, B, u); vmod = M2; vmax = (B - 1) / 21;
                    } else {               // {o3}|{7}: v=6v', (6v')^6≡B^6 (7^6)
                        f = 6; vres = scaled_roots(rt.r7, mod_pow6(B, M7), 6, M7); vmod = M7; vmax = (B - 1) / 6;
                    }
                    for (long long v : lift(vres, vmod, vmax)) {
                        const i128 R = B6 - u6 - ipow6(f * v);
                        if (R <= 0 || R % M42 != 0) continue;   // guaranteed by construction
                        const i128 Q = R / M42;
                        if (Q < 4) continue;
                        GpuCand C{};
                        C.B = (u32)B; C.u = (u32)u; C.cls = (u32)cls; C.lim = lim; C.w = (u32)(f * v);
                        C.q_lo = (u64)Q; C.q_hi = (u64)((u128)Q >> 64);
                        C.q504 = (u32)(long long)(Q % 504); C.q247 = (u32)(long long)(Q % 247);
                        if (set_window(C, Q)) out.push_back(C);
                    }
                }
            }
        }
    } else {
        // cls5 {o}|{3}|{7}: u=14u' (~3), w=6v' (~7); kernel grids w'=21w' (odd, mod 64)
        const std::vector<long long> wres = scaled_roots(rt.r2, mod_pow6(B, M2), 21, M2);
        const std::vector<long long> ures = classes_d_cls2(rt, B);                       // (14u')^6≡B^6 (3^6)
        const std::vector<long long> vres = scaled_roots(rt.r7, mod_pow6(B, M7), 6, M7); // (6v')^6≡B^6 (7^6)
        for (long long up : lift(ures, M3, (B - 1) / 14)) {
            const long long u = 14 * up;
            if (u < lo || u > hi) continue;                  // band applies to the u-term
            const i128 u6 = ipow6(u);
            for (long long vp : lift(vres, M7, (B - 1) / 6)) {
                const i128 base = B6 - u6 - ipow6(6 * vp);
                if (base <= 0) continue;
                GpuCand C{};
                C.B = (u32)B; C.u = (u32)u; C.cls = 5; C.lim = lim; C.w = (u32)(6 * vp);
                C.q_lo = (u64)base; C.q_hi = (u64)((u128)base >> 64);
                C.mod1 = M2; C.fmax1 = (u32)((B - 1) / 21);
                C.nres1 = (u32)wres.size();
                for (size_t i = 0; i < wres.size() && i < 24; ++i) C.res[i] = (u32)wres[i];
                out.push_back(C);
            }
        }
    }
}

#ifndef HOST_ONLY

// ------------------------------------------------------------ GPU run/drain --
struct RunResult {
    std::vector<Hit> hits;
    u32 raw_count = 0;
    u64 probes = 0;
    u64 calls = 0;
    u64 gated = 0;
    u64 tri_skipped = 0;
    u64 tri_total = 0;
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
    CU(cudaMemset(g.d_calls, 0, sizeof(u64)));
    CU(cudaMemset(g.d_gated, 0, sizeof(u64)));
    if (g.tri_ready) {
        CU(cudaMemset(g.d_tri_skipped, 0, sizeof(u64)));
        CU(cudaMemset(g.d_tri_total, 0, sizeof(u64)));
    }

    Params P{};
    P.tab = g.d_tab; P.mask = g.tab_slots - 1; P.S = g.S;
    P.hits = g.d_hits; P.hitc = g.d_hitc; P.cap = g.hit_cap; P.overflow = g.d_overflow;
    P.probes = g.d_probes;
    P.cands = g.d_cands; P.n_cand = (u32)v.size();
    P.inv216 = inv216;
    P.factor = (cls == 5) ? 21 : 0;
    P.use_gate = (g.gate_ready && g.gate_on) ? 1u : 0u;
    P.g504 = g.d_g504; P.g247 = g.d_g247; P.p6504 = g.d_p6504; P.p6247 = g.d_p6247;
    P.calls = g.d_calls; P.gated = g.d_gated;
    P.c_m504 = g.c_m504; P.c_m247 = g.c_m247; P.inv42_247 = g.inv42_247;
    // triple-sum window gate (cls5 only)
    P.use_tri_gate = (cls == 5 && g.tri_ready && g.tri_on) ? 1u : 0u;
    P.t504 = g.d_t504; P.t247 = g.d_t247;
    P.tri_skipped = g.d_tri_skipped; P.tri_total = g.d_tri_total;
    P.defer_gate_load = (cls == 5 && g.defer_gate_load) ? 1u : 0u;

    u32 ymax = 1;
    if (cls <= 4) ymax = 32;                       // 2048-c4 chunks (window <= ~11k at N=65535)
    else {
        for (const auto& C : v) {
            const u32 f = C.nres1 * (C.fmax1 / C.mod1 + 1);
            if (f > ymax) ymax = f;
        }
    }
    const auto t0 = Clock::now();
    if (cls <= 4) k_cls1<<<dim3((u32)v.size(), ymax), 256>>>(P);
    else          k_cls234<<<dim3((u32)v.size(), ymax), 256>>>(P);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
    R.kernel_ms = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count() / 1e3;

    u32 hc = 0, of = 0;
    CU(cudaMemcpy(&hc, g.d_hitc, sizeof(u32), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&of, g.d_overflow, sizeof(u32), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&R.probes, g.d_probes, sizeof(u64), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&R.calls, g.d_calls, sizeof(u64), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(&R.gated, g.d_gated, sizeof(u64), cudaMemcpyDeviceToHost));
    if (P.use_tri_gate) {
        CU(cudaMemcpy(&R.tri_skipped, g.d_tri_skipped, sizeof(u64), cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(&R.tri_total, g.d_tri_total, sizeof(u64), cudaMemcpyDeviceToHost));
    }
    R.raw_count = hc;
    R.overflow = (of != 0);
    const u32 kept = std::min(hc, g.hit_cap);
    R.hits.resize(kept);
    if (kept) CU(cudaMemcpy(R.hits.data(), g.d_hits, kept * sizeof(Hit), cudaMemcpyDeviceToHost));
    if (R.overflow)
        fprintf(stderr, "[gpu] WARNING: hit buffer overflow (%u hits > cap %u) — raise --hit-cap\n", hc, g.hit_cap);
    return R;
}

#endif // HOST_ONLY

// -------------------------------------------------- 21^6 inverse mod 2^64 --
// Newton iteration: x <- x(2 - a x) doubles correct bits; start x = a (mod 8).
static u64 inv216_mod_2_64() {
    u64 a = (u64)M21, x = a;
    for (int i = 0; i < 6; ++i) x = x * (2 - a * x);   // wrapping u64 arithmetic
    return x;
}

// ------------------------------------------------------------- verification --
// Returns 0 = fingerprint false positive, 1 = exact decomposition but tuple
// rejected (non-distinct / out of range), 2 = SOLUTION.
static int verify_hit(const GpuCand& C, const Hit& H, long& solutions,
                      std::set<std::array<long long, 6>>& reported) {
    const int cls = (int)C.cls;
    const u128 a6 = (u128)ipow6(H.a), b6 = (u128)ipow6(H.b), c6 = (u128)ipow6(H.c);
    const u128 q = ((u128)C.q_hi << 64) | C.q_lo;
    std::array<long long, 6> t;
    bool exact = false;
    if (cls <= 4) {
        // q = Q: the find4 target, divided exactly on the host.
        exact = (a6 + b6 + c6 + (u128)ipow6(H.d) == q);
        t = {42LL * H.a, 42LL * H.b, 42LL * H.c, 42LL * H.d, (long long)C.w, (long long)C.u};
    } else {
        // q = base = B^6-(14u')^6-(6v')^6; H.c = c3, H.d = w' (grid term 21w').
        const u128 g6 = (u128)ipow6(21LL * H.d);
        if (g6 <= q) {
            const u128 rhs = q - g6;
            exact = (rhs % (u128)M42 == 0) && (a6 + b6 + c6 == rhs / (u128)M42);
        }
        t = {42LL * H.a, 42LL * H.b, 42LL * H.c, 21LL * H.d, (long long)C.u, (long long)C.w};
    }
    if (!exact) return 0;
    std::sort(t.begin(), t.end());
    for (int i = 0; i < 5; ++i) if (t[i] == t[i + 1]) return 1;
    if (t.back() >= (long long)C.B || t.front() < 1) return 1;
    i128 lhs = ipow6((long long)C.B);       // subtractive: bail at first
    for (long long x : t) {                 // negative, so |lhs| < B^6 always
        lhs -= ipow6(x);
        if (lhs < 0) return 1;
    }
    if (lhs != 0) return 1;
    if (reported.insert(t).second) {
        long long g = std::gcd(t[0], t[1]);
        for (int i = 2; i < 6; ++i) g = std::gcd(g, t[i]);
        g = std::gcd(g, (long long)C.B);
        printf("SOLUTION cls=%d B=%u a1=%lld a2=%lld a3=%lld a4=%lld a5=%lld a6=%lld (anchor=%u) gcd=%lld %s\n",
               cls, C.B, t[0], t[1], t[2], t[3], t[4], t[5], C.u, g, g == 1 ? "primitive" : "NON-PRIMITIVE");
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
    fprintf(stderr, "[selftest] host math (616)\n");
    // (a) every seed class satisfies its defining congruences
    srand(616);
    for (int trial = 0; trial < 200; ++trial) {
        long long B = 1000 + rand() % 2000000;
        while (B % 2 == 0 || B % 3 == 0 || B % 7 == 0) ++B;
        const long long b2 = mod_pow6(B, M2), b3 = mod_pow6(B, M3), b7 = mod_pow6(B, M7);
        for (long long s : seeds616(rt, B, 1))
            if (mod_pow6(s, M2) != b2 || mod_pow6(s, M3) != b3 || mod_pow6(s, M7) != b7
                || !(s & 1) || std::gcd(s, 42LL) != 1) { fprintf(stderr, "seed fail cls=1 B=%lld\n", B); return false; }
        for (long long s : seeds616(rt, B, 2))
            if (mod_pow6(s, M2) != b2 || mod_pow6(s, M7) != b7 || !(s & 1)) { fprintf(stderr, "seed fail cls=2 B=%lld\n", B); return false; }
        for (long long s : seeds616(rt, B, 3))
            if (mod_pow6(s, M3) != b3 || mod_pow6(s, M7) != b7 || std::gcd(s, 21LL) != 1) { fprintf(stderr, "seed fail cls=3 B=%lld\n", B); return false; }
        for (long long x : seeds616(rt, B, 4))   // x-classes of the u=7x anchor
            if (mod_pow6(7 * x, M2) != b2 || mod_pow6(7 * x, M3) != b3) { fprintf(stderr, "seed fail cls=4 B=%lld\n", B); return false; }
        for (long long s : classes_d_cls2(rt, B))
            if (mod_pow6(14 * s, M3) != b3) { fprintf(stderr, "seed fail cls=5u B=%lld\n", B); return false; }
        for (long long s : scaled_roots(rt.r7, b7, 6, M7))
            if (mod_pow6(6 * s, M7) != b7) { fprintf(stderr, "seed fail cls=5v B=%lld\n", B); return false; }
        for (long long s : scaled_roots(rt.r2, b2, 21, M2))
            if (mod_pow6(21 * s, M2) != b2) { fprintf(stderr, "seed fail cls=5w B=%lld\n", B); return false; }
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
    // (g) probe gate: bitmaps must be exactly the pair-sum sets mod 504/247,
    //     of sizes 27 (3 mod 8 x 3 mod 9 x 3 mod 7) and 50 (5 mod 13 x 10 mod
    //     19), and must pass EVERY real pair sum (soundness fuzz).
    {
        GateData gd;
        build_gate(gd);
        int c504 = 0, c247 = 0;
        for (int x = 0; x < 504; ++x) c504 += gate_get(gd.g504, x);
        for (int x = 0; x < 247; ++x) c247 += gate_get(gd.g247, x);
        if (c504 != 27 || c247 != 50) {
            fprintf(stderr, "gate bitmap sizes %d/%d, want 27/50\n", c504, c247);
            return false;
        }
        for (int t = 0; t < 200000; ++t) {
            const u32 i = 1 + rand() % 65535, j = 1 + rand() % 65535;
            const u128 s = h_pow6_128(i) + h_pow6_128(j);   // exact (not wrapped!)
            if (!gate_get(gd.g504, (int)(long long)(s % 504)) || !gate_get(gd.g247, (int)(long long)(s % 247))) {
                fprintf(stderr, "gate soundness fail i=%u j=%u\n", i, j);
                return false;
            }
        }
        fprintf(stderr, "[selftest] probe gate: 27+50 residues, keep=%.3f%%, soundness fuzz OK\n",
                100.0 * 27.0 * 50.0 / 124488.0);
    }
    // (h) triple-sum window gate (cls5): bitmaps must be exactly the three-sum
    //     sets, mod 504 has 64 residues (4 mod 8 x 4 mod 9 x 4 mod 7), and EVERY
    //     real triple a^6+b^6+c^6 must pass both bitmaps (zero false negatives).
    {
        TriData td;
        build_tri_gate(td);
        if (td.n504 != 64) {
            fprintf(stderr, "tri gate mod504 size %d, want 64\n", td.n504);
            return false;
        }
        if (td.n247 <= 0 || td.n247 >= 247) {
            fprintf(stderr, "tri gate mod247 degenerate: %d\n", td.n247);
            return false;
        }
        const auto tget = [](const u64* b, int x) { return (b[x >> 6] >> (x & 63)) & 1ULL; };
        for (int t = 0; t < 200000; ++t) {
            const u32 a = 1 + rand() % 65535, b = 1 + rand() % 65535, c = 1 + rand() % 65535;
            const u128 s = h_pow6_128(a) + h_pow6_128(b) + h_pow6_128(c);   // exact
            if (!tget(td.t504, (int)(long long)(s % 504)) || !tget(td.t247, (int)(long long)(s % 247))) {
                fprintf(stderr, "tri gate soundness fail a=%u b=%u c=%u\n", a, b, c);
                return false;
            }
        }
        const double keep = (td.n504 / 504.0) * (td.n247 / 247.0);
        fprintf(stderr, "[selftest] triple-sum gate: mod504=%d mod247=%d, keep~%.2f%%, soundness fuzz OK\n",
                td.n504, td.n247, 100.0 * keep);
    }
    // (f) candidate-generation identities — the machine-checked proof of the
    //     new class math: EVERY emitted candidate must satisfy its defining
    //     identity exactly (u128):
    //       cls1-4: u^6 + w^6 + 42^6 * Q == B^6   (Q = find4 target)
    //       cls5:   u^6 + w^6 + base   == B^6   (base = find3 base)
    //     plus every class must actually produce candidates somewhere.
    long long tot[6] = {0};
    const auto check_ids = [&](long long B) -> bool {
        const u128 B6 = (u128)ipow6(B);
        for (int cls = 1; cls <= 5; ++cls) {
            std::vector<GpuCand> v;
            gen_class_cands(rt, B, cls, 0.0, 1.0, v);
            long long checked = 0;
            for (const GpuCand& C : v) {
                const u128 q = ((u128)C.q_hi << 64) | C.q_lo;
                const bool ok = (cls <= 4)
                    ? ((u128)ipow6(C.u) + (u128)ipow6(C.w) + (u128)M42 * q == B6)
                    : ((u128)ipow6(C.u) + (u128)ipow6(C.w) + q == B6);
                if (!ok) {
                    fprintf(stderr, "gen identity fail cls=%d B=%lld u=%u w=%u\n", cls, B, C.u, C.w);
                    return false;
                }
                if (++checked >= 300) break;      // cap per (B, cls); totals use v.size()
            }
            tot[cls] += (long long)v.size();
        }
        return true;
    };
    // phase 1: small B (dense coverage of the small-modulus classes 4 and 5)
    for (long long B = 1009; B <= 62000; B += 260) {
        if (!(B & 1) || B % 3 == 0 || B % 7 == 0) continue;
        if (!check_ids(B)) return false;
    }
    if (tot[4] == 0 || tot[5] == 0) { fprintf(stderr, "gen smoke: no cls4/cls5 candidates at small B\n"); return false; }
    // phase 2: large B near the i128 ceiling, so the big-modulus classes
    // (cls2 mod 14^6, cls3 mod 21^6) materialize anchors.
    srand(4242);
    for (int t = 0; t < 60; ++t) {
        long long B = 2250001 + rand() % 100000;
        while (B % 2 == 0 || B % 3 == 0 || B % 7 == 0) ++B;
        if (B > B_HARD_MAX) continue;
        if (!check_ids(B)) return false;
    }
    if (tot[2] == 0 || tot[3] == 0) { fprintf(stderr, "gen smoke: no cls2/cls3 candidates at large B\n"); return false; }
    // cls1 anchors are the rarest (144 classes mod 42^6 ~ 5.5e9): scan upward
    // deterministically until a B with a cls1 candidate turns up, then check it.
    for (long long B = 2250001; tot[1] == 0 && B <= B_HARD_MAX - 400; B += 2) {
        if (!(B & 1) || B % 3 == 0 || B % 7 == 0) continue;
        std::vector<GpuCand> v;
        gen_class_cands(rt, B, 1, 0.0, 1.0, v);
        if (!v.empty() && !check_ids(B)) return false;
    }
    if (tot[1] == 0) { fprintf(stderr, "gen smoke: no cls1 candidate found in scan\n"); return false; }
    fprintf(stderr, "[selftest] gen smoke candidates: c1=%lld c2=%lld c3=%lld c4=%lld c5=%lld\n",
            tot[1], tot[2], tot[3], tot[4], tot[5]);
    fprintf(stderr, "[selftest] host math OK\n");
    return true;
}

#ifndef HOST_ONLY

// ------------------------------------------------------- GPU plant tests --
// Build a small table, plant known targets, require the kernels to find them.
static bool selftest_gpu(GpuCtx& g, u64 inv216) {
    fprintf(stderr, "[selftest] GPU plant tests (small table N=4096)\n");
    const int N = 4096, S = 24;   // LF ~0.5: S must satisfy 2^S*0.95 >= N(N+1)/2
    std::vector<Slot> slots;
    table_build(N, S, slots);
    gpu_upload_table(g, slots, S);
    GateData gd;
    build_gate(gd);
    gpu_upload_gate(g, gd);   // plants double as the gate soundness test

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
            if (cls <= 4) {
                ex = (a6 + b6 + h_pow6_128(H.c) + h_pow6_128(H.d) == targets[H.cand]);
            } else {
                const u128 base = ((u128)C.q_hi << 64) | C.q_lo;
                ex = ((u128)M42 * (a6 + b6 + h_pow6_128(H.c)) + h_pow6_128(21 * H.d) == base);
            }
            if (ex) got[H.cand] = 1;
        }
        int cnt = 0;
        for (char c : got) cnt += c;
        return cnt;
    };

    srand(777);
    // -- find4 plant (cls1-4 shape): Q = sum of 4 distinct sixth powers --
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
            C.cls = 1; C.B = 42 * N + 1; C.u = 1; C.w = 0; C.lim = N;
            C.q_lo = (u64)Q; C.q_hi = (u64)((u128)Q >> 64);
            C.q504 = (u32)(long long)(Q % 504); C.q247 = (u32)(long long)(Q % 247);
            const long long hi4 = std::min(iroot6_i128(Q), (long long)N);
            long long lo4 = (long long)(hi4 * 0.7937005260); if (lo4 < 1) lo4 = 1;
            while (4 * ipow6(lo4) < Q) ++lo4;
            while (lo4 > 1 && 4 * ipow6(lo4 - 1) >= Q) --lo4;
            C.c4lo = (u32)lo4; C.c4hi = (u32)hi4;
            cands[t] = C; targets[t] = (u128)Q;
        }
        const int got = run_and_count_exact(1, cands, targets);
        fprintf(stderr, "[plant] find4 (cls1 shape): %d/%d\n", got, T);
        if (got != T) return false;
    }
    // -- find3 plant (cls5 shape): base = (21w)^6 + 42^6(i^6+j^6+k^6) --
    {
        const int T = 512;
        std::vector<GpuCand> cands(T);
        std::vector<u128> targets(T);
        for (int t = 0; t < T; ++t) {
            const int d = 1 + rand() % 280;
            int cs[3];
            do { for (int k = 0; k < 3; ++k) cs[k] = 1 + rand() % N; }
            while (cs[0] == cs[1] || cs[0] == cs[2] || cs[1] == cs[2]);
            const u128 base = (u128)ipow6(21LL * d) + (u128)M42 * ((u128)ipow6(cs[0]) + ipow6(cs[1]) + ipow6(cs[2]));
            GpuCand C{};
            C.cls = 5; C.B = 42 * N + 1; C.u = 14; C.w = 6; C.lim = N;
            C.q_lo = (u64)base; C.q_hi = (u64)(base >> 64);
            C.mod1 = M2; C.fmax1 = (u32)d; C.nres1 = 1; C.res[0] = (u32)(d % M2);
            cands[t] = C; targets[t] = base;
        }
        const int got = run_and_count_exact(5, cands, targets);
        fprintf(stderr, "[plant] find3 (cls5 shape): %d/%d\n", got, T);
        if (got != T) return false;
    }
    fprintf(stderr, "[selftest] GPU plant tests OK\n");
    return true;
}

#endif // HOST_ONLY

// =============================================================================
// XCHECK — CPU reference finders vs GPU kernels, same candidates.
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
    if (C.cls <= 4) return xc_find4(ix, q, lim).has_value();
    // cls5: q = base; replay the kernel's w' grid exactly.
    const u32 k1 = C.fmax1 / C.mod1 + 1;
    for (u32 t1 = 0; t1 < C.nres1 * k1; ++t1) {
        const u32 w = C.res[t1 % C.nres1] + (t1 / C.nres1) * C.mod1;
        if (!w || w > C.fmax1) continue;
        const i128 R = q - ipow6(21LL * w);
        if (R <= 0 || R % M42 != 0) continue;
        if (xc_find3(ix, R / M42, lim)) return true;
    }
    return false;
}
#endif

// =============================================================================
// MAIN
// =============================================================================
#ifdef HOST_ONLY
int main() {
    const RootTables rt;
    const u64 inv216 = inv216_mod_2_64();
    if (!selftest_host(rt, inv216)) { fprintf(stderr, "SELFTEST FAIL (host)\n"); return 1; }
    fprintf(stderr, "SELFTEST PASS (host-only build; run the nvcc build on the server for GPU tests)\n");
    return 0;
}
#else
int main(int argc, char** argv) {
    // ---- argument parsing ----
    std::vector<std::string> pos;
    long long chunk = 8192, bench_chunks = 0;
    int device = 0, slots_log2 = 0;
    u32 hit_cap = 1u << 20;
    std::string save_table, load_table;
    bool opt_selftest = false, opt_xcheck = false, quiet = false, opt_nogate = false, opt_notri = false;
    bool opt_no_defer_gate = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") opt_selftest = true;
        else if (a == "--xcheck") opt_xcheck = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--no-gate") opt_nogate = true;
        else if (a == "--no-tri-gate") opt_notri = true;
        else if (a == "--no-defer-gate-load") opt_no_defer_gate = true;
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
        // bare "--bench [K]": time K chunks over a representative night-1 slice
        pos.push_back("110267");
        pos.push_back("410000");
    }
    if (pos.size() < 2) {
        fprintf(stderr,
            "usage: %s <B_min> <B_max> [classes] [u_lo] [u_hi] [options]\n"
            "  (6,1,6): a1^6+..+a6^6 = B^6, primitive. New ground starts at 110267\n"
            "  (EulerNet covered <= 110266 by Jan 2000).\n"
            "  classes: \"all\" (default) or e.g. \"1,3\";  u band as fractions of B\n"
            "  options: --chunk K --device K --slots-log2 S --hit-cap N\n"
            "           --save-table F --load-table F --bench [K] --xcheck --quiet\n"
            "           --no-gate (disable the mod-124,488 probe gate, for A/B bench)\n"
            "           --no-tri-gate (disable the cls5 triple-sum window gate)\n"
            "           --no-defer-gate-load (cls5: gate_load before skip; A/B baseline)\n"
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
    g.gate_on = !opt_nogate;
    g.tri_on = !opt_notri;
    g.defer_gate_load = !opt_no_defer_gate;
    if (!quiet)
        fprintf(stderr, "[gpu] cls5 defer gate_load: %s\n",
                g.defer_gate_load ? "on (skip gate_load on dead blocks)" : "off (A/B baseline)");
    {
        GateData gd;
        build_gate(gd);
        gpu_upload_gate(g, gd);
        TriData td;
        build_tri_gate(td);
        gpu_upload_tri(g, td);
    }

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
        std::set<std::array<long long, 6>> reported;
        for (long long B = std::max(B_min, 11LL); B <= B_max; ++B) {
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
    fprintf(stderr, "solve_616_v1: B in [%lld, %lld], classes ", B_min, B_max);
    for (int c : classes) fprintf(stderr, "%d", c);
    fprintf(stderr, ", unit band [%.4f, %.4f)*B, chunk=%lld\n", u_lo, u_hi, chunk);

    long solutions = 0;
    long long stat_cands[6] = {0}, stat_exact[6] = {0}, stat_false[6] = {0};
    u64 stat_probes[6] = {0}, stat_calls[6] = {0}, stat_gated[6] = {0};
    u64 stat_tri_skip[6] = {0}, stat_tri_tot[6] = {0};
    double stat_ms[6] = {0};
    std::set<std::array<long long, 6>> reported;

    long long chunks_done = 0;
    for (long long c0 = B_min; c0 <= B_max; c0 += chunk) {
        const long long c1 = std::min(B_max, c0 + chunk - 1);
        std::vector<long long> Bl;
        for (long long B = std::max(c0, 11LL); B <= c1; ++B)
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
            stat_calls[cls] += R.calls;
            stat_gated[cls] += R.gated;
            stat_tri_skip[cls] += R.tri_skipped;
            stat_tri_tot[cls] += R.tri_total;
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
        if (stat_cands[cls]) {
            fprintf(stderr, "cls%d: cands=%lld probes=%.4e kernel=%.1fs rate=%.2e probes/s exact=%lld fp-false=%lld gate-kill=%.2f%%",
                    cls, stat_cands[cls], (double)stat_probes[cls], stat_ms[cls] / 1e3,
                    stat_probes[cls] / std::max(1e-9, stat_ms[cls] / 1e3), stat_exact[cls], stat_false[cls],
                    100.0 * (double)stat_gated[cls] / std::max(1.0, (double)(stat_gated[cls] + stat_calls[cls])));
            if (stat_tri_tot[cls])
                fprintf(stderr, " tri-skip=%.2f%% (%llu/%llu blocks)",
                        100.0 * (double)stat_tri_skip[cls] / (double)stat_tri_tot[cls],
                        (unsigned long long)stat_tri_skip[cls], (unsigned long long)stat_tri_tot[cls]);
            fprintf(stderr, "\n");
        }
    fprintf(stderr, "total solutions: %ld\n", solutions);
    return 0;
}
#endif // HOST_ONLY
