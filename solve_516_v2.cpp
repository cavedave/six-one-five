// =============================================================================
// solve_516_v2.cpp — five-class (6,1,5) solver
//     a1^6 + a2^6 + a3^6 + a4^6 + a5^6 = B^6,  1<=a1<...<a5<B, gcd=1
// =============================================================================
//
// Successor to solve_516.cpp. Changes (see 615-search-code-reading-notes.md):
//   * ALL FIVE Meyrignac congruence classes are searched, not only the
//     "unit coprime to 21" slice. Database evidence (178 known (6,1,7)
//     solutions): the distinguished unit is coprime to 21 in only ~15% of
//     solutions; ~85% have the unit divisible by 3 (classes 2/4/5).
//   * UNIT IS NOT ASSUMED TO BE THE LARGEST TERM. In the (6,1,7) database the
//     unit's rank among the seven terms is exactly uniform (11,12,10,14,10,11,
//     10 for ranks 1..7) and unit/z goes down to 0.019. Default unit range is
//     the FULL [1, B); the optional band is for targeted runs only.
//   * Class 1 uses the full 42^6 master congruence (parity law included):
//     144 seed classes, 16x fewer candidates than the 21^6 scheme, and the
//     pair table is built at N = B/42 (4x smaller than at B/21).
//   * Windowed finders: c_max is confined to [(Q/t)^{1/6}, Q^{1/6}] for t-term
//     sums (~6x cut for the 3-sum classes, ~15x for the 4-sum vs the original
//     square loop), with integer 6th-root bounds (no floating point on i128).
//   * COMPLETENESS FIX ("bad_order"): finders return the first ALL-DISTINCT
//     decomposition and skip non-distinct ones, instead of dropping the
//     candidate when the first-found quad has a repeated value.
//   * i128 overflow guard: B_max <= 2,200,000 (B^6 < 2^127).
//   * --selftest: verifies root tables, seed congruences, finders, iroot6,
//     and per-class 42^6-divisibility on synthetic inputs.
//
// THE FIVE CLASSES (distribution of the three distinguished indices: unique
// odd term, unique 3-free term, unique 7-free term; B odd, gcd(B,21)=1 always).
// u = unit (unique 7-free term). Master congruences from the valuation laws
// v2(B^6-u^6)=v2(B-u)+v2(B+u), v3(...)=v3(B∓u)+1, v7(...)=v7(B-ζ_c u):
//
//   cls  structure of the five terms        master            free terms & filters
//   ---  ---------------------------------  ----------------  -----------------------------
//   1    42c1..42c4 + u, gcd(u,42)=1        B^6≡u^6 (42^6)    —
//   2    42c1..42c3 + 14d + u=3w            B^6≡u^6 (14^6)    (14d)^6≡B^6 (3^6)
//   3    42c1..42c3 + 21d(odd) + u even,    B^6≡u^6 (21^6)    (21d)^6≡B^6-u^6 (2^6)
//        gcd(u,21)=1
//   4    42c1..42c3 + 7d(odd,3∤d) + u=6w    B^6≡u^6 (7^6)     (7d)^6≡B^6 (2^6·3^6)
//   5    42c1..42c2 + 21d + 14e + u=6w      B^6≡u^6 (7^6)     (21d)^6≡B^6-u^6 (2^6),
//                                                           (14e)^6≡B^6 (3^6)
//
// After the master congruence and the free-term filters, the remaining term
// Q = (B^6 - u^6 - free...)/42^6 must be a sum of n_small sixth powers
// (4,3,3,3,2 for classes 1..5), answered by windowed finders over the shared
// meet-in-the-middle pair table (Bernstein's paradigm; see quad_sum.hpp).
//
// Filters note: 13/19/25/37 filters are PROVABLY VACUOUS at the (B,u) stage of
// the aligned reduction (every attainable Q mod p is a sum of four 6th-power
// residues) and are intentionally NOT used. The mod-60 (B,u) pre-filter is
// applied for class 1 only; mod-300 final check + 128-bit ground truth for all.
//
// Build:  g++ -O2 -std=c++20 -fopenmp -o solve_516_v2 solve_516_v2.cpp
// Run:    OMP_NUM_THREADS=24 ./solve_516_v2 <B_min> <B_max> [classes] [u_lo] [u_hi]
//           classes: "all" (default) or any of 1,2,3,4,5 e.g. "1,3"
//           u_lo/u_hi: optional unit band as fractions of B (default full [0,1))
//           ./solve_516_v2 --selftest
//
// References: LPS 1967; Resta-Meyrignac 2003 (B<=730000 excluded);
// Gerbicz-Meyrignac-Beckert arXiv:1108.0462; Newton-Rouse arXiv:2101.09390;
// Bernstein Math. Comp. 70 (2001) 389-394; euler.free.fr database;
// MSE q/392857 (five classes), q/4753175, q/4746879.

#include "k14_common.hpp"
#include "quad_sum.hpp"
#include "mod60.hpp"

#include <chrono>
#include <iomanip>
#include <optional>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using Clock = chrono::steady_clock;

// ---------------------------------------------------------------- constants --
static constexpr long long M2 = 64LL;         // 2^6
static constexpr long long M3 = 729LL;        // 3^6
static constexpr long long M7 = 117649LL;     // 7^6
static constexpr long long M14 = 7529536LL;   // 14^6 = 2^6·7^6
static constexpr long long M21 = 85766121LL;  // 21^6 = 3^6·7^6
static constexpr long long M42 = 5489031744LL;// 42^6 = 2^6·3^6·7^6
static constexpr int K = 6;
static constexpr long long B_HARD_MAX = 2200000;  // B^6 < 2^127 (i128 guard)

// ------------------------------------------------------- small modular tools --
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

// Integer floor 6th root of nonnegative i128 (binary search; exact).
static long long iroot6(i128 n) {
    if (n <= 0) return 0;
    long long lo = 0, hi = 1;
    while (ipow(hi, 6) <= n) hi <<= 1;
    while (hi - lo > 1) {
        const long long mid = (lo + hi) >> 1;
        if (ipow(mid, 6) <= n) lo = mid; else hi = mid;
    }
    return lo;
}

// ------------------------------------------------------------- root tables --
// For every sixth-power residue r among UNITS mod m, the units a with a^6≡r.
// Counts (verified): mod 2^6: 8 residues x 4 roots; mod 3^6: 81 x 6; mod 7^6:
// 16807 x 6 (Teichmuller). Only units are stored — every seed/unit we ever
// need is coprime to its modulus (classes 1-3) or the modulus is 7^6 with
// 7∤u (classes 4-5).
struct RootTables {
    vector<vector<long long>> r2, r3, r7;
    RootTables() : r2(M2), r3(M3), r7(M7) {
        for (long long a = 1; a < M2; ++a) if (gcd(a, M2) == 1) r2[mod_pow6(a, M2)].push_back(a);
        for (long long a = 1; a < M3; ++a) if (gcd(a, M3) == 1) r3[mod_pow6(a, M3)].push_back(a);
        for (long long a = 1; a < M7; ++a) if (gcd(a, M7) == 1) r7[mod_pow6(a, M7)].push_back(a);
    }
};

// Seed classes for u given B, merging root tables over the class's moduli.
static vector<long long> seeds_for_B(const RootTables& rt, long long B, int cls) {
    const long long b2 = mod_pow6(B, M2), b3 = mod_pow6(B, M3), b7 = mod_pow6(B, M7);
    vector<long long> out;
    if (cls == 1) {          // 4 x 6 x 6 = 144 classes mod 42^6
        for (long long s2 : rt.r2[b2])
            for (long long s3 : rt.r3[b3]) {
                const long long s23 = crt2(s2, M2, s3, M3);
                for (long long s7 : rt.r7[b7]) out.push_back(crt2(s23, M2 * M3, s7, M7));
            }
    } else if (cls == 2) {   // 4 x 6 = 24 classes mod 14^6
        for (long long s2 : rt.r2[b2])
            for (long long s7 : rt.r7[b7]) out.push_back(crt2(s2, M2, s7, M7));
    } else if (cls == 3) {   // 6 x 6 = 36 classes mod 21^6
        for (long long s3 : rt.r3[b3])
            for (long long s7 : rt.r7[b7]) out.push_back(crt2(s3, M3, s7, M7));
    } else {                 // classes 4,5: 6 classes mod 7^6
        out = rt.r7[b7];
    }
    sort(out.begin(), out.end());
    out.erase(unique(out.begin(), out.end()), out.end());
    return out;
}

static long long class_modulus(int cls) {
    switch (cls) {
        case 1: return M42;
        case 2: return M14;
        case 3: return M21;
        default: return M7;
    }
}

// Per-class admissibility of the unit (the 2- and 3-structure; 7-structure is
// automatic since seeds are units mod 7^6).
static bool unit_ok(long long u, int cls) {
    switch (cls) {
        case 1: return (u & 1) && std::gcd(u, 42LL) == 1;          // odd, coprime 42
        case 2: return (u & 1) && (u % 3 == 0) && (u % 9 != 0);    // u = 3w, 3∤w
        case 3: return !(u & 1) && std::gcd(u, 21LL) == 1;         // even, coprime 21
        case 4:
        case 5: return (u % 6 == 0) && (u % 7 != 0);               // u = 6w, 7∤u
    }
    return false;
}

// Lift seed classes to unit values in [lo, hi] (fractions of B), applying
// unit_ok. With M >= 7^6 this yields only a handful of candidates per B.
static vector<long long> unit_candidates(const RootTables& rt, long long B, int cls,
                                         double lo_frac, double hi_frac) {
    const long long M = class_modulus(cls);
    const long long lo = max(1LL, (long long)(lo_frac * B));
    const long long hi = min(B - 1, (long long)(hi_frac * B));
    vector<long long> out;
    for (long long s : seeds_for_B(rt, B, cls)) {
        long long u = s % M;
        if (u < lo) {
            const long long skip = (lo - u + M - 1) / M;
            u += skip * M;
        }
        for (; u <= hi; u += M)
            if (unit_ok(u, cls)) out.push_back(u);
    }
    sort(out.begin(), out.end());
    out.erase(unique(out.begin(), out.end()), out.end());
    return out;
}

// ------------------------------------------------- windowed sum-2/3/4 finders --
// All use the shared sorted pair table. "Distinct" means the returned indices
// are pairwise distinct (the bad_order completeness fix): a decomposition with
// a repeated value is skipped, not fatal.

// find2: c1^6 + c2^6 = Q, 1 <= c1 < c2 <= lim.
static optional<array<int, 2>> find2(const QuadSumIndex& ix, i128 Q, int lim) {
    if (Q <= 0) return nullopt;
    const auto [lo, hi] = ix.equal_sum_range(Q);
    for (size_t p = lo; p < hi; ++p) {
        const int i = ix.pairs[p].i, j = ix.pairs[p].j;
        if (i > lim || j > lim) continue;
        if (i < j) return array<int, 2>{i, j};
    }
    return nullopt;
}

// find3: c1^6 + c2^6 + c3^6 = Q, distinct, <= lim.
// Window: c3 in [ceil((Q/3)^(1/6)), floor(Q^(1/6))] — a ~6x cut vs full loop.
static optional<array<int, 3>> find3(const QuadSumIndex& ix, i128 Q, int lim) {
    if (Q <= 0) return nullopt;
    long long hi = iroot6(Q);
    long long lo = iroot6(Q / 3);
    while (3 * ipow(lo, 6) < Q) ++lo;                      // exact: 3*c3^6 >= Q
    if (hi > lim) hi = lim;
    for (long long c3 = hi; c3 >= lo; --c3) {
        const i128 rem = Q - ix.pw[c3];
        if (rem <= 0) continue;
        auto p = find2(ix, rem, (int)min((long long)lim, c3));
        if (!p) continue;
        int a = (*p)[0], b = (*p)[1];
        if (a != b && b != (int)c3 && a != (int)c3) return array<int, 3>{a, b, (int)c3};
    }
    return nullopt;
}

// find4: c1^6 + ... + c4^6 = Q, distinct, <= lim.
// Windows on c4 and (via find3) c3 — ~15x cut vs the original O(lim^2) square.
static optional<array<int, 4>> find4(const QuadSumIndex& ix, i128 Q, int lim) {
    if (Q <= 0) return nullopt;
    long long hi = iroot6(Q);
    long long lo = iroot6(Q / 4);
    while (4 * ipow(lo, 6) < Q) ++lo;                      // exact: 4*c4^6 >= Q
    if (hi > lim) hi = lim;
    for (long long c4 = hi; c4 >= lo; --c4) {
        const i128 rem = Q - ix.pw[c4];
        if (rem <= 0) continue;
        auto t = find3(ix, rem, (int)min((long long)lim, c4));
        if (!t) continue;
        int a = (*t)[0], b = (*t)[1], c = (*t)[2];
        if (a != (int)c4 && b != (int)c4 && c != (int)c4)
            return array<int, 4>{a, b, c, (int)c4};
    }
    return nullopt;
}

// ----------------------------------------------------- free-term class sets --
// Allowed classes for the distinguished free terms (d, e), from the valuation
// filters of the header table. All targets are units, so the filters also
// enforce 3∤d, 3∤e, d odd automatically.
static vector<long long> classes_d_cls2(const RootTables& rt, long long B) {
    const long long t = mod_pow6(B, M3) * mod_inv(mod_pow6(14, M3), M3) % M3;
    return rt.r3[t];                                    // 6 classes mod 729
}
static vector<long long> classes_d_cls3(const RootTables& rt, long long B, long long u) {
    long long rhs = (mod_pow6(B, M2) - mod_pow6(u, M2)) % M2; if (rhs < 0) rhs += M2;
    const long long t = rhs * mod_inv(mod_pow6(21, M2), M2) % M2;
    return rt.r2[t];                                    // 4 classes mod 64
}
static vector<long long> classes_d_cls4(const RootTables& rt, long long B) {
    const long long t2 = mod_pow6(B, M2) * mod_inv(mod_pow6(7, M2), M2) % M2;
    const long long t3 = mod_pow6(B, M3) * mod_inv(mod_pow6(7, M3), M3) % M3;
    vector<long long> out;
    for (long long a : rt.r2[t2]) for (long long b : rt.r3[t3]) out.push_back(crt2(a, M2, b, M3));
    return out;                                         // 24 classes mod 64·729 = 46656
}
static vector<long long> classes_e_cls5(const RootTables& rt, long long B) {
    return classes_d_cls2(rt, B);                       // (14e)^6 ≡ B^6 (3^6)
}
static vector<long long> classes_d_cls5(const RootTables& rt, long long B, long long u) {
    return classes_d_cls3(rt, B, u);                    // (21d)^6 ≡ B^6-u^6 (2^6)
}

// Values of a free term f*base in [1, fmax] lying in the given residue classes.
static vector<long long> free_values(long long fmax, long long mod, const vector<long long>& classes) {
    vector<long long> out;
    for (long long r : classes) {
        long long v = r % mod;
        if (v == 0) v = mod;
        for (; v <= fmax; v += mod) out.push_back(v);
    }
    sort(out.begin(), out.end());
    return out;
}

// ------------------------------------------------------------ final assembly --
static int gcd6ll(long long a, long long b, long long c, long long d, long long e, long long B) {
    long long g = std::gcd(a, b);
    g = std::gcd(g, c); g = std::gcd(g, d); g = std::gcd(g, e); g = std::gcd(g, B);
    return (int)g;
}

// Check the full tuple: five DISTINCT terms, all < B, exact 128-bit identity.
// Unit is NOT required to be the largest term (database: unit rank is uniform).
static bool check_tuple(long long B, array<long long, 5> t, int cls, long long u, long& solutions) {
    sort(t.begin(), t.end());
    for (int i = 0; i < 4; ++i) if (t[i] == t[i + 1]) return false;
    if (t.back() >= B || t.front() < 1) return false;
    static mod60::SixthPowerFilter300 f300;
    if (!f300.passes(B, {t[0], t[1], t[2], t[3], t[4]})) return false;
    i128 lhs = 0;
    for (long long x : t) lhs += ipow(x, K);
    if (lhs != ipow(B, K)) return false;
    const int g = gcd6ll(t[0], t[1], t[2], t[3], t[4], B);
#pragma omp critical(log_line)
    {
        cout << "SOLUTION cls=" << cls << " B=" << B
             << " a1=" << t[0] << " a2=" << t[1] << " a3=" << t[2]
             << " a4=" << t[3] << " a5=" << t[4] << " (unit=" << u << ")"
             << " gcd=" << g << (g == 1 ? " primitive" : " NON-PRIMITIVE") << "\n";
        cerr << B << "\tSOLUTION\tcls=" << cls << "\n";
    }
    ++solutions;
    return true;
}

// --------------------------------------------------------- per-class engines --
// Shared preliminaries per (B, u): diff = B^6 - u^6 must be divisible by the
// class's master modulus (automatic by seed construction; cheap assert left in
// for classes with free terms, where the 42^6 divisibility also involves the
// free-term filters).

static void run_class1(const QuadSumIndex& ix, long long B, long long u, long& solutions) {
    const i128 diff = ipow(B, K) - ipow(u, K);
    if (diff <= 0 || diff % M42 != 0) return;
    const i128 Q = diff / M42;
    const int lim = (int)((B - 1) / 42);
    auto q = find4(ix, Q, lim);
    if (!q) return;
    check_tuple(B, {42LL * (*q)[0], 42LL * (*q)[1], 42LL * (*q)[2], 42LL * (*q)[3], u}, 1, u, solutions);
}

static void run_class2(const RootTables& rt, const QuadSumIndex& ix, long long B, long long u, long& solutions) {
    const i128 base = ipow(B, K) - ipow(u, K);
    const int lim = (int)((B - 1) / 42);
    const auto dvals = free_values((B - 1) / 14, M3, classes_d_cls2(rt, B));
    for (long long d : dvals) {
        const i128 R = base - ipow(14 * d, K);
        if (R <= 0 || R % M42 != 0) continue;
        auto t = find3(ix, R / M42, lim);
        if (!t) continue;
        check_tuple(B, {42LL * (*t)[0], 42LL * (*t)[1], 42LL * (*t)[2], 14 * d, u}, 2, u, solutions);
    }
}

static void run_class3(const RootTables& rt, const QuadSumIndex& ix, long long B, long long u, long& solutions) {
    const i128 base = ipow(B, K) - ipow(u, K);
    const int lim = (int)((B - 1) / 42);
    const auto dvals = free_values((B - 1) / 21, M2, classes_d_cls3(rt, B, u));
    for (long long d : dvals) {
        const i128 R = base - ipow(21 * d, K);
        if (R <= 0 || R % M42 != 0) continue;
        auto t = find3(ix, R / M42, lim);
        if (!t) continue;
        check_tuple(B, {42LL * (*t)[0], 42LL * (*t)[1], 42LL * (*t)[2], 21 * d, u}, 3, u, solutions);
    }
}

static void run_class4(const RootTables& rt, const QuadSumIndex& ix, long long B, long long u, long& solutions) {
    const i128 base = ipow(B, K) - ipow(u, K);
    const int lim = (int)((B - 1) / 42);
    const auto dvals = free_values((B - 1) / 7, M2 * M3, classes_d_cls4(rt, B));
    for (long long d : dvals) {
        const i128 R = base - ipow(7 * d, K);
        if (R <= 0 || R % M42 != 0) continue;
        auto t = find3(ix, R / M42, lim);
        if (!t) continue;
        check_tuple(B, {42LL * (*t)[0], 42LL * (*t)[1], 42LL * (*t)[2], 7 * d, u}, 4, u, solutions);
    }
}

static void run_class5(const RootTables& rt, const QuadSumIndex& ix, long long B, long long u, long& solutions) {
    const i128 base = ipow(B, K) - ipow(u, K);
    const int lim = (int)((B - 1) / 42);
    const auto evals = free_values((B - 1) / 14, M3, classes_e_cls5(rt, B));
    const auto dcls = classes_d_cls5(rt, B, u);
    for (long long e : evals) {
        const i128 R1 = base - ipow(14 * e, K);
        if (R1 <= 0) continue;
        const auto dvals = free_values((B - 1) / 21, M2, dcls);
        for (long long d : dvals) {
            const i128 R = R1 - ipow(21 * d, K);
            if (R <= 0 || R % M42 != 0) continue;
            auto p = find2(ix, R / M42, lim);
            if (!p) continue;
            check_tuple(B, {42LL * (*p)[0], 42LL * (*p)[1], 21 * d, 14 * e, u}, 5, u, solutions);
        }
    }
}

// ------------------------------------------------------------------ selftest --
static bool selftest() {
    cerr << "[selftest] root tables + seeds + finders + iroot6 + class congruences\n";
    const RootTables rt;
    // iroot6: exact floor, incl. boundary cases (values kept below 10^6 so x^6 fits i128)
    for (long long x : {0LL, 1LL, 2LL, 63LL, 64LL, 65LL, 729LL, 999983LL}) {
        const i128 n = ipow(x > 0 ? x : 1, 6);
        const long long r = iroot6(n);
        if (ipow(r, 6) > n || ipow(r + 1, 6) <= n) { cerr << "iroot6 fail at x=" << x << "\n"; return false; }
    }
    for (long long n = 0; n < 5000; ++n) {
        const long long r = iroot6(n);
        if (ipow(r, 6) > n || ipow(r + 1, 6) <= n) { cerr << "iroot6 fail at n=" << n << "\n"; return false; }
    }
    // seeds satisfy the master congruence, for random B (odd, coprime 21)
    srand(615);
    for (int trial = 0; trial < 200; ++trial) {
        long long B = 1000 + rand() % 2000000;
        while (B % 2 == 0 || B % 3 == 0 || B % 7 == 0) ++B;
        for (int cls = 1; cls <= 5; ++cls) {
            for (long long s : seeds_for_B(rt, B, cls)) {
                bool ok = mod_pow6(s, M7) == mod_pow6(B, M7);
                if (cls == 1) ok = ok && mod_pow6(s, M2) == mod_pow6(B, M2) && mod_pow6(s, M3) == mod_pow6(B, M3);
                if (cls == 2) ok = ok && mod_pow6(s, M2) == mod_pow6(B, M2);
                if (cls == 3) ok = ok && mod_pow6(s, M3) == mod_pow6(B, M3);
                if (!ok) { cerr << "seed fail cls=" << cls << " B=" << B << "\n"; return false; }
            }
        }
    }
    // per-class 42^6 divisibility with synthetic free terms
    for (int trial = 0; trial < 50; ++trial) {
        long long B = 5000 + rand() % 100000;
        while (B % 2 == 0 || B % 3 == 0 || B % 7 == 0) ++B;
        for (long long d : classes_d_cls2(rt, B)) {
            if (((mod_pow6(B, M3) - mod_pow6(14 * d, M3)) % M3 + M3) % M3 != 0) { cerr << "cls2 d-filter fail\n"; return false; }
        }
        for (long long d : classes_d_cls4(rt, B)) {
            const bool ok2 = ((mod_pow6(B, M2) - mod_pow6(7 * d, M2)) % M2 + M2) % M2 == 0;
            const bool ok3 = ((mod_pow6(B, M3) - mod_pow6(7 * d, M3)) % M3 + M3) % M3 == 0;
            if (!ok2 || !ok3) { cerr << "cls4 d-filter fail\n"; return false; }
        }
    }
    // finders on synthetic targets
    QuadSumIndex ix;
    ix.build(6, 300);
    {
        const i128 Q = ipow(10, 6) + ipow(20, 6) + ipow(30, 6) + ipow(40, 6);
        auto q = find4(ix, Q, 300);
        if (!q) { cerr << "find4 miss\n"; return false; }
        i128 s = 0; for (int c : *q) s += ipow(c, 6);
        if (s != Q) { cerr << "find4 wrong sum\n"; return false; }
    }
    {
        const i128 Q = ipow(7, 6) + ipow(77, 6) + ipow(111, 6);
        auto t = find3(ix, Q, 300);
        if (!t) { cerr << "find3 miss\n"; return false; }
        i128 s = 0; for (int c : *t) s += ipow(c, 6);
        if (s != Q) { cerr << "find3 wrong sum\n"; return false; }
    }
    {
        const i128 Q = ipow(13, 6) + ipow(101, 6);
        auto p = find2(ix, Q, 300);
        if (!p || ipow((*p)[0], 6) + ipow((*p)[1], 6) != Q) { cerr << "find2 fail\n"; return false; }
    }
    cerr << "[selftest] all OK\n";
    return true;
}

// ---------------------------------------------------------------------- main --
int main(int argc, char** argv) {
    if (argc >= 2 && string(argv[1]) == "--selftest") return selftest() ? 0 : 1;
    if (argc < 3) {
        cerr << "usage: " << argv[0] << " <B_min> <B_max> [classes] [u_lo] [u_hi]\n"
             << "  classes: \"all\" (default) or subset like \"1,3\";  u band as fractions of B\n"
             << "  " << argv[0] << " --selftest\n";
        return 1;
    }

    const long long B_min = parse_i128_arg(argv[1]);
    const long long B_max = parse_i128_arg(argv[2]);
    if (B_max > B_HARD_MAX) {
        cerr << "B_max capped at " << B_HARD_MAX << " (i128: B^6 < 2^127)\n";
        return 1;
    }
    vector<int> classes{1, 2, 3, 4, 5};
    if (argc >= 4) {
        classes.clear();
        const string s = argv[3];
        if (s != "all") for (char c : s) if (c >= '1' && c <= '5') classes.push_back(c - '0');
        if (classes.empty()) classes = {1, 2, 3, 4, 5};
    }
    const double u_lo = argc >= 5 ? atof(argv[4]) : 0.0;
    const double u_hi = argc >= 6 ? atof(argv[5]) : 1.0;

    cerr << "solve_516_v2: B in [" << B_min << ", " << B_max << "], classes ";
    for (int c : classes) cerr << c;
    cerr << ", unit band [" << u_lo << ", " << u_hi << ")*B\n";
#ifdef _OPENMP
    cerr << "OpenMP threads: " << omp_get_max_threads() << "\n";
#endif

    // Pair table at N = B_max/42 (4x smaller than the 21-scheme's B_max/21).
    const int N_max = (int)(B_max / 42);
    if (N_max < 4) { cerr << "B_max too small (need N >= 4)\n"; return 1; }

    vector<long long> B_list;
    for (long long B = max(B_min, 86LL); B <= B_max; ++B)
        if ((B & 1) && B % 3 != 0 && B % 7 != 0) B_list.push_back(B);   // N1
    cerr << "eligible B (odd, gcd(B,21)=1): " << B_list.size() << "\n";

    QuadSumIndex index;
    const RootTables rt;
    {
        const auto t0 = Clock::now();
        index.build(K, N_max);
        const auto t1 = Clock::now();
        cerr << "index built in " << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() << " ms\n";
    }

    long solutions = 0;
    long long cand_units = 0, cand_quads = 0;

#pragma omp parallel for schedule(dynamic, 8) reduction(+ : solutions, cand_units, cand_quads)
    for (size_t bi = 0; bi < B_list.size(); ++bi) {
        const long long B = B_list[bi];
        for (int cls : classes) {
            const auto units = unit_candidates(rt, B, cls, u_lo, u_hi);
            cand_units += (long long)units.size();
            for (long long u : units) {
                ++cand_quads;
                switch (cls) {
                    case 1: run_class1(index, B, u, solutions); break;
                    case 2: run_class2(rt, index, B, u, solutions); break;
                    case 3: run_class3(rt, index, B, u, solutions); break;
                    case 4: run_class4(rt, index, B, u, solutions); break;
                    case 5: run_class5(rt, index, B, u, solutions); break;
                }
            }
        }
    }

    cerr << "unit candidates tested: " << cand_units << "\n";
    cerr << "total solutions: " << solutions << "\n";
    return 0;
}
