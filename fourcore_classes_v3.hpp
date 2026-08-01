#pragma once
// =============================================================================
// fourcore_classes_v3.hpp — Meyrignac class contracts for (6,1,5), mirrored
// from solve_516_v3.cu (seeds_for_B / unit_ok / free-term residue classes).
//
// Spec table (Stage 1 of plan615Rest):
//
// | cls | master M | seed CRT            | #seed classes | unit_ok                         | free peel          | GPU later |
// |-----|----------|---------------------|---------------|---------------------------------|--------------------|-----------|
// | 1   | 42^6     | 2^6 × 3^6 × 7^6     | 144           | odd, gcd(u,42)=1                | — → find4 on T     | find4     |
// | 2   | 14^6     | 2^6 × 7^6           | 24            | odd, 3|u, 9∤u                   | (14·d) → find3     | find3     |
// | 3   | 21^6     | 3^6 × 7^6           | 36            | even, gcd(u,21)=1               | (21·d) → find3     | find3     |
// | 4   | 7^6      | 7^6                 | 6             | 6|u, 7∤u                        | (7·d)  → find3     | find3     |
// | 5   | 7^6      | 7^6                 | 6             | 6|u, 7∤u                        | (21d)+(14e)→find2  | find2     |
//
// Eligible B: gcd(B,42)=1 (same as v3 campaign loop).
// Core scaling after peels: always /42^6 into T (pair-table indices).
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace fc3 {

using u64 = std::uint64_t;
using u32 = std::uint32_t;

static constexpr long long M2 = 64LL;
static constexpr long long M3 = 729LL;
static constexpr long long M7 = 117649LL;
static constexpr long long M14 = 7529536LL;
static constexpr long long M21 = 85766121LL;
static constexpr long long M42 = 5489031744LL;

inline long long mod_pow6(long long base, long long mod) {
    long long r = 1;
    base %= mod;
    if (base < 0) base += mod;
    for (int e = 0; e < 6; ++e) r = (r * base) % mod;
    return r;
}
inline long long egcd(long long a, long long b, long long& x, long long& y) {
    if (b == 0) { x = 1; y = 0; return a; }
    long long x1, y1;
    const long long g = egcd(b, a % b, x1, y1);
    x = y1; y = x1 - y1 * (a / b);
    return g;
}
inline long long mod_inv(long long a, long long m) {
    long long x, y;
    egcd(a, m, x, y);
    x %= m; if (x < 0) x += m;
    return x;
}
inline long long crt2(long long r1, long long m1, long long r2, long long m2) {
    long long x, y;
    const long long g = egcd(m1, m2, x, y);
    const long long t = ((r2 - r1) / g) * x;
    const long long mod = m1 / g * m2;
    long long ans = r1 + m1 * (t % (m2 / g));
    ans %= mod;
    if (ans < 0) ans += mod;
    return ans;
}

struct RootTables {
    std::vector<std::vector<long long>> r2, r3, r7;
    RootTables() : r2(M2), r3(M3), r7(M7) {
        for (long long a = 1; a < M2; ++a)
            if (std::gcd(a, M2) == 1) r2[mod_pow6(a, M2)].push_back(a);
        for (long long a = 1; a < M3; ++a)
            if (std::gcd(a, M3) == 1) r3[mod_pow6(a, M3)].push_back(a);
        for (long long a = 1; a < M7; ++a)
            if (std::gcd(a, M7) == 1) r7[mod_pow6(a, M7)].push_back(a);
    }
};

inline bool admissible_B(long long B) {
    return B > 1 && std::gcd(B, 42LL) == 1;
}

inline long long class_modulus(int cls) {
    switch (cls) {
        case 1: return M42;
        case 2: return M14;
        case 3: return M21;
        default: return M7;  // 4,5
    }
}

inline int free_factor(int cls) {
    switch (cls) {
        case 2: return 14;
        case 3: return 21;
        case 4: return 7;
        default: return 0;
    }
}

inline bool unit_ok(long long u, int cls) {
    switch (cls) {
        case 1: return (u & 1) && std::gcd(u, 42LL) == 1;
        case 2: return (u & 1) && (u % 3 == 0) && (u % 9 != 0);
        case 3: return !(u & 1) && std::gcd(u, 21LL) == 1;
        case 4:
        case 5: return (u % 6 == 0) && (u % 7 != 0);
    }
    return false;
}

// Exact port of solve_516_v3::seeds_for_B.
inline std::vector<long long> seeds_for_B(const RootTables& rt, long long B, int cls) {
    const long long b2 = mod_pow6(B, M2), b3 = mod_pow6(B, M3), b7 = mod_pow6(B, M7);
    std::vector<long long> out;
    if (cls == 1) {
        for (long long s2 : rt.r2[b2])
            for (long long s3 : rt.r3[b3]) {
                const long long s23 = crt2(s2, M2, s3, M3);
                for (long long s7 : rt.r7[b7]) out.push_back(crt2(s23, M2 * M3, s7, M7));
            }
    } else if (cls == 2) {
        for (long long s2 : rt.r2[b2])
            for (long long s7 : rt.r7[b7]) out.push_back(crt2(s2, M2, s7, M7));
    } else if (cls == 3) {
        for (long long s3 : rt.r3[b3])
            for (long long s7 : rt.r7[b7]) out.push_back(crt2(s3, M3, s7, M7));
    } else {
        out = rt.r7[b7];
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

inline std::vector<long long> unit_candidates(const RootTables& rt, long long B, int cls,
                                              double lo_frac, double hi_frac) {
    const long long M = class_modulus(cls);
    const long long lo = std::max(1LL, (long long)(lo_frac * (double)B));
    const long long hi = std::min(B - 1, (long long)(hi_frac * (double)B));
    std::vector<long long> out;
    if (lo > hi) return out;
    for (long long s : seeds_for_B(rt, B, cls)) {
        long long u = s % M;
        if (u < 0) u += M;
        if (u < lo) u += (lo - u + M - 1) / M * M;
        for (; u <= hi; u += M)
            if (unit_ok(u, cls)) out.push_back(u);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

inline std::vector<long long> classes_d_cls2(const RootTables& rt, long long B) {
    const long long t = mod_pow6(B, M3) * mod_inv(mod_pow6(14, M3), M3) % M3;
    return rt.r3[t];
}
inline std::vector<long long> classes_d_cls3(const RootTables& rt, long long B, long long u) {
    long long rhs = (mod_pow6(B, M2) - mod_pow6(u, M2)) % M2;
    if (rhs < 0) rhs += M2;
    const long long t = rhs * mod_inv(mod_pow6(21, M2), M2) % M2;
    return rt.r2[t];
}
inline std::vector<long long> classes_d_cls4(const RootTables& rt, long long B) {
    const long long t2 = mod_pow6(B, M2) * mod_inv(mod_pow6(7, M2), M2) % M2;
    const long long t3 = mod_pow6(B, M3) * mod_inv(mod_pow6(7, M3), M3) % M3;
    std::vector<long long> out;
    for (long long a : rt.r2[t2])
        for (long long b : rt.r3[t3]) out.push_back(crt2(a, M2, b, M3));
    std::sort(out.begin(), out.end());
    return out;
}
inline std::vector<long long> classes_e_cls5(const RootTables& rt, long long B) {
    return classes_d_cls2(rt, B);
}
inline std::vector<long long> classes_d_cls5(const RootTables& rt, long long B, long long u) {
    return classes_d_cls3(rt, B, u);
}

struct FreeTermSpec {
    long long mod = 0;
    long long fmax = 0;
    std::vector<long long> residues;  // representatives mod `mod`
};

inline FreeTermSpec free_d_cls234(const RootTables& rt, long long B, long long u, int cls) {
    FreeTermSpec s;
    switch (cls) {
        case 2:
            s.residues = classes_d_cls2(rt, B);
            s.mod = M3;
            s.fmax = (B - 1) / 14;
            break;
        case 3:
            s.residues = classes_d_cls3(rt, B, u);
            s.mod = M2;
            s.fmax = (B - 1) / 21;
            break;
        case 4:
            s.residues = classes_d_cls4(rt, B);
            s.mod = M2 * M3;
            s.fmax = (B - 1) / 7;
            break;
        default: break;
    }
    return s;
}

inline void free_de_cls5(const RootTables& rt, long long B, long long u,
                         FreeTermSpec& e_spec, FreeTermSpec& d_spec) {
    e_spec.residues = classes_e_cls5(rt, B);
    e_spec.mod = M3;
    e_spec.fmax = (B - 1) / 14;
    d_spec.residues = classes_d_cls5(rt, B, u);
    d_spec.mod = M2;
    d_spec.fmax = (B - 1) / 21;
}

// Enumerate free-term values exactly as v3 kernels:
//   d = res[i] + k*mod, k>=0, skip d==0 or d>fmax.
template <typename Fn>
inline void for_each_free(const FreeTermSpec& s, Fn&& fn) {
    if (s.mod <= 0 || s.fmax < 1) return;
    for (long long r : s.residues) {
        for (long long d = r; d <= s.fmax; d += s.mod) {
            if (d == 0) continue;
            fn(d);
        }
    }
}

inline u64 count_free(const FreeTermSpec& s) {
    u64 n = 0;
    for_each_free(s, [&](long long) { ++n; });
    return n;
}

}  // namespace fc3
