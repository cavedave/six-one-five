#pragma once
// Host bigint helpers for post-i128 (6,1,5) four-core hunt.
// Requires GMP: -lgmpxx -lgmp  (apt: libgmp-dev; brew: gmp)

#include <cstdint>
#include <gmpxx.h>
#include <stdexcept>
#include <string>

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using u128 = unsigned __int128;

inline mpz_class mpz_pow6(u64 x) {
    mpz_class z((unsigned long)x);
    z *= z;           // x^2
    mpz_class z2 = z;
    z *= z;           // x^4
    z *= z2;          // x^6
    return z;
}

// Import non-negative mpz into u128; false if > 128 bits.
inline bool mpz_to_u128(const mpz_class& q, u128& T_out) {
    if (q < 0) return false;
    if (mpz_sizeinbase(q.get_mpz_t(), 2) > 128) return false;
    T_out = 0;
    if (q == 0) return true;
    mpz_export(&T_out, nullptr, -1, sizeof(u64), 0, 0, q.get_mpz_t());
    return true;
}

// T = (B^6 - u^6) / D^6  exactly; returns false if not divisible or negative.
inline bool compute_T_gmp(u64 B, u64 u, u64 D, u128& T_out) {
    if (u >= B || D < 2) return false;
    mpz_class num = mpz_pow6(B) - mpz_pow6(u);
    if (num <= 0) return false;
    mpz_class den = mpz_pow6(D);
    mpz_class r = num % den;
    if (r != 0) return false;
    return mpz_to_u128(num / den, T_out);
}

// T = (B^6 - u^6 - (scale*free)^6) / 42^6  (cls2/3/4 peel of one free term).
inline bool compute_T_peel1_gmp(u64 B, u64 u, u64 scale, u64 free, u128& T_out) {
    if (u >= B || scale < 1 || free < 1) return false;
    mpz_class num = mpz_pow6(B) - mpz_pow6(u) - mpz_pow6(scale * free);
    if (num <= 0) return false;
    mpz_class den = mpz_pow6(42);
    if (num % den != 0) return false;
    return mpz_to_u128(num / den, T_out);
}

// T = (B^6 - u^6 - (21*d)^6 - (14*e)^6) / 42^6  (cls5).
inline bool compute_T_peel2_gmp(u64 B, u64 u, u64 d, u64 e, u128& T_out) {
    if (u >= B || d < 1 || e < 1) return false;
    mpz_class num = mpz_pow6(B) - mpz_pow6(u) - mpz_pow6(21 * d) - mpz_pow6(14 * e);
    if (num <= 0) return false;
    mpz_class den = mpz_pow6(42);
    if (num % den != 0) return false;
    return mpz_to_u128(num / den, T_out);
}

// Cached cls5 peel: reuse B^6, u^6, 42^6 across the (d,e) grid for one unit.
struct Cls5PeelCtx {
    mpz_class base;   // B^6 - u^6
    mpz_class den;    // 42^6
    bool ok = false;

    void init(u64 B, u64 u) {
        ok = false;
        if (u >= B) return;
        base = mpz_pow6(B) - mpz_pow6(u);
        if (base <= 0) return;
        den = mpz_pow6(42);
        ok = true;
    }

    // T = (base - (21*d)^6 - (14*e)^6) / 42^6
    bool peel(u64 d, u64 e, u128& T_out) const {
        if (!ok || d < 1 || e < 1) return false;
        mpz_class num = base - mpz_pow6(21 * d) - mpz_pow6(14 * e);
        if (num <= 0) return false;
        if (num % den != 0) return false;
        return mpz_to_u128(num / den, T_out);
    }
};

// Exact: B^6 == u^6 + D^6 * (x1^6+...+x4^6)
inline bool verify_fourcore_gmp(u64 B, u64 u, u64 D,
                                u64 x1, u64 x2, u64 x3, u64 x4) {
    mpz_class rhs = mpz_pow6(u);
    mpz_class core = mpz_pow6(x1) + mpz_pow6(x2) + mpz_pow6(x3) + mpz_pow6(x4);
    rhs += mpz_pow6(D) * core;
    return rhs == mpz_pow6(B);
}

// Exact five-term (6,1,5) identity via GMP (post-i128 safe).
inline bool verify_615_gmp(u64 B, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    mpz_class rhs = mpz_pow6(a1) + mpz_pow6(a2) + mpz_pow6(a3) + mpz_pow6(a4) + mpz_pow6(a5);
    return rhs == mpz_pow6(B);
}

// cls234 shape: B^6 = u^6 + (f*d)^6 + 42^6 * (x1^6+x2^6+x3^6)
inline bool verify_cls234_gmp(u64 B, u64 u, u64 f, u64 d,
                              u64 x1, u64 x2, u64 x3) {
    mpz_class rhs = mpz_pow6(u) + mpz_pow6(f * d)
                  + mpz_pow6(42) * (mpz_pow6(x1) + mpz_pow6(x2) + mpz_pow6(x3));
    return rhs == mpz_pow6(B);
}

// cls5 shape: B^6 = u^6 + (21*d)^6 + (14*e)^6 + 42^6 * (x1^6+x2^6)
inline bool verify_cls5_gmp(u64 B, u64 u, u64 d, u64 e, u64 x1, u64 x2) {
    mpz_class rhs = mpz_pow6(u) + mpz_pow6(21 * d) + mpz_pow6(14 * e)
                  + mpz_pow6(42) * (mpz_pow6(x1) + mpz_pow6(x2));
    return rhs == mpz_pow6(B);
}

inline void split_u128(u128 T, u64& lo, u64& hi) {
    lo = (u64)T;
    hi = (u64)(T >> 64);
}

inline u128 join_u128(u64 lo, u64 hi) {
    return ((u128)hi << 64) | lo;
}
