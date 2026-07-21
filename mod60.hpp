// mod60.hpp — cheap necessary-condition filters for sixth-power sums.
//
// mod 60 (square identity): x^6 ≡ x^2 (mod 60) for all integers x.
// mod 100 (sixth-power residues): B^6 ≡ sum(a_i^6) (mod 100), ~99.7% reject on
//   false quad hits in the (5.1.6) pipeline (same strength as mod 300 there).
//
// Use before expensive __int128 work; no false negatives on true solutions.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace mod60 {

inline int sq(long long x) {
    long long r = x % 60;
    if (r < 0) r += 60;
    r = (r * r) % 60;
    return static_cast<int>(r);
}

inline int sum_sq(std::initializer_list<long long> terms) {
    int s = 0;
    for (long long t : terms) s = (s + sq(t)) % 60;
    return s;
}

inline bool passes(long long B, std::initializer_list<long long> lhs_terms) {
    return sq(B) == sum_sq(lhs_terms);
}

// Phase-1 (5.1.6): a1..a4 = 21*c_i. Pre-filter (B, a5) before inner 4-sum.
// ~44% of random (B, a5) pairs fail this (no false negatives on true hits).
struct ScaledInnerFilter {
    int sq21[60]{};
    bool reach4[60]{};

    ScaledInnerFilter() {
        for (int c = 0; c < 60; ++c) {
            const long long v = (21LL * c) % 60;
            sq21[c] = static_cast<int>((v * v) % 60);
        }
        for (int c1 = 0; c1 < 60; ++c1) {
            for (int c2 = 0; c2 < 60; ++c2) {
                for (int c3 = 0; c3 < 60; ++c3) {
                    for (int c4 = 0; c4 < 60; ++c4) {
                        const int s = (sq21[c1] + sq21[c2] + sq21[c3] + sq21[c4]) % 60;
                        reach4[s] = true;
                    }
                }
            }
        }
    }

    bool passes_B_a5(long long B, long long a5) const {
        const int need = (sq(B) - sq(a5) + 120) % 60;
        return reach4[need];
    }
};

// Post-quad / post-MITM filter: sixth-power congruence mod M.
template <int M>
struct SixthPowerFilter {
    bool valid[M]{};

    SixthPowerFilter() {
        for (int x = 0; x < M; ++x) valid[pow6(x)] = true;
    }

    static int pow6(long long x) {
        long long r = x % M;
        if (r < 0) r += M;
        long long out = 1;
        for (int e = 0; e < 6; ++e) out = (out * r) % M;
        return static_cast<int>(out);
    }

    bool passes(long long B, std::initializer_list<long long> terms) const {
        const int lhs = pow6(B);
        int rhs = 0;
        for (long long t : terms) rhs = (rhs + pow6(t)) % M;
        return lhs == rhs && valid[rhs];
    }
};

using SixthPowerFilter100 = SixthPowerFilter<100>;
using SixthPowerFilter300 = SixthPowerFilter<300>;

}  // namespace mod60
