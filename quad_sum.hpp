// Single-target four-sum: find c1 <= c2 <= c3 <= c4 with c1^k + ... + c4^k = Q.
#pragma once

#include "k14_common.hpp"

#include <optional>

struct QuadSumIndex {
    int k = 6;
    int N = 0;
    std::vector<i128> pw;
    // All (i,j) pairs i<=j, sorted by pw[i]+pw[j] — compact vs std::map (~17 GiB at N=33809).
    std::vector<PairEntry> pairs;

    void build(int k_in, int N_in) {
        k = k_in;
        N = N_in;
        pw.assign(N + 1, 0);
        for (int i = 0; i <= N; ++i) pw[i] = ipow(i, k);

        pairs.clear();
        pairs.reserve((size_t)N * (N + 1) / 2);
        for (int i = 1; i <= N; ++i) {
            for (int j = i; j <= N; ++j) {
                pairs.push_back({pw[i] + pw[j], i, j});
            }
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const PairEntry& a, const PairEntry& b) { return a.sum < b.sum; });

        const size_t bytes = pairs.size() * sizeof(PairEntry) + pw.size() * sizeof(i128);
        std::cerr << "QuadSumIndex: k=" << k << " N=" << N << " pairs=" << pairs.size()
                  << " RAM ~" << (bytes >> 20) << " MiB (sorted pairs + powers)\n";
    }

    // Equal range [lo, hi) of pair indices with pairs[idx].sum == target.
    std::pair<size_t, size_t> equal_sum_range(i128 target) const {
        auto lo_it = std::lower_bound(
            pairs.begin(), pairs.end(), target,
            [](const PairEntry& e, i128 val) { return e.sum < val; });
        auto hi_it = lo_it;
        while (hi_it != pairs.end() && hi_it->sum == target) ++hi_it;
        return {static_cast<size_t>(lo_it - pairs.begin()),
                static_cast<size_t>(hi_it - pairs.begin())};
    }

    std::optional<std::array<int, 4>> find(i128 Q, int N_limit = -1) const {
        if (Q <= 0) return std::nullopt;
        const int lim = (N_limit < 0 || N_limit > N) ? N : N_limit;

        for (int kk = 1; kk <= lim; ++kk) {
            for (int ll = kk; ll <= lim; ++ll) {
                const i128 tail = pw[kk] + pw[ll];
                if (tail > Q) break;
                const i128 head = Q - tail;
                const auto [p_lo, p_hi] = equal_sum_range(head);
                for (size_t p = p_lo; p < p_hi; ++p) {
                    const int i = pairs[p].i;
                    const int j = pairs[p].j;
                    if (i > lim || j > lim) continue;
                    if (j <= kk) return std::array<int, 4>{i, j, kk, ll};
                }
            }
        }
        return std::nullopt;
    }
};

inline long long parse_i128_arg(const char* s) {
    return std::strtoll(s, nullptr, 10);
}

inline i128 parse_i128_target(const char* s) {
    return (i128)parse_i128_arg(s);
}

inline void print_i128(std::ostream& os, i128 x) {
    if (x == 0) {
        os << '0';
        return;
    }
    if (x < 0) {
        os << '-';
        x = -x;
    }
    std::string digits;
    while (x > 0) {
        digits.push_back(char('0' + (int)(x % 10)));
        x /= 10;
    }
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) os << *it;
}
