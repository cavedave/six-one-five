// k14_common.hpp — MINIMAL local reconstruction for compiling solve_516_v2
// outside the original project. On your own system, keep using YOUR
// k14_common.hpp (this file only provides what quad_sum.hpp / solve_516_v2
// actually need: i128, ipow, PairEntry).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

using i128 = __int128_t;

inline i128 ipow(long long base, int e) {
    i128 r = 1, b = base;
    while (e > 0) {
        if (e & 1) r *= b;
        b *= b;
        e >>= 1;
    }
    return r;
}

struct PairEntry {
    i128 sum;
    int i, j;
};
