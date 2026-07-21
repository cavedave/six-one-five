// v3_host_logic_test.cpp — host mirror of the solve_516_v3.cu DEVICE math,
// validated against exact __int128 ground truth. No CUDA required.
// Compile: g++ -O2 -std=c++20 -o v3_host_logic_test v3_host_logic_test.cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

using u64 = unsigned long long;
using u32 = unsigned int;
using i128 = __int128_t;
using u128 = __uint128_t;

static constexpr long long M42 = 5489031744LL;
static constexpr long long M21 = 85766121LL;

static i128 ipow6(long long x) { i128 r = 1, b = x; for (int e = 0; e < 6; ++e) r *= b; return r; }

// ---- host stand-ins for device intrinsics ----
static inline u64 umul64hi(u64 a, u64 b) { return (u64)(((u128)a * b) >> 64); }

// ---- verbatim mirrors of the device functions (with __umul64hi -> umul64hi) ----
static inline void pow6_128(u32 x, u64& hi, u64& lo) {
    const u64 x2 = (u64)x * x;
    const u64 h4 = umul64hi(x2, x2), l4 = x2 * x2;
    lo = l4 * x2;
    hi = umul64hi(l4, x2) + h4 * x2;
}
static inline u64 pow6_64(u32 x) { const u64 x2 = (u64)x * x; return x2 * x2 * x2; }
static inline bool mul128_small(u64& hi, u64& lo, u64 m) {
    const u64 h1 = umul64hi(lo, m);
    const u64 h2 = hi * m;
    bool of = (hi != 0) && (umul64hi(hi, m) != 0);
    hi = h1 + h2;
    of |= (hi < h1);
    lo = lo * m;
    return of;
}
static inline bool gt128(u64 ah, u64 al, u64 bh, u64 bl) {
    return (ah > bh) || (ah == bh && al > bl);
}
static bool scaled_gt(u64 rh, u64 rl, u32 c, u64 scale) {
    u64 ch, cl; pow6_128(c, ch, cl);
    if (mul128_small(ch, cl, scale)) return true;
    return gt128(ch, cl, rh, rl);
}
static bool scaled_lt(u64 rh, u64 rl, u32 c, u64 scale) {
    u64 ch, cl; pow6_128(c, ch, cl);
    if (mul128_small(ch, cl, scale)) return false;
    return gt128(rh, rl, ch, cl);
}
static u32 iroot6_fix(u64 rh, u64 rl, double inv_scale, u64 scale) {
    const double d = (double)rh * 18446744073709551616.0 + (double)rl;
    u32 r = (u32)(pow(d * inv_scale, 1.0 / 6.0)) + 2;
    while (r > 0 && scaled_gt(rh, rl, r, scale)) --r;
    while (!scaled_gt(rh, rl, r + 1, scale)) ++r;
    return r;
}
static u32 min_c_fix(u64 rh, u64 rl, u32 r_hi, u64 scale) {
    u32 c = (u32)(r_hi * 0.8326831149);
    if (c < 1) c = 1;
    while (scaled_lt(rh, rl, c, 3 * scale)) ++c;
    while (c > 1 && !scaled_lt(rh, rl, c - 1, 3 * scale)) --c;
    return c;
}
static inline u64 funnel_fp(u64 rh, u64 rl, u64 inv216) {
    return ((rl >> 6) | (rh << 58)) * inv216;
}

// ---- exact references ----
static u32 ref_iroot6_scaled(u128 R, u64 scale) {   // largest r: scale*r^6 <= R
    u32 lo = 0, hi = 1;
    while ((u128)scale * ipow6(hi) <= R && hi < (1u << 22)) hi <<= 1;
    while (hi - lo > 1) { u32 m = (lo + hi) >> 1; if ((u128)scale * ipow6(m) <= R) lo = m; else hi = m; }
    return lo;
}
static u32 ref_min_c(u128 R, u64 scale) {           // smallest c>=1: 3*scale*c^6 >= R
    u32 c = 1;
    while ((u128)3 * scale * ipow6(c) < R) ++c;
    return c;
}

int main() {
    srand(20260720);
    // 1) pow6_128 vs i128
    for (int t = 0; t < 300000; ++t) {
        const u32 x = (t < 100000) ? (u32)t : (u32)(rand() % 2300000);
        u64 hi, lo; pow6_128(x, hi, lo);
        const u128 ref = (u128)ipow6(x);
        if (x < 3000000u && ((u64)(ref >> 64) != hi || (u64)ref != lo)) {
            printf("pow6_128 FAIL x=%u\n", x); return 1;
        }
    }
    printf("pow6_128 OK\n");
    // 2) pow6_64 low half
    for (int t = 0; t < 100000; ++t) {
        const u32 x = 1 + rand() % 3000000;
        if (pow6_64(x) != (u64)(u128)ipow6(x)) { printf("pow6_64 FAIL\n"); return 1; }
    }
    printf("pow6_64 OK\n");
    // 3) iroot6_fix / min_c_fix vs exact, scales 1 and 42^6, incl. huge R
    for (int t = 0; t < 60000; ++t) {
        u128 R;
        if (t % 3 == 0)      R = ((u128)rand() << 90) ^ ((u128)rand() << 45) ^ rand();   // up to ~2^120
        else if (t % 3 == 1) R = ((u128)rand() << 60) ^ rand();
        else                 R = rand() % 1000000;
        const u64 scale = (t % 2 == 0) ? 1ULL : (u64)M42;
        const double inv = 1.0 / (double)scale;
        const u64 rh = (u64)(R >> 64), rl = (u64)R;
        const u32 got = iroot6_fix(rh, rl, inv, scale);
        const u32 want = ref_iroot6_scaled(R, scale);
        if (got != want) { printf("iroot6_fix FAIL t=%d got=%u want=%u\n", t, got, want); return 1; }
        const u32 glo = min_c_fix(rh, rl, got, scale);
        const u32 wlo = ref_min_c(R, scale);
        if (glo != wlo) { printf("min_c_fix FAIL t=%d got=%u want=%u\n", t, glo, wlo); return 1; }
    }
    printf("iroot6_fix/min_c_fix OK\n");
    // 4) funnel identity
    u64 a = (u64)M21, inv216 = a;
    for (int i = 0; i < 6; ++i) inv216 = inv216 * (2 - a * inv216);
    if ((u64)((u128)M21 * inv216) != 1) { printf("inv216 FAIL\n"); return 1; }
    for (int t = 0; t < 200000; ++t) {
        const u128 T = ((u128)rand() << 90) ^ ((u128)rand() << 45) ^ rand();
        const u128 R = T * (u128)M42;
        if (funnel_fp((u64)(R >> 64), (u64)R, inv216) != (u64)T) { printf("funnel FAIL\n"); return 1; }
    }
    printf("funnel identity OK\n");
    // 5) window completeness: a planted quad's max element is inside the window
    for (int t = 0; t < 20000; ++t) {
        int c[4];
        for (int k = 0; k < 4; ++k) c[k] = 1 + rand() % 52000;
        int cmax = 0; for (int k = 0; k < 4; ++k) if (c[k] > cmax) cmax = c[k];
        const u128 Q = (u128)ipow6(c[0]) + ipow6(c[1]) + ipow6(c[2]) + ipow6(c[3]);
        // find4 window (host, exact): lo4 = smallest c with 4c^6 >= Q, hi4 = iroot6(Q)
        const u32 hi4 = ref_iroot6_scaled(Q, 1);
        if (cmax > (int)hi4) { printf("window hi FAIL\n"); return 1; }
        u32 lo4 = 1; while ((u128)4 * ipow6(lo4) < Q) ++lo4;
        if (cmax < (int)lo4) { printf("window lo FAIL\n"); return 1; }
        // find3 window on remainder after removing the max
        i128 rem = 0; bool skipped = false;
        for (int k = 0; k < 4; ++k) { if (!skipped && c[k] == cmax) { skipped = true; continue; } rem += ipow6(c[k]); }
        int cmax2 = 0; for (int k = 0; k < 4; ++k) { if (k == 0 && false) continue; if (c[k] == cmax && &c[k] == &c[k]) {} }
        // recompute second max properly
        int m1 = 0, m2 = 0;
        for (int k = 0; k < 4; ++k) { if (c[k] >= m1) { m2 = m1; m1 = c[k]; } else if (c[k] > m2) m2 = c[k]; }
        const u64 rh = (u64)((u128)rem >> 64), rl = (u64)rem;
        const u32 hi3 = iroot6_fix(rh, rl, 1.0, 1);
        const u32 lo3 = min_c_fix(rh, rl, hi3, 1);
        if (m2 < (int)lo3 || m2 > (int)hi3) { printf("find3 window FAIL t=%d m2=%d [%u,%u]\n", t, m2, lo3, hi3); return 1; }
    }
    printf("window completeness OK\n");
    printf("ALL HOST LOGIC TESTS PASS\n");
    return 0;
}
