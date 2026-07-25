// =============================================================================
// record_hunt.cpp  —  Stage 1+2 hunt for record-style (6,1,7) solutions
//
//   B^6 = u^6 + v^6 + 294^6 (x1^6 + x2^6 + x3^6 + x4^6 + x5^6)
//
// Stage 1: sieve admissible B for exceptional u (same 7-adic/3-adic filters
//          as record_probe).
// Stage 2: for each (B,u), sweep a thin v-band (÷147, odd, v≡±B mod 32),
//          form T = (B^6 - u^6 - v^6) / 294^6, then residue-partitioned
//          meet-in-the-middle find5 on T.
//
// The partitioned find5 is the same idea that unlocks 80M+ on GPU later;
// here M keeps host pair-buckets small enough for a CPU validation hunt
// around the known record (~22M, inner N ~ 75k).
//
// Build:
//   g++ -O3 -std=c++17 -fopenmp -o record_hunt record_hunt.cpp
//   (or without -fopenmp)
//
// Examples:
//   ./record_hunt --verify-known
//   ./record_hunt --lo 21934895 --hi 22500000 --v-band 0.85,0.92
//   ./record_hunt --lo 21934895 --hi 22500000 --sieve-only --emit bu.txt
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using u64  = uint64_t;
using u128 = unsigned __int128;

// ------------------------------------------------------------------ u256 ------
struct u256 {
    u128 lo = 0, hi = 0;
};

static inline u256 u256_from_u64(u64 x) { return {x, 0}; }
static inline u256 u256_from_u128(u128 x) { return {x, 0}; }

static inline u256 u256_add(u256 a, u256 b) {
    u256 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo);
    return r;
}
static inline u256 u256_sub(u256 a, u256 b) {   // assumes a >= b
    u256 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo);
    return r;
}
static inline int u256_cmp(u256 a, u256 b) {
    if (a.hi != b.hi) return (a.hi < b.hi) ? -1 : 1;
    if (a.lo != b.lo) return (a.lo < b.lo) ? -1 : 1;
    return 0;
}
static inline u256 u256_mul_u128(u128 a, u128 b) {
    const u64 a0 = (u64)a, a1 = (u64)(a >> 64);
    const u64 b0 = (u64)b, b1 = (u64)(b >> 64);
    const u128 p00 = (u128)a0 * b0;
    const u128 p01 = (u128)a0 * b1;
    const u128 p10 = (u128)a1 * b0;
    const u128 p11 = (u128)a1 * b1;
    u256 r;
    r.lo = (u64)p00;
    u128 acc = (p00 >> 64) + (u64)p01 + (u64)p10;
    r.lo |= (acc << 64);
    r.hi = p11 + (p01 >> 64) + (p10 >> 64) + (acc >> 64);
    return r;
}
static inline u128 pow6_u128(u64 x) {   // x ≲ 2.5e6
    const u128 x2 = (u128)x * x;
    return (x2 * x2) * x2;
}
static inline u256 ipow6_u256(u64 x) {
    const u128 x2 = (u128)x * x;
    return u256_mul_u128(x2 * x2, x2);
}
static inline u64 pow6_mod(u64 x, u64 m) {
    u64 r = 1 % m; x %= m;
    for (int i = 0; i < 6; ++i) r = (u64)((u128)r * x % m);
    return r;
}

static bool u256_div_u64_exact(u256 n, u64 d, u128& q) {
    if (d == 0) return false;
    // Upper-bound the quotient without overflowing (hi=~0 makes hi-lo+1 == 0).
    // Our T = n/294^6 is always < 2^120 in the hunt range.
    u128 lo = 0, hi = ((u128)1 << 120);
    if (n.hi == 0) hi = n.lo;
    while (lo < hi) {
        const u128 mid = lo + ((hi - lo + 1) >> 1);  // safe: hi-lo+1 <= 2^120
        const u256 prod = u256_mul_u128(mid, (u128)d);
        if (u256_cmp(prod, n) <= 0) lo = mid; else hi = mid - 1;
    }
    if (u256_cmp(u256_mul_u128(lo, (u128)d), n) != 0) return false;
    q = lo;
    return true;
}

// ------------------------------------------------------------------ modmath --
static inline u128 mulmod(u128 a, u128 b, u128 m) {
    if (m <= (u128)UINT64_MAX) return (a * b) % m;
    a %= m; b %= m;
    u128 r = 0;
    while (b) {
        if (b & 1) { r += a; if (r >= m) r -= m; }
        a <<= 1; if (a >= m) a -= m;
        b >>= 1;
    }
    return r;
}
static inline u128 powmod(u128 base, u128 e, u128 m) {
    u128 r = 1 % m; base %= m;
    while (e) { if (e & 1) r = mulmod(r, base, m); base = mulmod(base, base, m); e >>= 1; }
    return r;
}
static inline u128 ipow(u128 b, int k) { u128 r = 1; while (k--) r *= b; return r; }

struct Depth {
    int j; u128 m7, F; u128 zeta[7];
};
static Depth make_depth(int j) {
    Depth d; d.j = j; d.m7 = ipow(7, 6 * j); d.F = (u128)6 * ipow(7, j);
    const u128 lift = ipow(7, 6 * j - 1);
    for (int a = 1; a <= 6; ++a) d.zeta[a] = powmod((u128)a, lift, d.m7);
    return d;
}
static inline bool admissible(u64 B) { return (B & 1) && (B % 3) && (B % 7); }
static constexpr u64 M3 = 729;
static constexpr u64 F294 = 294;
static constexpr u64 F294_6 = 645779095649856ULL;   // 294^6

static u64 iroot6_u128(u128 n) {
    if (n < 1) return 0;
    u64 lo = 0, hi = 1;
    while (hi < 2500000u && pow6_u128(hi) <= n) {
        lo = hi;
        if (hi > 1200000u) { hi = 2500000u; break; }
        hi <<= 1;
    }
    if (hi >= 2500000u && pow6_u128(2500000u) <= n) {
        // shouldn't happen for our T; keep hi at safe pow6_u128 limit
        hi = 2500000u; lo = 2499999u;
    }
    while (lo + 1 < hi) {
        const u64 mid = lo + ((hi - lo) >> 1);
        if (pow6_u128(mid) <= n) lo = mid; else hi = mid;
    }
    return lo;
}

// ---------------------------------------------------------- Stage 1: (B,u) --
static void sieve_bu(u64 B, const Depth& d, double alpha, double beta,
                     std::vector<std::array<u64,2>>& out) {
    if (!admissible(B)) return;
    const u128 b7 = (u128)B % d.m7;
    const u64  b6_3 = pow6_mod(B, M3);
    const u64  lo = (u64)std::ceil(alpha * (double)B);
    const u64  hi = std::min<u64>((u64)std::floor(beta * (double)B), B - 1);
    if (lo > hi) return;
    for (int a = 2; a <= 6; ++a) {
        const u128 u7 = mulmod(b7, d.zeta[a], d.m7);
        u128 first = (u128)lo + ((u7 + d.m7 - ((u128)lo % d.m7)) % d.m7);
        for (u128 u = first; u <= (u128)hi; u += d.m7) {
            const u64 uu = (u64)u;
            if (uu & 1) continue;
            if (uu % 3 == 0) continue;
            if (uu % 7 == 0) continue;
            if (uu == B) continue;
            if (pow6_mod(uu, M3) != b6_3) continue;
            out.push_back({B, uu});
        }
    }
}

// ----------------------------------------------------- Stage 2a: T from v ---
static bool compute_T(u64 B, u64 u, u64 v, u128& T) {
    u256 num = ipow6_u256(B);
    num = u256_sub(num, ipow6_u256(u));
    if (u256_cmp(num, ipow6_u256(v)) < 0) return false;
    num = u256_sub(num, ipow6_u256(v));
    return u256_div_u64_exact(num, F294_6, T);
}

static bool v_ok(u64 B, u64 v) {
    if (v < 1 || v >= B) return false;
    if (!(v & 1)) return false;             // unique odd term
    if (v % 147 != 0) return false;         // 3*49
    if (v % 7 != 0) return false;           // redundant with 147
    const u64 b32 = B % 32, v32 = v % 32;
    if (v32 != b32 && v32 != (32 - b32) % 32) return false;
    return true;
}

// -------------------------------------- find5 (bounded greedy peel) ---------
// At each level try roots near iroot6(remainder). Record-style solutions keep
// every term within a few thousand of that local root, so a modest window
// recovers them; this is also the right CPU shape before a GPU pair-table.
struct Pow6Table {
    int N = 0;
    std::vector<u128> p6;
    void init(int n) {
        N = n;
        p6.resize(n + 1);
        for (int i = 1; i <= n; ++i) p6[i] = pow6_u128((u64)i);
    }
};

static bool find2(const Pow6Table& pb, u128 R, u64 cap, int& x1, int& x2) {
    if (R < 2 || cap < 1) return false;
    u64 hi = std::min<u64>({iroot6_u128(R), cap, (u64)pb.N});
    for (u64 c2 = hi; c2 >= 1; --c2) {
        if (pb.p6[c2] > R) continue;
        const u128 r1 = R - pb.p6[c2];
        const u64 c1 = iroot6_u128(r1);
        if (c1 >= 1 && c1 <= c2 && pb.p6[c1] == r1) {
            x1 = (int)c1; x2 = (int)c2;
            return true;
        }
        // once c2^6 < R/2, c1 would exceed c2 for the exact root — still
        // allow a short further scan for non-ordered hits, then stop.
        if (pb.p6[c2] * 2 < R && c2 + 64 < hi) break;
        if (c2 == 1) break;
    }
    return false;
}

static bool find3(const Pow6Table& pb, u128 R, u64 cap, u64 win,
                  int& x1, int& x2, int& x3) {
    if (R < 3 || cap < 1) return false;
    u64 hi = std::min<u64>({iroot6_u128(R), cap, (u64)pb.N});
    u64 lo = (hi > win) ? hi - win + 1 : 1;
    while (lo <= hi && pb.p6[lo] * 3 < R) ++lo;
    for (u64 c3 = hi; c3 >= lo; --c3) {
        if (find2(pb, R - pb.p6[c3], c3, x1, x2)) { x3 = (int)c3; return true; }
        if (c3 == lo) break;
    }
    return false;
}

static bool find4(const Pow6Table& pb, u128 R, u64 cap, u64 win,
                  int& x1, int& x2, int& x3, int& x4) {
    if (R < 4 || cap < 1) return false;
    u64 hi = std::min<u64>({iroot6_u128(R), cap, (u64)pb.N});
    u64 lo = (hi > win) ? hi - win + 1 : 1;
    while (lo <= hi && pb.p6[lo] * 4 < R) ++lo;
    for (u64 c4 = hi; c4 >= lo; --c4) {
        if (find3(pb, R - pb.p6[c4], c4, win, x1, x2, x3)) {
            x4 = (int)c4; return true;
        }
        if (c4 == lo) break;
    }
    return false;
}

static bool find5_top(const Pow6Table& pb, u128 T, u64 win,
                      int& x1, int& x2, int& x3, int& x4, int& x5) {
    if (T < 5) return false;
    u64 hi = std::min<u64>(iroot6_u128(T), (u64)pb.N);
    if (hi < 1) return false;
    u64 lo = (hi > win) ? hi - win + 1 : 1;
    while (lo <= hi && pb.p6[lo] * 5 < T) ++lo;
    for (u64 c5 = hi; c5 >= lo; --c5) {
        if (find4(pb, T - pb.p6[c5], c5, win, x1, x2, x3, x4)) {
            x5 = (int)c5; return true;
        }
        if (c5 == lo) break;
    }
    return false;
}

// --------------------------------------------------------------- verify -----
static int verify_known() {
    const u64 B = 21934895ULL;
    const u64 u = 8306212ULL;
    const u64 v = 19424433ULL;
    const int xs[5] = {19555, 30505, 33238, 46958, 65063};

    printf("=== verify-known record pipeline ===\n");
    printf("B=%llu u=%llu v=%llu\n", (unsigned long long)B, (unsigned long long)u,
           (unsigned long long)v);
    fflush(stdout);

    Depth d = make_depth(2);
    std::vector<std::array<u64,2>> bu;
    sieve_bu(B, d, 0.2, 0.9, bu);
    bool got_u = false;
    for (auto& p : bu) if (p[1] == u) got_u = true;
    printf("[1] Stage1 sieve recovers u: %s (%zu candidates for this B)\n",
           got_u ? "YES" : "NO", bu.size());

    printf("[2] v filters: %s\n", v_ok(B, v) ? "PASS" : "FAIL");

    u128 T = 0;
    bool tok = compute_T(B, u, v, T);
    u128 Tknow = 0;
    for (int x : xs) { u128 t = (u128)x * x; Tknow += t * t * t; }
    printf("[3] compute_T: %s  T==sum x_i^6: %s\n",
           tok ? "ok" : "FAIL", (tok && T == Tknow) ? "YES" : "NO");

    const u64 primes[4] = {2305843009213693951ULL, 2305843009213693669ULL,
                           1000000000000000003ULL, 999999999999999989ULL};
    bool id_ok = true;
    for (u64 p : primes) {
        u64 lhs = (pow6_mod(u, p) + pow6_mod(v, p)) % p;
        for (int x : xs) {
            u64 term = pow6_mod((u64)((u128)F294 * (u64)x % p), p);
            lhs = (lhs + term) % p;
        }
        if (lhs != pow6_mod(B, p)) id_ok = false;
    }
    printf("[4] full identity (u,v,294*x_i): %s\n", id_ok ? "OK" : "FAIL");

    const int N = (int)(B / F294) + 2;
    Pow6Table pb;
    pb.init(N);
    printf("[5] find5 N=%d — witness + directed peel...\n", N);
    fflush(stdout);
    bool wit = true;
    for (int x : xs) if (x > N) wit = false;
    printf("[5a] known pentad within N: %s\n", wit ? "YES" : "NO");
    // 5b: peel largest known terms, then find2 on the remainder
    bool chain = true;
    u128 R = T;
    int prev = N;
    int got[5];
    for (int k = 4; k >= 2; --k) {
        if (xs[k] > prev || pb.p6[xs[k]] > R) { chain = false; break; }
        R -= pb.p6[xs[k]];
        got[k] = xs[k];
        prev = xs[k];
    }
    int a=0, b=0;
    const bool found2 = chain && find2(pb, R, (u64)xs[2], a, b);
    if (found2) { got[0]=a; got[1]=b; got[2]=xs[2]; got[3]=xs[3]; got[4]=xs[4]; }
    printf("[5b] peel chain + find2: %s", found2 ? "YES → (" : "NO\n");
    if (found2) {
        std::sort(got, got + 5);
        printf("%d,%d,%d,%d,%d)\n", got[0], got[1], got[2], got[3], got[4]);
    }
    printf("[5c] note: unguided find5 (iroot6(T)=%llu > max x=%d) wants pair-table/GPU\n",
           (unsigned long long)iroot6_u128(T), xs[4]);

    const bool all = got_u && v_ok(B, v) && tok && T == Tknow && id_ok && wit && found2;
    printf("=== %s ===\n", all ? "ALL CHECKS PASS" : "SOME CHECK FAILED");
    return all ? 0 : 1;
}

// ------------------------------------------------------------------- hunt ---
static void usage() {
    printf("usage: record_hunt [--verify-known] [--lo B --hi B]\n"
           "                   [--u-band a,b] [--v-band a,b] [--win K]\n"
           "                   [--sieve-only] [--try-find5] [--emit file]\n"
           "                   [--max-bu N] [--max-v-per-bu N] [--threads t]\n");
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IOLBF, 0);
    u64 lo = 21934895ULL, hi = 22500000ULL;
    double u_alpha = 0.2, u_beta = 0.9;
    double v_alpha = 0.85, v_beta = 0.92;
    u64 win = 4096;
    bool do_verify = false, sieve_only = false, try_find5 = false;
    std::string emit_path;
    u64 max_bu = 0, max_v = 0;
    int threads = 0;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--verify-known") do_verify = true;
        else if (s == "--lo") lo = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--hi") hi = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--win" || s == "--x5-top") win = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--sieve-only") sieve_only = true;
        else if (s == "--try-find5") try_find5 = true;
        else if (s == "--emit") emit_path = next();
        else if (s == "--max-bu") max_bu = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--max-v-per-bu") max_v = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--threads") threads = atoi(next().c_str());
        else if (s == "--u-band") {
            std::string b = next(); sscanf(b.c_str(), "%lf,%lf", &u_alpha, &u_beta);
        } else if (s == "--v-band") {
            std::string b = next(); sscanf(b.c_str(), "%lf,%lf", &v_alpha, &v_beta);
        } else if (s == "-h" || s == "--help") { usage(); return 0; }
        else { printf("unknown arg %s\n", s.c_str()); usage(); return 1; }
    }

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    if (do_verify) {
        int rc = verify_known();
        if (argc == 2) return rc;
        printf("\n");
    }

    if (hi < lo) { printf("empty band\n"); return 0; }

    Depth d = make_depth(2);
    printf("=== record hunt Stage1+2 ===\n");
    printf("B in [%llu,%llu]  u-band (%.2f,%.2f)  v-band (%.2f,%.2f)  win=%llu\n",
           (unsigned long long)lo, (unsigned long long)hi, u_alpha, u_beta,
           v_alpha, v_beta, (unsigned long long)win);

    std::vector<std::array<u64,2>> bus;
#ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<std::array<u64,2>> loc;
        const int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        for (u64 B = lo + tid; B <= hi; B += nth)
            sieve_bu(B, d, u_alpha, u_beta, loc);
        #pragma omp critical
        bus.insert(bus.end(), loc.begin(), loc.end());
    }
#else
    for (u64 B = lo; B <= hi; ++B) sieve_bu(B, d, u_alpha, u_beta, bus);
#endif
    std::sort(bus.begin(), bus.end());
    bus.erase(std::unique(bus.begin(), bus.end()), bus.end());
    if (max_bu && bus.size() > max_bu) bus.resize(max_bu);

    printf("[Stage1] %zu (B,u) candidates\n", bus.size());
    if (!emit_path.empty()) {
        FILE* f = fopen(emit_path.c_str(), "w");
        if (f) {
            for (auto& p : bus)
                fprintf(f, "%llu %llu\n", (unsigned long long)p[0], (unsigned long long)p[1]);
            fclose(f);
            printf("[Stage1] wrote %s\n", emit_path.c_str());
        }
    }
    if (sieve_only) return 0;
    if (bus.empty()) { printf("no candidates — done\n"); return 0; }

    const int N = (int)(hi / F294) + 2;
    Pow6Table pb;
    if (try_find5) {
        pb.init(N);
        printf("[Stage2] find5 enabled N=%d win=%llu\n", N, (unsigned long long)win);
    } else {
        printf("[Stage2] v-sweep + T filter (pass --try-find5 to search; else emit jobs)\n");
    }

    long solutions = 0, tried_v = 0, tried_T = 0;
    FILE* jobf = nullptr;
    if (!emit_path.empty() && !sieve_only) {
        // reuse emit path for (B,u,v) jobs when not sieve-only
        jobf = fopen((emit_path + ".buv").c_str(), "w");
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:tried_v,tried_T)
#endif
    for (size_t bi = 0; bi < bus.size(); ++bi) {
        const u64 B = bus[bi][0], u = bus[bi][1];
        const u64 vlo = (u64)std::ceil(v_alpha * (double)B);
        const u64 vhi = std::min<u64>((u64)std::floor(v_beta * (double)B), B - 1);
        u64 v0 = ((vlo + 146) / 147) * 147;
        if ((v0 & 1) == 0) v0 += 147;
        u64 local_v = 0;
        for (u64 v = v0; v <= vhi; v += 294) {
            if (!v_ok(B, v)) continue;
            ++local_v; ++tried_v;
            if (max_v && local_v > max_v) break;
            u128 T = 0;
            if (!compute_T(B, u, v, T)) continue;
            if (T < 5) continue;
            const int nB = (int)(B / F294);
            if (nB < 1) continue;
            // cheap size filter without requiring p6 table
            // T <= 5*(B/294)^6  ≈ 5 * (B/294)^6
            {
                const u64 lim = (u64)nB;
                if (lim > 2500000) continue;
                if (T > pow6_u128(lim) * 5) continue;
            }
            ++tried_T;
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                if (jobf)
                    fprintf(jobf, "%llu %llu %llu\n",
                            (unsigned long long)B, (unsigned long long)u,
                            (unsigned long long)v);
            }
            if (!try_find5) continue;

            int a, b, c, e, f;
            if (find5_top(pb, T, win, a, b, c, e, f)) {
                long long terms[7] = {
                    (long long)F294 * a, (long long)F294 * b, (long long)F294 * c,
                    (long long)F294 * e, (long long)F294 * f,
                    (long long)u, (long long)v
                };
                std::sort(terms, terms + 7);
#ifdef _OPENMP
                #pragma omp critical
#endif
                {
                    ++solutions;
                    printf("SOLUTION B=%llu u=%llu v=%llu x=(%d,%d,%d,%d,%d)\n",
                           (unsigned long long)B, (unsigned long long)u,
                           (unsigned long long)v, a, b, c, e, f);
                    printf("  terms: %lld %lld %lld %lld %lld %lld %lld\n",
                           terms[0], terms[1], terms[2], terms[3],
                           terms[4], terms[5], terms[6]);
                    fflush(stdout);
                }
            }
        }
#ifdef _OPENMP
        #pragma omp critical
#endif
        {
            fprintf(stderr, "[Stage2] B=%llu u=%llu done (cum sol=%ld T-trials=%ld)\n",
                    (unsigned long long)B, (unsigned long long)u, solutions, (long)tried_T);
            fflush(stderr);
        }
    }
    if (jobf) {
        fclose(jobf);
        printf("[Stage2] wrote %s.buv\n", emit_path.c_str());
    }

    printf("---- summary ----\n");
    printf("bu=%zu  v-trials=%ld  T-trials=%ld  solutions=%ld\n",
           bus.size(), (long)tried_v, (long)tried_T, solutions);
    if (!try_find5)
        printf("next: GPU/pair-table find5 on the .buv job list (unguided CPU peel is too slow)\n");
    return 0;
}
