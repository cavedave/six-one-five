// =============================================================================
// record_probe.cpp  —  Experiment 1: the "constructibility" decision tool
//
// For high-B (6,1,7) solutions of the record type
//
//     B^6 = u^6 + v^6 + F^6 (x1^6 + x2^6 + x3^6 + x4^6 + x5^6),   F = 6*7^j,
//
// the whole search is feasible iff the *exceptional* term u exists cheaply.
// u is pinned by three congruences (the divisibility of the five inner terms
// by F, and of v by 3*7^j):
//
//     u^6 ≡ B^6 (mod 7^{6j})      [five inner terms and v carry 7^j -> 7^{6j}]
//     u^6 ≡ B^6 (mod 3^6)         [inner terms and v carry 3]
//     u  ≡ 0    (mod 2)           [u is the even, non-multiple-of-3 term]
//
// (The 2-adic condition on v, v ≡ ±B (mod 32), is a *constructive* v-condition
//  handled in the next stage, not here.)
//
// This tool does a CENSUS: over admissible B (gcd(B,42)=1) it counts how many
// carry a nontrivial u in a ratio band (alpha*B, beta*B).  The output is the
// hit-rate curve r_j(B) = P(a random admissible B has a usable u), which — set
// against the inner scale B/F_j our find5 must then solve — decides how far up
// in B each depth j can reach.
//
// Everything here is modular; B^6 (~1e44 at 22M) is never formed, so it stays
// exact in 128-bit.  The record identity check uses multi-prime fingerprints.
//
// Build:  g++ -O3 -std=c++17 -fopenmp -o record_probe record_probe.cpp
//   (drops to single-thread cleanly if -fopenmp is unavailable)
//
// Examples:
//   ./record_probe --verify-record
//   ./record_probe --lo 21934895 --hi 60000000 --sample 4000000 --j 2,3
//   ./record_probe --lo 100000000 --hi 100200000 --enumerate --j 2 --emit hits.txt
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

using u64  = uint64_t;
using u128 = unsigned __int128;

// ------------------------------------------------------------------ modmath --
static inline u128 mulmod(u128 a, u128 b, u128 m) {
    if (m <= (u128)UINT64_MAX) {          // a,b < m <= 2^64  =>  a*b < 2^128 exact
        return (a * b) % m;
    }
    a %= m; b %= m;                       // binary (double-and-add); our m < 2^68
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

// small (u64) modular sixth power, for the 3^6 and prime fingerprints
static inline u64 pow6_mod(u64 x, u64 m) {
    u64 r = 1 % m; x %= m;
    for (int i = 0; i < 6; ++i) r = (u64)((u128)r * x % m);
    return r;
}

// ------------------------------------------------------------- 7-adic setup --
struct Depth {
    int    j;            // 7-adic depth of the inner terms (F = 6*7^j)
    u128   m7;           // 7^{6j}
    u128   F;            // 6*7^j
    u128   zeta[7];      // zeta[a] = a-lifted 6th root of unity mod 7^{6j}, a=1..6
};

static Depth make_depth(int j) {
    Depth d;
    d.j  = j;
    d.m7 = ipow(7, 6 * j);
    d.F  = (u128)6 * ipow(7, j);
    // 6th roots of unity mod 7^{6j} are the Hensel lifts of 1..6 mod 7:
    //   zeta_a = a^{7^{6j-1}} (raises into the order-6 subgroup, keeps a mod 7).
    const u128 lift = ipow(7, 6 * j - 1);
    for (int a = 1; a <= 6; ++a) d.zeta[a] = powmod((u128)a, lift, d.m7);
    return d;
}

// admissible primitive Branch-A(code) B: coprime to 42 (odd, not /3, not /7)
static inline bool admissible(u64 B) { return (B & 1) && (B % 3) && (B % 7); }

// -------------------------------------------------------------- 3^6 filter ---
static constexpr u64 M3 = 729;   // 3^6

// -------------------------------------------------------- per-thread census --
struct Acc {
    u64 samples = 0;              // admissible B examined
    u64 hits    = 0;              // qualifying (B,u) pairs
    u64 bhit    = 0;              // admissible B with >=1 qualifying u
    std::array<u64, 20> ratio{};  // u/B histogram over [0,1) in 0.05 bins
    void operator+=(const Acc& o) {
        samples += o.samples; hits += o.hits; bhit += o.bhit;
        for (size_t i = 0; i < ratio.size(); ++i) ratio[i] += o.ratio[i];
    }
};

// count qualifying u for one admissible B at depth d; optionally record hits
static int census_one(u64 B, const Depth& d, double alpha, double beta,
                      Acc& acc, std::vector<std::array<u64,2>>* emit) {
    const u128 b7   = (u128)B % d.m7;
    const u64  b6_3 = pow6_mod(B, M3);
    const u64  lo   = (u64)std::ceil (alpha * (double)B);
    const u64  hi   = std::min<u64>((u64)std::floor(beta * (double)B), B - 1);
    if (lo > hi) return 0;
    int found = 0;
    for (int a = 2; a <= 6; ++a) {                 // 5 nontrivial 7-adic roots
        const u128 u7 = mulmod(b7, d.zeta[a], d.m7);  // residue of u mod 7^{6j}
        // first u >= lo with u ≡ u7 (mod m7)
        u128 first = (u128)lo + ((u7 + d.m7 - ((u128)lo % d.m7)) % d.m7);
        for (u128 u = first; u <= (u128)hi; u += d.m7) {
            const u64 uu = (u64)u;
            if (uu & 1) continue;                  // u even
            if (uu % 3 == 0) continue;             // u is the non-multiple-of-3 term
            if (uu % 7 == 0) continue;             // (never true for a nontrivial root)
            if (uu == B) continue;
            if (pow6_mod(uu, M3) != b6_3) continue;// u^6 ≡ B^6 (mod 3^6)
            ++found; ++acc.hits;
            int bin = (int)((double)uu / (double)B * 20.0);
            if (bin < 0) bin = 0; if (bin > 19) bin = 19;
            acc.ratio[bin]++;
            if (emit) emit->push_back({B, uu});
        }
    }
    if (found) ++acc.bhit;
    return found;
}

// ------------------------------------------------- analytic decision model ---
// Expected qualifying u per admissible B at depth j, scale B (band width w):
//   rate ≈ [5 * w * B / 7^{6j}]  *  (1/2)  *  (6/729)
// The bracket is the expected count of 7-adic roots landing in the band; the
// two factors are the u-even and u^6≡B^6 (mod 3^6) filters.
static double model_rate(int j, double B, double w) {
    const double m7 = std::pow(7.0, 6.0 * j);
    const double band7 = 5.0 * w * B / m7;
    return band7 * 0.5 * (6.0 / 729.0);
}

// ------------------------------------------------------------ record anchor --
static int verify_record() {
    // Meyrignac/known record (7 ∤ B):
    const u64 B  = 21934895ULL;
    const u64 a[7] = {19424433ULL, 19128522ULL, 13805652ULL, 9771972ULL,
                      8968470ULL, 8306212ULL, 5749170ULL};
    const u64 u  = 8306212ULL;   // exceptional: only term not divisible by 7 (and not by 3)
    const u64 v  = 19424433ULL;  // unique odd term (divisible by 147 = 3*49)

    printf("[record] B = %llu  (B mod 7 = %llu -> 7 %s B)\n",
           (unsigned long long)B, (unsigned long long)(B % 7),
           (B % 7) ? "does NOT divide" : "divides");

    // 1) full identity, verified via multi-prime fingerprints (B^6 overflows 128b)
    const u64 primes[4] = {2305843009213693951ULL /*2^61-1*/, 2305843009213693669ULL,
                           1000000000000000003ULL, 999999999999999989ULL};
    bool id_ok = true;
    for (u64 p : primes) {
        u64 lhs = 0; for (int i = 0; i < 7; ++i) lhs = (lhs + pow6_mod(a[i], p)) % p;
        if (lhs != pow6_mod(B, p)) id_ok = false;
    }
    printf("[record] identity  sum a_i^6 == B^6 : %s\n", id_ok ? "OK (4-prime)" : "FAIL");

    // 2) divisibility structure
    int nd49 = 0, nd294 = 0, nunit7 = 0, nodd = 0, nnot3 = 0;
    for (int i = 0; i < 7; ++i) {
        if (a[i] % 49 == 0)  ++nd49;
        if (a[i] % 294 == 0) ++nd294;
        if (a[i] % 7  != 0)  ++nunit7;
        if (a[i] & 1)        ++nodd;
        if (a[i] % 3  != 0)  ++nnot3;
    }
    printf("[record] structure : /49=%d(want6) /294=%d(want5) unit7=%d(want1) "
           "odd=%d(want1) not/3=%d(want1)\n", nd49, nd294, nunit7, nodd, nnot3);
    printf("[record] v=%llu : /147=%s  odd=%s   u=%llu : even=%s  not/3=%s  not/7=%s\n",
           (unsigned long long)v, (v % 147 == 0) ? "yes" : "no", (v & 1) ? "yes" : "no",
           (unsigned long long)u, (u % 2 == 0) ? "yes" : "no",
           (u % 3) ? "yes" : "no", (u % 7) ? "yes" : "no");

    // 3) the exceptional congruences on u
    bool c7  = true;  // u^6 ≡ B^6 mod 7^12
    { Depth d = make_depth(2); c7 = (powmod((u128)u, 6, d.m7) == powmod((u128)B, 6, d.m7)); }
    bool c3  = (pow6_mod(u, M3) == pow6_mod(B, M3));           // u^6 ≡ B^6 mod 3^6
    bool c3b = ((long long)(u % 729)) == ((729 - (long long)(B % 729)) % 729);  // u ≡ -B (mod 3^6)?
    printf("[record] u^6 ≡ B^6 (mod 7^12): %s | u^6 ≡ B^6 (mod 3^6): %s | u ≡ -B (mod 3^6): %s\n",
           c7 ? "OK" : "FAIL", c3 ? "OK" : "FAIL", c3b ? "yes" : "no");
    bool cv = ((v % 32) == (B % 32)) || ((v % 32) == (32 - B % 32) % 32);
    printf("[record] v ≡ ±B (mod 32): %s\n", cv ? "OK" : "FAIL");

    // 4) does our sieve (depth j=2) actually flag (B,u)?
    Depth d = make_depth(2);
    const u128 b7 = (u128)B % d.m7;
    int rootA = 0;
    for (int aa = 2; aa <= 6; ++aa)
        if (mulmod(b7, d.zeta[aa], d.m7) == (u128)u) { rootA = aa; break; }
    printf("[record] sieve@j=2 recovers u : %s%s\n",
           rootA ? "YES (7-adic root a=" : "NO",
           rootA ? (std::to_string(rootA) + ")").c_str() : "");
    printf("[record] u/B = %.4f  inner scale B/294 = %.0f\n",
           (double)u / (double)B, (double)B / 294.0);

    bool all = id_ok && nd49 == 6 && nd294 == 5 && nunit7 == 1 && nodd == 1 &&
               nnot3 == 1 && c7 && c3 && cv && rootA;
    printf("[record] %s\n", all ? "ALL CHECKS PASS" : "*** SOME CHECK FAILED ***");
    return all ? 0 : 1;
}

// ------------------------------------------------------------------- driver --
static void usage() {
    printf("usage: record_probe [--verify-record] [--lo B --hi B]\n"
           "                    [--sample N | --enumerate] [--j 2,3,4]\n"
           "                    [--band a,b] [--seed s] [--emit file] [--threads t]\n");
}

// splitmix64 for cheap per-thread admissible-B sampling
static inline u64 sm64(u64& s) {
    u64 z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

int main(int argc, char** argv) {
    u64 lo = 21934895ULL, hi = 60000000ULL, sample = 4000000ULL, seed = 12345ULL;
    double alpha = 0.2, beta = 0.9;
    std::vector<int> js = {2, 3, 4};
    bool enumerate = false, do_record = false;
    std::string emit_path;
    int threads = 0;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (s == "--verify-record") do_record = true;
        else if (s == "--lo")        lo = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--hi")        hi = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--sample")    sample = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--enumerate") enumerate = true;
        else if (s == "--seed")      seed = strtoull(next().c_str(), nullptr, 10);
        else if (s == "--emit")      emit_path = next();
        else if (s == "--threads")   threads = atoi(next().c_str());
        else if (s == "--band") { std::string b = next(); sscanf(b.c_str(), "%lf,%lf", &alpha, &beta); }
        else if (s == "--j") {
            js.clear(); std::string b = next(); size_t p = 0;
            while (p < b.size()) { js.push_back(atoi(b.c_str() + p)); size_t c = b.find(',', p); if (c == std::string::npos) break; p = c + 1; }
        } else if (s == "-h" || s == "--help") { usage(); return 0; }
        else { printf("unknown arg: %s\n", s.c_str()); usage(); return 1; }
    }

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    if (do_record) { int rc = verify_record(); printf("\n"); if (argc == 2) return rc; }

    if (hi <= lo) { printf("nothing to do (hi <= lo)\n"); return 0; }

    printf("=== 7-adic exceptional-root census ===\n");
    printf("B band [%llu, %llu]   ratio band (%.2f, %.2f)   %s   depths j=",
           (unsigned long long)lo, (unsigned long long)hi, alpha, beta,
           enumerate ? "ENUMERATE" : "SAMPLE");
    for (int j : js) printf("%d ", j);
    printf("\n");
    const double Bmid = 0.5 * ((double)lo + (double)hi);
    const double w    = beta - alpha;

    for (int j : js) {
        Depth d = make_depth(j);
        Acc total;
        std::vector<std::array<u64,2>> emit;   // only used if emit_path set (j-tagged below)
        const bool want_emit = !emit_path.empty();

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            Acc loc;
            std::vector<std::array<u64,2>> loc_emit;
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
            const int nth = omp_get_num_threads();
#else
            const int tid = 0, nth = 1;
#endif
            if (enumerate) {
                for (u64 B = lo + tid; B <= hi; B += nth) {
                    if (!admissible(B)) continue;
                    ++loc.samples;
                    census_one(B, d, alpha, beta, loc, want_emit ? &loc_emit : nullptr);
                }
            } else {
                u64 st = seed + 0x1000 * (u64)(j) + 0x9999 * (u64)tid + 1;
                const u64 span = hi - lo + 1;
                const u64 quota = sample / (u64)nth + 1;
                for (u64 k = 0; k < quota; ++k) {
                    u64 B = lo + sm64(st) % span;
                    B |= 1ULL;                              // odd
                    while (!admissible(B)) B += 2;
                    if (B > hi) continue;
                    ++loc.samples;
                    census_one(B, d, alpha, beta, loc, want_emit ? &loc_emit : nullptr);
                }
            }
#ifdef _OPENMP
            #pragma omp critical
#endif
            { total += loc; if (want_emit) emit.insert(emit.end(), loc_emit.begin(), loc_emit.end()); }
        }

        const double rate  = total.samples ? (double)total.hits / (double)total.samples : 0.0;
        const double brate = total.samples ? (double)total.bhit / (double)total.samples : 0.0;
        const double model = model_rate(j, Bmid, w);
        const double inner = Bmid / (double)d.F;

        printf("\n--- depth j=%d   F=%llu=6*7^%d   7^{6j}=%.4g ---\n",
               j, (unsigned long long)d.F, j, (double)d.m7);
        printf("  admissible B examined : %llu\n", (unsigned long long)total.samples);
        printf("  qualifying (B,u) hits : %llu   (u per admissible B = %.3e)\n",
               (unsigned long long)total.hits, rate);
        printf("  admissible B with u   : %llu   (fraction = %.3e)\n",
               (unsigned long long)total.bhit, brate);
        printf("  measured rate / model : %.3e / %.3e  (ratio %.2f)\n",
               rate, model, model > 0 ? rate / model : 0.0);
        printf("  inner scale B/F @ mid : %.4g   (find5 target ~ this^6)\n", inner);
        if (rate > 0) {
            const double W = 7.0 / (2.0 * rate);   // raw-B width to expect ~1 qualifying B
            printf("  raw-B width per hit   : %.3e  (scan this many B for ~1 usable u)\n", W);
        }
        if (total.hits) {
            printf("  u/B histogram (0.05 bins, only band %.2f-%.2f populated):\n   ", alpha, beta);
            for (int b = 0; b < 20; ++b) if (total.ratio[b]) printf("[%.2f:%llu]", b * 0.05, (unsigned long long)total.ratio[b]);
            printf("\n");
        }
        if (want_emit && !emit.empty()) {
            std::string path = emit_path + ".j" + std::to_string(j);
            FILE* f = fopen(path.c_str(), "w");
            if (f) {
                for (auto& e : emit) fprintf(f, "%llu %llu\n",
                                             (unsigned long long)e[0], (unsigned long long)e[1]);
                fclose(f);
                printf("  emitted %zu (B,u) pairs -> %s\n", emit.size(), path.c_str());
            }
        }
    }

    // decision table: analytic rate & inner scale across a few B checkpoints
    printf("\n=== decision table (analytic rate per admissible B; inner scale B/F) ===\n");
    printf("%14s", "B");
    for (int j : js) printf("   j=%d rate    j=%d inner", j, j);
    printf("\n");
    const double chk[] = {2.2e7, 5e7, 1e8, 3e8, 1e9, 3e9, 1e10};
    for (double B : chk) {
        printf("%14.3g", B);
        for (int j : js) {
            Depth d = make_depth(j);
            printf("   %.3e   %.3e", model_rate(j, B, w), B / (double)d.F);
        }
        printf("\n");
    }
    printf("\nread: pick (j,B) with rate high enough to find B by scanning, AND\n"
           "inner scale small enough for find5 (full pair table ~ up to a few e5).\n");
    return 0;
}
