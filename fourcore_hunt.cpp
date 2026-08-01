// =============================================================================
// fourcore_hunt.cpp  —  Stage 1 for (6,1,5) four-core branch
//
//   B^6 = u^6 + D^6 (x1^6 + x2^6 + x3^6 + x4^6)
//
// with D = 42*h (default D=42). Primitive-compatible:
//   gcd(B,42)=1, u odd, gcd(u,42)=1, u^6 ≡ B^6 (mod D^6).
//
// Emits job lines:  B u T_lo T_hi
// where T = (B^6 - u^6)/D^6 fits in u128 (GMP on host).
//
// Build:
//   g++ -O3 -std=c++17 -fopenmp -I$(brew --prefix gmp)/include \
//       -L$(brew --prefix gmp)/lib -o fourcore_hunt fourcore_hunt.cpp -lgmpxx -lgmp
//
// Examples:
//   ./fourcore_hunt --selftest
//   ./fourcore_hunt --lo 2353974 --hi 2355000 --emit runs/fc_smoke.but
//   ./fourcore_hunt --lo 2353974 --hi 3500000 --D 42 --emit runs/fc_42_2p35_3p5.but
// =============================================================================

#include "fourcore_gmp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr long long M2 = 64;
static constexpr long long M3 = 729;
static constexpr long long M7 = 117649;

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

// Sixth-root classes for u with u^6 ≡ B^6 (mod 42^6), unit mod 42.
// For general D=42*h we still require the 42-part; extra factors of h are
// enforced by computing T exactly (must be divisible by D^6).
static std::vector<long long> seeds_u_mod42(const RootTables& rt, long long B) {
    const long long b2 = mod_pow6(B, M2), b3 = mod_pow6(B, M3), b7 = mod_pow6(B, M7);
    std::vector<long long> out;
    for (long long s2 : rt.r2[b2])
        for (long long s3 : rt.r3[b3]) {
            const long long s23 = crt2(s2, M2, s3, M3);
            for (long long s7 : rt.r7[b7]) out.push_back(crt2(s23, M2 * M3, s7, M7));
        }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static bool admissible_B(u64 B) { return std::gcd((long long)B, 42LL) == 1; }
static bool unit_u(u64 u) { return (u & 1) && std::gcd((long long)u, 42LL) == 1; }

static constexpr long long M42 = 5489031744LL;  // 42^6

static void collect_u(const RootTables& rt, u64 B, double ulo, double uhi,
                      std::vector<u64>& out) {
    if (!admissible_B(B)) return;
    const u64 lo = std::max<u64>(1, (u64)std::ceil(ulo * (double)B));
    const u64 hi = std::min<u64>(B - 1, (u64)std::floor(uhi * (double)B));
    if (lo > hi) return;
    for (long long s : seeds_u_mod42(rt, (long long)B)) {
        long long u0 = s % M42;
        if (u0 < 0) u0 += M42;
        if ((u64)u0 < lo) {
            const long long k = (lo - u0 + M42 - 1) / M42;
            u0 += k * M42;
        }
        for (long long u = u0; (u64)u <= hi; u += M42) {
            if (unit_u((u64)u)) out.push_back((u64)u);
        }
    }
}

static int selftest() {
    printf("=== fourcore_hunt selftest (GMP) ===\n");
    // i128 boundary: B=2353973 has B^6 just under 2^127; next needs GMP.
    const u64 B0 = 2353973ULL;
    mpz_class b6 = mpz_pow6(B0);
    const long bits0 = (long)mpz_sizeinbase(b6.get_mpz_t(), 2);
    printf("[1] B=%llu B^6 bits=%ld (expect <=127)\n",
           (unsigned long long)B0, bits0);
    if (bits0 > 127) { printf("FAIL bits\n"); return 1; }

    const u64 B1 = 2353974ULL;
    mpz_class b61 = mpz_pow6(B1);
    const long bits1 = (long)mpz_sizeinbase(b61.get_mpz_t(), 2);
    printf("[2] B=%llu B^6 bits=%ld (expect >=128)\n",
           (unsigned long long)B1, bits1);
    if (bits1 < 128) { printf("FAIL bits\n"); return 1; }

    // Plant: pick x_i, u, D=42, form B^6 = u^6 + 42^6 sum x^6 — only if perfect sixth.
    // Instead check T round-trip on a random admissible pair from Stage-1.
    RootTables rt;
    std::vector<u64> us;
    // Find a B near 2.4M with at least one u
    u64 Btest = 0, utest = 0;
    for (u64 B = 2400001; B < 2401000; ++B) {
        us.clear();
        collect_u(rt, B, 0.05, 0.99, us);
        if (!us.empty()) { Btest = B; utest = us[0]; break; }
    }
    if (!Btest) { printf("FAIL no Stage1 candidate near 2.4M\n"); return 1; }
    u128 T = 0;
    if (!compute_T_gmp(Btest, utest, 42, T)) {
        printf("FAIL compute_T B=%llu u=%llu\n",
               (unsigned long long)Btest, (unsigned long long)utest);
        return 1;
    }
    printf("[3] Stage1+T ok  B=%llu u=%llu T_bits~%d\n",
           (unsigned long long)Btest, (unsigned long long)utest,
           (int)(T ? 64 + (T >> 64 ? 64 : 0) : 0));

    // Synthetic verify: 1^6+2^6+2^6+2^6 is not a useful plant; check identity
    // u^6 + 42^6 * T == B^6 via GMP for the pair above (T definition).
    {
        mpz_class lhs = mpz_pow6(utest) + mpz_pow6(42) * mpz_class(0);
        // export T into mpz
        mpz_class Tm;
        u64 halves[2];
        split_u128(T, halves[0], halves[1]);
        mpz_import(Tm.get_mpz_t(), 2, -1, sizeof(u64), 0, 0, halves);
        lhs = mpz_pow6(utest) + mpz_pow6(42) * Tm;
        if (lhs != mpz_pow6(Btest)) { printf("FAIL T identity\n"); return 1; }
    }
    printf("[4] GMP T identity holds\n");
    printf("SELFTEST PASS\n");
    return 0;
}

static void usage() {
    printf("usage: fourcore_hunt [--selftest] [--D d] [--lo a --hi b]\n"
           "                     [--u-band lo,hi] [--emit file.but]\n"
           "  job line: B u T_lo T_hi\n"
           "  default D=42; recommended start lo=2353974\n");
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IOLBF, 0);

    u64 D = 42, lo = 0, hi = 0;
    double ulo = 0.0, uhi = 1.0;
    std::string emit;
    bool do_self = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
        };
        if (a == "--selftest") do_self = true;
        else if (a == "--D") D = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--lo") lo = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--hi") hi = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--u-band") {
            std::string s = next();
            if (sscanf(s.c_str(), "%lf,%lf", &ulo, &uhi) != 2) {
                fprintf(stderr, "bad --u-band\n"); return 1;
            }
        } else if (a == "--emit") emit = next();
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown %s\n", a.c_str()); usage(); return 1; }
    }

    if (do_self || argc == 1) return selftest();
    if (!lo || !hi || lo > hi) { usage(); return 1; }
    if (D % 42 != 0) {
        fprintf(stderr, "[warn] D=%llu is not a multiple of 42; Stage-1 still uses mod 42^6 seeds\n",
                (unsigned long long)D);
    }

    RootTables rt;
    FILE* out = stdout;
    if (!emit.empty()) {
        out = fopen(emit.c_str(), "w");
        if (!out) { perror(emit.c_str()); return 1; }
    }

    u64 nB = 0, nJobs = 0, nFailT = 0;
    fprintf(stderr, "[hunt] D=%llu B=[%llu,%llu] u-band=[%.3f,%.3f)\n",
            (unsigned long long)D, (unsigned long long)lo, (unsigned long long)hi, ulo, uhi);

    for (u64 B = lo; B <= hi; ++B) {
        if (!admissible_B(B)) continue;
        ++nB;
        std::vector<u64> us;
        collect_u(rt, B, ulo, uhi, us);
        for (u64 u : us) {
            u128 T = 0;
            if (!compute_T_gmp(B, u, D, T)) { ++nFailT; continue; }
            u64 tlo, thi; split_u128(T, tlo, thi);
            fprintf(out, "%llu %llu %llu %llu\n",
                    (unsigned long long)B, (unsigned long long)u,
                    (unsigned long long)tlo, (unsigned long long)thi);
            ++nJobs;
        }
        if ((B - lo) % 10000 == 0) {
            fprintf(stderr, "[progress] B=%llu jobs=%llu failT=%llu\n",
                    (unsigned long long)B, (unsigned long long)nJobs,
                    (unsigned long long)nFailT);
        }
    }

    fprintf(stderr, "[done] eligible_B=%llu jobs=%llu failT=%llu\n",
            (unsigned long long)nB, (unsigned long long)nJobs, (unsigned long long)nFailT);
    if (out != stdout) fclose(out);
    return 0;
}
