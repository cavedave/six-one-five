// Host xor-filter tests (M1).
// Build: make xor-test   (from repo root)
//
// Checks:
//   1) no false negatives on all inserted keys
//   2) FPR on random non-keys ≈ 2^{-r} (within generous sampling bounds)
//   3) shard helpers partition stably
//   4) ribbon stub still links (permissive)

#include "ribbon_filter.hpp"
#include "shard.hpp"
#include "store_header.hpp"
#include "xor_filter.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

static int g_fails = 0;

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_fails;
    }
}

static std::vector<std::uint64_t> make_keys(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::unordered_set<std::uint64_t> seen;
    std::vector<std::uint64_t> keys;
    keys.reserve(n);
    while (keys.size() < n) {
        std::uint64_t k = rng();
        if (seen.insert(k).second) keys.push_back(k);
    }
    return keys;
}

static void test_tiny() {
    const std::vector<std::uint64_t> keys = {1, 42, 999, 123456789ULL, 0xdeadbeefcafeULL};
    const XorFilter f = build_xor(keys, 16);
    expect(store615::header_ok(f.hdr), "tiny: header_ok");
    expect(f.hdr.kind == static_cast<std::uint32_t>(store615::Kind::Xor), "tiny: kind");
    for (std::uint64_t k : keys) {
        expect(xor_might_contain(f, k), "tiny: FN on inserted key");
    }
    (void)xor_might_contain(f, 0x1111222233334444ULL);  // non-key query must not crash
}

static void test_no_fn_and_fpr(std::size_t n, int r, std::size_t n_probe) {
    const auto keys = make_keys(n, 0xC0FFEEULL + n + static_cast<std::size_t>(r));
    std::unordered_set<std::uint64_t> keyset(keys.begin(), keys.end());
    const XorFilter f = build_xor(keys, r);
    expect(f.hdr.n_keys == n, "size: n_keys");

    std::size_t fn = 0;
    for (std::uint64_t k : keys) {
        if (!xor_might_contain(f, k)) ++fn;
    }
    expect(fn == 0, "FN count must be 0");
    if (fn) std::fprintf(stderr, "  FN=%zu / %zu\n", fn, n);

    std::mt19937_64 rng(0xF15EEDULL + n);
    std::size_t fp = 0, trials = 0;
    while (trials < n_probe) {
        std::uint64_t k = rng();
        if (keyset.count(k)) continue;
        ++trials;
        if (xor_might_contain(f, k)) ++fp;
    }
    const double rate = static_cast<double>(fp) / static_cast<double>(trials);
    const double expect_rate = std::ldexp(1.0, -r);
    // Allow wide band: Poisson / hashing noise. Require rate < 8 * 2^{-r}
    // and (if enough expected FPs) rate > 2^{-r} / 8.
    const double lo = expect_rate / 8.0;
    const double hi = expect_rate * 8.0;
    const bool ok_hi = rate <= hi || fp <= 3;  // tiny absolute counts OK
    const bool ok_lo = (expect_rate * static_cast<double>(trials) < 8.0) || rate >= lo;
    expect(ok_hi && ok_lo, "FPR within 8x of 2^{-r}");
    std::printf("  n=%zu r=%d cells=%llu FN=%zu FPR=%.3e (expect ~%.3e) fp=%zu/%zu\n", n, r,
                (unsigned long long)f.hdr.m_cells, fn, rate, expect_rate, fp, trials);
}

static void test_duplicates_ok() {
    std::vector<std::uint64_t> keys = {7, 7, 8, 8, 8, 9};
    const XorFilter f = build_xor(keys, 20);
    expect(f.hdr.n_keys == 3, "dedup to 3 keys");
    expect(xor_might_contain(f, 7), "dup: 7");
    expect(xor_might_contain(f, 8), "dup: 8");
    expect(xor_might_contain(f, 9), "dup: 9");
}

static void test_shard_partition() {
    const std::uint32_t S = 8;
    const std::uint64_t seed = 0x615;
    std::size_t counts[8] = {};
    for (std::uint64_t k = 0; k < 8000; ++k) {
        const std::uint32_t s = shard_of_key(k * 0x9E3779B97F4A7C15ULL, seed, S);
        expect(s < S, "shard id in range");
        counts[s]++;
    }
    for (std::uint32_t s = 0; s < S; ++s) {
        expect(counts[s] > 500 && counts[s] < 1500, "shard load roughly even");
    }
}

static void test_ribbon_stub() {
    const std::vector<RibbonKey> keys = {1, 2, 3};
    const RibbonFilter f = build_ribbon(keys, 16);
    int pass = 0;
    for (RibbonKey k : keys)
        if (ribbon_might_contain(f, k)) ++pass;
    expect(pass == (int)keys.size(), "ribbon stub permissive");
}

// E0: real pair-sum keys k_ij = i^6+j^6 (mod 2^64), same as campaign store.
static void test_real_pair_keys(int N, int r) {
    std::vector<std::uint64_t> pw6((size_t)N + 1, 0);
    for (int x = 1; x <= N; ++x) {
        const std::uint64_t x2 = (std::uint64_t)x * (std::uint64_t)x;
        pw6[(size_t)x] = x2 * x2 * x2;
    }
    std::vector<std::uint64_t> keys;
    keys.reserve((size_t)N * (size_t)(N + 1) / 2);
    for (int i = 1; i <= N; ++i)
        for (int j = i; j <= N; ++j) keys.push_back(pw6[(size_t)i] + pw6[(size_t)j]);

    const XorFilter f = build_xor(keys, r);
    std::size_t fn = 0;
    for (std::uint64_t k : keys)
        if (!xor_might_contain(f, k)) ++fn;
    expect(fn == 0, "real-pair: FN count must be 0");
    const double packed = f.store_gb() * 1e9;
    const double u64eq = f.unpacked_u64_gb() * 1e9;
    expect(f.packed.size() == xor_packed_bytes(f.hdr.m_cells, f.hdr.r), "packed byte count");
    expect(packed < u64eq * 0.9 || r >= 64, "packed smaller than u64 cells when r<64");
    std::printf("  real-pair N=%d keys=%zu r=%d cells=%llu FN=%zu packed=%.3f MB "
                "(u64-equiv=%.3f MB, ratio=%.3f)\n",
                N, keys.size(), r, (unsigned long long)f.hdr.m_cells, fn, packed / 1e6,
                u64eq / 1e6, u64eq > 0 ? packed / u64eq : 0.0);
}

static void test_pack_size_table() {
    std::printf("  pack size model (logical cells = 1.23*n):\n");
    for (int N : {2000, 10000, 47857, 65535, 71428}) {
        const double n = (double)N * (N + 1) / 2.0;
        const std::size_t cells = xor_detail::capacity_for((std::size_t)n);
        for (int r : {32, 40, 48}) {
            const double packed = xor_packed_gb(cells, (std::uint32_t)r);
            const double u64 = cells * 8.0 / 1e9;
            std::printf("    N=%-5d r=%-2d  packed=%.2f GB  u64=%.2f GB  save=%.0f%%\n", N, r,
                        packed, u64, 100.0 * (1.0 - packed / u64));
        }
    }
}

static void test_peel_retry_keeps_keys() {
    // MEM-6 frees keys after incidence; a peel stall needs a fresh key vector.
    // Outer loop mirrors xor_build_pairs regeneration.
    XorFilter f;
    bool ok = false;
    for (int outer = 0; outer < 8 && !ok; ++outer) {
        std::vector<std::uint64_t> keys;
        keys.reserve(5000);
        for (std::size_t i = 0; i < 5000; ++i) keys.push_back(0xDEADBEEF00000000ULL + i);
        try {
            const std::uint64_t base =
                0x615615615615615ULL + (std::uint64_t)outer * 0x9E3779B97F4A7C15ULL;
            f = build_xor(std::move(keys), 16, base, /*max_seed_tries=*/1);
            ok = true;
        } catch (const std::runtime_error&) {
            // peel stall after MEM-6 — try next seed base with regenerated keys
        }
    }
    expect(ok, "peel: succeeded within outer retries");
    expect(f.hdr.n_keys == 5000, "peel: n_keys");
    expect(!f.packed.empty(), "peel: packed non-empty");
    expect(xor_might_contain(f, 0xDEADBEEF00000000ULL), "peel: FN on first key");
}

int main() {
    std::printf("xor_filter_test: starting\n");
    test_tiny();
    test_duplicates_ok();
    test_peel_retry_keeps_keys();
    test_shard_partition();
    test_ribbon_stub();

    // Medium sets: r=16 → expect ~1.5e-5; 2e6 probes → ~30 FP expected
    test_no_fn_and_fpr(/*n=*/10'000, /*r=*/16, /*n_probe=*/2'000'000);
    test_no_fn_and_fpr(/*n=*/100'000, /*r=*/16, /*n_probe=*/2'000'000);
    // Higher rank, fewer probes — just check no FN + no FP storm
    test_no_fn_and_fpr(/*n=*/50'000, /*r=*/32, /*n_probe=*/500'000);

    // E0 extension: real (i^6+j^6) keys — packing + r sweep.
    // N=200 first (lean MEM smoke); N=2000 catches packing at campaign r values.
    test_real_pair_keys(/*N=*/200, /*r=*/16);
    test_real_pair_keys(/*N=*/200, /*r=*/48);
    test_real_pair_keys(/*N=*/2000, /*r=*/48);
    test_real_pair_keys(/*N=*/2000, /*r=*/40);
    test_real_pair_keys(/*N=*/2000, /*r=*/32);
    test_pack_size_table();

    if (g_fails) {
        std::fprintf(stderr, "xor_filter_test: %d FAIL(s)\n", g_fails);
        return 1;
    }
    std::printf("xor_filter_test: ALL PASS\n");
    return 0;
}
