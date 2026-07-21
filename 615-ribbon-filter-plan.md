# Ribbon-Filter Variant of solve_516_v3 — Design Plan

Status: **plan for analysis — no code yet**. Drop-in replacement of the pair-sum table in
`solve_516_v3.cu` with a static probabilistic membership filter. Everything upstream
(candidate generation, class congruences, windows) and everything downstream (exact 128-bit
verification, dedup, solution printing) is **unchanged**.

---

## 1. The idea in one paragraph

The v3 pair table stores, for each of the n = 1.372×10⁹ pair sums
s(i,j) = (c_i⁶ + c_j⁶) mod 2⁶⁴, a 16-byte slot: 8-byte key + 4-byte payload (i,j) + padding,
in a 2³²-slot open-addressing table = 68.7 GB. Two observations:

1. **The payload is redundant.** Given a residual v that truly is a pair sum, the pair (i,j)
   can be re-derived on demand by an O(N) search over the 52,380 precomputed sixth powers
   (~1 ms). Hits are rare (≲10³ per campaign), so paying 1 ms per hit is free.
2. **The keys don't need to be exact.** If the store answers "definitely not present" always
   correctly and "maybe present" with probability 2⁻ʳ, then every real solution still gets
   found (its residual *is* a pair sum, hence *is* in the store), and the ~2⁻ʳ·P false
   alarms get killed by the re-derivation search (which finds no pair) or by the existing
   exact verification (which rejects the wrong pair).

A **ribbon filter** is the best known static structure for exactly this: ~r+2 bits per key,
**one** memory access per query, zero false negatives, false-positive rate ≈ 2⁻ʳ.
With r = 48: **8.6 GB instead of 68.7 GB, ~1 expected false alarm per campaign.**

This is your "reuse old landing spots" intuition, formalized: the landing spots (pair sums)
are computed once and reused by all ~3×10¹⁴ probes; the filter is a compressed directory of
landing spots that never loses one.

---

## 2. Why ribbon and not Bloom / xor / sorted-array

| Structure | Bits/key | Size @ n=1.372×10⁹ | Reads per query | Notes |
|---|---|---|---|---|
| Bloom filter, FPR 2⁻⁴⁸ | ~69 | 11.9 GB | **~33** | k = r/ln2 dependent reads — hopeless on GPU |
| Xor filter (Graf–Lemire 2020) | 1.23·r ≈ 59 | 10.1 GB | 3 | simple construction; 3× the memory traffic of ribbon |
| **Ribbon filter (BuRR-style)** | **~r+2 ≈ 50** | **8.6 GB** | **1** | near-optimal space; construction is the hard part |
| Sorted exact keys + binary search | 64 | 11.0 GB | ~31 (dependent) | deterministic; bandwidth-prohibitive |
| Current open-addressing table | 400 (incl. LF slack) | 68.7 GB | ~1.5 | the incumbent |

Per-probe memory traffic: incumbent ≈ 1.5 × 16 B = 24 B; ribbon ≈ 8–16 B (one unaligned
64-bit window). So the filter is not only 8× smaller — it should be **bandwidth-neutral to
~2× faster** per probe. The guaranteed win is capacity; speed is a plausible bonus.

References (construction algorithms only): Dillinger & Walzer, "Ribbon filter: practically
smaller than Bloom and Xor" (arXiv:2103.02515); Dillinger, Hübschle-Schneider, Sanders,
Walzer, "Fast Succinct Retrieval and Approximate Membership Using Ribbon" (BuRR,
arXiv:2109.01892); Graf & Lemire, "Xor Filters" (arXiv:1912.08258).

---

## 3. What does NOT change

- Candidate generation (`gen_class_cands`), the five class congruences, all window math.
- The probe *input*: the same 64-bit fingerprint `fp = funnel_fp(R)` the kernels compute today.
- `verify_hit` (exact 128-bit subtractive verification) — reused untouched.
- `--selftest`, `--xcheck`, `--bench` harnesses — extended, not replaced.
- Table persistence (`--save-table` / `--load-table`) — same pattern, smaller file.

Only the interior of `probe()` and the hit-record format change.

---

## 4. Assumptions

- **A1 — Key universe.** Keys are all s(i,j) = (c_i⁶ + c_j⁶) mod 2⁶⁴, 1 ≤ i < j ≤ N,
  N = ⌊B_max/42⌋ = 52,380 at B_max = 2.2×10⁶ → n = C(N,2) = 1,371,806,010 keys.
  (The true sums are 95-bit; we store the 64-bit residue, exactly as v3 does — the final
  verification recomputes everything in 128-bit, so no information is lost.)
- **A2 — Probe count.** Total campaign probes P ≈ 3×10¹⁴ (class 5 ≈ 80%). This is an
  *estimate from the design model*; `--bench` will measure the real rate, and the FP budget
  below is parametric in P so it can be re-derived with measured numbers.
- **A3 — Static set.** The key universe depends only on B_max. The filter is built once,
  saved to disk, reused for the whole campaign, exactly like the current table.
- **A4 — No false negatives.** Ribbon filters never answer "absent" for an inserted key.
  This is a *property of the construction*, not a probability — it will nonetheless be
  tested exhaustively on the host (all 1.372×10⁹ keys queried back, must be 100% positive)
  and on device (planted-candidate selftest).
- **A5 — Fingerprint budget.** With r = 48 and P = 3×10¹⁴: expected false positives
  E[FP] = P·2⁻⁴⁸ ≈ 1.1 per campaign. Each FP costs ~1 ms (re-derivation finds no pair).
  Even r = 32 (E[FP] ≈ 7×10⁴, ~70 s of overhead) would work; r = 48 is comfort margin
  for bigger future campaigns.
- **A6 — Rare true hits.** Expected true solutions in [730k, 2.2M] ≈ 2×10⁻⁴. The hit
  pipeline exists for the lottery ticket; its cost is irrelevant, only its correctness matters.
- **A7 — Hash decorrelation.** Pair-sum keys are arithmetically structured (sixth powers
  mod 2⁶ are 0/1, so sums mod 64 ∈ {0,1,2}; similar structure mod 9, mod 7). A strong
  64-bit mixer (splitmix64 finalizer) is applied to every key before deriving position /
  coefficients / fingerprint. Uniformity is validated empirically (§9).
- **A8 — Duplicate keys.** Expected 64-bit collisions among the n keys ≈ n²/2⁶⁵ ≈ 0.051.
  Duplicates are harmless to a membership filter (idempotent insert) and are *enumerated*
  at re-derivation time (§7), so colliding pairs can no longer shadow each other. See §8,
  R4: this actually **repairs a latent completeness gap in v3**.
- **A9 — Host resources.** Build machine: 4 cores, 283 GiB RAM. Peak construction RAM
  ≈ keys (11 GB) + filter (9 GB) + bookkeeping < 40 GB. Fine.

---

## 5. Component 1 — key generation (host, one-time)

```cpp
// K[i] = c_i^6 mod 2^64, i = 1..N            (52,380 x 8 B = 0.4 MB, kept forever)
// keys: stream all pairs i<j -> key = mix64(K[i] + K[j])
std::vector<u64> gen_keys(int N) {
    std::vector<u64> K(N+1);
    for (i = 1..N) K[i] = pow6_64(i);              // already have this
    std::vector<u64> keys;  keys.reserve(n);        // 11 GB
    #pragma omp parallel for
    for (i = 1..N)
        for (j = i+1..N)
            keys.push_back(mix64(K[i] + K[j]));     // mixed NOW, once
    return keys;                                     // ~seconds on 4 cores
}
```

No sort, no dedup (A8). Keys are mixed at birth so every downstream stage sees
near-uniform 64-bit values (A7).

---

## 6. Component 2 — filter construction (host, one-time)

Target structure (ribbon, w = 64): a bit array F of m ≈ n·(r+overhead) bits. Each key x
defines a start position p(x) ∈ [0, m−w) and a w-bit coefficient vector c(x) with c(x)[0]=1
(forces a unit-diagonal banded system → solvable by back-substitution). F is the solution of

    for every key x:  parity( F[p(x) .. p(x)+w)  AND  c(x) )  =  fp_r(x)

where fp_r(x) is an r-bit fingerprint derived from x.

```cpp
Ribbon build_ribbon(const std::vector<u64>& keys, int r /*=48*/, int w /*=64*/) {
    m = n * (r + 2) /*bits, ~2% slack + bump room*/;
    // 1. assign each key:  p = h1(x) % (m - w),  c = 1 | (h2(x) << 1),  fp = h3(x) & (2^r-1)
    // 2. bucket-sort keys by p                        (counting sort, 2 passes over 11 GB)
    // 3. sweep a w-wide window left->right, Gaussian-eliminating the banded system on the fly
    //    (standard ribbon construction; each step is a 64-bit word XOR pivot)
    // 4. keys whose row becomes dependent during the sweep (~tiny fraction) are "bumped"
    //    to a small overflow filter with a new hash seed (BuRR-style)
    // 5. emit F, seed, r, w, m, plus overflow structure
}
```

Pseudocode is deliberately at block level — the sliding-window elimination is the one
genuinely intricate component (the rest is plumbing), and §10 stages it behind the simpler
xor-filter build so the pipeline is validated before we touch it.

Fallback / stepping stone (xor filter, r = 48, 10.1 GB, 3 reads/query):

```cpp
XorF build_xor(const std::vector<u64>& keys, int r) {
    for (seed = 0;; ++seed) {
        size = 1.23 * n;  // three blocks
        // peel: count how many keys touch each cell; repeatedly remove cells of degree 1,
        // stacking (key, cell). If peeling stalls -> retry with next seed (rare).
        // assign fingerprints in reverse peel order so every key's XOR of its 3 cells
        // equals fp_r(x).  Construction succeeds w.h.p. at 1.23x load.
    }
}
```

---

## 7. Component 3 — device query + hit re-derivation

Device query (replaces the open-addressing walk inside `probe()`; signature unchanged):

```cpp
__device__ bool ribbon_might_contain(const u64* F, u64 m_bits, u64 x_mixed) {
    u64 p = h1(x_mixed) % (m_bits - W);          // start bit position
    u64 c = 1ULL | (h2(x_mixed) << 1);           // 64-bit coefficients, LSB = 1
    u64 z = load_bits64(F, p);                   // 1 unaligned 64-bit read (<=2 words)
    return (__popcll(z & c) & 1) == (u32)(h3(x_mixed) & FP_MASK);   // r-bit fingerprint
}
```

Kernel integration: everywhere v3 does `probe(P, fp, ci, v3, v4)` and records a
`Hit{cand, a, b, c, d}`, the variant does `if (ribbon_might_contain(F, fp))
record({cand, v3, v4})` — the hit record **shrinks from 20 B to 12 B** because the pair
(a,b) is no longer known on device.

Host hit processing (new — this is where the pair is recovered):

```cpp
void process_hit(cand, v3, v4) {
    u64 v = host_funnel_fp(cand, v3, v4);        // same math as device, already validated
                                                 // in v3_host_logic_test.cpp
    for (i = 1..N) {                              // 52,380 iterations ~ 1 ms
        u64 need = v - K[i];                      // exact 64-bit subtraction
        for (j : multimap_lookup(need))           // hash multimap: residue -> indices
            if (j > i) verify_hit(cand, i, j, v3, v4);   // EXISTING exact 128-bit path
    }
}
```

- `multimap_lookup` is a one-time hash map from K[j] → list of j (52,380 entries, ~50 MB).
- A **false positive** (filter said maybe, but v is no pair sum): the loop finds nothing —
  cost 1 ms, discarded.
- A **fingerprint collision** (v is the residue of *other* pairs too): all such pairs are
  enumerated and exact-verified; the wrong ones fail in 128-bit, the right one survives.
- A **true hit**: pair recovered, exact-verified, deduped, printed — identical to v3.

---

## 8. Correctness argument

**Completeness (exact, not probabilistic).** Let a real solution exist with terms
42·(c1..c5), unit u, target B. Its probe computes residual v = ((B⁶−u⁶)/42⁶ − c3⁶−c4⁶)
mod 2⁶⁴ = (c1⁶+c2⁶) mod 2⁶⁴ — which **is** one of the inserted keys. By A4 the filter
answers "maybe present" for every inserted key, so the kernel records the hit; §7 then
enumerates *all* pairs with that residue — including (c1,c2) — and `verify_hit` confirms it
in 128-bit. ∎ The filter can raise the hit count; it can never lower it.

**Soundness.** Anything printed passed `verify_hit` (exact 128-bit subtractive identity,
gcd, ordering — unchanged from v3). The filter only decides what gets *checked*, never what
gets *accepted*.

**Bonus: repairs a latent v3 gap.** v3's table stores one payload per key; if two distinct
pairs share a 64-bit residue (expected count over the whole table ≈ 0.051, A8), the second
pair is shadowed, and a solution using it would verify-fail against the first pair and be
lost — probability per solution ~10⁻¹⁰, but nonzero. The variant enumerates all pairs at
re-derivation time, so completeness is exact rather than 1−10⁻¹⁰.

---

## 9. Hash spec & uniformity validation

```
mix64(x)  = splitmix64 finalizer (x ^= x>>30; x *= 0xBF58476D1CE4E5B9; ... )
h1,h2,h3  = mix64 seeded with three independent constants (position / coefficients / fingerprint)
```

Because sixth-power sums are lattice-structured (A7), uniformity is *tested*, not assumed:
- chi-square on h1 over a 10⁷-key sample (bucket count ≈ 2¹⁶);
- coefficient weight distribution (should be ~Binomial(63,½) shifted by the forced 1);
- measured device FPR during `--bench` on a run with no planted solutions must equal
  2⁻⁴⁸ within sampling error. Any of these failing → swap in a different mixer (cost: rebuild).

---

## 10. Risks & mitigations

- **R1 — Ribbon construction complexity.** The sliding-window elimination is the only
  intricate piece; a bug there silently produces false negatives (violating A4).
  *Mitigation:* stage 0 builds the **xor filter first** (simple peeling, ~1 day of work),
  runs the full validation pipeline on it, and only then swaps in ribbon. The exhaustive
  host re-query test (all n keys must be positive) catches any construction bug before
  the GPU ever runs.
- **R2 — Construction failure (dependent rows).** Standard ribbon at w=64 with ~2% slack
  + BuRR-style bumping succeeds w.h.p.; xor filter retries with a new seed. Both are
  "retry until success" at build time, never a runtime issue.
- **R3 — FP storm if r is too small or P is underestimated.** Budget is recomputed from
  the `--bench`-measured probe rate before committing to r; r=48 tolerates P up to
  ~10¹⁶ (≈ 35 FPs) — 30× headroom over the A2 estimate.
- **R4 — Shadowed-pair regression risk: none.** The variant is strictly *more* complete
  than v3 (§8).
- **R5 — Memory-bandwidth assumption.** The "~2× faster" guess assumes the kernel is
  DRAM-bound; if it turns out latency/occupancy-bound, speed is merely neutral. Capacity
  win is unconditional.
- **R6 — Quadratic growth past 2.2M.** The filter shrinks memory 8× but n still grows as
  B_max²: the memory wall moves from ≈2.4M to ≈5M B (B_max=5M → n=7.1×10⁹ → ~44 GB,
  still fits 96 GB; B_max=10M → 177 GB, does not). Beyond that needs pair-set partitioning
  (multiple filters, multiple passes) — out of scope here.
- **R7 — Build time.** Sort-free key streaming + elimination over 1.372×10⁹ keys on 4
  cores: estimate 3–15 min (BuRR-class builders do ~10⁸ keys/s/core). Acceptable; a GPU
  builder is a later optimization, not a prerequisite.

---

## 11. Validation plan (mirrors the v3 strategy)

1. **Host filter test** (standalone, like `v3_host_logic_test.cpp`): build over all n keys;
   re-query all n → must be 100% positive (A4); query 10⁹ random non-keys → positive rate
   must be 2⁻⁴⁸ within noise; run re-derivation on 10⁵ random true keys → must recover
   the exact pair set every time.
2. **`--selftest` (device)**: existing plant tests re-pointed at the filter — 512 planted
   candidates per class must all produce hits and verify.
3. **`--xcheck` (CPU-vs-GPU, B ≤ 600k)**: the variant and v2's CPU finders must produce
   identical solution sets on identical candidate streams. PASS criterion unchanged:
   CPU-found-but-GPU-missed = 0.
4. **FPR field measurement** during `--bench` (no plants): positives/probes ≈ 2⁻⁴⁸.

---

## 12. Alternatives considered and rejected (for the record)

- **Payload-carrying retrieval filter** (ribbon *retrieval*, 16–24 fingerprint bits +
  32-bit payload (i,j) ≈ 10 GB): recovers the pair on-device with no re-derivation, but
  requires dedup at build, reintroduces shadowing (one payload per key), and its smaller
  fingerprint (2⁻¹⁶…2⁻²⁴) means 10⁷–10¹⁰ false retrievals per campaign hammering the CPU
  verifier. Membership + re-derivation is cleaner and strictly more complete.
- **Xor filter as final structure**: 3 reads/query ≈ 24 B/probe — same traffic as the
  incumbent, so capacity win only. Kept as the staging build (R1).
- **Plain Bloom**: ~33 dependent reads/query — disqualifying on GPU.
- **Sorted exact keys**: 11 GB, deterministic, but ~31 dependent reads/query.
- **Sharding the campaign by B and rebuilding smaller filters**: n depends on the global
  N = B_max/42, not on the shard — doesn't help (see R6 for the real scaling story).

---

## 13. Milestones

1. **M0** — host key-gen + xor-filter build + exhaustive re-query test (§11.1) on CPU.
2. **M1** — device xor query wired into a copy of the kernels; selftest + xcheck green.
3. **M2** — ribbon builder (the R1 component); passes the same §11.1 suite.
4. **M3** — swap ribbon in, re-run selftest/xcheck/bench; measure FPR and probes/s.
5. **M4** — persistence format (`--save-table` writes filter + header {seed,r,w,m,N}).
6. **M5** — campaign.

Total new code ≈ 600 lines host + ~40 lines device. v3 stays untouched as the oracle.

---

## 14. Open questions to poke at

- Is P = 3×10¹⁴ (A2) right? `--bench` on v3 answers this before M2 commits to r.
- Is one unaligned 64-bit read per probe actually faster than the LF-0.32 table walk on the
  RTX PRO 6000? M3's bench decides whether ribbon is also a *speed* upgrade or "only" 8× capacity.
- Do we want the overflow/bump structure (BuRR) or plain ribbon with seed-retry? Slight
  space vs. build-simplicity trade.
- Worth gating M2 on a real need? The xor filter (M1) already delivers a validated 6.8×
  capacity win; ribbon adds the last 1.2× plus the single-read query. If xor benches well,
  ribbon is optional polish.
