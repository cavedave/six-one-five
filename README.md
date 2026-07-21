# six-one-five: a GPU search for a₁⁶+a₂⁶+a₃⁶+a₄⁶+a₅⁶ = B⁶

A CUDA meet-in-the-middle solver for the Diophantine equation

```
a₁⁶ + a₂⁶ + a₃⁶ + a₄⁶ + a₅⁶ = B⁶        ("(6,1,5)" in Lander–Parkin–Selfridge notation)
```

**Headline result:** no primitive solution exists with **B ≤ 2,200,000**
(extension run to B ≤ 2,353,973 — the 128-bit arithmetic ceiling — in progress;
update this line when the final summary lands).
The previous published bound was B ≤ 730,000 (Resta & Meyrignac, 2002) [^1^].
No solution of (6,1,5) is known at all; if one exists, it is a counterexample
to Euler's sum-of-powers conjecture for k=6, and the Lander–Parkin–Selfridge
conjecture (k = m + n) would need k > 6 terms to fail [^1^].

## Result summary

| Item | Value |
|---|---|
| Range searched (this run) | B ∈ [1,010,231, 2,200,000], all classes |
| Prior coverage (same project, CPU) | B ∈ [0, 1,010,231] |
| Independent published coverage | B ≤ 730,000 (Resta–Meyrignac) [^1^] |
| Result | **0 solutions** |
| Probes executed | 4.60 × 10¹³ |
| Sustained probe rate | 1.42 × 10¹⁰ probes/s |
| GPU kernel time | 53.8 min (55 min wall) |
| Fingerprint false positives | 2,264 — all rejected by exact CPU verification, 0 missed |
| Hardware | NVIDIA RTX PRO 6000 Blackwell (188 SMs, 102 GB), 24 CPU threads |

Expectation value from the heuristic density (κ-law) for this range is
~1.6 × 10⁻⁴ solutions, so 0 is the statistically expected outcome; the
deliverable is the exclusion record.

## The seven tricks that make it fast

### 1. Only primitive B — and B must be coprime to 42

Sixth powers mod 8, 9, 7 are all in {0, 1} (odd²≡1 mod 8; φ(9)=φ(7)=6).
If 2 | B then B⁶ ≡ 0 (mod 8), so the number of odd aᵢ is ≡ 0 (mod 8);
with only five terms that means **all** aᵢ are even — a non-primitive solution.
The same counting works mod 9 (3 | B ⇒ all aᵢ divisible by 3) and mod 7.
Non-primitive solutions are d⁶-scalings of smaller primitive ones, so:

> **It suffices to test B with gcd(B, 42) = 1.** This discards 5/7 of all B
> values *by proof*, before any GPU time is spent. Coverage of the remaining
> 2/7 is complete, not heuristic search.

(The same counting, run forwards with gcd(B,42)=1, forces **exactly one** aᵢ
to be odd, exactly one to be coprime to 3, and exactly one to be coprime to 7
— which is the foundation of the class decomposition below.)

### 2. The five Meyrignac classes

The three "unique" roles (the odd term, the ¬3 term, the ¬7 term) can sit on
the same variable or be split across variables. Partitioning by how they are
distributed gives five classes; within each class the *pair* variables are
guaranteed to be divisible by 42, 14, 21, or 7 respectively:

| Class | Pair variables divisible by | Residue classes | Master modulus |
|---|---|---|---|
| 1 | 42 | 144 | 42⁶ = 5,489,031,744 |
| 2 | 14 | 24 | 14⁶ |
| 3 | 21 | 36 | 21⁶ |
| 4, 5 | 7 | 6 each | 7⁶ |

Running `classes = 1,2,3,4,5` ("all") is a **complete partition** — every legal
solution lies in exactly one class.

### 3. Seed variables ≤ B/42 instead of variables ≤ B

Because pair variables are divisible by 42/14/21/7 inside their class, write
them as 42c (resp. 14c, 21c, 7c): the seed c only needs c ≤ B/42. The
residues modulo the master modulus are enumerated once from precomputed
unit-root tables (mod 2⁶: 8 residues × 4 roots; mod 3⁶: 81 × 6; …).
This is what collapses the pair-sum table from (B)² pairs to (B/42)² pairs —
a factor of 42² ≈ 1764 in table size.

### 4. Meet-in-the-middle: a 68.7 GB fingerprint hash table

All ordered pair sums `cᵢ⁶ + cⱼ⁶ (mod 2⁶⁴)` for 1 ≤ cᵢ ≤ cⱼ ≤ B/42 are
inserted into an open-addressing table of 2³² slots (16 B/slot → 68.7 GB;
load factor 0.32; average insert probe length 1.23). For B_max = 2.2M that is
1.372 × 10⁹ pairs, built once in ~78 s on 24 CPU threads and uploaded to the
GPU in ~2 s. Duplicate fingerprints occupy multiple slots; `payload == 0`
marks empties.

The 64-bit value is a **fingerprint**, a bloom like filter not a proof — collisions are expected
(2,264 occurred) and every hit is re-verified exactly (trick 6). A ribbon data structure would be about a tenth the size but not implemented yet.

### 5. GPU probe engine at 1.4 × 10¹⁰ lookups/s

For each candidate (B, u, pair-seeds c₃, c₄) the kernel scans c₅ and computes

```
probe = ( B⁶ − u⁶ − (c₃⁶ + c₄⁶) − c₅⁶ ) mod 2⁶⁴
```

one hash lookup per probe. B⁶ itself is kept in signed 128-bit integers
(B ≤ 2,353,973 keeps B⁶ < 2¹²⁷ — the hard ceiling of this implementation);
everything else is u64 arithmetic mod 2⁶⁴. The c₅ loop bound keeps every
intermediate residual non-negative, so no intermediate exceeds B⁶.
Sustained rate on the RTX PRO 6000 Blackwell: **1.42 × 10¹⁰ probes/s**.

### 6. Exact verification — the hash never decides anything

Every fingerprint hit is re-checked on the CPU in full 128-bit exact
arithmetic. In the campaign: 2,264 hits, 2,264 false positives, 0 misses.
Sanity cross-check: observed false-hit rate ≈ 4.9 × 10⁻¹¹/probe matches the
theoretical table density (1.37 × 10⁹ fingerprints in 2⁶⁴ ≈ 7.4 × 10⁻¹¹)
to order of magnitude — the lookup layer behaves as modeled.

### 7. A validation chain, but there is more to check

* `--selftest`: host number theory (sixth roots, modular inverses via extended
  Euclid) + 20,000 planted pair sums + 3 × 512 planted class solutions driven
  through the **full** GPU pipeline (table → kernel → hit → verify). All found.
* `--xcheck`: an independent CPU reference engine (`QuadSumIndex`, sorted
  pair array) re-computes a slice and diffs against the GPU engine:
  `[300000, 350000]` → `CPU-found-but-GPU-missed=0 PASS`.
* The production campaign launches only if the xcheck gate passes
  (chained in `runs/run_campaign.sh`).
* Independent published overlap: Resta–Meyrignac covered B ≤ 730,000 [^1^];
  our joint range subsumes theirs.

## Engineering notes

* **Toolkit gap:** the card used is sm_120 (Blackwell) but CUDA 12.4 predates sm_120
  support, so the binary targets `compute_90` PTX and the driver JIT-compiles
  it on first launch (small one-time pause; slightly pessimistic codegen).
  With CUDA ≥ 12.8, build natively for sm_120 instead.
* **Build:**
  ```
  nvcc -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 -o solve_516_v3 solve_516_v3.cu -lineinfo
  ```
* **Validate then run:**
  ```
  ./solve_516_v3 --selftest
  ./solve_516_v3 300000 350000 all --xcheck
  ./solve_516_v3 1010231 2200000 all
  ```
* Long runs: wrap with `nohup ... & disown` (see `runs/run_campaign.sh`);
  logs land in `runs/`.

## Roadmap

1. **Mod-124,488 probe gate.** Pair-sum achievability mod 13 (5/13) and
   mod 19 (10/19), combined with the existing 504 = 8·9·7 structure, gives a
   124,488-entry bitmap (15.2 KiB) that kills 98.9% of probe streams before
   they touch the table — a projected ~90× probe-stage speedup.
2. **Beyond 2¹²⁷:** CRT over 2–3 64-bit primes (or 192-bit limbs) lifts the
   B ceiling past 2.35M; with trick 1 the next milestone is B = 10⁷.
3. **Ports of the same engine** (k=6 machinery is class-generic):
   **(6,2,4)** — sum of four sixth powers = sum of two — is *open* and has
   exactly this MITM shape (targets become pair sums instead of B⁶−u⁶);
   **(6,1,6)** is also open (nothing known; a 1967 bound of z > 38,300 is the
   only published exclusion [^2^][^3^]) but costs one more free variable.
4. **Near-miss logging:** threshold the residual stream to tabulate record
   `|a₁⁶+…+a₅⁶ − B⁶|` minima — a byproduct worth publishing on its own.

## References

[^1^]: G. Resta, J.-C. Meyrignac, "The smallest solutions to the Diophantine equation a⁶+b⁶+c⁶+d⁶+e⁶ = x⁶+y⁶", *Math. Comp.* 72 (2003), 1054–1057 (bound B ≤ 730,000 for (6,1,5) quoted via the Lander–Parkin–Selfridge conjecture page, https://en.wikipedia.org/wiki/Lander,_Parkin,_and_Selfridge_conjecture).

[^2^]: T. Piezas, "Timeline of Euler's Extended Conjecture", https://www.oocities.org/titus_piezas/Timeline1.htm — (6,1,6): no solutions known; Lander et al. (1967) exclude z ≤ 38,300; (7,1,7) solved by Dodrill (1999); (8,1,8) by Chase (2000); (9,1,9) open.

[^3^]: E. W. Weisstein, "Diophantine Equation — 6th Powers" (CRC Concise Encyclopedia of Mathematics), https://archive.lib.msu.edu/crcmath/math/math/d/d229.htm — "No solutions are known to the 6-1 or 6-2 equations."

[^4^]: L. J. Lander, T. R. Parkin, J. L. Selfridge, "A Survey of Equal Sums of Like Powers", *Math. Comput.* 21 (1967), 446–459, doi:10.1090/S0025-5718-1967-0222008-0.

[^5^]: S. Braun, "A fourth primitive solution to a⁵+b⁵+c⁵+d⁵ = e⁵" (MITM + modular filtering, 10.5M vCPU-hours), arXiv:2603.05549 (2026) — methodological sibling of this work.
