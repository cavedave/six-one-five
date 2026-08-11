# six-one-five: GPU searches for equal sums of sixth powers

## New (6,1,7) primitives found by this project

Nine new primitive solutions of \(a_1^6+\cdots+a_7^6=B^6\) (2nd kind: \(7\nmid B\)),
found with `solve_617_v1` Branch A. None appear in the prior EulerNet clean list
(previous max there was \(B=410455\)).

| \(B\) | Class | Identity (sorted terms) |
|---:|:---:|---|
| **400471** | cls5 | \(13860^6 + 83202^6 + 93240^6 + 129696^6 + 144756^6 + 355502^6 + 357609^6\) |
| **421663** | cls5 | \(85407^6 + 112560^6 + 231294^6 + 258888^6 + 296394^6 + 306102^6 + 392630^6\) |
| **423601** | cls4 | \(6720^6 + 8610^6 + 11473^6 + 36834^6 + 98994^6 + 344778^6 + 400014^6\) |
| **425003** | cls5 | \(38997^6 + 224130^6 + 260526^6 + 279454^6 + 282702^6 + 352800^6 + 369348^6\) |
| **425155** | cls5 | \(32067^6 + 84000^6 + 130914^6 + 278208^6 + 317506^6 + 324198^6 + 384894^6\) |
| **425729** | cls5 | \(8512^6 + 53466^6 + 119658^6 + 184338^6 + 293106^6 + 355551^6 + 385014^6\) |
| **427027** | cls5 | \(62132^6 + 66276^6 + 188685^6 + 189378^6 + 201540^6 + 352884^6 + 397992^6\) |
| **428195** | cls3 | \(128226^6 + 129213^6 + 169828^6 + 246792^6 + 290052^6 + 340158^6 + 394338^6\) |
| **428329** | cls5 | \(51450^6 + 156681^6 + 161100^6 + 257964^6 + 275478^6 + 342384^6 + 395038^6\) |

Each row satisfies \(\sum a_i^6 = B^6\) with \(\gcd(a_1,\ldots,a_7,B)=1\).

### Recombination of found relations

Taking known equal sums of sixth powers as generators and combining them
(pairwise / short triples with cancellation) yields further identities. This is
bookkeeping on the catalog, not a smaller search for (6,1,5). A short run over
198 generators (`recombination_report.md`) found three frontier improvements
(verify failures: 0), including:

| Form | Identity |
|:---|:---|
| (6,6,6) | \(130326^6+93240^6+87486^6+72254^6+26481^6+3486^6\) \(=\) \(121800^6+106260^6+96145^6+92400^6+54600^6+40950^6\) |
| (6,7,7) | \(244463^6+8598^6+7884^6+7601^6+6882^6+5496^6+4482^6\) \(=\) \(241962^6+138159^6+127176^6+105700^6+75894^6+13104^6+10031^6\) |
| (6,8,8) | \(10073^6+8598^6+7884^6+7601^6+6882^6+5844^6+5496^6+4482^6\) \(=\) \(10031^6+8858^6+7719^6+7584^6+6750^6+6000^6+5142^6+1122^6\) |

The same pass also lists many longer \((6,6,12)\) triples (six terms cancel
between signed generators). Full lists and overlap stats: `recombination_report.md`.

---

## (6,1,5) — five terms = one sixth power

A CUDA meet-in-the-middle solver for the Diophantine equation

```
a₁⁶ + a₂⁶ + a₃⁶ + a₄⁶ + a₅⁶ = B⁶        ("(6,1,5)" in Lander–Parkin–Selfridge notation)
```

**Headline result:** no primitive solution found in any Meyrignac class through
the bounds below (prior published bound: B ≤ 730,000, Resta & Meyrignac 2002
[^1^]). No solution of (6,1,5) is known at all; if one exists, it is a
counterexample to Euler's sum-of-powers conjecture for k=6, and the
Lander–Parkin–Selfridge conjecture (k = m + n) would need k > 6 terms to fail
[^1^].

### The three search branches (Meyrignac classes)

Every admissible primitive candidate (`gcd(B,42)=1`) falls in exactly one of
five Meyrignac classes. The class is fixed by **which factors of 42 divide
which terms** (the odd / not-divisible-by-3 / not-divisible-by-7 roles).
Operationally we run them as **three** pipelines (classes 2–4 share machinery):

| Branch | Classes | Forced divisibility (besides unit \(u\)) | Unit \(u\) |
|---|---|---|---|
| **÷42 four-core** | **cls1** | four terms each divisible by **42** | odd and \(\gcd(u,42)=1\) (144 seeds mod \(42^6\)) |
| **one free ×14 / ×21 / ×7** | **cls2–4** | three terms ÷**42**; free term ÷**14** (cls2), ÷**21** (cls3), or ÷**7** (cls4) | cls2: odd, divisible by 3 but not 9 (24 seeds mod \(14^6\)); cls3: even, \(\gcd(u,21)=1\) (36 mod \(21^6\)); cls4: divisible by 6 but not 7 (6 mod \(7^6\)) |
| **two frees ×21 and ×14** | **cls5** | two terms ÷**42**; frees ÷**21** and ÷**14** | divisible by 6 but not 7 (6 seeds mod \(7^6\)) |

Equivalently: after writing cores as \(42c_i\) (and frees as \(14d\), \(21e\), …),
each class reduces to a smaller equal-sum search on the seeds. Full residue
tables are under “The five Meyrignac classes” below.

**Classes partition solution shapes, not \(B\).** Every eligible \(B\)
(\(\gcd(B,42)=1\), about \(2/7\) of all integers) is searched in **all five**
classes: for each class we ask “if a solution at this \(B\) had this role
pattern, which units \(u\) (and frees) are allowed?” Your example is right —
the same \(B=4101101\) is run through cls5 *and* cls2 *and* …; a true hit
lands in exactly one class. Near \(B\sim3\)M, Stage‑1 **units per eligible \(B\)**
are roughly cls1 ~0.06, cls2 ~2, cls3 ~0.5, cls4 ~25, cls5 ~25 — so about
**~0.1% / 4% / 1% / 48% / 48%** of units (not of \(B\)). Almost every eligible
\(B\) has cls4 and cls5 work; cls1 is sparse.

### Coverage cleared (0 solutions)

Eligible \(B\) only (`gcd(B,42)=1`). Bounds are inclusive of the search windows run.

> **WARNING — cls5 production binary**
>
> For post-i128 **class 5**, always use **`fourcore_cls5_gpu_v4`** (or `_v3`):
> ```bash
> ./fourcore_hunt_v3 --lo LO --hi HI --classes 5 --emit-units runs/….buc
> ./fourcore_cls5_gpu_v4 --units runs/….buc --r 48
> # or prebuilt table (recommended past ~3.2M):
> ./fourcore_cls5_gpu_v4 --units runs/….buc --N K --load-table runs/xor_NK_r48.bin
> ```
> Do **not** use `fourcore_find_v4 --stream-cls5` for production. That path expands
> the \((d,e)\) grid on the **host** and is ~**1000× slower** (weeks vs minutes on
> the same band). `--stream-cls5` is debug/count-only. See `search_density_and_rates.md`.

| Branch | Cleared through | Notes |
|---|---|---|
| **All five classes** | **B ≤ 2,353,973** | `solve_516_v3` (i128 ceiling on \(B^6\)) |
| **cls2–4** | **B ≤ 3,000,000** | post-wall `fourcore_find_v4` (also logged through 2,752,470 as `fc_v3_234_*`) |
| **cls5** | **B ≤ 3,500,000** | Post-wall **`fourcore_cls5_gpu_v4`** + lean xor builder (`MEM-1..4`) and **`--load-table`**: 2.36→2.75 (~16 min), 2.75→3.0, 3.0→3.2, then **3.2→3.5** (`fc_cls5_gpu_3200001_3500000` — **done**, 0 sols, ~22 min GPU with prebuilt `xor_N83333_r48.bin`). Host peel at \(N=83333\) peaks ~**132 GiB** RSS (~23 min); packed store ~**26 GB** on disk. Next band 3.5→4.0M needs \(N=95238\), ~**34 GB** packed (~**40 GB** free disk to save). |
| **cls1** | **B ≤ 3,680,000** | post-wall `fourcore_hunt` + `fourcore_find4` (`D=42`); can push higher than cls2–5 because find4 allows larger \(N\) |

| Item | Value |
|---|---|
| Independent published coverage | B ≤ 730,000 (Resta–Meyrignac) [^1^] |
| Result in all cleared bands above | **0 solutions** |
| Hardware | NVIDIA RTX PRO 6000 Blackwell (188 SMs, ~98–102 GB) |

### Cls5 xor storage (disk + host RAM)

One packed table covers cls5 through \(B_{\max}\approx 42N\) (\(N=\lfloor B/42\rfloor\)).
Build with `xor_build_save --N K --r 48 --out runs/xor_NK_r48.bin`, then GPU
`--load-table` (no host peel on the search job). Allow **packed size + ~5 GB**
free disk when saving; delete the previous `.bin` if space is tight.

| Target \(B_{\max}\) | \(N\) | Packed `.bin` | Free disk (save) | Lean build peak RAM | Fits 102 GB GPU? |
|---:|---:|---:|---:|---:|:---:|
| 3.5M | 83333 | ~26 GB | ~31 GB | ~132 GB | yes |
| 4.0M | 95238 | ~34 GB | ~40 GB | ~**135 GB** (MEM-5) | yes |
| 4.5M | 107142 | ~42 GB | ~47 GB | ~220 GB | yes |
| **5.0M** | 119047 | ~**52 GB** | ~**60 GB** | ~**145 GB** (MEM-5) | yes |
| 5.04M (soft cap) | 120000 | ~53 GB | ~60 GB | ~146 GB | yes |
| **10M** | ~238095 | ~**210 GB** | — | ~**575 GB** | **no** |

**Practical disk today:** with ~28 GB free after clearing old tables, add about
**30 GB more** (~**60 GB total free**) to build and save the **5 M** table in one
shot. Stepping 3.5→4.0→4.5→5.0M needs only one table on disk at a time if you
delete the previous `.bin` after each band clears.

**10 M:** a single xor table does **not** fit current hardware (VRAM, host RAM,
or practical disk). Needs **pair-space shards** and/or **ribbon** (roadmap) —
not “more disk” alone. Soft code cap today is \(N\le120{,}000\) (`kXorNSoftMax`).

### (6,2,4) coverage

Primitive \(a^6+b^6=c^6+d^6+e^6+f^6\) (Resta cases A+B), max left term \(a\).

| Range | Result | Notes |
|---|---|---|
| \(a\le 30{,}400\) | **0 sols** | Matches Resta–Meyrignac published clear; local PRO 6000 ~957 s |
| \(a\in(30400,\;79100]\) | **0 sols** | Vast.ai `624_v2_*` campaign (checkpointed) |
| \(a\in[79100,\;100000]\) | **0 sols** | PRO 6000 `solve_624_v1`, `CLEARED` in **15744 s (~4.4 h)**, `seedN=16035` |
| \(a\in(100000,\;120000]\) | **0 sols** | `six-one-five-624-v2` `solve_624_v1`, `seedN=19242`, `CLEARED` in **28566 s (~7.9 h)**, `xor_N19242_r48.bin` (~1.3 GB) |

Published prior: no primitive solutions with \(a\le30400\) (Resta–Meyrignac).



## (6,1,6) — six terms, one side

Parallel search for

```
a₁⁶ + a₂⁶ + a₃⁶ + a₄⁶ + a₅⁶ + a₆⁶ = B⁶
```

using `solve_616_v1.cu` (same table/gate machinery as above, five Meyrignac
classes for six terms). EulerNet covered **B ≤ 110,266** (Jan 2000); new ground
starts at 110,267.

| Item | Value |
|---|---|
| Range cleared (this project) | B ∈ [110,267, **950,000**] |
| Result so far | **0 solutions** |
| Status | **Search ongoing** |

Build: `make v616`. Run: `./solve_616_v1 110267 530001 all` (see
`solve_616_v1.cu` header for options). cls5 uses a triple-sum window gate
(`--no-tri-gate` for A/B); see `cpuplan.md` for planned CPU/cache optimizations.

## (6,1,7) — seven terms, one side

Search for

```
a₁⁶ + a₂⁶ + a₃⁶ + a₄⁶ + a₅⁶ + a₆⁶ + a₇⁶ = B⁶
```

with `solve_617_v1.cu` (sibling of the 616 solver):

- **Branch A** (\(7\nmid B\)): GPU Meyrignac classes 1–5 (42-scaled seeds + find4/find5)
- **Branch B** (\(7\mid B\)): separate path (CPU MITM / limited GPU); not used in the
  overnight strips below

| Item | Value |
|---|---|
| Known list (EulerNet clean, prior) | 178 solutions, \(B \le 410455\) |
| **New primitives (this project)** | **7** — see table at top (all 2nd kind) |
| Branch A strip cleared | through \(B \approx 430000\) (from ~400k) |
| Solver | `make v617` → `./solve_617_v1` |

Typical overnight Branch A launch (use `--chunk 64` near 400k+ to bound host RAM):

```bash
stdbuf -oL -eL nohup ./solve_617_v1 425001 430000 all --branch-a-only \
  --chunk 64 \
  > runs/617_a_425k_430k.log 2>&1 &
```

Validate with `--selftest` and `--check-known --known-file 617-solutions-clean.txt`.
See `solve_617_v1.cu` header and `617plan.md`. Record-style hunt near \(B\sim 22\text{M}\)
is a separate tool path (`record_hunt` / `record_find5`).

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
* **Build:** `make v3` (or `make v616` / `make v617`). See `Makefile`.
* **Validate then run:**
  ```
  ./solve_516_v3 --selftest
  ./solve_516_v3 300000 350000 all --xcheck
  ./solve_516_v3 1010231 2200000 all
  ```
  Use `--no-gate` to disable the mod-124,488 pre-filter for A/B benchmarking.
* Long runs: wrap with `nohup ... & disown` (see `runs/run_campaign.sh`);
  logs land in `runs/`.

### 8. Mod-124,488 probe gate (v3, enabled by default)

Before each hash-table lookup, kernels test whether the target residue is
achievable as a pair sum `i⁶+j⁶` mod 124,488 (= 8·9·7·13·19, CRT-factored as
504×247). Only ~1.08% of candidates pass; the rest die in ~20 ALU cycles.
Sound by construction — real pair sums always pass. The same gate ships in
`solve_616_v1.cu` for the (6,1,6) port.

## Roadmap

1. **Ribbon filter** (`615-ribbon-filter-plan.md`, spike in `spike/ribbon/`):
   shrink the 68.7 GB pair table to ~8.6 GB with one read per query.
2. **Beyond 2¹²⁷:** CRT over 2–3 64-bit primes (or 192-bit limbs) lifts the
   B ceiling past 2.35M; with trick 1 the next milestone is B = 10⁷.
3. **Ports of the same engine** (k=6 machinery is class-generic):
   **(6,1,6)** — `solve_616_v1.cu` (in progress; see section above);
   **(6,1,7)** — `solve_617_v1.cu` (Branch A producing new primitives; see above);
   **(6,2,4)** — `solve_624_v1.cu` / `make v624` (Resta Case A=find2, Case B=find3;
   shared `fourcore_find_device.cuh` + xor). **Cleared primitive \(a\le120000\)** (0 sols);
   extending toward 150 k+. Use v2 seed \(N=\lfloor 2^{1/6}\texttt{hi}/7\rfloor\) and
   matching `--load-table`. Host: `./solve_624_v1_host --cpu --lo LO --hi HI --case all`.
   See coverage table above and `six-one-five-624-v2/README_VAST.txt`.
4. **Near-miss logging:** threshold the residual stream to tabulate record
   `|a₁⁶+…+a₅⁶ − B⁶|` minima — a byproduct worth publishing on its own.

## References

[^1^]: G. Resta, J.-C. Meyrignac, "The smallest solutions to the Diophantine equation a⁶+b⁶+c⁶+d⁶+e⁶ = x⁶+y⁶", *Math. Comp.* 72 (2003), 1054–1057 (bound B ≤ 730,000 for (6,1,5) quoted via the Lander–Parkin–Selfridge conjecture page, https://en.wikipedia.org/wiki/Lander,_Parkin,_and_Selfridge_conjecture).

[^2^]: T. Piezas, "Timeline of Euler's Extended Conjecture", https://www.oocities.org/titus_piezas/Timeline1.htm — (6,1,6): no solutions known; Lander et al. (1967) exclude z ≤ 38,300; (7,1,7) solved by Dodrill (1999); (8,1,8) by Chase (2000); (9,1,9) open.

[^3^]: E. W. Weisstein, "Diophantine Equation — 6th Powers" (CRC Concise Encyclopedia of Mathematics), https://archive.lib.msu.edu/crcmath/math/math/d/d229.htm — "No solutions are known to the 6-1 or 6-2 equations."

[^4^]: L. J. Lander, T. R. Parkin, J. L. Selfridge, "A Survey of Equal Sums of Like Powers", *Math. Comput.* 21 (1967), 446–459, doi:10.1090/S0025-5718-1967-0222008-0.

[^5^]: S. Braun, "A fourth primitive solution to a⁵+b⁵+c⁵+d⁵ = e⁵" (MITM + modular filtering, 10.5M vCPU-hours), arXiv:2603.05549 (2026) — methodological sibling of this work.
