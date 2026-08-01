Here's the porting plan, staged by what each step must prove before the next is allowed to run. The watching brief throughout: **every class is a new mathematical object, not a code clone** — the congruence contract changes per class, so each stage ends with a plant/identity test that a blind port would silently fail.

## Stage 0 — Lock the c1 contract as the reference (done)

The fourcore c1 path is your proven template: hunt emits `(B, u, T_lo, T_hi)` with `T = (B⁶−u⁶)/D⁶` [10][11], find4 decomposes T against the fingerprint pair table [6][28], and the gate kills ~83% of probes on the deep-filtered stream (predicted 98.9% on unfiltered [26]). Before touching cls2–5, freeze the A/B harness: same `.but` into old `fourcore_find4` and the ported find4, requiring sorted-identical `SOLUTION`/`exact hits` output. That harness is the certifier for everything below.

## Stage 1 — Per-class job anatomy (the real design work)

Class 1's reduction is the lucky case: the unit is coprime to all of 2,3,7, so after CRT seeding mod 42⁶ = 2⁶·3⁶·7⁶ (M2=64, M3=729, M7=117649) the remaining four terms all carry factor 42 [10][12]. For classes 2–5 the role assignments over the five slots differ, so each class has its **own master modulus and its own scaling vector**:

- **cls2** (f=14): the {odd, ~3, ~7} roles split differently — four terms aren't all 42-divisible; one carries only 14, one carries the rest. Master ~14⁶, scalings `42c₁, 42c₂, 42c₃, 14e` [12][22][27].
- **cls3** (f=21): master ~21⁶, scalings `42c₁, 42c₂, 42c₃, 21d` [12][22].
- **cls4** (f=7): master ~7⁶, scalings `42c₁, 42c₂, 42c₃, 7e` [12][22].
- **cls5**: the three-way split — `42c₁, 42c₂ + 21d + 14e`, master 7⁶ with independent congruence links on the 21d and 14e slots [12][22][26].

**Deliverable:** a per-class spec table (master modulus M, seed count — cls2 has 24 seed classes mod 14⁶ vs cls1's 144 mod 42⁶ [26][27], scaling vector, and the exact per-slot congruence links) verified against v3's `seeds_for_B` for each class [12][26][27], not against my memory.

## Stage 2 — per-class CRT emission (hunt port)

For each class: sixth-power root tables at M2/M3/M7 are reused unchanged [10][12], but the seed assembly changes: scaled roots via `x⁶ ≡ B⁶·(scale⁶)⁻¹ mod m` [26][27], then CRT2 merge, then per-class admissibility (u odd, gcd conditions from the class spec). Emission rate/purity per class validated by: the emitted **u count per B** must match v3's measured class cand-density (c2 ≈ 31×, c3 ≈ 8×, c4/c5 ≈ 377× the c1 rate at B≈2.2M [12]).

## Stage 3 — per-class T and the 13/19/25/37 filter revival

Here's the buried landmine I keep flagging: the "13/19/25/37 vacuous" result is **a class-1 theorem**, true because the aligned c1 congruence forces Q mod those primes into the foursum-cone automatically [22]. For cls2–5 the roles split, the equivalence class of Q mod p shifts, and those primes stop being vacuous — the class-specific `q504/q247`-style residue grids in v616/v617 exist precisely because of this [26][27]. So Stage 3 extends `fourcore_filter`'s tower tables per class [31], adds whichever of 13/19/25/37 are live for that class, and re-measures survival per class (expect joint survival *worse* than c1's 0.1434 [24][31]).

## Stage 4 — find4 port: same table, class-aware fingerprints

The pair table and hash machinery are table-of-sixth-powers — class-agnostic [6][12]. What changes is the *outer factor*: terms are `scale·cᵢ`, so the kernel needs the per-class scaling applied when forming `tfp = q_lo − scale⁶·c4⁶` and when recomposing solutions [12][26][27]. cls5 additionally needs the two-factor peel (21d, 14e) before the find4 core [26]. The CRT 504/247 gate machinery drops in unchanged, but the per-job `q504/q247` must be computed from the *class-adjusted* Q [26][27].

## Stage 5 — void-safety: hostile plants + per-class identity tests

The current plant suite only exercises class-1-shaped targets [12]. Mandate per class: (a) "in-domain" plant — constructed Q with all terms legally scaled for that class, must be found; (b) **hostile plant** — a true (B,u) job pair satisfying only the raw B⁶ identity with no class constraints, must be found in exactly one class; (c) **negative plant** — a Q that violates the class's role structure must be found in *no* class and must be caught as such at emission. These are what's missing today [12][22] and they are the difference between "port compiled" and "search is complete."

## Stage 6 — scale A/B against v3 before any production B

Two-arm validation at B ≈ 2.2M (inside v3's known-good domain): run v3 classes 2–5 on a strip, run the ported fourcore classes 2–5 on the same strip, require the same (empty, at 2.2M) exact-hits and cross-check per-class probe counts. Then the first genuinely new strip — [2,353,974 → 2,500,000] classes 2–5 — becomes production-valid, replacing the overnight v3 run I drafted earlier [12][25].

## The two hard-won cautions to carry through

1. **Units are not assumed largest** [22]: v2/v3 dropped that assumption deliberately, and the (6,1,7) evidence says the unit's rank is uniform over the terms. The port must not quietly re-impose an "u is the biggest" shortcut in class 1's style of CRT seeding.
2. **No skipped roots**: the 36-vs-144 seed-class point is already documented as mattering (branch statistics show no preferred Teichmüller branch [23]) — a port that trims the seed set for speed recreates my "role-splitting collapses to coincident" error from before, machine-fast and silently wrong [23][26].

Sequencing ETA on the GPU-wall clock you actually have: Stages 1–3 are the math (no GPU time); Stage 1's spec table is the one deliverable that, once it's written and cross-checked against `seeds_for_B` [12][26][27], makes everything else mechanical. I'd start there, and only let a kernel compile once the spec is pasted back and agreed.

One thing to bring forward from the search side while this porting happens: your c1 frontier just banked ~3.57–3.68M with v2 [24][31], and no (6,1,n) solution is known for n<7 (smallest (6,1,7) is 1141 [1][2][29]) — so classes 2–5, which carry ~85% of the observed (6,1,7)-shape solution mass [22], are where the next discovery most plausibly lives. That's why the port matters more than another class-1 strip.