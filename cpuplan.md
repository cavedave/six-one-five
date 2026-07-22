# CPU + memory hierarchy plan (615 / 616)

Status: **design notes / spike proposals — no code yet.** Companion to
`615-ribbon-filter-plan.md`. Single-box setup: RTX PRO 6000 Blackwell (~102 GB)
+ 20-core host (campaign currently uses one core).

*(Earlier draft lived in `useCPU.md`; this file is the canonical plan.)*

---

## 1. The situation

The production binary is built without OpenMP, so candidate generation
(`gen_class_cands`, the `#pragma omp parallel for` over the B-list in the
campaign loop) runs single-threaded. That is **not** the real waste — the job
is GPU-bound and CPU gen is only a few percent of wall time at 530k, and a
*shrinking* fraction as B climbs (GPU work grows ~B^4, gen grows ~B^2).

The interesting question is the inverse: **can the 19 idle cores (and the GPU’s
own cache hierarchy) be used in ways that remove work or hide latency?**

### The governing principle

> The CPU cannot out-arithmetic the GPU on the inner loop (gate + table probe):
> for that workload one RTX PRO 6000 is ~20-100x a 20-core host (bandwidth and
> parallelism). So the CPU’s job is **not** to do the same wide arithmetic in
> parallel. Its job is to **cheaply decide what the GPU can skip** — to move
> filtering *left*, onto the host, so the GPU launches and touches less.

Every dead unit of work the cores can reject before kernel launch is pure GPU
savings. Every false positive the cores can absorb after the fact lets the GPU
run a cheaper, looser test. Both are “CPU before/around GPU,” not “CPU instead
of GPU.”

---

## 2. Idea A (primary) — host-side block precompaction for cls5

### What the GPU does today

`k_cls234` (cls5 find3) is launched as a 2D grid `dim3(n_cand, ymax)`, i.e. one
block per `(candidate, w'-grid-index)`. Thread 0 of each block computes
`R = base - (21 w')^6`, then `T = R / 42^6`, then the **triple-sum window gate**
(does `T` reduce to a sum of three sixth powers mod 504 and 247?). On the
530k-630k bench this gate skipped **91.8%** of blocks.

The catch: those 18.8 billion skipped blocks are still **launched**. Each pays:

- block scheduling / occupancy slot,
- `gate_load` — a cooperative ~1.6 KB shared-memory copy of the gate bitmaps,
  done by *every* block including the ones about to die,
- thread-0’s residue computation,

before returning early. That is real overhead the tri-gate does **not** remove.

### The proposal

The tri-gate verdict depends only on `base` (per candidate) and `w'` — all
known on the host. So let the CPU cores compute it:

```
for each cls5 candidate (parallel over cores):
    for each w' in the candidate's grid:
        R = base - (21 w')^6            # i128
        if R <= 0 or R % 42^6 != 0: continue
        T = R / 42^6
        if tri504[T % 504] and tri247[T % 247]:
            emit (candidate_index, w')  # a survivor
```

Then launch the GPU over the **compacted survivor list** instead of the full 2D
grid. The GPU never sees the dead blocks: no launch, no `gate_load`, no residue
calc for 91.8% of them.

### Cost / benefit

- **CPU cost:** ~20.5e9 verdicts per 4 chunks at 530k. Across 20 cores, ~10 s
  per 4 chunks — cheap.
- **GPU saving:** whatever fraction of cls5 time is dead-block overhead.
  **Unknown until measured.**
- **Risk — H2D:** ~1.7e9 survivors/chunk × 8 B ≈ 13 GB per 4 chunks. Mitigate
  with packed indices, per-candidate w' bitmasks + GPU stream-compaction, or a
  GPU pre-pass kernel (fallback if H2D dominates).

### Measurement gate (do this first for Idea A)

1. Count launched vs active blocks — already known via `tri_total` / `tri_skipped`
   (91.8% skip).
2. Timing A/B: early-return **before** `gate_load` on skipped blocks vs current
   (`gate_load`-then-skip). Delta = dead-block overhead ceiling.
   - If >~20% of cls5 time → build precompaction.
   - If <~5% → skip; pivot to window-narrowing.

---

## 3. Idea B — stronger host anchor pre-filtering

Drop whole `(u',v')` anchors on the host before they become GpuCand rows.
Additional necessary conditions on `base` mod new independent primes (same
spirit as tri-gate, one level up). Parallel across cores; removes entire grids.
Measure density first (mod 11/25 were useless for triple sums).

---

## 4. Idea C — idle CPU tolerates more GPU false positives

Spend idle CPU on exact i128 verify so the GPU membership test can be **smaller
or looser** (ribbon `r`, two-level filter). Naive “shorten fp64 on the current
table” does not pay: probe chains set by load factor; table stays latency-bound
in DRAM; short keys → hit-buffer overflow.

Versions that can pay off:

1. **Ribbon with smaller `r`** — cores absorb extra FPs; measure if smaller
   filter is actually faster on GPU.
2. **Secondary in-cache filter** — cheap “no” skips the big table; FPs to CPU
   (see §7 Lever 1).

Hard constraints: `--hit-cap` must not overflow; verify cost depends on
whether payload `(i,j)` is kept in the store.

---

## 5. Idea D — use the GPU cache (L2), not “more cache” blindly

### Why the cache is unused today

Probe target = `hash_pos(funnel_fp(R))` — **uniform random** slot in a 4–8 GB
table. Random access into a structure far larger than L2 → ~0% hit rate; every
probe is an L2 miss and a DRAM round trip (~400–600 ns). A *bigger* cache does
not help: there is no working set and no locality.

Observed: cls5 ~1.77e9 probes/s at ~16 B/slot ≈ **28 GB/s** effective — nowhere
near the GPU’s ~1.5 TB/s peak. That gap usually means **latency-bound with low
memory-level parallelism (MLP)**, not “we should use more bandwidth.”

**Reframe:** our “shrinking” ideas are not anti-cache. **Shrinking the first-level
test until it fits L2 is how you go from 0% cache use to ~99% hits on the
common “not present” path.**

### Lever 1 — L2-resident first-level filter (the real cache play)

Put a **membership filter** small enough to live in L2 (~100 MB on Blackwell)
*in front of* the multi-GB exact table:

| Structure | Size @ ~1.1e8 keys | Fits L2? |
|---|---|---|
| Current open-address table | ~4.3 GB | No |
| Bloom / ribbon @ ~10 bits/key | ~140 MB | Borderline |
| Small filter @ ~6–8 bits/key | ~85–110 MB | Yes-ish |

Flow:

```
query fp
  → L2-resident filter says "definitely not"  (~99% of post-gate probes) → done in L2 (~tens of ns)
  → filter says "maybe"                       → probe big table in DRAM OR exact CPU verify
```

If the filter is L2-resident, the dominant “no” path avoids DRAM entirely —
potentially ~10× faster per rejected probe vs today’s random DRAM hop.

**Caveats:**

- Filter grows with B; at high B (~350 MB+) it spills L2 again — wins most in
  530k–1.0M band unless we tune bits/key down (more FPs → CPU absorbs).
- Overlaps Idea C / ribbon spike: same two-level architecture, viewed from cache.
- False negatives forbidden: filter must be a **superset** of true pair sums
  (Bloom/ribbon property).

**Spike:** build a host-side blocked Bloom on pair-sum keys; measure size vs
selectivity; prototype GPU query with filter in constant/L2-friendly layout;
A/B probes/s with `--no-gate` off (post-gate survivors only).

### Lever 2 — raise memory-level parallelism (bandwidth, not cache)

The 28 GB/s vs 1.5 TB/s headroom is largely **latency not hidden**. Random
open-addressing probes serialize: read slot → compare → maybe next slot. Few
independent DRAM requests in flight per SM.

Likely limiters today:

- **Warp divergence** in the c3 loop (gate pass differs per lane; probe chain
  length differs per lane).
- **Occupancy** — `gate_load` ~1.6 KB shared mem/block; register pressure.
- **Dependent load chains** — each probe step waits on the previous slot read.

Tactics (kernel engineering):

- **Batch probes per thread** — software-pipeline several c3 values so multiple
  loads are outstanding before any result is used.
- **Separate gate compaction from probe** — compact “gate passed” lanes, then run
  dense probe warps (less divergence).
- **Reduce shared-mem footprint** — e.g. don’t reload full gate every block if
  precompaction (Idea A) shrinks launches.
- **Software prefetch** next hash slot where chain length is predictable.

**Spike:** profile first; don’t tune blind.

```bash
ncu --set full -k k_cls234 -c 5 ./solve_616_v1 530000 538000 all --bench 1
```

Read: achieved occupancy, memory throughput %, L2 hit rate, warp stall reasons
(long scoreboard = waiting on memory). Add `prop.l2CacheSize` to startup print.

**Expectation:** Lever 1 (L2 filter) = potentially large win on common-case
probes in mid-B band. Lever 2 (MLP) = where raw bandwidth headroom lives, but
yield uncertain until profiled.

---

## 6. What does NOT work

- **CPU co-processing the inner gate+probe loop** — ~20–100× slower than GPU.
- **More i128 verify to reduce search** — verify is downstream; cannot shrink
  the search itself.
- **GPU high B + CPU low B split** — cost ~B^4 makes the cheap low-B slice
  negligible vs the frontier.
- **“Just use a bigger cache”** without changing access pattern — random probes
  into multi-GB tables never hit.

---

## 7. Recommended sequence (full plan)

| Step | What | Decides |
|------|------|---------|
| **0** | *(done)* Tri-gate ON for cls5; bench 530k–630k (91.8% block skip) | Baseline for cls5 |
| **1** | **Dead-block overhead A/B** (§2) — skip before `gate_load` vs after | Build Idea A or not |
| **2** | **Nsight profile** cls5 `k_cls234` (§5 Lever 2) | L2 filter vs MLP vs both |
| **3a** | If dead-block overhead material → **Idea A** precompaction (prefer GPU prepass or bitmask to avoid 13 GB H2D) | cls5 launch cost |
| **3b** | Cheap **Idea B** anchor modulus (measure density first) | Fewer GpuCand rows |
| **4** | **Lever 1:** L2-resident secondary filter in front of table (ties to ribbon / Idea C) | Probe latency on “no” |
| **5** | **Lever 2:** MLP kernel changes if profile shows memory latency stalls | Occupancy / pipelining |
| **6** | Ribbon spike with **`r` sweep** — smaller filter, CPU absorbs FPs | Table size + query cost |
| **7** | Rebuild with **OpenMP** (`-Xcompiler -fopenmp`) for free gen parallelism | +2–8% wall at high B |

All filters/gates are **necessary conditions** — zero false negatives on true
solutions; plant tests and `--xcheck` remain the safety net.

---

## 8. Quick reference — where each idea attacks cost

```
Campaign cost (cls5-dominated at high B)
│
├─ Block launches (91.8% dead today)     → Idea A (CPU/GPU precompaction)
├─ c3 window iterations                  → tri-gate (done), window math (future)
├─ Per-c3 pair gate                      → already ~99% kill on survivors
├─ Table probe (DRAM, random)            → Lever 1 (L2 filter), Lever 2 (MLP)
├─ Table size / build time               → ribbon (Idea C)
└─ CPU candidate gen (few %)             → OpenMP rebuild
```
