# A100 port via ribbon + residue sharding — code & math plan

Status: **implementation plan.** Companion to `615-ribbon-filter-plan.md`
(membership filter) and `j1plan.md` §4.3 / §9.2–9.3 (sharded pair tables).
Target binary family: `solve_516_v4` (fork of `solve_516_v3.cu`), with the same
pattern later applied to `solve_616_v1` / `solve_617_v1` / `record_find5`.

Goal: rent cheap A100s (sm_80, 40/80 GB) and get **correct, complete** campaigns
without needing Blackwell VRAM. Speed target: within ~2× of one RTX PRO 6000 per
card; multi-card B-range split for cost wins.

---

## 0. One-paragraph pitch

The v3 pair table is a 68.7 GB open-addressing store of
\(s(i,j)=(i^6+j^6)\bmod 2^{64}\). That fits a ~96–102 GB RTX PRO 6000 and is
tight or impossible on A100 40/80 GB. Replace the store with a **static membership
filter** (xor first, then ribbon) at ~8–10 GB, recover \((i,j)\) on the host for
the rare “maybe” hits, and optionally **residue-partition** the key set into
shards that fit any VRAM budget. Probe kernels stay the same arithmetic; only
lookup + hit recovery change. Compile for `sm_80` (and keep `sm_90`/`sm_120`
targets). Completeness is exact; soundness still rests solely on i128 verify.

---

## 1. What does not change (math or code)

| Piece | Role |
|---|---|
| Class congruences / candidate gen | Same admissible \((B,u,\ldots)\) stream |
| Mod-124,488 probe gate | Same CRT bitmaps; kills ~99% before lookup |
| `funnel_fp` / residual arithmetic | Same 64-bit fingerprint of the needed pair sum |
| `verify_hit` (i128, gcd, ordering) | Sole accept/reject authority |
| `--selftest` / `--xcheck` / `--bench` | Extended, not replaced |
| Campaign split by \([B_{\min},B_{\max}]\) | Already embarrassingly parallel across machines |

Only: (a) how membership is stored/queried, (b) how \((i,j)\) is recovered,
(c) optional shard index on the residual, (d) nvcc arch flags.

---

## 2. Math

### 2.1 Pair-sum key set

At campaign cap \(B_{\max}\),

\[
N=\Big\lfloor\frac{B_{\max}}{42}\Big\rfloor,\qquad
n=\binom{N}{2}\ \text{(or \(N(N+1)/2\) if diagonal pairs are kept as in v3)}.
\]

v3 inserts \(1\le i\le j\le N\) (includes \(i=j\)). Keep that convention in v4 so
xcheck against v3 stays apples-to-apples. Keys:

\[
k_{ij}=(i^6+j^6)\bmod 2^{64}.
\]

Lattice structure mod small \(m\) is real; apply a strong 64-bit mixer
(`mix64` / splitmix finalizer) **once at key birth** so filter positions and
shard IDs see near-uniform bits (same A7 as the ribbon plan).

### 2.2 Membership filter (completeness)

Let \(F\) be a static approximate membership structure on the mixed key set with:

- **No false negatives:** \(k\in K \Rightarrow F(k)=\mathsf{true}\).
- **FPR \(\approx 2^{-r}\)** on non-keys (xor/ribbon fingerprint width \(r\)).

For a genuine solution, the device residual fingerprint \(v\) equals some
\(k_{ij}\in K\), so \(F(v)=\mathsf{true}\), the hit is recorded, host
re-derivation enumerates **all** pairs with that 64-bit residue, and
`verify_hit` accepts the true one. Filter can only *increase* checks; it cannot
drop a real solution.

Expected false positives over a campaign of \(P\) probes:

\[
\mathbb{E}[\mathrm{FP}]=P\cdot 2^{-r}.
\]

| \(r\) | \(\mathbb{E}[\mathrm{FP}]\) at \(P=3\times10^{14}\) | Host re-derive cost (~1 ms each) |
|---|---|---|
| 48 | \(\sim 1\) | negligible |
| 40 | \(\sim 3\times10^{2}\) | ~0.3 s |
| 32 | \(\sim 7\times10^{4}\) | ~70 s |

Default \(r=48\). On A100 with idle host cores, \(r=40\) is an allowed knob if a
smaller filter helps L2 residency later.

**Sizes at \(n=1.372\times10^9\) (B_max=2.2M):**

| Structure | Bits/key | Size | Device reads/query |
|---|---|---|---|
| v3 open-address table | ~400 (w/ slack) | 68.7 GB | ~1.5 |
| Xor filter | \(\approx 1.23\cdot r\) | ~10.1 GB @ \(r=48\) | 3 |
| Ribbon (BuRR-style) | \(\approx r+2\) | ~8.6 GB @ \(r=48\) | 1 |

A100 40 GB: ribbon/xor fit with headroom. A100 80 GB: either fits easily;
optional denser **exact** table (`--slots-log2 31`, ~34 GB) is a stopgap that
needs **no** filter work but keeps payload shadowing (see §2.5).

### 2.3 Host re-derivation

On “maybe” hit with residual \(v\) (mixed or raw — pick one convention and stick
to it; prefer **store mixed keys, probe with the same mix**):

\[
\text{for }i=1..N:\quad \text{need}=v\ominus\mathrm{mix?}(K[i]),\quad
\text{lookup multimap }K\mapsto\{j\}.
\]

Cost \(O(N)\) ~1 ms with \(N\sim5\times10^4\). True collisions among 64-bit pair
residues are rare (\(\mathbb{E}\approx n^2/2^{65}\approx0.05\) over the whole
table); enumerating all matches **repairs** v3’s single-payload shadowing.

### 2.4 Residue sharding (when and how)

**When needed**

| Regime | Filter size | Need shards? |
|---|---|---|
| B ≤ 2.2M, ribbon/xor | ~9–10 GB | No (A100 ready) |
| B ~ 5M, ribbon | ~44 GB | Optional (fits 80 GB) |
| B ~ 10M, ribbon | ~177 GB | Yes |
| Fat exact table on 40 GB | — | Yes or denser LF |

**Partition.** Choose a small prime \(p\) (default **257**, as in j1plan) and
shard count \(S\) dividing a convenient range (32 or 64). For mixed key \(x\),

\[
\mathrm{shard}(x)=\bigl\lfloor S\cdot (x\bmod p)/p\bigr\rfloor
\quad\text{or simply}\quad
\mathrm{shard}(x)=(x\bmod S)
\]

after mix (prefer \(x\bmod S\) for equal load; use mod-\(p\) only if we later
want CRT synergy with other moduli). Document the chosen rule in the filter
header.

Shard \(s\) stores exactly the keys with \(\mathrm{shard}(x)=s\). Expected
keys/shard \(\approx n/S\); filter bytes scale \(1/S\).

**Residue-steered probing (mandatory for speed).**
Naive “run full campaign × \(S\)” is \(S\times\) too slow. Correct pattern
(j1plan §9.3): for each probe residual \(v\), compute \(s=\mathrm{shard}(v)\)
and query **only** filter/table \(s\).

\[
\text{one membership query per probe, same as monolithic.}
\]

Completeness: a true pair key \(k\) lives in shard \(\mathrm{shard}(k)\); the
probe computes \(v=k\), so it queries that shard. ∎

**Pass structure on one GPU**

1. Build or load filter for shard \(s\) (or keep a working set of shards if VRAM
   allows several).
2. Run the candidate stream; each probe routes to \(s_*=\mathrm{shard}(v)\);
   if \(s_*\neq s\), skip lookup (still run gate if cheaper to keep warp uniform —
   measure; see §5).
3. Advance \(s\), or swap the resident shard.

Better for multi-GPU: assign disjoint shard sets to cards **or** disjoint B
ranges (B-range split needs no shard logic and is week-1). Prefer B-range split
for rented fleets; implement steered sharding for single-card capacity.

### 2.5 Exact table stopgap (optional track T0)

`--slots-log2 31` → 34.4 GB table at LF≈0.64 on A100 80 GB. Completeness same as
v3 (including rare payload shadowing). Use only to:

- validate sm_80 build + rental ops before filter work lands;
- A/B probe rates vs Blackwell.

Do **not** treat T0 as the long-term A100 path.

### 2.6 Soundness / completeness summary

| Property | Argument |
|---|---|
| Soundness | Unchanged: only `verify_hit` prints solutions |
| Completeness (filter) | No FN ⇒ every true residual is recorded ⇒ re-derive finds pair |
| Completeness (shard) | \(\mathrm{shard}(v)=\mathrm{shard}(k)\) for true \(v=k\) |
| Cross-check | Monolithic vs sharded identical hit sets on a validated B band |

---

## 3. Performance model (A100 vs RTX PRO 6000)

Workload today: **latency-bound random lookups**, not peak-bandwidth-bound
(~28 GB/s effective vs ~1.5–2 TB/s peak; see `cpuplan.md`).

| Factor | RTX PRO 6000 | A100 80GB | Effect on us |
|---|---|---|---|
| Bandwidth | ~1.8 TB/s GDDR7 | ~2.0 TB/s HBM2e | Minor (we use ≪ peak) |
| SMs | ~170 | 108 | Fewer outstanding misses → main drag |
| VRAM | ~96–102 GB | 40/80 GB | Forces filter/shard |
| L2 | large (Blackwell) | 40 MB | Small front filter more valuable on A100 |
| Arch | sm_120 | sm_80 | Need explicit gencode |

**Planning number:** A100 ≈ **1.5–3× slower** than one PRO 6000 on the same
binary shape (gate + 1–3 random reads). Ribbon does not remove that gap; it
makes the A100 *fit*. Two A100s on disjoint B ranges ≈ one PRO 6000 wall-clock
if each is ~2× slower.

Cost: if A100-hr is \(\ll\) half of PRO-6000-hr, rent A100s even at 2× probes/s.

Bench protocol (mandatory before campaign):

```bash
./solve_516_v4 2000000 2200000 all --bench 12 --device 0
# compare cls5 rate= lines: PRO 6000 vs A100, table vs xor vs ribbon, S=1 vs S>1
```

---

## 4. Code architecture

### 4.1 Deliverables / tree

```
spike/ribbon/                  # existing stub — grow into real builders
  ribbon_filter.hpp            # API: build / query / header
  xor_filter.hpp               # M1 staging structure
  ribbon_filter_test.cpp       # host exhaustive + FPR tests
  shard.hpp                    # shard(x), shard header, key partition

solve_516_v4.cu                # fork of v3: Store backend + hit recovery
store_backend.hpp              # compile-time or runtime: Table | Xor | Ribbon

615-a100-ribbon-shard-plan.md  # this file
615-ribbon-filter-plan.md      # filter math detail (unchanged authority)
```

Keep `solve_516_v3.cu` as the oracle binary. v4 must pass xcheck against v2/v3
on B≤600k.

### 4.2 Store backend interface

```cpp
struct StoreHeader {
    u32 magic;          // '615S'
    u32 version;
    u32 kind;           // 0=table, 1=xor, 2=ribbon
    u32 r, w;           // fingerprint bits, ribbon width
    u64 n_keys, m_bits;
    u64 mix_seed;
    u32 shard_count;    // 1 = monolithic
    u32 shard_index;    // which shard this file is
    u32 shard_mode;     // 0 = key % S
    u32 N;              // pair index cap
};

// Device:
__device__ bool store_maybe(const Store& S, u64 fp_mixed);

// Host:
Store build_store(const std::vector<u64>& keys, const StoreConfig&);
void  save_store(const Store&, const char* path);
Store load_store(const char* path);          // mmap or cudaMalloc+H2D
std::vector<std::pair<u32,u32>> rederive(u64 v_mixed, const RederiveIndex&);
```

Hit record on device becomes `{cand, v3, v4}` (no payload). Host fills `(i,j)`
via `rederive`.

### 4.3 CLI additions

```text
--arch is a build concern, not CLI

--store table|xor|ribbon          default: table (v3 behavior) then xor then ribbon
--r 48                            fingerprint bits
--shards S                        default 1
--shard-index s                   build/load only shard s
--shard-mode mod                  key % S
--max-store-gb G                  refuse build/upload if over budget
--save-store PATH                 .bin with StoreHeader
--load-store PATH
--device K
--slots-log2 N                    table backend only (T0)
```

Campaign examples:

```bash
# A100 80GB, xor monolithic (week 2)
./solve_516_v4 730000 2200000 all --store xor --r 48 \
  --load-store /scratch/xor_2p2M.bin --device 0

# A100 40GB, 8 ribbon shards, steered probe (week 4)
./solve_516_v4 730000 1200000 all --store ribbon --shards 8 --shard-index 3 \
  --load-store /scratch/rib_2p2M_s3.bin

# Multi-card without shards: split B
# card0: 730000 1500000   card1: 1500001 2200000
```

### 4.4 Makefile / arch

```make
# Default rental / A100:
NVCCFLAGS_A100 = -O3 -std=c++20 \
  -gencode arch=compute_80,code=sm_80 \
  -Xcompiler -fopenmp

# Local Blackwell (optional fatbin):
NVCCFLAGS_FAT = -O3 -std=c++20 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_120,code=sm_120 \
  -Xcompiler -fopenmp

v4-a100: solve_516_v4.cu ...
	$(NVCC) $(NVCCFLAGS_A100) -o solve_516_v4 $<
```

Drop the current “compute_90 PTX JIT on Blackwell” stopgap for A100 builds —
native `sm_80` only.

### 4.5 Kernel integration (minimal diff)

Today (`solve_516_v3.cu`):

```cpp
cnt += probe(P, fp, ci, v3, v4);   // open-address walk, may push Hit with payload
```

v4:

```cpp
if (store_maybe(P.store, mix64(fp))) {
    // record Hit{ci, v3, v4}; payload recovered on host
}
// optional: if shards>1 && shard(mix64(fp)) != P.shard_index) skip store_maybe
```

Gate path unchanged. Warp divergence from shard skip: measure; if bad, run
one kernel launch per resident shard with a compacted candidate list (later
optimization, not M1).

---

## 5. Milestones

| ID | Deliverable | Exit criteria | Est. |
|---|---|---|---|
| **T0** | sm_80 build of v3 (`make v3-a100`) + `--slots-log2 31` on rented A100 | `--selftest` PASS; `--bench` numbers logged | Makefile done; rental bench pending |
| **M0** | This plan + CLI/arch sketch in Makefile | Reviewed | **done** |
| **M1** | Host **xor** builder + exhaustive re-query + FPR test | 100% FN-free on all keys; FPR ≈ 2⁻ʳ | **done** (`make xor-test`) |
| **M2** | `solve_516_v4` with xor device query + host rederive | `--selftest` + `--xcheck` vs v3, B≤600k, miss=0 | 1–2 days |
| **M3** | Persist `--save-store` / `--load-store`; A100 campaign smoke | 10k B band matches v3 hit set | 0.5 day |
| **M4** | Ribbon builder swap-in (same API as xor) | Same tests; size ≤ xor; bench ≥ xor | 2–4 days |
| **M5** | `--shards S` + residue-steered probe + mono↔shard equality | Identical hits on `[300k,400k]` | 1–2 days |
| **M6** | Multi-card runbook (B-split + optional shard-split) | Written + one 2×A100 dry run | 0.5 day |
| **M7** | Port store backend to `solve_616_v1` / record path | Same selftest bar | as needed |

**Critical path:** T0 → M1 → M2 → M3 (A100 useful). M4–M5 unlock bigger B and
40 GB cards. M6 is ops, can parallel M4.

Staging rule (from ribbon plan R1): **xor before ribbon**. Never debug ribbon
construction and GPU plumbing at the same time.

---

## 6. Validation matrix

| Test | Command / action | Pass |
|---|---|---|
| Host math | `v3_host_logic_test` (unchanged) | OK |
| Filter FN | re-query all \(n\) inserted keys | 0 negatives |
| Filter FPR | \(10^9\) random non-keys | rate \(\approx 2^{-r}\) |
| Rederive | \(10^5\) random true keys | exact pair set recovered |
| Device plant | `./solve_516_v4 --selftest` | plants 100% |
| CPU xcheck | `./solve_516_v4 … --xcheck` (B≤600k) | GPU-miss=0 |
| Mono vs shard | same range, `--shards 1` vs `S` steered | identical verified solutions / hit keys |
| A100 vs PRO | same `--bench` band | rates logged; no correctness delta |
| v3 oracle | same band, table backend | hit multiset agreement after rederive |

Negative results (no solution in a band) are only publishable after xcheck-class
gates, same as v3 runbook.

---

## 7. Rental / ops runbook (draft for M6)

```bash
# Build on box with CUDA ≥ 11.8 (A100) or ≥ 12.0
make v4-a100

# One-time store build can be on any fat-RAM host (need ~40 GB RAM for keys+filter)
./solve_516_v4 0 2200000 all --store xor --r 48 --save-store xor_2p2M.bin --build-only
# (add --build-only flag in M3; or a tiny host tool spike/ribbon/build_store)

# Upload + smoke
rsync -P xor_2p2M.bin solve_516_v4 user@a100:/scratch/
./solve_516_v4 --selftest
./solve_516_v4 300000 320000 all --xcheck --load-store xor_2p2M.bin
./solve_516_v4 2000000 2200000 all --bench 12 --load-store xor_2p2M.bin

# Campaign slice
./solve_516_v4 730000 1500000 all --load-store xor_2p2M.bin | tee c0.log
```

Watchdogs: `nvidia-smi` (SM util), hit-buffer overflow, store VRAM
(`--max-store-gb 35` on 40 GB cards).

---

## 8. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Ribbon construction FN bug | Xor first; exhaustive host re-query |
| R2 | A100 slower than 3× | Bench T0 early; multi-card B-split |
| R3 | Shard skip warp divergence | Measure; fallback per-shard compacted launch |
| R4 | FP storm (bad \(r\) or huge \(P\)) | Recompute \(r\) from measured probes/s |
| R5 | Host RAM during build | Stream keys; build per-shard (natural with M5) |
| R6 | Toolkit on rental image | Pin CUDA 11.8+/12.x; sm_80 gencode in CI/make |
| R7 | Mixing inconsistency host/device | Single `mix64` in shared header; plant tests |

---

## 9. Open decisions (poke before M2 locks them)

1. **Diagonal pairs:** keep \(i\le j\) (v3) vs \(i<j\) only? Prefer keep v3.
2. **Mix-then-store vs store-raw:** prefer mix-then-store (uniform shards).
3. **Xor vs jump straight to ribbon:** plan says xor first; skip only if M1 xor
   slips and a known-good ribbon lib is vendored.
4. **Default shards on 40 GB:** \(S=1\) with ribbon; \(S=2\)–\(4\) only if
   headroom fails.
5. **Primary scale-out:** B-range multi-card first; shard multi-card later.

---

## 10. Immediate next actions

1. ~~Land Makefile `v3-a100` / fatbin flags; M1 xor host builder.~~ **done**
2. Smoke T0 on a rented A100: `make v3-a100`, then `--selftest` / `--bench`
   (use `--slots-log2 31` on 80 GB if the full 68.7 GB table does not fit).
3. Fork `solve_516_v4.cu` with `store_maybe` + host rederive (M2).
4. Re-bench; decide \(r\) and whether ribbon (M4) is worth the week before
   sharding (M5).

---

## 11. Success definition

- A100 80 GB runs the full B≤2.2M (6,1,5) probe load with xor/ribbon store.
- `--xcheck` miss=0 vs v3 on validation bands.
- Measured A100 probes/s within ~3× of PRO 6000; cost/probe documented.
- Steered `--shards S` matches monolithic hits on a locked test range.
- Runbook: two-card B-split campaign without manual heroics.
