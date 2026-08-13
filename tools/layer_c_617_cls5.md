# Layer C: (6,1,7) cls5 — peel depth vs find2 fusion

Status: first pass (scored plans + CRT notes). Not a new Meyrignac class —
**same** Branch A cls5 completeness as `solve_617_v1.cu`.

Re-run:

```bash
python3 tools/euler_peel_score.py --617 --B 4e5
python3 tools/euler_peel_score.py --617 --B 1e6
```

---

## 1. Production class (what `solve_617` already is)

Seven terms = \(B^6\), \(\gcd(B,42)=1\), roles split \(\{odd\}\,|\,\{\sim3\}\,|\,\{\sim7\}\):

| Term | Scale | Role |
|------|-------|------|
| \(14u'\) | 14 | ~3 |
| \(6v'\) | 6 | ~7 |
| \(21w'\) | 21 | odd |
| \(42c_1\ldots c_4\) | 42 | role-free seeds |

Search identity:

\[
T=\frac{B^6-(14u')^6-(6v')^6-(21w')^6}{42^6}
=c_1^6+c_2^6+c_3^6+c_4^6
\]

| Property | Production |
|----------|------------|
| Stage-1 anchors | \((u',v')\) CRT classes |
| Kernel free | \(w'\) grid (mod 64) |
| Leaf | **find4** (`k_find4_cls5`) |
| Scorer | **not** `cls5_shaped` (P1 fails: residual arity 4) |

This is why “port the (6,1,5) cls5 kernel” does **not** give a 1000×: wrong leaf.

Header cost model (~\(B=10^6\)): cls5 ~\(1.1\times10^{11}\) probes/\(B\); dominates Branch A.

---

## 2. Deeper peel variants (still the same class)

Change **search strategy only**: treat extra \(42\cdot c_i\) as FREE peels.

| ID | Free peels | Residual | Scorer @ \(B=4\times10^5\) |
|----|------------|----------|----------------------------|
| **A production** | \(u',v',w'\) (3) | find4 | `cls5_shaped=no` |
| **B** | + one \(c_i\) (4) | find3 | `no` |
| **C** | + two \(c_i\) (5) | **find2** | **`CLS5_SHAPED=yes`** |
| **D** | C + \(w'\) CRT~64 in grid est. | find2 | **`CLS5_SHAPED=yes`** |

Fixture labels in `tools/euler_peel_score.py` → `known_617_cls5_plans()`.

**C/D pass P1–P6** with `S=42`: exact reduction exists; free grid is huge; leaf is one xor probe.

---

## 3. CRT / seed implications

| Free index | Thinning today | If peeled deeper |
|------------|----------------|------------------|
| \(u'\) | CRT mod \(3^6\) (cls2-style) | Keep Stage-1 emission |
| \(v'\) | CRT mod \(7^6\) | Keep Stage-1 emission |
| \(w'\) | CRT mod \(2^6=64\) on GPU | Keep |
| Extra \(c_i\) | **None** beyond \(1\ldots\lfloor B/42\rfloor\) | Dense loops — **no new seeds** |

So deeper peel does **not** need new Meyrignac moduli. It needs a kernel that nests the extra \(c\)-loops on device (or loses to host expand).

Completeness: unchanged — still full Branch A cls5.

---

## 4. Will find2 fusion actually be ~10× (or more)?

Scorer `cls5_shaped` only means “shape admits fusion,” **not** “cheaper than find4.”

Rough work compare **per** \((u',v',w')\):

| Strategy | Inner work |
|----------|------------|
| Production find4 | ~\(3\times10^7\) probes (header) |
| Deeper find2 | ~\(\binom{N}{2}\) xor probes, \(N=\lfloor B/42\rfloor\) |

At \(B\approx4\times10^5\): \(N\approx9524\), \(\binom{N}{2}\approx4.5\times10^7\) — **same order** as find4’s probe count.

So naive deeper peel **reorganizes** work that find4 already spends walking core indices; it does **not** automatically delete a 1000× host-grid the way (6,1,5) stream-cls5 did.

**Where a ~10× might still appear**

1. Find4 windows + OA table probes heavier than packed xor find2 (constants, gate, memory).
2. Fuse peel+\(c_i\)+xor in one kernel (less launch / less host).
3. Better gates on the 5-D residual before probe.
4. Skip dead find4 structure (iroot6 windows, ytiles) if xor FPR is low.

**Where it will not**

- Expecting (6,1,5)-style 1000× from “stop materializing jobs” — production 617 cls5 is **already** GPU find4.
- Ignoring \(\sim N^2\) growth of the two extra free cores.

**Verdict:** deeper find2 is a **plausible ~2–10× engineering bet**, worth a spike kernel + plant of known hits (e.g. \(B=400471\)), not a claimed miracle until benchmarked against `k_find4_cls5`.

---

## 5. Honest ~10× without new peel depth (parallel track)

On current `solve_617_v1` (no Layer C invention):

| Lever | Idea |
|-------|------|
| Packed xor / `--load-table` | If still on fat OA slots, match (6,1,5) v4 packing |
| Probe gate already ~99% kill | Keep; A/B with `--no-gate` |
| Chunk / host RAM | `--chunk` already; avoid rebuild thrash |
| Profile cls5 vs cls2–4 | cls5 is ~half of probes/\(B\); optimize that path first |
| Known-plant microbench | Time `k_find4_cls5` vs a toy fused find2 on same \((u',v',w')\) |

---

## 6. Next build steps (when you want code)

1. **Plant spike:** for known cls5 solution \(B=400471\), run production find4 vs a host-reference deeper find2 count (same peels, compare probe counts).
2. If find2 probe count ≲ find4 within ~3×, sketch `k_cls5_617_find2` (5-D free, xor leaf) beside `k_find4_cls5`.
3. Only then overnight strip vs current binary.

---

## 7. Score snapshot (`--617 --B 4e5`)

```
production 3FREE+find4     => no          (find4)
deeper 4FREE+find3         => no          (find3)
deeper 5FREE+find2         => CLS5_SHAPED (grid ~3e21 raw)
5FREE+find2 (w' CRT~64)    => CLS5_SHAPED (grid ~5e19 est.)
```
