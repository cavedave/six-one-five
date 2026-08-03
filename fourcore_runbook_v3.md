# (6,1,5) multi-class post-i128 — host Stage 1–2 (`*_v3`)

Goal band: **B = 2,353,974 → ~2,740,000** (all Meyrignac classes; last window with pair indices \(N=B/42\le 65535\)).

This stage is **host / GMP only**. GPU find3/find2 consumers come later.

## Build

```bash
make fourcore-hunt-v3
# needs libgmp (brew: gmp; apt: libgmp-dev)
```

## Tests to run now (no GPU)

### 1. Selftest (must PASS)

```bash
./fourcore_hunt_v3 --selftest
```

Checks: seed sizes 144/24/36/6/6, i128 wall bits, cls1–5 peel identities, density ordering.

### 2. Density on a v3-comparable strip (inside i128)

```bash
./fourcore_hunt_v3 --density --lo 2200000 --hi 2200099 --classes all
```

Optional A/B later: instrument `solve_516_v3` cand counts on the same strip; unit totals per class should match.

### 3. Density past the wall (smoke)

```bash
./fourcore_hunt_v3 --density --lo 2353974 --hi 2354973 --classes 2,3,4,5
```

Note `expanded_jobs~` for cls5 — that is why production emit is **units-only** until the GPU peels free terms (or we batch-expand).

### 4. Emit Stage-1 units (production-shaped)

```bash
mkdir -p runs
./fourcore_hunt_v3 --lo 2353974 --hi 2354100 --classes 2,3,4,5 \
  --emit-units runs/fc_v3_units_smoke.buc
head runs/fc_v3_units_smoke.buc
wc -l runs/fc_v3_units_smoke.buc
```

Line: `cls B u`

### 5. Tiny expanded T emit (cls2–4 only; capped)

```bash
./fourcore_hunt_v3 --lo 100100 --hi 100200 --classes 2,3,4 \
  --expand --emit runs/fc_v3_t_smoke.but --max-expand 100
head runs/fc_v3_t_smoke.but
```

Line: `cls B u free1 free2 T_lo T_hi`  
(`free2=0` for cls2–4; both set for cls5.)

**Do not** full-expand cls5 over large B ranges on disk.

## Files

| File | Role |
|---|---|
| `fourcore_classes_v3.hpp` | Spec + seeds / units / free-term grids (= v3) |
| `fourcore_gmp.hpp` | GMP \(T\) + peel + verify helpers |
| `fourcore_hunt_v3.cpp` | Emit + `--selftest` / `--density` |
| `fourcore_find*_v3.cu` | *(not yet)* GPU find4/3/2 consumers |

## GPU path (`fourcore_find_v3`) — run this when the card is free

### Build + selftests

```bash
make fourcore-hunt-v3 fourcore-find-v3
./fourcore_find_v3 --selftest-host
./fourcore_find_v3 --selftest-gpu
```

### Smoke (cls2–4 only — skip full cls5 expand)

```bash
mkdir -p runs
# tiny pre-expanded jobs (already have from host tests) OR:
./fourcore_hunt_v3 --lo 100100 --hi 100200 --classes 2,3,4 \
  --expand --emit runs/fc_v3_t_smoke.but --max-expand 5000

./fourcore_find_v3 --jobs runs/fc_v3_t_smoke.but --max-table-gb 16
```

### First past-wall strip (cls2–4, capped expand)

```bash
./fourcore_hunt_v3 --lo 2353974 --hi 2354100 --classes 2,3,4 \
  --emit-units runs/fc_v3_234_smoke.buc

# Expand+search in find (cap keeps cls4 volume sane for a smoke):
./fourcore_find_v3 --units runs/fc_v3_234_smoke.buc \
  --max-expand 200000 --max-table-gb 80 \
  > runs/fc_v3_234_smoke.log 2>&1
```

### First overnight shape (cls2–4 before cls5)

```bash
./fourcore_hunt_v3 --lo 2353974 --hi 2360000 --classes 2,3,4 \
  --emit-units runs/fc_v3_234_2p35_2p36.buc

stdbuf -oL -eL nohup ./fourcore_find_v3 --units runs/fc_v3_234_2p35_2p36.buc \
  --max-table-gb 80 \
  > runs/fc_v3_234_2p35_2p36.log 2>&1 &
```

**cls5:** same `.buc` emit, but expand is enormous — only after cls2–4 path is green; use tiny `--hi` windows or a later streaming/GPU-peel path.

### Job formats

| File | Line |
|---|---|
| `.buc` | `cls B u` |
| `.but` | `cls B u free1 free2 T_lo T_hi` |

---

## cls5 streaming (post cls2–4 clearance)

Do **not** full-expand cls5. Use `--stream-cls5` (batch find2 on the fly).

### Stage 1 — host checks (no GPU)

```bash
make fourcore-find-v3   # or fourcore-find-v3-host for count-only / selftest-host
./fourcore_find_v3 --selftest-host

# Emit a tiny cls5 unit file past the wall:
./fourcore_hunt_v3 --lo 2353974 --hi 2354000 --classes 5 \
  --emit-units runs/fc_v3_cls5_tiny.buc
wc -l runs/fc_v3_cls5_tiny.buc
head runs/fc_v3_cls5_tiny.buc

# Count expand rate only (caps jobs):
./fourcore_find_v3 --stream-cls5 --count-only \
  --units runs/fc_v3_cls5_tiny.buc --max-expand 200000
```

Paste selftest + count lines before Stage 2.

### Stage 2 — GPU smoke (capped)

```bash
./fourcore_find_v3 --selftest-gpu
./fourcore_find_v3 --stream-cls5 --units runs/fc_v3_cls5_tiny.buc \
  --max-expand 500000 --max-table-gb 80 --batch 4096 \
  > runs/fc_v3_cls5_smoke.log 2>&1
tail -30 runs/fc_v3_cls5_smoke.log
```

### Stage 3 — first real strip (after smoke OK)

```bash
./fourcore_hunt_v3 --lo 2353974 --hi 2355000 --classes 5 \
  --emit-units runs/fc_v3_cls5_2p35_2p355.buc
stdbuf -oL -eL nohup ./fourcore_find_v3 --stream-cls5 \
  --units runs/fc_v3_cls5_2p35_2p355.buc --max-table-gb 80 --batch 4096 \
  > runs/fc_v3_cls5_2p35_2p355.log 2>&1 &
```

---

## cls5 GPU peel (faster — after stream frees the GPU)

`fourcore_cls5_gpu_v3`: one GMP `B^6-u^6` per unit, then v3-style `(d,e)` grid on GPU (192-bit).

**VRAM:** past-wall cls5 needs `N≈56k` → `S≥31` (~34 GB) minimum; with `--max-table-gb 80` it picks `S=32` (~69 GB). That will OOM if `stream-cls5` already holds an ~80 GB table on the same GPU. Check `nvidia-smi` first; do not run both at once.

```bash
nvidia-smi   # confirm free memory / only one fourcore process
make fourcore-cls5-gpu-v3
./fourcore_cls5_gpu_v3 --selftest-host

./fourcore_cls5_gpu_v3 --units runs/fc_v3_cls5_2p35_2p354.buc --max-table-gb 80 \
  > runs/fc_v3_cls5_gpu_2p35_2p354.log 2>&1
# Compare wall time / solutions=0 to the stream-cls5 log on the same .buc.
```
