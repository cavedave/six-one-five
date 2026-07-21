# GPU solver runbook — `solve_516_v3.cu`

Companion to `solve_516_v2.cpp`. **Same mathematics, same candidate space, same completeness**
— only the decomposition stage (find4/find3/find2 over the pair table) moves to the GPU.
Every GPU hit is re-verified on the CPU with exact 128-bit arithmetic, so a GPU bug can
cause a *miss* (caught by the tests below) but never a *false solution*.

## What is where

| piece | runs on | notes |
|---|---|---|
| B-filter, root tables, CRT seeds, unit lifting, free-term residues | CPU (OpenMP) | identical to v2 |
| Q and window bounds for class 1 | CPU | one i128 division per candidate |
| pair table = hash of `(c_i⁶+c_j⁶) mod 2⁶⁴ → (i,j)` | **VRAM** | built once per campaign |
| find4/find3/find2 as hash probes | **GPU** | 1–3 random 16-byte reads per probe |
| exact verification of hits (distinctness, mod-300, i128 identity, gcd) | CPU | the only place truth is decided |

Pair-table size at `B_max = 2,200,000`: N = 52,380 → 1.372×10⁹ pairs → **68.7 GB**
(2³² slots, load factor 0.32). Fits the 96 GB card with headroom. Host RAM needs the
same again during the build (~2–4 min on 4 cores); the host copy is freed after upload.
Use `--slots-log2 31` for a 34 GB table at LF 0.64 (≈2× more reads per probe).

## 1. Build

```bash
nvcc -O3 -std=c++20 -arch=native -Xcompiler -fopenmp -o solve_516_v3 solve_516_v3.cu
```

- Needs CUDA ≥ 12.8 (Blackwell sm_120). `-arch=native` picks the right arch automatically.
- Optional headers in the same directory unlock extras:
  - `mod60.hpp` → re-enables the mod-300 pre-filter (optional; i128 identity is ground truth),
  - `quad_sum.hpp` + `k14_common.hpp` → enables `--xcheck` (CPU-vs-GPU cross-validation).
- If the Blackwell is not device 0 (the A400 may be), check `nvidia-smi` and pass `--device K`.

## 2. Test sequence (please paste back the outputs)

```bash
# (a) host math + GPU plant tests: table probe, cls1/cls234/cls5 kernels on planted targets
./solve_516_v3 --selftest
# expect: [selftest] host math OK / [plant] pairs: ok=20000 bad=0 /
#         [plant] cls1: 512/512, cls234: 512/512, cls5: 512/512 / SELFTEST PASS

# (b) cross-validation vs the CPU reference finders (needs quad_sum.hpp):
#     every candidate where the CPU finders succeed must also be hit by the GPU
./solve_516_v3 300000 320000 all --xcheck
# expect: ... CPU-found-but-GPU-missed=0 PASS

# (c) microbenchmark: measured probes/sec per class (the number I estimated as 2–4e9)
./solve_516_v3 2000000 2200000 all --bench 12
# report the "clsK: ... rate=... probes/s" lines

# (d) head-to-head vs v2 on a validated range (both must report 0 solutions)
./solve_516_v3 300000 400000 all --load-table t.bin   # v3
OMP_NUM_THREADS=4 ./solve_516_v2 300000 400000        # v2 (for comparison)
```

If (a) fails: stop — paste the output, something in the kernel plumbing needs fixing.
If (b) passes, the engine is trustworthy for negative results.

## 3. Campaign

```bash
# first run builds and saves the table (~2–4 min build + ~1 min upload)
./solve_516_v3 730000 2200000 all --save-table /scratch/table_2p2M.bin | tee campaign.log

# restarts / subsequent ranges reuse it (loads in seconds)
./solve_516_v3 730001 1200000 all --load-table /scratch/table_2p2M.bin
```

- `--chunk 8192` (B-values per batch) is a good default; raise to 32768 for slightly
  less launch overhead once stable.
- Progress lines show per-class candidate and probe counts; a `SOLUTION` line (stdout)
  is the only thing that interrupts the quiet.
- `!! hit buffer overflow` means re-run with `--hit-cap 8388608` (should never happen:
  expected hits per full campaign ~10⁴, cap default 10⁶).
- Watchdog: keep `nvidia-smi` open the first hour — you want ~95–100% SM utilisation on
  class-5 chunks and near-zero CPU in `top` (candidate generation is cheap).

## 4. Expected timeline

Full campaign 730k → 2.2M ≈ 1.6×10¹⁴ probes ≈ **a day** at 2–4×10⁹ probes/s
(vs ~1.5–3 months CPU). Class 5 is ~80% of the probes — its `rate=` line in (c) is the
number that decides the real timeline.

## 5. Known limits (unchanged from v2)

- B ≤ 2,200,000 (i128, B⁶ < 2¹²⁷) and N ≤ 65,535 (32-bit payload). Going further needs
  the 256-bit arithmetic upgrade first; the table format then takes an 8-byte payload
  (same 16-byte slot) and the RAM/NVMe tiers from the design note come into play.
- Completeness rests on the five-class master congruences + valuation filters, exactly
  as v2; the GPU changes nothing about *what* is searched.
