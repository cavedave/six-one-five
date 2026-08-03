# (6,1,5) four-core hunter — post-i128 runbook

Search branch:

\[
B^6 = u^6 + D^6(x_1^6+x_2^6+x_3^6+x_4^6),\qquad D=42
\]

Previous exhaustive `solve_516_v3` coverage ends at **B = 2,353,973** (i128 ceiling).
This stack continues from **B = 2,353,974** with GMP host math for \(B^6\) / \(T\) / verify.
GPU find4 still uses a **u128** reduced target \(T\approx(B/D)^6\).

## Build

```bash
# Mac (Homebrew GMP):
make fourcore-host          # hunt + host-only find4 smoke

# Server (CUDA + libgmp-dev):
sudo apt-get install -y libgmp-dev   # if needed
make fourcore                       # fourcore_hunt + fourcore_find4
```

## Validate

```bash
./fourcore_hunt --selftest
./fourcore_find4_host --selftest-host   # or ./fourcore_find4 --selftest-host
```

## Smoke emit + GPU (tiny window)

```bash
mkdir -p runs
./fourcore_hunt --lo 2353974 --hi 2355000 --D 42 --emit runs/fc_smoke.but
wc -l runs/fc_smoke.but

# Optional deep floursum filter (~14% keep; necessary condition on T):
./fourcore_filter --selftest
./fourcore_filter < runs/fc_smoke.but > runs/fc_smoke.deep.but \
  --write-rej runs/fc_smoke.rej.but 2> runs/fc_filter_smoke.log
wc -l runs/fc_smoke.deep.but

# Server GPU (N ≈ B/42 ≈ 56k at 2.35M — small table):
./fourcore_find4 --jobs runs/fc_smoke.deep.but --D 42 --max-table-gb 80
```

## First campaign (iteration goal)

VRAM-safe full pair table for \(D=42\) up to roughly **B ≈ 3.5M** (\(N\approx B/42\lesssim 85k\) at ~69 GB).

```bash
stdbuf -oL -eL nohup ./fourcore_hunt --lo 2353974 --hi 3500000 --D 42 \
  --emit runs/fc_42_2p35_3p5.but > runs/fc_hunt_2p35_3p5.log 2>&1 &

# After jobs exist (or pipeline on a second machine once hunt finishes a chunk):
stdbuf -oL -eL nohup ./fourcore_find4 --jobs runs/fc_42_2p35_3p5.but --D 42 \
  --max-table-gb 80 \
  > runs/fc_find4_2p35_3p5.log 2>&1 &
```

Optional find-ONE cut: `--x4-top 5000 --stop-first` on `fourcore_find4`.

## Files

| File | Role |
|---|---|
| `fourcore_gmp.hpp` | GMP \(T\) + verify |
| `fourcore_hunt.cpp` | Stage‑1 \((B,u)\) + emit `.but` |
| `fourcore_filter.cpp` | Deep floursum prefilter → `.deep.but` (~14% keep) |
| `fourcore_find4.cu` | Pair table + GPU find4 |

Job line format: `B u T_lo T_hi`.
