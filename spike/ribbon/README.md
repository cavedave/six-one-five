# Ribbon / xor store spike

Experimental track to replace the 68.7 GB open-addressing pair-sum table in
`solve_516_v3.cu` with a static membership filter (~8–10 GB), then run on A100.

**Status:** M1 host xor builder + tests. Ribbon still a stub.
See `615-ribbon-filter-plan.md` (filter math) and
`615-a100-ribbon-shard-plan.md` (A100 + xor→ribbon→shard sequence).

## Scope

| Milestone | Deliverable | Status |
|---|---|---|
| M0 | Design docs | done |
| M1 | Host **xor** builder + FN/FPR tests | **done (this spike)** |
| M2 | GPU query + `solve_516_v4` wiring | not started |
| M3 | `--save-store` / `--load-store` | not started |
| M4 | Ribbon builder (same API as xor) | stub only |
| M5 | Residue-steered `--shards S` | helpers only (`shard.hpp`) |

## Files

| File | Role |
|---|---|
| `xor_filter.hpp` | Classic 3-hash xor filter (build + host query) |
| `xor_filter_test.cpp` | FN-free + FPR + shard smoke |
| `mix64.hpp` | Shared mixer (host now; device later) |
| `store_header.hpp` | On-disk header sketch |
| `shard.hpp` | `shard_of_key` / `shard_of_mixed` |
| `ribbon_filter.hpp` | Ribbon stub (M4) |

## What stays unchanged

Candidate generation, mod-124,488 probe gate, exact 128-bit verification, and
campaign orchestration are untouched until M2. Only the interior of `probe()`
and hit recovery change in `solve_516_v4`.

## Build

```bash
make xor-test       # host xor FN/FPR suite (M1)
make ribbon-test    # ribbon stub smoke
make v3-a100        # solve_516_v3 with sm_80 (T0 rental binary)
make v4-a100        # solve_516_v4 when M2 lands (sm_80)
```

## Notes

- The mod-124,488 **probe gate** is orthogonal: it filters before table access;
  xor/ribbon shrink what table access costs.
- False negatives are forbidden; construction failure is seed-retry.
- Prefer **xor before ribbon** so GPU plumbing and construction bugs stay separate.
