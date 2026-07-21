# Ribbon filter spike

Experimental track to replace the 68.7 GB open-addressing pair-sum table in
`solve_516_v3.cu` with a static ribbon membership filter (~8.6 GB at r=48).

**Status:** spike — interface stub only. See `615-ribbon-filter-plan.md` for the
full design.

## Scope

| Milestone | Deliverable | Status |
|---|---|---|
| M0 | Design doc (`615-ribbon-filter-plan.md`) | done |
| M1 | Host ribbon builder + unit tests on synthetic keys | not started |
| M2 | GPU query kernel (`ribbon_might_contain`) | not started |
| M3 | Swap into `probe()`, re-run `--selftest` / `--xcheck` / `--bench` | not started |

## What stays unchanged

Candidate generation, mod-124,488 probe gate, exact 128-bit verification, and
campaign orchestration are untouched. Only the interior of `probe()` and hit
recovery change.

## Build (when implemented)

```bash
make ribbon-test    # host-only builder tests
make solve_516_v4   # gated + ribbon variant (future)
```

## Notes

- The mod-124,488 **probe gate** (now in v3) is orthogonal: it filters before
  table access; ribbon shrinks what table access costs.
- False negatives are forbidden; construction failure is handled by seed-retry
  (see plan §R2).
