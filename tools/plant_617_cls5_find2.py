#!/usr/bin/env python3
"""Plant spike: (6,1,7) Branch A cls5 — find4 work vs deeper find2 work.

Uses known solutions only as *regression plants* (same Meyrignac cls5 class).
Does not invent moduli from the hits.

For each cls5 plant:
  - recover u',v',w' and four 42-cores
  - count production-style find4 (c4,c3) table-probe *calls* (pre-gate)
  - apply ~1.08% gate keep (mod 124488 CRT) for expected table probes
  - count deeper find2: all 1<=ca<=cb<=N xor probes after peeling two cores

Usage:
  python3 tools/plant_617_cls5_find2.py
  python3 tools/plant_617_cls5_find2.py --file 617-solutions-clean.txt
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def iroot6(n: int) -> int:
    if n <= 0:
        return 0
    r = int(n ** (1.0 / 6.0))
    while (r + 1) ** 6 <= n:
        r += 1
    while r > 0 and r**6 > n:
        r -= 1
    return r


def classify_cls5(B: int, terms: Sequence[int]) -> Optional[dict]:
    """Return role peels if terms match Branch A cls5 scales; else None."""
    if math.gcd(B, 42) != 1:
        return None
    by_g = {42: [], 21: [], 14: [], 6: [], 1: []}
    for t in terms:
        g = math.gcd(t, 42)
        if g in (42, 21, 14, 6):
            by_g[g].append(t)
        else:
            by_g[1].append(t)
    # cls5: one each of 14,6,21 and four of 42
    if not (len(by_g[14]) == 1 and len(by_g[6]) == 1 and len(by_g[21]) == 1 and len(by_g[42]) == 4):
        return None
    if by_g[1]:
        return None
    t14, t6, t21 = by_g[14][0], by_g[6][0], by_g[21][0]
    # role checks
    if t14 % 3 == 0:  # ~3 should NOT be ÷3
        return None
    if t6 % 7 == 0:  # ~7 should NOT be ÷7
        return None
    if t21 % 2 == 0:  # odd role
        return None
    up, vp, wp = t14 // 14, t6 // 6, t21 // 21
    cores = sorted(t // 42 for t in by_g[42])
    N = (B - 1) // 42
    T = (B**6 - t14**6 - t6**6 - t21**6) // (42**6)
    if T != sum(c**6 for c in cores):
        return None
    return {
        "B": B,
        "u_p": up,
        "v_p": vp,
        "w_p": wp,
        "cores": cores,
        "N": N,
        "T": T,
        "t14": t14,
        "t6": t6,
        "t21": t21,
    }


def find4_calls(T: int, N: int) -> Tuple[int, int, int]:
    """Mirror k_find4_cls5 window: c4 in [lo4,hi4], c3 in [lo3,hi3], one call each.

    lo4 ≈ first c with 4*c^6 >= T (host set_window4 uses 4*; device min_c uses 3*).
    Use device-like 3* for lo (min_c_fix style) — closer to k_find4_cls5.
    Returns (calls, lo4, hi4).
    """
    hi4 = min(iroot6(T), N)
    if hi4 < 1:
        return 0, 0, 0
    # min_c: smallest c with 3*c^6 >= T  (approx device min_c_fix @ scale 1 on T)
    lo4 = 1
    while lo4 <= hi4 and 3 * (lo4**6) < T:
        lo4 += 1
    if lo4 > hi4:
        # fallback host-style 4*
        lo4 = max(1, int(hi4 * 0.7937005260))
        while lo4 <= hi4 and 4 * (lo4**6) < T:
            lo4 += 1
    if lo4 > hi4:
        return 0, lo4, hi4

    calls = 0
    for c4 in range(lo4, hi4 + 1):
        rem = T - c4**6
        if rem <= 0:
            continue
        hi3 = min(c4, iroot6(rem), N)
        if hi3 < 1:
            continue
        lo3 = 1
        while lo3 <= hi3 and 3 * (lo3**6) < rem:
            lo3 += 1
        if lo3 > hi3:
            continue
        calls += hi3 - lo3 + 1
    return calls, lo4, hi4


# Probe gate keep fraction (solve_617 header): 27*50/124488
GATE_KEEP = 27 * 50 / 124488.0


@dataclass
class PlantResult:
    B: int
    N: int
    cores: Tuple[int, ...]
    find4_calls: int
    find4_table_est: float
    find2_probes: int
    ratio_find2_over_find4_calls: float
    ratio_find2_over_find4_table: float
    lo4: int
    hi4: int


def score_plant(info: dict) -> PlantResult:
    T, N = info["T"], info["N"]
    calls, lo4, hi4 = find4_calls(T, N)
    find2 = N * (N + 1) // 2
    table_est = calls * GATE_KEEP
    return PlantResult(
        B=info["B"],
        N=N,
        cores=tuple(info["cores"]),
        find4_calls=calls,
        find4_table_est=table_est,
        find2_probes=find2,
        ratio_find2_over_find4_calls=(find2 / calls) if calls else float("inf"),
        ratio_find2_over_find4_table=(find2 / table_est) if table_est else float("inf"),
        lo4=lo4,
        hi4=hi4,
    )


def parse_clean_line(line: str) -> Optional[Tuple[int, List[int]]]:
    # 400471 = 357609 + ... + 13860  | 2nd kind | ...
    m = re.match(
        r"^\s*(\d+)\s*=\s*([0-9 +]+)\s*\|",
        line,
    )
    if not m:
        return None
    B = int(m.group(1))
    terms = [int(x) for x in m.group(2).replace(" ", "").split("+") if x]
    if len(terms) != 7:
        return None
    return B, terms


def load_plants(path: Path, only_project: bool) -> List[Tuple[int, List[int]]]:
    out = []
    for line in path.read_text().splitlines():
        if only_project and "this project" not in line:
            continue
        parsed = parse_clean_line(line)
        if parsed:
            out.append(parsed)
    return out


# Hard-coded project cls5 from README if file missing tags
README_CLS5 = [
    (400471, [13860, 83202, 93240, 129696, 144756, 355502, 357609]),
    (421663, [85407, 112560, 231294, 258888, 296394, 306102, 392630]),
    (425003, [38997, 224130, 260526, 279454, 282702, 352800, 369348]),
    (425155, [32067, 84000, 130914, 278208, 317506, 324198, 384894]),
    (425729, [8512, 53466, 119658, 184338, 293106, 355551, 385014]),
    (427027, [62132, 66276, 188685, 189378, 201540, 352884, 397992]),
    (428329, [51450, 156681, 161100, 257964, 275478, 342384, 395038]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "617-solutions-clean.txt",
    )
    ap.add_argument(
        "--all-2nd",
        action="store_true",
        help="scan all 2nd-kind lines (not only this project)",
    )
    ap.add_argument("--readme-fallback", action="store_true", default=True)
    args = ap.parse_args()

    plants: List[Tuple[int, List[int]]] = []
    if args.file.exists():
        plants = load_plants(args.file, only_project=not args.all_2nd)
    if not plants and args.readme_fallback:
        print(f"[warn] no plants from {args.file}; using README cls5 list")
        plants = README_CLS5

    print("=== (6,1,7) cls5 plant spike: find4 calls vs deeper find2 probes ===")
    print(f"gate keep ≈ {100*GATE_KEEP:.2f}% of find4 calls → table probes\n")

    results: List[PlantResult] = []
    skipped = 0
    for B, terms in plants:
        info = classify_cls5(B, terms)
        if not info:
            skipped += 1
            continue
        r = score_plant(info)
        results.append(r)
        print(
            f"B={r.B} N={r.N} cores={list(r.cores)} "
            f"find4_window=[{r.lo4},{r.hi4}]"
        )
        print(
            f"  find4_calls={r.find4_calls:.3e}  "
            f"find4_table≈{r.find4_table_est:.3e}  "
            f"find2_probes={r.find2_probes:.3e}"
        )
        print(
            f"  find2/find4_calls={r.ratio_find2_over_find4_calls:.2f}x  "
            f"find2/find4_table≈{r.ratio_find2_over_find4_table:.2f}x"
        )

    print()
    if not results:
        print("No cls5 plants classified.")
        return 1
    ratios = [r.ratio_find2_over_find4_calls for r in results]
    print(
        f"cls5 plants={len(results)} skipped_non_cls5={skipped}  "
        f"median find2/find4_calls={sorted(ratios)[len(ratios)//2]:.2f}x  "
        f"min={min(ratios):.2f}x max={max(ratios):.2f}x"
    )
    print()
    if min(ratios) > 2:
        print(
            "VERDICT: deeper find2 does MORE inner probes than find4 on these plants "
            f"(≥{min(ratios):.1f}×). Not a free win — fusion would need to beat find4 "
            "on constants/gates, not raw probe count."
        )
    elif max(ratios) < 1:
        print(
            "VERDICT: deeper find2 does FEWER probes than find4 — worth a fused kernel spike."
        )
    else:
        print(
            "VERDICT: mixed — plant-level probe counts similar order; only a kernel "
            "microbench can claim ~10×."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
