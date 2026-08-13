#!/usr/bin/env python3
"""Score Euler-sum search partitions for cls5-style fused peel+find2 speedups.

Layer A: score_partition(plan) → P1–P7 + cls5_shaped.
Layer B: enumerate_find2_plans(k, m, n) → candidate role assignments.

LPS form (k, m, n) means m kth powers = n kth powers.
Special case n==1: m terms = B^k (as in (6,1,5)).

This does NOT invent Meyrignac modular classes — only architectural shapes.

Examples:
  python3 tools/euler_peel_score.py --demo
  python3 tools/euler_peel_score.py --k 6 --m 5 --n 1
  python3 tools/euler_peel_score.py --k 9 --m 2 --n 8 --B 1e6
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Roles for one algebraic term (a slot in the identity).
PAIR = "PAIR"  # goes into prebuilt {i^k + j^k} store (exactly 2 such)
FREE = "FREE"  # free peel index (scaled term peeled on device)
UNIT = "UNIT"  # sparse Stage-1 unit / co-factor
FIXED = "FIXED"  # known / absorbed into RHS (e.g. B^k on the other side)


@dataclass
class SearchPlan:
    """Explicit search architecture for one side of an Euler sum."""

    k: int
    m: int  # terms on the multi-power side (or total unknowns if n>1)
    n: int  # other side (# of kth powers); 1 means single B^k
    roles: Tuple[str, ...]  # length m (unknowns on the peeled side)
    scale_S: Optional[int] = None  # master scale for ÷ S^k if known (e.g. 42)
    B_target: float = 3.0e6
    # Per free term: range ~ B / scale; default scale 1 → range ~ B
    free_scales: Tuple[int, ...] = ()
    label: str = ""

    def n_pair(self) -> int:
        return sum(1 for r in self.roles if r == PAIR)

    def n_free(self) -> int:
        return sum(1 for r in self.roles if r == FREE)

    def n_unit(self) -> int:
        return sum(1 for r in self.roles if r == UNIT)

    def residual_arity(self) -> int:
        return self.n_pair()


@dataclass
class Score:
    plan: SearchPlan
    P1: bool  # residual arity == 2
    P2: bool  # n_free >= 2
    P3: bool  # leaf is O(1) probe (equiv. residual arity == 2 here)
    P4: bool  # exact scale reduction available / assumed
    P5: bool  # funnel feasible heuristic
    P6: bool  # host expand catastrophic
    P7: bool  # stage-1 sparse (has UNIT or n==1 anchor)
    free_grid: float
    cls5_shaped: bool
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        flags = " ".join(
            f"P{i}={'Y' if v else 'n'}"
            for i, v in enumerate(
                [self.P1, self.P2, self.P3, self.P4, self.P5, self.P6, self.P7], 1
            )
        )
        tag = "CLS5_SHAPED" if self.cls5_shaped else "no"
        lab = self.plan.label or roles_str(self.plan.roles)
        return (
            f"{lab:40s} free={self.plan.n_free()} pair={self.plan.n_pair()} "
            f"grid~{self.free_grid:.3e} {flags} => {tag}"
        )


def roles_str(roles: Sequence[str]) -> str:
    return "[" + ",".join(roles) + "]"


def estimate_free_grid(plan: SearchPlan) -> float:
    """Rough ∏ ranges for free peels at B_target (CRT thinning ignored → upper bound)."""
    B = plan.B_target
    n_free = plan.n_free()
    if n_free == 0:
        return 1.0
    scales = list(plan.free_scales)
    while len(scales) < n_free:
        scales.append(1)
    prod = 1.0
    for s in scales[:n_free]:
        prod *= max(1.0, B / max(1, s))
    return prod


def score_partition(plan: SearchPlan, grid_threshold: float = 1e10) -> Score:
    """Layer A: evaluate P1–P7 on an explicit plan."""
    notes: List[str] = []
    P1 = plan.residual_arity() == 2
    P2 = plan.n_free() >= 2
    P3 = P1
    if plan.residual_arity() == 3:
        notes.append("residual find3 — fuse peels help little vs cls5-class win")
    if plan.residual_arity() >= 4:
        notes.append("residual find4+ — leaf dominates")

    P4 = plan.scale_S is not None or plan.n == 1
    if not P4:
        notes.append("no scale_S and n!=1 — mark P4 false until reduction known")

    if plan.scale_S is not None:
        P5 = True
        notes.append(f"funnel assumed feasible for S={plan.scale_S}")
    elif plan.n == 1:
        P5 = True
    else:
        P5 = False
        notes.append("P5 unknown without scale structure")

    grid = estimate_free_grid(plan)
    P6 = P2 and grid >= grid_threshold
    if P2 and not P6:
        notes.append(f"free grid {grid:.2e} < threshold {grid_threshold:.0e}")

    P7 = plan.n_unit() >= 1 or plan.n == 1

    cls5 = bool(P1 and P2 and P3 and P4 and P6)
    return Score(
        plan=plan,
        P1=P1,
        P2=P2,
        P3=P3,
        P4=P4,
        P5=P5,
        P6=P6,
        P7=P7,
        free_grid=grid,
        cls5_shaped=cls5,
        notes=notes,
    )


def enumerate_find2_plans(
    k: int,
    m: int,
    n: int,
    *,
    max_free: int = 4,
    require_unit: bool = True,
    B_target: float = 3.0e6,
    scale_S: Optional[int] = None,
) -> List[SearchPlan]:
    """Layer B: role assignments on m slots with exactly 2 PAIR and >=2 FREE."""
    if m < 4:
        return []
    plans: List[SearchPlan] = []
    for pair_idx in itertools.combinations(range(m), 2):
        rest = [i for i in range(m) if i not in pair_idx]
        for n_free in range(2, min(max_free, len(rest)) + 1):
            for free_idx in itertools.combinations(rest, n_free):
                leftover = [i for i in rest if i not in free_idx]
                if require_unit:
                    if not leftover:
                        continue
                    for u_i in leftover:
                        roles = [FIXED] * m
                        for i in pair_idx:
                            roles[i] = PAIR
                        for i in free_idx:
                            roles[i] = FREE
                        roles[u_i] = UNIT
                        for i in leftover:
                            if i != u_i:
                                roles[i] = FIXED
                        plans.append(
                            SearchPlan(
                                k=k,
                                m=m,
                                n=n,
                                roles=tuple(roles),
                                scale_S=scale_S,
                                B_target=B_target,
                                label=f"({k},{m},{n}) {roles_str(roles)}",
                            )
                        )
                else:
                    roles = [FIXED] * m
                    for i in pair_idx:
                        roles[i] = PAIR
                    for i in free_idx:
                        roles[i] = FREE
                    plans.append(
                        SearchPlan(
                            k=k,
                            m=m,
                            n=n,
                            roles=tuple(roles),
                            scale_S=scale_S,
                            B_target=B_target,
                            label=f"({k},{m},{n}) {roles_str(roles)}",
                        )
                    )
    uniq = {}
    for p in plans:
        uniq[p.roles] = p
    return list(uniq.values())


def known_controls(B_target: float = 3.0e6) -> List[Tuple[str, SearchPlan]]:
    """Golden / negative controls from (6,1,5) Meyrignac practice."""
    cls5 = SearchPlan(
        k=6,
        m=5,
        n=1,
        roles=(PAIR, PAIR, FREE, FREE, UNIT),
        scale_S=42,
        B_target=B_target,
        free_scales=(21, 14),
        label="(6,1,5) cls5-like [PAIR,PAIR,FREE,FREE,UNIT]",
    )
    cls2 = SearchPlan(
        k=6,
        m=5,
        n=1,
        roles=(PAIR, PAIR, PAIR, FREE, UNIT),
        scale_S=42,
        B_target=B_target,
        free_scales=(14,),
        label="(6,1,5) cls2-like [PAIR×3,FREE,UNIT] find3",
    )
    cls1 = SearchPlan(
        k=6,
        m=5,
        n=1,
        roles=(PAIR, PAIR, PAIR, PAIR, UNIT),
        scale_S=42,
        B_target=B_target,
        label="(6,1,5) cls1-like [PAIR×4,UNIT] find4",
    )
    return [("cls5+", cls5), ("cls2-", cls2), ("cls1-", cls1)]


def run_demo(B_target: float = 3.0e6) -> int:
    print("=== controls (expect cls5+ => CLS5_SHAPED, others not) ===")
    ok = True
    for name, plan in known_controls(B_target):
        sc = score_partition(plan)
        print(sc.summary())
        for n in sc.notes:
            print(f"  note: {n}")
        if name.startswith("cls5") and not sc.cls5_shaped:
            ok = False
            print("  FAIL: expected cls5_shaped")
        if name.startswith("cls2") and sc.cls5_shaped:
            ok = False
            print("  FAIL: cls2 should not be cls5_shaped")
        if name.startswith("cls1") and sc.cls5_shaped:
            ok = False
            print("  FAIL: cls1 should not be cls5_shaped")
    print()
    print("=== enumerate (6,1,5) find2-shaped plans ===")
    plans = enumerate_find2_plans(6, 5, 1, B_target=B_target, scale_S=42)
    shaped = [score_partition(p) for p in plans]
    shaped.sort(key=lambda s: (-int(s.cls5_shaped), -s.free_grid, s.plan.n_free()))
    print(f"plans={len(shaped)} cls5_shaped={sum(1 for s in shaped if s.cls5_shaped)}")
    for sc in shaped[:12]:
        print(sc.summary())
    print()
    print("=== sample (9,2,8) and (6,6,6) top shapes ===")
    for k, m, n in [(9, 2, 8), (6, 6, 6)]:
        side = max(m, n)
        plans = enumerate_find2_plans(
            k, side, 1 if min(m, n) == 1 else min(m, n), B_target=B_target
        )
        scored = [score_partition(p) for p in plans]
        scored.sort(key=lambda s: (-int(s.cls5_shaped), -s.free_grid))
        n_ok = sum(1 for s in scored if s.cls5_shaped)
        print(f"({k},{m},{n}) side_slots={side} plans={len(scored)} cls5_shaped={n_ok}")
        for sc in scored[:5]:
            print(" ", sc.summary())
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="run controls + samples")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--B", type=float, default=3.0e6, help="target B for grid estimate")
    ap.add_argument("--S", type=int, default=None, help="optional master scale S")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--all-shaped", action="store_true", help="only print cls5_shaped")
    args = ap.parse_args()

    if args.demo or (args.k is None):
        return run_demo(args.B)

    assert args.m is not None and args.n is not None
    side = args.m if args.n == 1 else max(args.m, args.n)
    plans = enumerate_find2_plans(
        args.k, side, args.n if args.n != 1 else 1, B_target=args.B, scale_S=args.S
    )
    scored = [score_partition(p) for p in plans]
    scored.sort(key=lambda s: (-int(s.cls5_shaped), -s.free_grid, -s.plan.n_free()))
    if args.all_shaped:
        scored = [s for s in scored if s.cls5_shaped]
    print(
        f"({args.k},{args.m},{args.n}) slots={side} plans={len(plans)} "
        f"shown={min(args.top, len(scored))} "
        f"cls5_shaped={sum(1 for s in scored if s.cls5_shaped)}"
    )
    for sc in scored[: args.top]:
        print(sc.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
