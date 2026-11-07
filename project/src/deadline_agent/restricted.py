"""Restricted deterministic result-consumption problem (PLAN.MD Section 6.2).

n admitted results with known release times r_i, one nonpreemptive model slot.
Results may be summarized in batches (cost c(A, B) after set A was summarized)
and the final invocation reads summaries of S plus all raw results N \\ S
(cost f(S)). Every result must be consumed. Sets are bitmasks over range(n).

`solve_dp` is the subset DP; `solve_exhaustive` is an independent forward
enumeration of every ordered sequence of disjoint batches, used only to verify it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

INF = math.inf
ConsumeCost = Callable[[int, int], float]  # (A mask, B mask) -> seconds, inf if infeasible
FinalCost = Callable[[int], float]  # (S mask) -> seconds, inf if infeasible


@dataclass(frozen=True)
class Instance:
    releases: tuple[float, ...]
    consume_cost: ConsumeCost
    final_cost: FinalCost

    @property
    def n(self) -> int:
        return len(self.releases)

    def release_max(self, mask: int) -> float:
        return max((self.releases[i] for i in range(self.n) if mask >> i & 1), default=0.0)


@dataclass
class Schedule:
    completion: float  # modeled final completion time; inf if infeasible
    batches: list[tuple[int, float, float]] = field(default_factory=list)  # (mask, start, end)
    final_summarized: int = 0
    final_start: float = INF

    @property
    def feasible(self) -> bool:
        return self.completion < INF


def iter_nonempty_submasks(mask: int):
    sub = mask
    while sub:
        yield sub
        sub = (sub - 1) & mask


def solve_dp(inst: Instance) -> Schedule:
    """O(3^n + n 2^n) subset DP. Returns an optimal schedule under the model."""
    n = inst.n
    full = (1 << n) - 1
    rmax = [inst.release_max(m) for m in range(1 << n)]
    F = [INF] * (1 << n)
    pred: list[tuple[int, int] | None] = [None] * (1 << n)
    F[0] = 0.0
    for S in sorted(range(1, 1 << n), key=int.bit_count):
        for B in iter_nonempty_submasks(S):
            A = S ^ B
            if F[A] == INF:
                continue
            cost = inst.consume_cost(A, B)
            if cost == INF:
                continue
            cand = max(F[A], rmax[B]) + cost
            if cand < F[S]:
                F[S], pred[S] = cand, (A, B)
    best, best_S = INF, 0
    for S in range(1 << n):
        if F[S] == INF:
            continue
        fc = inst.final_cost(S)
        if fc == INF:
            continue
        cand = max(F[S], rmax[full ^ S]) + fc
        if cand < best:
            best, best_S = cand, S
    if best == INF:
        return Schedule(INF)
    # reconstruct
    batches = []
    S = best_S
    while S:
        A, B = pred[S]  # type: ignore[misc]
        batches.append((B, max(F[A], rmax[B]), F[S]))
        S = A
    batches.reverse()
    return Schedule(best, batches, best_S, best - inst.final_cost(best_S))


def simulate(inst: Instance, batch_seq: list[int]) -> Schedule:
    """Earliest-start timing of an explicit batch sequence followed by finalization."""
    t, A, out = 0.0, 0, []
    for B in batch_seq:
        cost = inst.consume_cost(A, B)
        if cost == INF:
            return Schedule(INF)
        start = max(t, inst.release_max(B))
        t = start + cost
        out.append((B, start, t))
        A |= B
    fc = inst.final_cost(A)
    if fc == INF:
        return Schedule(INF)
    full = (1 << inst.n) - 1
    start = max(t, inst.release_max(full ^ A))
    return Schedule(start + fc, out, A, start)


def solve_exhaustive(inst: Instance) -> Schedule:
    """Enumerate every ordered sequence of disjoint nonempty batches (independent of the DP)."""
    n = inst.n
    best = Schedule(INF)

    def rec(used: int, seq: list[int]) -> None:
        nonlocal best
        s = simulate(inst, seq)
        if s.completion < best.completion:
            best = s
        remaining = ((1 << n) - 1) ^ used
        for B in iter_nonempty_submasks(remaining):
            seq.append(B)
            rec(used | B, seq)
            seq.pop()

    rec(0, [])
    return best
