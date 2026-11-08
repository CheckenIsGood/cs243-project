"""Executable checks for docs/general_theory.md (Theorem 1, Lemmas 2-3).

Small exact solvers and constructions used by tests/test_general_theory.py to verify the proofs on exhaustive
small instances and randomized property tests. Z3 proofs of the algebraic steps of Lemma 2 live in the tests.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


# ---------------------------------------------------------------- Theorem 1: 1 || sum w_j U_j  vs  Knapsack
def knapsack_opt(weights: list[int], values: list[int], capacity: int) -> int:
    best = [0] * (capacity + 1)
    for wt, v in zip(weights, values):
        for c in range(capacity, wt - 1, -1):
            best[c] = max(best[c], best[c - wt] + v)
    return best[capacity]


def single_machine_on_time_value(p: list[float], w: list[float], d: list[float], r: list[float] | None = None) -> float:
    """Exact max total weight of jobs finishing by their deadlines on ONE non-preemptive machine (batch size 1),
    with optional release dates. Exhaustive over ordered subsets; n <= 7."""
    n = len(p)
    r = r or [0.0] * n
    best = 0.0
    for k in range(n + 1):
        for subset in itertools.combinations(range(n), k):
            for order in itertools.permutations(subset):
                t, ok = 0.0, True
                for j in order:
                    t = max(t, r[j]) + p[j]
                    if t > d[j] + 1e-12:
                        ok = False
                        break
                if ok:
                    best = max(best, sum(w[j] for j in subset))
                    break  # this subset is feasible; other orders give the same value
    return best


# ---------------------------------------------------------------- Lemma 2: McNaughton's wrap-around rule
@dataclass(frozen=True)
class Piece:
    job: int
    machine: int
    start: float
    end: float


def mcnaughton(rates: list[float], m: int, s: float, length: float) -> list[Piece]:
    """Realize per-job rates (work per second) over [0, length] on m identical machines of speed s with preemption
    and migration. Requires every rate <= s and sum(rates) <= m*s (McNaughton 1959 wrap-around)."""
    if any(x < -1e-12 or x > s + 1e-12 for x in rates) or sum(rates) > m * s + 1e-9:
        raise ValueError("infeasible rate vector")
    pieces, machine, t = [], 0, 0.0
    for j, x in enumerate(rates):
        need = x * length / s  # machine-time needed at speed s
        while need > 1e-12:
            if machine >= m:
                raise AssertionError("ran out of machines (should be impossible)")
            take = min(need, length - t)
            pieces.append(Piece(j, machine, t, t + take))
            need -= take
            t += take
            if t >= length - 1e-12:
                machine, t = machine + 1, 0.0
    return pieces


def check_mcnaughton(pieces: list[Piece], rates: list[float], m: int, s: float, length: float) -> None:
    """Assert the schedule is valid: within [0, length], no machine runs two pieces at once, no job runs on two
    machines at once, and each job receives exactly rate*length work."""
    for pc in pieces:
        assert -1e-9 <= pc.start <= pc.end <= length + 1e-9 and pc.machine < m
    for key in ("machine", "job"):
        groups: dict[int, list[Piece]] = {}
        for pc in pieces:
            groups.setdefault(getattr(pc, key), []).append(pc)
        for g in groups.values():
            g = sorted(g, key=lambda q: q.start)
            for a, b in zip(g, g[1:]):
                assert a.end <= b.start + 1e-9, f"overlap on {key}"
    for j, x in enumerate(rates):
        work = sum((pc.end - pc.start) * s for pc in pieces if pc.job == j)
        assert abs(work - x * length) < 1e-6


# ---------------------------------------------------------------- Lemma 2(i): engine simulates M_m(1/(2a))
def engine_completion_times(machine_schedule: list[tuple[float, frozenset[int]]], work: dict[int, float],
                            a: float, c: float, dt: float = 1e-3) -> dict[int, float]:
    """Run the continuous-batching engine (per-sequence rate 1/(a + c*B)) following a machine schedule given as
    [(segment end time, set of jobs the m-machine schedule runs in that segment)]. A job the engine has already
    finished is dropped from the batch. Returns engine completion times."""
    left = dict(work)
    done: dict[int, float] = {}
    t, seg = 0.0, 0
    while left and seg < len(machine_schedule):
        end, running = machine_schedule[seg]
        if t >= end - 1e-12:
            seg += 1
            continue
        batch = [j for j in running if j in left]
        step = min(dt, end - t)
        if batch:
            rate = 1.0 / (a + c * len(batch))
            for j in batch:
                left[j] -= rate * step
                if left[j] <= 1e-9:
                    done[j] = t + step
                    del left[j]
        t += step
    return done


def machine_completion_times(machine_schedule: list[tuple[float, frozenset[int]]], work: dict[int, float],
                             s: float, dt: float = 1e-3) -> dict[int, float]:
    left = dict(work)
    done: dict[int, float] = {}
    t, seg = 0.0, 0
    while left and seg < len(machine_schedule):
        end, running = machine_schedule[seg]
        if t >= end - 1e-12:
            seg += 1
            continue
        step = min(dt, end - t)
        for j in [j for j in running if j in left]:
            left[j] -= s * step
            if left[j] <= 1e-9:
                done[j] = t + step
                del left[j]
        t += step
    return done


# ---------------------------------------------------------------- Lemma 3: eager tool issue (offline optimum)
def tool_then_llm_opt(tool: list[float], issue: list[float], p: list[float], w: list[float], d: list[float]) -> float:
    """Each task: a tool call issued at issue[j] taking tool[j] seconds (off-GPU, uncapacitated), then one LLM job
    of length p[j] on a single non-preemptive GPU slot; deadline d[j]. Exact max on-time value."""
    r = [s + t for s, t in zip(issue, tool)]
    return single_machine_on_time_value(p, w, d, r)
