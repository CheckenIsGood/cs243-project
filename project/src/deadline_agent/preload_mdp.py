"""Exact online policy with KV pre-loading (docs/theory.md, Section 8.2, Prop. B5).

Same assumptions as `online_mdp.ConsumptionMDP` (simultaneous launch, i.i.d. known bounded delays,
deterministic count-based costs rounded up to dt steps, decisions when the slot is idle), plus the
evidence-log prompt layout and cost model of `cache_model.PreloadCostModel`, restricted to
*removal-free* policies (a raw result that was pre-loaded is never summarized later). Then
    state = (t, r returned, k summarized, p raws in the cached log, j summaries not yet cached, z P0 cached)
is sufficient and backward induction is exact within that policy class.
Actions: ("consume", b) with b <= r-k-p; ("preload", q) appending q of the r-k-p returned, unsummarized,
not-yet-logged raws (q = 0 caches P0 / pending summaries only); WAIT (r < n); FINAL (r = n).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .affine import ceil_steps
from .cache_model import PreloadCostModel
from .online_mdp import FINAL, WAIT, Action, ConsumptionMDP, Policy, binom_pmf
from .restricted import INF

PState = tuple[int, int, int, int, int, int]  # (t, r, k, p, j, z)
PPolicy = Callable[[int, int, int, int, int, int], Action]


@dataclass
class PreloadMDP(ConsumptionMDP):
    cache: PreloadCostModel | None = None  # None -> PreloadCostModel(self.costs)
    max_summaries: int | None = None  # restrict the policy class (0 = never summarize); None = no limit

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cache is None:
            self.cache = PreloadCostModel(self.costs)
        if self.cache.base != self.costs:
            raise ValueError("cache.base must equal costs")
        self._tau_pre: dict = {}

    def tau_pre(self, k: int, p: int, j: int, z: int, q: int):
        key = (k, p, j, z, q)
        if key not in self._tau_pre:
            self._tau_pre[key] = ceil_steps(self.cache.preload(k, p, j, z, q), self.dt)
        return self._tau_pre[key]

    def final_cost(self, k: int, p: int, j: int, z: int) -> float:
        return self.cache.final(k, p, j, z, self.n)

    def pactions(self, r, k, p, j, z) -> list[Action]:
        u = r - k - p
        bmax = u if self.max_summaries is None else min(u, self.max_summaries - k)
        acts: list[Action] = [("consume", b) for b in range(1, bmax + 1) if self.tau[(k, b)] != INF]
        acts += [("preload", q) for q in range(0, u + 1)
                 if (q > 0 or j > 0 or z == 0) and self.tau_pre(k, p, j, z, q) != INF]
        if r < self.n:
            acts.append(WAIT)
        elif self.final_cost(k, p, j, z) < INF:
            acts.append(FINAL)
        return acts

    def psuccessors(self, s: PState, a: Action):
        t, r, k, p, j, z = s
        kind, b = a
        if kind == "final":
            yield 1.0, t * self.dt + self.final_cost(k, p, j, z)
            return
        if kind == "wait":
            t2, nxt = t + 1, (k, p, j, z)
        elif kind == "consume":
            if not 1 <= b <= r - k - p:
                raise ValueError(f"policy consumed {b} with only {r - k - p} returned, raw and unlogged")
            t2, nxt = t + self.tau[(k, b)], (k + b, p, j + b, z)
        elif kind == "preload":
            if not 0 <= b <= r - k - p or (b == 0 and j == 0 and z == 1):
                raise ValueError(f"invalid preload of {b}")
            t2, nxt = t + self.tau_pre(k, p, j, z, b), (k, p + b, 0, 1)
        else:
            raise ValueError(kind)
        if t2 == INF:
            raise ValueError("infeasible action")
        for jj, pr in enumerate(binom_pmf(self.n - r, self.release_prob(t, t2))):
            if pr > 0:
                yield pr, (t2, r + jj) + nxt

    def _pvalue(self, s: PState, policy: PPolicy | None, memo: dict, pi: dict | None) -> float:
        if s in memo:
            return memo[s]
        acts = [policy(*s)] if policy else self.pactions(*s[1:])
        best, best_a = INF, None
        for a in acts:
            v = 0.0
            for pr, nxt in self.psuccessors(s, a):
                v += pr * (self.loss(nxt) if a == FINAL else self._pvalue(nxt, policy, memo, pi))
            if v < best - 1e-12:
                best, best_a = v, a
        memo[s] = best
        if pi is not None:
            pi[s] = best_a if best_a is not None else FINAL
        return best

    def _pstart(self, fn) -> float:
        import sys

        sys.setrecursionlimit(max(sys.getrecursionlimit(), 8 * (self.S + 10 * self.n + 100)))
        return sum(pr * fn((0, jj, 0, 0, 0, 0)) for jj, pr in enumerate(binom_pmf(self.n, self.G(0))) if pr > 0)

    def solve(self) -> float:  # type: ignore[override]
        self._PV, self._Ppi = {}, {}
        return self._pstart(lambda s: self._pvalue(s, None, self._PV, self._Ppi))

    def evaluate(self, policy: PPolicy) -> float:  # type: ignore[override]
        memo: dict = {}
        return self._pstart(lambda s: self._pvalue(s, policy, memo, None))

    def optimal_policy(self) -> PPolicy:  # type: ignore[override]
        if not getattr(self, "_Ppi", None):
            self.solve()

        def pi(*s) -> Action:
            if s not in self._Ppi:
                self._pvalue(s, None, self._PV, self._Ppi)
            return self._Ppi[s]

        return pi

    def simulate(self, policy: PPolicy, delays: list[int]) -> float:  # type: ignore[override]
        t, k, p, j, z = 0, 0, 0, 0, 0
        while True:
            r = sum(d <= t for d in delays)
            kind, b = a = policy(t, r, k, p, j, z)
            if kind == "final":
                return t * self.dt + self.final_cost(k, p, j, z)
            if kind in ("consume", "preload") and b > r - k - p:
                raise RuntimeError("policy used unreturned results")
            for _, nxt in self.psuccessors((t, r, k, p, j, z), a):
                t, k, p, j, z = nxt[0], *nxt[2:]
                break


# ---- policy adapters ----------------------------------------------------------------------
def no_preload(base: Policy) -> PPolicy:
    """Lift a (t, r, k) policy; it never pre-loads, so p = 0, j = k, z = 0 throughout."""
    return lambda t, r, k, p, j, z: base(t, r, k)


def with_idle_preload(mdp: PreloadMDP, base: Policy) -> PPolicy:
    """Run `base`, but whenever it would WAIT, pre-load what can safely be cached.

    Caches P0 and pending summaries, and appends returned raws beyond those still needed for the
    min_summaries_needed() summaries the final prompt requires (removal-free: a logged raw stays raw).
    Consumes are clamped to the raws not yet logged.
    """
    need = mdp.costs.min_summaries_needed(mdp.n)

    def pi(t, r, k, p, j, z) -> Action:
        kind, b = base(t, r, k)
        u = r - k - p
        if kind == "consume":
            b = min(b, u)
            if b >= 1:
                return ("consume", b)
            kind = "wait" if r < mdp.n else "final"
        if kind == "final":
            return FINAL
        reserve = max(0, need - k)
        q = max(0, min(u - reserve, (mdp.n - need) - p))
        if (q > 0 or j > 0 or z == 0) and mdp.tau_pre(k, p, j, z, q) != INF:
            return ("preload", q)
        return WAIT

    return pi


def sample_and_run(mdp: PreloadMDP, policy: PPolicy, runs: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [mdp.simulate(policy, mdp.sample_delays(rng)) for _ in range(runs)]
