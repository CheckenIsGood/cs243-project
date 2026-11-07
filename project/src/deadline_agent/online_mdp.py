"""Exact online (nonanticipative) consumption policy for the exchangeable case.

Assumptions (docs/theory.md, Section 4):
  * n calls launched together at time 0 with i.i.d. admission-to-return delays
    drawn from a known discrete distribution (step dt);
  * count-based costs (AffineCostModel), deterministic, rounded up to steps;
  * decisions only when the model is idle; no new admissions.
Then the state (t, r = #returned, k = #summarized) is sufficient, and backward
induction gives the policy minimizing E[loss(C)] for any nondecreasing loss.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Literal

from .affine import AffineCostModel, ceil_steps
from .restricted import INF

Action = tuple[str, int]  # ("consume", b) | ("wait", 0) | ("final", 0)
WAIT: Action = ("wait", 0)
FINAL: Action = ("final", 0)
Policy = Callable[[int, int, int], Action]  # (t_step, r, k) -> action


def binom_pmf(m: int, p: float) -> list[float]:
    p = min(max(p, 0.0), 1.0)
    return [math.comb(m, j) * p**j * (1 - p) ** (m - j) for j in range(m + 1)]


@dataclass
class ConsumptionMDP:
    n: int
    costs: AffineCostModel
    delay_pmf: list[float]  # P(T = s*dt), s = 0..S
    dt: float
    deadline_s: float
    objective: Literal["miss", "tardiness", "time"] = "miss"
    tie_eps: float = 1e-3

    def __post_init__(self) -> None:
        tot = sum(self.delay_pmf)
        if not math.isclose(tot, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"delay pmf sums to {tot}")
        self.S = len(self.delay_pmf) - 1
        self.cdf, acc = [], 0.0
        for p in self.delay_pmf:
            acc += p
            self.cdf.append(min(acc, 1.0))
        # pin to exactly 1 from the last support point, so float error cannot leave
        # a call "pending forever" (which would make WAIT recurse without end)
        last = max(s for s, p in enumerate(self.delay_pmf) if p > 0)
        for s in range(last, self.S + 1):
            self.cdf[s] = 1.0
        self.tau = {
            (k, b): ceil_steps(self.costs.consume(k, b), self.dt)
            for k in range(self.n + 1)
            for b in range(1, self.n - k + 1)
        }
        self._V: dict[tuple[int, int, int], float] = {}
        self._pi: dict[tuple[int, int, int], Action] = {}

    # ---- model primitives -------------------------------------------------
    def G(self, s: int) -> float:
        return 0.0 if s < 0 else self.cdf[min(s, self.S)]

    def release_prob(self, t: int, t2: int) -> float:
        """P(pending call returns by t2 | still pending at t)."""
        surv = 1.0 - self.G(t)
        return 1.0 if surv <= 1e-15 else (self.G(t2) - self.G(t)) / surv

    def loss(self, completion_s: float) -> float:
        if completion_s == INF:
            return INF
        tie = self.tie_eps * completion_s / max(self.deadline_s, 1e-9)
        if self.objective == "miss":
            return float(completion_s > self.deadline_s + 1e-9) + tie
        if self.objective == "tardiness":
            return max(completion_s - self.deadline_s, 0.0) + tie
        return completion_s

    def actions(self, r: int, k: int) -> list[Action]:
        acts: list[Action] = [("consume", b) for b in range(1, r - k + 1) if self.tau[(k, b)] != INF]
        if r < self.n:
            acts.append(WAIT)
        elif self.costs.final(k, self.n) < INF:
            acts.append(FINAL)
        return acts

    def successors(self, t: int, r: int, k: int, a: Action):
        """Yield (prob, next_state) or (1.0, completion_seconds) for FINAL."""
        kind, b = a
        if kind == "final":
            yield 1.0, t * self.dt + self.costs.final(k, self.n)
            return
        if kind == "consume" and not 1 <= b <= r - k:
            raise ValueError(f"policy consumed {b} results with only {r - k} returned and raw")
        t2 = t + (1 if kind == "wait" else self.tau[(k, b)])
        k2 = k + b
        for j, p in enumerate(binom_pmf(self.n - r, self.release_prob(t, t2))):
            if p > 0:
                yield p, (t2, r + j, k2)

    # ---- solve / evaluate --------------------------------------------------
    def _value(self, state, policy: Policy | None, memo: dict, pi: dict | None) -> float:
        if state in memo:
            return memo[state]
        t, r, k = state
        acts = [policy(t, r, k)] if policy else self.actions(r, k)
        best, best_a = INF, None
        for a in acts:
            v = 0.0
            for p, nxt in self.successors(t, r, k, a):
                v += p * (self.loss(nxt) if a == FINAL else self._value(nxt, policy, memo, pi))
            if v < best - 1e-12:
                best, best_a = v, a
        memo[state] = best
        if pi is not None:
            # infeasible states (no finite action) fall back to FINAL; the runtime guard then handles it
            pi[state] = best_a if best_a is not None else FINAL
        return best

    def _start(self, value_fn) -> float:
        import sys

        sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * (self.S + 10 * self.n + 100)))
        return sum(p * value_fn((0, j, 0)) for j, p in enumerate(binom_pmf(self.n, self.G(0))) if p > 0)

    def solve(self) -> float:
        """Optimal expected loss from launch; fills the policy table."""
        self._V, self._pi = {}, {}
        return self._start(lambda s: self._value(s, None, self._V, self._pi))

    def evaluate(self, policy: Policy) -> float:
        """Exact expected loss of an arbitrary state-feedback policy."""
        memo: dict = {}
        return self._start(lambda s: self._value(s, policy, memo, None))

    def optimal_policy(self) -> Policy:
        if not self._pi:
            self.solve()

        def pi(t: int, r: int, k: int) -> Action:
            if (t, r, k) not in self._pi:
                self._value((t, r, k), None, self._V, self._pi)
            return self._pi[(t, r, k)]

        return pi

    # ---- Monte Carlo check -------------------------------------------------
    def sample_delays(self, rng: random.Random) -> list[int]:
        return rng.choices(range(self.S + 1), weights=self.delay_pmf, k=self.n)

    def simulate(self, policy: Policy, delays: list[int]) -> float:
        """Run a policy against concrete delays; returns completion seconds."""
        t, k = 0, 0
        while True:
            r = sum(d <= t for d in delays)
            kind, b = policy(t, r, k)
            if kind == "final":
                return t * self.dt + self.costs.final(k, self.n)
            if kind == "wait":
                t += 1
            else:
                if b > r - k:
                    raise RuntimeError("policy consumed unreturned results")
                t += self.tau[(k, b)]
                k += b


# ---- static baseline policies (same state, same information) -------------------
def static_incremental(mdp: ConsumptionMDP) -> Policy:
    def pi(t: int, r: int, k: int) -> Action:
        if r - k >= 1 and mdp.tau[(k, 1)] != INF:
            return ("consume", 1)
        return FINAL if r == mdp.n else WAIT

    return pi


def static_fold(mdp: ConsumptionMDP) -> Policy:
    """Wait for everything; summarize only the minimum needed to fit, in one batch."""
    need = mdp.costs.min_summaries_needed(mdp.n)

    def pi(t: int, r: int, k: int) -> Action:
        if r < mdp.n:
            return WAIT
        if k < need:
            b = max(b for b in range(1, need - k + 1) if mdp.tau[(k, b)] != INF)
            return ("consume", b)
        return FINAL

    return pi


def static_batch(mdp: ConsumptionMDP, threshold: int) -> Policy:
    """Summarize all ready results once `threshold` are waiting; fold the tail."""
    fold = static_fold(mdp)

    def pi(t: int, r: int, k: int) -> Action:
        u = r - k
        if u >= threshold:
            b = max((b for b in range(1, u + 1) if mdp.tau[(k, b)] != INF), default=0)
            if b:
                return ("consume", b)
        return fold(t, r, k)

    return pi


def bounded_delay_pmf(dt: float, lo_s: float, hi_s: float, slow_p: float = 0.0,
                      slow_lo_s: float = 0.0, slow_hi_s: float = 0.0) -> list[float]:
    """Uniform[lo, hi] mixed with probability slow_p of uniform[slow_lo, slow_hi]."""
    S = math.ceil(max(hi_s, slow_hi_s) / dt)
    pmf = [0.0] * (S + 1)

    def add(lo: float, hi: float, w: float) -> None:
        a, b = round(lo / dt), round(hi / dt)
        for s in range(a, b + 1):
            pmf[s] += w / (b - a + 1)

    add(lo_s, hi_s, 1.0 - slow_p)
    if slow_p > 0:
        add(slow_lo_s, slow_hi_s, slow_p)
    return pmf


def required_summary_slack(mdp: ConsumptionMDP, batch_wait_prob: float = 0.5) -> Policy:
    """Closed-form heuristic distilled from the optimal policy (docs/theory.md, Section 5).

    1. Summarize only the m = min_summaries_needed results the final prompt requires.
    2. Summarize them as soon as they are ready (use idle model time).
    3. If fewer are ready than still needed, wait for one more only when it is likely
       to arrive within the time a combined batch would save.
    4. After m summaries, wait for all results and fold the rest raw into the answer.
    """
    need = mdp.costs.min_summaries_needed(mdp.n)

    def pi(t: int, r: int, k: int) -> Action:
        u = r - k
        if k >= need:
            return FINAL if r == mdp.n else WAIT
        # largest batch that is both needed and feasible (a batch's own prompt must fit the context)
        b = max((b for b in range(1, min(u, need - k) + 1) if mdp.tau[(k, b)] != INF), default=0)
        if b == 0:
            return WAIT if r < mdp.n else FINAL  # nothing feasible to summarize; FINAL falls to the runtime guard
        if b < need - k and r < mdp.n and mdp.tau.get((k, b + 1), INF) != INF:
            save = mdp.tau[(k, b)] + mdp.tau[(k + b, 1)] - mdp.tau[(k, b + 1)]
            if save > 0:
                q = mdp.release_prob(t, t + save)
                if 1 - (1 - q) ** (mdp.n - r) >= batch_wait_prob:
                    return WAIT
        return ("consume", b)

    return pi
