"""Extension A: speculative decoding (docs/theory.md, Section 8.1).

Two pieces, both backward compatible with `online_mdp.ConsumptionMDP`:

* Deterministic per-type decode rates live in `AffineCostModel.consume_decode_s_per_tok` /
  `final_decode_s_per_tok` (default: the shared `decode_s_per_tok`). `speculative_costs` builds one.
* `StochasticDurationMDP` lets every action's duration be random: a duration distribution per
  CONSUME(k, b) and per FINAL(k). If each duration is drawn independently of the tool delays and of
  every other duration, with a law depending only on (action, k, b), the state (t, r, k) is still
  sufficient and averaging the transitions over the duration law is exact (Prop. 4A).

A speculative-decoding speedup is modeled as a random factor on the *decode* part only: prompt-lookup
decoding verifies drafts with the target model, so prefill is unchanged and c(k, b) >= beta_p * L * b
still holds pathwise. That inequality is all Props 1-2 need.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .affine import AffineCostModel, ceil_steps
from .online_mdp import FINAL, Action, ConsumptionMDP, Policy, binom_pmf
from .restricted import INF

Speedups = list[tuple[float, float]]  # (decode speedup factor >= 1, probability)
StepPmf = list[tuple[int, float]]  # (duration in dt steps, probability)
SecPmf = list[tuple[float, float]]  # (duration in seconds, probability)


def speculative_costs(base: AffineCostModel, consume_speedup: float = 1.0,
                      final_speedup: float = 1.0) -> AffineCostModel:
    """Deterministic per-type decode rates: beta_d / speedup for summaries and the final answer."""
    return replace(base, consume_decode_s_per_tok=base.decode_s_per_tok / consume_speedup,
                   final_decode_s_per_tok=base.decode_s_per_tok / final_speedup)


def mean_speedup_costs(base: AffineCostModel, consume: Speedups, final: Speedups) -> AffineCostModel:
    """Deterministic-mean approximation: decode time per token = beta_d * E[1/factor]."""
    def inv_mean(sp: Speedups) -> float:
        return sum(p / f for f, p in sp)

    return replace(base, consume_decode_s_per_tok=base.decode_s_per_tok * inv_mean(consume),
                   final_decode_s_per_tok=base.decode_s_per_tok * inv_mean(final))


def _merge(pairs):
    out: dict = {}
    for v, p in pairs:
        if p > 0:
            out[v] = out.get(v, 0.0) + p
    return sorted(out.items())


@dataclass
class StochasticDurationMDP(ConsumptionMDP):
    """ConsumptionMDP whose CONSUME and FINAL durations are random (independent across actions).

    consume_speedups / final_speedups: distributions of the decode speedup factor. Durations are
    alpha + beta_p * prompt + (beta_d / factor) * output, where beta_d is the per-type rate of `costs`.
    With both left as None this is exactly ConsumptionMDP (same transitions, same values).
    `tau` (used by the static policies and RSS) stays the deterministic rounding of `costs`.
    """

    consume_speedups: Speedups | None = None
    final_speedups: Speedups | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for sp in (self.consume_speedups, self.final_speedups):
            if sp is not None and (abs(sum(p for _, p in sp) - 1.0) > 1e-9 or min(f for f, _ in sp) <= 0):
                raise ValueError("speedup distribution must sum to 1 with positive factors")
        c = self.costs
        self.tau_pmf: dict[tuple[int, int], StepPmf] = {}
        for (k, b), tau in self.tau.items():
            if tau == INF or self.consume_speedups is None:
                self.tau_pmf[(k, b)] = [(tau, 1.0)]
                continue
            dec = c.consume_decode_s_per_tok if c.consume_decode_s_per_tok is not None else c.decode_s_per_tok
            out = c.consume_out_base + c.consume_out_per_result * b
            fixed = c.consume(k, b) - dec * out
            self.tau_pmf[(k, b)] = _merge((ceil_steps(fixed + dec * out / f, self.dt), p)
                                          for f, p in self.consume_speedups)
        self.final_pmf: dict[int, SecPmf] = {}
        for k in range(self.n + 1):
            fc = c.final(k, self.n)
            if fc == INF or self.final_speedups is None:
                self.final_pmf[k] = [(fc, 1.0)]
                continue
            dec = c.final_decode_s_per_tok if c.final_decode_s_per_tok is not None else c.decode_s_per_tok
            fixed = fc - dec * c.final_out
            self.final_pmf[k] = _merge((fixed + dec * c.final_out / f, p) for f, p in self.final_speedups)

    def successors(self, t: int, r: int, k: int, a: Action):
        kind, b = a
        if kind == "final":
            for d, p in self.final_pmf[k]:
                yield p, t * self.dt + d
            return
        if kind == "wait":
            yield from super().successors(t, r, k, a)
            return
        if not 1 <= b <= r - k:
            raise ValueError(f"policy consumed {b} results with only {r - k} returned and raw")
        for steps, q in self.tau_pmf[(k, b)]:
            t2 = t + steps
            for j, p in enumerate(binom_pmf(self.n - r, self.release_prob(t, t2))):
                if p > 0:
                    yield q * p, (t2, r + j, k + b)

    def simulate_random(self, policy: Policy, delays: list[int], rng: random.Random) -> float:
        """Run a policy against concrete delays, sampling every action duration."""
        t, k = 0, 0
        while True:
            r = sum(d <= t for d in delays)
            kind, b = policy(t, r, k)
            if kind == "final":
                ds, ps = zip(*self.final_pmf[k])
                return t * self.dt + rng.choices(ds, weights=ps)[0]
            if kind == "wait":
                t += 1
            else:
                if b > r - k:
                    raise RuntimeError("policy consumed unreturned results")
                ss, ps = zip(*self.tau_pmf[(k, b)])
                t += rng.choices(ss, weights=ps)[0]
                k += b


def final_summary_count_dist(mdp: ConsumptionMDP, policy: Policy) -> dict[int, float]:
    """Exact distribution of #summaries at FINAL under `policy` (for the extra-summary prediction)."""
    memo: dict = {}

    def go(state) -> dict[int, float]:
        if state in memo:
            return memo[state]
        t, r, k = state
        a = policy(t, r, k)
        out: dict[int, float] = {}
        if a == FINAL:
            out = {k: 1.0}
        else:
            for p, nxt in mdp.successors(t, r, k, a):
                for kk, q in go(nxt).items():
                    out[kk] = out.get(kk, 0.0) + p * q
        memo[state] = out
        return out

    total: dict[int, float] = {}
    for j, p in enumerate(binom_pmf(mdp.n, mdp.G(0))):
        if p > 0:
            for kk, q in go((0, j, 0)).items():
                total[kk] = total.get(kk, 0.0) + p * q
    return total

