"""Processing policies behind one interface. All of them act on the same count state as the theory:
(elapsed step since launch, #returned, #summarized). Which results to summarize: oldest-returned first
(Prop. 3). The runtime enforces context/contract guards; policies never see hidden future timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from .affine import AffineCostModel
from .online_mdp import (
    ConsumptionMDP,
    required_summary_slack,
    static_batch,
    static_fold,
    static_incremental,
)
from .state import Action, Consume, Finalize, Observation, Preload, Wait


@dataclass(frozen=True)
class PolicyContext:
    n: int
    remaining_at_launch_s: float
    costs: AffineCostModel
    delay_pmf: tuple[float, ...]
    dt: float
    objective: str = "miss"


@lru_cache(maxsize=256)
def _mdp(ctx: PolicyContext) -> ConsumptionMDP:
    return ConsumptionMDP(ctx.n, ctx.costs, list(ctx.delay_pmf), ctx.dt, ctx.remaining_at_launch_s,
                          objective=ctx.objective)  # type: ignore[arg-type]


def quantize(ctx: PolicyContext) -> PolicyContext:
    """Round the remaining deadline to the dt grid so MDP solutions are cached across tasks."""
    steps = int(ctx.remaining_at_launch_s / ctx.dt)
    return PolicyContext(ctx.n, steps * ctx.dt, ctx.costs, ctx.delay_pmf, ctx.dt, ctx.objective)


class CountStatePolicy:
    def __init__(self, name: str, ctx: PolicyContext, make: Callable[[ConsumptionMDP], Callable]):
        self.name, self.ctx = name, quantize(ctx)
        self.mdp = _mdp(self.ctx)
        self.pi = make(self.mdp)

    def decide(self, obs: Observation) -> Action:
        dt = self.ctx.dt
        step = int(obs.elapsed_since_launch_s / dt + 1e-9)
        kind, b = self.pi(step, obs.n_returned, obs.n_summarized)
        if kind == "consume":
            ids = obs.returned_raw_ids[: max(1, min(b, len(obs.returned_raw_ids)))]
            return Consume(tuple(ids)) if ids else Wait(dt)
        if kind == "final":
            return Finalize()
        return Wait(max(1e-6, (step + 1) * dt - obs.elapsed_since_launch_s))


def _optimal(mdp: ConsumptionMDP):
    mdp.solve()
    return mdp.optimal_policy()


POLICIES: dict[str, Callable[[ConsumptionMDP], Callable]] = {
    "incremental": static_incremental,
    "fold": static_fold,
    "batch2": lambda m: static_batch(m, 2),
    "batch3": lambda m: static_batch(m, 3),
    "rss": required_summary_slack,
    "optimal": _optimal,
}


class PreloadWrapper:
    """<base>_preload: run the base count-state policy; whenever it would idle-wait, instead prefill the returned,
    not-yet-logged raw results into the KV cache (evidence log). Preloaded raws are never summarized (the runtime
    guard drops them from Consume). With 'fold' as the base this is the fold+preload policy of theory.md B2."""

    def __init__(self, base: CountStatePolicy):
        self.base, self.name = base, base.name + "_preload"

    def decide(self, obs: Observation) -> Action:
        a = self.base.decide(obs)
        if isinstance(a, Wait):
            new = tuple(c for c in obs.returned_raw_ids if c not in obs.preloaded_ids)
            if new:
                return Preload(new)
        return a


def make_policy(name: str, ctx: PolicyContext):
    if name.endswith("_preload"):
        return PreloadWrapper(make_policy(name.removesuffix("_preload"), ctx))
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    return CountStatePolicy(name, ctx, POLICIES[name])
