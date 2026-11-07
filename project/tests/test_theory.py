"""Gate 0 checks for the restricted theory (docs/theory.md)."""

import math
import random

import pytest

from deadline_agent.affine import AffineCostModel, fold_completion, fold_gap_bound, prefix_dp
from deadline_agent.online_mdp import (
    FINAL,
    ConsumptionMDP,
    bounded_delay_pmf,
    static_batch,
    static_fold,
    static_incremental,
)
from deadline_agent.restricted import INF, Instance, simulate, solve_dp, solve_exhaustive


def random_general_instance(rng: random.Random, n: int) -> Instance:
    """Arbitrary set-dependent costs (c may depend on A), random infeasible masks."""
    releases = tuple(rng.choice([0.0, rng.uniform(0, 5)]) for _ in range(n))
    table_c, table_f = {}, {}

    def c(A: int, B: int) -> float:
        if (A, B) not in table_c:
            table_c[(A, B)] = INF if rng.random() < 0.1 else rng.uniform(0.1, 3.0)
        return table_c[(A, B)]

    def f(S: int) -> float:
        if S not in table_f:
            table_f[S] = INF if rng.random() < 0.15 else rng.uniform(0.1, 4.0)
        return table_f[S]

    # materialize tables so DP and exhaustive see identical costs
    for A in range(1 << n):
        f(A)
        rest = ((1 << n) - 1) ^ A
        sub = rest
        while sub:
            c(A, sub)
            sub = (sub - 1) & rest
    return Instance(releases, c, f)


@pytest.mark.parametrize("seed", range(150))
def test_dp_matches_exhaustive_general_costs(seed):
    rng = random.Random(seed)
    inst = random_general_instance(rng, rng.randint(0, 5))
    dp, ex = solve_dp(inst), solve_exhaustive(inst)
    assert dp.completion == pytest.approx(ex.completion) or dp.completion == ex.completion == INF
    if dp.feasible:
        # reconstruction replays to the same time and consumes every result exactly once
        replay = simulate(inst, [B for B, _, _ in dp.batches])
        assert replay.completion == pytest.approx(dp.completion)
        seen = 0
        for B, _, _ in dp.batches:
            assert seen & B == 0
            seen |= B
        assert seen == dp.final_summarized


def random_affine(rng: random.Random) -> AffineCostModel:
    return AffineCostModel(
        overhead_s=rng.uniform(0.0, 0.3),
        prefill_s_per_tok=rng.uniform(1 / 10000, 1 / 500),
        decode_s_per_tok=rng.uniform(1 / 80, 1 / 10),
        raw_tokens=rng.choice([400, 768, 1500, 2000]),
        summary_tokens=rng.choice([32, 64, 128]),
        consume_out_base=rng.choice([0, 32]),
        consume_out_per_result=rng.choice([32, 64]),
        context_limit=rng.choice([3000, 4096, 8192, 16384]),
        consume_reads_summaries=rng.random() < 0.5,
    )


@pytest.mark.parametrize("seed", range(150))
def test_prefix_dp_is_exact_for_count_costs(seed):
    """Proposition 3: O(n^2) prefix DP equals the general subset DP."""
    rng = random.Random(1000 + seed)
    n = rng.randint(1, 6)
    m = random_affine(rng)
    rel = tuple(rng.choice([0.0, rng.uniform(0, 10)]) for _ in range(n))
    general = solve_dp(m.to_instance(rel))
    fast = prefix_dp(rel, m)
    if general.feasible:
        assert fast.completion == pytest.approx(general.completion)
        assert simulate(m.to_instance(rel), [B for B, _, _ in fast.batches]).completion == pytest.approx(
            fast.completion
        )
    else:
        assert not fast.feasible


@pytest.mark.parametrize("seed", range(150))
def test_fold_gap_bound(seed):
    """Proposition 2: when the all-raw prompt fits, fold is within (n-1)*delta."""
    rng = random.Random(2000 + seed)
    n = rng.randint(1, 6)
    m = random_affine(rng)
    if m.final(0, n) == INF:
        pytest.skip("fold infeasible; proposition does not apply")
    rel = tuple(rng.uniform(0, 10) for _ in range(n))
    opt = prefix_dp(rel, m).completion
    gap = fold_completion(rel, m) - opt
    assert -1e-9 <= gap <= fold_gap_bound(m, n) + 1e-9


@pytest.mark.parametrize("seed", range(50))
def test_fold_optimal_when_released_together(seed):
    """Proposition 1: simultaneous releases + feasible fold => fold is optimal."""
    rng = random.Random(3000 + seed)
    n = rng.randint(1, 6)
    m = random_affine(rng)
    if m.final(0, n) == INF:
        pytest.skip("fold infeasible")
    t0 = rng.uniform(0, 5)
    rel = (t0,) * n
    assert fold_completion(rel, m) == pytest.approx(prefix_dp(rel, m).completion)


# ---- online MDP ----------------------------------------------------------------
def tight_model() -> AffineCostModel:
    # context binds: 4 raw results of 1500 tokens cannot all fit in 4096
    return AffineCostModel(raw_tokens=1500, summary_tokens=64, context_limit=4096)


def make_mdp(objective="miss", pmf=None, deadline=14.0, n=4) -> ConsumptionMDP:
    dt = 0.25
    pmf = pmf or bounded_delay_pmf(dt, 0.5, 6.0, slow_p=0.25, slow_lo_s=6.0, slow_hi_s=12.0)
    return ConsumptionMDP(n, tight_model(), pmf, dt, deadline, objective=objective)


@pytest.mark.parametrize("objective", ["miss", "tardiness", "time"])
def test_mdp_dominates_static_policies(objective):
    mdp = make_mdp(objective)
    opt = mdp.solve()
    for pol in [static_incremental(mdp), static_fold(mdp), static_batch(mdp, 2), static_batch(mdp, 3)]:
        assert opt <= mdp.evaluate(pol) + 1e-9


def test_mdp_policy_evaluation_matches_simulation():
    mdp = make_mdp("time")
    v = mdp.solve()
    pi = mdp.optimal_policy()
    assert mdp.evaluate(pi) == pytest.approx(v)
    rng = random.Random(7)
    sims = [mdp.simulate(pi, mdp.sample_delays(rng)) for _ in range(4000)]
    mean = sum(sims) / len(sims)
    sd = math.sqrt(sum((x - mean) ** 2 for x in sims) / (len(sims) - 1))
    assert abs(mean - v) < 4 * sd / math.sqrt(len(sims))


def test_mdp_with_deterministic_delay_matches_offline_dp():
    dt = 0.25
    for delay_steps in [0, 3, 10, 30]:
        pmf = [0.0] * delay_steps + [1.0]
        mdp = make_mdp("time", pmf=pmf)
        rel = (delay_steps * dt,) * mdp.n
        # offline DP with the same step-rounded consume costs = MDP value
        m = mdp.costs
        offline = solve_dp(
            Instance(
                rel,
                lambda A, B: mdp.tau[(A.bit_count(), B.bit_count())] * dt,
                lambda S: m.final(S.bit_count(), mdp.n),
            )
        ).completion
        assert mdp.solve() == pytest.approx(offline)


def test_mdp_never_consumes_unreturned_and_always_finishes():
    mdp = make_mdp("miss")
    mdp.solve()
    pi = mdp.optimal_policy()
    rng = random.Random(11)
    for _ in range(500):
        c = mdp.simulate(pi, mdp.sample_delays(rng))  # raises if it consumes unreturned results
        assert c < INF


def test_static_fold_is_infeasible_without_summaries_when_context_binds():
    m = tight_model()
    assert m.final(0, 4) == INF
    assert m.min_summaries_needed(4) >= 1
    mdp = make_mdp()
    assert static_fold(mdp)(0, 4, 0) != FINAL


@pytest.mark.parametrize("deadline", [16.0, 18.0, 20.0, 24.0])
def test_rss_heuristic_near_optimal_and_beats_fold_when_context_binds(deadline):
    from deadline_agent.online_mdp import required_summary_slack

    dt = 0.1
    pmf = bounded_delay_pmf(dt, 0.5, 4.0, slow_p=0.3, slow_lo_s=4.0, slow_hi_s=12.0)
    m = AffineCostModel(overhead_s=0.08, prefill_s_per_tok=1 / 3000, decode_s_per_tok=1 / 30,
                        raw_tokens=1500, context_limit=4096)
    mdp = ConsumptionMDP(4, m, pmf, dt, deadline, objective="miss", tie_eps=0.0)
    opt = mdp.solve()
    heur = mdp.evaluate(required_summary_slack(mdp))
    fold = mdp.evaluate(static_fold(mdp))
    assert opt - 1e-9 <= heur <= opt + 0.05
    assert heur <= fold + 1e-9


def test_optimal_policy_defined_on_infeasible_instances_and_evaluate_rejects_bad_consume():
    m = AffineCostModel(raw_tokens=2000, summary_tokens=64, context_limit=1000)  # nothing ever fits
    mdp = ConsumptionMDP(3, m, bounded_delay_pmf(0.25, 0.5, 2.0), 0.25, 10.0)
    assert mdp.solve() == INF
    assert mdp.optimal_policy()(0, 0, 0) is not None
    ok = make_mdp("time")
    with pytest.raises(ValueError):
        ok.evaluate(lambda t, r, k: ("consume", 1) if r == k else FINAL)


@pytest.mark.parametrize("n", [2, 4, 8, 12])
def test_rss_never_picks_infeasible_batch_when_batches_overflow_context(n):
    """Regression: RSS once chose a summary batch whose own prompt exceeded the context limit (value = inf)."""
    from deadline_agent.online_mdp import INF, bounded_delay_pmf, required_summary_slack, static_fold

    dt = 0.1
    pmf = bounded_delay_pmf(dt, 0.5, 4.0, slow_p=0.3, slow_lo_s=4.0, slow_hi_s=12.0)
    m = AffineCostModel(overhead_s=0.08, prefill_s_per_tok=1 / 3000, decode_s_per_tok=1 / 30, raw_tokens=600,
                        summary_tokens=60, context_limit=150 + 600 * n // 2 + 300, final_out=20)
    mdp = ConsumptionMDP(n, m, pmf, dt, 10 + 1.2 * n, objective="miss")
    pi = required_summary_slack(mdp)
    for t in (0, 40, 120):
        for r in range(n + 1):
            for k in range(r + 1):
                kind, b = pi(t, r, k)
                if kind == "consume":
                    assert 1 <= b <= r - k and mdp.tau[(k, b)] != INF
    v_rss, v_fold = mdp.evaluate(pi), mdp.evaluate(static_fold(mdp))
    assert v_rss < INF and v_rss <= v_fold + 1e-9
    assert mdp.solve() <= v_rss + 1e-9
