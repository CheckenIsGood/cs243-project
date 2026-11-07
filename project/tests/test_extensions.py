"""Checks for the engine-dependent extensions (docs/theory.md, Section 8).

A: per-type decode rates / random action durations (speculative decoding).
B: KV pre-loading through prefix caching (evidence-log layout), offline and online.
"""

import math
import random

import pytest

from deadline_agent.affine import AffineCostModel, fold_completion, fold_gap_bound, prefix_dp
from deadline_agent.cache_model import (
    PreloadCostModel,
    RoundedPreloadCosts,
    earliest_first_optimum,
    replacement_excess,
    solve_preload_dp,
    solve_preload_exhaustive,
)
from deadline_agent.online_mdp import (
    ConsumptionMDP,
    bounded_delay_pmf,
    required_summary_slack,
    static_batch,
    static_fold,
    static_incremental,
)
from deadline_agent.preload_mdp import PreloadMDP, no_preload, with_idle_preload
from deadline_agent.restricted import INF, Instance, solve_dp
from deadline_agent.speculation import (
    StochasticDurationMDP,
    final_summary_count_dist,
    mean_speedup_costs,
    speculative_costs,
)

SEC6 = dict(overhead_s=0.08, prefill_s_per_tok=1 / 3000, decode_s_per_tok=1 / 30)


def random_affine(rng: random.Random, adversarial: bool = False) -> AffineCostModel:
    dec = rng.uniform(1 / 80, 1 / 10)
    if adversarial and rng.random() < 0.5:
        dec = rng.uniform(1 / 2000, 1 / 100)
    return AffineCostModel(
        overhead_s=rng.uniform(0.0, 0.3),
        prefill_s_per_tok=rng.uniform(1 / 10000, 1 / 500),
        decode_s_per_tok=dec,
        consume_base_prompt=rng.choice([50, 300]),
        final_base_prompt=rng.choice([100, 300, 2000] if adversarial else [100, 300, 600]),
        raw_tokens=rng.choice([400, 768, 1500, 2000]),
        summary_tokens=rng.choice([32, 64, 128]),
        consume_out_base=rng.choice([0, 32]),
        consume_out_per_result=rng.choice([8, 32, 64]),
        context_limit=rng.choice([3000, 4096, 6000, 8192, 16384]),
        consume_reads_summaries=rng.random() < 0.5,
        consume_decode_s_per_tok=dec / rng.choice([1.0, 1.5, 2.0, 3.0]),
        final_decode_s_per_tok=dec / rng.choice([1.0, 1.2]),
    )


def random_preload_model(rng: random.Random, adversarial: bool = False) -> PreloadCostModel:
    return PreloadCostModel(random_affine(rng, adversarial),
                            preload_overhead_s=rng.uniform(0.0, 1.0) if adversarial and rng.random() < 0.5 else None,
                            final_suffix_tokens=rng.choice([0, 16]))


def random_releases(rng: random.Random, n: int) -> tuple[float, ...]:
    return tuple(round(rng.choice([0.0, rng.uniform(0, 10)]), 2) for _ in range(n))


# ================================ Extension A ====================================================
def test_default_decode_fields_are_bit_for_bit_unchanged():
    rng = random.Random(0)
    for _ in range(200):
        m = random_affine(rng)
        plain = AffineCostModel(**{f: getattr(m, f) for f in m.__dataclass_fields__
                                   if f not in ("consume_decode_s_per_tok", "final_decode_s_per_tok")})
        same = AffineCostModel(**{**plain.__dict__, "consume_decode_s_per_tok": plain.decode_s_per_tok,
                                  "final_decode_s_per_tok": plain.decode_s_per_tok})
        for k in range(5):
            for b in range(1, 5):
                assert plain.consume(k, b) == same.consume(k, b)
            assert plain.final(k, 5) == same.final(k, 5)


def test_speculative_costs_only_change_decode():
    m = AffineCostModel(**SEC6, raw_tokens=1500, context_limit=4096)
    s = speculative_costs(m, consume_speedup=3.0)
    assert s.final(2, 4) == m.final(2, 4)
    assert s.fold_saving_per_summary == m.fold_saving_per_summary
    out = m.consume_out_base + m.consume_out_per_result
    assert m.consume(0, 1) - s.consume(0, 1) == pytest.approx(m.decode_s_per_tok * out * (1 - 1 / 3))
    for b in range(1, 3):  # the key inequality of Props 1-2 survives any decode speedup
        assert s.consume(0, b) >= m.fold_saving_per_summary * b


@pytest.mark.parametrize("seed", range(120))
def test_props_1_to_3_with_per_type_decode_rates(seed):
    rng = random.Random(10_000 + seed)
    n = rng.randint(1, 6)
    m = random_affine(rng)
    rel = tuple(rng.uniform(0, 10) for _ in range(n))
    general = solve_dp(m.to_instance(rel))
    fast = prefix_dp(rel, m)
    assert fast.completion == pytest.approx(general.completion) or fast.completion == general.completion == INF
    if m.final(0, n) < INF:
        assert -1e-9 <= fold_completion(rel, m) - fast.completion <= fold_gap_bound(m, n) + 1e-9
        t0 = rng.uniform(0, 5)
        assert fold_completion((t0,) * n, m) == pytest.approx(prefix_dp((t0,) * n, m).completion)


@pytest.mark.parametrize("seed", range(60))
def test_props_1_2_hold_for_content_dependent_consume_speedups(seed):
    """Set-dependent (non-count) summary costs: Props 1-2 still hold exactly; only c >= beta_p*L*b is used."""
    rng = random.Random(20_000 + seed)
    n = rng.randint(1, 5)
    m = random_affine(rng)
    if m.final(0, n) == INF:
        pytest.skip("fold infeasible")
    factor: dict = {}

    def c(A: int, B: int) -> float:
        base = m.consume(A.bit_count(), B.bit_count())
        if base == INF:
            return INF
        out = m.consume_out_base + m.consume_out_per_result * B.bit_count()
        if (A, B) not in factor:
            factor[(A, B)] = rng.uniform(1.0, 4.0)  # speedup depends on which results are summarized
        return base - m.consume_decode_s_per_tok * out * (1 - 1 / factor[(A, B)])

    f = lambda S: m.final(S.bit_count(), n)  # noqa: E731
    rel = tuple(rng.uniform(0, 10) for _ in range(n))
    opt = solve_dp(Instance(rel, c, f)).completion
    assert -1e-9 <= fold_completion(rel, m) - opt <= fold_gap_bound(m, n) + 1e-9
    t0 = rng.uniform(0, 3)
    assert solve_dp(Instance((t0,) * n, c, f)).completion == pytest.approx(fold_completion((t0,) * n, m))


def sdmdp(D=18.0, objective="miss", cs=None, fs=None, base=None, dt=0.25, n=4):
    m = base or AffineCostModel(**SEC6, raw_tokens=1500, context_limit=4096)
    pmf = bounded_delay_pmf(dt, 0.5, 4.0, slow_p=0.3, slow_lo_s=4.0, slow_hi_s=12.0)
    return StochasticDurationMDP(n, m, pmf, dt, D, objective=objective, consume_speedups=cs, final_speedups=fs)


def test_stochastic_mdp_without_speedups_equals_consumption_mdp():
    a = sdmdp()
    b = ConsumptionMDP(a.n, a.costs, a.delay_pmf, a.dt, a.deadline_s, objective="miss")
    assert a.solve() == b.solve()
    for mk in (static_fold, static_incremental, required_summary_slack):
        assert a.evaluate(mk(a)) == b.evaluate(mk(b))


def test_point_mass_speedup_equals_deterministic_rates():
    base = AffineCostModel(**SEC6, raw_tokens=1500, context_limit=4096)
    a = sdmdp(cs=[(2.0, 1.0)], fs=[(1.25, 1.0)], base=base)
    det = speculative_costs(base, 2.0, 1.25)
    b = ConsumptionMDP(a.n, det, a.delay_pmf, a.dt, a.deadline_s, objective="miss")
    assert a.solve() == pytest.approx(b.solve())


@pytest.mark.parametrize("objective", ["miss", "time"])
def test_stochastic_durations_optimal_dominates_and_matches_mc(objective):
    sp = [(1.0, 0.3), (2.0, 0.4), (3.5, 0.3)]
    mdp = sdmdp(D=16.0, objective=objective, cs=sp, fs=[(1.0, 0.5), (1.3, 0.5)])
    v = mdp.solve()
    pi = mdp.optimal_policy()
    assert mdp.evaluate(pi) == pytest.approx(v)
    for mk in (static_fold, static_incremental, required_summary_slack, lambda x: static_batch(x, 2)):
        assert v <= mdp.evaluate(mk(mdp)) + 1e-9
    # the deterministic-mean policy is feasible here, so it cannot beat the exact one
    mean = ConsumptionMDP(mdp.n, mean_speedup_costs(mdp.costs, sp, [(1.0, 0.5), (1.3, 0.5)]),
                          mdp.delay_pmf, mdp.dt, mdp.deadline_s, objective=objective)
    mean.solve()
    assert v <= mdp.evaluate(mean.optimal_policy()) + 1e-9
    if objective == "time":
        rng = random.Random(3)
        sims = [mdp.simulate_random(pi, mdp.sample_delays(rng), rng) for _ in range(3000)]
        mu = sum(sims) / len(sims)
        sd = math.sqrt(sum((x - mu) ** 2 for x in sims) / (len(sims) - 1))
        assert abs(mu - v) < 4 * sd / math.sqrt(len(sims))


def test_final_summary_count_distribution_sums_to_one():
    mdp = sdmdp(D=18.0)
    mdp.solve()
    dist = final_summary_count_dist(mdp, mdp.optimal_policy())
    assert sum(dist.values()) == pytest.approx(1.0)
    assert min(dist) >= mdp.costs.min_summaries_needed(mdp.n)


# ================================ Extension B (offline) ==========================================
def test_preload_model_without_preloads_is_the_base_model():
    rng = random.Random(1)
    for _ in range(100):
        cm = random_preload_model(rng)
        n = rng.randint(1, 6)
        for k in range(n + 1):
            assert cm.final(k, 0, k, 0, n) == pytest.approx(cm.base.final(k, n)) or \
                cm.final(k, 0, k, 0, n) == cm.base.final(k, n) == INF


@pytest.mark.parametrize("seed", range(100))
def test_preload_dp_no_preload_equals_prefix_dp_and_preload_never_hurts(seed):
    rng = random.Random(30_000 + seed)
    cm = random_preload_model(rng, adversarial=seed % 2 == 1)
    n = rng.randint(1, 6)
    rel = random_releases(rng, n)
    base = prefix_dp(rel, cm.base).completion
    nopre = solve_preload_dp(rel, cm, preload=False).completion
    assert nopre == pytest.approx(base) or nopre == base == INF
    assert solve_preload_dp(rel, cm).completion <= base + 1e-9


def check_schedule(rel, cm, sched):
    """Replay a reconstructed schedule with explicit identities and the count cost model."""
    t, k, p, j, z, used = 0.0, 0, 0, 0, 0, set()
    n = len(rel)
    for kind, ids, start, end in sched.actions:
        assert not used & set(ids)
        used |= set(ids)
        assert start >= max([t] + [rel[i] for i in ids]) - 1e-9
        if kind == "consume":
            assert end == pytest.approx(start + cm.consume(k, len(ids)))
            k, j = k + len(ids), j + len(ids)
        else:
            assert end == pytest.approx(start + cm.preload(k, p, j, z, len(ids)))
            p, j, z = p + len(ids), 0, 1
        t = end
    rest = [i for i in range(n) if i not in used]
    start = max([t] + [rel[i] for i in rest])
    assert sched.completion == pytest.approx(start + cm.final(k, p, j, z, n))


@pytest.mark.parametrize("seed", range(160))
def test_preload_dp_matches_exhaustive(seed):
    """Prop. B1: count DP = brute force over identities, deletions, orders and an explicit prefix trie."""
    rng = random.Random(40_000 + seed)
    cm = random_preload_model(rng, adversarial=seed % 2 == 1)
    n = rng.randint(1, 3) if seed >= 40 else 4
    rel = random_releases(rng, n)
    dp = solve_preload_dp(rel, cm)
    ex = solve_preload_exhaustive(rel, cm)
    assert dp.completion == pytest.approx(ex) or dp.completion == ex == INF
    if dp.feasible:
        check_schedule(rel, cm, dp)


@pytest.mark.parametrize("seed", range(60))
def test_earliest_first_restriction_matches_exhaustive(seed):
    rng = random.Random(50_000 + seed)
    cm = random_preload_model(rng, adversarial=seed % 2 == 1)
    n = rng.randint(1, 3)
    rel = tuple(sorted(random_releases(rng, n)))  # index order = release order
    ef = earliest_first_optimum(rel, cm).completion
    ex = min(solve_preload_exhaustive(rel, cm, summarize_exactly=frozenset(range(K))) for K in range(n + 1))
    assert ef == pytest.approx(ex) or ef == ex == INF


def test_order_flip_counterexample():
    """Prop. B4: with pre-loading, summarizing the earliest results is no longer optimal (Section 6 costs)."""
    cm = PreloadCostModel(AffineCostModel(**SEC6, raw_tokens=1500, context_limit=4096))
    rel = (0.0, 0.7, 0.7, 0.7)
    assert cm.base.min_summaries_needed(4) == 2 and cm.condition_P()
    opt = solve_preload_dp(rel, cm)
    ef = earliest_first_optimum(rel, cm)
    assert opt.completion == pytest.approx(14.241333, abs=1e-5)
    assert ef.completion == pytest.approx(14.741333, abs=1e-5)
    assert opt.summarized == (1, 2) and ef.summarized == (0, 1)
    assert [a[0] for a in opt.actions] == ["preload", "consume"] and opt.actions[0][1] == (0,)
    # independent confirmation, including the earliest-first restriction
    assert solve_preload_exhaustive(rel, cm) == pytest.approx(opt.completion)
    assert min(solve_preload_exhaustive(rel, cm, summarize_exactly=frozenset(range(K))) for K in range(5)) \
        == pytest.approx(ef.completion)
    # without pre-loading the two orders tie (Prop. 3 holds there)
    no_pre = [solve_preload_exhaustive(rel, cm, summarize_exactly=frozenset(S), allow_preload=False)
              for S in [(0, 1), (1, 2)]]
    assert no_pre[0] == pytest.approx(no_pre[1]) == pytest.approx(prefix_dp(rel, cm.base).completion)
    # the current runtime layout (summaries before raws) cannot exploit it
    assert solve_preload_exhaustive(rel, cm, layout="summaries_first") == pytest.approx(ef.completion)


@pytest.mark.parametrize("seed", range(200))
def test_fold_with_preload_is_optimal_when_context_fits(seed):
    """Prop. B2: all-raw prompt fits and Delta_B = 0 => fold+preload is optimal; in general gap <= n*Delta_B."""
    rng = random.Random(60_000 + seed)
    cm = random_preload_model(rng, adversarial=seed % 2 == 1)
    n = rng.randint(1, 6)
    if cm.base.final(0, n) == INF:
        pytest.skip("fold infeasible")
    rel = random_releases(rng, n)
    opt = solve_preload_dp(rel, cm).completion
    fold = solve_preload_dp(rel, cm, summarized=0).completion
    excess = replacement_excess(cm, n)
    assert -1e-9 <= fold - opt <= n * excess + 1e-9
    if cm.condition_P():
        assert excess == 0.0


@pytest.mark.parametrize("seed", range(200))
def test_exactly_min_summaries_suffice_under_condition_P(seed):
    """Prop. B3: under (P), some optimal pre-load schedule summarizes exactly m = min_summaries_needed."""
    rng = random.Random(70_000 + seed)
    cm = random_preload_model(rng, adversarial=seed % 2 == 1)
    if not cm.condition_P():
        pytest.skip("(P) fails")
    n = rng.randint(1, 6)
    need = cm.base.min_summaries_needed(n)
    if need > n:
        pytest.skip("infeasible")
    rel = random_releases(rng, n)
    opt = solve_preload_dp(rel, cm).completion
    if opt == INF:
        pytest.skip("infeasible")
    assert solve_preload_dp(rel, cm, n_summaries=need).completion == pytest.approx(opt)


@pytest.mark.parametrize("seed", range(40))
def test_log_layout_dominates_summaries_first(seed):
    rng = random.Random(80_000 + seed)
    cm = random_preload_model(rng)
    n = rng.randint(1, 3)
    rel = random_releases(rng, n)
    assert solve_preload_exhaustive(rel, cm) <= solve_preload_exhaustive(rel, cm, layout="summaries_first") + 1e-9


# ================================ Extension B (online) ===========================================
def pmdp(D=18.0, objective="miss", base=None, dt=0.25, pmf=None, n=4):
    m = base or AffineCostModel(**SEC6, raw_tokens=1500, context_limit=4096)
    pmf = pmf or bounded_delay_pmf(dt, 0.5, 4.0, slow_p=0.3, slow_lo_s=4.0, slow_hi_s=12.0)
    return PreloadMDP(n, m, pmf, dt, D, objective=objective)


@pytest.mark.parametrize("ctx", [dict(raw_tokens=1500, context_limit=4096), dict(raw_tokens=768, context_limit=8192)])
@pytest.mark.parametrize("objective", ["miss", "tardiness", "time"])
def test_preload_mdp_lifts_and_dominates(ctx, objective):
    m = AffineCostModel(**SEC6, **ctx)
    pm = pmdp(D=16.0, objective=objective, base=m)
    bm = ConsumptionMDP(pm.n, m, pm.delay_pmf, pm.dt, pm.deadline_s, objective=objective)
    v = pm.solve()
    assert v <= bm.solve() + 1e-12
    for mk in (static_fold, static_incremental, required_summary_slack, lambda x: static_batch(x, 2)):
        lifted = pm.evaluate(no_preload(mk(pm)))
        assert lifted == pytest.approx(bm.evaluate(mk(bm)))  # never pre-loading = base model
        assert v <= pm.evaluate(with_idle_preload(pm, mk(pm))) + 1e-9


def test_preload_mdp_point_mass_matches_offline_dp():
    for steps in [0, 3, 10, 30]:
        pm = pmdp(objective="time", pmf=[0.0] * steps + [1.0])
        off = solve_preload_dp((steps * pm.dt,) * pm.n, RoundedPreloadCosts(pm.cache, pm.dt)).completion
        assert pm.solve() == pytest.approx(off)


def test_preload_mdp_evaluation_matches_simulation():
    pm = pmdp(objective="time")
    v = pm.solve()
    pi = pm.optimal_policy()
    assert pm.evaluate(pi) == pytest.approx(v)
    rng = random.Random(5)
    sims = [pm.simulate(pi, pm.sample_delays(rng)) for _ in range(3000)]
    mu = sum(sims) / len(sims)
    sd = math.sqrt(sum((x - mu) ** 2 for x in sims) / (len(sims) - 1))
    assert abs(mu - v) < 4 * sd / math.sqrt(len(sims))
    assert max(sims) < INF


def test_preload_mdp_rejects_invalid_actions():
    pm = pmdp(objective="time")
    with pytest.raises(ValueError):
        pm.evaluate(lambda t, r, k, p, j, z: ("preload", 1) if r - k - p == 0 else ("final", 0))
    with pytest.raises(ValueError):
        pm.evaluate(lambda t, r, k, p, j, z: ("consume", 1) if r - k - p == 0 else ("final", 0))


@pytest.mark.parametrize("ctx", [dict(raw_tokens=1500, context_limit=4096), dict(raw_tokens=768, context_limit=8192)])
@pytest.mark.parametrize("pred", [False, True])
def test_online_preload_optimum_needs_only_m_summaries(ctx, pred):
    """Online counterpart of Props. B2/B3 (Section 6 costs): capping summaries at m loses nothing."""
    m = AffineCostModel(**SEC6, **ctx)
    pmf = bounded_delay_pmf(0.25, 2.0, 3.0) if pred else None
    need = m.min_summaries_needed(4)
    for D in [12.0, 16.0, 17.0, 18.0]:
        full = pmdp(D=D, base=m, pmf=pmf)
        capped = PreloadMDP(4, m, full.delay_pmf, full.dt, D, objective="miss", max_summaries=need)
        assert capped.solve() == pytest.approx(full.solve(), abs=1e-12)
