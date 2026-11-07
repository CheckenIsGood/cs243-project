"""Machine checks for docs/stochastic_theory.md. Each test is cited in the doc as [T: name].

Conventions: every scheduling quantity is an exact Fraction; 'DP' optima are exact over the stated policy class
on integer data; random instances use fixed seeds (deterministic, reproducible)."""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction as F

import pytest
import z3

from codesign.stochastic_theory import (
    HCfg, HNode, HTask, KeepClass, MTask, chain_task, cv2, critical_path, det, disc_bound, dist, dmax,
    expected_cost, h_bound, h_constants, h_cp, h_Lmax, h_lower_bounds, h_order, h_realizations, h_sizes, hold_only_lower,
    index_vs_pmtn_ratio, keep_dual, keep_lp_z3, lag_trap, lb2_task_lp_z3, lb2_task_msu, lb_cp, lb_smith_det,
    lp_order, mean, msu_gamma, msu_lp, opt_clair_1m, opt_np_dp, opt_pmtn_dp, pathwise_bound, realizations,
    sigma_star_prices, sim_discrete, sim_hybrid, sim_machine, ski_worst_lp, smith, task_index_order, ttl_best,
    thmD_constant, ttl_clair, ttl_cost, var,
)

E_OVER_E1_UP = F(158198, 100000)  # > e/(e-1) = 1.5819767...
assert E_OVER_E1_UP > F(str(math.e / (math.e - 1)))


def _proved(claim, *hyps):
    s = z3.Solver()
    s.add(*hyps)
    s.add(z3.Not(claim))
    return s.check() == z3.unsat


# ----------------------------------------------------------------------------------- random instance makers
def _rdist(rng, lo, hi, k):
    vals = rng.sample(range(lo, hi + 1), k)
    return dist(*[(v, rng.randint(1, 3)) for v in vals])


def _rtask_m(rng, det_sizes=True, dag=False, pmax=4, lagmax=5):
    """A random machine-model task: chain of 1-3 nodes, or (dag=True) a fork-join plan -> {a, b} -> final."""
    sz = (lambda: det(rng.randint(1, pmax))) if det_sizes else (lambda: _rdist(rng, 1, pmax + 1, rng.choice([1, 2])))
    lg = lambda: _rdist(rng, 0, lagmax, rng.choice([1, 2]))
    if dag and rng.random() < 0.5:
        nodes = tuple(sz() for _ in range(4))
        arcs = ((0, 1, lg()), (0, 2, lg()), (1, 3, lg()), (2, 3, lg()))
        return MTask(F(rng.randint(0, 2)), F(rng.randint(1, 4)), nodes, arcs)
    n = rng.choice([1, 2, 2, 3])
    return chain_task(rng.randint(0, 2), rng.randint(1, 4), [sz() for _ in range(n)], [lg() for _ in range(n - 1)])


def _rtask_h(rng, qmax=9):
    plan = HNode(q=rng.randint(1, qmax), X=rng.randint(1, 4))
    k = rng.randint(1, 3)
    pcs = tuple((dist(*[(v, 1) for v in rng.sample(range(0, 6), rng.choice([1, 2]))]), rng.randint(1, qmax))
                for _ in range(k))
    fin = HNode(q=sum(t for _, t in pcs) + rng.randint(0, 2), X=rng.randint(1, 3), rho=plan.l_plus, pieces=pcs)
    return HTask(F(rng.randint(0, 4)), F(rng.randint(1, 4)), (plan, fin), ((0, 1, det(0)),))


def _wsum(tasks, C):
    return sum(T.w * c for T, c in zip(tasks, C))


# =========================================================================== Theorem A (deterministic sizes)
def test_thmA_pathwise_bound_every_realization():
    """C_J <= r_J + CP_J + (1/m) sum_{K ranked <= J} W_K for every realization (sizes and lags), DAG tasks."""
    rng = random.Random(1)
    checked = 0
    for _ in range(250):
        ts = [_rtask_m(rng, det_sizes=rng.random() < 0.5, dag=True) for _ in range(rng.randint(1, 4))]
        m = rng.choice([1, 2, 3])
        o = task_index_order(ts)
        for prob, P, D in itertools.islice(realizations(ts), 3):
            TC, _ = sim_machine(ts, P, D, m, o)
            for c, b in zip(TC, pathwise_bound(ts, P, D, m, o)):
                assert c <= b
                checked += 1
    assert checked > 1000


def test_thmA_pathwise_bound_mutations_fail():
    """Non-vacuity: dropping the task's own work, or shrinking 1/m to 1/(m+1), breaks the inequality."""
    par = MTask(F(0), F(1), (det(1), det(1)), ())  # two parallel nodes, one machine
    TC, _ = sim_machine([par], [[F(1), F(1)]], [[]], 1, [0])
    assert TC[0] == 2 and critical_path(par, [1, 1], []) == 1  # own work is needed: 0 + 1 + 0 < 2
    K, J = MTask(F(0), F(10), (det(2),)), MTask(F(0), F(1), (det(1),))
    TC, _ = sim_machine([K, J], [[F(2)], [F(1)]], [[], []], 1, [0, 1])
    assert TC[1] == 3 and 0 + 1 + F(3, 2) < 3  # 1/(m+1) instead of 1/m fails
    assert TC[1] <= pathwise_bound([K, J], [[2], [1]], [[], []], 1, [0, 1])[1]


def test_thmA_two_approx_vs_exact_clairvoyant_m1():
    """E[ALG] <= LB_cp + Smith <= 2 E[OPT_clairvoyant] (exact clairvoyant DP, m = 1, random lags)."""
    rng = random.Random(2)
    worst = F(0)
    for _ in range(50):
        ts = [_rtask_m(rng, det_sizes=True, dag=True, pmax=3, lagmax=3) for _ in range(2)]
        o = task_index_order(ts)
        alg = expected_cost(ts, 1, o)
        l1, l2 = lb_cp(ts), lb_smith_det(ts, 1)
        opt = opt_clair_1m(ts)
        assert alg <= l1 + l2 and l1 <= opt and l2 <= opt and alg <= 2 * opt
        worst = max(worst, alg / opt)
    assert worst > 1  # non-vacuous: the policy is not optimal on these instances
    # sharpness of the Smith term: 20 unit tasks at time 0, m = 1: ALG = 210 = LB_cp + Smith - sum w W exactly,
    # and alg > LB_cp + 0.8 Smith, so an understated Smith term fails here
    ts = [MTask(F(0), F(1), (det(1),)) for _ in range(20)]
    alg = expected_cost(ts, 1, task_index_order(ts))
    assert alg == 210 and lb_smith_det(ts, 1) == 210 and lb_cp(ts) == 20
    assert alg <= lb_cp(ts) + lb_smith_det(ts, 1) and alg > lb_cp(ts) + F(4, 5) * lb_smith_det(ts, 1)


def test_thmA_parallel_machines_vs_lower_bounds():
    """m = 2, 3: E[ALG] <= LB_cp + (1/m) Smith(W) (both are lower bounds on any schedule)."""
    rng = random.Random(3)
    for _ in range(150):
        ts = [_rtask_m(rng, det_sizes=True, dag=True) for _ in range(rng.randint(2, 4))]
        m = rng.choice([2, 3])
        o = task_index_order(ts)
        assert expected_cost(ts, m, o) <= lb_cp(ts) + lb_smith_det(ts, m)
    # sharpness (m = 2): 20 unit tasks at 0: ALG = 110 <= 20 + 105, but 20 + 0.8 * 105 = 104 < 110
    ts = [MTask(F(0), F(1), (det(1),)) for _ in range(20)]
    alg = expected_cost(ts, 2, task_index_order(ts))
    assert alg == 110 and lb_smith_det(ts, 2) == 105
    assert alg <= lb_cp(ts) + lb_smith_det(ts, 2) and alg > lb_cp(ts) + F(4, 5) * lb_smith_det(ts, 2)


def test_thmA_lag_trap_lower_bound_four_thirds():
    """Remark only (superseded as a lower bound by the Schulz-Skutella family below): a lag trap gives ratio
    (4L+1)/(3L+2) -> 4/3 against the exact clairvoyant optimum."""
    for L in (4, 10, 20):
        ts = lag_trap(L)
        o = task_index_order(ts)
        assert o == [1, 0]
        alg, opt = expected_cost(ts, 1, o), opt_clair_1m(ts)
        assert alg == 4 * L + 1 and opt == 3 * L + 2
    assert F(81, 62) > F(13, 10)


def _schulz_skutella(n, scale=1):
    """Schulz-Skutella (J. Sched. 2002) / Megow-Schulz (ORL 2004, Lemma 2.2) family for P-WSPT on one machine:
    w_j = 1, p_j = n - j/n, r_j = j n - j(j+1)/(2n), j = 0..n (single-node tasks, no lags)."""
    return [MTask(F(scale) * (j * n - F(j * (j + 1), 2 * n)), F(1), (det(F(scale) * (n - F(j, n))),))
            for j in range(n + 1)]


def _nonpreemptive_in_release_order(ts):
    t = tot = F(0)
    for T in ts:
        t = max(t, T.r) + T.nodes[0][0][0]
        tot += T.w * t
    return tot


def test_thmA_constant_two_is_tight_schulz_skutella():
    """Theorem A's 2 is tight (known): on single-node tasks without lags P-WSPT-DAG is P-WSPT, and the Schulz-
    Skutella family drives P-WSPT's ratio to 2 on one machine with unit weights. Exact: scaled n = 3 gives
    ALG = 228 vs OPT = 160 (exact preemptive DP = non-preemptive release-order schedule); n = 150 gives 1.9846."""
    ts = _schulz_skutella(3, 6)
    alg = expected_cost(ts, 1, task_index_order(ts))
    ub = _nonpreemptive_in_release_order(ts)
    assert alg == 228 and ub == 160 and opt_pmtn_dp(ts, 1) == 160
    ratios = []
    for n in (5, 20, 150):
        ts = _schulz_skutella(n)
        ratios.append(expected_cost(ts, 1, task_index_order(ts)) / _nonpreemptive_in_release_order(ts))
    assert ratios[0] < ratios[1] < ratios[2] and F(198, 100) < ratios[2] < 2
    assert ratios[2] > F(4, 3) + F(1, 2)  # far above the lag-trap remark (4/3) and the ext sqrt(2) family


def test_thmA_clairvoyant_below_nonanticipatory_dp():
    """Sanity of the benchmark chain LB <= OPT_clair <= OPT_na (preemptive DP), deterministic sizes, m = 1."""
    rng = random.Random(4)
    gap = False
    for _ in range(40):
        ts = [_rtask_m(rng, det_sizes=True, pmax=3, lagmax=4) for _ in range(2)]
        oc, ona = opt_clair_1m(ts), opt_pmtn_dp(ts, 1)
        assert max(lb_cp(ts), lb_smith_det(ts, 1)) <= oc <= ona
        gap |= oc < ona
    assert gap  # information about lags has value: the clairvoyant optimum is sometimes strictly better


def test_fanout_collapse_to_max_lag():
    d1, d2 = dist((1, 1), (4, 1)), dist((2, 1), (3, 1))
    assert dmax(d1, d2) == dist((2, F(1, 4)), (3, F(1, 4)), (4, F(1, 2)))


# ============================================================================ Theorem B (random node sizes)
def _rhoB(m, Delta, chain):
    x = max(1, (m - 1) * (Delta - 1))
    return 2 + x / (2 * m) if chain else 2 + F(x) / 2


def test_thmB_online_index_policy_vs_nonpreemptive_dp():
    """E[ALG] <= LB1 + LB2 + (1/(2m) - gamma) sum w E[W] and the resulting ratio vs the exact non-preemptive
    non-anticipatory optimum (chains: 2 + max(1,(m-1)(Delta-1))/(2m); DAGs: 2 + max(1,(m-1)(Delta-1))/2)."""
    rng = random.Random(5)
    n_inst = 0
    for _ in range(150):
        dag = rng.random() < 0.3
        ts = [_rtask_m(rng, det_sizes=False, dag=dag, pmax=3, lagmax=3) for _ in range(2)]
        m = rng.choice([1, 2])
        Delta = max(cv2(d) for T in ts for d in T.nodes)
        o = task_index_order(ts)
        alg = expected_cost(ts, m, o)
        l1, l2 = lb_cp(ts), lb2_task_msu(ts, m)
        ew = sum(T.w * sum(mean(d) for d in T.nodes) for T in ts)
        assert alg <= l1 + l2 + (F(1, 2 * m) - msu_gamma(ts, m)) * ew
        opt = opt_np_dp(ts, m)
        assert l1 <= opt and l2 <= opt
        chain = all(len(T.arcs) == len(T.nodes) - 1 for T in ts)
        assert alg <= _rhoB(m, Delta, chain) * opt
        n_inst += 1
    assert n_inst == 150
    # sharpness of LB2: (i) high variance (8 i.i.d. tasks {1 w.p. 9/10, 40 w.p. 1/10}, m = 2): LB2 <= E[list
    # schedule] (a non-preemptive policy, so >= OPT_np), while LB2 with gamma := 0 is 392/5 > E[list] = 69.97 >= OPT_np
    # (the variance correction is load-bearing); (ii) 11 unit jobs on m = 1: LB2 = 121/2 vs OPT_np = 66 (ratio
    # 12/11 < 11/10), so an LB2 overstated by 10% would exceed the exact optimum
    R = MTask(F(0), F(1), (dist((1, F(9, 10)), (40, F(1, 10))),))
    ts = [R] * 8
    ub = expected_cost(ts, 2, list(range(8)), preemptive=False)
    assert msu_gamma(ts, 2) < 0
    assert lb2_task_msu(ts, 2) <= ub < lb2_task_msu(ts, 2, Delta=F(0)) == F(392, 5)  # Delta = 0 <=> gamma = 0
    ts = [MTask(F(0), F(1), (det(1),)) for _ in range(11)]
    opt = opt_np_dp(ts, 1)
    assert opt == 66 and lb2_task_msu(ts, 1) <= opt < F(11, 10) * lb2_task_msu(ts, 1)


def test_thmB_lb2_greedy_equals_task_lp():
    rng = random.Random(6)
    for _ in range(25):
        ts = [_rtask_m(rng, det_sizes=False, pmax=6) for _ in range(rng.randint(1, 4))]
        m = rng.choice([1, 2, 4])
        assert lb2_task_msu(ts, m) == lb2_task_lp_z3(ts, m)


def test_thmB_offline_lp_policy():
    """Offline: LP <= OPT_np (DP) and E[ALG with LP order] <= (2 + max(1,(m-1)Delta/m)) LP."""
    rng = random.Random(7)
    for _ in range(60):
        ts = [_rtask_m(rng, det_sizes=False, pmax=3, lagmax=3) for _ in range(2)]
        m = rng.choice([1, 2])
        Delta = max(cv2(d) for T in ts for d in T.nodes)
        lpv = msu_lp(ts, m)[0]
        assert lpv <= opt_np_dp(ts, m)
        alg = expected_cost(ts, m, lp_order(ts, m))
        assert alg <= (2 + max(1, F(m - 1) * Delta / m)) * lpv


def test_msu_variance_correction_is_necessary():
    """Without the -(m-1)(Delta-1)/(2m) term the 'LP' exceeds the exact non-preemptive optimum."""
    R = MTask(F(0), F(1), (dist((1, F(9, 10)), (30, F(1, 10))),))
    ts = [R] * 4
    opt = opt_np_dp(ts, 2)
    assert msu_lp(ts, 2, drop_variance=True)[0] > opt >= msu_lp(ts, 2)[0]


def test_nonpreemptive_benchmark_is_necessary():
    """The MSU bound is not valid against preemptive non-anticipatory policies (probing beats it)."""
    R = MTask(F(0), F(1), (dist((1, F(3, 4)), (8, F(1, 4))),))
    ts = [R] * 3
    assert msu_lp(ts, 1)[0] == opt_np_dp(ts, 1) == F(33, 2)
    assert opt_pmtn_dp(ts, 1) < msu_lp(ts, 1)[0]


def test_thmB_constant_algebra_z3():
    """(i) C S >= S^2/(2m) + b Q, 0 <= Q <= C S  =>  S/m <= (1 + max(1, (m-1)Delta/m)) C  (SU2005 Lemma 3.3 form);
    (ii) 1/(2m) - min(0, 1/2 - (m-1)Delta/(2m)) = max(1, (m-1)(Delta-1))/(2m)."""
    C, S, Q, m, Dl = z3.Reals("C S Q m Delta")
    b = F(1, 2) - (m - 1) * Dl / (2 * m)
    base = [C > 0, S > 0, Q >= 0, Q <= C * S, m >= 1, Dl >= 0, C * S >= S * S / (2 * m) + b * Q]
    kap = z3.If((m - 1) * Dl / m >= 1, 1 + (m - 1) * Dl / m, 2)
    assert _proved(S / m <= kap * C, *base)
    assert not _proved(S / m <= F(19, 10) * C, *base)  # mutation: a constant below 2 is false
    gam = z3.If(b < 0, b, 0)
    lhs = 1 / (2 * m) - gam
    rhs = z3.If((m - 1) * (Dl - 1) >= 1, (m - 1) * (Dl - 1), 1) / (2 * m)
    assert _proved(lhs == rhs, m >= 1, Dl >= 0)


# ===================================================================== Theorem D (hybrid model, KV memory)
def _h_cases(seed, n, mem=True):
    rng = random.Random(seed)
    for _ in range(n):
        ts = [_rtask_h(rng) for _ in range(rng.randint(2, 4))]
        L = h_Lmax(ts)
        Mp = rng.choice([None, L + rng.randint(L // 2, 3 * L)]) if mem else None
        yield rng, ts, Mp


def _kappa_stress():
    """Tiny high-priority arrivals repeatedly kick a long prefill (needs the kick charge kappa in Z')."""
    qz, alpha, a = 200, F(1, 10), F(1, 100)
    J = HTask(F(0), F(1), (HNode(q=qz, X=1),))
    ys, t = [], F(0)
    for _ in range(5):
        t += alpha * qz - F(1, 10)
        ys.append(HTask(t, F(1000), (HNode(q=1, X=1),)))
        t += alpha + a
    ts = [J] + ys
    return ts, [{} for _ in ts], HCfg(alpha=alpha, a=a, m=2, Mp=qz + 2)


def _margin_stress():
    """Needs the area term a X l+/M with M = M' - L_max (not M')."""
    hi, x = HTask(F(0), F(100), (HNode(q=5, X=5),)), HTask(F(0), F(1), (HNode(q=9, X=1),))
    return [hi, x], [{}, {}], HCfg(alpha=F(1, 1000), a=F(1), m=10, Mp=19)


def test_thmD_pathwise_bound_any_keep_policy_and_preload():
    """C_J <= r_J + CP'_J + sum_{K ranked <= J} Z'_K under kicks, every keep policy (evict-always, LRU, random
    eviction order of held KV) and lowest-priority preloads; memory invariant asserted inside the simulator."""
    checked = kicked = hits = 0
    for rng, ts, Mp in _h_cases(8, 200):
        for hold, pre in (("none", False), ("lru", False), ("rand", True), ("lru", True)):
            cfg = HCfg(alpha=F(1, 4), a=F(1), m=rng.choice([1, 2, 3]), Mp=Mp, hold=hold, preload=pre, seed=3)
            o = h_order(ts, HCfg(alpha=cfg.alpha, a=cfg.a, m=cfg.m, Mp=Mp))
            for prob, lag in itertools.islice(h_realizations(ts), 2):
                st = sim_hybrid(ts, lag, cfg, order=o)
                for c, b in zip(st["C"], h_bound(ts, lag, cfg, o)):
                    assert c <= b
                    checked += 1
                kicked += st["kicks"] > 0
                hits += st["held_hits"] > 0
    assert checked > 1500 and kicked > 10 and hits > 50
    # stress instances on which Z' without kappa, or without (or with a smaller) area term, is violated
    for ts, lag, cfg in (_kappa_stress(), _margin_stress()):
        st = sim_hybrid(ts, lag, cfg)
        for c, b in zip(st["C"], h_bound(ts, lag, cfg)):
            assert c <= b


def test_thmD_ratio_against_engine_lower_bounds():
    """E[ALG] <= (1+beta_cp) LB_cp + max(1+beta, theta) S(sigma + A/M), with OPT >= LB_cp and OPT >= S(.)/2."""
    for rng, ts, Mp in _h_cases(9, 200):
        cfg = HCfg(alpha=F(1, 4), a=F(1), m=rng.choice([1, 2]), Mp=Mp)
        alg = sum(prob * _wsum(ts, sim_hybrid(ts, lag, cfg)["C"]) for prob, lag in h_realizations(ts))
        lbcp, scos = h_lower_bounds(ts, cfg)
        bcp, b, th = h_constants(ts, cfg)
        assert alg <= (1 + bcp) * lbcp + max(1 + b, th) * scos
        # the two proof steps, per instance, with the exact constants (an understated constant fails these):
        # (i) per node alpha(q+rho) + aX <= (1+beta_cp)(alpha q + a X), alpha rho + kappa <= beta sigma,
        #     l+ <= theta l-, each with equality at some node; (ii) Z'_K <= max(1+beta, theta)(sigma_K + A_K/M)
        L = h_Lmax(ts)
        kap = (lambda n: cfg.alpha * (n.l_plus + L)) if Mp is not None else (lambda n: F(0))
        r_cp = [cfg.alpha * n.rho / (cfg.alpha * n.q + cfg.a * n.X) for T in ts for n in T.nodes]
        r_b = [(cfg.alpha * n.rho + kap(n)) / (cfg.alpha * n.q + cfg.c * n.X) for T in ts for n in T.nodes]
        r_th = [F(n.l_plus, n.l_minus) for T in ts for n in T.nodes] if Mp is not None else [F(1)]
        assert max(r_cp) == bcp and max(r_b) == b and max(r_th) == th
        Z, S, A = h_sizes(ts, cfg)
        assert all(z <= max(1 + b, th) * (x + y) for z, x, y in zip(Z, S, A))
        for prob, lag in itertools.islice(h_realizations(ts), 2):
            for T, t in zip(ts, range(len(ts))):
                assert h_cp(T, lag[t], cfg, "alg") <= (1 + bcp) * h_cp(T, lag[t], cfg, "opt")
    # constants used in the proof are attained: a one-node task with rho > 0 makes beta_cp = alpha rho/(alpha q + aX)
    T = HTask(F(0), F(1), (HNode(q=1, X=1), HNode(q=1, X=1, rho=20, pieces=((det(0), 1),))), ((0, 1, det(0)),))
    cfg = HCfg(alpha=F(1, 4), a=F(1), m=1)
    assert h_constants([T], cfg) == (F(4), F(4), F(1))


def test_thmD_hypotheses_are_necessary():
    # (1) without kicks (non-preemptive memory) a high-priority arrival waits for a long low-priority decode
    low, hi = HTask(F(0), F(1), (HNode(q=1, X=40),)), HTask(F(1, 2), F(100), (HNode(q=1, X=1),))
    cfg = HCfg(alpha=F(1, 10), a=F(1), m=4, Mp=42)
    lag = [{}, {}]
    assert h_order([low, hi], cfg) == [1, 0]
    bd = h_bound([low, hi], lag, cfg)
    assert sim_hybrid([low, hi], lag, cfg)["C"][1] <= bd[1] < sim_hybrid([low, hi], lag, cfg, kicks=False)["C"][1]
    # (2) the kick charge kappa is needed: tiny high-priority arrivals repeatedly kick a long prefill
    ts, lag, cfg = _kappa_stress()
    J, alpha = ts[0], cfg.alpha
    st = sim_hybrid(ts, lag, cfg)
    o = h_order(ts, cfg)
    Z, _, _ = h_sizes(ts, cfg)
    L = h_Lmax(ts)
    pos = {k: i for i, k in enumerate(o)}
    Zk = [z - sum(alpha * (n.l_plus + L) for n in T.nodes) for z, T in zip(Z, ts)]
    no_kappa = h_cp(J, lag[0], cfg, "alg") + sum(Zk[k] for k in range(len(ts)) if pos[k] <= pos[0])
    assert st["kicks"] == 5 and no_kappa < st["C"][0] <= h_bound(ts, lag, cfg)[0]
    # (3) the memory margin M' = M + L_max is needed: normalizing the area by M' instead of M' - L_max fails
    ts, lag, cfg = _margin_stress()
    x = ts[1]
    st = sim_hybrid(ts, lag, cfg)
    Lm = h_Lmax(ts)
    Zm = [sum(cfg.alpha * (n.q + n.rho) + cfg.c * n.X + cfg.alpha * (n.l_plus + Lm) + cfg.a * n.X * n.l_plus / cfg.Mp
              for n in T.nodes) for T in ts]
    mutated = h_cp(x, lag[1], cfg, "alg") + Zm[0] + Zm[1]
    assert mutated < st["C"][1] <= h_bound(ts, lag, cfg)[1]


def _l1_counterexample():
    """Verifier's instance: plan (q=1, X=1) -> final with two fan-out pieces, 10 tokens returning at lag 0 and
    1 token at lag 10; alpha = a = 1, m = 1 (c = 1), unbounded memory."""
    T = HTask(F(0), F(1), (HNode(q=1, X=1), HNode(q=11, X=1, rho=0, pieces=((det(0), 10), (det(10), 1)))),
              ((0, 1, det(0)),))
    return T, HCfg(alpha=F(1), a=F(1), m=1)


def test_COUNTEREXAMPLE_L1_needs_restricted_benchmark():
    """(L1) is a lower bound only for engines that do not prefill a node before it is available. An explicit
    speed-1 schedule that prefills the returned 10-token piece during the remaining wait costs 16 < LB_cp = 26;
    the preload-aware LB0 (node length alpha q0 + (a+c) X, q0 = 0 with pieces) is 15 <= 16."""
    T, cfg = _l1_counterexample()
    al, a, c = cfg.alpha, cfg.a, cfg.c
    t = al * 1 + (a + c) * 1  # plan prefill then batch-1 decode: 3
    t_plan = t
    t += al * 10  # piece A returned at t_plan; prefill it before the node is available
    assert t <= t_plan + 10
    t = t_plan + 10 + al * 1 + (a + c) * 1  # piece B returns, prefill 1 token, decode 1 token
    assert t == 16
    lb, _ = h_lower_bounds([T], cfg)
    lb0, _ = h_lower_bounds([T], cfg, preload_aware=True)
    assert lb == 26 > t and lb0 == 15 <= t


def test_thmD_unrestricted_benchmark_corrected_bound():
    """Unrestricted benchmark (may preload fan-out pieces): E[ALG] <= min{(1+beta_cp0) LB0, LB0 + (1+beta) sum w
    sigma} + max(1+beta, theta) S(sigma + A/M), each piece <= its multiple of OPT (LB0 <= OPT, sum w sigma <= OPT,
    S <= 2 OPT); node-wise CP' <= (1+beta_cp0) CP0 on every realization. Calibrated Thm D at M' = 64k: 9.985 ->
    10.007. Mutation: the restricted beta_cp against LB0 fails on the (L1) counterexample."""
    for rng, ts, Mp in _h_cases(19, 120):
        cfg = HCfg(alpha=F(1, 4), a=F(1), m=rng.choice([1, 2]), Mp=Mp)
        alg = sum(prob * _wsum(ts, sim_hybrid(ts, lag, cfg)["C"]) for prob, lag in h_realizations(ts))
        lb0, scos = h_lower_bounds(ts, cfg, preload_aware=True)
        bcp0, b, th = h_constants(ts, cfg, preload_aware=True)
        wsig = sum(T.w * sum(cfg.alpha * n.q + cfg.c * n.X for n in T.nodes) for T in ts)
        assert alg <= min((1 + bcp0) * lb0, lb0 + (1 + b) * wsig) + max(1 + b, th) * scos
        assert lb0 <= h_lower_bounds(ts, cfg)[0]
        for prob, lag in itertools.islice(h_realizations(ts), 2):
            for t, T in enumerate(ts):
                assert h_cp(T, lag[t], cfg, "alg") <= (1 + bcp0) * h_cp(T, lag[t], cfg, "opt0")
                assert h_cp(T, lag[t], cfg, "alg") <= h_cp(T, lag[t], cfg, "opt0") + sum(
                    (1 + b) * (cfg.alpha * n.q + cfg.c * n.X) for n in T.nodes)
        second = (1 + b) if Mp is None else 2 * max(1 + b, th)
        assert thmD_constant(ts, cfg, preload_aware=True) == min(1 + bcp0, 2 + b) + second
    T, cfg = _l1_counterexample()
    lag = [{("a", 0): F(0), ("p", (1, 0)): F(0), ("p", (1, 1)): F(10)}]
    alg = sim_hybrid([T], lag, cfg)["C"][0]
    lb0 = h_lower_bounds([T], cfg, preload_aware=True)[0]
    assert alg == 24 and h_constants([T], cfg)[0] == 0 and h_constants([T], cfg, True)[0] == 11
    assert (1 + 0) * lb0 < h_cp(T, lag[0], cfg, "alg") == alg <= (1 + 11) * lb0
    al, a = F("0.000114"), F("0.027767")
    TK = HTask(F(0), F(1), (HNode(q=2000, X=200), HNode(q=1500, X=300, rho=2200, pieces=((det(1), 750), (det(1), 750)))),
               ((0, 1, det(1)),))
    c64 = HCfg(alpha=al, a=a, m=51, Mp=64000)
    assert F(998, 100) < thmD_constant([TK], c64) < F(999, 100)
    assert F(1000, 100) < thmD_constant([TK], c64, preload_aware=True) < F(1001, 100)
    cinf = HCfg(alpha=al, a=a, m=51)
    assert F(277, 100) < thmD_constant([TK], cinf) < F(278, 100) < F(280, 100) < thmD_constant([TK], cinf, True) < F(281, 100)


def test_lemmaD0_charge_tight_family():
    """Verifier's family: z1 (l+ = L) decoding, z2 (l+ = s) released at L - s, top-priority y (l+ = s + 1) arriving
    at L - 3/2, M' = L + s. One admission event kicks both: kicked reservation l+_y + L_max - 1, and the recompute
    exceeds l+_y + L_max - 3, so the per-event charge l+_y + L_max of Lemma D0 is tight up to an additive 1."""
    for L, sz in ((10, 3), (50, 20), (200, 50)):
        z1 = HTask(F(0), F(1), (HNode(q=1, X=L - 1),))
        z2 = HTask(F(L - sz), F(1), (HNode(q=1, X=sz - 1),))
        y = HTask(F(L) - F(3, 2), F(10**6), (HNode(q=1, X=sz),))
        ts = [z1, z2, y]
        cfg = HCfg(alpha=F(1, 1000), a=F(1), m=3, Mp=L + sz)
        st = sim_hybrid(ts, [{}, {}, {}], cfg, order=[2, 0, 1])
        ly = y.nodes[0].l_plus
        assert h_Lmax(ts) == L and st["kicked_res"] == [ly + L - 1] and st["recompute"] > ly + L - 3
        for c, b in zip(st["C"], h_bound(ts, [{}, {}, {}], cfg, [2, 0, 1])):
            assert c <= b


def test_R5c_keep_policies_never_increase_recompute_bound_but_per_task_dominance_fails():
    """Held KV in the lowest class keeps the guarantee (above); per-task 'never later' is false (exact instance)."""
    t0 = HTask(F(0), F(4), (HNode(7, 3), HNode(3, 3, rho=10, pieces=((det(2), 1),))), ((0, 1, det(0)),))
    t1 = HTask(F(4), F(4), (HNode(5, 2), HNode(9, 2, rho=7, pieces=((det(2), 8), (dist((1, 1), (2, 1)), 1)))),
               ((0, 1, det(0)),))
    t2 = HTask(F(0), F(2), (HNode(5, 3), HNode(10, 2, rho=8, pieces=((dist((1, 1), (5, 1)), 1),
                                                                    (dist((2, 1), (4, 1)), 4), (det(5), 4)))),
               ((0, 1, det(0)),))
    t3 = HTask(F(3), F(4), (HNode(8, 1), HNode(5, 3, rho=9, pieces=((dist((0, 1), (5, 1)), 3),))), ((0, 1, det(0)),))
    ts = [t0, t1, t2, t3]
    lag = [{("a", 0): F(0), ("p", (1, 0)): F(2)},
           {("a", 0): F(0), ("p", (1, 0)): F(2), ("p", (1, 1)): F(1)},
           {("a", 0): F(0), ("p", (1, 0)): F(1), ("p", (1, 1)): F(2), ("p", (1, 2)): F(5)},
           {("a", 0): F(0), ("p", (1, 0)): F(0)}]
    base = HCfg(alpha=F(1, 4), a=F(1), m=1)
    o = h_order(ts, base)
    c0 = sim_hybrid(ts, lag, base, order=o)["C"]
    c1 = sim_hybrid(ts, lag, HCfg(alpha=F(1, 4), a=F(1), m=1, hold="lru", preload=True), order=o)["C"]
    assert c1[2] > c0[2]  # one task is later with holding + preloading
    assert _wsum(ts, c1) < _wsum(ts, c0)  # but the weighted total improves here
    for c, b in zip(c1, h_bound(ts, lag, base, o)):
        assert c <= b


def test_R6_isolated_preload_saving_is_exact():
    """One CHAIN task alone: completion(evict-always) - completion(hold + preload) = alpha * (held tokens used)."""
    rng = random.Random(10)
    pos = 0
    for _ in range(60):
        ts = [_rtask_h(rng)]
        cfg0 = HCfg(alpha=F(1, 4), a=F(1), m=1)
        cfg1 = HCfg(alpha=F(1, 4), a=F(1), m=1, hold="lru", preload=True)
        for prob, lag in itertools.islice(h_realizations(ts), 3):
            s0, s1 = sim_hybrid(ts, lag, cfg0), sim_hybrid(ts, lag, cfg1)
            assert s0["C"][0] - s1["C"][0] == cfg0.alpha * s1["held_hits"]
            pos += s1["held_hits"] > ts[0].nodes[1].rho
    assert pos > 20  # preloading (beyond the parent context) strictly helps on many instances


def test_COUNTEREXAMPLE_R6_not_exact_for_DAGs_either_direction():
    """For DAG tasks the isolated saving is not alpha * held hits, in either direction (Graham-type anomalies on
    the shared prefill server / decoders): (i) verifier: 1/2 < alpha hits = 1 (hit off the critical path);
    (ii) 1 > alpha hits = 3/4 (holding lets b finish its prefill before a arrives); (iii) -1/4 < 0 (holding makes
    the isolated task LATER). All cross-checked with the verifier's independent simulator. Thm D still holds."""
    def run(T, lag, al, m, pre):
        c0, c1 = HCfg(alpha=al, a=F(1), m=m), HCfg(alpha=al, a=F(1), m=m, hold="lru", preload=pre)
        s0, s1 = sim_hybrid([T], lag, c0), sim_hybrid([T], lag, c1)
        for c, bd in zip(s1["C"], h_bound([T], lag, c0)):
            assert c <= bd
        return s0["C"][0] - s1["C"][0], al * s1["held_hits"]
    T = HTask(F(0), F(1), (HNode(q=1, X=1), HNode(q=1, X=1, rho=2), HNode(q=1, X=100, rho=2)),
              ((0, 1, det(5)), (0, 2, det(0))))
    sv, ah = run(T, [{("a", 0): F(5), ("a", 1): F(0)}], F(1, 4), 2, True)
    assert sv == F(1, 2) and ah == 1
    T = HTask(F(0), F(1), (HNode(q=1, X=1), HNode(q=2, X=1, rho=1), HNode(q=3, X=1, rho=2)),
              ((0, 1, det(1)), (0, 2, det(0))))
    sv, ah = run(T, [{("a", 0): F(1), ("a", 1): F(0)}], F(1, 4), 1, False)
    assert sv == 1 and ah == F(3, 4) and sv > ah
    T = HTask(F(0), F(1), (HNode(q=3, X=1), HNode(q=2, X=1), HNode(q=1, X=2, rho=1), HNode(q=3, X=3)),
              ((0, 1, det(1)), (0, 2, det(1)), (0, 3, det(0))))
    sv, ah = run(T, [{("a", 0): F(1), ("a", 1): F(1), ("a", 2): F(0)}], F(1, 4), 2, False)
    assert sv == F(-1, 4) and ah == F(1, 4)


def test_R0a_engine_lift_and_converse_z3():
    """Fluid engine at speed s with prefill share u: decode rate s(1-u)/(a+cB), prefill rate s u/alpha.
    s = 3, u = 1/3 dominates the hybrid rates (1/a per decode machine, 1/alpha prefill) for B <= m = a/c, and no
    s < 3 does; conversely every engine allocation at speed 1 is within the hybrid rates."""
    a, c, al, B, u, s = z3.Reals("a c alpha B u s")
    hyp = [a > 0, c > 0, al > 0, B >= 0, B <= a / c]
    assert _proved(z3.And(3 * (1 - F(1, 3)) / (a + c * B) >= 1 / a, 3 * F(1, 3) / al >= 1 / al), *hyp)
    worst = hyp + [B == a / c, s < 3, s > 0, u >= 0, u <= 1]
    assert _proved(z3.Or(s * (1 - u) / (a + c * B) < 1 / a, s * u / al < 1 / al), *worst)
    conv = [a > 0, c > 0, al > 0, B >= 0, u >= 0, u <= 1]
    assert _proved(z3.And((1 - u) / (a + c * B) <= 1 / a, B * (1 - u) / (a + c * B) <= (a / c) * (1 / a),
                          u / al <= 1 / al), *conv)
    # discrete step: (a + c m + alpha Pbar)/s <= a with c m = a and alpha Pbar = a iff s >= 3
    assert _proved(z3.Implies(s >= 3, 3 * a / s <= a), a > 0, s > 0)
    assert _proved(z3.Implies(s < 3, 3 * a / s > a), a > 0, s > 0)


def test_R0b_discrete_engine_bound():
    """Atomic steps at speed 3 with Pbar = a/alpha: C_J <= hybrid bound + a(|J| + N_J)."""
    checked = 0
    for rng, ts, Mp in _h_cases(12, 250):
        cfg = HCfg(alpha=F(1, 4), a=F(1), m=rng.choice([1, 2, 3]), Mp=Mp)
        for prob, lag in itertools.islice(h_realizations(ts), 2):
            st = sim_discrete(ts, lag, cfg)
            for c, b in zip(st["C"], disc_bound(ts, lag, cfg, st["completions"])):
                assert c <= b
                checked += 1
    assert checked > 400


def test_R0c_machine_cap_loses_m_over_meff():
    """(i) m = 1 vs m = 8, unbounded memory: 8 equal decodes, ratio exactly 1286/287 ~ (n+1)/2.
    (ii) Consistent instance of the cap rule m_eff = min(m, floor(M'/L_max)) (verifier): add one low-weight
    long-context task (l+ = 50) so L_max = 50, M' = 99: m_eff = 1, while all eight 11-token decodes fit for P;
    the eight short tasks lose a factor > 4 under the cap."""
    ts = [HTask(F(0), F(1), (HNode(q=1, X=10),)) for _ in range(8)]
    cfg_cap = HCfg(alpha=F(1, 100), a=F(1), m=1)  # m_eff = 1
    cfg_full = HCfg(alpha=F(1, 100), a=F(1), m=8)
    lag = [{} for _ in ts]
    w_cap = _wsum(ts, sim_hybrid(ts, lag, cfg_cap)["C"])
    w_full = _wsum(ts, sim_hybrid(ts, lag, cfg_full)["C"])
    assert w_cap / w_full == F(1286, 287) and w_cap / w_full > 4
    big = HTask(F(0), F(1, 1000), (HNode(q=40, X=10),))
    ts2 = ts + [big]
    lag2 = [{} for _ in ts2]
    Mp = 99
    assert h_Lmax(ts2) == 50 and min(8, Mp // h_Lmax(ts2)) == 1
    rP = sim_hybrid(ts2, lag2, HCfg(alpha=F(1, 100), a=F(1), m=8, Mp=Mp))["C"]
    rcap = sim_hybrid(ts2, lag2, HCfg(alpha=F(1, 100), a=F(1), m=1, Mp=Mp))["C"]
    assert sum(rcap[:8]) / sum(rP[:8]) > 4


# ============================================================================ R5: keep / evict layer
def test_R5a_best_ttl_is_optimal_and_within_e_over_e_minus_1():
    rng = random.Random(13)
    for _ in range(300):
        D = _rdist(rng, 0, 12, rng.randint(1, 4))
        h, R = F(rng.randint(1, 5)), F(rng.randint(1, 30))
        tau, cost = ttl_best(D, h, R)
        # every TTL in a fine grid (including non-candidates) is no better
        for g in range(0, 4 * 13):
            assert ttl_cost(D, h, R, F(g, 4)) >= cost
        assert cost <= ttl_cost(D, h, R, None)
        clair = ttl_clair(D, h, R)
        assert cost <= E_OVER_E1_UP * clair
    z, Dw = ski_worst_lp(16)
    assert z > F(155, 100)  # near-tight: the best TTL can be forced to ratio > 1.55 (limit e/(e-1))
    assert ttl_best(Dw, F(1), F(1))[1] == z * ttl_clair(Dw, F(1), F(1))
    assert ttl_best(Dw, F(1), F(1))[1] <= E_OVER_E1_UP * ttl_clair(Dw, F(1), F(1))  # a constant <= 1.55 fails


def test_R5a_exponential_wait_rule_z3():
    """Exponential wait (rate th): d/dtau cost = e^{-th tau}(h - R th), so hold iff h E[D] < R (sign algebra)."""
    h, R, th, Ex = z3.Reals("h R th E")
    hyp = [h > 0, R > 0, th > 0, Ex > 0, Ex <= 1]
    assert _proved((Ex * (h - R * th) > 0) == (h * (1 / th) > R), *hyp)


def test_R5a_randomized_ski_rental_density_numeric():
    """Karlin et al. density p(t) = e^{t/B}/(B(e-1)) on [0,B]: E cost = e/(e-1) min(D, B) for every D (float)."""
    B = 1.0
    n = 20000
    for D in (0.05, 0.3, 0.7, 1.0, 3.0):
        tot = 0.0
        for i in range(n):
            t = (i + 0.5) * B / n
            p = math.exp(t / B) / (B * (math.e - 1)) * (B / n)
            tot += p * ((t + B) if D > t else D)
        assert abs(tot - math.e / (math.e - 1) * min(D, B)) < 1e-6


def test_R5b_keep_lp_strong_duality_and_sigma_star():
    rng = random.Random(14)
    for _ in range(25):
        k = rng.randint(1, 3)
        cls = [KeepClass(F(rng.randint(1, 3)), F(rng.randint(1, 5)), F(rng.randint(1, 8)), _rdist(rng, 0, 6, rng.randint(1, 3)))
               for _ in range(k)]
        u, rs = F(rng.randint(1, 8), 10), F(rng.randint(1, 30), 10)
        nu_cp, nu_ld, sstar = sigma_star_prices(u, rs)
        Mh = F(rng.randint(0, 30))
        lpv, sol = keep_lp_z3(cls, nu_cp, nu_ld, Mh)
        dv, eta = keep_dual(cls, nu_cp, nu_ld, Mh)
        assert lpv == dv
        if eta == 0:  # memory price zero: issue (with full hold) iff survival 1 > sigma*, i.e. always
            for i, c in enumerate(cls):
                assert sol[(i, None)] == 0 or sstar >= 1
    # sigma* is the GPU-price ratio of BE' and the reduced-cost sign rule
    u, rs, sg, P, X, lam = z3.Reals("u rs sigma P X lam")
    ncp, nld = 1 / (1 - u), rs / (1 - u) ** 2
    hyp = [u >= 0, u < 1, rs > 0, sg >= 0, sg <= 1, P > 0]
    assert _proved(nld / (nld + ncp) == rs / (1 - u + rs), *hyp)
    assert _proved((P * (nld * (1 - sg) - ncp * sg) < 0) == (sg > rs / (1 - u + rs)), *hyp)
    # BE' saving / P -> sigma nu_cp - (1-sigma) nu_ld as P -> 0 (exact closed form of the saving)
    den = 1 - u - lam * (1 - sg) * P
    delta = X / (1 - u) - (X - sg * P) / den
    hyp2 = hyp + [X > 0, lam > 0, rs == lam * X, den > 0]
    assert _proved(delta == P * (sg * (1 - u) - rs * (1 - sg)) / ((1 - u) * den), *hyp2)


# ============================================================================ R7: lower bounds / necessity
def test_R7_hold_only_is_unbounded():
    """Hold-only (never evict a context during a tool wait). Each task holds >= 4 tokens from plan start to final end,
    so with M' = 11 at most k = 2 tasks are in flight and the i-th completion is >= ceil(i/k) (t_plan + D + t_final)
    for EVERY hold-only policy. Evict-always overlaps all waits: the ratio grows with D toward
    sum_i ceil(i/k) / n = 7/2 (n = 12), and ~ n/(2k) in general, i.e. unbounded in n."""
    al, a, n, Mp = F(1, 4), F(1), 12, 11
    ratios = []
    for D in (8, 32, 128):
        ts = [HTask(F(0), F(1), (HNode(q=3, X=1), HNode(q=1, X=1, rho=4, pieces=((det(D), 1),))), ((0, 1, det(0)),))
              for _ in range(n)]
        cfg = HCfg(alpha=al, a=a, m=1, Mp=Mp)
        lag = [{("a", 0): F(0), ("p", (1, 0)): F(D)} for _ in ts]
        ev = _wsum(ts, sim_hybrid(ts, lag, cfg)["C"])
        k = Mp // ts[0].nodes[0].l_plus
        t_in_flight = (al * 3 + a) + D + (al * 1 + a)  # minimum plan + wait + final (context held)
        hold_lb = sum(-(-i // k) * t_in_flight for i in range(1, n + 1))
        ratios.append(hold_lb / ev)
    assert ratios[0] < ratios[1] < ratios[2] and ratios[2] > 3
    assert hold_only_lower(4, F(3), F(1)) == sum(i * 5 for i in range(1, 5))


def test_R7_index_policies_lose_one_plus_delta_against_preemption():
    """P = 0 w.p. 1-q, L w.p. q: exact enumeration of both policies' values gives (n+1)/(nq+2-q) -> 1 + Delta."""
    for n, q, L in ((3, F(1, 4), 5), (6, F(1, 10), 7), (10, F(1, 20), 3)):
        Delta = (1 - q) / q
        fixed = q * L * F(n * (n + 1), 2)
        probe = F(0)
        for K in range(n + 1):
            probe += math.comb(n, K) * q**K * (1 - q) ** (n - K) * L * F(K * (K + 1), 2)
        assert fixed / probe == index_vs_pmtn_ratio(n, q)
    assert index_vs_pmtn_ratio(10**6, F(1, 20)) > F(199, 10)  # -> 20 = 1 + Delta at q = 1/20
    # the same phenomenon with P >= 1 (exact preemptive DP on the integer grid, m = 1)
    R = MTask(F(0), F(1), (dist((1, F(4, 5)), (9, F(1, 5))),))
    ts = [R] * 3
    assert expected_cost(ts, 1, [0, 1, 2]) / opt_pmtn_dp(ts, 1) > F(6, 5)
