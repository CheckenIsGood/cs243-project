"""Machine checks for docs/stochastic_theory_ext.md. Each test is cited in the doc as [X: name].

Exact rational arithmetic throughout; fixed seeds. 'DP' optima are exact over the stated policy class."""

from __future__ import annotations

import itertools
import random
from fractions import Fraction as F

from codesign.stochastic_theory import (
    HCfg, HNode, HTask, det, dist, expected_cost, h_bound, h_constants, h_lower_bounds, h_realizations, h_sizes,
    opt_pmtn_dp, sim_hybrid, smith, task_index_order,
)
from codesign.stochastic_theory_ext import (
    DONE, PNode, PTask, RNode, RTask, chain_cp, expected_gittins_machines, expected_work, gittins,
    gittins_single_value, hyb_cp, lambda_D, lb_cp_h, lb_cp_p, machine_ptask, p_realizations, quanta,
    r_realizations, realized_path, s_process, sim_gittins_hybrid, sim_gittins_machines, single_machine_opt,
    start_state, strip_release_and_lags, thmD_new_constant, thmD_old_constant, thmG_pathwise_bound,
    thmH_constants, thmH_pathwise_bound, to_mtask, transitions, v_matrix, wsept_index, wspt_sqrt2_instance,
    z_process,
)


def _rd(rng, lo, hi, k):
    vals = rng.sample(range(lo, hi + 1), k)
    return dist(*[(v, rng.randint(1, 3)) for v in vals])


def _rt(rng, pmax=4, lagmax=3):
    n = rng.choice([1, 2, 2, 3])
    return machine_ptask(rng.randint(0, 2), rng.randint(1, 4), [_rd(rng, 1, pmax, rng.choice([1, 2])) for _ in range(n)],
                         [_rd(rng, 0, lagmax, rng.choice([1, 2])) for _ in range(n - 1)])


def _rtask_h(rng):
    n = rng.choice([1, 2, 2])
    nodes = [RNode(q=rng.randint(1, 6), law=_rd(rng, 1, 4, rng.choice([1, 2])),
                   rho=(0 if i == 0 else rng.randint(0, 6)), ctx=(0 if i == 0 else rng.choice([0, 1])))
             for i in range(n)]
    return RTask(F(rng.randint(0, 3)), F(rng.randint(1, 4)), tuple(nodes),
                 tuple(_rd(rng, 0, 3, rng.choice([1, 2])) for _ in range(n - 1)))


# ============================================================================ Gittins machinery (exactness)
def _brute_index(T, x):
    """Gittins index by brute force over ALL deterministic stopping sets on the reachable states."""
    reach, stack = [], [x]
    while stack:
        s = stack.pop()
        if s in reach:
            continue
        reach.append(s)
        stack += [nx for _, _, nx in transitions(T, s) if nx != DONE]
    others = [s for s in reach if s != x]
    best = None
    for k in range(len(others) + 1):
        for cont in itertools.combinations(others, k):
            cont = set(cont)

            def ev(s, first):
                if not first and s not in cont:
                    return F(0), F(0)
                n = d = F(0)
                for p, wk, nx in transitions(T, s):
                    d += p * wk
                    if nx == DONE:
                        n += p * T.w
                    else:
                        n2, d2 = ev(nx, False)
                        n, d = n + p * n2, d + p * d2
                return n, d

            n, d = ev(x, True)
            if best is None or n / d > best:
                best = n / d
    return best


def test_gittins_index_exact_and_quanta_strictly_decreasing():
    rng = random.Random(1)
    for _ in range(25):
        T = machine_ptask(0, rng.randint(1, 4), [_rd(rng, 1, 3, rng.choice([1, 2])) for _ in range(rng.choice([1, 2]))],
                          [det(0)] * 0 + [det(0)] * 0)
        T = PTask(T.r, T.w, T.nodes, tuple(det(0) for _ in range(len(T.nodes) - 1)))
        x = start_state(T)
        assert gittins(T, x) == _brute_index(T, x)
        for prob, Ns, _ in p_realizations([T]):
            Q = quanta(T, Ns[0])
            assert all(Q[i][0] > Q[i + 1][0] for i in range(len(Q) - 1))
            assert sum(wk for _, wk, _ in Q) == sum(Ns[0])


def test_gittins_single_machine_is_optimal_and_beats_wsept():
    """Gittins quanta order = exact single-machine optimum (two independent DPs), machine-model chain tasks."""
    rng = random.Random(2)
    for _ in range(40):
        ts = [machine_ptask(0, rng.randint(1, 4), [_rd(rng, 1, 4, rng.choice([1, 2])) for _ in range(rng.choice([1, 2]))], [])
              for _ in range(rng.choice([2, 3]))]
        ts = [PTask(T.r, T.w, T.nodes, tuple(det(0) for _ in range(len(T.nodes) - 1))) for T in ts]
        g = gittins_single_value(ts)
        assert g == single_machine_opt(ts) == opt_pmtn_dp([to_mtask(T) for T in ts], 1)
        assert g <= gittins_single_value(ts, index=wsept_index)
    d = dist((1, 1), (4, 1))
    ts = [machine_ptask(0, 1, [d], []) for _ in range(3)]
    assert gittins_single_value(ts) == F(57, 4) < gittins_single_value(ts, index=wsept_index) == 15


# ===================================================================== Theorem G (machine model, OPT_na)
def test_thmG_pathwise_bound_every_realization():
    """C_J <= r_J + CP_J + (1/m) sum_{K != J} V_KJ (V = K's work in quanta ahead of J's last quantum)."""
    rng = random.Random(3)
    checked = 0
    for _ in range(150):
        ts = [_rt(rng) for _ in range(rng.choice([2, 3]))]
        m = rng.choice([1, 2, 3])
        for prob, Ns, Ds in itertools.islice(p_realizations(ts), 6):
            C = sim_gittins_machines(ts, Ns, Ds, m)
            for c, b in zip(C, thmG_pathwise_bound(ts, Ns, Ds, m)):
                assert c <= b
                checked += 1
    assert checked > 1500


def test_thmG_pathwise_mutations_fail():
    """Using J's FIRST quantum key instead of its last, or dropping the lags from CP, breaks the bound."""
    rng = random.Random(3)
    v_first = v_lag = 0
    for _ in range(200):
        ts = [_rt(rng) for _ in range(rng.choice([2, 3]))]
        m = rng.choice([1, 2])
        n = len(ts)
        for prob, Ns, Ds in itertools.islice(p_realizations(ts), 6):
            C = sim_gittins_machines(ts, Ns, Ds, m)
            Q = [quanta(T, Ns[t]) for t, T in enumerate(ts)]
            first = [(-Q[j][0][0], j) for j in range(n)]
            bf = [T.r + chain_cp(T, Ns[t], Ds[t]) +
                  sum(sum(wk for g, wk, _ in Q[k] if (-g, k) < first[t]) for k in range(n) if k != t) / m
                  for t, T in enumerate(ts)]
            v_first += any(c > b for c, b in zip(C, bf))
            B = thmG_pathwise_bound(ts, Ns, Ds, m)
            v_lag += any(c > b - sum(Ds[t]) for t, (c, b) in enumerate(zip(C, B)))
    assert v_first > 0 and v_lag > 0


def test_thmG_two_approx_vs_exact_preemptive_optimum_m1():
    """E[ALG] <= LB_cp + Gitt_1 - sum w E[W] <= 2 OPT_na, with OPT_na the exact preemptive DP (m = 1)."""
    rng = random.Random(4)
    worst = F(0)
    for _ in range(60):
        ts = [_rt(rng, pmax=3, lagmax=2) for _ in range(2)]
        alg = expected_gittins_machines(ts, 1)
        opt = opt_pmtn_dp([to_mtask(T) for T in ts], 1)
        lb, g = lb_cp_p(ts), gittins_single_value(strip_release_and_lags(ts))
        ew = sum(T.w * expected_work(T) for T in ts)
        assert alg <= lb + g - ew and lb <= opt and g <= opt and alg <= 2 * opt
        worst = max(worst, alg / opt)
    assert F(1) < worst < F(11, 10)


def test_thmG_parallel_machines_consistency():
    """m = 2: E[ALG] <= LB_cp + (Gitt_1 - sum w E W)/2; Gitt_1/2 and LB_cp lie below the grid-DP upper bound."""
    rng = random.Random(5)
    for _ in range(30):
        ts = [_rt(rng, pmax=3, lagmax=2) for _ in range(2)]
        alg = expected_gittins_machines(ts, 2)
        dp = opt_pmtn_dp([to_mtask(T) for T in ts], 2)
        lb, g = lb_cp_p(ts), gittins_single_value(strip_release_and_lags(ts))
        ew = sum(T.w * expected_work(T) for T in ts)
        assert alg <= lb + (g - ew) / 2 and g / 2 <= dp and lb <= dp
    # the 1/m factor is exercised: four unit jobs on m = 2: ALG = 6 <= 4 + (10 - 4)/2 = 7, but 4 + (10 - 4)/4 < 6
    ts = [machine_ptask(0, 1, [det(1)], []) for _ in range(4)]
    alg, lb, g = expected_gittins_machines(ts, 2), lb_cp_p(ts), gittins_single_value(ts)
    ew = sum(T.w * expected_work(T) for T in ts)
    assert (alg, lb, g, ew) == (6, 4, 10, 4)
    assert alg <= lb + (g - ew) / 2 and alg > lb + (g - ew) / 4


def test_thmG_hypotheses_are_necessary():
    """(1) A fixed-priority index (WSEPT) violates the Thm G bound: Bernoulli-type sizes.
    (2) The fast single machine must have speed m: OPT_na(m = 2) < Gitt_1 at speed 1."""
    for n, L in ((3, 6), (4, 8)):
        d = dist((1, F(3, 4)), (L, F(1, 4)))
        ts = [machine_ptask(0, 1, [d], []) for _ in range(n)]
        bound = lb_cp_p(ts) + gittins_single_value(ts) - sum(T.w * expected_work(T) for T in ts)
        assert expected_gittins_machines(ts, 1) <= bound < expected_gittins_machines(ts, 1, index=wsept_index)
    d = dist((1, F(1, 2)), (3, F(1, 2)))
    ts = [machine_ptask(0, 1, [d], []) for _ in range(3)]
    assert opt_pmtn_dp([to_mtask(T) for T in ts], 2) == F(59, 8) < gittins_single_value(ts) == 12


# ===================================================================== Theorem H (engine, random X)
def test_thmH_pathwise_bound_hybrid_random_outputs():
    """C_J <= r_J + CP'_J + sum_{K != J} V'_KJ in the hybrid model (Z'-process quanta), random X and lags."""
    rng = random.Random(8)
    checked = 0
    for _ in range(150):
        ts = [_rtask_h(rng) for _ in range(rng.choice([2, 3]))]
        cfg = HCfg(alpha=F(1, 4), a=F(1), m=rng.choice([1, 2, 3]))
        for prob, Xs, Ds in itertools.islice(r_realizations(ts), 5):
            C = sim_gittins_hybrid(ts, Xs, Ds, cfg)
            for c, b in zip(C, thmH_pathwise_bound(ts, Xs, Ds, cfg)):
                assert c <= b
                checked += 1
    assert checked > 1500


def test_thmH_pathwise_mutation_s_process_fails():
    """Charging interference with the s-process (no recompute) instead of Z' breaks the pathwise bound."""
    rng = random.Random(9)
    viol = 0
    for _ in range(120):
        ts = [_rtask_h(rng) for _ in range(2)]
        cfg = HCfg(alpha=F(1, 4), a=F(1), m=rng.choice([1, 2]))
        S = [s_process(T, cfg) for T in ts]
        for prob, Xs, Ds in itertools.islice(r_realizations(ts), 4):
            C = sim_gittins_hybrid(ts, Xs, Ds, cfg)
            V = v_matrix(S, [quanta(s_, Xs[t]) for t, s_ in enumerate(S)])
            B = [T.r + hyb_cp(T, Xs[t], Ds[t], cfg, "alg") + V[1 - t][t] for t, T in enumerate(ts)]
            viol += any(c > b for c, b in zip(C, B))
    assert viol > 0


def test_thmH_ratio_domination_and_gittins_optimality():
    """Gitt_1(Z') = OPT_1(Z'), Gitt_1(s) = OPT_1(s) (exact DP); Gitt_1(Z') <= lambda Gitt_1(s);
    E[ALG] <= (1+beta_cp) LB_cp + Gitt_1(Z') - sum w E Z' <= (1 + beta_cp + lambda) max(LB_cp, Gitt_1(s))."""
    rng = random.Random(9)
    for _ in range(40):
        ts = [_rtask_h(rng) for _ in range(2)]
        cfg = HCfg(alpha=F(1, 4), a=F(1), m=rng.choice([1, 2]))
        alg = sum(prob * sum(T.w * c for T, c in zip(ts, sim_gittins_hybrid(ts, Xs, Ds, cfg)))
                  for prob, Xs, Ds in r_realizations(ts))
        Z = strip_release_and_lags([z_process(T, cfg) for T in ts])
        S = strip_release_and_lags([s_process(T, cfg) for T in ts])
        gz, gs = gittins_single_value(Z), gittins_single_value(S)
        assert gz == single_machine_opt(Z) and gs == single_machine_opt(S)
        bcp, lam = thmH_constants(ts, cfg)
        lb = lb_cp_h(ts, cfg)
        ez = sum(T.w * expected_work(z) for T, z in zip(ts, Z))
        assert gz <= lam * gs
        assert alg <= (1 + bcp) * lb + gz - ez <= (1 + bcp + lam) * max(lb, gs)
    # beta_cp is load-bearing: one chain task, final node with rho = 20: ALG = 15/2 = (1+0) LB_cp + 3 > LB_cp,
    # and the bound with the exact beta_cp = 4 holds
    T = RTask(F(0), F(1), (RNode(q=1, law=det(1)), RNode(q=1, law=det(1), rho=20)), (det(0),))
    cfg = HCfg(alpha=F(1, 4), a=F(1), m=1)
    alg = sum(prob * sim_gittins_hybrid([T], Xs, Ds, cfg)[0] for prob, Xs, Ds in r_realizations([T]))
    bcp, lam = thmH_constants([T], cfg)
    Z = [z_process(T, cfg)]
    lb, gz, ez = lb_cp_h([T], cfg), gittins_single_value(strip_release_and_lags(Z)), expected_work(Z[0])
    assert alg == F(15, 2) and lb == F(9, 2) and bcp == 4 and gz == ez
    assert lb + gz - ez < alg <= (1 + bcp) * lb + gz - ez


def test_thmH_fixed_priority_mutation_fails():
    """Running a FIXED priority (WSEPT on Z') instead of Gittins quanta violates the sharp Thm H bound
    (Bernoulli output lengths: X = 1 w.p. 9/10, 12 w.p. 1/10; six tasks; m = 1)."""
    law = dist((1, F(9, 10)), (12, F(1, 10)))
    ts = [RTask(F(0), F(1), (RNode(q=1, law=law),)) for _ in range(6)]
    cfg = HCfg(alpha=F(1, 100), a=F(1), m=1)
    val = lambda idx: sum(prob * sum(sim_gittins_hybrid(ts, Xs, Ds, cfg, index=idx)) for prob, Xs, Ds in r_realizations(ts))
    bcp, lam = thmH_constants(ts, cfg)
    Z = [z_process(T, cfg) for T in ts]
    sharp = (1 + bcp) * lb_cp_h(ts, cfg) + gittins_single_value(Z) - sum(expected_work(z) for z in Z)
    assert val(gittins) <= sharp < val(wsept_index)


def test_thmH_calibrated_constant():
    """Calibrated a, c (m = 51), alpha; plan q=2000, final q=1500, rho=2200; random X laws: 1 + beta_cp + lambda."""
    al, a = F("0.000114"), F("0.027767")
    cfg = HCfg(alpha=al, a=a, m=51)
    lp, lf = dist((50, 1), (200, 2), (800, 1)), dist((100, 1), (300, 2), (1200, 1))
    T = RTask(F(0), F(1), (RNode(q=2000, law=lp), RNode(q=1500, law=lf, rho=2200)), (det(1),))
    bcp, lam = thmH_constants([T], cfg)
    assert F(267, 100) < 1 + bcp + lam < F(268, 100)
    Td = RTask(F(0), F(1), (RNode(q=2000, law=det(200)), RNode(q=1500, law=det(300), rho=2200)), (det(1),))
    bcp, lam = thmH_constants([Td], cfg)
    assert F(252, 100) < 1 + bcp + lam < F(253, 100)


def test_preload_aware_constants_H_DR_K():
    """(L1) fix inherited by Thm H, Thm DR, Prop K. Unrestricted benchmark (the final prompt is a staggered fan-out
    the benchmark may prefill early): beta_cp -> beta_cp0 = alpha(q + rho - q0)/(alpha q0 + a X_min), q0 = 0.
    Calibrated: H 2.52 -> 2.54 (det X), 2.67 -> 2.74 (random X); DR 8.68 -> 8.74; Prop K 5.80 -> 5.82 (64k);
    Thm D 9.99 -> 10.01. Exact on random Thm H instances: E[ALG] <= (1+beta_cp0) LB0 + Gitt_1(Z') - sum w E Z'."""
    old, new = calibrated_constants(False), calibrated_constants(True)
    assert F(252, 100) < old["H_det"] < F(253, 100) and F(254, 100) < new["H_det"] < F(255, 100)
    assert F(267, 100) < old["H_rand"] < F(268, 100) and F(274, 100) < new["H_rand"] < F(275, 100)
    assert 8.67 < old["DR"] < 8.68 and 8.74 < new["DR"] < 8.75
    assert F(579, 100) < old["K64"] < F(580, 100) and F(582, 100) < new["K64"] < F(583, 100)
    assert F(998, 100) < old["D64"] < F(999, 100) and F(1000, 100) < new["D64"] < F(1001, 100)
    al, a = F("0.000114"), F("0.027767")
    assert new["H_rand"] - old["H_rand"] == al * 3700 / (a * 100) - al * 2200 / (al * 1500 + a * 100)
    def fam_prompt(rng):  # long fan-out final prompts: the beta_cp0 term binds
        ts = []
        for _ in range(2):
            plan = RNode(q=rng.randint(1, 3), law=_rd(rng, 1, 3, rng.choice([1, 2])))
            fin = RNode(q=rng.randint(4, 12), law=_rd(rng, 1, 3, rng.choice([1, 2])), rho=rng.randint(0, 8),
                        ctx=rng.choice([0, 1]), fanout=True)
            ts.append(RTask(F(0), F(rng.randint(1, 3)), (plan, fin), (_rd(rng, 0, 2, rng.choice([1, 2])),)))
        return ts, HCfg(alpha=F(1, 4), a=F(1), m=rng.choice([1, 2]))

    def fam_load(rng):  # five short-prompt tasks competing for the decoders: Gitt_1(Z') - sum w E Z' binds
        ts = []
        for k in range(5):
            nodes = [RNode(q=1, law=_rd(rng, 2, 5, 2 if k < 2 else 1))]
            if k == 0:
                nodes.append(RNode(q=1, law=_rd(rng, 2, 5, 1), fanout=True))
            ts.append(RTask(F(0), F(rng.randint(1, 3)), tuple(nodes),
                            tuple(det(rng.randint(0, 2)) for _ in range(len(nodes) - 1))))
        return ts, HCfg(alpha=F(1, 4), a=F(1), m=rng.choice([1, 2]))

    rng = random.Random(21)
    need_bcp = need_gitt = 0
    for fam, n in ((fam_prompt, 30), (fam_load, 20)):
        for _ in range(n):
            ts, cfg = fam(rng)
            alg = sum(prob * sum(T.w * c for T, c in zip(ts, sim_gittins_hybrid(ts, Xs, Ds, cfg)))
                      for prob, Xs, Ds in r_realizations(ts))
            Z = strip_release_and_lags([z_process(T, cfg) for T in ts])
            gz, ez = gittins_single_value(Z), sum(T.w * expected_work(z) for T, z in zip(ts, Z))
            bcp0, _ = thmH_constants(ts, cfg, preload_aware=True)
            lb0 = lb_cp_h(ts, cfg, preload_aware=True)
            assert lb0 <= lb_cp_h(ts, cfg)
            assert alg <= (1 + bcp0) * lb0 + gz - ez
            need_bcp += alg > lb0 + gz - ez  # dropping beta_cp0 would fail
            need_gitt += alg > (1 + bcp0) * lb0  # dropping the interference term would fail
    assert need_bcp > 0 and need_gitt > 0
    # an instance where the 2 + beta branch of min(1 + beta_cp0, 2 + beta) binds (Prop K, bounded memory): the
    # (L1) counterexample, whose fan-out prompt dominates its decode (beta_cp0 = 11 > 1 + beta = 8)
    T = HTask(F(0), F(1), (HNode(q=1, X=1), HNode(q=11, X=1, rho=0, pieces=((det(0), 10), (det(10), 1)))),
              ((0, 1, det(0)),))
    cfg = HCfg(alpha=F(1), a=F(1), m=1, Mp=100)
    lag = [{("a", 0): F(0), ("p", (1, 0)): F(0), ("p", (1, 1)): F(10)}]
    bcp0, b, _ = h_constants([T], cfg, preload_aware=True)
    lD = lambda_D([T], cfg)
    assert (bcp0, b) == (11, 7)
    assert thmD_new_constant([T], cfg, preload_aware=True) == 2 + b + 2 * lD < 1 + bcp0 + 2 * lD
    alg = sim_hybrid([T], lag, cfg)["C"][0]
    lb0, S = h_lower_bounds([T], cfg, preload_aware=True)
    wsig = sum(cfg.alpha * n.q + cfg.c * n.X for n in T.nodes)
    assert alg == 24 and alg <= lb0 + (1 + b) * wsig + lD * S
    assert h_cp(T, lag[0], cfg, "alg") <= h_cp(T, lag[0], cfg, "opt0") + (1 + b) * wsig


# ===================================================================== Prop K (sharper Thm D constant)
def test_propK_sharpened_thmD_constant():
    """E[ALG] <= (1+beta_cp) LB_cp + lambda_D S(sigma + A/M), lambda_D = max_K Z'_K/(sigma_K + A_K/M) <=
    max(1+beta, theta); exact on random Thm D instances; tight for one task (mutation lambda_D - eps fails)."""
    rng = random.Random(11)
    for _ in range(80):
        k = rng.randint(1, 3)
        ts = []
        for _ in range(k):
            plan = HNode(q=rng.randint(1, 9), X=rng.randint(1, 4))
            fin = HNode(q=rng.randint(1, 9), X=rng.randint(1, 3), rho=plan.l_plus, pieces=((dist((0, 1), (3, 1)), 1),))
            ts.append(HTask(F(rng.randint(0, 3)), F(rng.randint(1, 4)), (plan, fin), ((0, 1, det(0)),)))
        L = max(n.l_plus for T in ts for n in T.nodes)
        cfg = HCfg(alpha=F(1, 4), a=F(1), m=rng.choice([1, 2]), Mp=L + rng.randint(L // 2, 3 * L))
        alg = sum(prob * sum(T.w * c for T, c in zip(ts, sim_hybrid(ts, lag, cfg)["C"])) for prob, lag in h_realizations(ts))
        lbcp, scos = h_lower_bounds(ts, cfg)
        bcp, b, th = h_constants(ts, cfg)
        lD = lambda_D(ts, cfg)
        assert lD <= max(1 + b, th)
        assert alg <= (1 + bcp) * lbcp + lD * scos
        Z, S, A = h_sizes(ts, cfg)
        assert smith([T.w for T in ts], Z) <= lD * scos
        if k == 1:
            assert smith([T.w for T in ts], Z) == lD * scos  # tight: lambda_D - eps would fail
    al, a = F("0.000114"), F("0.027767")
    T = HTask(F(0), F(1), (HNode(q=2000, X=200), HNode(q=1500, X=300, rho=2200)), ((0, 1, det(1)),))
    c2 = HCfg(alpha=al, a=a, m=51, Mp=64000)
    assert F(998, 100) < thmD_old_constant([T], c2) < F(999, 100)
    assert F(579, 100) < thmD_new_constant([T], c2) < F(580, 100)


def test_D0_kick_charge_is_tight_per_event():
    """Under priority-prefix admission one 2-token arrival can kick a whole L_max reservation: the per-event charge
    l+_y + L_max of Lemma D0 cannot be reduced without changing the admission rule."""
    z = HTask(F(0), F(1), (HNode(q=1, X=9),))  # l+ = 10 = L_max
    y = HTask(F(17, 2), F(1000), (HNode(q=1, X=1),))  # l+ = 2, high priority, arrives late in z's decode
    cfg = HCfg(alpha=F(1, 100), a=F(1), m=2, Mp=11)  # y + z = 12 > 11
    st = sim_hybrid([z, y], [{}, {}], cfg)
    assert st["kicks"] == 1 and st["recompute"] >= 2 + 10 - 3  # >= l+_y + L_max - 3 tokens recomputed


# ===================================================================== Thm A lower bound (sqrt 2)
def test_thmA_lower_bound_sqrt2():
    """Remark only (superseded: Schulz-Skutella's known family gives 2, so Thm A is tight; see the base test
    test_thmA_constant_two_is_tight_schulz_skutella). P-WSPT on one machine: job A (p = w = L) and k unit jobs of
    weight 1+eps released at L-1; exact simulation = closed form; the ratio tends to sqrt(2) (never reached)."""
    for L, k in ((3, 4), (4, 5)):
        ts, alg, ub = wspt_sqrt2_instance(L, k)
        assert expected_cost(ts, 1, task_index_order(ts)) == alg
        assert opt_pmtn_dp(ts, 1) == ub
    ts, alg, ub = wspt_sqrt2_instance(1000, 1414, F(1, 10**6))
    assert alg / ub > F(1413, 1000) > F(4, 3)
    ts, alg, ub = wspt_sqrt2_instance(1000, 0)
    assert alg == ub  # mutation: without the unit jobs there is no loss


# ===================================================================== Theorem P (index policy vs clairvoyant)
from codesign.stochastic_theory import (  # noqa: E402
    cv2, h_cp, index_vs_pmtn_ratio, lb_cp, mean, opt_clair_1m, MTask, chain_task,
)
from codesign.stochastic_theory_ext import (  # noqa: E402
    CapNode, DRNode, DRTask, calibrated_constants, dr_expected_sizes, dr_order, dr_realizations, emin,
    emin_lower_bound_holds, normalize, pair_ratio_bound, smith_clairvoyant, smith_expected,
)
import z3  # noqa: E402


def _proved(claim, *hyps):
    s = z3.Solver()
    s.add(*hyps)
    s.add(z3.Not(claim))
    return s.check() == z3.unsat


def test_lemmaP1_emin_lower_bound_and_witness():
    """Independent X, Y >= 0, mean 1, CV^2 <= Delta: E min(X,Y) >= max(16/(27(1+Delta)), 1 - sqrt(Delta/2)).
    Exact on 3000 random laws; Z3 for the algebra; an exact witness shows the constant is < 0.78/(1+Delta), so
    the natural guess 1/(1+Delta) is FALSE."""
    rng = random.Random(1)
    for _ in range(3000):
        d1 = dist(*[(rng.randint(0, 30), rng.randint(1, 9)) for _ in range(rng.choice([1, 2, 3]))] + [(rng.randint(1, 30), 1)])
        d2 = dist(*[(rng.randint(0, 30), rng.randint(1, 9)) for _ in range(rng.choice([1, 2, 3]))] + [(rng.randint(1, 30), 1)])
        assert emin_lower_bound_holds(d1, d2)
    x, T, c = z3.Reals("x T c")
    assert _proved(z3.If(x > T, x - T, 0) <= x * x / (4 * T), x >= 0, T > 0)
    Tq = 3 * c / 4
    assert _proved((1 / Tq) * (1 - c / (4 * Tq)) ** 2 == 16 / (27 * c), c > 0)
    X, Y = dist((0, F(2, 3)), (3, F(1, 3))), dist((F(11, 20), F(10, 11)), (F(11, 2), F(1, 11)))
    D = max(cv2(X), cv2(Y))
    assert mean(X) == mean(Y) == 1 and D == F(81, 40)
    assert F(16, 27) < (1 + D) * emin(X, Y) == F(187, 240) < F(78, 100)  # 0.7792
    # Bernoulli: E min = q = 1/(1+Delta) exactly (so the 1/(1+Delta) scaling is right)
    q = F(1, 7)
    B = dist((0, 1 - q), (1 / q, q))
    assert emin(B, B) == q * q * (1 / q) and (1 + cv2(B)) * emin(B, B) == q * (1 / q) * q * (1 / q) * 1


def test_lemmaP1_witness_58_77_refutes_larger_constants():
    """Verifier's witness: X = {0 w.p. 19/20, 20 w.p. 1/20}, Y = {1/2 w.p. 76/77, 39 w.p. 1/77}; both mean 1, CV^2 =
    19, (1+Delta) E min(X,Y) = 58/77 = 0.7532. So the best constant c* in E min >= c*/(1+Delta) lies in
    [16/27, 58/77]; candidate constants 0.76 and 0.80 are false (computed directly, not via the checker)."""
    X = dist((0, F(19, 20)), (20, F(1, 20)))
    Y = dist((F(1, 2), F(76, 77)), (39, F(1, 77)))
    assert mean(X) == mean(Y) == 1 and cv2(X) == cv2(Y) == 19
    c = (1 + 19) * emin(X, Y)
    assert c == F(58, 77) and F(16, 27) <= c < F(76, 100) < F(80, 100)
    assert F(16, 27) / (1 + 19) <= emin(X, Y)  # the proved bound holds on the witness
    assert emin_lower_bound_holds(X, Y)  # the checker accepts it; a checker constant above 58/77 would fail here


def test_thmP_index_policy_vs_clairvoyant_m1():
    """P-WSPT-DAG with index w/E[W] (random sizes, DAGs, lags, releases): E[ALG] <= LB_cp + Smith(E W)/m,
    Smith(E W) <= pair-constant * E[Smith(W)], E[Smith(W)]/m <= OPT_clair (exact clairvoyant DP, m = 1)."""
    import sys
    sys.path.insert(0, "tests")
    from test_stochastic_theory import _rtask_m
    rng = random.Random(5)
    for _ in range(14):
        ts = [_rtask_m(rng, det_sizes=False, dag=rng.random() < 0.4, pmax=3, lagmax=3) for _ in range(2)]
        alg = expected_cost(ts, 1, task_index_order(ts))
        se, sc, pr = smith_expected(ts), smith_clairvoyant(ts), pair_ratio_bound(ts)
        oc = opt_clair_1m(ts)
        assert alg <= lb_cp(ts) + se and se <= pr * sc and sc <= oc and lb_cp(ts) <= oc
        # pair constant obeys Lemma P1: pr <= (27/16)(1 + Delta_W)
        from codesign.stochastic_theory import realizations
        laws = []
        for T in ts:
            acc = {}
            for prob, P, D in realizations([T]):
                acc[sum(P[0])] = acc.get(sum(P[0]), F(0)) + prob
            laws.append(tuple(sorted(acc.items())))
        Dw = max(cv2(l) for l in laws)
        assert pr <= F(27, 16) * (1 + Dw)


def test_thmP_parallel_machines_and_delta_dependence():
    """m = 2, 3 chain: E[ALG] <= LB_cp + Smith(E W)/m with Smith(E W) <= pair * E Smith(W) (both lower-bound
    pieces valid pathwise). Mutation: a Delta-free constant is impossible for index policies (ratio -> 1 + Delta)."""
    import sys
    sys.path.insert(0, "tests")
    from test_stochastic_theory import _rtask_m
    rng = random.Random(6)
    for _ in range(40):
        ts = [_rtask_m(rng, det_sizes=False, dag=True, pmax=3, lagmax=3) for _ in range(rng.choice([2, 3]))]
        m = rng.choice([2, 3])
        alg = expected_cost(ts, m, task_index_order(ts))
        se, sc, pr = smith_expected(ts), smith_clairvoyant(ts), pair_ratio_bound(ts)
        assert alg <= lb_cp(ts) + se / m and se <= pr * sc
    assert index_vs_pmtn_ratio(10**6, F(1, 20)) > F(199, 10) > 1 + F(27, 16)  # Delta = 19: must grow with Delta


# ===================================================================== Theorem DR (Thm D with random X)
def _dr_cases(seed, n):
    rng = random.Random(seed)
    for _ in range(n):
        ts = []
        for _ in range(rng.randint(2, 3)):
            plan = DRNode(q=rng.randint(1, 8), law=_rd(rng, 1, 4, rng.choice([1, 2])))
            fin = DRNode(q=rng.randint(1, 8), law=_rd(rng, 1, 3, rng.choice([1, 2])), rho=rng.randint(0, 8))
            ts.append(DRTask(F(rng.randint(0, 3)), F(rng.randint(1, 4)), (plan, fin), _rd(rng, 0, 3, rng.choice([1, 2]))))
        Lc = max(nd.rho + nd.q + nd.xcap for T in ts for nd in T.nodes)
        cfg = HCfg(alpha=F(1, 4), a=F(1), m=rng.choice([1, 2]), Mp=Lc + rng.randint(Lc // 2, 3 * Lc))
        yield ts, cfg


def test_thmDR_random_outputs_with_cap_reservation():
    """Random X (support <= cap, reserve l- + cap), index w/E[Z']: pathwise Lemma D1 on every realization;
    E[ALG] <= (1+beta_cp) LB_cp + Smith(E Z') ; Smith(E Z') <= lambda_E Smith(E s) ; Smith(E s) <= pair * E Smith(s)
    with E Smith(s) <= 2 OPT_clair (concurrent open shop, pathwise)."""
    chk = 0
    for ts, cfg in _dr_cases(3, 60):
        o = dr_order(ts, cfg)
        alg = F(0)
        bcp = lbcp = F(0)
        for prob, hts, lag in dr_realizations(ts):
            st = sim_hybrid(hts, lag, cfg, order=o)
            for c, b in zip(st["C"], h_bound(hts, lag, cfg, o)):
                assert c <= b
                chk += 1
            alg += prob * sum(T.w * c for T, c in zip(ts, st["C"]))
            bcp = max(bcp, h_constants(hts, cfg)[0])
            lbcp += prob * sum(T.w * (T.r + h_cp(T, lag[t], cfg, "opt")) for t, T in enumerate(hts))
        EZ, slaws, es = dr_expected_sizes(ts, cfg)
        Es = [sum(v * p for v, p in l) for l in slaws]
        w = [T.w for T in ts]
        pr = F(1)
        for j, k in itertools.combinations(range(len(ts)), 2):
            a_, b_ = ts[k].w * Es[j], ts[j].w * Es[k]
            e = sum(p * q * min(ts[k].w * x, ts[j].w * y) for x, p in slaws[j] for y, q in slaws[k])
            pr = max(pr, min(a_, b_) / e)
        lamE = max(z / e for z, e in zip(EZ, Es))
        assert alg <= (1 + bcp) * lbcp + smith(w, EZ)
        assert smith(w, EZ) <= lamE * smith(w, Es) and smith(w, Es) <= pr * es
        assert pr <= F(27, 16) * (1 + max(cv2(l) for l in slaws))
    assert chk > 2000


def test_thmDR_calibrated_constant():
    import math
    al, a = F("0.000114"), F("0.027767")
    cfg = HCfg(alpha=al, a=a, m=51, Mp=64000)
    T = DRTask(F(0), F(1), (DRNode(q=2000, law=dist((50, 1), (200, 2), (800, 1))),
                            DRNode(q=1500, law=dist((100, 1), (300, 2), (1200, 1)), rho=2200)), det(1))
    EZ, sl, es = dr_expected_sizes([T], cfg)
    Es = sum(v * p for v, p in sl[0])
    D = cv2(sl[0])
    lamE = EZ[0] / Es
    bcp = max(h_constants(h, cfg)[0] for _, h, _ in dr_realizations([T]))
    rho = min(27 / 16 * (1 + float(D)), 1 / (1 - math.sqrt(float(D) / 2)))
    total = float(1 + bcp) + 2 * float(lamE) * rho
    assert 8.6 < total < 8.7 and F(3, 10) < D < F(31, 100)
