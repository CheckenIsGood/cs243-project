"""Independent verification of docs/stochastic_theory.md (commit e17bd7f). See docs/stochastic_theory_verification.md.

Tests named test_COUNTEREXAMPLE_* demonstrate a claim that is false as stated (they pass because the
counterexample exists). All other tests check claims that were verified."""

from __future__ import annotations

import itertools
import math
import random
import time
from fractions import Fraction as Q

import z3

from codesign.stochastic_verify import (
    E, Eng, Job, LBcp, Lmax, Node, Agent, agent_reals, discrete_ski_ratio, engine_lower_bounds, exp_cost,
    hold_only_value_lb, index_vs_probe, is_chain, joint, karlin_cost, keep_dual_scan, keep_lp_vertex, law, lb2,
    lb2_dual_certificate, lemmaA1_rhs, lemmaD1_rhs, msu_check_policy, msu_node_lp, opt_clair_exp, opt_na_pmtn_1m,
    opt_np, point, preload_L1_counterexample, r6_dag_counterexample, rand_agent, rand_job, rand_law,
    random_np_policy, rank_P, run_discrete, run_P, run_priority, scv, smith_bruteforce, smith_value, thmB_rho,
    thmBp_rho, thmD_constants, ttl_cost_, ttl_opt_grid, wspt_rank, zprime,
)


def proved(claim, *hyp):
    s = z3.Solver()
    s.add(*hyp)
    s.add(z3.Not(claim))
    return s.check() == z3.unsat


# ======================================================================================= Theorem A
def test_lemmaA1_pathwise_own_simulator():
    rng = random.Random(101)
    n = 0
    for _ in range(300):
        ts = [rand_job(rng, det_sizes=rng.random() < 0.5, pmax=4, lagmax=5) for _ in range(rng.randint(1, 4))]
        m = rng.choice([1, 2, 3])
        rk = wspt_rank(ts)
        for pr, P, D in itertools.islice(joint(ts), 4):
            C, _, _ = run_priority(ts, P, D, m, rk)
            for c, b in zip(C, lemmaA1_rhs(ts, P, D, m, rk)):
                assert c <= b
                n += 1
    assert n > 2000


def test_smith_rule_is_bruteforce_optimal():
    rng = random.Random(1)
    for _ in range(200):
        k = rng.randint(1, 5)
        ws = [Q(rng.randint(1, 9)) for _ in range(k)]
        ps = [Q(rng.randint(1, 9), rng.randint(1, 3)) for _ in range(k)]
        assert smith_value(ws, ps) == smith_bruteforce(ws, ps)


def test_thmA_vs_exact_clairvoyant_and_nonanticipatory_m1():
    """E[ALG] <= LB_cp + Smith <= 2 OPT_clair <= 2 OPT_na with own exact DPs (m = 1)."""
    rng = random.Random(7)
    gaps = 0
    for _ in range(25):
        ts = [rand_job(rng, det_sizes=True, pmax=3, lagmax=3) for _ in range(2)]
        rk = wspt_rank(ts)
        alg = exp_cost(ts, 1, rk)
        oc, ona = opt_clair_exp(ts), opt_na_pmtn_1m(ts)
        l1 = LBcp(ts)
        l2 = smith_value([J.w for J in ts], [sum(E(d) for d in J.sizes) for J in ts])
        assert alg <= l1 + l2 and l1 <= oc and l2 <= oc and oc <= ona and alg <= 2 * oc
        gaps += oc < ona
    assert gaps > 0


def test_unit_grid_is_exact_for_m1_dps():
    """Integrality check behind the DPs: refining the time grid to 1/2 never changes the m = 1 optima."""
    rng = random.Random(5)
    for _ in range(8):
        ts = [rand_job(rng, det_sizes=True, pmax=2, lagmax=2, rmax=1) for _ in range(2)]
        assert opt_clair_exp(ts, 1) == opt_clair_exp(ts, 2)
    rng = random.Random(32)
    for _ in range(6):
        ts = [rand_job(rng, det_sizes=False, pmax=2, lagmax=2, rmax=1, shape=rng.choice(["single", "chain2"]))
              for _ in range(2)]
        m = rng.choice([1, 2])
        assert opt_np(ts, m, 1) == opt_np(ts, m, 2)


def test_lag_trap_exact():
    for L in (4, 10, 30):
        J = Job(Q(0), Q(1), (point(1), point(L)), ((0, 1, point(L)),))
        K = Job(Q(0), Q(1), (point(L),))  # w_K = (L+1)/(L+eps) = 1 at eps = 1
        ts = [J, K]
        rk = wspt_rank(ts)
        assert rk == [1, 0]
        assert exp_cost(ts, 1, rk) == 4 * L + 1
        assert opt_clair_exp(ts) == 3 * L + 2 == opt_na_pmtn_1m(ts)


# ======================================================================================= Theorem B / B'
def test_msu_inequality_on_random_nonanticipatory_nonpreemptive_policies():
    """The MSU load inequality (with lags, releases, DAGs, idling, adaptive rules) on arbitrary np policies."""
    rng = random.Random(3)
    n = 0
    for _ in range(200):
        ts = [rand_job(rng, det_sizes=False, pmax=4, lagmax=3) for _ in range(rng.randint(1, 3))]
        if sum(len(J.sizes) for J in ts) > 6:
            continue
        m = rng.choice([1, 2, 3])
        assert msu_check_policy(ts, m, random_np_policy, rng) >= 0
        n += 1
    assert n > 100


def test_msu_fails_for_preemptive_policies():
    """Three i.i.d. {1 w.p. 3/4, 8 w.p. 1/4} jobs, m = 1: preemptive OPT 213/16 < LP 33/2 = OPT_np."""
    R = Job(Q(0), Q(1), (law((1, Q(3, 4)), (8, Q(1, 4))),))
    ts = [R] * 3
    lp = msu_node_lp(ts, 1)[0]
    assert lp == opt_np(ts, 1) == Q(33, 2)
    assert opt_na_pmtn_1m(ts) == Q(213, 16) < lp


def test_msu_correction_needed():
    R = Job(Q(0), Q(1), (law((1, Q(9, 10)), (30, Q(1, 10))),))
    ts = [R] * 4
    opt = opt_np(ts, 2)
    assert opt == Q(2358, 125)  # 18.864
    # Delta = 1 in f(A) removes the -(m-1)(Delta-1)/(2m) correction: that 'LP' exceeds OPT_np
    assert msu_node_lp(ts, 2, Delta=Q(1))[0] > opt >= msu_node_lp(ts, 2)[0]


def test_lb2_greedy_has_matching_dual_certificate():
    rng = random.Random(6)
    for _ in range(60):
        ts = [rand_job(rng, det_sizes=False, pmax=6) for _ in range(rng.randint(1, 5))]
        m = rng.choice([1, 2, 4])
        p, d, feas = lb2_dual_certificate(ts, m)
        assert feas and p == d == lb2(ts, m)[0]


def test_thmB_vs_exact_nonpreemptive_optimum():
    rng = random.Random(21)
    for _ in range(150):
        ts = [rand_job(rng, det_sizes=False, pmax=3, lagmax=2, rmax=1,
                       shape=rng.choice(["single", "chain2", "fork", "vee"])) for _ in range(2)]
        m = rng.choice([1, 2, 3])
        Delta = max(scv(d) for J in ts for d in J.sizes)
        rk = wspt_rank(ts)
        alg = exp_cost(ts, m, rk)
        l1 = LBcp(ts)
        l2, g = lb2(ts, m)
        ew = sum(J.w * sum(E(d) for d in J.sizes) for J in ts)
        assert alg <= l1 + l2 + (Q(1, 2 * m) - g) * ew
        opt = opt_np(ts, m)
        assert l1 <= opt and l2 <= opt
        chain = all(is_chain(J) for J in ts)
        assert alg <= thmB_rho(m, Delta, chain) * opt


def test_thmB_high_variance_gamma_negative():
    """Regime where gamma < 0 (the variance correction is active): m = 2, Delta = 81/19 ... checked exactly."""
    big = law((1, Q(9, 10)), (12, Q(1, 10)))
    ts = [Job(Q(0), Q(1), (big,)), Job(Q(0), Q(2), (big,)), Job(Q(1), Q(1), (point(1), big), ((0, 1, point(1)),))]
    m = 2
    Delta = max(scv(d) for J in ts for d in J.sizes)
    l2, g = lb2(ts, m)
    assert g < 0
    opt = opt_np(ts, m)
    assert LBcp(ts) <= opt and l2 <= opt
    alg = exp_cost(ts, m, wspt_rank(ts))
    assert alg <= thmB_rho(m, Delta, True) * opt


def test_thmBprime_lp_policy():
    rng = random.Random(22)
    for _ in range(30):
        ts = [rand_job(rng, det_sizes=False, pmax=3, lagmax=2, rmax=1, shape=rng.choice(["single", "chain2"]))
              for _ in range(2)]
        m = rng.choice([1, 2])
        Delta = max(scv(d) for J in ts for d in J.sizes)
        lpv, CJ = msu_node_lp(ts, m)
        assert lpv <= opt_np(ts, m)
        rk = sorted(range(len(ts)), key=lambda t: (CJ[t], t))
        EC = [Q(0)] * len(ts)
        for pr, P, D in joint(ts):
            C, _, _ = run_priority(ts, P, D, m, rk)
            for j in range(len(ts)):
                EC[j] += pr * C[j]
        for j in range(len(ts)):  # per-task form of the proof
            assert EC[j] <= thmBp_rho(m, Delta) * CJ[j]


def test_thmB_algebra_z3():
    m, D, C, S, Qs = z3.Reals("m D C S Q")
    b = z3.RealVal(1) / 2 - (m - 1) * D / (2 * m)
    gam = z3.If(b < 0, b, 0)
    hyp = [m >= 1, D >= 0]
    assert proved(1 / (2 * m) - gam == z3.If((m - 1) * (D - 1) > 1, (m - 1) * (D - 1), 1) / (2 * m), *hyp)
    # SU2005 Lemma 3.3 form from C S >= S^2/(2m) + b Q, 0 <= Q <= C S
    base = hyp + [C > 0, S > 0, Qs >= 0, Qs <= C * S, C * S >= S * S / (2 * m) + b * Qs]
    assert proved(S / m <= (1 + z3.If((m - 1) * D / m > 1, (m - 1) * D / m, 1)) * C, *base)
    # the MSU f(A) in the doc equals (1/2m)(sum)^2 + b * sum of squares
    s, s2 = z3.Reals("s s2")
    f = (s * s + s2) / (2 * m) - (m - 1) * (D - 1) / (2 * m) * s2
    assert proved(f == s * s / (2 * m) + b * s2, *hyp)
    # chain/DAG constants: LB_cp + LB2 + k*sum wEW with sum wEW <= OPT (chains) or <= m OPT (DAGs)
    k = (1 / (2 * m) - gam)
    o = z3.Real("o")
    assert proved(o + o + k * m * o == (2 + z3.If((m - 1) * (D - 1) > 1, (m - 1) * (D - 1), 1) / 2) * o, *hyp)


# ======================================================================================= Theorem D
def test_lemmaD0_D1_own_hybrid_simulator():
    rng = random.Random(41)
    checks = kick_events = hit_runs = 0
    for _ in range(400):
        ts = [rand_agent(rng, dag=True) for _ in range(rng.randint(2, 4))]
        L = Lmax(ts)
        Mp = rng.choice([None, L + rng.randint(1, L // 2 + 1), L + rng.randint(L // 2, 3 * L)])
        m = rng.choice([1, 2, 3])
        cfg = Eng(alpha=Q(1, rng.choice([1, 2, 4, 8])), a=Q(1), m=m, Mp=Mp, hold=rng.random() < 0.5,
                  preload=rng.random() < 0.5)
        rk = rank_P(ts, Eng(cfg.alpha, cfg.a, m, Mp))
        for pr, lag in itertools.islice(agent_reals(ts), 2):
            r = run_P(ts, lag, cfg, rk)  # memory invariants asserted inside
            for c, b in zip(r["C"], lemmaD1_rhs(ts, lag, cfg, rk)):
                assert c <= b
                checks += 1
            for (_, arr, kicked, n_arr) in r["kick_events"]:
                assert n_arr > 0  # kicks only at availability events
                assert kicked < arr + L  # Lemma D0 (even with several simultaneous arrivals)
                kick_events += 1
            hit_runs += r["held_hits"] > 0
    assert checks > 1500 and kick_events > 100 and hit_runs > 100


def test_lemmaD1_mutations_detected_by_own_checker():
    """Non-vacuity of our checker: without kicks the bound fails; without kappa the bound fails."""
    low, hi = Agent(Q(0), Q(1), (Node(q=1, X=40),)), Agent(Q(1, 2), Q(100), (Node(q=1, X=1),))
    cfg = Eng(alpha=Q(1, 10), a=Q(1), m=4, Mp=42)
    lag = [{}, {}]
    rk = rank_P([low, hi], cfg)
    b = lemmaD1_rhs([low, hi], lag, cfg, rk)
    assert run_P([low, hi], lag, cfg, rk)["C"][1] <= b[1] < run_P([low, hi], lag, cfg, rk, kicks=False)["C"][1]
    qz, alpha, a = 200, Q(1, 10), Q(1, 100)
    J = Agent(Q(0), Q(1), (Node(q=qz, X=1),))
    ys, t = [], Q(0)
    for _ in range(5):
        t += alpha * qz - Q(1, 10)
        ys.append(Agent(t, Q(1000), (Node(q=1, X=1),)))
        t += alpha + a
    ts = [J] + ys
    cfg = Eng(alpha=alpha, a=a, m=2, Mp=qz + 2)
    lag = [{} for _ in ts]
    rk = rank_P(ts, cfg)
    r = run_P(ts, lag, cfg, rk)
    L = Lmax(ts)
    Z = zprime(ts, cfg)
    pos = {k: i for i, k in enumerate(rk)}
    nokap = [z - sum(alpha * (n.lp + L) for n in T.nodes) for z, T in zip(Z, ts)]
    bound_nokap = alpha * qz + a + sum(nokap[k] for k in range(len(ts)) if pos[k] <= pos[0])
    assert bound_nokap < r["C"][0] <= lemmaD1_rhs(ts, lag, cfg, rk)[0]


def test_thmD_constant_algebra_z3():
    al, a, c, q, rho, X, bcp, M, lp, lm, th, bt, kap = z3.Reals("al a c q rho X bcp M lp lm th bt kap")
    hyp = [al > 0, a > 0, c > 0, q >= 0, rho >= 0, X >= 1, al * q + a * X > 0, bcp * (al * q + a * X) >= al * rho]
    assert proved(al * (q + rho) + a * X <= (1 + bcp) * (al * q + (a + c) * X), *hyp)
    sig = al * q + c * X
    hyp2 = [al > 0, a > 0, c > 0, q >= 0, rho >= 0, X >= 1, M > 0, lm > 0, lp >= lm, kap >= 0, sig > 0,
            bt * sig >= al * rho + kap, th * lm >= lp]
    Zp = al * (q + rho) + kap + c * X + a * X * lp / M
    A = lm * (a + c) * X
    mx = z3.If(1 + bt >= th, 1 + bt, th)
    assert proved(Zp <= mx * (sig + A / M), *hyp2)
    # Lemma D0 bookkeeping: S_old <= M', y admitted: S_new + l_next > M', l_next <= L  => kicked < l_y + L
    Sold, Snew, ly, ln, Mp, L = z3.Reals("Sold Snew ly ln Mp L")
    assert proved(Sold - (Snew - ly) < ly + L, Sold <= Mp, Snew + ln > Mp, ln <= L, ly >= 0)
    # y not admitted (head-of-line stop at y): kicked = S_old - S_above, S_above + l_y > M'
    assert proved(Sold - Snew < ly + L, Sold <= Mp, Snew + ly > Mp, L >= 0)


def test_thmD_calibrated_constants_table():
    a, c, al = Q("0.027767"), Q("0.000543"), Q("1.14e-4")
    plan, final = (0, 2000, 200), (2200, 1500, 300)  # (rho, q, X)
    bcp = max(al * r / (al * q + a * X) for r, q, X in (plan, final))
    b0 = max(al * r / (al * q + c * X) for r, q, X in (plan, final))
    assert round(float(bcp), 2) == 0.03 and round(float(b0), 2) == 0.75
    assert round(float(2 + bcp + b0), 2) == 2.78
    L = max(r + q + X for r, q, X in (plan, final))
    b1 = max((al * r + al * (r + q + X + L)) / (al * q + c * X) for r, q, X in (plan, final))
    th = max(Q(r + q + X, r + q) for r, q, X in (plan, final))
    assert round(float(b1), 2) == 3.48 and round(float(th), 2) == 1.1
    assert round(float(1 + bcp + 2 * max(1 + b1, th)), 2) == 9.99


def test_COUNTEREXAMPLE_L1_invalid_when_benchmark_may_preload_pieces():
    """(L1) assumes alpha*q_y is spent after y becomes available. A speed-1 non-anticipatory engine schedule that
    prefills an already-returned fan-out piece during the remaining tool wait (what R6 lets ALG do) costs 16, but
    the doc's LB_cp is 26 > 16. The preload-aware bound (node length (a+c)X for nodes with pieces) is 15 <= 16."""
    sched, lb, lb_fix, trace, T, cfg = preload_L1_counterexample()
    assert sched == 16 and lb == 26 and lb > sched
    assert lb_fix <= sched


def test_thmD_corrected_bound_with_preload_aware_L1():
    """Corrected Thm D (benchmark may preload): E[ALG] <= LB0 + (1+beta) sum w sigma + max(1+beta,theta) S, and
    each term is <= OPT, OPT, 2 OPT respectively; check the first inequality exactly on random instances."""
    rng = random.Random(9)
    for _ in range(120):
        ts = [rand_agent(rng) for _ in range(rng.randint(2, 3))]
        L = Lmax(ts)
        Mp = rng.choice([None, L + rng.randint(L // 2, 3 * L)])
        cfg = Eng(alpha=Q(1, 4), a=Q(1), m=rng.choice([1, 2]), Mp=Mp)
        alg = sum(pr * sum(T.w * c for T, c in zip(ts, run_P(ts, lag, cfg)["C"])) for pr, lag in agent_reals(ts))
        lb0 = Q(0)
        for pr, lag in agent_reals(ts):
            for t, T in enumerate(ts):
                from codesign.stochastic_verify import cp_path
                lb0 += pr * T.w * (T.r + cp_path(T, lag[t], lambda nd: (cfg.a + cfg.c) * nd.X))
        _, S = engine_lower_bounds(ts, cfg)
        bcp, b, th = thmD_constants(ts, cfg)
        wsig = sum(T.w * sum(cfg.alpha * nd.q + cfg.c * nd.X for nd in T.nodes) for T in ts)
        assert alg <= lb0 + (1 + b) * wsig + max(1 + b, th) * S
        lb_cp, _ = engine_lower_bounds(ts, cfg)
        assert alg <= (1 + bcp) * lb_cp + max(1 + b, th) * S  # the doc's first inequality (for comparison)


def test_COUNTEREXAMPLE_R6_exact_saving_fails_for_DAG_tasks():
    saving, alpha_hits, _, _ = r6_dag_counterexample()
    assert saving == Q(1, 2) and alpha_hits == 1 and saving < alpha_hits


def test_R6_exact_saving_holds_for_chains():
    rng = random.Random(10)
    for _ in range(60):
        ts = [rand_agent(rng)]
        c0, c1 = Eng(alpha=Q(1, 4), a=Q(1), m=1), Eng(alpha=Q(1, 4), a=Q(1), m=1, hold=True, preload=True)
        for pr, lag in itertools.islice(agent_reals(ts), 3):
            s0, s1 = run_P(ts, lag, c0), run_P(ts, lag, c1)
            assert s0["C"][0] - s1["C"][0] == c0.alpha * s1["held_hits"]


def test_R5c_per_task_dominance_fails_own_simulator():
    t0 = Agent(Q(0), Q(4), (Node(7, 3), Node(3, 3, rho=10, pieces=((point(2), 1),))), ((0, 1, point(0)),))
    t1 = Agent(Q(4), Q(4), (Node(5, 2), Node(9, 2, rho=7, pieces=((point(2), 8), (law((1, 1), (2, 1)), 1)))),
               ((0, 1, point(0)),))
    t2 = Agent(Q(0), Q(2), (Node(5, 3), Node(10, 2, rho=8, pieces=((law((1, 1), (5, 1)), 1),
                                                                 (law((2, 1), (4, 1)), 4), (point(5), 4)))),
               ((0, 1, point(0)),))
    t3 = Agent(Q(3), Q(4), (Node(8, 1), Node(5, 3, rho=9, pieces=((law((0, 1), (5, 1)), 3),))), ((0, 1, point(0)),))
    ts = [t0, t1, t2, t3]
    lag = [{("arc", 0): Q(0), ("pc", 1, 0): Q(2)},
           {("arc", 0): Q(0), ("pc", 1, 0): Q(2), ("pc", 1, 1): Q(1)},
           {("arc", 0): Q(0), ("pc", 1, 0): Q(1), ("pc", 1, 1): Q(2), ("pc", 1, 2): Q(5)},
           {("arc", 0): Q(0), ("pc", 1, 0): Q(0)}]
    base = Eng(alpha=Q(1, 4), a=Q(1), m=1)
    rk = rank_P(ts, base)
    c0 = run_P(ts, lag, base, rk)["C"]
    c1 = run_P(ts, lag, Eng(alpha=Q(1, 4), a=Q(1), m=1, hold=True, preload=True), rk)["C"]
    assert (c0[2], c1[2]) == (24, 27)
    assert sum(T.w * c for T, c in zip(ts, c1)) < sum(T.w * c for T, c in zip(ts, c0))
    for c, b in zip(c1, lemmaD1_rhs(ts, lag, base, rk)):
        assert c <= b


# ======================================================================================= R0
def test_R0a_z3_own():
    a, c, al, B, u, s = z3.Reals("a c al B u s")
    hyp = [a > 0, c > 0, al > 0, B >= 0, a - c * B >= 0]
    assert proved(z3.And(2 / (a + c * B) >= 1 / a, 3 * (z3.RealVal(1) / 3) / al >= 1 / al), *hyp)
    # tightness: at B = m = a/c, prefill >= 1/alpha and decode >= 1/a need s u >= 1 and s (1-u) >= 2
    assert proved(s >= 3, a > 0, c > 0, al > 0, B * c == a, u >= 0, u <= 1, s > 0,
                  s * u / al >= 1 / al, s * (1 - u) / (a + c * B) >= 1 / a)
    # time-sharing realizes exactly (1/alpha, each of B decoders at 1/a) from speed-3 configurations
    lam = (a + c * B) / (2 * a)
    assert proved(z3.And(lam >= 0, lam <= 1, lam * 2 / (a + c * B) == 1 / a), *hyp)
    # converse: speed-1 rates lie inside the hybrid rates
    hyp3 = [a > 0, c > 0, al > 0, B >= 1, u >= 0, u <= 1]
    assert proved(z3.And((1 - u) / (a + c * B) <= 1 / a, B * (1 - u) / (a + c * B) <= 1 / c, u / al <= 1 / al), *hyp3)


def test_R0b_discrete_own_simulator():
    rng = random.Random(51)
    n = 0
    for _ in range(250):
        ts = [rand_agent(rng, dag=True) for _ in range(rng.randint(2, 4))]
        L = Lmax(ts)
        Mp = rng.choice([None, L + rng.randint(1, L // 2 + 1), L + rng.randint(L // 2, 3 * L)])
        cfg = Eng(alpha=Q(1, rng.choice([1, 2, 4])), a=Q(1), m=rng.choice([1, 2, 3]), Mp=Mp)
        rk = rank_P(ts, cfg)
        pos = {t: i for i, t in enumerate(rk)}
        for pr, lag in itertools.islice(agent_reals(ts), 2):
            C, nc = run_discrete(ts, lag, cfg, rank=rk)
            hb = lemmaD1_rhs(ts, lag, cfg, rk)
            for t, (c, h) in enumerate(zip(C, hb)):
                N = sum(nc[(k, v)] for k in range(len(ts)) if pos[k] <= pos[t] for v in range(len(ts[k].nodes)))
                assert c <= h + cfg.a * (len(ts[t].nodes) + N)
                n += 1
    assert n > 500


def test_R0c_numbers_and_consistent_instance():
    """The doc's 1286/287 is m = 1 vs m = 8 with unbounded memory (not an instance of the m_eff rule). A consistent
    instance: add one low-weight long-context task so L_max = 50, M' = 99: m_eff = min(8, floor(99/50)) = 1 while
    the eight 11-token decodes all fit for policy P."""
    ts = [Agent(Q(0), Q(1), (Node(q=1, X=10),)) for _ in range(8)]
    lag = [{} for _ in ts]
    cap = sum(run_P(ts, lag, Eng(Q(1, 100), Q(1), 1))["C"])
    full = sum(run_P(ts, lag, Eng(Q(1, 100), Q(1), 8))["C"])
    assert cap / full == Q(1286, 287)
    big = Agent(Q(0), Q(1, 1000), (Node(q=40, X=10),))
    ts2 = ts + [big]
    lag2 = [{} for _ in ts2]
    Mp = 99
    assert min(8, Mp // Lmax(ts2)) == 1
    rP = run_P(ts2, lag2, Eng(Q(1, 100), Q(1), 8, Mp=Mp))["C"]
    rcap = run_P(ts2, lag2, Eng(Q(1, 100), Q(1), 1, Mp=Mp))["C"]
    assert sum(rcap[:8]) / sum(rP[:8]) > 4


# ======================================================================================= R5
def test_R5a_ttl_candidates_optimal_and_ratio():
    rng = random.Random(13)
    worst = Q(0)
    for _ in range(300):
        D = rand_law(rng, 0, 12, rng.randint(1, 4))
        h, R = Q(rng.randint(1, 5)), Q(rng.randint(1, 30))
        cand = ttl_opt_grid(D, h, R, [Q(0)] + [v for v, _ in D])
        assert cand == ttl_opt_grid(D, h, R, [Q(g, 8) for g in range(0, 8 * 14)])
        clair = sum(p * min(h * v, R) for v, p in D)
        if clair:
            worst = max(worst, cand / clair)
    assert worst <= Q(158198, 100000)
    # Karlin density gives exactly e/(e-1) * min(d, B) for every d (closed form, floats)
    for d in (0.01, 0.3, 0.999, 1.0, 1.5, 7.0):
        assert abs(karlin_cost(d, 1.0) - math.e / (math.e - 1) * min(d, 1.0)) < 1e-12
    # the doc's Yao-LP numbers are the discrete ski-rental values 1/(1-(1-1/n)^n)
    for n, v in ((2, 1.333), (4, 1.463), (8, 1.523), (16, 1.553)):
        assert abs(discrete_ski_ratio(n) - v) < 6e-4


def test_R5a_exponential_rule_z3():
    h, R, lam, t = z3.Reals("h R lam t")
    # cost(tau) = h/lam + (R - h/lam) e^{-lam tau}: monotone with sign of (R - h E[D]); check the sign identity
    assert proved((R - h / lam > 0) == (h * (1 / lam) < R), h > 0, R > 0, lam > 0)


def test_R5b_vertex_enumeration_equals_dual():
    rng = random.Random(14)
    for _ in range(25):
        k = rng.randint(1, 3)
        cls = [(Q(rng.randint(1, 3)), Q(rng.randint(1, 5)), Q(rng.randint(1, 8)), rand_law(rng, 0, 6, rng.randint(1, 3)))
               for _ in range(k)]
        u, rs = Q(rng.randint(1, 8), 10), Q(rng.randint(1, 30), 10)
        ncp, nld = 1 / (1 - u), rs / (1 - u) ** 2
        Mh = Q(rng.randint(0, 30))
        prim = keep_lp_vertex(cls, ncp, nld, Mh)
        # breakpoints of the piecewise-linear dual
        lines = []
        for (lam, P, L, D) in cls:
            for tau in sorted({Q(0)} | {v for v, _ in D}):
                sig = sum(p for v, p in D if v <= tau)
                lines.append((lam * P * (nld * (1 - sig) - ncp * sig), lam * L * sum(p * min(v, tau) for v, p in D)))
        etas = {Q(0)}
        for (b1, s1), (b2, s2) in itertools.combinations(lines + [(Q(0), Q(0))], 2):
            if s1 != s2 and (b2 - b1) / (s1 - s2) > 0:
                etas.add((b2 - b1) / (s1 - s2))
        assert keep_dual_scan(cls, ncp, nld, Mh, etas) == prim


def test_R5b_sigma_star_and_BEprime_z3():
    u, rs, sg, P, X, lam = z3.Reals("u rs sg P X lam")
    ncp, nld = 1 / (1 - u), rs / (1 - u) ** 2
    hyp = [u >= 0, u < 1, rs > 0, sg >= 0, sg <= 1, P > 0]
    assert proved(nld / (nld + ncp) == rs / (1 - u + rs), *hyp)
    # issue rule at eta = 0: reduced cost nu_ld(1-sigma) - nu_cp sigma < 0 iff sigma > sigma*
    assert proved((nld * (1 - sg) - ncp * sg < 0) == (sg > rs / (1 - u + rs)), *hyp)
    den = 1 - u - lam * (1 - sg) * P
    delta = X / (1 - u) - (X - sg * P) / den
    assert proved(delta == P * (sg * (1 - u) - rs * (1 - sg)) / ((1 - u) * den),
                  *(hyp + [X > 0, lam > 0, rs == lam * X, den > 0]))
    # first-order term: delta/P at lam*P -> 0 equals sigma nu_cp - (1 - sigma) nu_ld
    assert proved((sg * (1 - u) - rs * (1 - sg)) / ((1 - u) * (1 - u)) == sg * ncp - (1 - sg) * nld, *hyp)


# ======================================================================================= R7
def test_R7_index_vs_probe_exact():
    for n, q, L in ((3, Q(1, 4), 5), (6, Q(1, 10), 7), (12, Q(1, 20), 3)):
        f, p = index_vs_probe(n, q, L)
        assert f / p == Q(n + 1) / (n * q + 2 - q)
    # ratio -> 1/q = 1 + Delta
    q = Q(1, 20)
    assert Q(10**6 + 1) / (10**6 * q + 2 - q) > Q(199, 10)
    d = law((0, 1 - q), (3, q))
    assert 1 + scv(d) == 1 / q


def test_R7_hold_only_numbers():
    assert hold_only_value_lb(12, 2, 1) == 42 and Q(42, 12) == Q(7, 2)


# =========================================================================== Re-verification of commit c46b893
from codesign.stochastic_verify import (  # noqa: E402
    beta_cp0, cp_path, d0_tight_family, lb0_q0, r6_isolated, release_order_np, schulz_skutella,
    thmD_unrestricted_constant,
)


def test_reverify_thmA_tight_schulz_skutella_own_code():
    """The constant 2 of Thm A is tight: own simulator + own exact one-machine DP."""
    ts = schulz_skutella(3, 6)
    alg = exp_cost(ts, 1, wspt_rank(ts))
    assert alg == 228 and release_order_np(ts) == 160 == opt_clair_exp(ts)
    prev = Q(0)
    for n in (5, 20, 150):
        ts = schulz_skutella(n)
        r = exp_cost(ts, 1, wspt_rank(ts)) / release_order_np(ts)
        assert prev < r < 2
        prev = r
    assert prev > Q(198, 100)
    # structure behind the family: the release-order schedule never idles (each job ends after the next release),
    # and P-WSPT preempts each job j < n with exactly 1/n left at r_{j+1}
    n = 7
    ts = schulz_skutella(n)
    t = Q(0)
    for j, J in enumerate(ts[:-1]):
        t = max(t, J.r) + J.sizes[0][0][0]
        assert t >= ts[j + 1].r
        assert J.sizes[0][0][0] - (ts[j + 1].r - J.r) == Q(1, n)


def test_reverify_thmA_smith_sharpness_numbers():
    ts = [Job(Q(0), Q(1), (point(1),)) for _ in range(20)]
    W = [Q(1)] * 20
    assert exp_cost(ts, 1, wspt_rank(ts)) == 210 == smith_value([J.w for J in ts], W)
    assert exp_cost(ts, 2, wspt_rank(ts)) == 110 and smith_value([J.w for J in ts], W) / 2 == 105


def test_reverify_lb2_sharpness_numbers():
    R = Job(Q(0), Q(1), (law((1, Q(9, 10)), (40, Q(1, 10))),))
    ts = [R] * 8
    lst = exp_cost(ts, 2, list(range(8)), preemptive=False)  # a non-preemptive policy: >= OPT_np
    l2, g = lb2(ts, 2)
    l2_gamma0, g0 = lb2(ts, 2, Delta=Q(0))
    assert g < 0 and g0 == 0 and l2 <= lst < l2_gamma0 == Q(392, 5)
    assert round(float(lst), 2) == 69.97
    ts = [Job(Q(0), Q(1), (point(1),)) for _ in range(11)]
    assert lb2(ts, 1)[0] == Q(121, 2) and opt_np(ts, 1) == 66 < Q(11, 10) * Q(121, 2)


def test_reverify_R6_DAG_saving_both_directions_and_negative():
    """Author's c46b893 instances, reproduced with OUR simulator and hold rule: saving > alpha*hits, and < 0."""
    T = Agent(Q(0), Q(1), (Node(1, 1), Node(2, 1, rho=1), Node(3, 1, rho=2)), ((0, 1, point(1)), (0, 2, point(0))))
    lag = [{("arc", 0): Q(1), ("arc", 1): Q(0)}]
    for pre in (False, True):
        c0, c1, ah = r6_isolated(T, lag, Q(1, 4), 1, pre)
        assert (c0, c1, ah) == (5, 4, Q(3, 4)) and c0 - c1 > ah
    T = Agent(Q(0), Q(1), (Node(3, 1), Node(2, 1), Node(1, 2, rho=1), Node(3, 3)),
              ((0, 1, point(1)), (0, 2, point(1)), (0, 3, point(0))))
    lag = [{("arc", 0): Q(1), ("arc", 1): Q(1), ("arc", 2): Q(0)}]
    for pre in (False, True):
        c0, c1, ah = r6_isolated(T, lag, Q(1, 4), 2, pre)
        assert (c0, c1, ah) == (6, Q(25, 4), Q(1, 4)) and c0 - c1 < 0


def test_reverify_lemmaD0_tight_family_own_simulator():
    for L, s in ((10, 3), (50, 20), (200, 50)):
        ev, rec, ly, Lm, ok = d0_tight_family(L, s)
        assert ok and Lm == L and len(ev) == 1
        _, arr, kicked, n_arr = ev[0]
        assert n_arr == 1 and arr == ly and kicked == ly + L - 1 < ly + L
        assert rec > ly + L - 3


def test_reverify_thmD_unrestricted_both_branches():
    """E[ALG] <= min{(1+beta_cp0) LB0, LB0 + (1+beta) sum w sigma} + max(1+beta, theta) S, plus the two pathwise
    step-1 forms, on random instances (chains and DAGs) with our simulator; any keep/preload configuration."""
    rng = random.Random(77)
    nchk = 0
    for _ in range(150):
        ts = [rand_agent(rng, dag=True) for _ in range(rng.randint(1, 3))]
        L = Lmax(ts)
        Mp = rng.choice([None, L + rng.randint(1, L // 2 + 1), L + rng.randint(L // 2, 3 * L)])
        cfg = Eng(alpha=Q(1, rng.choice([1, 4])), a=Q(1), m=rng.choice([1, 2]), Mp=Mp,
                  hold=rng.random() < 0.5, preload=rng.random() < 0.5)
        base = Eng(cfg.alpha, cfg.a, cfg.m, Mp)
        rk = rank_P(ts, base)
        alg = sum(pr * sum(T.w * c for T, c in zip(ts, run_P(ts, lag, cfg, rk)["C"])) for pr, lag in agent_reals(ts))
        lb0 = lb0_q0(ts, cfg)
        _, S = engine_lower_bounds(ts, cfg)
        _, b, th = thmD_constants(ts, cfg)
        bc0 = beta_cp0(ts, cfg)
        wsig = sum(T.w * sum(cfg.alpha * nd.q + cfg.c * nd.X for nd in T.nodes) for T in ts)
        second = (1 + b) * S if Mp is None else max(1 + b, th) * S
        assert alg <= min((1 + bc0) * lb0, lb0 + (1 + b) * wsig) + second
        assert lb0_q0(ts, cfg, zero=True) <= lb0 <= engine_lower_bounds(ts, cfg)[0]
        for pr, lag in itertools.islice(agent_reals(ts), 2):
            for t, T in enumerate(ts):
                cpa = cp_path(T, lag[t], lambda nd: cfg.alpha * (nd.q + nd.rho) + cfg.a * nd.X)
                cp0 = cp_path(T, lag[t], lambda nd: cfg.alpha * nd.q0 + (cfg.a + cfg.c) * nd.X)
                cpz = cp_path(T, lag[t], lambda nd: (cfg.a + cfg.c) * nd.X)
                sig = sum(cfg.alpha * nd.q + cfg.c * nd.X for nd in T.nodes)
                assert cpa <= (1 + bc0) * cp0 and cpa <= cpz + (1 + b) * sig
                nchk += 1
    assert nchk > 200


def test_reverify_thmD_unrestricted_algebra_z3():
    al, a, c, q, q0, rho, X, b0, bt, kap = z3.Reals("al a c q q0 rho X b0 bt kap")
    hyp = [al > 0, a > 0, c > 0, q0 >= 0, q >= q0, rho >= 0, X >= 1, b0 * (al * q0 + a * X) >= al * (q + rho - q0)]
    assert proved(al * (q + rho) + a * X <= (1 + b0) * (al * q0 + (a + c) * X), *hyp)
    hyp2 = [al > 0, a > 0, c > 0, q >= 0, rho >= 0, X >= 1, kap >= 0, bt * (al * q + c * X) >= al * rho + kap]
    assert proved(al * (q + rho) + a * X <= (a + c) * X + (1 + bt) * (al * q + c * X), *hyp2)
    # final constants: LB0 <= OPT, sum w sigma <= OPT, S <= 2 OPT
    o, lb0, ws, S, mx = z3.Reals("o lb0 ws S mx")
    h3 = [lb0 <= o, ws <= o, S <= 2 * o, o > 0, b0 >= 0, bt >= 0, lb0 >= 0, ws >= 0, S >= 0, mx >= 1]
    lhs = z3.If((1 + b0) * lb0 <= lb0 + (1 + bt) * ws, (1 + b0) * lb0, lb0 + (1 + bt) * ws) + mx * S
    assert proved(lhs <= (z3.If(1 + b0 <= 2 + bt, 1 + b0, 2 + bt) + 2 * mx) * o, *h3)


def test_reverify_calibrated_unrestricted_table():
    al, a = Q("0.000114"), Q("0.027767")
    TK = Agent(Q(0), Q(1), (Node(q=2000, X=200), Node(q=1500, X=300, rho=2200, pieces=((point(1), 750), (point(1), 750)))),
               ((0, 1, point(1)),))
    cinf, c64 = Eng(al, a, 51), Eng(al, a, 51, Mp=64000)
    assert round(float(beta_cp0([TK], cinf)), 4) == 0.0506
    assert round(float(thmD_unrestricted_constant([TK], cinf)), 2) == 2.80
    assert round(float(thmD_unrestricted_constant([TK], c64)), 2) == 10.01
    bcp, b, th = thmD_constants([TK], c64)
    assert round(float(1 + bcp + 2 * max(1 + b, th)), 2) == 9.99


def test_reverify_L1_counterexample_numbers_in_new_doc():
    sched, lb, lb_fix, _, T, cfg = preload_L1_counterexample()
    assert (sched, lb, lb_fix) == (16, 26, 15)
    assert beta_cp0([T], cfg) == 11 and thmD_constants([T], cfg)[0] == 0
    lag = [{("arc", 0): Q(0), ("pc", 1, 0): Q(0), ("pc", 1, 1): Q(10)}]
    assert run_P([T], lag, cfg)["C"][0] == 24


def test_CROSS_CHECK_author_new_code_matches_ours():
    """CROSS-CHECK (imports the author's module): q0, h_lower_bounds(preload_aware), h_constants(preload_aware),
    thmD_constant and sim_hybrid's kicked_res agree with our independent implementations."""
    import codesign.stochastic_theory as ST

    def conv(ts):
        return [ST.HTask(T.r, T.w, tuple(ST.HNode(q=nd.q, X=nd.X, rho=nd.rho,
                                                  pieces=tuple((ST.dist(*d), tk) for d, tk in nd.pieces))
                                         for nd in T.nodes), tuple((u, v, ST.dist(*d)) for u, v, d in T.arcs))
                for T in ts]

    def convlag(lag):
        return [{(("a", k[1]) if k[0] == "arc" else ("p", (k[1], k[2]))): v for k, v in l.items()} for l in lag]

    rng = random.Random(88)
    for _ in range(80):
        ts = [rand_agent(rng, dag=True) for _ in range(rng.randint(1, 3))]
        L = Lmax(ts)
        Mp = rng.choice([None, L + rng.randint(1, L // 2 + 1), L + rng.randint(L // 2, 3 * L)])
        cfg = Eng(alpha=Q(1, 4), a=Q(1), m=rng.choice([1, 2, 3]), Mp=Mp)
        A = conv(ts)
        acfg = ST.HCfg(alpha=cfg.alpha, a=cfg.a, m=cfg.m, Mp=Mp)
        assert [n.q0 for T in A for n in T.nodes] == [n.q0 for T in ts for n in T.nodes]
        assert ST.h_lower_bounds(A, acfg, preload_aware=True)[0] == lb0_q0(ts, cfg)
        assert ST.h_lower_bounds(A, acfg)[0] == engine_lower_bounds(ts, cfg)[0]
        assert ST.h_constants(A, acfg, True)[0] == beta_cp0(ts, cfg)
        assert ST.thmD_constant(A, acfg, True) == thmD_unrestricted_constant(ts, cfg)
        rk = rank_P(ts, cfg)
        for pr, lag in itertools.islice(agent_reals(ts), 2):
            mine = run_P(ts, lag, cfg, rk)
            theirs = ST.sim_hybrid(A, convlag(lag), acfg, order=rk)
            assert mine["C"] == theirs["C"]
            assert [k for (_, _, k, _) in mine["kick_events"]] == theirs["kicked_res"]
