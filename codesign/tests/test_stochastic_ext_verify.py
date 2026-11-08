"""Independent verification tests for docs/stochastic_theory_ext.md (see docs/stochastic_theory_ext_verification.md).

All instance generators, simulators, DPs and bounds come from codesign.stochastic_ext_verify (written from
scratch; it imports nothing from the author's modules). Only test_mutations_* import the AUTHOR's tests, to check
that those tests are not vacuous. Exact Fractions; fixed seeds."""

from __future__ import annotations

import itertools
import math
import random
import sys
from fractions import Fraction as Fr

import z3

from codesign.stochastic_ext_verify import (
    Cfg, DRN, DRT, MN, MT, RN, RT, Node, Task, beta_cp_h, calib_DR, calib_propK, calib_thmH, cumulative_lambda,
    d_realizations, det_opt_single_int, dtask_indep, DTask, emin, expected_alg_machines, expected_clair_ub,
    expected_total_work, gitt1, gittins_brute, gittins_dinkelbach, grid_opt_machines, lb_cp_machine, lcv2, law,
    lemma_G3_bounds, lemma_H2_bounds, lemmaP1_holds, lmean, mtask, nonpreemptive_in_order, path, point,
    propK_parts, pwspt_single, quanta, reachable, realizations, rho_bound_ok, schulz_skutella_instance,
    sim_hybrid_unbounded, sim_machines, sim_memory, single_machine_opt, single_machine_quanta_C, sproc,
    sqrt2_instance, start, strip, task_law_W, thmDR_parts, thmG_check, thmH_parts, thmP_parts, total_work,
    v_matrix, wval, zproc,
)
from codesign.stochastic_ext_verify import (  # section I (re-verification, c46b893)
    beta_cp0_h, calib_unrestricted, cp_alg_mem, cp_opt_mem, hyb_cp, hyb_cp0, propK0_parts, thmDR0_parts,
    thmH0_parts,
)

G = gittins_dinkelbach  # cross-checked against gittins_brute in the first test


def _rl(rng, lo, hi, k, wmax=3):
    return law(*[(v, rng.randint(1, wmax)) for v in rng.sample(range(lo, hi + 1), k)])


def _rtask(rng, pmax=3, lagmax=2, nmax=3, rmax=2):
    n = rng.randint(1, nmax)
    return mtask(rng.randint(0, rmax), rng.randint(1, 4), [_rl(rng, 1, pmax, rng.choice([1, 2, 2])) for _ in range(n)],
                 [_rl(rng, 0, lagmax, rng.choice([1, 2])) for _ in range(n - 1)])


def _proved(claim, *hyps):
    s = z3.Solver()
    s.add(*hyps)
    s.add(z3.Not(claim))
    return s.check() == z3.unsat


# ============================================================================== Gittins machinery (own, exact)
def test_gittins_brute_equals_dinkelbach_and_quanta_strictly_decrease():
    rng = random.Random(0)
    n = 0
    for _ in range(50):
        T = _rtask(rng, pmax=3)
        for s in reachable(T, start(T)):
            assert gittins_brute(T, s) == gittins_dinkelbach(T, s)
            n += 1
        for _, Ns, _ in realizations([T]):
            Q = quanta(T, Ns[0], G)
            assert all(Q[i][0] > Q[i + 1][0] for i in range(len(Q) - 1))
            assert sum(w for _, w in Q) == total_work(T, Ns[0])
    assert n > 250


def test_G2_identity_and_gittins_single_machine_optimality():
    """Lemma G2 (C1_J = W_J + sum_K V_KJ) on every realization, = direct simulation; Gitt_1 = exact DP optimum
    (chains with independent nodes); doc's WSEPT example 15 > 57/4."""
    rng = random.Random(1)
    for _ in range(30):
        ts = strip([_rtask(rng) for _ in range(rng.choice([2, 3]))])
        for _, Ns, _ in realizations(ts):
            C = single_machine_quanta_C(ts, Ns, G)
            Cs = sim_machines(ts, Ns, [[Fr(0)] * len(T.lags) for T in ts], 1, G)
            V = v_matrix(ts, [quanta(T, Ns[t], G) for t, T in enumerate(ts)])
            for t, T in enumerate(ts):
                assert C[t] == Cs[t] == total_work(T, Ns[t]) + sum(V[k][t] for k in range(len(ts)))
        assert gitt1(ts, G) == single_machine_opt(ts)
    d = law((1, 1), (4, 1))
    ts = [mtask(0, 1, [d]) for _ in range(3)]
    ew = [Fr(1) / expected_total_work(T) for T in ts]
    assert gitt1(ts, G) == Fr(57, 4) and expected_alg_machines(ts, 1, fixed=ew) == 15


# =============================================================================================== Theorem G
def test_thmG_lemma_G3_pathwise_every_realization():
    rng = random.Random(2)
    chk = tight = 0
    for _ in range(120):
        ts = [_rtask(rng, pmax=4, lagmax=3) for _ in range(rng.choice([2, 3]))]
        m = rng.choice([1, 2, 3])
        for _, Ns, Ds in realizations(ts):
            for c, b in zip(sim_machines(ts, Ns, Ds, m, G), lemma_G3_bounds(ts, Ns, Ds, m, G)):
                assert c <= b
                chk += 1
                tight += c == b
    assert chk > 15000 and tight > 1000  # the bound is often attained


def test_thmG_chain_and_lower_bounds_vs_grid_dp():
    """E[ALG] <= LB_cp + (Gitt_1 - sum w E W)/m; LB_cp and Gitt_1/m below an upper bound on OPT_na (grid DP with
    lag information, so also refuting-capable); for m = 1 the grid-1/2 DP equals the grid-1 DP."""
    rng = random.Random(3)
    worst = Fr(0)
    for _ in range(25):
        ts = [_rtask(rng, pmax=3, lagmax=2, nmax=2) for _ in range(2)]
        for m in (1, 2):
            R = thmG_check(ts, m, G, grid=1)
            assert R["alg"] <= R["bound"]
            assert R["lb_cp"] <= R["grid_ub"] and R["gitt1"] / m <= R["grid_ub"]
            assert R["alg"] <= 2 * max(R["lb_cp"], R["gitt1"] / m)
            if m == 1:
                assert grid_opt_machines(ts, 1, 2) == R["grid_ub"]
                worst = max(worst, R["alg"] / R["grid_ub"])
    assert Fr(1) <= worst < 2


def test_thmG_lemma_G1_adversarial_m2_m3():
    """Try to refute OPT_na >= Gitt_1/m with fine-grid policies (time step 1/2) on 3-4 tasks; also the doc's
    instance 3 x {1,3} on m = 2: grid OPT 59/8 >= Gitt_1/2 = 6 (and < Gitt_1 = 12, so speed m is needed)."""
    rng = random.Random(7)
    worst = None
    for _ in range(8):
        ts = [mtask(0, rng.randint(1, 3), [_rl(rng, 1, 4, 2, 4)]) for _ in range(3)]
        m = 2
        ub = grid_opt_machines(ts, m, 2)
        r = ub / (gitt1(ts, G) / m)
        assert r >= 1
        worst = r if worst is None else min(worst, r)
    d = law((1, 1), (3, 1))
    ts = [mtask(0, 1, [d]) for _ in range(3)]
    assert grid_opt_machines(ts, 2, 2) == grid_opt_machines(ts, 2, 1) == Fr(59, 8)
    assert gitt1(ts, G) / 2 == 6 < Fr(59, 8) < gitt1(ts, G) == 12


def test_thmG_wsept_mutation_breaks_the_sharp_bound():
    for n, L in ((3, 6), (4, 8)):
        d = law((1, Fr(3, 4)), (L, Fr(1, 4)))
        ts = [mtask(0, 1, [d]) for _ in range(n)]
        R = thmG_check(ts, 1, G)
        fixed = [Fr(1) / expected_total_work(T) for T in ts]
        assert R["alg"] <= R["bound"] < expected_alg_machines(ts, 1, fixed=fixed)


# =============================================================================================== Theorem P
def _rdtask(rng, corr):
    n = rng.choice([1, 2, 3])
    arcs = []
    if n == 2:
        arcs = [(0, 1, _rl(rng, 0, 2, rng.choice([1, 2])))]
    if n == 3:
        arcs = ([(0, 1, _rl(rng, 0, 2, 1)), (1, 2, _rl(rng, 0, 2, 2))] if rng.random() < 0.5
                else [(0, 1, _rl(rng, 0, 2, 1)), (0, 2, _rl(rng, 0, 2, 2))])
    if corr:  # arbitrary joint law of the node sizes inside the task
        sup = [tuple(rng.randint(1, 3) for _ in range(n)) for _ in range(rng.choice([2, 3]))]
        acc = {}
        for s in sup:
            acc[s] = acc.get(s, Fr(0)) + rng.randint(1, 3)
        tot = sum(acc.values())
        return DTask(Fr(rng.randint(0, 2)), Fr(rng.randint(1, 4)), tuple((s, p / tot) for s, p in acc.items()),
                     tuple(arcs))
    return dtask_indep(rng.randint(0, 2), rng.randint(1, 4), [_rl(rng, 1, 3, rng.choice([1, 2])) for _ in range(n)],
                       arcs)


def test_thmP_chain_dags_correlated_nodes_m123():
    """E ALG <= LB_cp + Smith(EW)/m; Smith(EW) <= pair * E Smith(W); pair <= rho(Delta_W); Delta_W <= max node CV^2
    even with correlated nodes; LB_cp and E Smith(W)/m below a clairvoyant grid upper bound."""
    rng = random.Random(3)
    for it in range(50):
        ts = [_rdtask(rng, it % 2 == 1) for _ in range(rng.choice([2, 2, 3]))]
        m = rng.choice([1, 2, 3])
        R = thmP_parts(ts, m)
        assert R["alg"] <= R["lb_cp"] + R["smith_E"] / m
        assert R["smith_E"] <= R["pair"] * R["E_smith"]
        assert rho_bound_ok(R["pair"], R["delta_W"])
        for T in ts:
            mx = max(lcv2(law(*[(sz[v], p) for sz, p in T.joint])) for v in range(T.n))
            assert lcv2(task_law_W(T)) <= mx
        if len(ts) == 2 and m <= 2:
            ub = expected_clair_ub(ts, m)
            assert R["lb_cp"] <= ub and R["E_smith"] / m <= ub


def test_lemmaP1_exact_random_and_witnesses():
    rng = random.Random(9)
    for _ in range(3000):
        X = law(*[(rng.randint(0, 40), rng.randint(1, 9)) for _ in range(rng.choice([1, 2, 3, 4]))] + [(rng.randint(1, 40), 1)])
        Y = law(*[(rng.randint(0, 40), rng.randint(1, 9)) for _ in range(rng.choice([1, 2, 3, 4]))] + [(rng.randint(1, 40), 1)])
        assert lemmaP1_holds(X, Y)[0]
    # doc's witness
    X, Y = law((0, Fr(2, 3)), (3, Fr(1, 3))), law((Fr(11, 20), Fr(10, 11)), (Fr(11, 2), Fr(1, 11)))
    ok, e, D = lemmaP1_holds(X, Y)
    assert ok and D == Fr(81, 40) and (1 + D) * e == Fr(187, 240)
    # OUR sharper witness: c* <= 58/77 < 0.7533 (doc: c* in [16/27, 0.78], 'local search found 0.776')
    X, Y = law((0, Fr(19, 20)), (20, Fr(1, 20))), law((Fr(1, 2), Fr(76, 77)), (39, Fr(1, 77)))
    ok, e, D = lemmaP1_holds(X, Y)
    assert ok and lmean(X) == lmean(Y) == 1 and lcv2(X) == lcv2(Y) == D == 19 and (1 + D) * e == Fr(58, 77)
    assert Fr(16, 27) < Fr(58, 77) < Fr(776, 1000)


def test_lemmaP1_and_rho_algebra_z3():
    x, T, c, D, s, u = z3.Reals("x T c D s u")
    assert _proved(z3.If(x > T, x - T, 0) <= x * x / (4 * T), x >= 0, T > 0)
    # T = 3c/4 is the optimal truncation: (1/T)(1 - c/(4T))^2 <= 16/(27 c) whenever 4T >= c, equality at T = 3c/4
    assert _proved((1 / T) * (1 - c / (4 * T)) ** 2 <= 16 / (27 * c), c > 0, T > 0, 4 * T >= c)
    assert _proved((1 / (3 * c / 4)) * (1 - c / (3 * c)) ** 2 == 16 / (27 * c), c > 0)
    # second bound: E min = 1 - E|X-Y|/2 with E|X-Y| <= sqrt(2 Delta): 1 - sqrt(2D)/2 = 1 - sqrt(D/2)
    assert _proved(1 - u / 2 == 1 - s, u >= 0, s >= 0, u * u == 2 * D, s * s == D / 2, D >= 0)
    # Delta_W <= max node CV^2 for ANY correlation (Minkowski): v1 <= D m1^2, v2 <= D m2^2, cov^2 <= v1 v2
    m1, m2, v1, v2, cv = z3.Reals("m1 m2 v1 v2 cv")
    assert _proved(v1 + v2 + 2 * cv <= D * (m1 + m2) ** 2, m1 > 0, m2 > 0, v1 >= 0, v2 >= 0, D >= 0,
                   v1 <= D * m1 * m1, v2 <= D * m2 * m2, cv * cv <= v1 * v2)
    # calibrated Delta_s = 0.30012 < 0.3002: rho <= 1/(1 - sqrt(0.1501)) < 1.6327, and (27/16)(1.3002) > 2.19
    assert _proved(1 / (1 - s) < Fr(16327, 10000), s >= 0, s * s == Fr(3002, 20000))
    # second branch requires Delta < 2 (it blows up at 2): document
    assert _proved(1 - s <= 0, s >= 0, s * s == D / 2, D >= 2)


def test_thmP_necessity_family_one_plus_delta():
    """Fixed-priority loses (n+1)/(nq+2-q) -> 1/q = 1+Delta (exact enumeration of n i.i.d. {0, L} jobs)."""
    for n, q in ((5, Fr(1, 4)), (6, Fr(1, 3))):
        L = Fr(10)
        P = law((0, 1 - q), (L, q))
        fixed = probe = Fr(0)
        for combo in itertools.product(P, repeat=n):
            pr = math.prod((p for _, p in combo), start=Fr(1))
            sizes = [v for v, _ in combo]
            fixed += pr * sum(sum(sizes[: i + 1]) for i in range(n))
            probe += pr * sum(sum(sorted(sizes)[: i + 1]) for i in range(n))  # clairvoyant = probing here
        assert fixed / probe == Fr(n + 1) / (n * q + 2 - q)
    n, q = z3.Reals("n q")
    assert _proved((n + 1) / (n * q + 2 - q) <= 1 / q, n >= 1, q > 0, q <= 1)


# =============================================================================================== Theorem H
def _rt(rng):
    n = rng.choice([1, 2, 2, 3])
    nodes = tuple(RN(q=rng.randint(1, 6), law=_rl(rng, 1, 4, rng.choice([1, 2])), rho=(0 if i == 0 else rng.randint(0, 8)),
                     ctx=(0 if i == 0 else rng.choice([0, 1, 2]))) for i in range(n))
    return RT(Fr(rng.randint(0, 3)), Fr(rng.randint(1, 4)), nodes,
              tuple(_rl(rng, 0, 3, rng.choice([1, 2])) for _ in range(n - 1)))


def _rt_reals(ts):
    return realizations([Task(T.r, T.w, tuple(Node(nd.law) for nd in T.nodes), T.lags) for T in ts])


def test_thmH_lemma_H2_pathwise_and_s_process_mutation():
    rng = random.Random(11)
    chk = viol_s = 0
    for _ in range(150):
        ts = [_rt(rng) for _ in range(rng.choice([2, 3]))]
        cfg = Cfg(Fr(rng.choice([1, 2, 3]), rng.choice([4, 8])), Fr(1), rng.choice([1, 2, 3]))
        S = [sproc(T, cfg) for T in ts]
        for _, Xs, Ds in _rt_reals(ts):
            C = sim_hybrid_unbounded(ts, Xs, Ds, cfg, G)
            for c, b in zip(C, lemma_H2_bounds(ts, Xs, Ds, cfg, G)):
                assert c <= b
                chk += 1
            # mutation: interference charged on the s-process (no recompute)
            from codesign.stochastic_ext_verify import hyb_cp
            Vs = v_matrix(S, [quanta(s_, Xs[t], G) for t, s_ in enumerate(S)])
            viol_s += any(c > T.r + hyb_cp(T, Xs[t], Ds[t], cfg, True) + sum(Vs[k][t] for k in range(len(ts)))
                          for t, (T, c) in enumerate(zip(ts, C)))
    assert chk > 15000 and viol_s > 0


def test_thmH_chain_domination_and_delta_free_lambda():
    """Gitt_1 = exact DP for Z' and s; H3 (Gitt_1(Z') <= lambda Gitt_1(s)); full chain; and lambda <= max per-step
    ratio <= 1 + max (rho + ctx * Xparent_max)/q (mediant), i.e. Delta-free (support-dependent only)."""
    rng = random.Random(12)
    for _ in range(35):
        ts = [_rt(rng) for _ in range(2)]
        cfg = Cfg(Fr(1, 4), Fr(1), rng.choice([1, 2]))
        R = thmH_parts(ts, cfg, G)
        assert R["gz"] == single_machine_opt(R["Z"]) and R["gs"] == single_machine_opt(R["S"])
        assert R["gz"] <= R["lam"] * R["gs"]
        mid = (1 + R["bcp"]) * R["lb_cp"] + R["gz"] - R["ez"]
        assert R["alg"] <= mid <= (1 + R["bcp"] + R["lam"]) * max(R["lb_cp"], R["gs"])
        cap = max(Fr(nd.q + nd.rho + nd.ctx * (int(max(v for v, _ in T.nodes[i - 1].law)) if i else 0), nd.q)
                  for T in ts for i, nd in enumerate(T.nodes))
        assert R["lam"] <= cap


def test_thmH_engine_accrual_and_cp_algebra_z3():
    """H1's core: a speed-1 fluid engine step accrues s-work at rate u + cB(1-u)/(a+cB) <= 1; discrete step:
    cB + alpha P <= a + cB + alpha P. CP: alpha(q+rho) + aX <= (1+beta_cp)(alpha q + (a+c)X)."""
    u, c, a, B, al, q, rho, X, b = z3.Reals("u c a B al q rho X b")
    assert _proved(u + c * B * (1 - u) / (a + c * B) <= 1, u >= 0, u <= 1, a > 0, c > 0, B >= 0)
    assert _proved(al * (q + rho) + a * X <= (1 + b) * (al * q + (a + c) * X), al > 0, a > 0, c > 0, q >= 0, X >= 1,
                   rho >= 0, b >= 0, al * rho <= b * (al * q + a * X))


def test_thmH_fixed_priority_mutation():
    lw = law((1, Fr(9, 10)), (12, Fr(1, 10)))
    ts = [RT(Fr(0), Fr(1), (RN(q=1, law=lw),)) for _ in range(6)]
    cfg = Cfg(Fr(1, 100), Fr(1), 1)
    R = thmH_parts(ts, cfg, G)
    sharp = (1 + R["bcp"]) * R["lb_cp"] + R["gz"] - R["ez"]
    fixed = [Fr(1)] * 6
    wsept = sum(pr * sum(sim_hybrid_unbounded(ts, Xs, Ds, cfg, fixed=fixed)) for pr, Xs, Ds in _rt_reals(ts))
    assert R["alg"] <= sharp < wsept
    assert abs(float(sharp) - 43.575) < 1e-3 and abs(float(wsept) - 44.16) < 1e-3


# ================================================================================ KV memory: Prop K / DR / kappa
def test_propK_pathwise_chain_and_single_task_equality():
    rng = random.Random(4)
    kicking = 0
    for _ in range(150):
        ts = []
        for _ in range(rng.randint(1, 3)):
            q1, X1, q2, X2 = rng.randint(1, 9), rng.randint(1, 4), rng.randint(1, 9), rng.randint(1, 3)
            rho = q1 + X1
            ts.append(MT(Fr(rng.randint(0, 3)), Fr(rng.randint(1, 4)),
                         (MN(q1, 0, X1, q1 + X1), MN(q2, rho, X2, rho + q2 + X2)), (Fr(rng.randint(0, 3)),)))
        L = max(nd.res for T in ts for nd in T.nodes)
        cfg = Cfg(Fr(1, 4), Fr(1), rng.choice([1, 2]), Fr(L + rng.randint(L // 2, 3 * L)))
        P = propK_parts(ts, cfg)
        assert P["path"]
        assert P["lamD"] <= max(1 + P["beta"], P["theta"])
        assert P["alg"] <= (1 + P["bcp"]) * P["lb_cp"] + P["smithZ"] and P["smithZ"] <= P["lamD"] * P["smithS"]
        if len(ts) == 1:
            assert P["smithZ"] == P["lamD"] * P["smithS"]
        assert all(e == "avail" for e in P["sim"]["kick_events"])  # kicks only at availability events
        kicking += P["sim"]["kicks"] > 0
    assert kicking > 5


def test_thmDR_pathwise_chain_random_outputs_cap_reservation():
    rng = random.Random(21)
    chk = kicked = 0
    for _ in range(80):
        ts = [DRT(Fr(rng.randint(0, 3)), Fr(rng.randint(1, 4)),
                  (DRN(rng.randint(1, 8), _rl(rng, 1, 4, rng.choice([1, 2]))),
                   DRN(rng.randint(1, 8), _rl(rng, 1, 3, rng.choice([1, 2])), rng.randint(0, 8))),
                  (_rl(rng, 0, 3, rng.choice([1, 2])),)) for _ in range(rng.randint(2, 3))]
        Lc = max(nd.rho + nd.q + nd.cap for T in ts for nd in T.nodes)
        cfg = Cfg(Fr(1, 4), Fr(1), rng.choice([1, 2]), Fr(Lc + rng.randint(Lc // 3, 2 * Lc)))
        R = thmDR_parts(ts, cfg)
        assert R["path_ok"] and all(e == "avail" for e in R["kick_events"])
        assert R["alg"] <= (1 + R["bcp"]) * R["lb_cp"] + R["smithEZ"]
        assert R["smithEZ"] <= R["lamE"] * R["smithEs"] and R["smithEs"] <= R["pair"] * R["E_smith_s"]
        assert rho_bound_ok(R["pair"], R["delta_s"])
        chk += R["checks"]
        kicked += bool(R["kick_events"])
    assert chk > 5000 and kicked > 10


def test_kappa_instance_and_D0_tightness_family():
    z = MT(Fr(0), Fr(1), (MN(1, 0, 9, 10),))
    y = MT(Fr(17, 2), Fr(1000), (MN(1, 0, 1, 2),))
    out = sim_memory([z, y], Cfg(Fr(1, 100), Fr(1), 2, Fr(11)), [1, 0])
    assert out["kicks"] == 1 and out["recompute"] == Fr(949, 100) and out["kicked_res"] == [10]
    # general family: kicked RESERVATION = l+_y + L_max - 1 for any sizes, recompute / charge -> 1
    for L, s in ((10, 3), (50, 20), (200, 50)):
        z1 = MT(Fr(0), Fr(1), (MN(1, 0, L - 1, L),))
        z2 = MT(Fr(L - s), Fr(1), (MN(1, 0, s - 1, s),))
        yy = MT(Fr(L) - Fr(3, 2), Fr(10**6), (MN(1, 0, s, s + 1),))
        o = sim_memory([z1, z2, yy], Cfg(Fr(1, 1000), Fr(1), 3, Fr(L + s)), [2, 0, 1])
        assert o["kicked_res"] == [(s + 1) + L - 1]
        assert o["recompute"] > (s + 1) + L - 3


# ============================================================================================ calibrated values
def test_calibrated_constants_recomputed():
    b, lam, tot = calib_thmH(False)
    assert Fr(252, 100) < tot < Fr(253, 100) and Fr(29, 1000) < b < Fr(30, 1000) and Fr(1493, 1000) < lam < Fr(1494, 1000)
    b, lam, tot = calib_thmH(True)
    assert Fr(267, 100) < tot < Fr(268, 100) and Fr(85, 1000) < b < Fr(86, 1000) and Fr(1588, 1000) < lam < Fr(1589, 1000)
    k = {M: calib_propK(M) for M in (32000, 64000, 128000)}
    assert Fr(482, 100) < k[32000]["new"] < Fr(483, 100)
    assert Fr(579, 100) < k[64000]["new"] < Fr(580, 100) and Fr(998, 100) < k[64000]["old"] < Fr(999, 100)
    assert Fr(675, 100) < k[128000]["new"] < Fr(676, 100)
    d = calib_DR()
    assert 8.67 < d["total"] < 8.68 and Fr(3, 10) < d["delta_s"] < Fr(301, 1000) and 1.632 < d["rho"] < 1.633
    assert Fr(232, 100) < d["lamE"] < Fr(233, 100) and abs(float(d["cv2_Xp"]) - 0.85) < 0.01


def test_thmH_calibrated_lambda_matches_path_enumeration():
    """Closed-form calibration = brute-force cumulative ratio over all paths of the actual Z'/s processes."""
    cfg = Cfg(Fr("0.000114"), Fr("0.027767"), 51)
    T = RT(Fr(0), Fr(1), (RN(2000, law((50, 1), (200, 2), (800, 1))), RN(1500, law((100, 1), (300, 2), (1200, 1)), 2200)),
           (point(1),))
    assert cumulative_lambda(zproc(T, cfg), sproc(T, cfg)) == calib_thmH(True)[1]
    assert beta_cp_h([T], cfg) == calib_thmH(True)[0]


# ===================================================================================== Theorem A lower bounds
def test_sqrt2_family_correct_but_never_reaches_sqrt2():
    for L, k in ((3, 4), (4, 5)):
        J = sqrt2_instance(L, k, Fr(1, 1000))
        ub = wval(J, nonpreemptive_in_order(J, list(range(len(J)))))
        assert det_opt_single_int(J) == ub  # 'A first' is optimal on small sizes
    J = sqrt2_instance(1000, 1414, Fr(1, 10**6))
    r = wval(J, pwspt_single(J)) / wval(J, nonpreemptive_in_order(J, list(range(len(J)))))
    assert Fr(14137, 10000) < r < Fr(14138, 10000)
    x, s, L, k, e = z3.Reals("x s L k e")
    assert _proved(x / (1 + x + x * x / 2) <= s - 1, x >= 0, s > 0, s * s == 2)  # sup = sqrt 2 at x = sqrt 2
    alg = (1 + e) * (k * (L - 1) + k * (k + 1) / 2) + L * (L + k)
    ub = L * L + (1 + e) * (k * L + k * (k + 1) / 2)
    assert _proved(alg < s * ub, L >= 1, k >= 0, e >= 0, s > 0, s * s == 2)  # never attained


def test_thmA_tight_lower_bound_two_is_known_schulz_skutella():
    """FALSIFIES the doc's 'gap sqrt2 vs 2 / improves 4/3 / Thm A tightness open': the known instance of
    Schulz-Skutella (J. Sched. 2002) as restated in Megow-Schulz (ORL 2004, Lemma 2.2), m = 1, unit weights,
    deterministic, no lags: P-WSPT (= P-WSPT-DAG on single-node tasks) has ratio -> 2."""
    J = schulz_skutella_instance(150)
    alg = wval(J, pwspt_single(J))
    ub = wval(J, nonpreemptive_in_order(J, list(range(len(J)))))
    assert alg / ub > Fr(198, 100) > Fr(1415, 1000)
    # exact OPT on a scaled small member equals the non-preemptive schedule
    Ji = [(r * 6, p * 6, w) for r, p, w in schulz_skutella_instance(3)]
    assert det_opt_single_int(Ji) == wval(Ji, nonpreemptive_in_order(Ji, list(range(4)))) == 160
    assert wval(Ji, pwspt_single(Ji)) == 228


# ======================================================================= are the AUTHOR's tests discriminating?
def _author_tests():
    sys.path.insert(0, "tests")
    import test_stochastic_theory_ext as T
    import codesign.stochastic_theory_ext as X
    return T, X


def _detects(fn, patches):
    saved = [(mod, a, getattr(mod, a)) for mod, a, _ in patches]
    for mod, a, v in patches:
        setattr(mod, a, v)
    try:
        fn()
        return False
    except AssertionError:
        return True
    finally:
        for mod, a, v in saved:
            setattr(mod, a, v)


def test_mutations_author_tests_detect_false_claims():
    T, X = _author_tests()
    og = X.gittins_single_value
    assert _detects(T.test_thmG_pathwise_bound_every_realization,
                    [(T, "chain_cp", lambda Tk, Ns, Ds: sum(Ds, Fr(0))), (X, "chain_cp", lambda Tk, Ns, Ds: sum(Ds, Fr(0)))])
    assert _detects(T.test_thmG_two_approx_vs_exact_preemptive_optimum_m1,
                    [(T, "gittins_single_value", lambda ts, index=X.gittins: Fr(9, 10) * og(ts, index))])
    assert _detects(T.test_thmP_index_policy_vs_clairvoyant_m1, [(T, "pair_ratio_bound", lambda ts: Fr(1))])
    old = X.lambda_D
    assert _detects(T.test_propK_sharpened_thmD_constant,
                    [(T, "lambda_D", lambda ts, cfg: Fr(99, 100) * old(ts, cfg))])


def test_mutations_author_lemmaP1_checker_now_protected():
    """Final check (87186f5), ruling (b) resolved. The witness test now feeds the 58/77 witness through
    emin_lower_bound_holds, so raising the checker's constant to 0.76 or 0.80 makes the author's suite FAIL (the
    whole suite was run under both mutations by hand; here the two callers are run, which is where it is called)."""
    import inspect
    T, X = _author_tests()
    from codesign.stochastic_theory import cv2
    callers = sorted(n for n, f in inspect.getmembers(T, inspect.isfunction)
                     if n.startswith("test_") and "emin_lower_bound_holds" in inspect.getsource(f))
    assert callers == ["test_lemmaP1_emin_lower_bound_and_witness", "test_lemmaP1_witness_58_77_refutes_larger_constants"]

    def checker(c):
        def h(d1, d2):
            A, B = X.normalize(d1), X.normalize(d2)
            D, e = max(cv2(A), cv2(B)), X.emin(A, B)
            return e >= c / (1 + D) and (e >= 1 or (1 - e) ** 2 <= D / 2)
        return h

    T.test_lemmaP1_witness_58_77_refutes_larger_constants()  # passes on the true checker
    for c in (Fr(76, 100), Fr(80, 100), Fr(1)):
        patch = [(T, "emin_lower_bound_holds", checker(c)), (X, "emin_lower_bound_holds", checker(c))]
        assert _detects(T.test_lemmaP1_witness_58_77_refutes_larger_constants, patch)
    assert not _detects(T.test_lemmaP1_witness_58_77_refutes_larger_constants,
                        [(T, "emin_lower_bound_holds", checker(Fr(58, 77))), (X, "emin_lower_bound_holds", checker(Fr(58, 77)))])


def test_mutations_author_final_87186f5():
    """5b: with the calibrated values held fixed, the two random families of test_preload_aware_constants_H_DR_K
    fail when beta_cp0 := 0 and when Gitt_1(Z') - sum w E Z' is dropped. 5c: dropping the min (keeping only
    1 + beta_cp0) fails, and so does beta off by one. Plus an independent recomputation of the 5c instance."""
    T, X = _author_tests()
    import codesign.stochastic_theory as B
    cache = {True: X.calibrated_constants(True), False: X.calibrated_constants(False)}
    cal = [(T, "calibrated_constants", lambda pa=False: cache[pa])]
    oc, ohc = X.thmH_constants, B.h_constants
    t = T.test_preload_aware_constants_H_DR_K
    assert not _detects(t, cal)
    assert _detects(t, cal + [(T, "thmH_constants", lambda ts, cfg, preload_aware=False:
                               (Fr(0), oc(ts, cfg, True)[1]) if preload_aware else oc(ts, cfg, False))])
    assert _detects(t, cal + [(T, "gittins_single_value",
                               lambda ts, index=X.gittins: sum(u.w * X.expected_work(u) for u in ts))])
    assert _detects(t, cal + [(T, "thmD_new_constant", lambda ts, cfg, preload_aware=False:
                               (lambda bcp, b, _: 1 + bcp + 2 * X.lambda_D(ts, cfg))(*ohc(ts, cfg, preload_aware)))])
    assert _detects(t, cal + [(T, "h_constants", lambda ts, cfg, preload_aware=False:
                               (lambda bcp, b, th: (bcp, b + 1, th))(*ohc(ts, cfg, preload_aware)))])
    # 5c recomputed with this module (pieces -> one lag of 10; ALG never prefills before availability)
    Tm = MT(Fr(0), Fr(1), (MN(1, 0, 1, 2), MN(11, 0, 1, 12, True)), (Fr(10),))
    P = propK0_parts([Tm], Cfg(Fr(1), Fr(1), 1, Fr(100)))
    assert (P["bcp0"], P["beta0"], P["alg"], P["lb0"], P["wsig"]) == (11, 7, 24, 15, 14)
    assert P["lamD"] == Fr(2295, 628) and P["smithS"] == Fr(157, 11) and P["nodewise"] and P["additive"]
    assert min(1 + P["bcp0"], 2 + P["beta0"]) == 9 < 12 and P["alg"] <= P["bound0"] == Fr(7883, 44)
    # ...and it equals the author's lambda_D / thmD_new_constant on the author's pieces version
    Ta = B.HTask(Fr(0), Fr(1), (B.HNode(q=1, X=1), B.HNode(q=11, X=1, rho=0, pieces=((B.det(0), 10), (B.det(10), 1)))),
                 ((0, 1, B.det(0)),))
    ca = B.HCfg(alpha=Fr(1), a=Fr(1), m=1, Mp=100)
    assert X.lambda_D([Ta], ca) == P["lamD"] and X.thmD_new_constant([Ta], ca, True) == 9 + 2 * P["lamD"]


def test_mutations_author_new_tests_c46b893():
    """The new or tightened author tests fail under false mutations: Thm G m = 2 (Gitt_1 shifted), Thm H beta_cp
    instance (beta_cp scaled), preload-aware calibration (beta_cp0 -> beta_cp, q0 -> q, min branch -> 2 + beta)."""
    T, X = _author_tests()
    import codesign.stochastic_theory as B
    og, oc, ohc = X.gittins_single_value, X.thmH_constants, B.h_constants
    ew = lambda ts: sum(t.w * X.expected_work(t) for t in ts)
    assert _detects(T.test_thmG_parallel_machines_consistency,
                    [(T, "gittins_single_value", lambda ts, index=X.gittins: (og(ts, index) + ew(ts)) / 2)])
    for f in (Fr(0), Fr(4, 5)):
        assert _detects(T.test_thmH_ratio_domination_and_gittins_optimality,
                        [(T, "thmH_constants", lambda ts, cfg, pa=False, f=f: (f * oc(ts, cfg, pa)[0], oc(ts, cfg, pa)[1]))])
    t = T.test_preload_aware_constants_H_DR_K
    assert _detects(t, [(X, "thmH_constants", lambda ts, cfg, pa=False: oc(ts, cfg, False))])
    assert _detects(t, [(X, "h_constants", lambda ts, cfg, pa=False: ohc(ts, cfg, False))])
    assert _detects(t, [(X.RNode, "q0", property(lambda s: s.q))])
    assert _detects(t, [(X.CapNode, "q0", property(lambda s: s.q))])
    assert _detects(t, [(X, "thmD_new_constant", lambda ts, cfg, pa=False: (
        (lambda bcp, b, _: (2 + b if pa else 1 + bcp) + 2 * X.lambda_D(ts, cfg))(*ohc(ts, cfg, pa))))])

    def dr_no_small_delta_branch(tasks, cfg, preload_aware=False):
        EZ, sl, _ = X.dr_expected_sizes(tasks, cfg)
        Es = [sum(v * p for v, p in l) for l in sl]
        D, lamE = max(B.cv2(l) for l in sl), max(z / e for z, e in zip(EZ, Es))
        cs = [ohc(h, cfg, preload_aware) for _, h, _ in X.dr_realizations(tasks)]
        bcp, b = max(c[0] for c in cs), max(c[1] for c in cs)
        first = min(1 + bcp, 2 + b) if preload_aware else 1 + bcp
        return float(first) + 2 * float(lamE) * 27 / 16 * (1 + float(D))

    assert _detects(t, [(X, "thmDR_constant", dr_no_small_delta_branch)])
    # the author's unrestricted constants equal this module's independent recomputation EXACTLY
    mine, au = calib_unrestricted(), X.calibrated_constants(True)
    assert all(mine[k] == au[k] for k in au)


# ===================================================== re-verification (c46b893): unrestricted benchmark ((L1) fix)
def _rt_fanout(rng):
    T = _rt(rng)
    return RT(T.r, T.w, tuple(RN(nd.q, nd.law, nd.rho, nd.ctx, fanout=(i > 0 and rng.random() < 0.7))
                              for i, nd in enumerate(T.nodes)), T.lags)


def test_unrestricted_thmH_chain_with_fanout_nodes():
    """Thm H, unrestricted benchmark: node-wise CP_alg <= (1+beta_cp0) CP0 on every realization; E ALG <=
    (1+beta_cp0) LB0 + Gitt_1(Z') - sum w E Z' <= (1+beta_cp0+lambda) max(LB0, Gitt_1(s)); LB0 <= LB_cp. Necessity:
    with the restricted beta_cp against LB0 the bound FAILS on a fan-out instance ((L1) counterexample shape)."""
    rng = random.Random(31)
    fo = 0
    for _ in range(30):
        ts = [_rt_fanout(rng) for _ in range(rng.choice([1, 2, 2]))]
        fo += any(nd.fanout for T in ts for nd in T.nodes)
        cfg = Cfg(Fr(rng.choice([1, 2]), 4), Fr(1), rng.choice([1, 2]))
        R = thmH0_parts(ts, cfg, G)
        assert R["nodewise"] and R["lb0"] <= R["lb_cp"] and R["bcp0"] >= R["bcp"]
        mid = (1 + R["bcp0"]) * R["lb0"] + R["gz"] - R["ez"]
        assert R["alg"] <= mid and R["gz"] <= R["lam"] * R["gs"]
        assert mid <= (1 + R["bcp0"] + R["lam"]) * max(R["lb0"], R["gs"])
    assert fo > 15
    T = RT(Fr(0), Fr(1), (RN(1, point(1)), RN(10, point(1), fanout=True)), (point(0),))
    cfg = Cfg(Fr(1, 4), Fr(1), 1)
    R = thmH0_parts([T], cfg, G)
    assert R["bcp"] == 0 and R["bcp0"] == Fr(5, 2) and R["gz"] == R["ez"]
    assert (1 + R["bcp"]) * R["lb0"] == Fr(17, 4) < R["alg"] == Fr(19, 4) <= (1 + R["bcp0"]) * R["lb0"] and R["lb_cp"] == Fr(27, 4)


def test_unrestricted_propK_and_DR_both_branches():
    """Prop K and Thm DR, unrestricted benchmark, final prompt a fan-out: on every realization CP_alg <= (1+bcp0)CP0
    (node-wise) and CP_alg <= CP0 + (1+beta) sigma (additive, (L0)); hence E ALG <= min{(1+bcp0) LB0, LB0 + (1+beta)
    sum w E sigma} + (lambda_D S or Smith(E Zbar)). The 2 + beta branch is strictly better on some instances."""
    rng = random.Random(41)
    binds = 0
    for _ in range(60):
        ts = []
        for _ in range(rng.randint(1, 3)):
            q1, X1, q2, X2 = rng.randint(1, 9), rng.randint(1, 4), rng.randint(1, 9), rng.randint(1, 3)
            rho = q1 + X1
            ts.append(MT(Fr(rng.randint(0, 3)), Fr(rng.randint(1, 4)),
                         (MN(q1, 0, X1, q1 + X1), MN(q2, rho, X2, rho + q2 + X2, True)), (Fr(rng.randint(0, 3)),)))
        L = max(nd.res for T in ts for nd in T.nodes)
        cfg = Cfg(Fr(rng.choice([1, 4]), 4), Fr(1), rng.choice([1, 2]), Fr(L + rng.randint(L // 2, 3 * L)))
        P = propK0_parts(ts, cfg)
        assert P["path"] and P["nodewise"] and P["additive"]
        assert P["alg"] <= P["bound0"] and P["lb0"] <= P["lb_cp"] and P["wsig"] <= P["lb_cp"]
        binds += 2 + P["beta0"] < 1 + P["bcp0"]
    assert binds > 3
    binds = 0
    for _ in range(30):
        ts = [DRT(Fr(rng.randint(0, 3)), Fr(rng.randint(1, 4)),
                  (DRN(rng.randint(1, 8), _rl(rng, 1, 4, rng.choice([1, 2]))),
                   DRN(rng.randint(1, 8), _rl(rng, 1, 3, rng.choice([1, 2])), rng.randint(0, 8), True)),
                  (_rl(rng, 0, 3, rng.choice([1, 2])),)) for _ in range(rng.randint(2, 3))]
        Lc = max(nd.rho + nd.q + nd.cap for T in ts for nd in T.nodes)
        cfg = Cfg(Fr(rng.choice([1, 4]), 4), Fr(1), rng.choice([1, 2]), Fr(Lc + rng.randint(Lc // 3, 2 * Lc)))
        R = thmDR0_parts(ts, cfg)
        assert R["path_ok"] and R["nodewise"] and R["additive"]
        assert R["alg"] <= R["bound0"] and R["lb0"] <= R["lb_cp"] and R["ewsig"] <= R["lb_cp"]
        binds += 2 + R["beta0"] < 1 + R["bcp0"]
    assert binds > 3


def test_unrestricted_calibrated_constants_recomputed():
    """2.54 / 2.74 (Thm H), 8.74 (DR), 4.84 / 5.82 / 6.78 (Prop K at 32k/64k/128k), 10.01 (old Thm D form); the
    min{1 + beta_cp0, 2 + beta} takes the 1 + beta_cp0 branch at the calibrated values (1.05 vs 5.48; 1.15 vs 8.07)."""
    u = calib_unrestricted()
    al, a = Fr("0.000114"), Fr("0.027767")
    assert u["H_det"] - calib_thmH(False)[2] == al * 3700 / (a * 300) - al * 2200 / (al * 1500 + a * 300)
    assert u["H_rand"] - calib_thmH(True)[2] == al * 3700 / (a * 100) - al * 2200 / (al * 1500 + a * 100)
    assert Fr(254, 100) < u["H_det"] < Fr(255, 100) and Fr(274, 100) < u["H_rand"] < Fr(275, 100)
    assert 8.742 < u["DR"] < 8.743 and abs(u["DR"] - (calib_DR()["total"] - float(al * 2200 / (al * 1500 + a * 100))
                                                       + float(al * 3700 / (a * 100)))) < 1e-12
    assert Fr(484, 100) < u["K32"] < Fr(485, 100) and Fr(582, 100) < u["K64"] < Fr(583, 100)
    assert Fr(677, 100) < u["K128"] < Fr(678, 100) and Fr(1000, 100) < u["D64"] < Fr(1001, 100)
    assert u["first_K64"] < u["two_plus_beta_K64"] and u["first_DR"] < u["two_plus_beta_DR"]
