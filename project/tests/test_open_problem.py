"""Machine checks for docs/open_problem.md.

Every [proved] / [refuted] claim of the document has at least one test here:
- Z3 (unsat of the negation over the reals) for the algebraic steps of each proof;
- exact (fractions.Fraction) simulation of the online policies and explicit offline schedules for every
  construction and counterexample;
- an exact offline optimum (Z3 encoding, itself validated three ways) for randomized small-instance searches;
- randomized property tests on instances that are feasible by construction ("planted").
"""

import random
from fractions import Fraction as F

import pytest
import z3

from deadline_agent.open_problem import (
    GPU,
    TOOL,
    Node,
    Task,
    chain,
    cpedf_search_instance,
    cpedf_two_task_family,
    critical_path,
    edf_tool_counterexample,
    evict_on_suspend,
    has_slack,
    offline_feasible_1m,
    peak_suspended_kv,
    planted_instance,
    r2_feasibility_adversary,
    r2_offline_witness,
    r2_parameters,
    r2_throughput_adversary,
    r2_throughput_offline_set,
    r4_general_release_counterexample,
    r4_at_5_2,
    r4_offline_keys,
    realized,
    ridouard_instance,
    simulate,
    static_order_feasible,
    witness_check,
)


def z3_proves(*constraints_and_negated_goal) -> bool:
    s = z3.Solver()
    s.add(*constraints_and_negated_goal)
    return s.check() == z3.unsat


POLICIES = ["edf", "fifo", "ps", "key"]


def keys_for(tasks, seed):
    rng = random.Random(seed)
    return {(t.tid, n): rng.random() for t in tasks for n in t.nodes}


# ===================================================================================== offline solver validation
@pytest.mark.parametrize("seed", range(40))
def test_offline_solver_validated_three_ways(seed):
    """(i) On single-node tasks without tools, Z3 feasibility == EDF feasibility (EDF is optimal on one machine);
    (ii) every Z3 'feasible' answer comes with a witness that an exact replay confirms;
    (iii) if some static node-priority order is feasible (exhaustive), Z3 must say feasible."""
    rng = random.Random(seed)
    for _ in range(4):
        ts = []
        for i in range(rng.randint(1, 3)):
            if rng.random() < 0.5:
                ts.append(chain(i, rng.randint(0, 6), 0, [(GPU, F(rng.randint(1, 6), 2))]))
            else:
                ts.append(chain(i, rng.randint(0, 6), 0, [(GPU, F(rng.randint(1, 4), 2)),
                                                          (TOOL, F(rng.randint(0, 8), 2)),
                                                          (GPU, F(rng.randint(1, 4), 2))]))
        for t in ts:
            t.deadline = t.release + critical_path(t) + F(rng.randint(0, 6), 2)
        ok, C = offline_feasible_1m(ts)
        if all(len(t.nodes) == 1 for t in ts):
            assert ok == simulate(ts, "edf", 1, 1).all_met(ts)
        if ok:
            assert witness_check(ts, C)
        else:
            assert not static_order_feasible(ts)


# ===================================================================================== Fact A (known): EDF unbounded
@pytest.mark.parametrize("s", [F(1), F(2), F(5), F(10), F(100)])
def test_factA_edf_misses_brief_instance_at_every_speed(s):
    eps = 1 / (4 * s)  # any eps < 1/(2s)
    ts = edf_tool_counterexample(eps)
    r = simulate(ts, "edf", 1, s)
    assert r.finish[1] == (1 + 2 * eps) / s + 10 - 2 * eps > 10  # closed form from the brief
    assert not r.all_met(ts)
    ok, C = offline_feasible_1m(ts)
    assert ok and witness_check(ts, C)
    assert simulate(ts, "cpedf", 1, 1).all_met(ts)  # critical-path EDF handles it even at speed 1


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_factA_ridouard_theorem6_instance(k):
    """Ridouard-Richard-Cottet RTSS'04 Thm 6: EDF at speed k misses; speed 1 is feasible (tau2 first)."""
    ts = ridouard_instance(k)
    assert not simulate(ts, "edf", 1, k).all_met(ts)
    keys = {(1, "gpu0"): 0, (0, "gpu0"): 1, (1, "gpu2"): 2}
    assert simulate(ts, "key", 1, 1, keys=keys).all_met(ts)


def test_factA_z3_edf_formula_exceeds_deadline_iff_eps_small():
    s, e = z3.Reals("s e")
    # (1+2e)/s + 10 - 2e > 10  <=>  1 + 2e > 2es  <=  e < 1/(2s)
    assert z3_proves(s >= 1, e > 0, e < 1 / (2 * s), (1 + 2 * e) / s + 10 - 2 * e <= 10)


# ===================================================================================== Theorem 2 (feasibility LB)
CASES_T2 = [(F(0), F(3, 2)), (F(0), F(4)), (F(0), F(10)), (F(1, 2), F(2)), (F(1, 2), F(5, 2)),
            (F(1), F(3, 2)), (F(1), F(19, 10)), (F(2), F(13, 10)), (F(3), F(5, 4))]


@pytest.mark.parametrize("delta,s", CASES_T2)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("m", [1, 2])
def test_thm2_adversary_defeats_every_tested_online_policy(delta, s, policy, m):
    q = r2_parameters(s, delta)
    k = m * q
    tasks, make_oracle, L_long = r2_feasibility_adversary(q, delta)
    # m machines: k = m*q tasks with the same deadline D = q*x (offline runs m pre-nodes in parallel)
    tasks = [chain(i, 0, tasks[0].deadline, [(GPU, 1), (TOOL, L_long)]) for i in range(k)]
    issued = []

    def oracle(tid, node, when):
        issued.append(tid)
        return L_long if len(issued) == k else F(0)

    keys = keys_for(tasks, 7) if policy == "key" else None
    r = simulate(tasks, policy, m, s, tool_oracle=oracle, keys=keys)
    real = realized(tasks, r.tool_len)
    assert not r.all_met(real)
    assert all(has_slack(t, delta) for t in real)
    # replaying the realized instance gives the same run (the adversary is consistent)
    assert not simulate(real, policy, m, s, keys=keys).all_met(real)
    # explicit offline speed-1 schedule: long task's GPU node first
    assert simulate(real, "key", m, 1, keys=r2_offline_witness(real)).all_met(real)


def test_thm2_z3_algebra():
    s, d, x, k, D, L = z3.Reals("s d x k D L")
    hyp = [s >= 1, d >= 0, x > 0, k >= 1 + d, k * (1 / s - d / (1 + d)) > 1, D == k * x, L == D / (1 + d) - x]
    # online: the last pre-node finishes no earlier than k x / s, so the long task ends >= kx/s + L > D
    assert z3_proves(*hyp, k * x / s + L <= D)
    # offline: long first ends at x + L = D/(1+d) <= D; slack of the long task is exactly delta
    assert z3_proves(*hyp, x + L > D)
    assert z3_proves(*hyp, (1 + d) * (x + L) != D)
    # L >= 0 and the short tasks have slack, provided k >= 1 + d
    assert z3_proves(*hyp, z3.Or(L < 0, (1 + d) * x > D))
    # a valid k exists whenever s < (1+d)/d  (i.e. 1/s - d/(1+d) > 0)
    assert z3_proves(s >= 1, d >= 0, s * d < 1 + d, 1 / s - d / (1 + d) <= 0)


@pytest.mark.parametrize("delta", [F(1, 2), F(1), F(2), F(4)])
@pytest.mark.parametrize("k", [5, 9, 16, 25])
def test_thm2_is_tight_edf_meets_adversary_at_threshold(delta, k):
    """At s = (1+delta)/delta global EDF meets the adversary's instance (as Theorem 4 guarantees)."""
    s = (1 + delta) / delta
    tasks, make_oracle, _ = r2_feasibility_adversary(k, delta)
    r = simulate(tasks, "edf", 1, s, tool_oracle=make_oracle())
    assert r.all_met(realized(tasks, r.tool_len))


@pytest.mark.parametrize("delta,s", [(F(0), F(2)), (F(1, 2), F(2)), (F(1), F(3, 2))])
@pytest.mark.parametrize("policy", POLICIES)
def test_thm2_randomized_version_counts_failing_indices(delta, s, policy):
    """Corollary 2' (randomized / known-distribution version): for the OBLIVIOUS family 'task i is the long one',
    every deterministic policy fails for at least k - floor(s*theta/x) of the k indices, theta = D - L_long.
    Averaging over a uniform i gives the bound for randomized policies."""
    k = 12
    x = F(1)
    D = k * x
    L_long = D / (1 + delta) - x
    theta = D - L_long
    fails = 0
    for i in range(k):
        tasks = [chain(j, 0, D, [(GPU, x), (TOOL, L_long if j == i else 0)]) for j in range(k)]
        keys = keys_for(tasks, 3) if policy == "key" else None
        if not simulate(tasks, policy, 1, s, keys=keys).all_met(tasks):
            fails += 1
    import math
    assert fails >= k - math.floor(s * theta / x)
    assert k - math.floor(s * theta / x) > 0


# ===================================================================================== Theorem 3 (throughput LB)
@pytest.mark.parametrize("s", [F(1), F(2), F(4), F(8)])
@pytest.mark.parametrize("policy", POLICIES)
def test_thm3_online_gets_zero_offline_gets_k_over_s(s, policy):
    k = 24
    tasks, oracle = r2_throughput_adversary(k)
    keys = keys_for(tasks, 5) if policy == "key" else None
    r = simulate(tasks, policy, 1, s, tool_oracle=oracle, keys=keys)
    real = realized(tasks, r.tool_len)
    assert r.value(real) == 0
    issue = {tid: r.node_finish[(tid, "gpu0")] for tid in range(k)}
    chosen = r2_throughput_offline_set(real, issue)
    import math
    assert len(chosen) >= math.floor(F(k) / s - F(1, 2))
    # replay the offline plan: serve `chosen` in that order on one speed-1 machine, drop the rest
    sub = [real[t] for t in chosen]
    keys_off = {(t, "gpu0"): i for i, t in enumerate(chosen)}
    assert simulate(sub, "key", 1, 1, keys=keys_off).all_met(sub)


def test_thm3_z3_algebra():
    s, k, g, l, x = z3.Reals("s k g l x")
    hyp = [s >= 1, x > 0, k >= 1, g <= k / s - z3.Q(1, 2), l >= 1, l <= g]
    # the l-th chosen task finishes offline at l*x <= tau_(k-g+l) - x/2, using tau_(j) >= j x / s
    assert z3_proves(*hyp, l * x > (k - g + l) * x / s - x / 2)


# ===================================================================================== Theorem 4 (EDF upper bound)
def test_thm4_z3_algebra():
    s, d, C, D, Y, T, P, m, Wdone, Wtot = z3.Reals("s d C D Y T P m Wdone Wtot")
    # path step: D <= T + P/s + Y with T + P <= C, s >= 1  =>  Y >= D - C
    assert z3_proves(s >= 1, T >= 0, P >= 0, T + P <= C, D <= T + P / s + Y, Y < D - C)
    # counting step
    hyp = [d > 0, s >= (1 + d) / d, C >= 0, D >= (1 + d) * C, D > 0, m >= 1, Y >= D - C,
           Wdone >= m * s * Y, Wdone < Wtot, Wtot <= m * D]
    assert z3_proves(*hyp)  # contradiction: no missed deadline
    # sanity: without the speed hypothesis the system is satisfiable (so the proof uses it)
    assert not z3_proves(d > 0, s >= 1, C >= 0, D >= (1 + d) * C, D > 0, m >= 1, Y >= D - C,
                         Wdone >= m * s * Y, Wdone < Wtot, Wtot <= m * D)


@pytest.mark.parametrize("seed", range(3000))
def test_thm4_planted_instances(seed):
    rng = random.Random(seed)
    m = rng.randint(1, 3)
    delta = rng.choice([F(1, 4), F(1, 2), F(1), F(2), F(3)])
    lifo = seed % 2 == 1
    ts = planted_instance(rng, rng.randint(1, 6), m, delta, common_release=not lifo, lifo=lifo,
                          p_tool=rng.choice([0.3, 0.6]), max_tool=rng.choice([4, 12]))
    for a in ts:
        assert has_slack(a, delta)
        for b in ts:
            if a.release < b.release:
                assert a.deadline > b.deadline
    assert simulate(ts, "edf", m, (1 + delta) / delta).all_met(ts)


@pytest.mark.parametrize("seed", range(300))
def test_thm4_exact_offline_random_single_machine(seed):
    """Random (not planted) small instances, common release, minimum slack; exact offline check."""
    rng = random.Random(1000 + seed)
    delta = rng.choice([F(1, 2), F(1), F(2)])
    ts = []
    for i in range(rng.randint(2, 4)):
        st = rng.choice([[(GPU, F(rng.randint(1, 8), 2))],
                         [(GPU, F(rng.randint(1, 4), 4)), (TOOL, F(rng.randint(1, 12), 2)), (GPU, F(rng.randint(1, 8), 2))]])
        t = chain(i, 0, 0, st)
        t.deadline = (1 + delta) * critical_path(t) + F(rng.choice([0, 0, 1]), 2)
        ts.append(t)
    ok, _ = offline_feasible_1m(ts)
    if ok:
        assert simulate(ts, "edf", 1, (1 + delta) / delta).all_met(ts)


# ===================================================================================== R4: refutation (general releases)
def test_r4_edf_at_one_plus_one_over_delta_fails_with_general_releases():
    ts, s, delta = r4_general_release_counterexample()
    assert all(has_slack(t, delta) for t in ts) and delta == 1
    for speed in [F(2), s]:  # 2 = 1 + 1/delta, and 9/4 > 2
        r = simulate(ts, "edf", 1, speed)
        assert r.finish[2] > ts[2].deadline  # task K misses
    r = simulate(ts, "edf", 1, s)
    assert r.finish[2] == F(7340, 90)  # exact: K finishes at 81 + 5/9 ... (73400/900)
    # explicit speed-1 schedule meets everything; exact Z3 agrees
    assert simulate(ts, "key", 1, 1, keys=r4_offline_keys(ts)).all_met(ts)
    ok, C = offline_feasible_1m(ts)
    assert ok and witness_check(ts, C)
    # the hypothesis of Theorem 4 is violated: A (released 0) has an earlier deadline than K (released later)
    A, K = ts[1], ts[2]
    assert A.release < K.release and A.deadline < K.deadline
    # critical-path EDF (needs tool durations) meets this instance at the same speed
    assert simulate(ts, "cpedf", 1, s).all_met(ts)


def test_r4_member_at_speed_5_2():
    """The same family with other parameters defeats EDF at speed 5/2 (delta = 1)."""
    ts, s, delta = r4_at_5_2()
    assert s == F(5, 2) and all(has_slack(t, delta) for t in ts)
    for speed in [F(2), F(9, 4), F(12, 5), F(5, 2)]:
        assert not simulate(ts, "edf", 1, speed).all_met(ts)
    assert simulate(ts, "key", 1, 1, keys=r4_offline_keys(ts)).all_met(ts)
    ok, C = offline_feasible_1m(ts)
    assert ok and witness_check(ts, C)


def test_r4_family_z3_failure_region():
    """The R4 family (parameters y, D, s; eps -> 0) makes EDF miss iff (D + y)/s > D/2 subject to the slack
    constraint y <= D/(s+1); there is a feasible point for every s < (1+sqrt 17)/2 ~ 2.56 (delta = 1)."""
    s, y, D = z3.Reals("s y D")
    sol = z3.Solver()
    sol.add(D == 40, s == z3.Q(5, 2), y > 0, y <= D / (s + 1), (D + y) / s > D / 2)
    assert sol.check() == z3.sat
    # and no such point at s >= (1+sqrt 17)/2: s^2 - s - 4 >= 0 kills it
    assert z3_proves(D > 0, y > 0, s > 0, s * s - s - 4 >= 0, y <= D / (s + 1), (D + y) / s > D / 2)


# ===================================================================================== Prop 5 (single GPU node)
@pytest.mark.parametrize("seed", range(150))
def test_prop5_single_gpu_node_tasks_cpedf_optimal_on_one_machine(seed):
    """Tasks whose DAG has exactly one GPU node (tools before and after): CP-EDF (= EDF on d - tail) meets all
    deadlines iff the exact offline optimum does (one machine, speed 1, arbitrary releases)."""
    rng = random.Random(seed)
    ts = []
    for i in range(rng.randint(1, 4)):
        nodes = {}
        pre = [f"a{j}" for j in range(rng.randint(0, 2))]
        for n in pre:
            nodes[n] = Node(n, TOOL, F(rng.randint(0, 6), 2), ())
        nodes["g"] = Node("g", GPU, F(rng.randint(1, 6), 2), tuple(pre))
        post = [f"b{j}" for j in range(rng.randint(0, 2))]
        for n in post:
            nodes[n] = Node(n, TOOL, F(rng.randint(0, 6), 2), ("g",))
        t = Task(i, F(rng.randint(0, 6)), F(0), nodes)
        t.deadline = t.release + critical_path(t) + F(rng.randint(0, 6), 2)
        ts.append(t)
    ok, _ = offline_feasible_1m(ts)
    assert simulate(ts, "cpedf", 1, 1).all_met(ts) == ok


# ===================================================================================== Theorem 6 (engine lift)
def test_thm6_engine_rate_and_lower_bound_algebra():
    a, c, B, m, lam, s, n, k = z3.Reals("a c B m lam s n k")
    # upper bound: engine sped up by lam = 2s, batch B <= m = a/c, per-sequence token rate >= s/a
    assert z3_proves(a > 0, c > 0, s >= 1, lam == 2 * s, B >= 1, B <= a / c, lam / (a + c * B) < s / a)
    # lower bound: engine throughput B*lam/(a+cB) < lam/c, so k pre-nodes of n tokens finish after k n c / lam;
    # offline runs the long one alone in n(a+c); online is later whenever k > lam (m+1), m = a/c
    assert z3_proves(a > 0, c > 0, lam > 0, B >= 1, B * lam / (a + c * B) >= lam / c)
    assert z3_proves(a > 0, c > 0, lam > 0, n > 0, m == a / c, k > lam * (m + 1), k * n * c / lam <= n * (a + c))


# ===================================================================================== Theorem 7 / Props 8-9 (KV)
@pytest.mark.parametrize("seed", range(1000))
def test_thm7_evict_on_suspend_costs_exactly_a_speed_factor(seed):
    rng = random.Random(seed)
    m = rng.randint(1, 3)
    delta = rng.choice([F(1, 2), F(1), F(2)])
    beta = rng.choice([F(1, 4), F(1), F(3)])
    ts = planted_instance(rng, rng.randint(1, 5), m, delta, p_tool=0.5)
    inflated = evict_on_suspend(ts, lambda t, n: beta * t.nodes[n].size)  # worst allowed recompute
    assert simulate(inflated, "edf", m, (1 + beta) * (1 + delta) / delta).all_met(inflated)


def test_thm7_z3_slack_preserved_at_offline_speed_one_plus_beta():
    T, P, R, b = z3.Reals("T P R b")
    # inflated path time at speed (1+b): T + (P + R)/(1+b) <= T + P whenever R <= b P
    assert z3_proves(b >= 0, P >= 0, T >= 0, R >= 0, R <= b * P, T + (P + R) / (1 + b) > T + P)


@pytest.mark.parametrize("s", [F(1), F(2), F(4)])
def test_prop8_keep_kv_edf_needs_k_times_the_memory(s):
    k = 8
    ts = [chain(i, 0, 12 * k, [(GPU, 1), (TOOL, 10), (GPU, 1)]) for i in range(k)]
    kv = {i: F(1) for i in range(k)}
    r = simulate(ts, "edf", 1, s)
    assert r.all_met(ts) and peak_suspended_kv(r, ts, kv) == k
    # an offline schedule that runs the tasks one after another (idling is allowed) meets every deadline and
    # never holds more than one task's KV
    staggered = [chain(i, 12 * i, 12 * k, [(GPU, 1), (TOOL, 10), (GPU, 1)]) for i in range(k)]
    r2 = simulate(staggered, "edf", 1, 1)
    assert r2.all_met(staggered) and peak_suspended_kv(r2, staggered, kv) == 1


def test_prop9_min_recompute_cost_policy_misses_at_every_speed():
    """Memory holds one KV. A: GPU(1) -> TOOL(5) -> GPU(1), d=100; B released at 2: GPU(1), d=7/2.
    Never-evict (recompute cost 0, optimal for caching cost) makes B wait for A: B ends at 5 + 3/s > 7/2.
    Evict A at time 2 (cost 1): B ends at 2 + 1/s <= 3; A ends at 5 + 3/s <= 100."""
    for s in [F(1), F(2), F(10), F(1000)]:
        assert 5 + 3 / s > F(7, 2)
        assert 2 + 1 / s <= F(7, 2) and 5 + 3 / s <= 100
    s = z3.Real("s")
    assert z3_proves(s >= 1, 5 + 3 / s <= z3.Q(7, 2))


# ===================================================================================== R5: CP-EDF (known tools)
@pytest.mark.parametrize("s", [F(1), F(6, 5), F(13, 10), F(7, 5), F(29, 20)])
def test_r5_cpedf_two_task_family_fails(s):
    ts = cpedf_two_task_family()
    assert not simulate(ts, "cpedf", 1, s).all_met(ts)
    if s > 1:
        assert simulate(ts, "edf", 1, s).all_met(ts)  # EDF by task deadline meets this family
    ok, C = offline_feasible_1m(ts)
    assert ok and witness_check(ts, C)
    assert simulate(ts, "key", 1, 1, keys={(1, "gpu0"): 0, (0, "gpu0"): 1, (1, "gpu2"): 2, (0, "gpu2"): 3}).all_met(ts)


def test_r5_z3_family_condition():
    """Family algebra: the blocking of T0's last node exceeds its spare time iff a > (a + c + u)(s - 1);
    the priority constraints force a < c, hence the family can only reach s < 3/2."""
    a, c, u, s = z3.Reals("a c u s")
    assert z3_proves(a > 0, c > a, u >= 0, s >= z3.Q(3, 2), a > (a + c + u) * (s - 1))


def test_r5_cpedf_search_instance_general_release():
    ts = cpedf_search_instance()
    assert not simulate(ts, "cpedf", 1, F(6, 5)).all_met(ts)
    ok, C = offline_feasible_1m(ts)
    assert ok and witness_check(ts, C)


# ===================================================================================== Prop 0 / Cor 5'
@pytest.mark.parametrize("seed", range(30))
def test_prop0_common_release_known_tools_offline_schedule_is_executable_online(seed):
    """With common release and known tools the online algorithm can compute the offline schedule at time 0 and
    replay it: every Z3 witness replays exactly (static EDF on the witness completion times)."""
    rng = random.Random(500 + seed)
    ts = []
    for i in range(rng.randint(2, 4)):
        t = chain(i, 0, 0, [(GPU, F(rng.randint(1, 4), 2)), (TOOL, F(rng.randint(0, 8), 2)), (GPU, F(rng.randint(1, 4), 2))])
        t.deadline = critical_path(t) + F(rng.randint(0, 6), 2)
        ts.append(t)
    ok, C = offline_feasible_1m(ts)
    if ok:
        assert witness_check(ts, C)


def test_cor5_slack_translates_to_job_slack():
    """Single GPU node with upstream tool path I and downstream tool path O: window d - r - I - O >= (1+d) p
    whenever d - r >= (1+d)(I + p + O)."""
    W, I, O, p, d = z3.Reals("W I O p d")
    assert z3_proves(d >= 0, I >= 0, O >= 0, p > 0, W >= (1 + d) * (I + p + O), W - I - O < (1 + d) * p)
