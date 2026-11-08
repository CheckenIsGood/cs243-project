"""Independent adversarial verification of docs/general_theory.md and docs/open_problem.md.

Uses only `deadline_agent.verify_independent` (written from the definitions), Z3, and exact Fractions.
The section "CROSS-CHECKS" at the end is the only place the project's own modules are imported, and only to
compare their answers against the independent ones.

Tests that demonstrate a flaw in a claim *as written* are ordinary passing tests that exhibit the
counterexample, paired with an `xfail(strict=True)` test that states the claim literally.
See docs/verification_report.md for the verdicts.
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction as F

import pytest
import z3

from deadline_agent.verify_independent import (
    GPU,
    ONE,
    TOOL,
    ZERO,
    LastIssuedLong,
    Node,
    Piece,
    Task,
    Thm3Adversary,
    check_pieces,
    crit_path,
    discrete_engine_feasible,
    discrete_steps_to_machine,
    edf,
    edf_key,
    engine_min_makespan,
    engine_on_time_value_common_deadline,
    engine_policy,
    enforce_release_order,
    evict_inflate,
    factA_instance,
    fifo_key,
    fluid_engine_following,
    knapsack,
    lifo_key,
    machine_segments_completion,
    make_cpedf_key,
    mcnaughton,
    mk_chain,
    offline_1m_by_orders,
    offline_mm_lp,
    one_machine_on_time_value,
    peak_held_kv,
    planted,
    priority_policy,
    ps_policy,
    r4_family,
    r4_instance,
    r5_instance,
    random_feasible_run,
    random_task,
    release_order_ok,
    run_to_pieces,
    simulate,
    slack_ok,
    srpt_key,
    static_key,
    tails,
    thm2_instance,
    thm2_offline_pieces,
    thm2_q,
    validate_schedule,
    with_tools,
)


def unsat(*cs) -> bool:
    s = z3.Solver()
    s.add(*cs)
    r = s.check()
    assert r != z3.unknown
    return r == z3.unsat


def sat(*cs) -> bool:
    return not unsat(*cs)


def policies(m, s, tasks, seed=0):
    """A spread of deterministic online policies (all non-clairvoyant about tool durations except cpedf, which
    sees only the stored *upper bound*)."""
    rng = random.Random(seed)
    keys = {(t.tid, v): rng.random() for t in tasks for v in range(len(t.nodes))}

    def lazy(now, ready, ctx):  # idles one machine whenever it can (non-work-conserving)
        ranked = sorted(ready, key=lambda k: (edf_key(k, ctx), k))
        return {k: F(s) for k in ranked[:max(1, m - 1)]}

    return {
        "edf": edf(m, s),
        "fifo": priority_policy(m, s, fifo_key),
        "lifo": priority_policy(m, s, lifo_key),
        "srpt": priority_policy(m, s, srpt_key),
        "cpedf_ub": priority_policy(m, s, make_cpedf_key(tasks)),
        "ps": ps_policy(m, s),
        "random_static": priority_policy(m, s, static_key(keys)),
        "lazy": lazy,
    }


# =============================================================================================== Theorem 1
@pytest.mark.parametrize("seed", range(120))
def test_thm1_restricted_problem_is_knapsack(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 7)
    w = [rng.randint(1, 12) for _ in range(n)]
    v = [rng.randint(1, 30) for _ in range(n)]
    cap = rng.randint(0, sum(w))
    assert one_machine_on_time_value(w, v, [cap] * n) == knapsack(w, v, cap)


@pytest.mark.parametrize("seed", range(40))
def test_thm1_engine_makespan_with_batching_is_a_max_plus_c_sum(seed):
    """Discrete engine, all released at 0: the minimum time to finish a set equals a*max_j n_j + c*sum_j n_j
    (brute force over every step sequence)."""
    rng = random.Random(seed)
    tok = tuple(rng.randint(1, 3) for _ in range(rng.randint(1, 4)))
    a, c = F(rng.randint(1, 5), rng.randint(1, 3)), F(rng.randint(1, 4), rng.randint(1, 3))
    assert engine_min_makespan(tok, a, c) == a * max(tok) + c * sum(tok)


@pytest.mark.parametrize("seed", range(25))
def test_thm1_strengthening_hardness_survives_unbounded_batching(seed):
    """Correction/strengthening of Thm 1: batching need not be forbidden. With c = 1, a = 1/(2*sum n + 1) and
    common deadline D + 1/2, the batching engine's max on-time value equals Knapsack(capacity D)."""
    rng = random.Random(100 + seed)
    n = rng.randint(1, 4)
    tok = [rng.randint(1, 3) for _ in range(n)]
    val = [rng.randint(1, 9) for _ in range(n)]
    D = rng.randint(0, sum(tok))
    a = F(1, 2 * sum(tok) + 1)
    assert engine_on_time_value_common_deadline(tok, val, a, 1, D + F(1, 2)) == knapsack(tok, val, D)


# =============================================================================================== Lemma 2
a_, c_, B_, s_, lam_ = z3.Reals("a c B s lam")


def test_lemma2_z3_algebra_and_hypothesis_necessity():
    base = [a_ > 0, c_ > 0, B_ >= 0]
    # (i) c B <= a  =>  1/(a + cB) >= 1/(2a); dropping c B <= a makes it false
    assert unsat(*base, c_ * B_ <= a_, 1 / (a_ + c_ * B_) < 1 / (2 * a_))
    assert sat(*base, 1 / (a_ + c_ * B_) < 1 / (2 * a_))
    # (ii) per-sequence rate <= 1/a and total B/(a+cB) < 1/c = m * (1/a)
    assert unsat(*base, 1 / (a_ + c_ * B_) > 1 / a_)
    assert unsat(*base, B_ >= 1, B_ / (a_ + c_ * B_) >= 1 / c_)
    # discrete (ii): within one step of length a + cB, B tokens of machine-time a fit on m = a/c machines
    assert unsat(*base, B_ >= 1, B_ * a_ > (a_ / c_) * (a_ + c_ * B_))
    # (ii) needs c > 0: with c = 0 the engine would exceed every finite machine count
    assert sat(a_ > 0, c_ == 0, B_ >= 1, B_ / (a_ + c_ * B_) * a_ > 51)


def test_lemma2_floor_m_needs_only_a_over_c_at_least_one():
    """Doc: 'further factor <= 2 when a/c >= 2'. True, but a/c >= 1 already suffices; a/c < 1 breaks it."""
    x, mi = z3.Real("x"), z3.Int("mi")
    fl = [z3.ToReal(mi) <= x, x < z3.ToReal(mi) + 1]
    assert unsat(*fl, x >= 1, 2 * z3.ToReal(mi) < x)        # floor(x) >= x/2 for x >= 1
    assert sat(*fl, x > 0, x < 1, 2 * z3.ToReal(mi) < x)     # fails below 1 (floor = 0 machines)
    assert math.floor(F(278, 10) / F(543, 1000)) == 51       # the measured m


@pytest.mark.parametrize("seed", range(60))
def test_lemma2_mcnaughton_exact(seed):
    rng = random.Random(seed)
    m = rng.randint(1, 5)
    L = F(rng.randint(1, 12), rng.randint(1, 4))
    dem = {j: L * F(rng.randint(0, 8), 8) for j in range(rng.randint(1, 12))}
    tot = sum(dem.values(), ZERO)
    if tot > m * L:
        dem = {j: d * m * L / tot for j, d in dem.items()}
    pcs = mcnaughton(dem, m, F(3), F(3) + L)
    check_pieces(pcs)
    for j, d in dem.items():
        assert sum((p.end - p.start for p in pcs if p.key == j), ZERO) == d
    with pytest.raises(ValueError):
        mcnaughton({0: L + 1}, m, ZERO, L)


@pytest.mark.parametrize("seed", range(80))
def test_lemma2_i_fluid_engine_never_later_than_machines(seed):
    """Fluid model (rates, fractional tokens): following any M_m(1/(2a)) schedule, every engine completion is
    no later than the machine completion. Exact."""
    rng = random.Random(seed)
    c = F(rng.randint(1, 4), rng.randint(1, 3))
    m = rng.randint(1, 5)
    a = c * m
    jobs = list(range(rng.randint(1, 3 * m + 2)))
    work = {j: F(rng.randint(1, 12), rng.randint(1, 3)) for j in jobs}
    segs, t = [], ZERO
    for _ in range(rng.randint(5, 40)):
        t1 = t + F(rng.randint(1, 30), rng.randint(1, 4))
        segs.append((t, t1, frozenset(rng.sample(jobs, min(len(jobs), rng.randint(0, m))))))
        t = t1
    s = 1 / (2 * a)
    tm = machine_segments_completion(segs, work, s)
    te = fluid_engine_following(segs, work, a, c)
    for j, x in tm.items():
        assert j in te and te[j] <= x


def test_lemma2_i_fails_for_the_discrete_step_engine_common_release():
    """FLAW (as stated with discrete steps): m = a/c = 2, a = 2, c = 1, three 1-token jobs released at 0.
    McNaughton on M_2(1/(2a)) finishes them at 4, 6, 6; no discrete engine schedule meets (4, 6, 6)."""
    a, c = F(2), F(1)
    pcs = mcnaughton({"J1": F(4), "J2": F(4), "J3": F(4)}, 2, ZERO, F(6))
    ends = {j: max(p.end for p in pcs if p.key == j) for j in ("J1", "J2", "J3")}
    assert ends == {"J1": 4, "J2": 6, "J3": 6}
    assert not discrete_engine_feasible([(ZERO, 1, F(4)), (ZERO, 1, F(6)), (ZERO, 1, F(6))], a, c)
    # lateness of one extra a/2 suffices here (not a proven bound)
    assert discrete_engine_feasible([(ZERO, 1, F(5)), (ZERO, 1, F(7)), (ZERO, 1, F(7))], a, c)
    # the fluid relaxation does meet them (the proof's model)
    segs = [(ZERO, F(2), frozenset({"J1", "J2"})), (F(2), F(4), frozenset({"J1", "J3"})),
            (F(4), F(6), frozenset({"J2", "J3"}))]
    assert fluid_engine_following(segs, {"J1": ONE, "J2": ONE, "J3": ONE}, a, c) == ends


def test_lemma2_i_fails_for_the_discrete_step_engine_with_release():
    """FLAW, second form: a job released mid-step cannot join it. a = 2, c = 1 (m = 2): A (r 0, d 4) and
    B (r 1, d 5) are met on M_2(1/4) but by no discrete engine schedule."""
    assert not discrete_engine_feasible([(ZERO, 1, F(4)), (ONE, 1, F(5))], F(2), F(1))


@pytest.mark.xfail(strict=True, reason="Lemma 2(i) is false for the discrete-step engine (holds only in the fluid model)")
def test_lemma2_i_as_literally_stated_for_discrete_engine():
    assert discrete_engine_feasible([(ZERO, 1, F(4)), (ZERO, 1, F(6)), (ZERO, 1, F(6))], F(2), F(1))


@pytest.mark.parametrize("seed", range(60))
def test_lemma2_ii_holds_for_discrete_steps(seed):
    """Every discrete engine schedule (any batch sizes, also B > m) is executable on M_m(1/a), each token
    finishing inside its own step."""
    rng = random.Random(seed)
    c = F(rng.randint(1, 3))
    m = rng.randint(1, 4)
    a = c * m
    jobs = list(range(rng.randint(1, 8)))
    steps, t = [], ZERO
    for _ in range(rng.randint(1, 15)):
        t += F(rng.randint(0, 3), 2)
        b = frozenset(rng.sample(jobs, rng.randint(1, len(jobs))))
        steps.append((t, b))
        t += a + c * len(b)
    pcs = discrete_steps_to_machine(steps, a, c, m)
    check_pieces(pcs)
    # a job is never on two machines at once across steps either
    check_pieces([Piece(p.key[0], -1 - p.machine, p.start, p.end) for p in pcs])
    for st, b in steps:
        for j in b:
            assert sum((p.end - p.start for p in pcs if p.key == (j, st)), ZERO) == a


# =============================================================================================== Lemma 3
@pytest.mark.parametrize("seed", range(60))
def test_lemma3_offline_feasibility_is_monotone_in_tool_end_times(seed):
    """Lemma 3 (eager issue) <=> lengthening a tool never helps the offline optimum. One machine, exact."""
    rng = random.Random(seed)
    ts = [random_task(rng, i, F(rng.randint(0, 4)), n=(1, 3), p_tool=0.5) for i in range(rng.randint(1, 3))]
    for t in ts:
        t.deadline = t.release + crit_path(t) + F(rng.randint(0, 8), 2)
    tools = [(t.tid, v) for t in ts for v, nd in enumerate(t.nodes) if nd.kind == TOOL]
    delayed = with_tools(ts, {k: ts[k[0]].nodes[k[1]].size + F(rng.randint(0, 6), 2) for k in tools})
    if offline_1m_by_orders(delayed) is not None:
        assert offline_1m_by_orders(ts) is not None


@pytest.mark.parametrize("seed", range(15))
def test_lemma3_monotone_two_machines(seed):
    rng = random.Random(500 + seed)
    ts = [random_task(rng, i, ZERO, n=(1, 3), p_tool=0.5) for i in range(rng.randint(2, 3))]
    for t in ts:
        t.deadline = crit_path(t) + F(rng.randint(0, 4), 2)
    tools = [(t.tid, v) for t in ts for v, nd in enumerate(t.nodes) if nd.kind == TOOL]
    delayed = with_tools(ts, {k: ts[k[0]].nodes[k[1]].size + F(rng.randint(0, 6), 2) for k in tools})
    if offline_mm_lp(delayed, 2)[0]:
        assert offline_mm_lp(ts, 2)[0]


def test_lemma3_limit_online_edf_is_not_monotone():
    """Supports the doc's caveat: shortening a tool (earlier result) makes EDF miss at speed 1."""
    ts = [Task(0, ZERO, F(15, 2), (Node(GPU, ONE), Node(TOOL, F(1, 2), (0,)), Node(GPU, F(2), (1,)))),
          Task(1, ZERO, F(8), (Node(GPU, F(1, 2)), Node(TOOL, F(6), (0,)))),
          Task(2, ZERO, F(10), (Node(GPU, F(3, 2)), Node(TOOL, F(15, 2))))]
    assert simulate(ts, edf(1, 1)).all_met(ts)
    shorter = with_tools(ts, {(0, 1): ZERO})
    assert not simulate(shorter, edf(1, 1)).all_met(shorter)
    assert offline_1m_by_orders(shorter) is not None


# =============================================================================================== Fact A
@pytest.mark.parametrize("s", [F(1), F(3, 2), F(2), F(5), F(100)])
def test_factA_closed_form_and_offline(s):
    eps = 1 / (4 * s)
    ts = factA_instance(eps)
    r = simulate(ts, edf(1, s))
    assert r.task_end[1] == (1 + 2 * eps) / s + 10 - 2 * eps > 10
    assert offline_1m_by_orders(ts) is not None


def test_factA_iff_condition_in_doc_is_wrong():
    """Doc: 'finishes B at (1+2e)/s + 10 - 2e > 10 iff e < 1/(2s)'. Only 'if' holds; the exact condition is
    e < 1/(2(s-1)) for s > 1 (and every e for s = 1)."""
    s, e = z3.Reals("s e")
    late = (1 + 2 * e) / s + 10 - 2 * e > 10
    assert unsat(s > 1, e > 0, late != (e < 1 / (2 * (s - 1))))
    assert sat(s > 1, e > 0, late != (e < 1 / (2 * s)))
    ts = factA_instance(F(1, 3))  # s = 2, e = 1/3 >= 1/(2s) = 1/4, yet EDF misses
    assert not simulate(ts, edf(1, 2)).all_met(ts)


# =============================================================================================== Theorem 2
T2_CASES = [(F(0), F(1)), (F(0), F(3)), (F(1, 2), F(2)), (F(1, 2), F(29, 10)), (F(1), F(3, 2)),
            (F(1), F(19, 10)), (F(2), F(7, 5)), (F(3), F(13, 10))]


@pytest.mark.parametrize("delta,s", T2_CASES)
@pytest.mark.parametrize("m", [1, 2, 3])
def test_thm2_adversary_beats_every_policy(delta, s, m):
    q = thm2_q(s, delta)
    base, D, L = thm2_instance(m, q, delta)
    assert L >= 0
    for name in ["edf", "fifo", "lifo", "srpt", "cpedf_ub", "ps", "random_static", "lazy"]:
        tasks = [t.copy() for t in base]
        adv = LastIssuedLong(m * q, L)
        run = simulate(tasks, policies(m, s, tasks, seed=q)[name], tool_oracle=adv, check=(m, s))
        real = with_tools(tasks, {k: d for k, d in run.tool_dur.items()})
        real = with_tools(real, {(t.tid, 1): ZERO for t in real if (t.tid, 1) not in run.tool_dur})
        assert not run.all_met(real), name
        assert all(slack_ok(t, delta) for t in real)
        assert validate_schedule(real, m, ONE, thm2_offline_pieces(real, m))
        if m * q <= 6:
            assert offline_mm_lp(real, m)[0]


@pytest.mark.parametrize("delta", [F(1, 2), F(1), F(2), F(4)])
def test_thm2_threshold_is_sharp_both_ways(delta):
    """Just below 1 + 1/delta the adversary defeats EDF; at 1 + 1/delta no adversary instance exists and EDF meets
    the instance for every q and every choice of which task is long. Refutes any claimed threshold < 1+1/delta
    (e.g. a mutation to 1 + 1/(2 delta))."""
    thr = 1 + 1 / delta
    s = thr - F(1, 50)
    q = thm2_q(s, delta)
    tasks, D, L = thm2_instance(1, q, delta)
    run = simulate(tasks, edf(1, s), tool_oracle=LastIssuedLong(q, L))
    assert not run.all_met(tasks)
    assert 1 + 1 / (2 * delta) < s  # a mutated threshold 1 + 1/(2 delta) would be contradicted
    with pytest.raises(ValueError):
        thm2_q(thr, delta)
    for q in range(max(2, math.ceil(1 + delta)), 14):
        tasks, D, L = thm2_instance(1, q, delta)
        for longi in range(q):
            real = with_tools(tasks, {(t.tid, 1): (L if t.tid == longi else ZERO) for t in tasks})
            assert simulate(real, edf(1, thr)).all_met(real)
        assert simulate(tasks, edf(1, thr), tool_oracle=LastIssuedLong(q, L)).all_met(tasks)


def test_thm2_z3_algebra_with_hypothesis_necessity():
    s, d, q = z3.Reals("s d q")
    L = q / (1 + d) - 1
    hyp = [s >= 1, d >= 0, q >= 1 + d, q * (1 / s - d / (1 + d)) > 1]
    assert unsat(*hyp, q / s + L <= q)                  # online misses
    assert unsat(*hyp, z3.Or(1 + L > q, L < 0, (1 + d) * 1 > q))  # offline feasible, L >= 0, short tasks slack
    assert unsat(*hyp, (1 + d) * (1 + L) != q)           # long task slack exactly delta
    # necessity of the gap hypothesis: without it the online can meet the deadline
    assert sat(s >= 1, d >= 0, q >= 1 + d, q / s + L <= q)
    # existence of q iff s*d < 1 + d
    assert unsat(s >= 1, d >= 0, s * d < 1 + d, 1 / s - d / (1 + d) <= 0)
    assert unsat(s >= 1, d > 0, s * d >= 1 + d, 1 / s - d / (1 + d) > 0)


@pytest.mark.parametrize("delta,s", [(F(0), F(2)), (F(1, 2), F(2)), (F(1), F(3, 2)), (F(2), F(5, 4))])
def test_cor2prime_failing_indices_and_randomized_mixture(delta, s):
    k = 12
    D = F(k)
    L = D / (1 + delta) - 1
    theta = D - L
    bound_fail = k - math.floor(s * theta)
    assert bound_fail > 0
    names = ["edf", "fifo", "lifo", "srpt", "ps", "random_static", "lazy"]
    total_fail = 0
    for name in names:
        fails = 0
        for i in range(k):
            ts = [mk_chain(j, 0, D, [(GPU, 1), (TOOL, L if j == i else 0)]) for j in range(k)]
            if not simulate(ts, policies(1, s, ts, seed=3)[name]).all_met(ts):
                fails += 1
        assert fails >= bound_fail, name
        total_fail += fails
    # a randomized algorithm = uniform mixture of these deterministic ones; exact miss probability
    assert F(total_fail, k * len(names)) >= 1 - F(math.floor(s * theta), k)


# =============================================================================================== Theorem 3
@pytest.mark.parametrize("s", [F(1), F(3, 2), F(2), F(4)])
def test_thm3_online_zero_offline_at_least_floor(s):
    k = 14
    D = F(k)
    for name in ["edf", "fifo", "lifo", "srpt", "ps", "random_static", "lazy"]:
        tasks = [mk_chain(i, 0, D, [(GPU, 1), (TOOL, D + 1)]) for i in range(k)]
        run = simulate(tasks, policies(1, s, tasks, seed=11)[name], tool_oracle=Thm3Adversary(D))
        real = with_tools(tasks, {(t.tid, 1): run.tool_dur.get((t.tid, 1), ZERO) for t in tasks})
        assert run.value(real) == 0
        issue = sorted(range(k), key=lambda i: (run.node_end.get((i, 0), F(10 ** 9)), i))
        g = math.floor(F(k) / s - F(1, 2))
        chosen = issue[k - g:] if g > 0 else []
        sub = [real[i] for i in chosen]
        pcs = [Piece((tid, 0), 0, F(l), F(l + 1)) for l, tid in enumerate(chosen)]
        assert validate_schedule(sub, 1, ONE, pcs)
        assert len(chosen) >= g


def test_thm3_z3_algebra():
    s, k, g, l = z3.Reals("s k g l")
    assert unsat(s >= 1, k >= 1, g <= k / s - z3.Q(1, 2), l >= 1, l <= g, l > (k - g + l) / s - z3.Q(1, 2))
    assert sat(s >= 1, k >= 1, l >= 1, l <= g, l > (k - g + l) / s - z3.Q(1, 2))  # g-bound is needed


# =============================================================================================== Theorem 4
def _thm4_case(seed):
    rng = random.Random(seed)
    m = rng.randint(1, 3)
    delta = rng.choice([F(1, 4), F(1, 3), F(1, 2), F(1), F(2), F(3)])
    common = rng.random() < 0.5
    ts, pcs = planted(rng, rng.randint(1, 6), m, delta, common=common, idle_prob=rng.choice([0, 0.3]),
                      p_tool=rng.choice([0.3, 0.6]), tmax=rng.choice([3, 12]), zero_tools=rng.random() < 0.3)
    return rng, m, delta, ts, pcs


@pytest.mark.parametrize("seed", range(1200))
def test_thm4_planted_at_exact_threshold(seed):
    rng, m, delta, ts, pcs = _thm4_case(seed)
    assert validate_schedule(ts, m, ONE, pcs)
    assert all(slack_ok(t, delta) for t in ts) and release_order_ok(ts)
    assert simulate(ts, edf(m, (1 + delta) / delta), check=(m, (1 + delta) / delta)).all_met(ts)


@pytest.mark.parametrize("seed", range(40))
def test_thm4_minimum_slack_random_instances_exact_offline(seed):
    """Not planted: every deadline is exactly r + (1 + delta) C; offline feasibility decided exactly (LP)."""
    rng = random.Random(7000 + seed)
    m = rng.choice([1, 2])
    delta = rng.choice([F(1, 2), F(1), F(2)])
    ts = [random_task(rng, i, ZERO, n=(1, 3), p_tool=0.5) for i in range(rng.randint(2, 2 + m))]
    for t in ts:
        t.deadline = (1 + delta) * crit_path(t)
    if offline_mm_lp(ts, m)[0]:
        assert simulate(ts, edf(m, (1 + delta) / delta)).all_met(ts)


def _thm4_climb(seed, speed_of_delta, iters=40):
    """Lateness-maximizing local search over planted instances (deadlines from random offline schedules)."""
    def score(tasks, m, delta, s, rng):
        best = None
        for k in range(4):
            ts = [t.copy() for t in tasks]
            run = random_feasible_run(ts, m, random.Random(rng.random()), idle_prob=0.25 if k % 2 else 0)
            for t in ts:
                t.deadline = max(run.task_end[t.tid], (1 + delta) * crit_path(t))
            r = simulate(ts, edf(m, s))
            sc = max((r.task_end[t.tid] - t.deadline) / t.deadline for t in ts)
            best = sc if best is None else max(best, sc)
        return best

    rng = random.Random(seed)
    m, delta = rng.choice([1, 2]), rng.choice([F(1, 2), F(1)])
    s = speed_of_delta(delta)
    tasks = [random_task(rng, i, ZERO, n=(1, 4), p_tool=0.6) for i in range(3)]
    cur = score(tasks, m, delta, s, rng)
    for _ in range(iters):
        cand = [t.copy() for t in tasks]
        i = rng.randrange(len(cand))
        nodes = list(cand[i].nodes)
        v = rng.randrange(len(nodes))
        nd = nodes[v]
        new = F(rng.choice([1, 2, 4, 8, 16]), rng.choice([1, 4, 8]))
        nodes[v] = Node(nd.kind, new if nd.kind == TOOL else max(F(1, 8), new), nd.preds)
        cand[i] = cand[i].copy(nodes=tuple(nodes))
        sc = score(cand, m, delta, s, rng)
        if sc >= cur:
            tasks, cur = cand, sc
    return cur


@pytest.mark.parametrize("seed", range(15))
def test_thm4_hill_climb_finds_nothing_at_threshold(seed):
    assert _thm4_climb(seed, lambda d: (1 + d) / d) <= 0


def test_thm4_hill_climb_is_not_vacuous():
    """The same search at speed 1 does find EDF failures (so 'nothing found at threshold' is informative).
    A larger scratch run (300 seeds x 300 steps) found 0 failures at the threshold and 24 at 0.95x threshold."""
    assert any(_thm4_climb(seed, lambda d: ONE, iters=60) > 0 for seed in range(12))


def test_thm4_proof_counting_step_gap_and_repair():
    """The written step (iii) uses 'k is unfinished, so m s |Y| < W(S)'. If k's unfinished node is a TOOL, k's GPU
    work may be complete and only '<=' follows; then the written algebra is satisfiable at s = (1+d)/d exactly.
    Repair: GPU nodes have p > 0, so a path containing one gives p/s < p strictly; a tool-only path gives
    d_k - r_k <= C_k, contradicting slack. Both branches are UNSAT; dropping p > 0 or d > 0 reopens them."""
    s, d, C, D, Y, T, P, m, W, Wd = z3.Reals("s d C D Y T P m W Wd")
    common = [d > 0, s == (1 + d) / d, C >= 0, D >= (1 + d) * C, D > 0, m >= 1, W <= m * D, Wd >= m * s * Y]
    # as written, but with the strict '<' replaced by the justified '<=': satisfiable (gap)
    assert sat(*common, Y >= D - C, Wd <= W)
    # repaired branch 1: path contains a GPU node (P > 0)
    path = [T >= 0, T + P <= C, D <= T + P / s + Y]
    assert unsat(*common, *path, P > 0, Wd <= W)
    assert sat(*common, *path, P >= 0, Wd <= W)  # p > 0 is needed
    # repaired branch 2: tool-only path
    assert unsat(*common, T <= C, D <= T)
    assert sat(d >= 0, s == 1, C >= 0, D >= (1 + d) * C, D > 0, T <= C, D <= T, d == 0)  # delta > 0 needed


def test_thm4_release_condition_is_needed_r4():
    ts = r4_instance()
    assert all(slack_ok(t, 1) for t in ts) and not release_order_ok(ts)
    for s in [F(2), F(21, 10), F(9, 4), F(23, 10), F(12, 5)]:
        r = simulate(ts, edf(1, s))
        assert not r.all_met(ts) and r.task_end[2] > ts[2].deadline
    assert simulate(ts, edf(1, F(9, 4))).task_end[2] == F(734, 9)
    assert simulate(ts, edf(1, F(5, 2))).all_met(ts)  # this member is met from 5/2 on
    assert offline_1m_by_orders(ts) is not None and offline_mm_lp(ts, 1)[0]
    assert simulate(ts, priority_policy(1, F(9, 4), make_cpedf_key(ts))).all_met(ts)


def test_r4_member_at_5_2_and_c4_boundary():
    ts = r4_family(F(5, 2), 11, F(1, 100), F(1099, 100), F(197, 10), 77, 55, F(79, 2))
    assert all(slack_ok(t, 1) for t in ts)
    for s in [F(2), F(9, 4), F(12, 5), F(5, 2)]:
        assert not simulate(ts, edf(1, s)).all_met(ts)
    assert simulate(ts, edf(1, F(13, 5))).all_met(ts)
    assert offline_1m_by_orders(ts) is not None and offline_mm_lp(ts, 1)[0]
    s, y, D = z3.Reals("s y D")
    assert unsat(D > 0, y > 0, s > 0, s * s - s - 4 >= 0, y <= D / (s + 1), (D + y) / s > D / 2)


def test_r5_cpedf_exact_threshold_321_over_221():
    ts = r5_instance()
    key = make_cpedf_key(ts)
    for s in [F(1), F(6, 5), F(13, 10), F(7, 5), F(29, 20), F(321, 221) - F(1, 10 ** 9)]:
        assert not simulate(ts, priority_policy(1, s, key)).all_met(ts)
    for s in [F(321, 221), F(3, 2), F(2)]:
        assert simulate(ts, priority_policy(1, s, key)).all_met(ts)
    for s in [F(1), F(6, 5), F(29, 20)]:
        assert simulate(ts, edf(1, s)).all_met(ts)
    assert offline_1m_by_orders(ts) is not None
    # 'zero slack' is approximate: T0 has slack 1/710
    assert (ts[0].deadline - crit_path(ts[0])) / crit_path(ts[0]) == F(1, 710)
    a, c, u, s = z3.Reals("a c u s")
    assert unsat(a > 0, c > a, u >= 0, s >= z3.Q(3, 2), a > (a + c + u) * (s - 1))


def test_r5_general_release_search_instance():
    ts = [mk_chain(0, F(5, 2), F(45, 4), [(GPU, 2), (TOOL, F(5, 2)), (GPU, F(15, 4))]),
          mk_chain(1, 0, 1, [(GPU, 1)]),
          mk_chain(2, 2, 12, [(GPU, 1), (TOOL, F(9, 4)), (GPU, 2), (TOOL, F(3, 2)), (GPU, F(1, 2))])]
    assert not simulate(ts, priority_policy(1, F(6, 5), make_cpedf_key(ts))).all_met(ts)
    assert offline_1m_by_orders(ts) is not None


# =============================================================================================== Prop 5 / Cor 5'
def _single_gpu_task(rng, tid, side_tools=True):
    nodes = []
    pre = rng.randint(0, 2)
    for _ in range(pre):
        nodes.append(Node(TOOL, F(rng.randint(0, 6), 2)))
    nodes.append(Node(GPU, F(rng.randint(1, 6), 2), tuple(range(pre))))
    g = pre
    for _ in range(rng.randint(0, 2)):
        nodes.append(Node(TOOL, F(rng.randint(0, 6), 2), (g,)))
    if side_tools and rng.random() < 0.3:
        nodes.append(Node(TOOL, F(rng.randint(0, 12), 2)))  # incomparable to g
    t = Task(tid, F(rng.randint(0, 6)), ZERO, tuple(nodes))
    t.deadline = t.release + crit_path(t) + F(rng.randint(0, 6), 2)
    return t, g


@pytest.mark.parametrize("seed", range(60))
def test_prop5_cpedf_optimal_one_machine(seed):
    rng = random.Random(seed)
    ts = [_single_gpu_task(rng, i)[0] for i in range(rng.randint(1, 4))]
    opt = offline_1m_by_orders(ts) is not None
    assert simulate(ts, priority_policy(1, 1, make_cpedf_key(ts))).all_met(ts) == opt


@pytest.mark.parametrize("seed", range(20))
def test_prop5_window_reduction_two_machines(seed):
    rng = random.Random(900 + seed)
    pairs = [_single_gpu_task(rng, i, side_tools=False) for i in range(rng.randint(2, 4))]
    ts = [p[0] for p in pairs]
    jobs = []
    for t, g in pairs:
        I = crit_path(t.copy(nodes=tuple(t.nodes[:g])) ) if g else ZERO
        O = tails(t)[g]
        jobs.append(Task(t.tid, t.release + I, t.deadline - O, (Node(GPU, t.nodes[g].size),)))
    ok_jobs = all(j.deadline - j.release >= j.nodes[0].size for j in jobs) and offline_mm_lp(jobs, 2)[0]
    assert offline_mm_lp(ts, 2)[0] == ok_jobs


def test_cor5_slack_translation_z3():
    W, I, O, p, d = z3.Reals("W I O p d")
    assert unsat(d >= 0, I >= 0, O >= 0, p > 0, W >= (1 + d) * (I + p + O), W - I - O < (1 + d) * p)
    assert sat(d >= 0, I >= 0, O < 0, p > 0, W >= (1 + d) * (I + p + O), W - I - O < (1 + d) * p)


# =============================================================================================== Theorem 6 / 6b
@pytest.mark.parametrize("seed", range(40))
def test_thm6_engine_lift_end_to_end(seed):
    """Fluid engine, decode-only, unlimited KV. (1) a random unit-engine schedule (any batch sizes) is turned into
    an exact M_m(1/a) schedule (Lemma 2(ii) on DAG tasks); (2) EDF on m machines at s = (1+d)/d meets every
    deadline; (3) the engine at lam = 2s following that simulation finishes every node no later."""
    rng = random.Random(seed)
    c = F(1)
    m = rng.randint(1, 3)
    a = c * m
    delta = rng.choice([F(1, 2), F(1), F(2)])
    tok = [random_task(rng, i, ZERO, n=(1, 4), p_tool=0.4, wmax=3, den=1) for i in range(rng.randint(1, 5))]
    # unit engine: random batches of random size (possibly > m)
    def batch(now, ready, ctx, _r=random.Random(seed)):
        return [k for k in ready if _r.random() < 0.7] or ready[:1]
    run_e = simulate(tok, engine_policy(a, c, 1, batch), record=True)
    p_tasks = [t.copy(nodes=tuple(Node(nd.kind, nd.size * a if nd.kind == GPU else nd.size, nd.preds)
                                  for nd in t.nodes)) for t in tok]
    for tt, pt in zip(tok, p_tasks):
        pt.deadline = tt.deadline = max(run_e.task_end[tt.tid], (1 + delta) * crit_path(pt))
    pcs = []
    for t0, t1, rates in run_e.segments:
        pcs.extend(mcnaughton({k: r * (t1 - t0) * a for k, r in rates.items()}, m, t0, t1))
    assert validate_schedule(p_tasks, m, ONE, pcs)
    s = (1 + delta) / delta
    sim = simulate(p_tasks, edf(m, s), record=True)
    assert sim.all_met(p_tasks)
    segs = [(t0, t1, frozenset(rates)) for t0, t1, rates in sim.segments]
    work = {(t.tid, v): nd.size for t in tok for v, nd in enumerate(t.nodes) if nd.kind == GPU and nd.size > 0}
    eng = fluid_engine_following(segs, work, a, c, lam=2 * s)
    for k in work:
        assert eng[k] <= sim.node_end[k]


@pytest.mark.parametrize("lam", [F(1), F(2), F(4)])
def test_thm6b_engine_lower_bound(lam):
    a, c, n = F(2), F(1), 2
    m = int(a / c)
    k = math.floor(lam * (m + 1)) + 1
    D = n * (a + c) + n * (a + c * (k - 1))
    L = D - n * (a + c)
    base = [mk_chain(i, 0, D, [(GPU, n), (TOOL, L)]) for i in range(k)]
    batchers = {"all": None, "m": lambda now, r, ctx: r[:m], "one": lambda now, r, ctx: r[:1],
                "rev": lambda now, r, ctx: r[::-1][:m + 1]}
    for name, b in batchers.items():
        tasks = [t.copy() for t in base]
        run = simulate(tasks, engine_policy(a, c, lam, b), tool_oracle=LastIssuedLong(k, L))
        real = with_tools(tasks, {(t.tid, 1): run.tool_dur.get((t.tid, 1), ZERO) for t in tasks})
        assert not run.all_met(real), name
        longi = [t.tid for t in real if t.nodes[1].size > 0][0]
        off = simulate(real, engine_policy(a, c, 1, lambda now, r, ctx: [x for x in r if x[0] == longi] or r))
        assert off.all_met(real)
    assert math.floor(4 * (F(278, 10) + F(543, 1000)) / F(543, 1000)) + 1 == 209  # doc: k = 209 at lam = 4


# =============================================================================================== Theorem 7 / Props 8-9
@pytest.mark.parametrize("seed", range(300))
def test_thm7_evict_on_suspend(seed):
    rng = random.Random(seed)
    m = rng.randint(1, 3)
    delta = rng.choice([F(1, 2), F(1), F(2)])
    beta = rng.choice([F(1, 4), F(1), F(3)])
    ts, _ = planted(rng, rng.randint(1, 5), m, delta, common=rng.random() < 0.5, p_tool=0.5)
    inf = evict_inflate(ts, beta)
    assert all(slack_ok(t, delta, speed=1 + beta) for t in inf)
    assert simulate(inf, edf(m, (1 + beta) * (1 + delta) / delta)).all_met(inf)


@pytest.mark.parametrize("seed", range(12))
def test_thm7_inflated_instance_feasible_at_offline_speed_1_plus_beta(seed):
    rng = random.Random(40 + seed)
    beta = rng.choice([F(1, 2), F(1)])
    ts, _ = planted(rng, rng.randint(2, 3), 1, F(1), n=(1, 3), p_tool=0.5)
    assert offline_mm_lp(evict_inflate(ts, beta), 1, speed=1 + beta)[0]


def test_prop8_keep_kv_edf_peak():
    k = 8
    for s in [F(1), F(2), F(4)]:
        ts = [mk_chain(i, 0, 12 * k, [(GPU, 1), (TOOL, 10), (GPU, 1)]) for i in range(k)]
        r = simulate(ts, edf(1, s))
        assert r.all_met(ts) and peak_held_kv(r, ts) == k
    pcs = [Piece((i, v), 0, F(12 * i + (0 if v == 0 else 11)), F(12 * i + (1 if v == 0 else 12)))
           for i in range(k) for v in (0, 2)]
    ts = [mk_chain(i, 0, 12 * k, [(GPU, 1), (TOOL, 10), (GPU, 1)]) for i in range(k)]
    assert validate_schedule(ts, 1, ONE, pcs)  # sequential offline schedule, one task in flight at a time


def test_prop9_never_evict_misses_at_every_speed():
    for s in [F(1), F(2), F(10), F(1000)]:
        A = mk_chain(0, 0, 100, [(GPU, 1), (TOOL, 5), (GPU, 1)])
        endA = simulate([A], edf(1, s)).task_end[0]
        assert endA == 5 + 2 / s and endA + 1 / s > F(7, 2)          # never-evict: B waits for A
        # evict A at time 2: B runs at once; A's resumed node pays 1 unit of recompute
        A2 = mk_chain(0, 0, 100, [(GPU, 1), (TOOL, 5), (GPU, 2)])
        Bt = mk_chain(1, 2, F(7, 2), [(GPU, 1)])
        r = simulate([A2, Bt], edf(1, s))
        assert r.all_met([A2, Bt])
    x = z3.Real("x")
    assert unsat(x > 0, 5 + 3 / x <= z3.Q(7, 2))


# =============================================================================================== CROSS-CHECKS
# Only here are the project's own modules imported, to compare with the independent implementation.
@pytest.mark.parametrize("seed", range(30))
def test_crosscheck_project_offline_solver_and_simulator(seed):
    op = pytest.importorskip("deadline_agent.open_problem")
    rng = random.Random(3000 + seed)
    mine = [random_task(rng, i, F(rng.randint(0, 4)), n=(1, 3), p_tool=0.5) for i in range(rng.randint(1, 3))]
    for t in mine:
        t.deadline = t.release + crit_path(t) + F(rng.randint(0, 6), 2)
    theirs = []
    for t in mine:
        nodes = {f"v{v}": op.Node(f"v{v}", op.GPU if nd.kind == GPU else op.TOOL, nd.size,
                                  tuple(f"v{u}" for u in nd.preds)) for v, nd in enumerate(t.nodes)}
        theirs.append(op.Task(t.tid, t.release, t.deadline, nodes))
    assert op.offline_feasible_1m(theirs)[0] == (offline_1m_by_orders(mine) is not None)
    for s in [F(1), F(3, 2)]:
        a = op.simulate(theirs, "edf", 1, s)
        b = simulate(mine, edf(1, s))
        assert a.finish == b.task_end


def test_crosscheck_project_r4_r5_constructions_match_doc_text():
    op = pytest.importorskip("deadline_agent.open_problem")
    theirs, s, delta = op.r4_general_release_counterexample()
    mine = r4_instance()
    for a, b in zip(theirs, mine):
        assert (a.release, a.deadline) == (b.release, b.deadline)
        assert [n.size for n in a.nodes.values()] == [n.size for n in b.nodes]
    t5 = op.cpedf_two_task_family()
    for a, b in zip(t5, r5_instance()):
        assert a.deadline == b.deadline and [n.size for n in a.nodes.values()] == [n.size for n in b.nodes]
