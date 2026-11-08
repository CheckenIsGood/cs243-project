"""Machine checks for docs/preload_theory.md. Exact rational simulation (fractions.Fraction) for every theorem
about the work model; Z3 for the algebra; hypothesis for random instances; the data checks read
results/real/preload_contention (skipped if absent)."""

from __future__ import annotations

import itertools
import random
from fractions import Fraction as F
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from deadline_agent.preload_theory import (
    Calib, EngineCfg, TaskIn, TaskSpec, cex_amplification, cex_endogenous_makespan, cex_fcfs_on_per_task,
    cex_low_endogenous, cex_ps_on_per_task, cumulative_at, fcfs_harm_bound, gap_ci, grid_tasks, load_contention,
    make_tasks, makespan, measured_savings, microbench_fit, predict_fcfs, predict_ps, reflection_departures,
    search_dominance, simulate_engine, simulate_work, workload_moments,
)

ROOT = Path(__file__).resolve().parents[1]
HAS_DATA = (ROOT / "results/real/preload_contention/tasks_r0.6_fcfs_pre-on.csv").exists()
SET = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])

frac = st.integers(0, 6).map(F)
small = st.integers(0, 3).map(F)


@st.composite
def task(draw, pos_f=False):
    n = draw(st.integers(1, 3))
    return TaskSpec(draw(frac), draw(st.integers(0, 2).map(F)), tuple(draw(frac) for _ in range(n)),
                    tuple(draw(small) for _ in range(n)), draw(st.integers(1 if pos_f else 0, 2).map(F)))


instances = st.lists(task(), min_size=1, max_size=4)
eps_s = st.sampled_from([F(0), F(1, 4), F(1, 2)])


def last_piece(tk: TaskSpec):
    i = max(range(tk.n), key=lambda i: (tk.D[i], i))
    return tk.w[i]


# ------------------------------------------------------------------------------------------- Lemma 1 (reflection)
@SET
@given(instances, st.sampled_from(["fcfs", "ps"]), st.sampled_from(["off", "on", "low"]), eps_s)
def test_lemma1_reflection_formula_equals_simulated_cumulative_work(inst, disc, mode, eps):
    res = simulate_work(inst, mode, disc, eps, "exo", merge=False)
    for t, v in res.breakpoints:
        assert reflection_departures(res.arrivals, t) == v


# --------------------------------------------------------------------------------------- Lemma 2 (size monotone)
@SET
@given(st.lists(st.tuples(frac, st.integers(0, 4)), min_size=1, max_size=5), st.data(),
       st.sampled_from(["fcfs", "ps"]))
def test_lemma2_completion_times_monotone_in_job_sizes(jobs, data, disc):
    """Single-job tasks (n=1, P=0 exo, w=0): the final job of size f is released at r + D. Shrinking any sizes
    never makes any completion later, under FCFS and under PS."""
    big = [TaskSpec(F(0), F(0), (r,), (F(0),), F(x)) for r, x in jobs]
    cut = [TaskSpec(t.r, t.P, t.D, t.w, F(data.draw(st.integers(0, int(t.f))))) for t in big]
    cb = simulate_work(big, "off", disc).completion
    cc = simulate_work(cut, "off", disc).completion
    assert all(a <= b for a, b in zip(cc, cb))


# --------------------------------------------------------------------------------- Theorem 1 (aggregate, exogenous)
@SET
@given(instances, st.sampled_from(["fcfs", "ps"]), st.sampled_from(["exo", "top"]), eps_s, st.booleans(),
       st.sampled_from(["on", "low"]))
def test_thm1_cumulative_work_and_makespan(inst, disc, pm, eps, merge, mode):
    off = simulate_work(inst, "off", disc, 0, pm)
    on = simulate_work(inst, mode, disc, eps, pm, merge)
    ts = sorted({t for t, _ in off.breakpoints} | {t for t, _ in on.breakpoints})
    assert all(cumulative_at(on.breakpoints, t) >= cumulative_at(off.breakpoints, t) for t in ts)
    N = sum(len(p) for p in on.piece_release)
    assert makespan(on) <= makespan(off) + eps * N
    if eps == 0:
        assert makespan(on) <= makespan(off)


def test_thm1_makespan_bound_is_attained():
    """C_on = C_off + eps*N when there is no idle time to absorb the overhead: B keeps the server busy on [0, 10],
    A's piece (returned at 1) is not merged away (merge=False), so its overhead is spent."""
    tk = [TaskSpec(F(0), F(0), (F(1), F(2)), (F(2), F(1)), F(1)), TaskSpec(F(0), F(0), (F(0),), (F(0),), F(10))]
    e = F(1, 2)
    assert makespan(simulate_work(tk, "off", "fcfs")) == 14
    assert makespan(simulate_work(tk, "on", "fcfs", e, merge=False)) == 14 + e
    assert makespan(simulate_work(tk, "on", "fcfs", e, merge=True)) == 14  # merged before it started: no overhead


# -------------------------------------------------------------------------- Refutation 1 (endogenous plans)
@pytest.mark.parametrize("disc", ["fcfs", "ps"])
def test_ref1_endogenous_plans_preloading_can_increase_makespan_with_zero_overhead(disc):
    inst = cex_endogenous_makespan()
    off, on = simulate_work(inst, "off", disc, 0, "work"), simulate_work(inst, "on", disc, 0, "work")
    assert makespan(on) > makespan(off)
    if disc == "fcfs":
        assert (makespan(off), makespan(on)) == (F(121, 10), F(13))
    # the same instance with exogenous plans obeys Theorem 1
    assert makespan(simulate_work(inst, "on", disc, 0, "exo")) <= makespan(simulate_work(inst, "off", disc, 0, "exo"))


@pytest.mark.parametrize("delta", [F(1, 10), F(1, 100), F(1, 1000)])
def test_ref1_per_task_harm_unbounded_relative_to_preload_work(delta):
    inst = cex_amplification(delta)
    off, on = simulate_work(inst, "off", "fcfs", 0, "work"), simulate_work(inst, "on", "fcfs", 0, "work")
    harm = on.completion[1] - off.completion[1]
    assert harm >= 1 and harm / delta >= 1 / delta  # B is ~1 later because of a preload of work delta


# --------------------------------------------------------------------------- Theorem 2 (per-task harm, FCFS-on)
@SET
@given(instances, eps_s, st.booleans())
def test_thm2_fcfs_on_harm_bound(inst, eps, merge):
    off = simulate_work(inst, "off", "fcfs", 0, "exo")
    on = simulate_work(inst, "on", "fcfs", eps, "exo", merge)
    for j in range(len(inst)):
        assert on.completion[j] - off.completion[j] <= fcfs_harm_bound(inst, on, j, eps)


@pytest.mark.parametrize("eps", [F(0), F(1, 4)])
def test_thm2_tight_instance(eps):
    d = F(1, 10)
    inst = cex_fcfs_on_per_task(d)
    off, on = simulate_work(inst, "off", "fcfs", 0, "exo"), simulate_work(inst, "on", "fcfs", eps, "exo")
    harm = on.completion[0] - off.completion[0]
    assert harm == 2 + eps - d  # bound is 2 + eps: gap delta -> 0
    assert fcfs_harm_bound(inst, on, 0, eps) == 2 + eps


def test_thm2_per_task_dominance_fails_under_ps_too():
    inst = cex_ps_on_per_task()
    off, on = simulate_work(inst, "off", "ps", 0, "exo"), simulate_work(inst, "on", "ps", 0, "exo")
    assert on.completion[0] > off.completion[0] and on.completion[1] < off.completion[1]


@SET
@given(instances, eps_s)
def test_prop_fcfs_identity_saving_decomposition(inst, eps):
    """S_j = (W_j - F_j) + V_off(R_j) - V_on(R_j) exactly (FCFS, exogenous, no merge), where V is the unfinished
    work just before j's final is released and F_j = last piece + f_j."""
    off = simulate_work(inst, "off", "fcfs", 0, "exo")
    on = simulate_work(inst, "on", "fcfs", eps, "exo", merge=False)
    for j, tk in enumerate(inst):
        W = sum(tk.w, F(0)) + tk.f
        Fj = last_piece(tk) + tk.f
        lhs = off.completion[j] - on.completion[j]
        rhs = (W - Fj) + off.unfinished_at_R[j]["V"] - on.unfinished_at_R[j]["V"]
        assert lhs == rhs


@SET
@given(task(pos_f=True), eps_s)
def test_prop_isolated_task_saving(tk, eps):
    """One task alone: S = (preloaded work) - (own useful residual at R) exactly; = all preloaded work if every
    piece finishes before R."""
    off = simulate_work([tk], "off", "fcfs", 0, "exo")
    on = simulate_work([tk], "on", "fcfs", eps, "exo")
    pre = sum(tk.w, F(0)) - last_piece(tk)
    assert off.completion[0] - on.completion[0] == pre - on.unfinished_at_R[0]["own"]


# ---------------------------------------------------------------------- Theorem 3 (low priority: per-task dominance)
@SET
@given(instances, st.sampled_from(["fcfs", "ps"]), st.sampled_from(["exo", "top"]), eps_s)
def test_thm3_low_priority_preloads_never_delay_any_task(inst, disc, pm, eps):
    off = simulate_work(inst, "off", disc, 0, pm)
    low = simulate_work(inst, "low", disc, eps, pm)
    assert all(a <= b for a, b in zip(low.completion, off.completion))


def test_thm3_exhaustive_small_grid():
    """All unordered pairs of tasks from a grid (r, P in {0,1}; n in {1,2}; D in {0,2}; w in {0,1}; f = 1):
    80 task types, 3240 pairs, FCFS, exogenous plans."""
    pool = grid_tasks([0, 1], [0, 1], [0, 2], [0, 1], [1], [1, 2])
    assert len(pool) == 80
    r = search_dominance(pool, 2, ("off", "low"), "fcfs", "exo", F(1, 2))
    assert r["count"] == 3240 and r["worst_task"][0] <= 0
    r2 = search_dominance(pool, 2, ("off", "on"), "fcfs", "exo", 0)
    assert r2["worst_task"][0] > 0  # 'on' is not per-task dominant even on this grid


@pytest.mark.parametrize("disc,harm", [("fcfs", F(9, 2)), ("ps", F(1, 2))])
def test_ref2_low_priority_with_endogenous_plans(disc, harm):
    inst = cex_low_endogenous()
    off, low = simulate_work(inst, "off", disc, 0, "work"), simulate_work(inst, "low", disc, 0, "work")
    assert low.completion[2] - off.completion[2] == harm
    # with plans at top priority (exogenous plan completions) the anomaly disappears
    o2, l2 = simulate_work(inst, "off", disc, 0, "top"), simulate_work(inst, "low", disc, 0, "top")
    assert all(a <= b for a, b in zip(l2.completion, o2.completion))


@SET
@given(instances, st.sampled_from(["exo", "top"]), eps_s)
def test_thm3b_clairvoyant_jit_dominance(inst, pm, eps):
    off = simulate_work(inst, "off", "fcfs", 0, pm)
    jit = simulate_work(inst, "jit", "fcfs", eps, pm)
    for j in range(len(inst)):
        Rj = off.release_final[j]
        n_ahead = sum(len(jit.piece_release[k]) for k in range(len(inst)) if jit.release_final[k] <= Rj)
        assert jit.completion[j] <= off.completion[j] + eps * n_ahead


# ------------------------------------------------------------------------ Theorem 4 (priority insensitivity)
@SET
@given(instances)
def test_thm4_bracket_and_difference_bound(inst):
    res = {m: simulate_work(inst, m, "fcfs", 0, "exo") for m in ("on", "low", "jit")}
    for j, tk in enumerate(inst):
        V = res["on"].unfinished_at_R[j]["V"]
        Fj = last_piece(tk) + tk.f
        for m, r in res.items():
            u = r.unfinished_at_R[j]
            assert u["V"] == V  # the total backlog at R_j is the same in all three systems
            Rj = r.release_final[j]
            assert Rj + V - u["other_pre"] + Fj <= r.completion[j] <= Rj + V + Fj
        for a, b in itertools.combinations(res, 2):
            pi = max(res[a].unfinished_at_R[j]["other_pre"], res[b].unfinished_at_R[j]["other_pre"])
            assert abs(res[a].completion[j] - res[b].completion[j]) <= pi


def test_thm4_needs_exogenous_plans():
    """With top-priority plan work arriving after R_j, the upper bracket can fail (plans preempt the final)."""
    inst = [TaskSpec(F(2), F(0), (F(1), F(1)), (F(2), F(1)), F(2)), TaskSpec(F(5), F(2), (F(2), F(3)), (F(2), F(3)), F(1))]
    r = simulate_work(inst, "on", "fcfs", 0, "top")
    j = 0
    assert r.completion[j] > r.release_final[j] + r.unfinished_at_R[j]["V"] + last_piece(inst[j]) + inst[j].f


# ----------------------------------------------------------------------------------------------- Z3 algebra
z3 = pytest.importorskip("z3")


def _prove(claim, *hyps):
    s = z3.Solver()
    s.add(*hyps)
    s.add(z3.Not(claim))
    return s.check() == z3.unsat


def test_z3_thm6b_ps_breakeven():
    x, w, rho, lam, eps = z3.Reals("x w rho lam eps")
    hyp = [x > w, w > 0, rho >= 0, lam > 0, eps >= 0, rho + lam * eps < 1]
    gain = x / (1 - rho) > (x - w) / (1 - rho - lam * eps)
    assert _prove(gain == (eps < w * (1 - rho) / (lam * x)), *hyp)
    # hypothesis needed: without rho + lam*eps < 1 the equivalence fails
    assert not _prove(gain == (eps < w * (1 - rho) / (lam * x)), *hyp[:-1])


def test_z3_thm6c_capacity_loss():
    lam, s, k, eps = z3.Reals("lam s k eps")
    hyp = [s > 0, k > 0, eps > 0, lam > 0]
    # an arrival rate that is stable without preloading but unstable with it always exists
    assert _prove(z3.Exists([lam], z3.And(lam * s < 1, lam * (s + k * eps) >= 1)), *hyp)


def test_z3_thm4_bracket_implies_difference_bound():
    R, V, Fj, a, b, ca, cb = z3.Reals("R V F pa pb ca cb")
    hyp = [a >= 0, b >= 0, R + V - a + Fj <= ca, ca <= R + V + Fj, R + V - b + Fj <= cb, cb <= R + V + Fj]
    assert _prove(z3.And(ca - cb <= z3.If(a > b, a, b), cb - ca <= z3.If(a > b, a, b)), *hyp)


def test_z3_prop_fcfs_discriminant_algebra():
    """E[S|wide] = E[P - U] + D_w, E[S|narrow] = D_n, D_w <= D (own pieces only add backlog), D_n = D
    => gap <= E[P]."""
    P, U, Dw, Dn, D = z3.Reals("P U Dw Dn D")
    assert _prove(P - U + Dw - Dn <= P, U >= 0, Dw <= D, Dn == D)


# ------------------------------------------------------------------------------------------- engine model
def _single(n=8, delays=None, sizes=None):
    d = delays or tuple(1.0 + i for i in range(n))
    s = sizes or tuple([1400] * n)
    return [TaskIn(0.0, n, d, s, 127)]


def test_engine_isolated_saving_matches_preloaded_prefill_cost():
    cal = Calib()
    tasks = _single()
    off = simulate_engine(tasks, "off").latency[0]
    for mode in ("block", "nb"):
        on = simulate_engine(tasks, mode).latency[0]
        ctx, pre = cal.header, 0.0
        for s in tasks[0].sizes[:-1]:
            pre += cal.prefill_cost(s, ctx)
            ctx += s
        # saving = preloaded prefill + per-chunk step overheads of the cold final (5 extra chunk steps of a),
        # minus partial-block recompute; within 0.2 s of the preloaded prefill cost
        assert abs((off - on) - pre) < 0.2, (mode, off - on, pre)


def test_engine_isolated_low_equals_on():
    tasks = _single()
    assert abs(simulate_engine(tasks, "nb", False).latency[0] - simulate_engine(tasks, "nb", True).latency[0]) < 1e-9


def test_blocking_low_priority_chain_starves_but_nonblocking_does_not():
    """A wide task plus a stream of tasks whose long finals keep the prefill budget busy: the blocking chain at the
    lowest priority waits for idle budget before the final can even be sent; the non-blocking version cancels."""
    hog = [TaskIn(0.3 + 1.8 * i, 1, (0.0,), (9000,), 1) for i in range(40)]
    me = TaskIn(0.0, 8, tuple(0.5 + 0.1 * i for i in range(8)), tuple([1400] * 8), 20)
    tasks = [me] + hog
    blk = simulate_engine(tasks, "block", True).latency[0]
    nb = simulate_engine(tasks, "nb", True).latency[0]
    on = simulate_engine(tasks, "block", False).latency[0]
    assert blk > nb + 1.0 and blk > on + 1.0


def test_lemma_k_trajectory_checks_with_eviction():
    if not HAS_DATA:
        pytest.skip("no data")
    d = load_contention(ROOT)["0.3"]
    tasks = make_tasks(d["arrival"], d["width"], seed=1000)
    big = simulate_engine(tasks, "nb", cfg=EngineCfg(2048, kv_capacity=163904), horizon=3000)
    small = simulate_engine(tasks, "nb", cfg=EngineCfg(2048, kv_capacity=40000), horizon=3000)
    assert big.evicted_preloaded == 0 and big.kv_violations["necessary_flagged"] == 0
    assert small.evicted_preloaded > 0
    for o in (big, small):
        assert o.kv_violations["necessary_without_eviction"] == 0  # Lemma K (i) on the trajectory
        assert o.kv_violations["evicted_while_sufficient"] == 0  # Lemma K (ii) on the trajectory


# ----------------------------------------------------------------------------------------------- data checks
needs_data = pytest.mark.skipif(not HAS_DATA, reason="results/real/preload_contention not present")


@needs_data
def test_measured_savings_match_reported_numbers():
    m = measured_savings(load_contention(ROOT))
    got = [round(m[r]["all"][0], 2) for r in ("0.15", "0.3", "0.45", "0.6")]
    assert got == [1.20, 2.47, 4.74, 8.25]


@needs_data
def test_fcfs_work_model_rejected_by_width_discriminant():
    """Prop 5: under the FCFS work model, E[S|wide] - E[S|narrow] <= E[preloaded work | wide] (~1.7 s here). The
    measured gap's 95% bootstrap lower bound exceeds it at 0.45 and 0.6 tasks/s."""
    data = load_contention(ROOT)
    pre = workload_moments(1.0)["preloaded_per_task"]
    assert 1.4 < pre < 2.0
    for r in ("0.45", "0.6"):
        g, lo, hi = gap_ci(data, r)
        assert lo > pre, (r, g, lo, hi, pre)


@needs_data
def test_ps_formula_within_measured_ci_and_fcfs_formula_not():
    data = load_contention(ROOT)
    m = measured_savings(data)
    for r in ("0.15", "0.3", "0.45", "0.6"):
        q = m[r]["q_wide"]
        ps = predict_ps(float(r), q)["mean"]
        assert m[r]["all"][1] <= ps <= m[r]["all"][2], (r, ps, m[r]["all"])
    for r in ("0.45", "0.6"):  # FCFS heuristic misses the wide-task saving
        assert predict_fcfs(float(r), m[r]["q_wide"])["wide"] < m[r]["wide"][1]


@needs_data
def test_engine_reproduces_off_latency():
    data = load_contention(ROOT)
    for r, meas in (("0.15", 13.71), ("0.3", 16.52), ("0.6", 32.90)):
        d = data[r]
        lat = []
        for s in range(4):
            out = simulate_engine(make_tasks(d["arrival"], d["width"], seed=1000 + s), "off")
            lat.append(sum(out.latency) / len(out.latency))
        assert abs(sum(lat) / 4 - meas) / meas < 0.15, (r, lat)


@needs_data
def test_microbenchmark_saving_per_cached_token():
    fit = microbench_fit(ROOT)
    assert fit["n"] == 15 and 1.5e-4 < fit["slope"] < 1.9e-4 and abs(fit["intercept"]) < 0.1
