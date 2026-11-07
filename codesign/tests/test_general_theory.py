"""Verification of docs/general_theory.md: Theorem 1 and Lemmas 2-3.

Algebraic steps are proved with Z3 (unsatisfiability of the negation over the reals); combinatorial steps are
checked exhaustively on small instances and by randomized property tests (Hypothesis).
"""

import random

import pytest
import z3
from hypothesis import given, settings
from hypothesis import strategies as st

from deadline_agent.general_theory import (
    check_mcnaughton,
    engine_completion_times,
    knapsack_opt,
    machine_completion_times,
    mcnaughton,
    single_machine_on_time_value,
    tool_then_llm_opt,
)


def z3_proves(*constraints_and_negated_goal) -> bool:
    s = z3.Solver()
    s.add(*constraints_and_negated_goal)
    return s.check() == z3.unsat


# ------------------------------------------------------------------------------------------- Theorem 1
@pytest.mark.parametrize("seed", range(300))
def test_theorem1_restricted_problem_equals_knapsack(seed):
    """Common deadline D, batch size 1, no tools: max on-time value == Knapsack(capacity D)."""
    rng = random.Random(seed)
    n = rng.randint(1, 6)
    wts = [rng.randint(1, 9) for _ in range(n)]
    vals = [rng.randint(1, 20) for _ in range(n)]
    cap = rng.randint(0, sum(wts))
    assert single_machine_on_time_value(wts, vals, [cap] * n) == knapsack_opt(wts, vals, cap)


# ------------------------------------------------------------------------------------------- Lemma 2 (Z3)
a, c, B, m = z3.Reals("a c B m")


def test_lemma2_i_per_sequence_rate_at_least_half_single_rate_when_batch_le_a_over_c():
    # a,c>0, 0<=B, c*B<=a  =>  1/(a+cB) >= 1/(2a)
    assert z3_proves(a > 0, c > 0, B >= 0, c * B <= a, 1 / (a + c * B) < 1 / (2 * a))


def test_lemma2_ii_per_sequence_rate_at_most_single_rate():
    assert z3_proves(a > 0, c > 0, B >= 0, 1 / (a + c * B) > 1 / a)


def test_lemma2_ii_total_rate_below_m_machines_at_speed_1_over_a():
    # B>=1: B/(a+cB) < 1/c, and m = a/c machines at speed 1/a have total speed exactly 1/c
    assert z3_proves(a > 0, c > 0, B >= 1, B / (a + c * B) >= 1 / c)
    assert z3_proves(a > 0, c > 0, m == a / c, m * (1 / a) != 1 / c)


def test_lemma2_noninteger_m_floor_costs_at_most_factor_two_more_speed():
    # m = floor(a/c) >= a/c - 1, a/c >= 2  =>  m machines at speed 2/a have total speed >= 1/c (part ii),
    # and part (i) still holds because m <= a/c.
    assert z3_proves(a > 0, c > 0, a >= 2 * c, m >= a / c - 1, m * (2 / a) < 1 / c)


def test_lemma2_z3_catches_a_false_variant():
    # sanity: the bound fails without the batch condition c*B <= a (so Z3 is not vacuously returning unsat)
    assert not z3_proves(a > 0, c > 0, B >= 0, 1 / (a + c * B) < 1 / (2 * a))


# ------------------------------------------------------------------------------------------- McNaughton
@settings(max_examples=400, deadline=None)
@given(st.integers(1, 6), st.floats(0.1, 5.0), st.floats(0.5, 3.0), st.lists(st.floats(0, 1), min_size=1, max_size=12))
def test_mcnaughton_realizes_any_feasible_rate_vector(mm, s, length, fracs):
    rates = [f * s for f in fracs]
    total = sum(rates)
    if total > mm * s:  # scale into the feasible region
        rates = [x * (mm * s) / total for x in rates]
    check_mcnaughton(mcnaughton(rates, mm, s, length), rates, mm, s, length)


def test_mcnaughton_rejects_infeasible_vectors():
    with pytest.raises(ValueError):
        mcnaughton([1.1], 1, 1.0, 1.0)  # one job faster than a machine
    with pytest.raises(ValueError):
        mcnaughton([1.0, 1.0, 1.0], 2, 1.0, 1.0)  # more total work than machines


# ------------------------------------------------------------------------------------------- Lemma 2(i) by simulation
@pytest.mark.parametrize("seed", range(40))
def test_lemma2_i_engine_finishes_every_job_no_later_than_m_machines_at_speed_1_over_2a(seed):
    rng = random.Random(seed)
    a_, c_ = rng.uniform(0.01, 0.05), rng.uniform(0.0005, 0.005)
    mm = max(1, int(a_ / c_))
    n = rng.randint(1, 3 * mm + 2)
    work = {j: rng.uniform(1, 30) for j in range(n)}
    s = 1 / (2 * a_)
    # random valid m-machine schedule: segments, each running at most mm jobs
    sched, t = [], 0.0
    for _ in range(400):
        t += rng.uniform(0.05, 0.5)
        sched.append((t, frozenset(rng.sample(range(n), min(n, rng.randint(0, mm))))))
    tm = machine_completion_times(sched, work, s, dt=2e-3)
    te = engine_completion_times(sched, work, a_, c_, dt=2e-3)
    for j, t_machine in tm.items():
        assert j in te and te[j] <= t_machine + 4e-3, (j, te.get(j), t_machine)


# ------------------------------------------------------------------------------------------- Lemma 3
@pytest.mark.parametrize("seed", range(300))
def test_lemma3_eager_tool_issue_never_lowers_the_offline_optimum(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 5)
    tool = [rng.uniform(0, 5) for _ in range(n)]
    p = [rng.uniform(0.2, 3) for _ in range(n)]
    w = [rng.randint(1, 5) for _ in range(n)]
    d = [rng.uniform(1, 12) for _ in range(n)]
    late = [rng.uniform(0, 4) if rng.random() < 0.7 else 0.0 for _ in range(n)]
    assert tool_then_llm_opt(tool, [0.0] * n, p, w, d) >= tool_then_llm_opt(tool, late, p, w, d)


def test_lemma3_counterexample_when_tools_share_capacity():
    """Limit of Lemma 3 (documented): with ONE shared tool slot, the issue ORDER matters, so 'eager' is not a
    complete policy. Here issuing the long call first delays the short one past its deadline."""
    # two tasks, tool durations 5 and 1, LLM jobs 1 each, deadlines 7 and 2.5; one tool slot
    def opt_with_order(first):
        order = [first, 1 - first]
        tool, issue, t = [5.0, 1.0], [0.0, 0.0], 0.0
        for j in order:
            issue[j] = t
            t += tool[j]
        return tool_then_llm_opt(tool, issue, [1.0, 1.0], [1, 1], [7.0, 2.5])
    assert opt_with_order(1) == 2 and opt_with_order(0) == 1
