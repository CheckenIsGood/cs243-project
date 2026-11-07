"""CPU tests of the scheduler's planning logic (codesign.vllm_sched.plan_step), no vLLM needed.

They check the engine rules of docs/vllm_codesign_spec.md against the theory:
- E1: admission is the longest priority prefix that fits, with a head-of-line stop; running requests outside it
  are kicked; Lemma D0 (one arrival kicks < l+_y + L_max) holds on random instances;
- E2: one prefill at a time, top-m decoders;
- E4: preloads never take reserved memory or the prefill slot.
"""

import random

from codesign.vllm_sched import Req, blocks_for, plan_step


def R(rid, prio, res, running=False, prefill=True, preload=False, arrival=None):
    return Req(rid, prio, float(prio if arrival is None else arrival), res, running, prefill, preload)


def test_blocks_for_rounds_up():
    assert blocks_for(1, 16) == 1 and blocks_for(16, 16) == 1 and blocks_for(17, 16) == 2 and blocks_for(0, 16) == 1


def test_prefix_admission_with_head_of_line_stop():
    # priorities 1,2,3; sizes 6,5,1; capacity 10: admit 1 (6), stop at 2 (6+5>10) even though 3 would fit
    p = plan_step([R("a", 1, 6), R("b", 2, 5), R("c", 3, 1)], 10)
    assert p.admitted == ["a"] and p.hide_waiting == ["b", "c"] and p.kick == [] and p.reserved_blocks == 6


def test_higher_priority_arrival_kicks_lower_priority_running():
    # running low-priority request holds 8 of 10 blocks; a priority-0 arrival of 5 blocks displaces it
    p = plan_step([R("low", 5, 8, running=True, prefill=False), R("new", 0, 5)], 10)
    assert p.admitted == ["new"] and p.kick == ["low"]


def test_stock_admission_never_kicks():
    p = plan_step([R("low", 5, 8, running=True, prefill=False), R("new", 0, 5)], 10, admission="stock")
    assert p.kick == [] and p.hide_waiting == []


def test_completions_only_extend_the_prefix():
    """Removing an admitted request (a completion) never kicks anyone (Lemma D0, last step)."""
    rng = random.Random(0)
    for _ in range(500):
        n = rng.randint(1, 8)
        reqs = [R(f"r{i}", rng.randint(0, 5), rng.randint(1, 6), arrival=i) for i in range(n)]
        cap = rng.randint(3, 20)
        adm = plan_step(reqs, cap).admitted
        running = [Req(r.rid, r.priority, r.arrival, r.reserve_blocks, r.rid in adm, False) for r in reqs]
        if not adm:
            continue
        gone = rng.choice(adm)
        p2 = plan_step([r for r in running if r.rid != gone], cap)
        assert p2.kick == []
        assert set(adm) - {gone} <= set(p2.admitted)


def test_lemma_D0_one_arrival_kicks_less_than_its_size_plus_Lmax():
    rng = random.Random(1)
    worst = 0.0
    for _ in range(3000):
        n = rng.randint(1, 10)
        Lmax = rng.randint(1, 8)
        cap = rng.randint(Lmax, 4 * Lmax + 4)
        reqs = [R(f"r{i}", rng.randint(0, 9), rng.randint(1, Lmax), arrival=i) for i in range(n)]
        adm = set(plan_step(reqs, cap).admitted)
        running = [Req(r.rid, r.priority, r.arrival, r.reserve_blocks, r.rid in adm, False) for r in reqs]
        y = R("y", rng.randint(0, 9), rng.randint(1, Lmax), arrival=n)
        p = plan_step(running + [y], cap)
        kicked = sum(r.reserve_blocks for r in running if r.rid in p.kick)
        assert kicked < y.reserve_blocks + Lmax
        worst = max(worst, kicked / (y.reserve_blocks + Lmax))
    assert worst > 0.5  # the bound is exercised, not vacuous


def test_prefill_top1_keeps_only_the_highest_priority_prefill():
    reqs = [R("a", 1, 2, running=True, prefill=True), R("b", 2, 2, running=True, prefill=True), R("c", 3, 2),
            R("d", 0, 2, running=True, prefill=False)]
    p = plan_step(reqs, 100, prefill_top1=True)
    assert set(p.hide_running) == {"b"} and p.hide_waiting == ["c"]  # "a" prefills; decoder "d" untouched


def test_decode_top_m():
    reqs = [R(f"d{i}", i, 1, running=True, prefill=False) for i in range(5)]
    p = plan_step(reqs, 100, decode_top_m=2)
    assert p.hide_running == ["d2", "d3", "d4"]


def test_preloads_wait_while_an_admitted_request_prefills_and_never_take_reserved_memory():
    reqs = [R("a", 1, 6, running=True, prefill=True), R("pre1", 9, 3, preload=True), R("pre2", 9, 3, preload=True,
                                                                                         arrival=10)]
    p = plan_step(reqs, 10, preload_gate=True)
    assert p.admitted == ["a"]
    assert set(p.hide_waiting) == {"pre1", "pre2"}  # busy server: both wait; pre2 also does not fit (6+3+3>10)
    # once "a" decodes, the preload that fits runs and the other one waits
    reqs[0] = R("a", 1, 6, running=True, prefill=False)
    p = plan_step(reqs, 10, preload_gate=True)
    assert p.hide_waiting == ["pre2"] and "pre1" not in p.hide_waiting
    # a running preload is kicked when a new admitted request needs its memory
    reqs = [R("a", 1, 6, running=True, prefill=False), R("pre1", 9, 3, running=True, prefill=False, preload=True),
            R("b", 2, 3)]
    p = plan_step(reqs, 10, preload_gate=True)
    assert p.admitted == ["a", "b"] and p.kick == ["pre1"]


def test_preloads_are_not_counted_in_admission():
    reqs = [R("pre", 0, 9, running=True, prefill=False, preload=True), R("a", 1, 9)]
    p = plan_step(reqs, 10, preload_gate=True)
    assert p.admitted == ["a"] and p.kick == ["pre"]


def test_matches_the_theory_simulators_admission_rule():
    """Same prefix as stochastic_theory.sim_hybrid's recompute_admitted (sorted by key, stop at first misfit)."""
    rng = random.Random(2)
    for _ in range(1000):
        n = rng.randint(1, 9)
        cap = rng.randint(1, 30)
        reqs = [R(f"r{i}", rng.randint(0, 6), rng.randint(1, 8), arrival=rng.random()) for i in range(n)]
        ref, used = [], 0
        for r in sorted(reqs, key=lambda r: (r.priority, r.arrival, r.rid)):
            if used + r.reserve_blocks > cap:
                break
            ref.append(r.rid)
            used += r.reserve_blocks
        assert plan_step(reqs, cap).admitted == ref
