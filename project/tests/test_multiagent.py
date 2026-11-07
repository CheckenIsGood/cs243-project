"""Multi-tenant / multi-agent serving layer on the simulated batching engine (virtual clock)."""

import asyncio

import pytest

from deadline_agent.clock import now, run_virtual
from deadline_agent.datasets import fixture_task
from deadline_agent.engine_sim import SimBackend, SimEngine
from deadline_agent.model_client import FakeBackend
from deadline_agent.multiagent import AgentTask, CostEstimate, ServingLayer, make_workload, serve
from deadline_agent.tools import DelayModel

DELAY = DelayModel(0.5, 4.0, 0.3, 4.0, 12.0)


def test_engine_single_request_time_matches_the_step_model():
    async def go():
        e = SimEngine(a=0.03, c=0.001, prefill_s_per_tok=1e-4)
        sub, first, end = await e.submit(1000, 10)
        await e.aclose()
        return first - sub, end - sub
    ttft, total = run_virtual(go())
    assert ttft == pytest.approx(0.03 + 0.1)  # prefill step emits the first token
    assert total == pytest.approx(0.13 + 9 * (0.03 + 0.001))


def test_engine_batches_concurrent_requests_and_slows_each_one():
    async def go():
        e = SimEngine(a=0.03, c=0.01, prefill_s_per_tok=0.0)
        outs = await asyncio.gather(*(e.submit(1, 11) for _ in range(4)))
        await e.aclose()
        return outs, e.stats.max_batch
    outs, mb = run_virtual(go())
    assert mb == 4
    for sub, first, end in outs:  # 1 prefill step + 10 decode steps at batch 4
        assert end - sub == pytest.approx(0.03 + 10 * (0.03 + 0.04))


def test_engine_priority_policy_serves_lower_priority_value_first_when_capped():
    async def go():
        e = SimEngine(a=0.01, c=0.0, prefill_s_per_tok=0.0, max_seqs=1, policy="priority")
        blocker = asyncio.create_task(e.submit(1, 5, priority=0))
        await asyncio.sleep(0)
        order = []

        async def req(p):
            await e.submit(1, 3, priority=p)
            order.append(p)
        await asyncio.gather(req(5), req(1), req(3), blocker)
        await e.aclose()
        return order
    assert run_virtual(go()) == [1, 3, 5]


def test_engine_fcfs_ignores_priority():
    async def go():
        e = SimEngine(a=0.01, c=0.0, prefill_s_per_tok=0.0, max_seqs=1, policy="fcfs")
        order = []

        async def req(p, delay):
            await asyncio.sleep(delay)
            await e.submit(1, 3, priority=p)
            order.append(p)
        await asyncio.gather(req(5, 0.0), req(1, 0.001), req(3, 0.002))
        await e.aclose()
        return order
    assert run_virtual(go()) == [5, 1, 3]


def _run(policy, rate, duration=60.0, seed=0, max_seqs=256, preload=False, shapes=("fanout", "orchestrator")):
    task, _ = fixture_task()

    async def go():
        e = SimEngine(max_seqs=max_seqs, policy="fcfs" if policy.startswith("fcfs") else "priority")
        be = SimBackend(e)
        layer = ServingLayer(policy, CostEstimate(), slowdown=lambda: (e.a + e.c * len(e._running)) / (e.a + e.c))
        wl = make_workload([task], rate, duration, seed, shapes=shapes)
        res = await serve(wl, be, layer, DELAY, seed=1, preload=preload)
        await e.aclose()
        return res, e
    return run_virtual(go())


@pytest.mark.parametrize("policy", ["fcfs", "edf", "llf", "edf+ac", "llf+ac"])
def test_light_load_every_policy_finishes_every_task_on_time(policy):
    res, _ = _run(policy, rate=0.05, duration=200)
    assert res and all(r.status == "ontime" for r in res), [r.status for r in res]


def test_orchestrator_workers_are_batched_together():
    res, e = _run("edf", rate=0.05, duration=100, shapes=("orchestrator",))
    assert all(r.status == "ontime" for r in res) and e.stats.max_batch >= 2


def test_admission_control_drops_only_under_overload_and_dropped_tasks_stop_using_the_gpu():
    light, _ = _run("edf+ac", rate=0.05, duration=200)
    heavy, e = _run("edf+ac", rate=8.0, duration=30, max_seqs=8)
    assert not any(r.status == "dropped" for r in light)
    assert any(r.status == "dropped" for r in heavy)
    assert all(r.model_calls <= 1 + 8 for r in heavy)


def test_preload_issues_prefix_requests_during_tool_waits():
    res, e = _run("edf", rate=0.05, duration=100, shapes=("fanout",), preload=True)
    assert all(r.status == "ontime" for r in res)
    assert all(r.model_calls >= 3 for r in res)  # plan + >=1 preload + final


# ------------------------------------------------------------------------------------------- just-in-time preloading
def test_expected_wait_for_all_matches_monte_carlo():
    import random

    from deadline_agent.multiagent import expected_wait_for_all

    d = DelayModel(0.5, 4.0, 0.3, 4.0, 12.0)
    rng = random.Random(0)
    for elapsed in ([0.0], [0.0] * 4, [1.0] * 3, [3.5] * 2, [5.0]):
        samples = []
        while len(samples) < 20000:
            rem = []
            for e in elapsed:
                while True:  # conditional on not having returned yet
                    x = d.sample(rng.random(), rng.random())
                    if x > e:
                        rem.append(x - e)
                        break
            samples.append(max(rem))
        mc = sum(samples) / len(samples)
        assert abs(expected_wait_for_all(d, elapsed, dt=0.01) - mc) < 0.08, (elapsed, mc)
    assert expected_wait_for_all(d, []) == 0.0
    # more outstanding calls -> longer expected wait
    assert expected_wait_for_all(d, [1.0] * 4) > expected_wait_for_all(d, [1.0])


def _spy_backend(engine):
    seen = []

    class Spy(SimBackend):
        async def generate(self, messages, max_tokens, phase, schema=None, extra=None):
            seen.append((phase, now(), extra.get("priority")))
            return await super().generate(messages, max_tokens, phase, schema, extra)

    return Spy(engine), seen


def test_jit_preload_priority_is_the_expected_time_of_need():
    async def go():
        e = SimEngine(policy="priority")
        be, seen = _spy_backend(e)
        wl = make_workload([fixture_task()[0]], 0.05, 60, 0, shapes=("fanout",), width=4, slack=(50, 50))
        res = await serve(wl, be, ServingLayer("fcfs", CostEstimate()), DelayModel(0.5, 4.0, 0.3, 4.0, 12.0),
                          seed=1, preload="jit")
        await e.aclose()
        return res, seen

    res, seen = run_virtual(go())
    assert res and all(r.status == "ontime" for r in res)
    pre = [(t, p) for ph, t, p in seen if ph == "preload"]
    other = [(t, p) for ph, t, p in seen if ph != "preload"]
    assert pre, "jit issued no preloads"
    assert all(p > t for t, p in pre)  # needed in the future
    assert all(abs(p - t) < 1e-9 for t, p in other)  # plan/final: needed now


def test_final_call_is_never_blocked_by_a_starved_preload():
    """Engine runs one sequence at a time and is hogged by a long request; a low-priority preload queued behind it
    must not delay the agent's final call (it is cancelled when the final call is ready)."""
    async def go(mode):
        e = SimEngine(policy="priority", max_seqs=1)
        be, seen = _spy_backend(e)
        task = fixture_task()[0]
        at = AgentTask("t#0", task, 0.0, 1000.0, "fanout", 4)

        async def hog():  # high-priority long decode that keeps the only slot busy
            await asyncio.sleep(0.4)
            await e.submit(10, 2000, priority=-1.0)

        h = asyncio.create_task(hog())
        from deadline_agent.multiagent import run_agent_task
        r = await run_agent_task(at, be, ServingLayer("fcfs", CostEstimate()), DelayModel(0.5, 1.0), seed=1,
                                 preload=mode)
        h.cancel()
        await asyncio.gather(h, return_exceptions=True)
        await e.aclose()
        return r, seen

    r, seen = run_virtual(go("low"))
    assert r.status == "ontime"
    finals = [t for ph, t, _ in seen if ph == "final"]
    preloads = [t for ph, t, _ in seen if ph == "preload"]
    assert preloads, "expected at least one preload to be queued behind the hog"
    # the hog occupies the only slot for ~2000 * 29 ms = 58 s; a blocking implementation would send the final call
    # only after the queued preload ran (> 58 s). Tools all return within 1 s of the first preload's issue.
    assert finals and finals[0] <= preloads[0] + 1.0 + 1e-6


def test_task_result_records_final_call_timing_and_cache_hits():
    """Instrumentation: with prefix caching, preloading makes most of the final prompt a cache hit; timings are
    relative to the run start and ordered (last tool return <= final submit <= first token <= end)."""
    from deadline_agent.multiagent import run_agent_task

    async def go(mode):
        be = FakeBackend(prefix_cache=True)
        at = AgentTask("t#0", fixture_task()[0], 0.0, 1000.0, "fanout", 4)
        return await run_agent_task(at, be, ServingLayer("fcfs", CostEstimate()), DelayModel(0.5, 2.0), seed=1,
                                    preload=mode)

    off, on = run_virtual(go(False)), run_virtual(go(True))
    for r in (off, on):
        assert r.status == "ontime"
        assert r.t_last_tool <= r.final_submit <= r.final_first <= r.end
        assert r.final_prompt_tokens > 0
    assert off.n_preloads == 0 and on.n_preloads >= 1
    assert on.final_cached_tokens > off.final_cached_tokens


def test_chained_preload_covers_results_that_arrive_while_a_preload_runs():
    """With slow preloads (expensive prefill) and closely spaced tool returns, the one-in-flight rule leaves
    results un-prefilled; chaining covers them, so the final prompt has at least as many cache hits, and the
    final call is still never blocked (it is sent within one engine call of the last tool return)."""
    from deadline_agent.model_client import LatencyModel
    from deadline_agent.multiagent import run_agent_task

    async def go(mode, seed):
        be = FakeBackend(prefix_cache=True, costs=LatencyModel(prefill_s_per_tok=1.5e-3))
        at = AgentTask("t#0", fixture_task()[0], 0.0, 1000.0, "fanout", 4)
        return await run_agent_task(at, be, ServingLayer("fcfs", CostEstimate()), DelayModel(0.5, 3.0), seed=seed,
                                    preload=mode, top_k=4, result_cap=400)

    gains = 0
    for seed in range(12):
        on, ch = run_virtual(go(True, seed)), run_virtual(go("chain", seed))
        assert on.status == ch.status == "ontime"
        assert on.answer == ch.answer  # same evidence, same order
        assert ch.final_cached_tokens >= on.final_cached_tokens
        assert ch.final_submit - ch.t_last_tool < 1e-6  # the final call is sent as soon as the last tool returns
        gains += ch.final_cached_tokens > on.final_cached_tokens
    assert gains >= 1
