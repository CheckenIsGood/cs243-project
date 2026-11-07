"""Multi-tenant / multi-agent serving layer on the simulated batching engine (virtual clock)."""

import asyncio

import pytest

from deadline_agent.clock import now, run_virtual
from deadline_agent.datasets import fixture_task
from deadline_agent.engine_sim import SimBackend, SimEngine
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
