"""Client-side KV-aware preload controller (preload_controller.py) driven through the real agent loop on the fake
backend (virtual clock). These check the mechanics: decisions, pins held open and released, yielding, feedback and
the exploration probe. They do not check latency benefit (the fake backend has no KV capacity)."""

import asyncio

from deadline_agent.clock import now, run_virtual
from deadline_agent.datasets import fixture_task
from deadline_agent.model_client import FakeBackend
from deadline_agent.multiagent import AgentTask, CostEstimate, ServingLayer, run_agent_task, serve, make_workload
from deadline_agent.preload_controller import PreloadController
from deadline_agent.state import Generation
from deadline_agent.tools import DelayModel

DELAY = DelayModel(0.5, 4.0, 0.3, 4.0, 12.0)
DT = (DELAY.lo, DELAY.hi, DELAY.slow_p, DELAY.slow_lo, DELAY.slow_hi)


class HoldBackend(FakeBackend):
    """FakeBackend whose ignore_eos requests decode for max_tokens steps (so a pin really stays open)."""

    async def generate(self, messages, max_tokens, phase, schema=None, extra=None):
        if not (extra or {}).get("ignore_eos"):
            return await super().generate(messages, max_tokens, phase, schema, extra)
        submit = now()
        prompt = sum(self.count_tokens(m["content"]) + 4 for m in messages)
        cached = self._cached_tokens(messages) if self.prefix_cache else None
        await asyncio.sleep(self.costs.overhead_s + self.costs.prefill_s_per_tok * (prompt - (cached or 0)))
        first = now()
        extra["_on_first"]()
        await asyncio.sleep(self.costs.decode_s_per_tok * max_tokens)
        return Generation("x" * max_tokens, prompt, max_tokens, "length", submit, first, now(), cached_tokens=cached)


def one_task(mode, ctrl=None, seed=0, backend=None):
    async def go():
        be = backend or FakeBackend(prefix_cache=True)
        if ctrl is not None:
            ctrl.task_arrived  # noqa: B018 (the agent registers itself)
        at = AgentTask("t#0", fixture_task()[0], 0.0, 1000.0, "fanout", 4)
        return await run_agent_task(at, be, ServingLayer("fcfs", CostEstimate()), DELAY, seed=seed, preload=mode,
                                    top_k=4, result_cap=400, controller=ctrl)
    return run_virtual(go())


def test_ctrl_with_ample_kv_acts_like_low_priority_preloading():
    """Theorem A-ii: when capacity suffices the rule is always-preload (unpinned, lowest priority)."""
    for seed in range(6):
        c = PreloadController(M=10**7, delay=DT)
        lo, ct = one_task("low", seed=seed), one_task("ctrl", c, seed=seed)
        assert lo.status == ct.status == "ontime"
        assert lo.answer == ct.answer
        assert ct.n_preloads >= 1 and ct.final_cached_tokens >= lo.final_cached_tokens * 0.9
        assert c.stats["pin"] == 0 and c.stats["unpinned"] == ct.n_preloads
        assert not c.tasks and not c.reqs  # all bookkeeping released


def test_ctrl_with_tiny_kv_never_preloads():
    """No capacity: the TTL estimate is 0, so survival p = 0 < sigma*, and no pin fits the budget -> skip."""
    c = PreloadController(M=300, delay=DT)  # below the log size: TTL 0
    r = one_task("ctrl", c)
    assert r.status == "ontime" and r.n_preloads == 0
    assert c.stats["skip"] >= 1 and c.stats["unpinned"] == c.stats["pin"] == 0


def test_pin_is_held_open_then_released_at_the_final_first_token():
    """Force every decision to 'pin': the pin decodes in the background (ignore_eos), the agent never waits on it,
    and it is cancelled once superseded or once the final call's first token arrives."""
    c = PreloadController(M=10**7, delay=DT)
    c.on_return = (lambda f: (lambda key, L: (f(key, L), "pin")[1]))(c.on_return)
    c.last_pin_tokens = 2000  # 2000 x 29 ms = 58 s if it were never cancelled
    r = one_task("ctrl", c, backend=HoldBackend(prefix_cache=True))
    assert r.status == "ontime" and r.n_preloads >= 1
    assert r.end - r.arrival < 30  # the task did not wait for its pins
    assert c.stats["cancel_final_started"] + c.stats["cancel_superseded"] >= 1
    assert not c.reqs  # nothing left open


def test_feedback_turns_unpinned_preloads_off_and_the_probe_keeps_measuring():
    """sigma_hat <= sigma* after min_obs observations stops unpinned preloads, except every explore_every-th."""
    c = PreloadController(M=10**7, delay=DT, explore_every=3)
    c._sig = [(1000, 0)] * 10  # ten tasks whose preloads all died
    outs = []
    for seed in range(4):
        before = c.stats["unpinned"] + c.stats["skip"]
        one_task("ctrl", c, seed=seed)
        outs.append(before)
    assert c.stats["unpinned"] == c.stats["explore"]  # every unpinned preload came from the probe
    assert c.stats["explore"] >= 1 and c.stats["skip"] >= 2 * c.stats["explore"]


def test_ctrl_runs_a_multi_tenant_workload_and_feeds_its_estimators():
    async def go():
        be = FakeBackend(prefix_cache=True)
        c = PreloadController(M=10**6, delay=DT)
        wl = make_workload([fixture_task()[0]], 0.5, 20.0, seed=3, shapes=("fanout",), width=4, slack=(1000, 1000))
        res = await serve(wl, be, ServingLayer("fcfs", CostEstimate()), DELAY, seed=1, preload="ctrl", top_k=4,
                          result_cap=400, controller=c)
        return res, c
    res, c = run_virtual(go())
    assert res and all(r.status == "ontime" for r in res)
    s = c.summary()
    assert s["n_sigma_obs"] >= 1 and 0 <= s["sigma_hat"] <= 1
    assert len(c._xcrit) == len(res)
    assert all(d["act"] in ("pin", "unpinned", "skip") for d in s["decisions"])
    assert not c.reqs and not c.tasks
