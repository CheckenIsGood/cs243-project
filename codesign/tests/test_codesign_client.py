"""Client side of the co-design on the fake backend: the react shape reuses prefixes across tool waits, and the
CodesignLayer sends a priority and vllm_xargs on every request."""

from deadline_agent.clock import run_virtual
from deadline_agent.datasets import fixture_task
from deadline_agent.model_client import FakeBackend
from deadline_agent.multiagent import AgentTask, make_workload, run_agent_task, serve
from deadline_agent.tools import DelayModel

from codesign.client import Calib, CodesignLayer, ShapeEst, delay_quantile, task_nodes, zprime

DELAY = DelayModel(0.5, 4.0, 0.3, 4.0, 12.0)


class Recording(FakeBackend):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.sent = []

    async def generate(self, messages, max_tokens, phase, schema=None, extra=None):
        self.sent.append((phase, dict(extra or {})))
        return await super().generate(messages, max_tokens, phase, schema, {k: v for k, v in (extra or {}).items()
                                                                              if k == "priority"})


def test_react_shape_runs_and_reuses_the_prefix():
    be = Recording(prefix_cache=True)
    layer = CodesignLayer("zprime", M_tokens=50_000, L_max=8000, ttl=5.0)
    at = AgentTask("t#0", fixture_task()[0], 0.0, 1000.0, "react", 3)
    r = run_virtual(run_agent_task(at, be, layer, DELAY, seed=0, top_k=4, result_cap=400))
    assert r.status == "ontime" and r.model_calls == 4  # 3 steps + final
    assert r.final_cached_tokens and r.final_cached_tokens > 0.5 * r.final_prompt_tokens
    phases = [p for p, _ in be.sent]
    assert phases == ["step", "step", "step", "final"]
    prios = {x["priority"] for _, x in be.sent}
    assert len(prios) == 1  # one fixed index per task
    assert all(x["vllm_xargs"]["cd_task"] == "t#0" for _, x in be.sent)
    assert [("cd_ttl" in x["vllm_xargs"]) for _, x in be.sent] == [True, True, True, False]


def test_rules_and_index_ordering():
    e, cal = ShapeEst(), Calib()
    small, big = task_nodes("fanout", 2, e), task_nodes("react", 6, e)
    assert zprime(small, cal, 50_000, 8000) < zprime(big, cal, 50_000, 8000)
    assert zprime(small, cal, None, 0) < zprime(small, cal, 50_000, 8000)  # memory terms only add
    for rule in ("fcfs", "zprime", "task_fcfs", "plas", "area"):
        be = Recording(prefix_cache=True)
        wl = make_workload([fixture_task()[0]], 0.5, 10.0, seed=3, shapes=("fanout", "react"),
                           width={"fanout": [2, 4], "react": [2, 3]}, slack=(1000, 1000))
        res = run_virtual(serve(wl, be, CodesignLayer(rule, M_tokens=50_000, L_max=8000), DELAY, seed=1,
                                top_k=4, result_cap=300))
        assert res and all(x.status == "ontime" for x in res), rule
        assert all("priority" in x and "vllm_xargs" in x for _, x in be.sent)


def test_preloads_are_tagged_lowest():
    be = Recording(prefix_cache=True)
    at = AgentTask("t#1", fixture_task()[0], 0.0, 1000.0, "fanout", 4)
    r = run_virtual(run_agent_task(at, be, CodesignLayer("zprime"), DELAY, seed=0, top_k=4, result_cap=300,
                                   preload="low"))
    pre = [x for p, x in be.sent if p == "preload"]
    assert r.n_preloads >= 1 and pre and all(x["vllm_xargs"].get("cd_preload") == 1 and x["priority"] == 1e9
                                             for x in pre)


def test_delay_quantile():
    assert 2.9 <= delay_quantile(DELAY, 0.5) <= 3.1 and 9 <= delay_quantile(DELAY, 0.9) <= 9.6


def test_aging_priority_bounds_overtaking():
    """zprime_age: priority r_J + kappa*Z'_J. A later task K overtakes J only if r_K - r_J < kappa*(Z'_J - Z'_K)."""
    from deadline_agent.multiagent import AgentTask as AT
    t = fixture_task()[0]
    big, small = AT("big#0", t, 0.0, 1e3, "react", 6), AT("small#1", t, 0.0, 1e3, "fanout", 2)
    for kappa in (0.0, 1.0, 4.0):
        lay = CodesignLayer("zprime_age", M_tokens=16_000, L_max=8184, kappa=kappa)
        zb, zs = lay.task_index(big), lay.task_index(small)
        window = kappa * (zb - zs)
        for dt in (0.0, 0.5 * window, window + 1.0, 30.0):
            small.arrival = big.arrival + dt
            pb = lay.request_extra(big, "s", "step", [], 48, None)["priority"]
            ps = lay.request_extra(small, "s", "plan", [], 128, None)["priority"]
            assert (ps < pb) == (dt < window), (kappa, dt)  # overtakes exactly inside the window
    # kappa = 0 is task FCFS; large kappa reproduces the zprime order
    lay = CodesignLayer("zprime_age", M_tokens=16_000, L_max=8184, kappa=1e6)
    small.arrival = big.arrival + 100.0
    assert lay.request_extra(small, "s", "plan", [], 128, None)["priority"] < \
        lay.request_extra(big, "s", "step", [], 48, None)["priority"]
