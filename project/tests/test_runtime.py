"""Gate 0.5: runtime invariants on the virtual clock with the fake backend (PLAN.MD Section 10)."""

from dataclasses import fields, replace

import pytest

from deadline_agent.affine import AffineCostModel
from deadline_agent.clock import run_virtual
from deadline_agent.datasets import fixture_task
from deadline_agent.model_client import FakeBackend
from deadline_agent.runtime import RuntimeConfig, run_task
from deadline_agent.state import Observation
from deadline_agent.tools import DelayModel, SearchTool

POLICIES = ["incremental", "batch2", "fold", "rss", "optimal"]
COSTS = AffineCostModel(overhead_s=0.08, prefill_s_per_tok=1 / 3000, decode_s_per_tok=1 / 30,
                        raw_tokens=300, summary_tokens=40, final_base_prompt=120, consume_base_prompt=120,
                        context_limit=4096)
DELAY = DelayModel(0.5, 4.0, 0.3, 4.0, 12.0)


def run(policy="fold", deadline=40.0, delays=None, costs=COSTS, backend=None, cap=4096, top_k=2,
        result_cap=400, charge_controller=True, tool_raises=None):
    task, _ = fixture_task()

    def override(i, q):
        if tool_raises is not None and i == tool_raises:
            raise RuntimeError("tool exploded")
        return delays[i]

    async def go():
        tool = SearchTool(task.task_id, list(task.passages), DELAY, latency_seed=7, top_k=top_k,
                          result_token_cap=result_cap,
                          delay_override=override if (delays is not None or tool_raises is not None) else None)
        cfg = RuntimeConfig(deadline_s=deadline, final_context_cap=cap, charge_controller=charge_controller)
        return await run_task(task, policy, replace(costs, context_limit=cap), DELAY.pmf(0.1),
                              backend or FakeBackend(costs), tool, cfg, "test")

    return run_virtual(go())


def model_intervals(events):
    starts = {e["inv"]: e["t"] for e in events if e["event"].endswith("_start") and "inv" in e}
    ends = {e["inv"]: e["t"] for e in events if e["event"].endswith("_end") and "inv" in e}
    return sorted((starts[i], ends.get(i, float("inf")), i) for i in starts)


def first(events, name, **match):
    return next(e for e in events if e["event"] == name and all(e.get(k) == v for k, v in match.items()))


@pytest.mark.parametrize("policy", POLICIES)
def test_valid_run_consumes_every_call_exactly_once_after_return(policy):
    res, ev = run(policy, delays=[1.0, 6.0, 2.5])
    assert res.valid, res.miss_reason
    returned = {e["call_id"]: e["t"] for e in ev if e["event"] == "tool_return"}
    consumed = {}
    for e in ev:
        if e["event"] in ("consume_end", "final_end"):
            start = first(ev, e["event"].replace("_end", "_start"), inv=e["inv"])["t"]
            for c in e["call_ids"]:
                assert c not in consumed, f"{c} consumed twice"
                assert start >= returned[c] - 1e-9, f"{c} consumed before it returned"
                consumed[c] = e["inv"]
    assert set(consumed) == set(returned) and len(returned) == res.n_calls


@pytest.mark.parametrize("policy", POLICIES)
def test_one_generation_at_a_time(policy):
    _, ev = run(policy, delays=[1.0, 6.0, 2.5])
    iv = model_intervals(ev)
    for (s1, e1, _), (s2, _, _) in zip(iv, iv[1:]):
        assert e1 <= s2 + 1e-9


def test_tools_overlap_and_model_works_while_tools_pending():
    _, ev = run("incremental", delays=[1.0, 6.0, 2.5])
    launches = [e["t"] for e in ev if e["event"] == "tool_launch"]
    assert len(set(launches)) == 1 and len(launches) == 3  # launched together (K = n)
    last_return = max(e["t"] for e in ev if e["event"] == "tool_return")
    assert any(e["event"] == "consume_start" and e["t"] < last_return for e in ev)


def test_deadline_miss_is_explicit_and_cleanup_follows():
    res, ev = run("incremental", deadline=6.0, delays=[1.0, 6.0, 2.5])
    assert not res.valid and res.miss_reason == "deadline" and res.answer is None
    miss = first(ev, "deadline_miss")
    assert miss["t"] == pytest.approx(6.0)
    assert not any(e["event"] == "final_end" for e in ev)
    names = [e["event"] for e in ev]
    assert names.index("cleanup_start") > names.index("deadline_miss")
    assert "cleanup_end" in names


def test_malformed_final_never_counts_and_retry_is_charged():
    res, ev = run("fold", delays=[1.0, 2.0, 2.5], backend=FakeBackend(COSTS, malformed_phases={"final"}))
    assert not res.valid and res.miss_reason == "malformed_final"
    assert sum(e["event"] == "final_parse_failed" for e in ev) == 2  # original + 1 retry, both on the clock


def test_truncated_final_is_not_success():
    res, _ = run("fold", delays=[1.0, 2.0, 2.5], backend=FakeBackend(COSTS))
    assert res.valid
    tiny = replace(COSTS)
    task_cfg_backend = FakeBackend(tiny)
    # a 2-token final budget truncates the JSON answer
    task, _ = fixture_task()

    async def go():
        tool = SearchTool(task.task_id, list(task.passages), DELAY, 7, top_k=2, result_token_cap=400,
                          delay_override=lambda i, q: [1.0, 2.0, 2.5][i])
        return await run_task(task, "fold", COSTS, DELAY.pmf(0.1), task_cfg_backend, tool,
                              RuntimeConfig(deadline_s=40, final_max_tokens=2), "test")

    res2, _ = run_virtual(go())
    assert not res2.valid and res2.miss_reason == "malformed_final"


def test_tool_error_is_a_failure():
    res, ev = run("fold", tool_raises=1, delays=[1.0, 2.0, 2.5])
    assert not res.valid and res.miss_reason == "tool_error"
    assert any(e["event"] == "tool_error" for e in ev)


def test_observation_exposes_no_future_timing():
    names = {f.name for f in fields(Observation)}
    assert not names & {"delay", "injected_delay_s", "return_t", "release", "release_times", "result_text"}


@pytest.mark.parametrize("policy", POLICIES)
def test_decisions_do_not_depend_on_unrevealed_futures(policy):
    """Two worlds identical until c2 returns (t=6 vs t=10 after launch): decisions before 6 s must match."""
    _, ev_a = run(policy, delays=[1.0, 6.0, 2.5], charge_controller=False)
    _, ev_b = run(policy, delays=[1.0, 10.0, 2.5], charge_controller=False)
    launch = first(ev_a, "tool_launch")["t"]

    def decisions(ev):
        return [(round(e["t"], 6), e["action"]) for e in ev if e["event"] == "decision" and e["t"] < launch + 6.0 - 1e-6]  # strictly before c2 can return

    assert decisions(ev_a) == decisions(ev_b)


def test_runs_are_deterministic_without_controller_charging():
    strip = lambda ev: [(e["event"], round(e["t"], 6), e.get("call_id"), e.get("action")) for e in ev]  # noqa: E731
    a = run("optimal", charge_controller=False)[1]
    b = run("optimal", charge_controller=False)[1]
    assert strip(a) == strip(b)


def test_binding_context_forces_summaries_and_respects_cap():
    cap = 900  # three ~300-token raw results + instructions + 192-token answer cannot all fit
    for policy in POLICIES:
        res, ev = run(policy, delays=[1.0, 6.0, 2.5], cap=cap, top_k=6)
        assert res.valid, (policy, res.miss_reason)
        assert res.n_summarized >= 1, policy
        for e in ev:
            if e["event"].endswith("_start") and "est_prompt_tokens" in e and e["event"] != "plan_start":
                assert e["est_prompt_tokens"] + e["max_tokens"] <= cap, (policy, e)


def test_adaptive_policies_summarize_during_waits_when_context_binds():
    _, ev = run("rss", delays=[1.0, 8.0, 1.5], cap=900, top_k=6)
    last_return = max(e["t"] for e in ev if e["event"] == "tool_return")
    assert any(e["event"] == "consume_start" and e["t"] < last_return for e in ev)
    _, ev_fold = run("fold", delays=[1.0, 8.0, 1.5], cap=900, top_k=6)
    last_return = max(e["t"] for e in ev_fold if e["event"] == "tool_return")
    assert not any(e["event"] == "consume_start" and e["t"] < last_return for e in ev_fold)


def test_salvage_consume_keeps_complete_records_and_falls_back_for_the_rest():
    from deadline_agent.prompts import salvage_consume

    truncated = '{"records": [{"call_id": "c2", "facts": ["Founded in 1871", "Arden River"], "sources": ["A"]}, ' \
                '{"call_id": "c1", "facts": ["The Merrow Bay Sailing Soc'
    recs, fb = salvage_consume(truncated, ["c2", "c1"], {"c1": "<source id=\"x\">Merrow Bay was founded in 1894.</source>"})
    assert recs["c2"]["facts"] == ["Founded in 1871", "Arden River"]
    assert fb == ["c1"] and recs["c1"]["fallback"] and "1894" in recs["c1"]["facts"][0]
    ok, fb2 = salvage_consume('{"records": [{"call_id": "c1", "facts": ["x"]}]}', ["c1"], {})
    assert fb2 == [] and ok["c1"]["facts"] == ["x"]


def test_unparseable_consume_never_loops_and_each_result_is_summarized_once(tmp_path):
    """Regression (Gate 1): a truncated consume left results raw and the policy re-consumed them until the
    deadline. Now the batch is salvaged/fallback-summarized exactly once and the task completes."""
    from deadline_agent.affine import AffineCostModel
    from deadline_agent.clock import run_virtual
    from deadline_agent.datasets import fixture_task
    from deadline_agent.model_client import FakeBackend
    from deadline_agent.online_mdp import bounded_delay_pmf
    from deadline_agent.runtime import RuntimeConfig, run_task
    from deadline_agent.tools import DelayModel, SearchTool

    task, _ = fixture_task()
    costs = AffineCostModel(raw_tokens=120, summary_tokens=30, context_limit=400, final_out=20)
    backend = FakeBackend(costs, malformed_phases={"consume"})
    delay = DelayModel(lo=0.5, hi=3.0, slow_p=0.0, slow_lo=0.0, slow_hi=0.0)
    tool = SearchTool(task.task_id, list(task.passages), delay, 1, top_k=1, result_token_cap=120)
    cfg = RuntimeConfig(deadline_s=60.0, final_context_cap=400, dt=0.1)
    res, ev = run_virtual(run_task(task, "rss", costs, tuple(bounded_delay_pmf(0.1, 0.5, 3.0)), backend, tool, cfg,
                                   "t", str(tmp_path / "t.jsonl")))
    consumed = [c for e in ev if e["event"] == "consume_start" for c in e["call_ids"]]
    assert len(consumed) == len(set(consumed)), "a result was summarized twice"
    assert any(e["event"] == "consume_salvaged" for e in ev)
    assert res.valid, res.miss_reason


def test_parse_consume_normalizes_ids_the_model_actually_writes():
    """Regression (Gate 1): Qwen3 wrote "call_id": "1" for "c1"; the parser silently emptied the summary."""
    from deadline_agent.prompts import consume_schema, parse_consume

    out = parse_consume('{"records": [{"call_id": "1", "facts": ["a"]}, {"call_id": "[c3]", "facts": ["b"]}]}', ["c1", "c3"])
    assert out["c1"]["facts"] == ["a"] and out["c3"]["facts"] == ["b"]
    out = parse_consume('{"records": [{"call_id": "x", "facts": ["a"]}, {"call_id": "y", "facts": ["b"]}]}', ["c2", "c4"])
    assert out["c2"]["facts"] == ["a"] and out["c4"]["facts"] == ["b"]  # unmatched fill ids in request order
    sch = consume_schema(["c2", "c4"])["properties"]["records"]
    assert sch["minItems"] == sch["maxItems"] == 2 and sch["items"]["properties"]["call_id"]["enum"] == ["c2", "c4"]
