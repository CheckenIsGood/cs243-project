"""KV preloading (theory.md B1-B2): evidence-log layout, Preload action, fold+preload, fake prefix cache."""

from dataclasses import dataclass, field, replace

import pytest

from deadline_agent.affine import AffineCostModel
from deadline_agent.clock import run_virtual
from deadline_agent.datasets import fixture_task
from deadline_agent.model_client import FakeBackend
from deadline_agent.prompts import evidence_log_messages
from deadline_agent.runtime import ContractViolation, RuntimeConfig, run_task
from deadline_agent.tools import DelayModel, SearchTool

COSTS = AffineCostModel(overhead_s=0.02, prefill_s_per_tok=1 / 500, decode_s_per_tok=1 / 30,  # slow prefill
                        raw_tokens=300, summary_tokens=40, final_base_prompt=120, consume_base_prompt=120,
                        context_limit=8192)
DELAY = DelayModel(0.5, 4.0, 0.3, 4.0, 12.0)


@dataclass
class Recording(FakeBackend):
    log: list = field(default_factory=list)

    async def generate(self, messages, max_tokens, phase, schema=None, extra=None):
        g = await super().generate(messages, max_tokens, phase, schema, extra)
        self.log.append((phase, messages, g))
        return g


def run(policy, delays=(1.0, 6.0, 2.5), evidence_log=True, prefix_cache=True, cap=8192, deadline=60.0):
    task, _ = fixture_task()
    be = Recording(COSTS, prefix_cache=prefix_cache)

    async def go():
        tool = SearchTool(task.task_id, list(task.passages), DELAY, latency_seed=7, top_k=2, result_token_cap=400,
                          delay_override=lambda i, q: delays[i])
        cfg = RuntimeConfig(deadline_s=deadline, final_context_cap=cap, evidence_log=evidence_log)
        return await run_task(task, policy, replace(COSTS, context_limit=cap), DELAY.pmf(0.1), be, tool, cfg, "t")

    res, ev = run_virtual(go())
    return res, ev, be


def test_evidence_log_entries_are_string_prefixes():
    a = evidence_log_messages("Q?", ["e1", "e2"])
    b = evidence_log_messages("Q?", ["e1", "e2", "e3"])
    assert a[0] == b[0] and b[1]["content"].startswith(a[1]["content"])


def test_fold_preload_prefills_during_idle_time_and_final_reuses_the_cache():
    res, ev, be = run("fold_preload")
    assert res.valid and res.n_summarized == 0 and res.n_folded == res.n_calls
    pre = [(m, g) for ph, m, g in be.log if ph == "preload"]
    fin = [(m, g) for ph, m, g in be.log if ph == "final"]
    assert pre and len(fin) == 1
    final_content = fin[0][0][1]["content"]
    for m, g in pre:  # every preload prompt is a prefix of the final prompt; 1 output token
        assert final_content.startswith(m[1]["content"].rstrip())
        assert g.completion_tokens <= 1
    # preloads happen only while some tool is still running
    last_ret = max(e["t"] for e in ev if e["event"] == "tool_return")
    assert all(e["t"] <= last_ret + 1e-9 for e in ev if e["event"] == "preload_start")
    assert fin[0][1].cached_tokens >= sum(g.prompt_tokens for _, g in pre[-1:]) - 2 * be.block_tokens


def test_fold_preload_finishes_earlier_than_fold_with_the_same_final_content():
    r0, ev0, b0 = run("fold")
    r1, ev1, b1 = run("fold_preload")
    assert r0.valid and r1.valid and r1.answer == r0.answer
    f0 = next(m for ph, m, _ in b0.log if ph == "final")[1]["content"]
    f1 = next(m for ph, m, _ in b1.log if ph == "final")[1]["content"]
    assert sorted(f0.split("\n\n")) == sorted(f1.split("\n\n"))  # same evidence, possibly different order
    assert r1.completion_s < r0.completion_s - 0.1


def test_preload_requires_evidence_log():
    with pytest.raises(ContractViolation):
        run("fold_preload", evidence_log=False)


def test_rss_preload_never_summarizes_a_preloaded_result():
    # cap 920: RSS must summarize one result; c1 was preloaded first, so the guard redirects RSS to c2
    res, ev, be = run("rss_preload", cap=920, delays=(1.0, 3.0, 9.0))
    logged = {c for e in ev if e["event"] == "preload_start" for c in e["call_ids"]}
    summarized = {c for e in ev if e["event"] == "consume_start" for c in e["call_ids"]}
    assert res.valid and logged and summarized and not (logged & summarized)
    fin = next(g for ph, _, g in be.log if ph == "final")
    assert fin.cached_tokens > 0  # the summary sits after the logged raws, so the logged prefix is still reused


def test_fake_prefix_cache_reports_no_reuse_without_a_prior_prompt():
    _, _, be = run("fold", prefix_cache=True)
    assert be.log[0][2].cached_tokens == 0
