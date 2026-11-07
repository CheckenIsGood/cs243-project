"""Multi-tenant, multi-agent serving layer (docs/general_theory.md, settings a and b).

Many tasks arrive over time and share one engine. Each task is an agent program:
- "fanout"        (one agent):     plan -> n parallel searches (async, issued eagerly: Lemma 3) -> final answer;
- "orchestrator"  (collaborating): orchestrator plans k sub-questions -> k worker agents in parallel, each
                                   search -> read -> report -> orchestrator answers from the reports.
The serving layer decides, per model request, its engine priority, and whether a task is still worth serving:
  fcfs        no priorities (stock vLLM default);
  edf         priority = task deadline (earliest deadline first);
  llf         priority = task deadline - estimated remaining critical path after this request (least laxity:
              per-request sub-deadlines from the task deadline);
  <p>+ac      admission control on top of <p>: a task whose estimated finish (now + remaining critical path,
              stretched by the current batch slowdown) is past its deadline is dropped before it uses more GPU
              (overload handling in the spirit of D^over).
Optional KV preloading during tool waits (fanout): each returned result is prefilled into the prefix cache.
The same code runs on SimBackend (virtual clock) and on VLLMBackend (real server, --scheduling-policy priority).
"""

from __future__ import annotations

import asyncio
import json
import random
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any

from .clock import now
from .datasets import Task
from .model_client import messages_tokens
from .prompts import FINAL_SCHEMA, evidence_log_messages, parse_final, parse_plan, plan_messages, plan_schema, raw_block
from .tools import DelayModel, SearchTool, normalize_query

WORKER_SCHEMA = {"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]}


def worker_messages(question: str, subq: str, result: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": "You are a research worker. Using the search results, report in at most "
             "two sentences what they say about your sub-question. Return JSON {\"report\": ...}."},
            {"role": "user", "content": f"Main question: {question}\nSub-question: {subq}\n\nSearch results:\n{result}"}]


def orchestrator_final_messages(question: str, reports: list[tuple[str, str]]) -> list[dict[str, str]]:
    body = "\n".join(f"- ({q}) {r}" for q, r in reports) or "(no reports)"
    return [{"role": "system", "content": "Answer the question from your workers' reports. Return JSON "
             "{\"answer\": ...} where answer is a short span (an entity, date, number, or yes/no)."},
            {"role": "user", "content": f"Question: {question}\n\nWorker reports:\n{body}"}]


@dataclass
class AgentTask:
    task_id: str
    task: Task
    arrival: float
    deadline: float  # absolute
    shape: str  # fanout | orchestrator
    width: int = 4  # n searches or k workers
    value: float = 1.0


@dataclass
class TaskResult:
    task_id: str
    shape: str
    arrival: float
    deadline: float
    status: str  # ontime | late | dropped | error
    end: float | None
    answer: str | None
    model_calls: int
    # final-call instrumentation (fan-out / orchestrator final); times are on the same clock as `end`
    t_last_tool: float | None = None  # last tool result returned
    final_submit: float | None = None
    final_first: float | None = None  # first output token of the final call
    final_prompt_tokens: int | None = None
    final_cached_tokens: int | None = None  # prompt tokens served from the prefix cache (vLLM usage details)
    n_preloads: int = 0


@dataclass
class CostEstimate:
    """Batch-1 latency model (profiled) used for critical-path estimates."""
    prefill_s_per_tok: float = 1.8e-4
    decode_s_per_tok: float = 0.0293
    out_tokens: dict[str, int] = field(default_factory=lambda: {"plan": 40, "worker": 80, "final": 12})
    prompt_tokens: dict[str, int] = field(default_factory=lambda: {"plan": 90, "worker": 700, "final": 2500,
                                                                     "ofinal": 400})
    tool_s: float = 4.0  # expected max tool delay of the parallel calls (set from the delay model)

    def node(self, phase: str) -> float:
        out = self.out_tokens.get("final" if phase == "ofinal" else phase, 12)
        return self.prefill_s_per_tok * self.prompt_tokens.get(phase, 500) + self.decode_s_per_tok * out

    def remaining(self, shape: str, stage: str) -> float:
        """Estimated critical-path time from the START of `stage` to task end, at batch 1."""
        if shape == "fanout":
            seq = {"plan": ["plan", "tool", "final"], "tool": ["tool", "final"], "final": ["final"]}[stage]
        else:
            seq = {"plan": ["plan", "tool", "worker", "ofinal"], "tool": ["tool", "worker", "ofinal"],
                   "worker": ["worker", "ofinal"], "ofinal": ["ofinal"]}[stage]
        return sum(self.tool_s if s == "tool" else self.node(s) for s in seq)


@lru_cache(maxsize=64)
def _delay_cdf(delay: DelayModel, dt: float) -> tuple[float, ...]:
    out, acc = [], 0.0
    for p in delay.pmf(dt):
        acc += p
        out.append(min(1.0, acc))
    return tuple(out)


def expected_wait_for_all(delay: DelayModel, elapsed: list[float], dt: float = 0.05) -> float:
    """E[time until ALL outstanding tool calls return], given each has already run for elapsed[i] seconds and
    durations are i.i.d. from `delay` (the conditional remaining time of call i is D - e_i given D > e_i)."""
    if not elapsed:
        return 0.0
    cdf = _delay_cdf(delay, dt)
    n = len(cdf)

    def F(t: float) -> float:
        k = int(t / dt + 1e-9)
        return 1.0 if k >= n else (0.0 if k < 0 else cdf[k])

    base = [F(e) for e in elapsed]
    total = 0.0
    for k in range(n + 1):
        x = k * dt
        prod = 1.0
        for e, fe in zip(elapsed, base):
            if fe < 1.0:
                prod *= (F(e + x) - fe) / (1.0 - fe)
        total += (1.0 - prod) * dt
        if prod >= 1.0:
            break
    return total


def expected_max_delay(delay: DelayModel, n: int, dt: float = 0.01) -> float:
    pmf = delay.pmf(dt)
    cdf, acc = [], 0.0
    for p in pmf:
        acc += p
        cdf.append(min(1.0, acc))
    return sum((1 - cdf[i] ** n) * dt for i in range(len(cdf)))


class ServingLayer:
    def __init__(self, policy: str, est: CostEstimate, slowdown=lambda: 1.0):
        base, _, ac = policy.partition("+")
        if base not in ("fcfs", "edf", "llf") or ac not in ("", "ac"):
            raise ValueError(f"unknown serving policy {policy!r}")
        self.base, self.ac, self.est, self.slowdown = base, ac == "ac", est, slowdown

    def priority(self, at: AgentTask, stage: str) -> float:
        if self.base == "fcfs":
            return 0.0
        if self.base == "edf":
            return at.deadline
        # llf: sub-deadline = deadline - remaining critical path after this request
        after = self.est.remaining(at.shape, stage) - self.est.node(stage)
        return at.deadline - after

    def keep(self, at: AgentTask, stage: str) -> bool:
        if not self.ac:
            return True
        return now() + self.est.remaining(at.shape, stage) * self.slowdown() <= at.deadline


LOWEST_PRIORITY = 1e9  # seconds; above any absolute deadline, so it sorts after every other request


class _Drop(Exception):
    pass


async def run_agent_task(at: AgentTask, backend, layer: ServingLayer, delay: DelayModel, seed: int,
                         top_k: int = 4, result_cap: int = 576, preload: bool | str = False, inline_tools: bool = True,
                         final_max_tokens: int = 64, controller=None) -> TaskResult:
    """preload modes (fan-out tasks): False (off); True (preload each returned result at the task's normal
    priority); "low" (preloads at the lowest engine priority); "jit" (just-in-time: every request's priority is
    the time by which its work is needed -- now for plan/final calls, and for a preload the expected time at which
    the agent's last outstanding tool returns, from the tool-latency distribution given elapsed times); "chain" (as True,
    and when a preload finishes while results that arrived during it are not yet prefilled, the next preload is sent
    at once instead of at the next tool return); "ctrl" (the client-side KV-aware controller of
    docs/controller_theory.md Section 7, `controller` = a shared preload_controller.PreloadController: per tool
    return it pins, sends an unpinned lowest-priority preload, or skips; plan and final calls go through it too, so
    it can account for KV demand and yield pins before they are submitted).
    Preloads never block the agent: at most one is in flight; when the final call is ready, an unfinished preload
    is cancelled (its already-computed KV blocks stay in the prefix cache) and the final call is sent at once."""
    calls = 0
    n_pre = 0
    t_last = gfin = None
    ctrl = controller if preload == "ctrl" else None
    if preload == "ctrl" and ctrl is None:
        raise ValueError("preload='ctrl' needs a PreloadController")
    key = at.task_id
    status = "error"  # until the task finishes (also covers outside cancellation of the whole run)
    if ctrl is not None:
        ctrl.task_arrived(key)

    async def llm(stage: str, msgs, max_tokens: int, schema, phase: str | None = None, prio: float | None = None):
        nonlocal calls, n_pre
        if not layer.keep(at, stage):
            raise _Drop()
        calls += 1
        n_pre += phase == "preload"
        if prio is None:
            if phase == "preload" and preload == "low":
                prio = LOWEST_PRIORITY
            elif preload == "jit":
                prio = now()  # needed now (preloads pass their expected need time explicitly)
            else:
                prio = layer.priority(at, stage)
        if ctrl is not None:
            kind = phase if phase in ("plan", "final") else "other"
            ctrl.yield_for(key, messages_tokens(backend, msgs))
            return await ctrl.generate(backend, key, kind, msgs, max_tokens, phase or stage, schema, {"priority": prio})
        return await backend.generate(msgs, max_tokens, phase or stage, schema, {"priority": prio})

    tool = SearchTool(at.task_id, list(at.task.passages), delay, seed, top_k=top_k, result_token_cap=result_cap,
                      inline=inline_tools)
    answer = None
    tools: list[asyncio.Task] = []
    pending: asyncio.Task | None = None  # in-flight preload (fan-out)
    pre_tasks: list[asyncio.Task] = []  # every preload / pin of the controller mode
    try:
        async with asyncio.timeout_at(at.deadline):
            g = await llm("plan", plan_messages(at.task.question, at.width), 128, plan_schema(at.width), "plan")
            try:
                qs = parse_plan(g.text)
            except ValueError:
                qs = []
            seen, uniq = set(), []
            for q in qs:
                k = normalize_query(q)
                if k and k not in seen:
                    seen.add(k)
                    uniq.append(q)
            uniq = (uniq or [at.task.question])[: at.width]
            # eager asynchronous tool issue (Lemma 3)
            t_issue = now()
            tools = [asyncio.create_task(tool.run(f"c{i + 1}", i, q)) for i, q in enumerate(uniq)]
            if ctrl is not None:
                ctrl.tools_issued(key, len(tools))
            if at.shape == "fanout" and preload == "ctrl":
                logged = []
                for fut in asyncio.as_completed(tools):
                    r = await fut
                    if r.error:
                        raise RuntimeError(r.error)
                    logged.append(raw_block(r.call_id, uniq[int(r.call_id[1:]) - 1], r.text))
                    if len(logged) == len(tools):
                        break
                    msgs = evidence_log_messages(at.task.question, list(logged))
                    act = ctrl.on_return(key, messages_tokens(backend, msgs))
                    if act == "skip":
                        continue
                    pinned = act == "pin"
                    mt = ctrl.last_pin_tokens if pinned else 1
                    extra = {"priority": LOWEST_PRIORITY, **({"ignore_eos": True} if pinned else {})}
                    req = ctrl.register(backend, key, "preload", msgs, mt, pinned)
                    calls += 1
                    n_pre += 1
                    pre_tasks.append(asyncio.create_task(
                        ctrl.generate(backend, key, "preload", msgs, mt, "preload", None, extra, pinned, req=req)))
                t_last = now()
                ctrl.final_ready(key)
                g = gfin = await llm("final", evidence_log_messages(at.task.question, logged), final_max_tokens,
                                     FINAL_SCHEMA, "final")
            elif at.shape == "fanout" and preload == "chain":
                # like True, but a finished preload immediately chains another if results arrived meanwhile, so
                # results that land while a preload is in flight are covered without waiting for the next arrival
                logged, remaining, covered = [], set(tools), 0
                while remaining:
                    live = {pending} if pending is not None and not pending.done() else set()
                    done, _ = await asyncio.wait(remaining | live, return_when=asyncio.FIRST_COMPLETED)
                    for fut in sorted(done & remaining, key=lambda f: int(f.result().call_id[1:])):
                        remaining.discard(fut)
                        r = fut.result()
                        if r.error:
                            raise RuntimeError(r.error)
                        logged.append(raw_block(r.call_id, uniq[int(r.call_id[1:]) - 1], r.text))
                    if remaining and len(logged) > covered and (pending is None or pending.done()):
                        covered = len(logged)
                        pending = asyncio.create_task(llm("tool", evidence_log_messages(at.task.question, list(logged)),
                                                          1, None, "preload"))
                t_last = now()
                if pending is not None and not pending.done():
                    pending.cancel()
                await asyncio.gather(*([pending] if pending else []), return_exceptions=True)
                g = gfin = await llm("final", evidence_log_messages(at.task.question, logged), final_max_tokens,
                                     FINAL_SCHEMA, "final")
            elif at.shape == "fanout":
                logged: list[str] = []
                for fut in asyncio.as_completed(tools):
                    r = await fut
                    if r.error:
                        raise RuntimeError(r.error)
                    logged.append(raw_block(r.call_id, uniq[int(r.call_id[1:]) - 1], r.text))
                    # prefill the log so far while other calls still run; never block on it
                    if preload and len(logged) < len(tools) and (pending is None or pending.done()):
                        prio = None
                        if preload == "jit":
                            waiting = [now() - t_issue] * (len(tools) - len(logged))
                            prio = now() + expected_wait_for_all(delay, waiting)
                        pending = asyncio.create_task(llm("tool", evidence_log_messages(at.task.question, list(logged)),
                                                          1, None, "preload", prio))
                t_last = now()
                if pending is not None and not pending.done():  # final is ready: do not wait for the preload
                    pending.cancel()
                await asyncio.gather(*([pending] if pending else []), return_exceptions=True)
                g = gfin = await llm("final", evidence_log_messages(at.task.question, logged), final_max_tokens,
                                     FINAL_SCHEMA, "final")
            else:  # orchestrator with k workers, each: its search -> its own model call
                async def worker(i: int, q: str):
                    r = await tools[i]
                    if r.error:
                        raise RuntimeError(r.error)
                    w = await llm("worker", worker_messages(at.task.question, q, r.text), 128, WORKER_SCHEMA, "worker")
                    try:
                        return q, json.loads(w.text[w.text.find("{"): w.text.rfind("}") + 1]).get("report", "")
                    except (ValueError, AttributeError):
                        return q, w.text[:300]
                reports = await asyncio.gather(*(worker(i, q) for i, q in enumerate(uniq)))
                t_last = now()
                g = gfin = await llm("ofinal", orchestrator_final_messages(at.task.question, list(reports)), final_max_tokens,
                              FINAL_SCHEMA, "final")
            try:
                answer = parse_final(g.text)
            except ValueError:
                answer = ""
        status, end = "ontime", now()
    except _Drop:
        status, end = "dropped", None
    except TimeoutError:
        status, end = "late", None
    except Exception as e:  # e.g. a transport error: count the task as failed instead of aborting the whole run
        status, end, answer = "error", None, f"{type(e).__name__}: {e}"[:200]
    finally:
        for tk in tools + ([pending] if pending else []) + pre_tasks:  # a dropped / late task stops its tool calls
            tk.cancel()                                                 # and preloads (pins are held open otherwise)
        await asyncio.gather(*tools, *([pending] if pending else []), *pre_tasks, return_exceptions=True)
        if ctrl is not None:
            ctrl.task_done(key, gfin and gfin.submit_t, gfin and gfin.end_t if status == "ontime" else None,
                           gfin and gfin.cached_tokens)
    return TaskResult(at.task_id, at.shape, at.arrival, at.deadline, status, end, answer, calls, t_last,
                      gfin and gfin.submit_t, gfin and gfin.first_token_t, gfin and gfin.prompt_tokens,
                      gfin and gfin.cached_tokens, n_pre)


def make_workload(tasks: list[Task], rate: float, duration: float, seed: int, shapes=("fanout", "orchestrator"),
                  width: int = 4, slack=(1.5, 4.0), isolated: dict[str, float] | None = None) -> list[AgentTask]:
    """Poisson arrivals at `rate` tasks/s for `duration` s. Deadline = arrival + U(slack) x the task shape's
    isolated (unloaded) completion estimate."""
    rng = random.Random(seed)
    out, t, i = [], 0.0, 0
    isolated = isolated or {"fanout": 8.0, "orchestrator": 10.0}
    while True:
        t += rng.expovariate(rate)
        if t > duration:
            break
        shape = shapes[rng.randrange(len(shapes))]
        base = tasks[rng.randrange(len(tasks))]
        out.append(AgentTask(f"{base.task_id}#{i}", base, t, t + rng.uniform(*slack) * isolated[shape], shape, width))
        i += 1
    return out


async def serve(workload: list[AgentTask], backend, layer: ServingLayer, delay: DelayModel, seed: int,
                **kw: Any) -> list[TaskResult]:
    """Release each task at its arrival time (relative to now) and run all of them concurrently."""
    t0 = now()

    async def one(at: AgentTask):
        await asyncio.sleep(max(0.0, at.arrival - (now() - t0)))
        shifted = AgentTask(at.task_id, at.task, t0 + at.arrival, t0 + at.deadline, at.shape, at.width, at.value)
        r = await run_agent_task(shifted, backend, layer, delay, seed, **kw)
        r.arrival, r.deadline = at.arrival, at.deadline
        for k in ("end", "t_last_tool", "final_submit", "final_first"):
            if getattr(r, k) is not None:
                setattr(r, k, getattr(r, k) - t0)
        return r

    return list(await asyncio.gather(*(one(at) for at in workload)))
