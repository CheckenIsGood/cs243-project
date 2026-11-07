"""Event-driven runtime for one deadline-bound task (PLAN.MD Section 5).

One event loop owns all state transitions. Tool workers post immutable completion events to a queue and
never touch model context. One model generation at a time. The whole task runs under a single deadline
(`asyncio.timeout_at`), which covers planning, tools, waiting, evidence updates, controller time, retries
and the complete final answer.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .affine import AffineCostModel
from .clock import now
from .datasets import Task
from .model_client import ModelBackend, messages_tokens
from .policies import PolicyContext, make_policy
from .prompts import (
    consume_schema,
    FINAL_SCHEMA,
    plan_schema,
    consume_messages,
    evidence_log_messages,
    raw_block,
    summary_block,
    final_messages,
    salvage_consume,
    parse_final,
    parse_plan,
    plan_messages,
)
from .state import CallStatus, Consume, Finalize, Observation, Preload, RunResult, ToolCall, Wait
from .tools import SearchTool, ToolReturn, normalize_query
from .trace import Tracer


@dataclass(frozen=True)
class RuntimeConfig:
    deadline_s: float
    n_calls: int = 4
    plan_max_tokens: int = 128
    consume_out_base: int = 32
    consume_out_per_result: int = 64
    final_max_tokens: int = 192
    final_context_cap: int = 4096  # prompt + max output tokens allowed for any invocation
    dt: float = 0.1
    max_final_retries: int = 1
    charge_controller: bool = True  # virtual mode: advance the clock by measured controller CPU time
    evidence_log: bool = False  # append-only final prompt (required for Preload; see prompts.py)


class ToolFailure(Exception):
    pass


class ContractViolation(AssertionError):
    pass


async def run_task(
    task: Task,
    policy_name: str,
    controller_costs: AffineCostModel,
    delay_pmf: list[float],
    backend: ModelBackend,
    tool: SearchTool,
    cfg: RuntimeConfig,
    run_id: str,
    trace_path: str | None = None,
) -> tuple[RunResult, list[dict]]:
    """Returns the run result and its in-memory event list (also written to `trace_path`)."""
    t0 = now()
    deadline_abs = t0 + cfg.deadline_s
    tr = Tracer(trace_path, run_id, task.task_id, policy_name, t0)
    calls: dict[str, ToolCall] = {}
    return_order: list[str] = []
    summaries: dict[str, dict] = {}  # insertion order = summarization order
    logged: list[str] = []  # preloaded raw results, in preload order (evidence-log prefix)
    q: asyncio.Queue[ToolReturn] = asyncio.Queue()
    tool_tasks: list[asyncio.Task] = []
    gen_lock = asyncio.Lock()
    stats: dict[str, Any] = dict(n_model=0, ptoks=0, ctoks=0, ctrl=0.0)
    answer: str | None = None
    completion: float | None = None
    miss: str | None = None
    t_launch = t0

    async def model_call(phase: str, messages, max_tokens: int, schema, ids: list[str]):
        if gen_lock.locked():
            raise ContractViolation("concurrent model generation")
        async with gen_lock:
            inv = f"m{stats['n_model'] + 1}"
            stats["n_model"] += 1
            est = messages_tokens(backend, messages)
            tr.emit(now(), f"{phase}_start", inv=inv, call_ids=ids, est_prompt_tokens=est, max_tokens=max_tokens)
            gen = await backend.generate(messages, max_tokens, phase, schema)
            stats["ptoks"] += gen.prompt_tokens
            stats["ctoks"] += gen.completion_tokens
            tr.emit(gen.end_t, f"{phase}_end", inv=inv, call_ids=ids, prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens, finish=gen.finish_reason,
                    first_token_t=None if gen.first_token_t is None else round(gen.first_token_t - t0, 6),
                    cached_tokens=gen.cached_tokens, latency_s=round(gen.end_t - gen.submit_t, 6))
            return inv, gen

    def apply(ev: ToolReturn) -> None:
        c = calls[ev.call_id]
        if c.status is not CallStatus.RUNNING:
            raise ContractViolation(f"{c.call_id} returned twice")
        c.return_t = now() - t0
        c.retrieval_s, c.injected_delay_s = ev.retrieval_s, ev.injected_delay_s
        if ev.error:
            c.status = CallStatus.ERROR
            tr.emit(now(), "tool_error", call_id=c.call_id, error=ev.error)
            raise ToolFailure(ev.error)
        c.status, c.result_text, c.result_tokens, c.source_ids = (
            CallStatus.RETURNED, ev.text, ev.tokens, list(ev.source_ids))
        return_order.append(c.call_id)
        tr.emit(now(), "tool_return", call_id=c.call_id, tokens=ev.tokens, sources=list(ev.source_ids),
                retrieval_s=round(ev.retrieval_s, 6), injected_delay_s=round(ev.injected_delay_s, 6),
                total_s=round(c.return_t - c.launch_t, 6))

    def drain_queue() -> None:
        while not q.empty():
            apply(q.get_nowait())

    async def tool_worker(call: ToolCall, idx: int) -> None:
        try:
            ev = await tool.run(call.call_id, idx, call.query)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - surfaced as an explicit tool failure
            ev = ToolReturn(call.call_id, "", 0, (), 0.0, 0.0, error=repr(e))
        await q.put(ev)

    def raw_ids() -> list[str]:
        return [c for c in return_order if calls[c].status is CallStatus.RETURNED]

    def observation() -> Observation:
        ids = raw_ids()
        t = now()
        return Observation(
            elapsed_since_launch_s=t - t_launch,
            remaining_s=deadline_abs - t,
            n_admitted=len(calls),
            returned_raw_ids=tuple(ids),
            n_summarized=sum(c.status is CallStatus.SUMMARIZED for c in calls.values()),
            n_running=sum(c.status is CallStatus.RUNNING for c in calls.values()),
            raw_tokens=tuple(calls[c].result_tokens or 0 for c in ids),
            preloaded_ids=tuple(logged),
        )

    def log_entry_raw(c: str) -> str:
        return raw_block(c, calls[c].query, calls[c].result_text or "")

    def log_msgs(extra: list[str] = ()):
        """Evidence log: preloaded raws (preload order), [extra preloads], summaries, remaining raws."""
        ent = [log_entry_raw(c) for c in [*logged, *extra]]
        ent += [summary_block(c, summaries[c]) for c in summaries]
        ent += [log_entry_raw(c) for c in raw_ids() if c not in logged and c not in extra]
        return evidence_log_messages(task.question, ent)

    def final_msgs():
        if cfg.evidence_log:
            return log_msgs()
        summ = [(c, summaries[c]) for c in sorted(summaries)]
        raws = [(c, calls[c].query, calls[c].result_text or "") for c in raw_ids()]
        return final_messages(task.question, summ, raws)

    def guard(action, obs: Observation):
        """Runtime safety rules shared by all policies. Returns (action, override_reason)."""
        if isinstance(action, Finalize):
            if obs.n_running:
                return Wait(None), "final_before_all_returned"
            if messages_tokens(backend, final_msgs()) + cfg.final_max_tokens > cfg.final_context_cap and obs.returned_raw_ids:
                unlogged = [c for c in obs.returned_raw_ids if c not in logged]
                return Consume(((unlogged or list(obs.returned_raw_ids))[0],)), "final_context_guard"
            return action, None
        if isinstance(action, Preload):
            if not cfg.evidence_log:
                raise ContractViolation("Preload requires the evidence-log prompt layout")
            ids = [c for c in action.call_ids if calls[c].status is CallStatus.RETURNED and c not in logged]
            if not ids:
                return guard(Wait(None) if obs.n_running else Finalize(), obs)[0], "preload_nothing_new"
            # only the logged prefix + these raws are sent; it must leave room for the final answer later
            # (summaries sit after all logged raws in the log, so logged + ids stays a prefix of the final prompt)
            while ids and messages_tokens(backend, evidence_log_messages(
                    task.question, [log_entry_raw(c) for c in [*logged, *ids]])) + cfg.final_max_tokens > cfg.final_context_cap:
                ids = ids[:-1]
            if not ids:
                return guard(Wait(None) if obs.n_running else Finalize(), obs)[0], "preload_context_guard"
            return Preload(tuple(ids)), None
        if isinstance(action, Consume):
            # never summarize a preloaded raw: its prefill is already paid and it is folded by the final answer
            ids = [c for c in action.call_ids if calls[c].status is CallStatus.RETURNED and c not in logged]
            if not ids:
                return (Wait(None), "consume_nothing_ready") if obs.n_running else (Finalize(), "consume_nothing_ready")
            reason = None
            while len(ids) > 1:
                msgs = consume_messages(task.question, [(c, calls[c].query, calls[c].result_text or "") for c in ids])
                out = cfg.consume_out_base + cfg.consume_out_per_result * len(ids)
                if messages_tokens(backend, msgs) + out <= cfg.final_context_cap:
                    break
                ids, reason = ids[:-1], "consume_context_guard"
            return Consume(tuple(ids)), reason
        if isinstance(action, Wait) and not obs.n_running:
            return guard(Finalize(), obs)[0], "wait_with_nothing_running"
        return action, None

    try:
        async with asyncio.timeout_at(deadline_abs):
            tr.emit(t0, "task_start", deadline_s=cfg.deadline_s, question=task.question, backend=backend.name)
            # ---- planning (charged to the deadline) --------------------------------------------
            tr.emit(now(), "planning_start")
            _, gen = await model_call("plan", plan_messages(task.question, cfg.n_calls), cfg.plan_max_tokens,
                                      plan_schema(cfg.n_calls), [])
            try:
                queries = parse_plan(gen.text)
            except ValueError as e:
                queries = []
                tr.emit(now(), "plan_parse_failed", error=str(e))
            seen, uniq = set(), []
            for qy in queries:
                k = normalize_query(qy)
                if k and k not in seen:
                    seen.add(k)
                    uniq.append(qy)
            uniq = uniq[: cfg.n_calls] or [task.question]
            tr.emit(now(), "planning_end", queries=uniq)
            # ---- launch the fixed admitted set together (Experiment A) -------------------------
            t_launch = now()
            for i, qy in enumerate(uniq):
                c = ToolCall(f"c{i + 1}", qy, launch_t=t_launch - t0)
                calls[c.call_id] = c
                tr.emit(t_launch, "tool_launch", call_id=c.call_id, query=qy)
                tool_tasks.append(asyncio.create_task(tool_worker(c, i)))
            tc = time.perf_counter()
            policy = make_policy(policy_name, PolicyContext(len(calls), deadline_abs - t_launch, controller_costs,
                                                             tuple(delay_pmf), cfg.dt))
            setup = time.perf_counter() - tc
            stats["ctrl"] += setup
            tr.emit(now(), "policy_setup", seconds=round(setup, 6), remaining_at_launch_s=round(deadline_abs - t_launch, 6))
            if cfg.charge_controller:
                await asyncio.sleep(setup)
            final_attempts = 0
            # ---- decision loop ------------------------------------------------------------------
            while True:
                drain_queue()
                obs = observation()
                tc = time.perf_counter()
                proposed = policy.decide(obs)
                dc = time.perf_counter() - tc
                stats["ctrl"] += dc
                if cfg.charge_controller:
                    await asyncio.sleep(dc)
                action, override = guard(proposed, obs)
                tr.emit(now(), "decision", step=int(obs.elapsed_since_launch_s / cfg.dt + 1e-9),
                        returned=obs.n_returned, summarized=obs.n_summarized, running=obs.n_running,
                        raw_ready=list(obs.returned_raw_ids), proposed=repr(proposed), action=repr(action),
                        override=override, controller_s=round(dc, 6))
                if isinstance(action, Consume):
                    ids = list(action.call_ids)
                    msgs = consume_messages(task.question, [(c, calls[c].query, calls[c].result_text or "") for c in ids])
                    inv, gen = await model_call("consume", msgs, cfg.consume_out_base + cfg.consume_out_per_result * len(ids),
                                                consume_schema(ids), ids)
                    # Never leave a consumed batch raw: a retry loop on the same results burned whole deadlines
                    # at Gate 1. Salvage complete records; mechanical fallback for the rest (logged).
                    recs, fallback_ids = salvage_consume(gen.text, ids, {c: calls[c].result_text or "" for c in ids})
                    tr.emit(now(), "consume_records", inv=inv, n_facts={c: len(recs[c]["facts"]) for c in ids})
                    if fallback_ids or gen.finish_reason == "length":
                        tr.emit(now(), "consume_salvaged", inv=inv, call_ids=ids, fallback_ids=fallback_ids,
                                finish=gen.finish_reason)
                    for c in ids:
                        if calls[c].status is not CallStatus.RETURNED:
                            raise ContractViolation(f"{c} consumed twice")
                        calls[c].status, calls[c].consumed_by = CallStatus.SUMMARIZED, inv
                        summaries[c] = recs[c]
                elif isinstance(action, Preload):
                    ids = list(action.call_ids)
                    msgs = evidence_log_messages(task.question, [log_entry_raw(c) for c in [*logged, *ids]])
                    await model_call("preload", msgs, 1, None, ids)
                    logged.extend(ids)
                elif isinstance(action, Wait):
                    tr.emit(now(), "wait_start", max_s=action.max_s)
                    try:
                        apply(await asyncio.wait_for(q.get(), action.max_s))
                    except TimeoutError:
                        pass
                    tr.emit(now(), "wait_end")
                else:  # Finalize
                    raws = raw_ids()
                    if final_attempts == 0:
                        tr.emit(now(), "drain_start")
                    inv, gen = await model_call("final", final_msgs(), cfg.final_max_tokens, FINAL_SCHEMA, raws)
                    try:
                        if gen.finish_reason == "length":
                            raise ValueError("final answer truncated")
                        answer = parse_final(gen.text)
                    except ValueError as e:
                        final_attempts += 1
                        tr.emit(now(), "final_parse_failed", inv=inv, error=str(e), attempt=final_attempts)
                        if final_attempts > cfg.max_final_retries:
                            miss = "malformed_final"
                            break
                        continue
                    for c in raws:
                        calls[c].status, calls[c].consumed_by = CallStatus.FOLDED, inv
                    completion = now() - t0
                    break
    except TimeoutError:
        miss = "deadline"
        tr.emit(deadline_abs, "deadline_miss")
    except ToolFailure:
        miss = "tool_error"
    finally:
        pending = [t for t in tool_tasks if not t.done()]
        if pending:
            tr.emit(now(), "cleanup_start", cancel_tools=len(pending))
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            tr.emit(now(), "cleanup_end")

    consumed_all = bool(calls) and all(c.status in (CallStatus.SUMMARIZED, CallStatus.FOLDED) for c in calls.values())
    valid = miss is None and answer is not None and consumed_all and completion is not None \
        and completion <= cfg.deadline_s + 1e-9
    if miss is None and not valid:
        miss = "contract_incomplete"
    tr.emit(now(), "task_end", valid=valid, miss_reason=miss, answer=answer,
            completion_s=None if completion is None else round(completion, 6))
    tr.close()
    return RunResult(
        run_id=run_id, task_id=task.task_id, policy=policy_name, deadline_s=cfg.deadline_s, valid=valid,
        miss_reason=miss, completion_s=completion, answer=answer, n_calls=len(calls),
        n_summarized=sum(c.status is CallStatus.SUMMARIZED for c in calls.values()),
        n_folded=sum(c.status is CallStatus.FOLDED for c in calls.values()),
        n_model_calls=stats["n_model"], prompt_tokens=stats["ptoks"], completion_tokens=stats["ctoks"],
        controller_s=stats["ctrl"], trace_path=str(trace_path or ""),
    ), tr.events
