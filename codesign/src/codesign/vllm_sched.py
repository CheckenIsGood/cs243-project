"""CodesignScheduler: the engine rules of docs/vllm_codesign_spec.md, as a vLLM 0.32 scheduler subclass.

Load it with `vllm serve ... --scheduling-policy priority --scheduler-cls codesign.vllm_sched.CodesignScheduler`.
Configure it with the env var CODESIGN (a JSON object); every switch defaults to stock behaviour:

  admission       "stock" | "prefix"    E1: longest priority prefix that fits (reservation = prompt + max_tokens),
                                        head-of-line stop, kicks of running requests outside the prefix [proved]
  prefill_top1    bool                  E2: only the highest-priority admitted prefill gets prefill tokens [proved]
  prefill_chunk   int | null            E2: long_prefill_token_threshold (a/alpha ~ 244 on A40/Qwen3-8B)
  decode_top_m    int (0 = all)         E2: decode only the top m admitted decoders
  preload_gate    bool                  E4: requests with xarg cd_preload run only when no admitted request
                                        needs prefill, use only unreserved blocks, are kicked first [proved]
  protect_queued  bool                  E6: every step, a queued request's cached prefix blocks move to the
                                        MRU end of the free queue (no reference taken, so P2 holds) [benefit]
  hold            "lru" | "ttl" | "pin" after a request finishes: stock LRU; "ttl" = soft hold (xarg cd_ttl
                                        seconds; afterwards its blocks move to the eviction head) [benefit,
                                        safe by R5c]; "pin" = Continuum-like hard pin for cd_ttl seconds
                                        (blocks referenced; unpinned at expiry, when the task's next request
                                        runs, or to avoid deadlock, latest task first) [baseline]
  pin_boost       bool                  "pin" baseline: a request of a task with a live pin is ranked first
  stats_path      str | null            write counters as JSON here every stats_every steps

Requests carry per-request info in `vllm_xargs` (sampling_params.extra_args):
  cd_task (str), cd_preload (0/1), cd_ttl (float seconds).

All the logic that decides *which* requests are admitted, kicked or hidden is in `plan_step`, a pure function of
plain tuples, so it is unit-tested on CPU without vLLM (tests/test_vllm_sched_logic.py). The subclass only maps
vLLM objects to tuples and applies the plan.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# ----------------------------------------------------------------------------------------------------------------
# Pure planning logic (no vLLM imports)


@dataclass(frozen=True)
class Req:
    rid: str
    priority: int
    arrival: float
    reserve_blocks: int  # ceil((prompt + max_tokens) / block_size): the reservation l+ of the theory
    running: bool  # currently in scheduler.running
    needs_prefill: bool  # more than one token left to compute (fresh, chunked or kicked)
    preload: bool = False

    @property
    def key(self):
        return (self.priority, self.arrival, self.rid)


@dataclass
class StepPlan:
    admitted: list[str] = field(default_factory=list)  # priority order
    kick: list[str] = field(default_factory=list)  # running requests to preempt now
    hide_waiting: list[str] = field(default_factory=list)  # waiting requests not schedulable this step
    hide_running: list[str] = field(default_factory=list)  # running requests that get no tokens this step
    reserved_blocks: int = 0


def plan_step(reqs: Sequence[Req], total_blocks: int, *, admission: str = "prefix", prefill_top1: bool = False,
              decode_top_m: int = 0, preload_gate: bool = False) -> StepPlan:
    """One scheduling step's plan.

    E1 (admission="prefix"): walk the non-preload requests in (priority, arrival) order, summing reservations, and
    stop at the first one that does not fit (head-of-line stop; never skip ahead to a smaller one). The prefix is
    admitted; running requests outside it are kicked, waiting ones are hidden.
    E2: among admitted requests that need prefill, only the first keeps its prefill (prefill_top1); among admitted
    decoders, only the first decode_top_m run.
    E4: preloads are hidden while any admitted request needs prefill, and kicked (lowest first) when their
    reservations do not fit in the blocks the prefix leaves free.
    admission="stock" leaves admission to vLLM and only applies E2/E4 to the running/waiting sets as they are.
    """
    plan = StepPlan()
    main = sorted((r for r in reqs if not r.preload), key=lambda r: r.key)
    pre = sorted((r for r in reqs if r.preload), key=lambda r: r.key)
    if admission == "prefix":
        used = 0
        for r in main:
            if used + r.reserve_blocks > total_blocks:
                break
            used += r.reserve_blocks
            plan.admitted.append(r.rid)
        adm = set(plan.admitted)
        for r in main:
            if r.rid not in adm:
                (plan.kick if r.running else plan.hide_waiting).append(r.rid)
        plan.reserved_blocks = used
    elif admission == "stock":
        plan.admitted = [r.rid for r in main]
        plan.reserved_blocks = sum(r.reserve_blocks for r in main if r.running)
    else:
        raise ValueError(f"unknown admission {admission!r}")
    adm = set(plan.admitted)
    adm_reqs = [r for r in main if r.rid in adm]
    if prefill_top1:
        pf = [r for r in adm_reqs if r.needs_prefill]
        for r in pf[1:]:
            (plan.hide_running if r.running else plan.hide_waiting).append(r.rid)
    if decode_top_m > 0:
        dec = [r for r in adm_reqs if r.running and not r.needs_prefill]
        plan.hide_running.extend(r.rid for r in dec[decode_top_m:])
    if pre and preload_gate:
        busy = any(r.needs_prefill for r in adm_reqs)
        free = total_blocks - plan.reserved_blocks
        used = 0
        for r in pre:  # highest-priority preload first; the tail is kicked / hidden
            fits = used + r.reserve_blocks <= free
            if fits:
                used += r.reserve_blocks
            if not fits:
                (plan.kick if r.running else plan.hide_waiting).append(r.rid)
            elif busy and r.needs_prefill:
                (plan.hide_running if r.running else plan.hide_waiting).append(r.rid)
    return plan


def blocks_for(tokens: int, block_size: int) -> int:
    return max(1, math.ceil(tokens / block_size))


# ----------------------------------------------------------------------------------------------------------------
# vLLM subclass

try:  # vLLM is only installed on the GPU pod
    from vllm.v1.core.sched.scheduler import Scheduler as _VllmScheduler
    from vllm.v1.request import RequestStatus
except ImportError:  # pragma: no cover - CPU tests import only the pure part
    _VllmScheduler = object
    RequestStatus = None


def _xargs(request) -> dict:
    sp = getattr(request, "sampling_params", None)
    return (getattr(sp, "extra_args", None) or {}) if sp is not None else {}


DEFAULTS = dict(admission="stock", prefill_top1=False, prefill_chunk=None, decode_top_m=0, preload_gate=False,
                protect_queued=False, hold="lru", pin_boost=False, stats_path=None, stats_every=200)


class CodesignScheduler(_VllmScheduler):  # type: ignore[misc, valid-type]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(DEFAULTS, **json.loads(os.environ.get("CODESIGN", "{}") or "{}"))
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown CODESIGN keys {sorted(unknown)}")
        self.cd = cfg
        if cfg["admission"] != "stock" or cfg["pin_boost"]:
            assert self.policy.name == "PRIORITY", "codesign admission needs --scheduling-policy priority"
        if cfg["prefill_chunk"]:
            self.scheduler_config.long_prefill_token_threshold = int(cfg["prefill_chunk"])
        self.cd_total_blocks = self.kv_cache_manager.block_pool.num_gpu_blocks - 1  # minus the null block
        self.cd_groups = list(range(len(self.kv_cache_config.kv_cache_groups)))
        self.cd_ttl: list[tuple[float, list]] = []  # soft hold: (expiry, blocks)
        self.cd_pins: dict[str, dict] = {}  # task -> {expiry, blocks, task_arrival}
        self.cd_task_arrival: dict[str, float] = {}
        self.cd_stats = dict(steps=0, kicks=0, kick_steps=0, kicks_without_arrival=0, alloc_preempts=0,
                             hidden_waiting=0, hidden_running=0, preload_hidden=0, protect_moves=0,
                             ttl_demotions=0, pins=0, unpins_expiry=0, unpins_used=0, unpins_deadlock=0,
                             reserve_violations=0, max_admitted=0, max_running=0)
        self.cd_seen: set[str] = set()
        self.cd_last_arrivals = 0
        self.cd_arrivals = 0
        print(f"[codesign] config {json.dumps(cfg)} total_blocks={self.cd_total_blocks}", flush=True)

    # -- helpers ------------------------------------------------------------------------------------------------
    def _cd_req(self, r, running: bool) -> Req:
        x = _xargs(r)
        reserve = blocks_for(r.num_prompt_tokens + r.max_tokens, self.block_size)
        needs_prefill = (r.num_tokens - r.num_computed_tokens) > 1
        return Req(r.request_id, int(r.priority), float(r.arrival_time), reserve, running, needs_prefill,
                   bool(int(x.get("cd_preload", 0) or 0)))

    def _cd_block_list(self, request_id: str) -> list:
        blocks = self.kv_cache_manager.get_blocks(request_id).blocks
        return [b for grp in blocks for b in grp if not getattr(b, "is_null", False)]

    def _cd_free_queue(self):
        return self.kv_cache_manager.block_pool.free_block_queue

    def _cd_in_free_queue(self, b) -> bool:
        return b.ref_cnt == 0 and b.prev_free_block is not None and b.next_free_block is not None

    # -- E6: protect queued prefixes --------------------------------------------------------------------------
    def _cd_protect_queued(self, waiting: Iterable) -> None:
        pool, fq = self.kv_cache_manager.block_pool, self._cd_free_queue()
        for r in waiting:
            for h in getattr(r, "block_hashes", []) or []:
                blks = pool.get_cached_block(h, self.cd_groups)
                if blks is None:
                    break  # prefix chain ends at the first miss
                for b in blks:
                    if self._cd_in_free_queue(b):
                        fq.remove(b)
                        fq.append(b)
                        self.cd_stats["protect_moves"] += 1

    # -- E5 / pin baseline ------------------------------------------------------------------------------------
    def _cd_expire(self, now: float) -> None:
        fq = self._cd_free_queue()
        keep = []
        for exp, blks in self.cd_ttl:
            if exp > now:
                keep.append((exp, blks))
                continue
            # tail of the prefix goes first; skip blocks reused for other content since
            demote = [b for b, h in reversed(blks) if h is not None and b.block_hash == h and self._cd_in_free_queue(b)]
            for b in demote:
                fq.remove(b)
            fq.prepend_n(demote)
            self.cd_stats["ttl_demotions"] += len(demote)
        self.cd_ttl = keep
        if not self.cd_pins:
            return
        waiting_tasks = {_xargs(r).get("cd_task") for r in self.waiting}
        running_tasks = {_xargs(r).get("cd_task") for r in self.running}
        for task in list(self.cd_pins):
            p = self.cd_pins[task]
            if task in running_tasks:
                self._cd_unpin(task, "unpins_used")
            elif p["expiry"] <= now and task not in waiting_tasks:
                self._cd_unpin(task, "unpins_expiry")

    def _cd_unpin(self, task: str, why: str) -> None:
        p = self.cd_pins.pop(task)
        self.kv_cache_manager.block_pool.free_blocks(p["blocks"])
        self.cd_stats[why] += 1

    def _cd_unpin_for_deadlock(self) -> None:
        """Continuum: if the head waiting request cannot be allocated, unpin latest-arrival tasks until it fits."""
        if not self.cd_pins or not self.waiting:
            return
        head = self.waiting.peek_request()
        need = blocks_for(head.num_tokens, self.block_size)
        pool = self.kv_cache_manager.block_pool
        for task in sorted(self.cd_pins, key=lambda t: -self.cd_pins[t]["task_arrival"]):
            if pool.get_num_free_blocks() >= need:
                break
            self._cd_unpin(task, "unpins_deadlock")

    def _free_request(self, request, delay_free_blocks: bool = False):
        hold = self.cd["hold"]
        x = _xargs(request)
        ttl = float(x.get("cd_ttl", 0) or 0)
        blks = None
        if hold in ("ttl", "pin") and ttl > 0:
            blks = self._cd_block_list(request.request_id) or None
        if hold == "pin" and blks:
            task = x.get("cd_task")
            if task is not None:
                if task in self.cd_pins:
                    self._cd_unpin(task, "unpins_used")
                self.kv_cache_manager.block_pool.touch(blks)  # hold one reference until unpinned
                self.cd_pins[task] = dict(expiry=time.monotonic() + ttl, blocks=blks,
                                          task_arrival=self.cd_task_arrival.get(task, request.arrival_time))
                self.cd_stats["pins"] += 1
        out = super()._free_request(request, delay_free_blocks)
        if hold == "ttl" and blks:
            self.cd_ttl.append((time.monotonic() + ttl, [(b, b.block_hash) for b in blks]))
        return out

    def add_request(self, request) -> None:
        x = _xargs(request)
        task = x.get("cd_task")
        if task is not None:
            self.cd_task_arrival.setdefault(task, request.arrival_time)
            if self.cd["pin_boost"] and task in self.cd_pins:
                request.priority = int(request.priority) - 10**15  # pinned programs first (Continuum)
        self.cd_arrivals += 1
        super().add_request(request)

    # -- the step ---------------------------------------------------------------------------------------------
    def schedule(self, *args, **kwargs):
        cd = self.cd
        st = self.cd_stats
        st["steps"] += 1
        now = time.monotonic()
        if cd["hold"] in ("ttl", "pin"):
            self._cd_expire(now)
        if cd["hold"] == "pin":
            self._cd_unpin_for_deadlock()
        waiting = list(self.waiting)
        if cd["protect_queued"]:
            self._cd_protect_queued(waiting)
        active = (cd["admission"] != "stock" or cd["prefill_top1"] or cd["decode_top_m"] or cd["preload_gate"])
        hidden_w, hidden_r = [], []
        n_kick = 0
        if active:
            by_id = {r.request_id: r for r in waiting}
            by_id.update({r.request_id: r for r in self.running})
            reqs = [self._cd_req(r, False) for r in waiting] + [self._cd_req(r, True) for r in self.running]
            plan = plan_step(reqs, self.cd_total_blocks, admission=cd["admission"], prefill_top1=cd["prefill_top1"],
                             decode_top_m=cd["decode_top_m"], preload_gate=cd["preload_gate"])
            st["max_admitted"] = max(st["max_admitted"], len(plan.admitted))
            if plan.reserved_blocks > self.cd_total_blocks:
                st["reserve_violations"] += 1
            # kicks (E1/E4): preempt now, through vLLM's own path (frees blocks; recompute later)
            for rid in plan.kick:
                r = by_id[rid]
                if r.status != RequestStatus.RUNNING or not self._request_blocks_can_be_freed(r):
                    continue
                self.running.remove(r)
                self._preempt_request(r, now)
                n_kick += 1
            # hide: requests that get nothing this step are taken out and put back after the stock step
            hw = set(plan.hide_waiting) | set(plan.kick)  # kicked requests are back in waiting: not this step
            if hw:
                hidden_w = [r for r in self.waiting if r.request_id in hw]
                self.waiting.remove_requests(hidden_w)
            hr = set(plan.hide_running)
            if hr:
                hidden_r = [r for r in self.running if r.request_id in hr]
                self.running = [r for r in self.running if r.request_id not in hr]
            if cd["admission"] != "stock" or cd["prefill_top1"]:
                self.running.sort(key=lambda r: (r.priority, r.arrival_time))
            st["hidden_waiting"] += len(hidden_w)
            st["hidden_running"] += len(hidden_r)
            st["preload_hidden"] += sum(1 for r in hidden_w + hidden_r if _xargs(r).get("cd_preload"))
        if n_kick:
            st["kicks"] += n_kick
            st["kick_steps"] += 1
            if self.cd_arrivals == self.cd_last_arrivals:
                st["kicks_without_arrival"] += 1  # theory: kicks only at availability events
        self.cd_last_arrivals = self.cd_arrivals
        self.cd_in_stock = True
        try:
            out = super().schedule(*args, **kwargs)
        finally:
            self.cd_in_stock = False
            if hidden_r:
                self.running.extend(hidden_r)
            for r in hidden_w:
                if not r.is_finished():
                    self.waiting.add_request(r)
        st["max_running"] = max(st["max_running"], len(self.running))
        if cd["stats_path"] and st["steps"] % cd["stats_every"] == 0:
            self.cd_dump()
        return out

    def _preempt_request(self, request, timestamp, drop_stale_output: bool = False):
        if getattr(self, "cd_in_stock", False):  # stock preemption on allocation failure (0 expected under E1)
            self.cd_stats["alloc_preempts"] += 1
        return super()._preempt_request(request, timestamp, drop_stale_output)

    def cd_dump(self) -> None:
        path = self.cd["stats_path"]
        if path:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(dict(self.cd_stats, config=self.cd, t=time.time()), f)
            os.replace(tmp, path)
