"""Client-side KV-aware preload controller (docs/controller_theory.md Section 7), for the live serving layer.

The controller runs entirely in the client: it sees only its own requests (we are the server's only tenant), their
usage fields (prompt_tokens, cached_tokens), first-token times, the tool-latency law, and the server's KV capacity
M. It never modifies the engine. A "pin" is a low-priority preload kept open (ignore_eos, max_tokens from the wait
quantile), so its prefix blocks stay referenced until it is cancelled; an "unpinned" preload is the 'low' mode's
1-token request.

Every tool return of a fan-out task (with other calls outstanding) is a decision (`decide` in controller_theory):
PIN if Proposition C says holding beats LRU and the pin fits the KV budget; else UNPINNED if the TTL survival
estimate exceeds the break-even survival sigma* (Theorem BE') and the cached_tokens feedback has not shown
sigma_hat <= sigma*; else SKIP. New here (not in the theory doc's M4 controller): an exploration probe. When only
the feedback blocks an unpinned preload, every `explore_every`-th such opportunity is still preloaded, so sigma_hat
keeps being measured and can recover.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from .clock import now
from .controller_theory import (CtrlParams, CtrlState, TaskView, cond_delay_sample, decide, p_survive_unpinned,
                                pin_value, sigma_star, step_time_hat, yield_before_submit)
from .preload_theory import Calib

INF = float("inf")
LOWEST_PRIORITY = 1e9  # same value as multiagent.LOWEST_PRIORITY


@dataclass
class _Req:
    task: str
    kind: str  # plan | preload | final | other
    prompt: float  # estimated prompt tokens (server units)
    max_tokens: int
    submit: float
    first: float | None = None
    done: bool = False
    pinned: bool = False
    aio: asyncio.Task | None = None  # the asyncio task running this request (pins and preloads), for cancellation
    approx: float = 0.0  # prompt tokens by the client's approximate count


@dataclass
class _TaskState:
    arrival: float
    plan_done: float | None = None
    issue_t: float | None = None
    n_tools: int = 0
    returned: int = 0
    pend: _Req | None = None  # latest preload sent
    pin: _Req | None = None
    pin_prev: _Req | None = None  # superseded pin, cancelled when the new pin's first token arrives
    completed_pre: int = 0  # tokens covered by the largest preload whose first token arrived (16-token blocks)
    unpinned: int = 0
    pins: int = 0


@dataclass
class PreloadController:
    M: int  # server KV capacity (tokens)
    delay: tuple  # DelayModel (lo, hi, slow_p, slow_lo, slow_hi)
    cal: Calib = field(default_factory=Calib)
    params: CtrlParams | None = None
    explore_every: int = 10  # KNOB (new, set a priori): 1 in N feedback-blocked opportunities is still preloaded
    block: int = 16
    reqs: dict = field(default_factory=dict)
    tasks: dict = field(default_factory=dict)
    stats: Counter = field(default_factory=Counter)
    decisions: list = field(default_factory=list)

    def __post_init__(self):
        if self.params is None:
            self.params = CtrlParams(M=self.M)
        self._fin: deque = deque()  # (t, prefill GPU s) of finished requests: u_hat
        self._dead: deque = deque()  # (t, tokens released): v_dead for the TTL estimate
        self._arr: deque = deque()  # task arrival times: lambda_hat
        self._xcrit: list = []  # plan latency + final latency per finished task
        self._sig: list = []  # (preloaded tokens, survived tokens) per final of a task with unpinned preloads
        self._ratio = [1.0, 1.0]  # server tokens / approx tokens (sum real, sum approx), calibrated online
        self._t0: float | None = None
        self._probe = 0

    # ------------------------------------------------------------------ estimators (all client-observable)
    def _span(self, t):
        return min(self.params.window_s, max(t - (self._t0 if self._t0 is not None else t), 1.0))

    def _trim(self, q, t):
        while q and q[0][0] < t - self.params.window_s:
            q.popleft()

    def scale(self, approx: float) -> float:
        return approx * self._ratio[0] / self._ratio[1]

    def _gen(self, r: _Req, t: float, u: float) -> float:
        if r.first is None:
            return 0.0
        return min(r.max_tokens, 1 + (t - r.first) / step_time_hat(u, self.cal))

    def demand(self, t: float, u: float | None = None) -> float:
        """D_hat: prompt + generated tokens of in-flight requests; a task's preload/final prefix counted once."""
        u = self.u_hat(t) if u is None else u
        tot, per = 0.0, {}
        for r in self.reqs.values():
            x = r.prompt + self._gen(r, t, u)
            if r.kind in ("preload", "final"):
                per[r.task] = max(per.get(r.task, 0.0), x)
            else:
                tot += x
        return tot + sum(per.values())

    def u_hat(self, t: float) -> float:
        self._trim(self._fin, t)
        return sum(x for _, x in self._fin) / self._span(t)

    def rho_s_hat(self, t: float, u: float) -> float:
        while self._arr and self._arr[0] < t - self.params.window_s:
            self._arr.popleft()
        lam = len(self._arr) / self._span(t)
        if len(self._xcrit) >= self.params.min_obs:
            last = self._xcrit[-50:]
            X = sum(last) / len(last) * (1 - min(u, 0.95))
        else:  # prior before feedback: isolated plan decode + mean final prefill (cost model)
            X = 128 * (self.cal.a + self.cal.c) + self.cal.prefill_cost(7000, 0)
        return lam * X

    def ttl_hat(self, t: float, demand: float) -> float:
        self._trim(self._dead, t)
        v = sum(x for _, x in self._dead) / self._span(t)
        return INF if v <= 0 else max(0.0, (self.M - demand) / v)

    def sigma_hat(self) -> tuple[float, int]:
        if not self._sig:
            return 1.0, 0
        last = self._sig[-50:]
        pre = sum(a for a, _ in last)
        return (sum(b for _, b in last) / pre if pre else 1.0), len(self._sig)

    def wait_samples(self, key: str, t: float) -> list[float]:
        """64 draws of the time until the task's last outstanding tool returns, from the delay law only."""
        s = self.tasks[key]
        el = t - s.issue_t
        nrem = s.n_tools - s.returned
        rng = random.Random(f"{key}|{s.returned}")
        xs = sorted(max(cond_delay_sample(rng, el, self.delay) - el for _ in range(nrem)) if nrem else 0.0
                    for _ in range(64))
        return xs

    def pins_view(self, t: float, u: float, rs: float) -> dict:
        out = {}
        for key, s in self.tasks.items():
            p = s.pin
            if p is None or p.done or p.first is None:
                continue
            L = p.prompt
            if s.returned < s.n_tools:
                xs = self.wait_samples(key, t)
                w = sum(xs) / len(xs)
                out[key] = (L, pin_value(TaskView(int(L), int(L), w, w, 0.0, False), u, self.cal, rs) /
                            (L * max(w, 1e-3)))
            else:
                out[key] = (L, INF)  # final already submitted: its pin is needed now
        return out

    # ------------------------------------------------------------------ request lifecycle
    def register(self, backend, key: str, kind: str, messages, max_tokens: int, pinned: bool = False) -> _Req:
        """Register a request synchronously (before its coroutine starts), so decisions taken in the same event-loop
        turn already see it. A preload becomes the task's pending preload (and its pin if pinned)."""
        t = now()
        if self._t0 is None:
            self._t0 = t
        approx = sum(backend.count_tokens(m["content"]) + 4 for m in messages)
        r = _Req(key, kind, self.scale(approx), max_tokens, t, pinned=pinned)
        r.approx = approx
        self.reqs[id(r)] = r
        s = self.tasks.get(key)
        if kind == "preload" and s is not None:
            s.pend = r
            if pinned:
                s.pin_prev, s.pin = s.pin, r
                s.pins += 1
            else:
                s.unpinned += 1
        return r

    async def generate(self, backend, key: str, kind: str, messages, max_tokens: int, phase: str, schema=None,
                       extra: dict | None = None, pinned: bool = False, on_first=None, req: _Req | None = None):
        """Run one request through the backend with bookkeeping. `kind` in plan | preload | final."""
        r = req or self.register(backend, key, kind, messages, max_tokens, pinned)
        rid, approx = id(r), r.approx
        r.aio = asyncio.current_task()
        if r.done:  # cancelled before it started
            raise asyncio.CancelledError()

        def first_cb():
            r.first = now()
            self._on_first(r)
            if on_first is not None:
                on_first()

        ex = dict(extra or {})
        ex["_on_first"] = first_cb
        g = None
        try:
            g = await backend.generate(messages, max_tokens, phase, schema, ex)
            return g
        finally:
            r.done = True
            self.reqs.pop(rid, None)
            t2 = now()
            u = self.u_hat(t2)
            if g is not None:
                self._ratio[0] += g.prompt_tokens
                self._ratio[1] += approx
                computed = g.prompt_tokens - (g.cached_tokens or 0)
                self._fin.append((t2, self.cal.prefill_cost(computed, g.cached_tokens or 0)))
                self._dead.append((t2, g.prompt_tokens + g.completion_tokens))
            else:  # cancelled: its computed prefix (if any) becomes dead in LRU
                self._dead.append((t2, (r.prompt if r.first is not None else 0) + self._gen(r, t2, u)))
            if kind == "preload" and g is not None:  # coverage from the server's own count
                s = self.tasks.get(key)
                if s is not None:
                    cov = max(0, g.prompt_tokens - self.cal.suffix) // self.block * self.block
                    s.completed_pre = max(s.completed_pre, cov)

    def _on_first(self, r: _Req):
        s = self.tasks.get(r.task)
        if s is None:
            return
        if r.kind == "preload":
            cov = int(max(0, r.prompt - self.cal.suffix)) // self.block * self.block
            s.completed_pre = max(s.completed_pre, cov)
            if r.pinned and s.pin_prev is not None and s.pin_prev is not r:
                self._cancel(s.pin_prev, "superseded")
                s.pin_prev = None
        elif r.kind == "final":
            for p in (s.pin, s.pin_prev):
                self._cancel(p, "final_started")
            s.pin = s.pin_prev = None

    def _cancel(self, r: _Req | None, why: str):
        if r is None or r.done:
            return
        self.stats["cancel_" + why] += 1
        if r.aio is None:  # registered but its coroutine has not started: it will see r.done and exit
            r.done = True
            self.reqs.pop(id(r), None)
        elif not r.aio.done():
            r.aio.cancel()

    # ------------------------------------------------------------------ agent-facing events
    def task_arrived(self, key: str):
        t = now()
        if self._t0 is None:
            self._t0 = t
        self._arr.append(t)
        self.tasks[key] = _TaskState(arrival=t)

    def tools_issued(self, key: str, n: int):
        s = self.tasks[key]
        s.plan_done = s.issue_t = now()
        s.n_tools = n

    def preload_busy(self, key: str) -> bool:
        p = self.tasks[key].pend
        return p is not None and not p.done and p.first is None

    def on_return(self, key: str, log_tokens_approx: float) -> str:
        """A tool result of task `key` arrived and others are outstanding. Returns 'skip' | 'pin' | 'unpinned'
        (with max_tokens in self.last_pin_tokens for 'pin'); cancels displaced pins itself."""
        s = self.tasks[key]
        s.returned += 1
        if self.preload_busy(key):
            self.stats["skip_busy"] += 1
            return "skip"
        t = now()
        xs = self.wait_samples(key, t)
        w = sum(xs) / len(xs)
        wq = xs[min(len(xs) - 1, int(self.params.pin_quantile * len(xs)))]
        L = self.scale(log_tokens_approx)
        pin_tok = s.pin.prompt if (s.pin is not None and not s.pin.done) else 0
        u = self.u_hat(t)
        sh, nobs = self.sigma_hat()
        st = CtrlState(self.demand(t, u), u, 0.0, sh, nobs, {}, 0.0)
        st.rho_s = self.rho_s_hat(t, u)
        st.pins = self.pins_view(t, u, st.rho_s)
        st.ttl_hat = self.ttl_hat(t, st.demand)
        # a prefix larger than the free KV cannot stay resident whatever the TTL says (guard added for the live
        # client: the fluid TTL ignores the prefix's own size)
        p_un = 0.0 if st.demand + L > self.M else p_survive_unpinned(xs, st.ttl_hat)
        tv = TaskView(int(L), int(pin_tok), w, wq, p_un, False)
        act, victims = decide(tv, st, self.params, self.cal)
        thr = sigma_star(u, st.rho_s)
        if act == "skip" and tv.p_unpinned > thr and nobs >= self.params.min_obs and sh <= thr:
            self._probe += 1
            if self._probe % self.explore_every == 0:
                act = "unpinned"
                self.stats["explore"] += 1
        self.decisions.append(dict(t=round(t - self._t0, 3), task=key, act=act, u=round(u, 4),
                                   rho_s=round(st.rho_s, 4), thr=round(thr, 4), ttl=round(min(st.ttl_hat, 1e6), 2),
                                   p=round(tv.p_unpinned, 3), sigma=round(sh, 3), nobs=nobs, D=int(st.demand),
                                   L=int(L), wait=round(w, 2)))
        for tj in victims:
            self._cancel(self.tasks[tj].pin, "displaced")
            self.tasks[tj].pin = None
        self.stats[act] += 1
        self.last_pin_tokens = int(math.ceil(wq / self.cal.a)) + 1
        return act

    def final_ready(self, key: str):
        """The last tool returned: cancel a preload still in prefill (its computed blocks stay cached)."""
        s = self.tasks[key]
        p = s.pend
        if p is not None and not p.done and p.first is None:
            self._cancel(p, "final_ready")
            if p is s.pin:
                s.pin, s.pin_prev = s.pin_prev, None

    def yield_for(self, key: str, need_approx: float):
        """Before a plan/final submit: cancel other tasks' pins (lowest index first) so it passes the full-prompt
        admission gate demand + need <= M."""
        t = now()
        need = self.scale(need_approx)
        s = self.tasks.get(key)
        if s is not None and s.pin is not None and not s.pin.done:
            need = max(0.0, need - s.pin.prompt)
        u = self.u_hat(t)
        st = CtrlState(self.demand(t, u), 0.0, 0.0, 1.0, 0, self.pins_view(t, u, self.rho_s_hat(t, u)))
        for tj in yield_before_submit(need, key, st, self.M):
            self._cancel(self.tasks[tj].pin, "yield")
            self.tasks[tj].pin = None

    def task_done(self, key: str, final_submit: float | None, end: float | None, final_cached: int | None):
        s = self.tasks.pop(key, None)
        if s is None:
            return
        for p in (s.pin, s.pin_prev, s.pend):
            self._cancel(p, "task_end")
        if end is None or final_submit is None or s.plan_done is None:
            return
        self._xcrit.append((s.plan_done - s.arrival) + (end - final_submit))
        if s.completed_pre and s.unpinned:
            self._sig.append((s.completed_pre, min(s.completed_pre, final_cached or 0)))

    def summary(self) -> dict[str, Any]:
        sh, n = self.sigma_hat()
        return dict(M=self.M, stats=dict(self.stats), sigma_hat=sh, n_sigma_obs=n,
                    token_ratio=self._ratio[0] / self._ratio[1], decisions=self.decisions)
