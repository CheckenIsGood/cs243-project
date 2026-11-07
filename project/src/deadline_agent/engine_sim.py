"""Simulated continuous-batching LLM engine (virtual clock), for CPU tests of multi-agent serving policies.

Mirrors the vLLM behaviour the serving layer relies on:
- iteration-level batching: each step decodes one token for every running sequence and prefills newly admitted
  ones; step time = a + c * (#decoding sequences) + prefill_s_per_tok * (#prefill tokens in the step);
- a cap on concurrently running sequences (vLLM --max-num-seqs);
- waiting requests admitted in FCFS order, or by (priority, arrival) with lower priority first
  (vLLM --scheduling-policy priority);
- optional KV capacity in tokens: a request is admitted only if prompt + max output fits next to the running ones.
Running requests are not preempted for priority (a documented simplification; the real engine may preempt under
KV pressure). Latency parameters come from profiling (a + c = measured per-token decode time at batch 1).
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field

from .clock import now
from .model_client import FakeBackend, messages_tokens
from .state import Generation
from .tools import approx_tokens


@dataclass
class _Req:
    rid: int
    prompt: int
    out: int
    priority: float
    arrival: float
    fut: asyncio.Future
    left: int = 0
    prefilled: bool = False
    first_t: float | None = None


@dataclass
class EngineStats:
    steps: int = 0
    busy_s: float = 0.0
    batch_token_sum: int = 0  # sum over steps of #decoding sequences
    max_batch: int = 0
    finished: int = 0


@dataclass
class SimEngine:
    a: float = 0.0288  # per-step fixed cost (s)
    c: float = 0.0005  # per running sequence per step (s)
    prefill_s_per_tok: float = 1.8e-4
    max_seqs: int = 256
    policy: str = "fcfs"  # fcfs | priority
    kv_capacity: int | None = None
    stats: EngineStats = field(default_factory=EngineStats)
    _waiting: list[_Req] = field(default_factory=list)
    _running: list[_Req] = field(default_factory=list)
    _wake: asyncio.Event | None = None
    _task: asyncio.Task | None = None
    _ids: itertools.count = field(default_factory=itertools.count)

    def _key(self, r: _Req):
        return (r.priority, r.arrival, r.rid) if self.policy == "priority" else (r.arrival, r.rid)

    async def submit(self, prompt_tokens: int, out_tokens: int, priority: float = 0.0) -> tuple[float, float, float]:
        """Returns (submit_t, first_token_t, end_t)."""
        if self._task is None:
            self._wake = asyncio.Event()
            self._task = asyncio.create_task(self._loop())
        fut = asyncio.get_running_loop().create_future()
        r = _Req(next(self._ids), max(1, prompt_tokens), max(1, out_tokens), priority, now(), fut, left=max(1, out_tokens))
        self._waiting.append(r)
        self._wake.set()
        try:
            first, end = await fut
        except asyncio.CancelledError:  # caller gave up (e.g. its task's deadline passed): abort the request
            if r in self._waiting:
                self._waiting.remove(r)
            if r in self._running:
                self._running.remove(r)
            raise
        return r.arrival, first, end

    def _kv_used(self) -> int:
        return sum(r.prompt + r.out for r in self._running)

    def _admit(self) -> None:
        self._waiting.sort(key=self._key)
        keep = []
        for r in self._waiting:
            fits_seq = len(self._running) < self.max_seqs
            fits_kv = self.kv_capacity is None or self._kv_used() + r.prompt + r.out <= self.kv_capacity
            if fits_seq and fits_kv:
                self._running.append(r)
            else:
                # head-of-line blocking in queue order, like vLLM: later / lower-priority requests never jump ahead
                keep.extend(self._waiting[self._waiting.index(r):])
                break
        self._waiting = keep

    async def _loop(self) -> None:
        while True:
            if not self._waiting and not self._running:
                self._wake.clear()
                await self._wake.wait()
                continue
            self._admit()
            if not self._running:  # KV full and nothing running cannot happen unless a request is too big
                big = self._waiting.pop(0)
                big.fut.set_exception(RuntimeError("request exceeds KV capacity"))
                continue
            prefill = [r for r in self._running if not r.prefilled]
            decode = [r for r in self._running if r.prefilled]
            step = self.a + self.c * len(decode) + self.prefill_s_per_tok * sum(r.prompt for r in prefill)
            await asyncio.sleep(step)
            t = now()
            self.stats.steps += 1
            self.stats.busy_s += step
            self.stats.batch_token_sum += len(decode)
            self.stats.max_batch = max(self.stats.max_batch, len(self._running))
            for r in prefill:  # the prefill step also emits the first token (as in vLLM)
                if r in self._running:
                    r.prefilled, r.first_t = True, t
                    r.left -= 1
            for r in decode:
                if r in self._running:
                    r.left -= 1
            for r in [r for r in self._running if r.left <= 0]:
                self._running.remove(r)
                self.stats.finished += 1
                if not r.fut.done():
                    r.fut.set_result((r.first_t, t))

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


DEFAULT_OUT_TOKENS = {"plan": 40, "worker": 80, "final": 12, "preload": 1}


@dataclass
class SimBackend:
    """ModelBackend over a shared SimEngine. Output TEXT comes from FakeBackend (mechanical, quality not
    meaningful); output LENGTH comes from `out_tokens` per phase (measured typical sizes), capped by max_tokens.
    The request priority is taken from extra={"priority": p}, exactly as the vLLM client sends it."""

    engine: SimEngine
    out_tokens: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_OUT_TOKENS))
    name: str = "sim"
    _fake: FakeBackend = field(default_factory=FakeBackend)

    def count_tokens(self, text: str) -> int:
        return approx_tokens(text)

    async def generate(self, messages, max_tokens, phase, schema=None, extra=None) -> Generation:
        text = self._fake._output(messages, phase)
        out = min(max_tokens, self.out_tokens.get(phase, approx_tokens(text)))
        prompt = messages_tokens(self, messages)
        sub, first, end = await self.engine.submit(prompt, out, (extra or {}).get("priority", 0.0))
        return Generation(text, prompt, out, "stop", sub, first, end)
