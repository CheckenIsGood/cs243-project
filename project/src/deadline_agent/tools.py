"""Local BM25 search over one task's paragraphs, with reproducible injected delays.

The injected delay is derived from (task_id, normalized query, occurrence index, latency seed) via SHA-256,
never from policy name or launch order, and is hidden from the scheduler until the call returns.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .online_mdp import bounded_delay_pmf

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token); replaced by the real tokenizer count where available."""
    return max(1, math.ceil(len(text) / 4))


def normalize_query(q: str) -> str:
    return " ".join(words(q))


@dataclass(frozen=True)
class Passage:
    source_id: str
    text: str


class BM25:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.docs = [Counter(words(p.text)) for p in passages]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(w for d in self.docs for w in d)
        n = len(passages)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, top_k: int) -> list[Passage]:
        q = words(query)
        scores = []
        for i, (d, ln) in enumerate(zip(self.docs, self.lens)):
            s = 0.0
            for w in q:
                tf = d.get(w, 0)
                if tf:
                    s += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * ln / self.avg))
            scores.append((-s, i))  # deterministic tie-break by index
        scores.sort()
        return [self.passages[i] for _, i in scores[:top_k]]


@dataclass(frozen=True)
class DelayModel:
    """Uniform[lo, hi] w.p. 1-slow_p, Uniform[slow_lo, slow_hi] w.p. slow_p (seconds, admission-to-return)."""

    lo: float = 0.5
    hi: float = 4.0
    slow_p: float = 0.0
    slow_lo: float = 0.0
    slow_hi: float = 0.0

    def sample(self, u1: float, u2: float) -> float:
        if self.slow_p > 0 and u1 < self.slow_p:
            return self.slow_lo + u2 * (self.slow_hi - self.slow_lo)
        return self.lo + u2 * (self.hi - self.lo)

    def pmf(self, dt: float) -> list[float]:
        return bounded_delay_pmf(dt, self.lo, self.hi, self.slow_p, self.slow_lo, self.slow_hi)


def stable_uniforms(*key: object) -> tuple[float, float]:
    h = hashlib.sha256("|".join(map(str, key)).encode()).digest()
    return (int.from_bytes(h[:8], "big") / 2**64, int.from_bytes(h[8:16], "big") / 2**64)


@dataclass(frozen=True)
class ToolReturn:
    """Immutable completion event posted by a tool worker."""

    call_id: str
    text: str
    tokens: int
    source_ids: tuple[str, ...]
    retrieval_s: float
    injected_delay_s: float
    error: str | None = None


class SearchTool:
    def __init__(
        self,
        task_id: str,
        passages: list[Passage],
        delay: DelayModel,
        latency_seed: int,
        top_k: int = 4,
        result_token_cap: int = 1500,
        count_tokens: Callable[[str], int] = approx_tokens,
        inline: bool = True,  # True in virtual-clock mode; False runs retrieval in a thread
        delay_override: Callable[[int, str], float] | None = None,  # tests only: (call index, query) -> s
    ):
        self.task_id, self.delay, self.seed = task_id, delay, latency_seed
        self.index = BM25(passages)
        self.top_k, self.cap, self.count_tokens, self.inline = top_k, result_token_cap, count_tokens, inline
        self.delay_override = delay_override
        self._occ: Counter[str] = Counter()

    def planned_delay(self, call_index: int, query: str) -> float:
        nq = normalize_query(query)
        occ = self._occ[nq]
        self._occ[nq] += 1
        if self.delay_override:
            return self.delay_override(call_index, query)
        return self.delay.sample(*stable_uniforms(self.task_id, nq, occ, self.seed))

    def _retrieve(self, query: str) -> tuple[str, tuple[str, ...], int]:
        hits = self.index.search(query, self.top_k)
        parts, srcs, used = [], [], 0
        for p in hits:
            block = f"<source id=\"{p.source_id}\">\n{p.text}\n</source>"
            t = self.count_tokens(block)
            if used + t > self.cap:
                # deterministic truncation of the last passage to fit the shared cap
                room = self.cap - used
                if room > 32:
                    block = block[: max(0, room * 4 - 16)] + "…</source>"
                    parts.append(block)
                    srcs.append(p.source_id)
                break
            parts.append(block)
            srcs.append(p.source_id)
            used += t
        text = "\n".join(parts) if parts else "(no results)"
        return text, tuple(srcs), self.count_tokens(text)

    async def run(self, call_id: str, call_index: int, query: str) -> ToolReturn:
        """Admission-to-return: real retrieval, then injected delay. Total = retrieval + injected."""
        injected = self.planned_delay(call_index, query)
        t = time.perf_counter()
        if self.inline:
            text, srcs, tok = self._retrieve(query)
        else:
            text, srcs, tok = await asyncio.to_thread(self._retrieve, query)
        retrieval_s = time.perf_counter() - t
        await asyncio.sleep(injected)
        return ToolReturn(call_id, text, tok, srcs, retrieval_s, injected)
