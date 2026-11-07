"""Count-based (affine token) cost model and its structural results.

Invocation latency = overhead + prefill_per_tok * prompt + decode_per_tok * output.
All raw results have the same token length L and all summaries length s < L,
so costs depend only on how many results are summarized / in the batch.
See docs/theory.md, Propositions 2-4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .restricted import INF, Instance, Schedule


@dataclass(frozen=True)
class AffineCostModel:
    overhead_s: float = 0.05
    prefill_s_per_tok: float = 1 / 4000
    decode_s_per_tok: float = 1 / 30
    consume_base_prompt: int = 300  # instructions + question
    final_base_prompt: int = 300
    raw_tokens: int = 768  # L, per raw result
    summary_tokens: int = 64  # s, per summarized result
    consume_out_base: int = 32
    consume_out_per_result: int = 64
    final_out: int = 192
    context_limit: int = 8192
    consume_reads_summaries: bool = False  # does an evidence update see earlier summaries?
    # Extension A (docs/theory.md Section 8.1): effective decode rates per invocation type, e.g. under
    # prompt-lookup speculative decoding. None means "use decode_s_per_tok" (bit-for-bit unchanged).
    consume_decode_s_per_tok: float | None = None
    final_decode_s_per_tok: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.summary_tokens < self.raw_tokens:
            raise ValueError("model assumes 0 <= summary_tokens < raw_tokens")

    def _latency(self, prompt: int, out: int, decode_s_per_tok: float | None = None) -> float:
        if prompt + out > self.context_limit:
            return INF
        dec = self.decode_s_per_tok if decode_s_per_tok is None else decode_s_per_tok
        return self.overhead_s + self.prefill_s_per_tok * prompt + dec * out

    def consume(self, k_done: int, b: int) -> float:
        """Summarize b raw results after k_done were summarized."""
        prompt = self.consume_base_prompt + self.raw_tokens * b
        if self.consume_reads_summaries:
            prompt += self.summary_tokens * k_done
        return self._latency(prompt, self.consume_out_base + self.consume_out_per_result * b,
                             self.consume_decode_s_per_tok)

    def final(self, k_done: int, n: int) -> float:
        prompt = self.final_base_prompt + self.summary_tokens * k_done + self.raw_tokens * (n - k_done)
        return self._latency(prompt, self.final_out, self.final_decode_s_per_tok)

    @property
    def fold_saving_per_summary(self) -> float:
        """delta: final-invocation time saved by one summary instead of raw text."""
        return self.prefill_s_per_tok * (self.raw_tokens - self.summary_tokens)

    def min_summaries_needed(self, n: int) -> int:
        """Smallest k such that the final prompt fits (n+1 if none does)."""
        return next((k for k in range(n + 1) if self.final(k, n) < INF), n + 1)

    def to_instance(self, releases: tuple[float, ...]) -> Instance:
        n = len(releases)
        return Instance(
            releases,
            lambda A, B: self.consume(A.bit_count(), B.bit_count()),
            lambda S: self.final(S.bit_count(), n),
        )


def prefix_dp(releases: tuple[float, ...], m: AffineCostModel) -> Schedule:
    """O(n^2) exact solver for count-based costs (Proposition 3).

    Some optimal schedule summarizes a prefix of the release order in batches that
    are contiguous in that order, so DP over prefix lengths suffices.
    Returned batch masks refer to indices of the *original* releases tuple.
    """
    n = len(releases)
    order = sorted(range(n), key=lambda i: releases[i])
    r = [releases[i] for i in order]
    F = [INF] * (n + 1)
    pred = [-1] * (n + 1)
    F[0] = 0.0
    for k in range(1, n + 1):
        for j in range(k):
            if F[j] == INF:
                continue
            cost = m.consume(j, k - j)
            if cost == INF:
                continue
            cand = max(F[j], r[k - 1]) + cost
            if cand < F[k]:
                F[k], pred[k] = cand, j
    best, best_k = INF, 0
    for k in range(n + 1):
        fc = m.final(k, n)
        if F[k] == INF or fc == INF:
            continue
        cand = max(F[k], r[n - 1] if k < n else 0.0) + fc
        if cand < best:
            best, best_k = cand, k
    if best == INF:
        return Schedule(INF)
    batches, k = [], best_k
    while k:
        j = pred[k]
        mask = sum(1 << order[i] for i in range(j, k))
        batches.append((mask, max(F[j], r[k - 1]), F[k]))
        k = j
    batches.reverse()
    final_mask = sum(1 << order[i] for i in range(best_k))
    return Schedule(best, batches, final_mask, best - m.final(best_k, n))


def fold_completion(releases: tuple[float, ...], m: AffineCostModel) -> float:
    """Completion of pure folding: wait for every result, one final invocation."""
    return max(releases, default=0.0) + m.final(0, len(releases))


def fold_gap_bound(m: AffineCostModel, n: int) -> float:
    """Proposition 2: fold is within (n-1)*delta of optimal when its prompt fits."""
    return max(n - 1, 0) * m.fold_saving_per_summary


def ceil_steps(seconds: float, dt: float) -> int:
    return INF if seconds == INF else max(1, math.ceil(seconds / dt - 1e-9))  # type: ignore[return-value]
