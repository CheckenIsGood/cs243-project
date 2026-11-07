"""Extension B: KV pre-loading through automatic prefix caching (docs/theory.md, Section 8.2).

Prompt layout ("evidence log", append-only):
    final prompt = P0 . W . X . suffix
  P0     system prompt + answer instructions + question (known at time 0, never changes);
  W      evidence blocks in the order they *entered* the working context: a raw result enters when a
         PRELOAD appends it; a summary enters (at the end) when its CONSUME finishes. Summarizing a
         result whose raw block is in W deletes that block (this invalidates every cached token after it);
  X      returned raw results not yet in W, appended by the final invocation;
  suffix the chat-template tail (end of user turn + assistant header), never cached.
Consume prompts use their own system prompt, so they share no cached prefix with P0.W (their cost is the
base model's c(k, b)). No eviction: everything prefilled stays cached for the task.

Costs (beta_p = prefill s/token, alpha' = preload overhead):
  PRELOAD(q raws)   alpha' + beta_p * (uncached tokens of P0.W'), no decode (max_tokens=1);
  CONSUME(b)        base c(k, b);
  FINAL             alpha + beta_p * (uncached prompt tokens) + beta_d^fin * o_f.

`solve_preload_dp` is exact for this model (count state, release-order use, removal-free plans; proofs in
docs/theory.md). `solve_preload_exhaustive` is an independent brute force over explicit identities,
explicit block sequences (with deletions) and a prefix-trie cache, used only to verify it. It also
supports the "summaries_first" layout (P0 . summaries . raws), which is what the current runtime builds.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .affine import AffineCostModel, ceil_steps
from .restricted import INF


@dataclass(frozen=True)
class PreloadCostModel:
    base: AffineCostModel
    preload_overhead_s: float | None = None  # alpha'; None -> base.overhead_s
    final_suffix_tokens: int = 16  # part of base.final_base_prompt that follows the evidence

    def __post_init__(self) -> None:
        if not 0 <= self.final_suffix_tokens <= self.base.final_base_prompt:
            raise ValueError("suffix must be part of the final base prompt")

    @property
    def alpha_pre(self) -> float:
        return self.base.overhead_s if self.preload_overhead_s is None else self.preload_overhead_s

    @property
    def p0(self) -> int:
        """Cacheable prefix of the final prompt before any evidence."""
        return self.base.final_base_prompt - self.final_suffix_tokens

    @property
    def final_decode(self) -> float:
        b = self.base
        return b.decode_s_per_tok if b.final_decode_s_per_tok is None else b.final_decode_s_per_tok

    # ---- count-state costs (state: k summarized, p raws in W, j uncached summaries, z = P0 cached) ----
    def consume(self, k: int, b: int) -> float:
        return self.base.consume(k, b)

    def preload(self, k: int, p: int, j: int, z: int, q: int) -> float:
        b = self.base
        prompt = self.p0 + b.summary_tokens * k + b.raw_tokens * (p + q)
        if prompt + 1 > b.context_limit:
            return INF
        new = (0 if z else self.p0) + b.summary_tokens * j + b.raw_tokens * q
        return self.alpha_pre + b.prefill_s_per_tok * new

    def final(self, k: int, p: int, j: int, z: int, n: int) -> float:
        b = self.base
        prompt = b.final_base_prompt + b.summary_tokens * k + b.raw_tokens * (n - k)
        if prompt + b.final_out > b.context_limit:
            return INF
        new = (0 if z else self.p0) + b.summary_tokens * j + b.raw_tokens * (n - k - p) + self.final_suffix_tokens
        return b.overhead_s + b.prefill_s_per_tok * new + self.final_decode * b.final_out

    def condition_P(self) -> bool:
        """Sufficient condition (P) of Props B2-B3: alpha' + beta_p*P0 <= beta_d^sum * o (one summary's decode)."""
        b = self.base
        dec = b.decode_s_per_tok if b.consume_decode_s_per_tok is None else b.consume_decode_s_per_tok
        return self.alpha_pre + b.prefill_s_per_tok * self.p0 <= dec * b.consume_out_per_result + 1e-12


def replacement_excess(cm: PreloadCostModel, n: int) -> float:
    """Delta_B = max over feasible (k, b) of [alpha' + beta_p*(P0 + L*b) - c(k, b)]^+ (Prop. B2).

    0 means pre-loading b raws together with the whole final prefix is never slower than summarizing them.
    """
    b0 = cm.base
    worst = 0.0
    for k in range(n):
        for b in range(1, n - k + 1):
            c = cm.consume(k, b)
            if c < INF:
                worst = max(worst, cm.alpha_pre + b0.prefill_s_per_tok * (cm.p0 + b0.raw_tokens * b) - c)
    return worst


@dataclass(frozen=True)
class RoundedPreloadCosts:
    """Consume/preload durations rounded up to dt steps (what PreloadMDP uses); final kept exact."""
    cm: PreloadCostModel
    dt: float

    def consume(self, k, b):
        c = self.cm.consume(k, b)
        return INF if c == INF else ceil_steps(c, self.dt) * self.dt

    def preload(self, k, p, j, z, q):
        c = self.cm.preload(k, p, j, z, q)
        return INF if c == INF else ceil_steps(c, self.dt) * self.dt

    def final(self, k, p, j, z, n):
        return self.cm.final(k, p, j, z, n)


@dataclass
class PreloadSchedule:
    completion: float
    actions: list[tuple[str, tuple[int, ...], float, float]] = field(default_factory=list)  # kind, ids, start, end
    summarized: tuple[int, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.completion < INF


def solve_preload_dp(releases: tuple[float, ...], cm, *, preload: bool = True,
                     summarized: int | None = None, n_summaries: int | None = None) -> PreloadSchedule:
    """Exact optimum of the deterministic preload model (Prop. B1), O(n^4) time.

    preload=False forbids PRELOAD (reduces to the base model / `affine.prefix_dp`).
    summarized=K restricts to schedules that summarize exactly the K earliest-released results
    (K=0: fold with preloading); None = unrestricted.
    n_summaries=m restricts to exactly m summaries, in any order.
    `cm` needs consume(k,b), preload(k,p,j,z,q), final(k,p,j,z,n) (PreloadCostModel or RoundedPreloadCosts).
    """
    n = len(releases)
    order = sorted(range(n), key=lambda i: releases[i])
    r = [releases[i] for i in order]
    K = summarized
    cap = K if K is not None else n_summaries

    def rel(idx: int) -> float:  # release of the idx-th (1-based) result in release order
        return r[idx - 1] if idx > 0 else 0.0

    states = [(k, p, j, z) for k in range(n + 1) for p in range(n - k + 1) for j in range(k + 1) for z in (0, 1)
              if (cap is None or k <= cap) and (preload or (p == 0 and j == k and z == 0))]
    states.sort(key=lambda s: (s[0] + s[1], s[0], s[1], s[3], -s[2]))
    F = {s: INF for s in states}
    pred: dict = {}
    F[(0, 0, 0, 0)] = 0.0
    best, best_s = INF, None
    for s in states:
        t = F[s]
        if t == INF:
            continue
        k, p, j, z = s
        u = k + p
        # final
        if cap is None or k == cap:
            fc = cm.final(k, p, j, z, n)
            if fc < INF:
                cand = max(t, r[n - 1] if u < n else 0.0) + fc
                if cand < best:
                    best, best_s = cand, s
        # consume b results
        for b in range(1, n - u + 1):
            if cap is not None and k + b > cap:
                break
            c = cm.consume(k, b)
            if c == INF:
                continue
            start = max(t, rel(u + b) if K is None else rel(k + b))
            nxt = (k + b, p, j + b, z)
            if start + c < F.get(nxt, INF):
                F[nxt], pred[nxt] = start + c, (s, "consume", b, start)
        # preload q raws (q = 0: cache P0 / pending summaries only)
        if preload:
            hi = n - u if K is None else n - K - p
            for q in range(0, hi + 1):
                if q == 0 and j == 0 and z == 1:
                    continue
                c = cm.preload(k, p, j, z, q)
                if c == INF:
                    continue
                start = max(t, (rel(u + q) if K is None else rel(K + p + q)) if q else 0.0)
                nxt = (k, p + q, 0, 1)
                if start + c < F.get(nxt, INF):
                    F[nxt], pred[nxt] = start + c, (s, "preload", q, start)
    if best == INF:
        return PreloadSchedule(INF)
    # reconstruct and assign identities in release order
    seq, s = [], best_s
    while s != (0, 0, 0, 0):
        prev, kind, cnt, start = pred[s]
        seq.append((kind, cnt, start, F[s]))
        s = prev
    seq.reverse()
    nxt_any, nxt_sum, nxt_raw = 0, 0, K or 0
    actions, summ = [], []
    for kind, cnt, start, end in seq:
        if K is None:
            ids = tuple(order[nxt_any:nxt_any + cnt])
            nxt_any += cnt
        elif kind == "consume":
            ids = tuple(order[nxt_sum:nxt_sum + cnt])
            nxt_sum += cnt
        else:
            ids = tuple(order[nxt_raw:nxt_raw + cnt])
            nxt_raw += cnt
        if kind == "consume":
            summ.extend(ids)
        actions.append((kind, ids, start, end))
    return PreloadSchedule(best, actions, tuple(sorted(summ)))


def earliest_first_optimum(releases: tuple[float, ...], cm, preload: bool = True) -> PreloadSchedule:
    """Best schedule that summarizes an earliest-released prefix (Prop. 3's structure)."""
    best = PreloadSchedule(INF)
    for K in range(len(releases) + 1):
        s = solve_preload_dp(releases, cm, preload=preload, summarized=K)
        if s.completion < best.completion:
            best = s
    return best


# ---- independent brute force (explicit blocks + prefix trie) ----------------------------------
def solve_preload_exhaustive(releases: tuple[float, ...], cm: PreloadCostModel,
                             layout: str = "log", summarize_exactly: frozenset | None = None,
                             allow_preload: bool = True) -> float:
    """Minimum completion over every action sequence, with explicit cache semantics.

    Allows everything the DP excludes: summarizing preloaded raws (deleting their block), preloading in
    any order, and any subset/identity choice. Cost of a prompt = tokens not matching the longest
    previously prefilled prefix (no eviction). Exponential; use for n <= 4.
    summarize_exactly: if given, only these results may be summarized and all of them must be.
    """
    if layout not in ("log", "summaries_first"):
        raise ValueError(layout)
    b = cm.base
    n = len(releases)
    tok = {"P0": cm.p0}
    for i in range(n):
        tok[("r", i)] = b.raw_tokens
        tok[("s", i)] = b.summary_tokens

    def tokens(blocks) -> int:
        return sum(tok[x] for x in blocks)

    def cached(prompt: tuple, trie: frozenset) -> int:
        for i in range(len(prompt), 0, -1):
            if prompt[:i] in trie:
                return tokens(prompt[:i])
        return 0

    def with_prompt(trie: frozenset, prompt: tuple) -> frozenset:
        return trie | {prompt[:i] for i in range(1, len(prompt) + 1)}

    def blocks(W) -> tuple:
        return ("P0",) + (W if layout == "log" else W[0] + W[1])

    def summarize(W, B):
        drop = {("r", i) for i in B}
        add = tuple(("s", i) for i in sorted(B))
        if layout == "log":
            return tuple(x for x in W if x not in drop) + add
        return (W[0] + add, tuple(x for x in W[1] if x not in drop))

    def append_raw(W, Q):
        add = tuple(("r", i) for i in Q)
        return W + add if layout == "log" else (W[0], W[1] + add)

    def in_w(W) -> set:
        return {i for kind, i in (W if layout == "log" else W[0] + W[1]) if kind == "r"}

    best = INF
    rel = releases

    floor = b.overhead_s + cm.final_decode * b.final_out  # every completion is >= t + this

    def rec(t: float, S: frozenset, W, trie: frozenset) -> None:
        nonlocal best
        if t + floor >= best:
            return
        raw_in_w = in_w(W)
        unsumm = [i for i in range(n) if i not in S]
        can_sum = [i for i in unsumm if summarize_exactly is None or i in summarize_exactly]
        # final
        rest = tuple(("r", i) for i in unsumm if i not in raw_in_w)
        prompt = blocks(W) + rest
        total = tokens(prompt) + cm.final_suffix_tokens
        if total + b.final_out <= b.context_limit and (summarize_exactly is None or summarize_exactly <= S):
            start = max([t] + [rel[i] for _, i in rest])
            cost = b.overhead_s + b.prefill_s_per_tok * (total - cached(prompt, trie)) + cm.final_decode * b.final_out
            best = min(best, start + cost)
        # consume any nonempty subset of unsummarized results
        for size in range(1, len(can_sum) + 1):
            c = b.consume(len(S), size)
            if c == INF:
                continue
            for B in itertools.combinations(can_sum, size):
                start = max([t] + [rel[i] for i in B])
                rec(start + c, S | set(B), summarize(W, B), trie)
        # preload: append an ordered tuple of raws not yet in W (possibly none)
        avail = [i for i in unsumm if i not in raw_in_w]
        for size in (range(0, len(avail) + 1) if allow_preload else ()):
            for Q in itertools.permutations(avail, size):
                W2 = append_raw(W, Q)
                p2 = blocks(W2)
                hit = cached(p2, trie)
                if hit == tokens(p2):
                    continue  # nothing new to prefill
                if tokens(p2) + 1 > b.context_limit:
                    continue
                start = max([t] + [rel[i] for i in Q])
                rec(start + cm.alpha_pre + b.prefill_s_per_tok * (tokens(p2) - hit), S, W2, with_prompt(trie, p2))

    rec(0.0, frozenset(), () if layout == "log" else ((), ()), frozenset())
    return best
