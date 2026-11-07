"""Theory checks for a KV-aware preload controller (docs/controller_theory.md).

Contents, in the order of the document:

1. `BlockPool`: an exact model of vLLM v1's KV block pool with prefix caching (free queue = LRU over unreferenced
   cached blocks; freed blocks appended tail-first; uncached blocks reused first). `stack_resident` is the LRU
   stack-distance predicate of Lemma L; tests compare the two.
2. Fluid LRU / TTL model (Che-type approximation): `ttl_fixed_point`, `fluid_survival`, the thrashing map.
3. Break-even algebra under the M/G/1-PS approximation: `ps_net_saving`, `breakeven_survival` (Theorem BE).
4. Knapsack formulations (Target B): fractional greedy, exact LP optimum via Z3, integral brute force for the
   time-indexed (interval packing) version, index rule.
5. Pinning cost model (Target C).
6. The controller's decision rule as a pure function (`decide`), with every parameter derived in the doc.
7. `simulate_vllm` (model M4): a step-level engine with vLLM 0.32 scheduling semantics (full-prompt admission gate
   with head-of-line `break`, running-first token budget, priority waiting queue, lowest-priority recompute
   preemption, block-level LRU prefix cache with tail-first freeing) running the agent program of
   `multiagent.py` with preload policies off / on / low / chain / pin / ctrl.
8. Loaders for the local data (seed-2 workloads, vLLM log totals) and the comparison helpers.

Reuses `Calib`, `sample_delay`, `result_token_samples`, `bootstrap_ci` from preload_theory (not modified).
"""

from __future__ import annotations

import collections
import csv
import heapq
import itertools
import math
import random
import re
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence

from .preload_theory import Calib, bootstrap_ci, result_token_samples, sample_delay

INF = float("inf")
LOW = 1e9  # lowest engine priority used by multiagent.py (LOWEST_PRIORITY)


# ============================================================================== 1. block pool and Lemma L
class BlockPool:
    """vLLM v1 KV block pool (block_pool.py, v0.32.0 semantics as read from the source):
    - the free queue holds never-used / uncached blocks (reused first) and cached blocks with ref_cnt 0, which
      are evicted from the head in LRU order (`get_new_blocks` pops the head);
    - `touch` removes a hit block with ref_cnt 0 from the free queue and increments ref_cnt;
    - `free(blocks)` decrements ref counts; blocks reaching 0 are appended to the tail. Requests free their blocks
      in REVERSE order, so a request's tail blocks are evicted before its head blocks.
    Keys identify cached content (a prefix hash); `None` marks an uncached (partial) block.
    This class tracks only keys and counts, which is all that eviction order depends on."""

    def __init__(self, num_blocks: int):
        self.n = num_blocks
        self.uncached = num_blocks  # free, holding nothing
        self.lru: OrderedDict = OrderedDict()  # cached, ref_cnt 0; first = next evicted
        self.ref: dict = {}  # cached key -> ref_cnt > 0
        self.anon_ref = 0  # referenced uncached blocks
        self.evicted: list = []  # log of evicted keys (for checks)
        self.now = 0.0  # set by the simulator; release times are stored in the LRU for eviction-age statistics
        self.evict_ages: list = []

    def free_count(self) -> int:
        return self.uncached + len(self.lru)

    def present(self, key) -> bool:
        return key in self.lru or key in self.ref

    def touch(self, key) -> None:
        if key in self.lru:
            del self.lru[key]
            self.ref[key] = 1
        else:
            self.ref[key] += 1

    def allocate(self, keys: Sequence) -> bool:
        """Allocate len(keys) new blocks (key None = will hold uncached content). Fails (no change) if not enough
        free blocks. Evicts LRU heads when the uncached pool is exhausted."""
        if len(keys) > self.free_count():
            return False
        for k in keys:
            if self.uncached > 0:
                self.uncached -= 1
            else:
                old, rel = self.lru.popitem(last=False)
                self.evicted.append(old)
                if isinstance(old, tuple) and old and old[0] != "x":
                    self.evict_ages.append(self.now - rel)
            if k is None or k in self.ref or k in self.lru:  # duplicate content is not cached twice
                self.anon_ref += 1
            else:
                self.ref[k] = 1
        return True

    def free(self, keys: Sequence) -> None:
        """Free a request's blocks; `keys` in allocation order (freed in reverse). None = uncached block."""
        for k in reversed(keys):
            if k is None or k not in self.ref:
                self.anon_ref -= 1
                self.uncached += 1
                continue
            self.ref[k] -= 1
            if self.ref[k] == 0:
                del self.ref[k]
                self.lru[k] = self.now

    def referenced(self) -> int:
        return len(self.ref) + self.anon_ref


def stack_resident(trace: Sequence[tuple], n_blocks: int, key) -> bool:
    """Lemma L predicate, computed from the operation trace alone, without simulating eviction.

    trace: events (op, keys) with op in {'alloc', 'touch', 'free'} as issued to a BlockPool in which every
    allocation succeeded (allocated keys are fresh or None; touched keys were present; a 'free' lists one request's
    blocks in allocation order). Let s be the position of `key`'s last release (its ref count reaching 0). For each
    later state define
        D(t) = (# referenced blocks, cached or not) + (# cached unreferenced blocks released after `key`),
    the number of blocks LRU must keep ahead of `key`. Lemma L: `key` is resident at the end iff key is
    unreferenced-cached in the infinite-capacity replay and max over states after s of D(t) <= n_blocks - 1."""
    ref: dict = {}
    anon = 0
    rel_seq: dict = {}  # unreferenced cached key -> release sequence number
    seq = itertools.count()
    s_key = None
    dmax = -1

    def D():
        young = sum(1 for k, q in rel_seq.items() if k != key and q > s_key)
        return len(ref) + anon + young

    for op, keys in trace:
        if op == "alloc":
            for k in keys:
                if k is None:
                    anon += 1
                else:
                    ref[k] = 1
                    rel_seq.pop(k, None)
                if s_key is not None:
                    dmax = max(dmax, D())
        elif op == "touch":
            for k in keys:
                if k in ref:
                    ref[k] += 1
                else:
                    rel_seq.pop(k, None)
                    ref[k] = 1
                if s_key is not None:
                    dmax = max(dmax, D())
        else:
            for k in reversed(keys):
                if k is None:
                    anon -= 1
                    continue
                ref[k] -= 1
                if ref[k] == 0:
                    del ref[k]
                    q = next(seq)
                    rel_seq[k] = q
                    if k == key:
                        s_key, dmax = q, -1
                if s_key is not None:
                    dmax = max(dmax, D())
    if key not in rel_seq or s_key is None:
        return False
    return max(dmax, D()) <= n_blocks - 1


def replay_pool(trace: Sequence[tuple], n_blocks: int) -> BlockPool:
    pool = BlockPool(n_blocks)
    for (op, keys) in trace:
        if op == "alloc":
            ok = pool.allocate(keys)
            assert ok, "trace must not fail an allocation"
        elif op == "touch":
            for k in keys:
                pool.touch(k)
        else:
            pool.free(keys)
    return pool


# ======================================================================== 2. fluid LRU (TTL) approximation
def ttl_fixed_point(free_kv: float, v_dead: float, live: Callable[[float], float], iters: int = 200) -> float:
    """Characteristic time T of an LRU cache (Che et al. 2002 approximation adapted to referenced blocks):
    the unreferenced part of the cache, free_kv = M - mean referenced KV, is full and holds every dead block for T
    seconds after release and every live (to-be-re-referenced) block for min(T, gap):
        free_kv = v_dead * T + live(T),   live(T) = sum over live streams of rate * size * E[min(T, G)].
    The right side is continuous and strictly increasing in T (v_dead > 0), so the root is unique: bisection."""
    if free_kv <= 0:
        return 0.0
    if v_dead <= 0:
        return INF
    lo, hi_ = 0.0, free_kv / v_dead  # live(T) >= 0, so the root is at most free_kv / v_dead
    for _ in range(iters):
        mid = (lo + hi_) / 2
        if v_dead * mid + live(mid) < free_kv:
            lo = mid
        else:
            hi_ = mid
    return (lo + hi_) / 2


def thrash_map(T: float, free_kv: Callable[[float], float], v_dead: float,
               live: Callable[[float], float]) -> float:
    """One step of the feedback map T -> T' when the referenced KV itself depends on survival (free_kv(T)):
    used to exhibit multiple fixed points (Proposition T)."""
    return ttl_fixed_point(free_kv(T), v_dead, live)


# ================================================================= 3. break-even under the PS approximation
def ps_net_saving(xbar, u, w, lam):
    """Mean latency saving per task under the M/G/1-PS approximation (Approximation B of preload_theory):
    off: xbar/(1-rho); preloading moves useful work u off each task's critical path and adds waste w per task
    (evicted-and-recomputed prefill plus overhead): xbar/(1-rho) - (xbar-u)/(1-rho-lam*w), rho = lam*xbar.
    Works with Fractions (exact)."""
    rho = lam * xbar
    one = Fraction(1) if isinstance(xbar, Fraction) else 1.0
    if rho >= one or rho + lam * w >= one:
        return -INF
    return xbar / (one - rho) - (xbar - u) / (one - rho - lam * w)


def breakeven_survival(rho):
    """Theorem BE: with useful u = sigma*P and waste w = (1-sigma)*P, the PS saving is > 0 iff sigma > rho."""
    return rho


# =========================================================================== 4. knapsack (Target B)
@dataclass(frozen=True)
class Item:
    value: object  # expected saved GPU work (s), already net of overhead
    cost: object  # KV-seconds (tokens x s)


def _zfrac(v) -> Fraction:
    """Z3 numeral (int or rational) -> Fraction."""
    import z3
    if z3.is_int_value(v):
        return Fraction(v.as_long())
    return Fraction(v.numerator_as_long(), v.denominator_as_long())


def greedy_fractional(items: Sequence[Item], budget):
    """Dantzig's greedy for the fractional knapsack: sort by value/cost, fill. Returns (value, x)."""
    order = sorted(range(len(items)), key=lambda i: -(items[i].value / items[i].cost) if items[i].cost else -INF)
    left = budget
    x = [0] * len(items)
    val = 0
    for i in order:
        it = items[i]
        if it.value <= 0:
            continue
        if it.cost <= left:
            x[i] = 1
            left -= it.cost
            val += it.value
        else:
            frac = left / it.cost
            x[i] = frac
            val += it.value * frac
            left = 0
            break
    return val, x


def mckp_lp_greedy(groups: Sequence[Sequence[Item]], budget):
    """LP relaxation of the multiple-choice knapsack (choose at most one option per group, e.g. skip / unpinned /
    pinned for one preload): remove options that are dominated or below the upper concave hull of
    (cost, value) in each group (the zero option included), then fill the budget greedily by incremental
    efficiency. This is the classical LP-MCKP solution (Sinha & Zoltners 1979; Zemel 1980)."""
    segs = []
    for g, opts in enumerate(groups):
        pts = sorted({(Fraction(0), Fraction(0))} | {(Fraction(o.cost), Fraction(o.value)) for o in opts})
        hull = []
        for c, v in pts:
            if hull and v <= hull[-1][1]:
                continue  # dominated: costs more, worth no more
            while len(hull) >= 2:
                (c1, v1), (c2, v2) = hull[-2], hull[-1]
                if (v2 - v1) * (c - c1) <= (v - v1) * (c2 - c1):  # middle point not above the chord
                    hull.pop()
                else:
                    break
            hull.append((c, v))
        for (c1, v1), (c2, v2) in zip(hull, hull[1:]):
            segs.append(((v2 - v1) / (c2 - c1) if c2 > c1 else Fraction(10**9), c2 - c1, v2 - v1))
    segs.sort(key=lambda x: -x[0])
    left, val = Fraction(budget), Fraction(0)
    for eff, dc, dv in segs:
        if left <= 0:
            break
        if dc <= left:
            left -= dc
            val += dv
        else:
            val += dv * left / dc
            left = 0
    return val


def mckp_lp_z3(groups: Sequence[Sequence[Item]], budget):
    """Exact LP optimum of the multiple-choice knapsack relaxation (Z3)."""
    import z3
    opt = z3.Optimize()
    obj, cst = [], []
    for g, opts in enumerate(groups):
        xs = [z3.Real(f"x{g}_{i}") for i in range(len(opts))]
        for x in xs:
            opt.add(x >= 0)
        opt.add(z3.Sum(xs + [z3.RealVal(0)]) <= 1)
        obj += [z3.RealVal(str(Fraction(o.value))) * x for o, x in zip(opts, xs)]
        cst += [z3.RealVal(str(Fraction(o.cost))) * x for o, x in zip(opts, xs)]
    opt.add(z3.Sum(cst + [z3.RealVal(0)]) <= z3.RealVal(str(Fraction(budget))))
    h = opt.maximize(z3.Sum(obj + [z3.RealVal(0)]))
    assert opt.check() == z3.sat
    return _zfrac(opt.upper(h))


def lp_opt_z3(items: Sequence[Item], budget):
    """Exact LP optimum of max sum v_i x_i s.t. sum c_i x_i <= budget, 0 <= x_i <= 1 (Z3 Optimize, rationals)."""
    import z3
    opt = z3.Optimize()
    xs = [z3.Real(f"x{i}") for i in range(len(items))]
    for x in xs:
        opt.add(x >= 0, x <= 1)
    opt.add(z3.Sum([z3.RealVal(str(Fraction(it.cost))) * x for it, x in zip(items, xs)]) <= z3.RealVal(str(Fraction(budget))))
    obj = z3.Sum([z3.RealVal(str(Fraction(it.value))) * x for it, x in zip(items, xs)])
    h = opt.maximize(obj)
    assert opt.check() == z3.sat
    return _zfrac(opt.upper(h))


@dataclass(frozen=True)
class IntervalItem:
    """A preload held in KV from `start` to `end` (its need time) with `size` tokens, worth `value` if held."""
    start: int
    end: int
    size: int
    value: object


def interval_opt(items: Sequence[IntervalItem], cap: int):
    """Integral optimum of the time-indexed problem (unsplittable flow on a path with uniform capacity):
    choose a subset whose total size at every time point is <= cap. Brute force over subsets (small n)."""
    n = len(items)
    pts = sorted({it.start for it in items})
    best, arg = 0, ()
    for mask in range(1 << n):
        sel = [items[i] for i in range(n) if mask >> i & 1]
        ok = all(sum(it.size for it in sel if it.start <= p < it.end) <= cap for p in pts)
        if ok:
            v = sum(it.value for it in sel)
            if v > best:
                best, arg = v, tuple(i for i in range(n) if mask >> i & 1)
    return best, arg


def interval_index_greedy(items: Sequence[IntervalItem], cap: int):
    """Online-at-start greedy by the KV-second index value/(size*(end-start)): process items in start order;
    admit if it fits now; otherwise evict currently held items of lower index if that makes room
    (a held item that is evicted loses its value). This is the rule of the controller (Section 6)."""
    held: list = []
    order = sorted(range(len(items)), key=lambda i: (items[i].start, i))
    lost = set()

    def idx(i):
        it = items[i]
        return it.value / (it.size * max(1, it.end - it.start))

    for i in order:
        it = items[i]
        held = [h for h in held if items[h].end > it.start]
        used = sum(items[h].size for h in held)
        if used + it.size <= cap:
            held.append(i)
            continue
        victims = sorted([h for h in held if idx(h) < idx(i)], key=idx)
        freed, take = 0, []
        for h in victims:
            if used - freed + it.size <= cap:
                break
            take.append(h)
            freed += items[h].size
        if used - freed + it.size <= cap:
            for h in take:
                held.remove(h)
                lost.add(h)
            held.append(i)
        else:
            lost.add(i)
    val = sum(items[i].value for i in range(len(items)) if i not in lost)
    return val


def lp_interval_z3(items: Sequence[IntervalItem], cap: int):
    """LP relaxation of the time-indexed problem (Z3, exact rationals)."""
    import z3
    opt = z3.Optimize()
    xs = [z3.Real(f"x{i}") for i in range(len(items))]
    for x in xs:
        opt.add(x >= 0, x <= 1)
    for p in sorted({it.start for it in items}):
        opt.add(z3.Sum([it.size * x for it, x in zip(items, xs) if it.start <= p < it.end] + [z3.RealVal(0)]) <= cap)
    h = opt.maximize(z3.Sum([z3.RealVal(str(Fraction(it.value))) * x for it, x in zip(items, xs)]))
    assert opt.check() == z3.sat
    return _zfrac(opt.upper(h))


def preload_index(L: float, wait: float, cal: Calib = Calib(), p_unpinned: float = 0.0) -> float:
    """Saved prefill GPU-seconds per KV token-second of holding a log prefix of L tokens for `wait` more seconds:
    cost(L) (1 - p_unpinned) / (L * wait), cost(L) = alpha L + kappa L^2 / 2 (prefill of L tokens from scratch).
    The GDS analog: GDS ranks by cost/size; here the size is a KV-time product with the expected need time."""
    c = cal.alpha * L + cal.kappa * L * L / 2
    return c * (1.0 - p_unpinned) / (max(L, 1.0) * max(wait, 1e-3))


# ============================================================================ 5. pinning cost (Target C)
def pin_gpu_cost(hold_s: float, step_s: float, cal: Calib = Calib()) -> float:
    """GPU seconds a pinned request costs: one decode slot per step (c each) for hold_s / step_s steps."""
    return cal.c * hold_s / step_s


def pin_beats_lru(L: float, p_survive: float, hold_s: float, step_s: float, rho: float, eps: float = 0.0,
                  cal: Calib = Calib()) -> bool:
    """Proposition C: pinning a prefix of L tokens for hold_s instead of leaving it to LRU (survival p) lowers
    mean latency (PS approximation, Theorem BE accounting) iff the expected recompute it avoids, counted as useful
    work, outweighs the pin's GPU overhead counted as waste: (1-rho) * (1-p) * cost(L) > rho * pin_cost."""
    cost = cal.alpha * L + cal.kappa * L * L / 2
    return (1 - rho) * (1 - p_survive) * cost > rho * (pin_gpu_cost(hold_s, step_s, cal) + eps)


# ===================================================================== 6. the controller's decision rule
@dataclass
class CtrlParams:
    """Every field is derived in docs/controller_theory.md Section 6; `window_s` and `min_obs` are the only
    labelled tuning knobs (statistical smoothing of the feedback estimates)."""
    M: int  # KV capacity (tokens), from the server log
    a: float = Calib.a  # decode step fixed cost: lower bound on step time -> pin length
    headroom: int = 110 + 8 * (1400 + 35) + 15  # one admission of the largest final prompt (header + width x
    # (result cap + block header) + suffix): pins are budgeted so that this always fits without yielding
    pin_quantile: float = 0.99  # pin until the q-quantile of the remaining wait (beyond it LRU takes over)
    window_s: float = 30.0  # TUNING KNOB: EWMA window of rho_hat and sigma_hat
    min_obs: int = 5  # TUNING KNOB: observations before the sigma feedback may switch unpinned preloads off
    lever: str = "pin"  # keep-alive lever for preloads that LRU would evict: "pin" (hold the request open) or
    # "refresh" (re-send the cached prefix as a 1-token request every ttl_hat/2 until need: Proposition C')


@dataclass
class TaskView:
    """What the controller knows about one agent (all client-observable)."""
    log_tokens: int  # tokens of the evidence log so far (prompt of the next preload, without chat suffix)
    pinned_tokens: int  # tokens of this task's prefix held by an open (pinned) preload, 0 if none
    remaining_wait: float  # E[time until the last tool returns], from the delay law and elapsed times
    remaining_wait_q: float  # pin_quantile of the remaining wait (sets the pin's max_tokens)
    p_unpinned: float  # P(remaining wait <= ttl_hat): TTL survival estimate if left to LRU
    preload_in_prefill: bool  # a preload of this task was sent and its first token has not arrived


@dataclass
class CtrlState:
    demand: float  # D_hat: sum over in-flight requests of prompt + generated tokens, shared prefixes counted once
    rho_hat: float  # u_hat: prefill utilization (prefill GPU seconds per second, cost model x computed tokens)
    ttl_hat: float  # LRU characteristic time estimate (s)
    sigma_hat: float  # realized survival fraction of unpinned preloads (cached_tokens feedback); 1 if no data
    n_sigma_obs: int
    pins: dict  # task -> (tokens, index) of currently open pins
    rho_s: float = 0.0  # stretchable critical-path rate lam_hat * X_hat (Theorem BE'); 0 -> use rho_hat (PS case)


def p_survive_unpinned(wait_samples: Sequence[float], ttl: float) -> float:
    """TTL (Che) approximation: an unpinned prefix survives iff it is re-referenced within the characteristic time
    ttl; the probability is taken over the conditional law of the remaining wait."""
    if not wait_samples:
        return 1.0
    return sum(1 for w in wait_samples if w <= ttl) / len(wait_samples)


def step_time_hat(u: float, cal: Calib = Calib()) -> float:
    """Mean engine step time from the stretch identity (Proposition S): decode steps per second = (1-u)/(a+cB);
    with B small, s_bar ~ a/(1-u). Bounded to [a, 20a]."""
    return min(20 * cal.a, cal.a / max(0.05, 1 - min(u, 0.95)))


def sigma_star(u: float, rho_s: float) -> float:
    """Theorem BE' (stretch model): preloading with survival fraction sigma lowers mean latency iff
    sigma > rho_s / (1 - u + rho_s). With rho_s = u (all stretchable time is prefill work: the PS case of
    Theorem BE) this is sigma > u."""
    rs = u if rho_s <= 0 else rho_s
    return rs / (1 - u + rs) if (1 - u + rs) > 0 else 1.0


def refresh_gpu_cost(L: float, cal: Calib = Calib()) -> float:
    """GPU seconds of one refresh: recompute the last partial block plus the chat suffix (<= 16 + suffix tokens)
    at context L, plus one decode slot."""
    x = 16 + cal.suffix
    return cal.prefill_cost(x, L) + cal.c


def pin_value(task: TaskView, u: float, cal: Calib = Calib(), rho_s: float = 0.0, lever: str = "pin",
              ttl: float = INF) -> float:
    """Proposition C, net value in Theorem-BE' units of keeping the prefix alive until need instead of leaving it
    to LRU: (1-u)(1-p) cost(L) - rho_s * (keep-alive GPU seconds). Pin: c * hold / s_bar. Refresh: one refresh
    per ttl/2 of remaining hold."""
    rs = u if rho_s <= 0 else rho_s
    L = task.log_tokens
    cost = cal.alpha * L + cal.kappa * L * L / 2
    hold = task.remaining_wait
    if lever == "pin":
        keep = pin_gpu_cost(hold, step_time_hat(u, cal), cal)
    else:
        period = max(1.0, ttl / 2) if ttl < INF else INF
        keep = 0.0 if period == INF else math.ceil(hold / period) * refresh_gpu_cost(L, cal)
    return (1 - u) * (1 - task.p_unpinned) * cost - rs * keep


def decide(task: TaskView, st: CtrlState, P: CtrlParams, cal: Calib = Calib()) -> tuple:
    """Pure decision for one preload opportunity (a tool result arrived and others are outstanding).

    Returns (action, victims): action in {'skip', 'pin', 'refresh', 'unpinned'}; victims = pins to cancel first.
    Rule (doc Section 6), all preloads at the lowest engine priority:
      1. never preload while this task's previous preload is still in prefill (at most one in flight);
      2. PIN if pinning beats LRU (pin_value > 0, Proposition C) and the pin fits the KV budget
         demand + new tokens <= M, possibly after yielding open pins of strictly lower index (Section 4 rule);
      3. else UNPINNED iff the TTL survival estimate exceeds the break-even survival, p_unpinned > sigma_star
         (Theorem BE'), and the cached_tokens feedback has not shown sigma_hat <= sigma_star;
      4. else SKIP."""
    if task.preload_in_prefill:
        return "skip", ()
    thr = sigma_star(st.rho_hat, st.rho_s)
    feedback_off = st.n_sigma_obs >= P.min_obs and st.sigma_hat <= thr
    unpinned_ok = task.p_unpinned > thr and not feedback_off
    pv = pin_value(task, st.rho_hat, cal, st.rho_s, P.lever, st.ttl_hat)
    if pv > 0 and P.lever == "refresh":
        return "refresh", ()
    if pv > 0:
        new = max(0, task.log_tokens - task.pinned_tokens)
        idx = pv / (max(task.log_tokens, 1) * max(task.remaining_wait, 1e-3))
        if st.demand + new + P.headroom <= P.M:
            return "pin", ()
        need = st.demand + new + P.headroom - P.M
        victims, freed = [], 0.0
        for tj, (tok, ix) in sorted(st.pins.items(), key=lambda kv: kv[1][1]):
            if ix >= idx or freed >= need:
                break
            victims.append(tj)
            freed += tok
        if freed >= need:
            return "pin", tuple(victims)
    if unpinned_ok:
        return "unpinned", ()
    return "skip", ()


def yield_before_submit(need: float, own_task, st: CtrlState, M: float) -> tuple:
    """Pins to cancel (lowest index first, never the submitting task's own) so that a plan/final of `need`
    tokens passes the server's full-prompt admission gate: demand + need <= M."""
    over = st.demand + need - M
    out = []
    for tj, (tok, ix) in sorted(st.pins.items(), key=lambda kv: kv[1][1]):
        if over <= 0:
            break
        if tj == own_task:
            continue
        out.append(tj)
        over -= tok
    return tuple(out)


# ================================================================ 7. engine model M4 (vLLM 0.32 semantics)
@dataclass
class VCfg:
    kv_capacity: int = 167_152  # tokens (default server: 167,152 in results/logs_pod/kv_pressure.log)
    block: int = 16
    budget: int = 2048  # max_num_batched_tokens (vLLM default, DEFAULT_MAX_NUM_BATCHED_TOKENS)
    max_seqs: int = 256
    full_isl_gate: bool = True  # scheduler_reserve_full_isl (default True in v0.32.0)
    hol_break: bool = True  # stop admitting when the head of the waiting queue does not fit
    pin_waiting_prefix: bool = False  # True = M3-like idealisation: a request references its prefix on arrival


@dataclass
class VTask:
    arrival: float
    n: int
    delays: tuple
    sizes: tuple  # tokens per result (with block header), in tool index order
    plan_out: int


@dataclass
class _VReq:
    rid: int
    task: int
    kind: str  # plan | preload | final
    prompt: int
    shared: int  # leading prompt blocks that are task-log blocks (shareable)
    out: int
    prio: float
    t_sub: float  # client submission time
    t_arr: float = 0.0  # engine arrival
    computed: int = 0
    gen: int = 0
    blocks: list = field(default_factory=list)
    running: bool = False
    first_t: float | None = None
    done_t: float | None = None
    cancelled: bool = False
    cached: int = 0
    log_k: int = 0  # number of results in the prompt (log requests)
    pinned: bool = False
    keep: bool = False  # refresh lever: schedule refreshes of this prefix after it completes
    refresh: bool = False


@dataclass
class VOut:
    latency: list
    stage: dict  # per task: plan_done, last_tool, final_submit, final_first, end
    computed_prefill: int  # prompt tokens computed by the engine (vLLM 'prompt throughput' x time)
    computed_by_kind: dict
    final_cached: list
    final_prompt: list
    preloads: list
    preemptions: int
    steps: int
    busy: float
    horizon_hit: bool = False
    kv_ref_peak: int = 0
    hol_blocked_steps: int = 0
    sigma_obs: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    evict_ages: list = field(default_factory=list)
    diag: dict = field(default_factory=dict)
    prefill_time: float = 0.0
    decode_slot_steps: int = 0
    pre_max: list = field(default_factory=list)
    yielded: list = field(default_factory=list)


def cond_delay_sample(rng: random.Random, elapsed: float, dm=Calib.delay) -> float:
    """Sample D from the DelayModel mixture U[lo,hi] w.p. 1-p, U[slo,shi] w.p. p, conditional on D > elapsed
    (inverse transform; returns `elapsed` if the law has no mass above it)."""
    lo, hi, p, slo, shi = dm
    def tail(a, b):
        return 0.0 if elapsed >= b else (1.0 if elapsed <= a else (b - elapsed) / (b - a))
    w1, w2 = (1 - p) * tail(lo, hi), p * tail(slo, shi)
    if w1 + w2 <= 0:
        return elapsed
    if rng.random() * (w1 + w2) < w1:
        a = max(lo, elapsed)
        return a + rng.random() * (hi - a)
    a = max(slo, elapsed)
    return a + rng.random() * (shi - a)


def simulate_vllm(tasks: Sequence[VTask], policy: str = "off", cal: Calib = Calib(), cfg: VCfg = VCfg(),
                  ctrl: CtrlParams | None = None, horizon: float = 5000.0) -> VOut:
    """policy: off | on | low (nb at prio 0 / LOW; multiagent.py) | chain (nb + re-issue on completion, prio 0) |
    pin (every preload LOW with max_tokens from the wait quantile, held open until superseded or the final's first
    token; yield-before-submit) | ctrl (the full controller of Section 6: `decide` + `yield_before_submit`).

    Engine step (vLLM v1 Scheduler.schedule, v0.32.0, as read from the source):
      1. running requests in running-list order: prefill chunk min(remaining, budget) or one decode token; block
         allocation per chunk; on failure preempt max(running, key=(priority, arrival)) and retry;
      2. only if nothing was preempted: waiting requests in (priority, arrival) order; prefix hit = longest run of
         resident task-log blocks (at most (prompt-1)//block); full-prompt gate: new blocks + evictable hit blocks
         <= free blocks (cached-unreferenced count as free); on failure stop admitting (`break`);
      3. step time a + c * decodes + sum prefill_cost(chunk, context); finished requests free their blocks.
    Client-server latency delta/2 each way (calibrated in preload_theory)."""
    assert policy in ("off", "on", "low", "chain", "pin", "ctrl")
    B = cfg.block
    nblk = cfg.kv_capacity // B
    pool = BlockPool(nblk)
    rid = itertools.count()
    anon = itertools.count()
    sq = itertools.count()
    ev: list = []
    waiting: list = []  # heap (prio, t_arr, rid, req)
    running: list = []
    nT = len(tasks)
    orders = [sorted(range(tk.n), key=lambda i: (tk.delays[i], i)) for tk in tasks]
    cum = []
    for j, tk in enumerate(tasks):
        acc = [cal.header]
        for i in orders[j]:
            acc.append(acc[-1] + tk.sizes[i])
        cum.append(acc)
    stt = [dict(returned=0, pend=None, pin=None, covered=0, final=None, plan_done=None, last=None,
                elapsed0=None) for _ in range(nT)]
    lat = [None] * nT
    stage = {k: [None] * nT for k in ("plan_done", "last_tool", "final_submit", "final_first", "end")}
    computed_by = dict(plan=0, preload=0, final=0)
    npre = [0] * nT
    preempt = 0
    hol = 0
    kv_peak = 0
    inflight: dict = {}  # rid -> req (client view)
    rho_win: deque = deque()  # (t, gpu_work) of finished requests, for rho_hat
    decisions: list = []
    xcrit: list = []  # observed plan latency + final latency per finished task (stretched critical time)
    pre_max = [0] * nT
    yielded = [False] * nT
    diag_why: dict = {}
    diag: collections.Counter = collections.Counter()
    sig_obs: list = []  # (preloaded tokens, survived tokens) per final of a task with unpinned preloads
    dead_win: deque = deque()  # (t, tokens) released as dead into LRU, for ttl_hat

    def log_tokens(j, k):
        return cum[j][k]

    def mk(j, kind, prompt, shared_tokens, out, prio, now, k=0):
        r = _VReq(next(rid), j, kind, prompt, shared_tokens // B, out, prio, now)
        r.t_arr = now + cal.delta / 2
        r.log_k = k
        heapq.heappush(ev, (r.t_arr, next(sq), "arrive", r))
        inflight[r.rid] = r
        if kind == "preload":
            npre[j] += 1
        return r

    def cancel(r, now, why="other"):
        if r is None or r.done_t is not None or r.cancelled:
            return
        diag["cancel_" + why] += 1
        diag_why[r.rid] = why
        r.cancelled = True
        inflight.pop(r.rid, None)
        heapq.heappush(ev, (now + cal.delta / 2, next(sq), "cancel", r))

    # ---- client-side controller state ---------------------------------------------------------------------
    def demand_now():
        per_task: dict = {}
        tot = 0
        for r in inflight.values():
            if r.kind == "plan":
                tot += r.prompt + r.gen
            else:
                per_task[r.task] = max(per_task.get(r.task, 0), r.prompt + r.gen)
        return tot + sum(per_task.values())

    def rho_hat(now):
        w = ctrl.window_s if ctrl else 30.0
        while rho_win and rho_win[0][0] < now - w:
            rho_win.popleft()
        span = min(w, max(now - (tasks[0].arrival if tasks else 0.0), 1.0))
        return sum(x for _, x in rho_win) / span

    def rho_s_hat(now, u):
        w = ctrl.window_s if ctrl else 30.0
        span = min(w, max(now - (tasks[0].arrival if tasks else 0.0), 1.0))
        lam = sum(1 for tk in tasks if now - w <= tk.arrival <= now) / span
        if len(xcrit) >= (ctrl.min_obs if ctrl else 5):
            X = sum(xcrit[-50:]) / len(xcrit[-50:]) * (1 - min(u, 0.95))
        else:  # prior before feedback: isolated plan decode + mean final prefill (cost model)
            X = 128 * (cal.a + cal.c) + cal.prefill_cost(7000, 0)
        return lam * X

    def ttl_hat(now, demand):
        w = ctrl.window_s if ctrl else 30.0
        while dead_win and dead_win[0][0] < now - w:
            dead_win.popleft()
        span = min(w, max(now - (tasks[0].arrival if tasks else 0.0), 1.0))
        v = sum(x for _, x in dead_win) / span
        free = cfg.kv_capacity - demand
        return INF if v <= 0 else max(0.0, free / v)

    def sigma_hat():
        if not sig_obs:
            return 1.0, 0
        pre = sum(a for a, _ in sig_obs[-50:])
        sur = sum(b for _, b in sig_obs[-50:])
        return (sur / pre if pre else 1.0), len(sig_obs)

    def wait_samples(j, now):
        tk = tasks[j]
        s = stt[j]
        el = now - s["plan_done"]
        nrem = tk.n - s["returned"]
        rng = random.Random(1_000_003 * j + s["returned"])
        xs = []
        for _ in range(64):  # the client knows only the delay law: max over outstanding of (D - el | D > el)
            mx = 0.0
            for _r in range(nrem):
                mx = max(mx, cond_delay_sample(rng, el, cal.delay) - el)
            xs.append(mx)
        xs.sort()
        return xs

    def wait_stats(j, now):
        xs = wait_samples(j, now)
        q = ctrl.pin_quantile if ctrl else 0.99
        return sum(xs) / len(xs), xs[min(len(xs) - 1, int(q * len(xs)))]

    def pins_view(now):
        out = {}
        rh = rho_hat(now) if policy == "ctrl" else 0.5
        rs = rho_s_hat(now, rh) if policy == "ctrl" else 0.0
        for j in range(nT):
            p = stt[j]["pin"]
            if p is not None and p.done_t is None and not p.cancelled and p.first_t is not None:
                L = p.prompt
                if stt[j]["returned"] < tasks[j].n:
                    xs = wait_samples(j, now)
                    w = sum(xs) / len(xs)
                    tv = TaskView(L, L, w, w, 0.0, False)
                    out[j] = (L, pin_value(tv, rh, cal, rs) / (L * max(w, 1e-3)))
                else:
                    out[j] = (L, INF)  # final already submitted: its pin is needed now
        return out

    def pin_tokens(j, now):
        w, wq = wait_stats(j, now)
        return int(math.ceil(wq / cal.a)) + 1

    def send_preload(j, now, pinned):
        stt[j]["refresh_of"] = None
        k = stt[j]["returned"]
        L = log_tokens(j, k)
        lowp = policy in ("low", "pin", "ctrl")
        out = pin_tokens(j, now) if pinned else 1
        r = mk(j, "preload", L + cal.suffix, L, out, LOW if lowp else 0.0, now, k)
        r.pinned = pinned
        stt[j]["pend"] = r
        stt[j]["covered"] = k
        return r

    def yield_pins(j, need, now):
        if policy not in ("pin", "ctrl"):
            return
        own = stt[j]["pin"]
        if own is not None and own.done_t is None and not own.cancelled:
            need = max(0, need - own.prompt)
        st = CtrlState(demand_now(), 0.0, 0.0, 1.0, 0, pins_view(now))
        for tj in yield_before_submit(need, j, st, cfg.kv_capacity):
            cancel(stt[tj]["pin"], now, "yield")
            stt[tj]["pin"] = None

    def send_final(j, now):
        s = stt[j]
        L = log_tokens(j, tasks[j].n)
        yield_pins(j, L + cal.suffix, now)
        s["final"] = mk(j, "final", L + cal.suffix, L, cal.final_out, 0.0, now, tasks[j].n)
        stage["final_submit"][j] = now

    def on_return(j, now):
        s = stt[j]
        s["returned"] += 1
        k, n = s["returned"], tasks[j].n
        if k == n:
            s["last"] = now
            stage["last_tool"][j] = now
            p = s["pend"]
            if p is not None and p.done_t is None and p.first_t is None:
                cancel(p, now, "final_ready")  # still in prefill: its computed blocks stay cached
                if p is s["pin"]:
                    s["pin"], s["pin_prev"] = s.get("pin_prev"), None
            send_final(j, now)
            return
        if policy == "off":
            return
        p = s["pend"]
        busy = p is not None and p.done_t is None and not p.cancelled and (p.first_t is None or not p.pinned)
        if policy in ("on", "low", "chain"):
            if not busy:
                send_preload(j, now, False)
            return
        if p is not None and p.first_t is None and p.done_t is None and not p.cancelled:
            return  # previous preload still in prefill
        if policy == "pin":
            act, victims = "pin", ()
        else:
            xs = wait_samples(j, now)
            w = sum(xs) / len(xs)
            wq = xs[min(len(xs) - 1, int(ctrl.pin_quantile * len(xs)))]
            pin_tok = s["pin"].prompt if (s["pin"] is not None and s["pin"].done_t is None
                                           and not s["pin"].cancelled) else 0
            sh, nobs = sigma_hat()
            st = CtrlState(demand_now(), rho_hat(now), 0.0, sh, nobs, pins_view(now))
            st.rho_s = rho_s_hat(now, st.rho_hat)
            st.ttl_hat = ttl_hat(now, st.demand)
            tv = TaskView(log_tokens(j, k), pin_tok, w, wq, p_survive_unpinned(xs, st.ttl_hat), False)
            act, victims = decide(tv, st, ctrl, cal)
            decisions.append((now, act, st.rho_hat, st.ttl_hat, tv.p_unpinned, st.rho_s))
        for tj in victims:
            cancel(stt[tj]["pin"], now, "displaced")
            stt[tj]["pin"] = None
        if act == "skip":
            return
        if act == "refresh":
            r = send_preload(j, now, False)
            r.keep = True
            s["unpinned"] = s.get("unpinned", 0) + 1
            return
        old_pin = s["pin"]
        r = send_preload(j, now, act == "pin")
        if act == "pin":
            s["pin_prev"] = old_pin  # cancelled when r's first token arrives (r then references the prefix)
            s["pin"] = r
        else:
            s["unpinned"] = s.get("unpinned", 0) + 1

    def on_client(kind, r, now):
        j = r.task
        s = stt[j]
        if kind == "first":
            if r.kind == "preload":
                s["completed_pre"] = max(s.get("completed_pre", 0), log_tokens(j, r.log_k) // B * B)
                if r.keep and s["returned"] < tasks[j].n:
                    s["refresh_of"] = r
                    period = max(1.0, ttl_hat(now, demand_now()) / 2)
                    heapq.heappush(ev, (now + period, next(sq), "refresh", r))
            if r.kind == "preload" and r.pinned:
                prev = s.get("pin_prev")
                if prev is not None and prev is not r:
                    cancel(prev, now, "superseded")
                    s["pin_prev"] = None
            if r.kind == "final":
                stage["final_first"][j] = now
                for key in ("pin", "pin_prev"):
                    if s.get(key) is not None:
                        cancel(s[key], now, "final_started")
                        s[key] = None
            return
        # done
        inflight.pop(r.rid, None)
        if r.kind == "plan":
            s["plan_done"] = now
            stage["plan_done"][j] = now
            for i in range(tasks[j].n):
                heapq.heappush(ev, (now + tasks[j].delays[i], next(sq), "ret", j))
        elif r.kind == "final":
            lat[j] = now - tasks[j].arrival
            stage["end"][j] = now
            xcrit.append((s["plan_done"] - tasks[j].arrival) + (now - stage["final_submit"][j]))
            pre = s.get("completed_pre", 0)  # log tokens of the largest preload whose prefill completed
            if pre and s.get("unpinned", 0):
                sig_obs.append((pre, min(pre, r.cached)))
        elif r.kind == "preload" and policy == "chain":
            if s["returned"] < tasks[j].n and s["returned"] > s["covered"]:
                send_preload(j, now, False)

    for j, tk in enumerate(tasks):
        heapq.heappush(ev, (tk.arrival, next(sq), "task", j))

    def grow(r, new_total):
        """Allocate blocks so that r holds new_total tokens. A block's content key is fixed at allocation: task-log
        blocks are keyed (task, b) (shareable prefix); other blocks that will be full by the end of the request are
        unique cached content ('x', id); the rest are uncached."""
        have = len(r.blocks)
        need_blocks = (new_total + B - 1) // B
        end_tokens = r.prompt + r.out
        keys = []
        for b in range(have, need_blocks):
            if b < r.shared:
                keys.append((r.task, b))
            elif (b + 1) * B <= end_tokens:
                keys.append(("x", next(anon)))
            else:
                keys.append(None)
        if not pool.allocate(keys):
            return False
        r.blocks.extend(keys)
        return True

    def release(r, now):
        if r.kind == "preload":
            pre_max[r.task] = max(pre_max[r.task], min(r.computed, r.shared * B))
            if r.cancelled and diag_why.get(r.rid):
                yielded[r.task] |= diag_why[r.rid] == "yield"
        if r.blocks:
            if r.kind in ("plan", "final"):
                ndead = sum(1 for k in r.blocks if k is not None)
            else:
                ndead = sum(1 for k in r.blocks if k is not None and k[0] == "x")
            pool.free(r.blocks)
            dead_win.append((now, ndead * B))
        r.blocks = []
        r.running = False

    def hit_blocks(r):
        h = 0
        cap = (r.prompt - 1) // B
        while h < min(r.shared, cap) and pool.present((r.task, h)):
            h += 1
        return h

    t = 0.0
    steps = 0
    busy = 0.0
    pf_time = 0.0
    dec_steps = 0
    horizon_hit = False
    while ev or running or waiting:
        if t > horizon:
            horizon_hit = True
            break
        pool.now = t
        while ev and ev[0][0] <= t + 1e-12:
            et, _, kind, payload = heapq.heappop(ev)
            if kind == "task":
                j = payload
                yield_pins(j, cal.plan_prompt + tasks[j].plan_out, et)
                mk(j, "plan", cal.plan_prompt, 0, tasks[j].plan_out, 0.0, et)
            elif kind == "ret":
                on_return(payload, et)
            elif kind == "arrive":
                r = payload
                if not r.cancelled:
                    if cfg.pin_waiting_prefix and r.kind != "plan":
                        h = hit_blocks(r)
                        for b in range(h):
                            pool.touch((r.task, b))
                        r.blocks = [(r.task, b) for b in range(h)]
                        r.computed = h * B
                        r.cached = r.computed
                    heapq.heappush(waiting, (r.prio, r.t_arr, r.rid, r))
            elif kind == "cancel":
                r = payload
                if r.running:
                    running.remove(r)
                    release(r, et)
                elif r.blocks:
                    release(r, et)
            elif kind in ("first", "done"):
                on_client(kind, payload, et)
            elif kind == "refresh":
                r0 = payload
                j = r0.task
                s = stt[j]
                if s.get("refresh_of") is r0 and s["returned"] < tasks[j].n and s["final"] is None:
                    rr = mk(j, "preload", r0.prompt, r0.shared * B, 1, 0.0, et, r0.log_k)
                    npre[j] -= 1
                    rr.refresh = True
                    rr.keep = True
                    diag["refresh"] += 1
                    s["refresh_of"] = rr
        # drop cancelled from waiting lazily
        if not running and not waiting:
            if ev:
                t = max(t, ev[0][0])
                continue
            break
        budget = cfg.budget
        sched: list = []  # (req, tokens)
        preempted = False
        i = 0
        while i < len(running) and budget > 0:
            r = running[i]
            if r.computed < r.prompt:
                x = min(r.prompt - r.computed, budget)
                target = r.computed + x
            else:
                x = 1
                target = r.prompt + r.gen
            while not grow(r, target):
                victim = max(running, key=lambda q: (q.prio, q.t_arr, q.rid))
                diag["preempt_" + victim.kind + ("_pin" if victim.pinned else "")] += 1
                preempt += 1
                preempted = True
                running.remove(victim)
                release(victim, t)
                for q, y in sched:
                    if q is victim:
                        budget += y
                sched = [(q, y) for q, y in sched if q is not victim]
                victim.prompt += victim.gen  # recompute preemption: generated tokens are recomputed as prompt
                victim.out = max(1, victim.out - victim.gen)
                victim.gen = 0
                victim.computed = 0
                heapq.heappush(waiting, (victim.prio, victim.t_arr, victim.rid, victim))
                if victim is r:
                    break
            if r not in running:
                continue
            sched.append((r, x))
            budget -= x
            i = running.index(r) + 1
        if not preempted:
            while waiting and budget > 0 and len(running) < cfg.max_seqs:
                pr, ta, _, r = waiting[0]
                if r.cancelled:
                    heapq.heappop(waiting)
                    continue
                if r.blocks:  # pin_waiting_prefix idealisation: prefix already referenced
                    h = len(r.blocks)
                    evictable_hits = 0
                else:
                    h = hit_blocks(r)
                    evictable_hits = sum(1 for b in range(h) if (r.task, b) in pool.lru)
                full_blocks = (r.prompt + r.gen + B - 1) // B
                gate_need = (full_blocks - h) + evictable_hits if cfg.full_isl_gate else 0
                if gate_need > pool.free_count():
                    hol += 1
                    if cfg.hol_break:
                        break
                    heapq.heappop(waiting)  # (non-vLLM variant) skip it this step
                    heapq.heappush(ev, (t + 1e-6, next(sq), "arrive", r))
                    continue
                heapq.heappop(waiting)
                if not r.blocks:
                    for b in range(h):
                        pool.touch((r.task, b))
                    r.blocks = [(r.task, b) for b in range(h)]
                    r.computed = h * B
                    r.cached = r.computed
                x = min(r.prompt - r.computed, budget)
                if not grow(r, r.computed + x):
                    # cannot happen after the gate unless the gate is disabled: put back and stop
                    heapq.heappush(waiting, (r.prio, r.t_arr, r.rid, r))
                    break
                r.running = True
                running.append(r)
                sched.append((r, x))
                budget -= x
        if not sched:
            if ev:
                t = max(t, ev[0][0])
                continue
            if not running:  # the head of the queue can never fit (prompt > capacity): stop
                horizon_hit = True
                break
            t += cal.a
            continue
        dec = sum(1 for r, x in sched if r.computed >= r.prompt)
        dt = cal.a + cal.c * dec
        for r, x in sched:
            if r.computed < r.prompt:
                pc = cal.prefill_cost(x, r.computed)
                dt += pc
                pf_time += pc
        dec_steps += dec
        t += dt
        pool.now = t
        steps += 1
        busy += dt
        kv_peak = max(kv_peak, pool.referenced() * B)
        for r, x in sched:
            if r.computed < r.prompt:
                r.computed += x
                computed_by[r.kind] += x
                rho_win.append((t, cal.prefill_cost(x, r.computed - x)))
                if r.computed >= r.prompt:
                    r.gen = 1
                    r.first_t = t
                    heapq.heappush(ev, (t + cal.delta / 2, next(sq), "first", r))
            else:
                r.gen += 1
        for r in [r for r, _ in sched if r.gen >= r.out and r.running]:
            running.remove(r)
            r.done_t = t
            release(r, t)
            heapq.heappush(ev, (t + cal.delta / 2, next(sq), "done", r))
    fin_c = [stt[j]["final"].cached if stt[j]["final"] else None for j in range(nT)]
    fin_p = [stt[j]["final"].prompt if stt[j]["final"] else None for j in range(nT)]
    return VOut(lat, stage, sum(computed_by.values()), computed_by, fin_c, fin_p, npre, preempt, steps, busy,
                horizon_hit, kv_peak, hol, sig_obs, decisions, pool.evict_ages, dict(diag), pf_time, dec_steps, pre_max, yielded)


# ===================================================================================== 8. data and loaders
SEED2 = {
    # tag: (directory, rate, modes) of the measured runs (vLLM 0.32, A40, Qwen3-8B, arrival seed 2)
    "full": ("results/real/preload_seeds/seed2", "0.6", ("off", "on")),
    "full_chain": ("results/real/preload_seeds/seed2_chain", "0.6", ("on", "chain")),
    "kv80k": ("results/real/preload_kv/kv80k", "0.6", ("off", "on")),
    "kv50k": ("results/real/preload_kv/kv50k", "0.6", ("off", "on", "low")),
    "kv40k": ("results/real/preload_kv/kv40k", "0.6", ("off", "on", "low")),
    "kv40k_r03": ("results/real/preload_kv/kv40k", "0.3", ("off", "on")),
}
CAPACITY = {"full": 167_152, "full_chain": 167_152, "kv80k": 80_000, "kv50k": 50_000, "kv40k": 40_000,
            "kv40k_r03": 40_000}


def load_runs(tag: str, root: str | Path = ".") -> dict:
    d, rate, modes = SEED2[tag]
    out = {}
    for m in modes:
        with open(Path(root) / d / f"tasks_r{rate}_fcfs_pre-{m}.csv") as f:
            out[m] = {r["task_id"]: r for r in csv.DictReader(f)}
    return out


def width_from_prompt(final_prompt_tokens: int, cal: Calib = Calib()) -> int:
    """Number of results in a final prompt (results are min(t, 1400) + 35 header tokens; most hit the cap)."""
    return max(1, round((final_prompt_tokens - cal.header - cal.suffix) / (cal.result_cap + cal.block_hdr)))


def seed2_workload(tag: str = "kv50k", root: str | Path = ".", seed: int = 0, cal: Calib = Calib(),
                   narrow_plan_out: int = 128) -> list[VTask]:
    """Engine tasks from a measured off run: exact arrival times and final prompt sizes; the number of results
    from the prompt size; per-result sizes from the traced size distribution rescaled to the measured total; tool
    delays sampled from the run's DelayModel (not observable); plan output N(127, 28) capped at max_tokens 128 for
    wide plans and 128 for one-query plans (ASSUMED: the planner-truncation hypothesis; preload_theory used 25)."""
    runs = load_runs(tag, root)
    base = runs["off"] if "off" in runs else runs[next(iter(runs))]
    rows = sorted(base.values(), key=lambda r: float(r["arrival"]))
    rng = random.Random(seed)
    toks = result_token_samples(root)
    out = []
    for r in rows:
        fp = int(r["final_prompt_tokens"])
        n = width_from_prompt(fp, cal)
        total = max(n, fp - cal.header - cal.suffix)
        raw = [min(rng.choice(toks), cal.result_cap) + cal.block_hdr for _ in range(n)]
        scale = total / sum(raw)
        sizes = [max(1, int(round(x * scale))) for x in raw]
        sizes[-1] += total - sum(sizes)
        delays = tuple(sample_delay(rng, cal.delay) for _ in range(n))
        po = narrow_plan_out if n == 1 else min(128, max(40, int(round(rng.gauss(*cal.plan_out_wide)))))
        out.append(VTask(float(r["arrival"]), n, delays, tuple(sizes), po))
    return out


def measured_paired(tag: str, base: str, treat: str, root: str | Path = ".", other: str | None = None) -> tuple:
    """Mean paired saving (base - treat latency) with a 95% bootstrap CI. `other` = tag of a second directory
    holding `treat` (e.g. off from 'full' vs chain from 'full_chain')."""
    A = load_runs(tag, root)[base]
    Bm = load_runs(other or tag, root)[treat]
    ids = sorted(set(A) & set(Bm))
    d = [float(A[i]["latency"]) - float(Bm[i]["latency"]) for i in ids]
    return bootstrap_ci(d)


def preload_survival_measured(tag: str, mode: str, computed_preload: float, root: str | Path = ".") -> float:
    """sigma = (final prompt tokens served from cache) / (prompt tokens computed by preloads); the second number
    comes from the vLLM log totals (`LOG_TOTALS`)."""
    R = load_runs(tag, root)[mode]
    cached = sum(int(r["final_cached_tokens"] or 0) for r in R.values())
    return cached / computed_preload


# Computed prompt tokens per run = sum over 10-s log intervals of vLLM's "Avg prompt throughput" x 10 s, from
# results/logs_pod/serve_kv{5000,3125,2500}.log; run windows identified by the prefix-cache-hit reset between runs.
# Reproduced by `log_totals()` when the logs are present.
LOG_WINDOWS = {
    "serve_kv5000": [("kv80k", "off", "10:52:21", "10:55:31"), ("kv80k", "on", "10:55:41", "10:58:51")],
    "serve_kv3125": [("kv50k", "off", "11:00:25", "11:03:45"), ("kv50k", "on", "11:03:55", "11:07:35"),
                     ("kv50k", "low", "11:07:45", "11:10:55")],
    "serve_kv2500": [("kv40k", "off", "11:12:25", "11:15:35"), ("kv40k", "on", "11:15:45", "11:19:15"),
                     ("kv40k", "low", "11:19:25", "11:22:55"), ("kv40k_r03", "off", "11:23:05", "11:26:05"),
                     ("kv40k_r03", "on", "11:26:15", "11:29:05")],
}
_STAT = re.compile(r"INFO \d\d-\d\d (\d\d:\d\d:\d\d) \[loggers.py:\d+\] Engine 000: Avg prompt throughput: ([\d.]+) "
                   r"tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) "
                   r"reqs,(?: Deferred: (\d+) reqs,)? GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: "
                   r"([\d.]+)%")


def log_totals(root: str | Path = ".") -> dict:
    """(tag, mode) -> dict(computed prompt tokens, generated tokens, mean KV usage fraction (referenced),
    final prefix-cache hit rate, mean waiting, mean deferred)."""
    out = {}
    for f, wins in LOG_WINDOWS.items():
        p = Path(root) / "results/logs_pod" / f"{f}.log"
        if not p.exists():
            continue
        rows = []
        for line in open(p):
            m = _STAT.search(line)
            if m:
                rows.append((m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4)), int(m.group(5)),
                             int(m.group(6) or 0), float(m.group(7)) / 100, float(m.group(8)) / 100))
        for tag, mode, a, b in wins:
            s = [r for r in rows if a <= r[0] <= b]
            out[(tag, mode)] = dict(computed=sum(r[1] for r in s) * 10, generated=sum(r[2] for r in s) * 10,
                                    kv_ref=sum(r[6] for r in s) / len(s), hit=s[-1][7],
                                    waiting=sum(r[4] for r in s) / len(s), deferred=sum(r[5] for r in s) / len(s),
                                    wall=10 * len(s))
    return out


def predict_vllm(tag: str, policies=("off", "on", "low"), seeds=range(4), cal: Calib = Calib(),
                 cfg: VCfg | None = None, ctrl_factory=None, root: str | Path = ".") -> dict:
    """Mean over seeds of: mean latency per policy, paired saving vs off (and vs on), computed prefill tokens."""
    cap = CAPACITY[tag]
    cfg = cfg or VCfg(kv_capacity=cap)
    acc: dict = {p: dict(lat=[], comp=[], pre=[], fcached=[], plan=[]) for p in policies}
    sav: dict = {p: [] for p in policies}
    sav_on: dict = {p: [] for p in policies}
    for s in seeds:
        tasks = seed2_workload(tag, root, seed=s, cal=cal)
        res = {}
        for p in policies:
            ctrl = ctrl_factory(cap) if (p == "ctrl" and ctrl_factory) else (CtrlParams(cap) if p == "ctrl" else None)
            res[p] = simulate_vllm(tasks, p, cal, cfg, ctrl)
        for p in policies:
            L = res[p].latency
            acc[p]["lat"].append(sum(L) / len(L))
            acc[p]["comp"].append(res[p].computed_prefill)
            acc[p]["pre"].append(res[p].computed_by_kind["preload"])
            acc[p]["fcached"].append(sum(x or 0 for x in res[p].final_cached))
            pdur = [res[p].stage["plan_done"][j] - tasks[j].arrival for j in range(len(tasks))]
            acc[p]["plan"].append(sum(pdur) / len(pdur))
            if "off" in res:
                d = [x - y for x, y in zip(res["off"].latency, L)]
                sav[p].append(sum(d) / len(d))
            if "on" in res:
                d = [x - y for x, y in zip(res["on"].latency, L)]
                sav_on[p].append(sum(d) / len(d))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {p: dict(lat=mean(acc[p]["lat"]), saving_vs_off=mean(sav[p]), saving_vs_on=mean(sav_on[p]),
                    saving_range=(min(sav[p]), max(sav[p])) if sav[p] else None, computed=mean(acc[p]["comp"]),
                    preload_computed=mean(acc[p]["pre"]), final_cached=mean(acc[p]["fcached"]),
                    plan_stage=mean(acc[p]["plan"])) for p in policies}


MEASURED_SAVING = {  # paired mean saving vs off (s), results/real/preload_kv/paired.txt
    ("kv80k", "on"): 2.52, ("kv50k", "on"): -11.28, ("kv50k", "low"): -1.71, ("kv40k", "on"): -11.12,
    ("kv40k", "low"): -4.51, ("kv40k_r03", "on"): 0.65}


def breakeven_table(root: str | Path = ".", cal: Calib = Calib()) -> list[dict]:
    """Theorem BE' against the data, zero fitted parameters. Inputs per condition:
    - u: prefill utilization of the OFF run = computed prompt tokens (vLLM log) x mean prefill cost per token of
      the off finals (cost model) / wall time of the run;
    - X: stretchable critical-path time = (1 - u) x mean(plan stage + final stage) of the off run, where the plan
      stage = (last tool return - arrival) - E[max tool delay | width] (the delay law; plan end is not logged);
    - sigma: survival = final cached tokens / prompt tokens computed by preloads (log total - final computed -
      plan prompts) in the TREATMENT run; waste w = (computed by preloads - cached) per task.
    The magnitude column uses the treatment run's measured waste: it is a decomposition, not a forecast."""
    from .preload_theory import expected_max_delay
    import statistics as _st
    lt = log_totals(root)
    emax = {n: expected_max_delay(n) for n in range(1, 9)}
    rows = []
    for (tag, mode), meas in MEASURED_SAVING.items():
        if (tag, mode) not in lt:
            continue
        runs = load_runs(tag, root)
        off, tr = list(runs["off"].values()), list(runs[mode].values())
        lam = len(off) / 180.0
        ct = sum(cal.prefill_cost(int(r["final_prompt_tokens"]), 0) for r in off) / sum(
            int(r["final_prompt_tokens"]) for r in off)
        u = lt[(tag, "off")]["computed"] * ct / lt[(tag, "off")]["wall"]
        plan = [float(r["t_last_tool"]) - float(r["arrival"]) - emax[width_from_prompt(int(r["final_prompt_tokens"]))]
                for r in off]
        fin = [float(r["end"]) - float(r["final_submit"]) for r in off]
        X = (1 - u) * (_st.mean(plan) + _st.mean(fin))
        pre = lt[(tag, mode)]["computed"] - sum(int(r["final_prompt_tokens"]) - int(r["final_cached_tokens"] or 0)
                                                for r in tr) - cal.plan_prompt * len(tr)
        cached = sum(int(r["final_cached_tokens"] or 0) for r in tr)
        s_bar, w_bar = cached * ct / len(tr), max(0, pre - cached) * ct / len(tr)
        den = 1 - u - lam * w_bar
        pred = X / (1 - u) - (X - s_bar) / den if den > 0 else -INF
        rows.append(dict(tag=tag, mode=mode, u=u, X=X, rho_s=lam * X, sigma=cached / pre,
                         sigma_star=sigma_star(u, lam * X), s_bar=s_bar, w_bar=w_bar, predicted=pred, measured=meas))
    return rows


def fluid_sigma(M: float, Rbar: float, v_dead: float, lam_w: float, q: float = 0.0, n: int = 8,
                samples: int = 4000, seed: int = 0, cal: Calib = Calib()) -> tuple:
    """Fluid (Che/TTL) prediction of the 'on'-mode survival fraction for wide tasks. Each preload k is released
    at the k-th return and next referenced at the (k+1)-th (instantaneous preloads; q = extra unreferenced
    queueing time per reference); the TTL T solves M - Rbar = v_dead T + lam_w E[sum_k L_k min(T, G_k + q)].
    A prefix survives a gap iff G_k + q <= T; an evicted prefix is recomputed whole by the next preload.
    Returns (T, sigma = useful / computed preload tokens)."""
    rng = random.Random(seed)
    toks = result_token_samples()
    S = []
    for _ in range(samples):
        d = sorted(sample_delay(rng, cal.delay) for _ in range(n))
        L = [cal.header]
        for _i in range(n):
            L.append(L[-1] + min(rng.choice(toks), cal.result_cap) + cal.block_hdr)
        S.append([(L[k], d[k] - d[k - 1]) for k in range(1, n)])

    def live(T):
        return lam_w * sum(sum(Lk * min(T, g + q) for Lk, g in s) for s in S) / len(S)
    T = ttl_fixed_point(M - Rbar, v_dead, live)
    comp = use = 0.0
    for s in S:
        prev_ok, prevL = True, 0
        for Lk, g in s:
            comp += Lk - (prevL if prev_ok else 0)
            prev_ok, prevL = (g + q <= T), Lk
        use += s[-1][0] if prev_ok else 0
    return T, use / comp


def fluid_table(root: str | Path = ".", q: float = 0.0) -> list[dict]:
    """fluid_sigma for each measured capacity, inputs from the OFF run of the same condition: Rbar = mean
    referenced KV (vLLM 'GPU KV cache usage' x M), v_dead = computed prompt tokens / wall time."""
    lt = log_totals(root)
    out = []
    for tag in ("kv80k", "kv50k", "kv40k", "kv40k_r03"):
        if (tag, "off") not in lt:
            continue
        o = lt[(tag, "off")]
        runs = load_runs(tag, root)["off"]
        lam = len(runs) / 180.0
        qw = sum(width_from_prompt(int(r["final_prompt_tokens"])) > 1 for r in runs.values()) / len(runs)
        M = CAPACITY[tag]
        T, sg = fluid_sigma(M, o["kv_ref"] * M, o["computed"] / o["wall"], lam * qw, q)
        out.append(dict(tag=tag, M=M, Rbar=o["kv_ref"] * M, v_dead=o["computed"] / o["wall"], T=T, sigma=sg))
    return out
