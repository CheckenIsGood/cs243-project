"""Exact machinery for docs/stochastic_theory.md (agent serving with random tool lags, bounded KV, batching).

All arithmetic that decides anything is exact (fractions.Fraction); Z3 is used for LPs and algebra.

Contents
1. Distributions: finite discrete laws as tuples of (value, probability) with Fraction entries.
2. Machine model M_m (Sections 1-3 of the doc): tasks are DAGs of LLM nodes on m identical machines (preemption,
   migration); arcs carry random lags (tool calls, unlimited parallelism, not sped up).
   - `sim_machine`: exact event-driven simulation of the preemptive task-index policy (P-WSPT-DAG).
   - lower bounds: critical path (LB_cp), fast-single-machine Smith bound (deterministic sizes), the task-level
     Moehring-Schulz-Uetz polymatroid bound LB2 (random sizes), the node-level MSU LP with lag and critical-path
     constraints (`msu_lp`, Z3).
   - exact optimal policies on tiny instances (integer data): `opt_np_dp` (non-preemptive non-anticipatory),
     `opt_pmtn_dp` (preemptive non-anticipatory on the integer time grid), `opt_clair_1m` (clairvoyant, m = 1).
3. Hybrid model H (Section 4): a prefill server (rate 1/alpha tokens/s), m decode machines (rate 1/a tokens/s
   each), memory budget M' for admitted nodes, priority-prefix admission with kicks, held KV in the lowest
   eviction class (any keep policy), lowest-priority preload chunks. `sim_hybrid` (fluid, exact) and
   `sim_discrete` (atomic engine steps).
4. Keep/evict layer (Section 5): single-prefix stochastic ski rental (`ttl_*`), the many-tenant LP with a memory
   price (`keep_lp_z3`, `keep_dual`).
5. Lower-bound constructions (Section 7).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction as F
from functools import lru_cache
from typing import Iterable, Sequence

import z3

# ============================================================================================ 1. distributions
Dist = tuple  # tuple[tuple[F, F], ...], values distinct, probabilities > 0 summing to 1


def Fr(x) -> F:
    return x if isinstance(x, F) else F(str(x)) if isinstance(x, float) else F(x)


def dist(*pairs) -> Dist:
    """dist((v1, p1), (v2, p2), ...) -> normalized law (merges equal values, drops zero mass)."""
    acc: dict[F, F] = {}
    for v, p in pairs:
        v, p = Fr(v), Fr(p)
        if p:
            acc[v] = acc.get(v, F(0)) + p
    tot = sum(acc.values())
    assert tot > 0
    return tuple(sorted((v, p / tot) for v, p in acc.items()))


def det(v) -> Dist:
    return ((Fr(v), F(1)),)


def mean(d: Dist) -> F:
    return sum(v * p for v, p in d)


def var(d: Dist) -> F:
    mu = mean(d)
    return sum(p * (v - mu) ** 2 for v, p in d)


def cv2(d: Dist) -> F:
    mu = mean(d)
    return var(d) / mu**2 if mu else F(0)


def dmax(*ds: Dist) -> Dist:
    """Law of the max of independent variables (used to collapse a tool fan-out into one arc lag)."""
    out = det(0)
    for d in ds:
        out = dist(*[(max(a, b), p * q) for a, p in out for b, q in d])
    return out


def is_det(d: Dist) -> bool:
    return len(d) == 1


# ============================================================================================ 2. machine model
@dataclass(frozen=True)
class MTask:
    """A task of the machine model: release r, weight w, nodes 0..n-1 with processing-time laws (topologically
    ordered: every arc goes from a lower to a higher index), arcs (u, v, lag law)."""

    r: F
    w: F
    nodes: tuple  # tuple[Dist, ...]
    arcs: tuple = ()  # tuple[(u, v, Dist), ...]

    def preds(self, v: int) -> list[int]:
        return [u for (u, x, _) in self.arcs if x == v]


def chain_task(r, w, sizes: Sequence[Dist], lags: Sequence[Dist]) -> MTask:
    assert len(lags) == len(sizes) - 1
    return MTask(Fr(r), Fr(w), tuple(sizes), tuple((i, i + 1, lags[i]) for i in range(len(lags))))


def realizations(tasks: Sequence[MTask]):
    """Enumerate joint realizations (prob, p[t][v], lag[t][k]) assuming independence of all quantities."""
    slots = []
    for t, T in enumerate(tasks):
        for v, d in enumerate(T.nodes):
            slots.append(("p", t, v, d))
        for k, (_, _, d) in enumerate(T.arcs):
            slots.append(("d", t, k, d))
    for combo in itertools.product(*[s[3] for s in slots]):
        prob = F(1)
        P = [[None] * len(T.nodes) for T in tasks]
        D = [[None] * len(T.arcs) for T in tasks]
        for s, (val, pr) in zip(slots, combo):
            prob *= pr
            (P if s[0] == "p" else D)[s[1]][s[2]] = val
        yield prob, P, D


def task_index_order(tasks: Sequence[MTask], size=None) -> list[int]:
    """Task priority order of P-WSPT-DAG: decreasing w_J / E[W_J] (Smith ratio on expected total work), ties by
    task id. `size(t)` may override E[W_J]."""
    def ew(t):
        return size(t) if size else sum(mean(d) for d in tasks[t].nodes)

    def key(t):
        s = ew(t)
        return (-(tasks[t].w / s) if s > 0 else -F(10**18), t)

    return sorted(range(len(tasks)), key=key)


def sim_machine(tasks: Sequence[MTask], P, D, m: int, order: Sequence[int], preemptive: bool = True,
                speed: F = F(1)):
    """Exact simulation of the task-priority list policy on m machines.

    preemptive=True: at every instant run the m highest-priority available nodes (priority = (rank of task in
    `order`, node index)), preempting and migrating freely.  preemptive=False: Graham list scheduling (a started
    node runs to completion; a free machine takes the highest-priority available node).
    Returns (task completion times, node completion times)."""
    rank = {t: i for i, t in enumerate(order)}
    nodes = [(t, v) for t in range(len(tasks)) for v in range(len(tasks[t].nodes))]
    rem = {(t, v): F(P[t][v]) for (t, v) in nodes}
    C: dict = {}
    avail_at: dict = {}
    for t, T in enumerate(tasks):
        for v in range(len(T.nodes)):
            if not T.preds(v):
                avail_at[(t, v)] = T.r
    started: set = set()
    now = F(0)

    def try_avail():
        changed = True
        while changed:
            changed = False
            for t, T in enumerate(tasks):
                for v in range(len(T.nodes)):
                    if (t, v) in avail_at:
                        continue
                    ins = [(u, k) for k, (u, x, _) in enumerate(T.arcs) if x == v]
                    if all((t, u) in C for u, _ in ins):
                        avail_at[(t, v)] = max([T.r] + [C[(t, u)] + D[t][k] for u, k in ins])
                        changed = True

    try_avail()
    while len(C) < len(nodes):
        # instantaneous completions of zero-work available nodes
        zero = [n for n in nodes if n not in C and n in avail_at and avail_at[n] <= now and rem[n] == 0]
        if zero:
            for n in zero:
                C[n] = now
            try_avail()
            continue
        avail = sorted([n for n in nodes if n not in C and n in avail_at and avail_at[n] <= now],
                       key=lambda n: (rank[n[0]], n[1]))
        if preemptive:
            run = avail[:m]
        else:
            run = [n for n in avail if n in started]
            for n in avail:
                if len(run) >= m:
                    break
                if n not in started:
                    run.append(n)
                    started.add(n)
        future = [avail_at[n] for n in nodes if n not in C and n in avail_at and avail_at[n] > now]
        nxt = [now + rem[n] / speed for n in run] + future
        assert nxt, "deadlock"
        t1 = min(nxt)
        for n in run:
            rem[n] -= (t1 - now) * speed
            if rem[n] == 0:
                C[n] = t1
        now = t1
        try_avail()
    TC = [max(C[(t, v)] for v in range(len(T.nodes))) for t, T in enumerate(tasks)]
    return TC, C


def critical_path(T: MTask, p, d) -> F:
    """Longest r-relative path (processing + lags) of one realization of task T."""
    fin = [F(0)] * len(T.nodes)
    for v in range(len(T.nodes)):
        ins = [(u, k) for k, (u, x, _) in enumerate(T.arcs) if x == v]
        start = max([F(0)] + [fin[u] + d[k] for u, k in ins])
        fin[v] = start + p[v]
    return max(fin)


def pathwise_bound(tasks, P, D, m: int, order) -> list[F]:
    """Theorem A/B pathwise bound: C_J <= r_J + CP_J + (1/m) * sum_{K ranked <= J} W_K."""
    pos = {t: i for i, t in enumerate(order)}
    out = []
    for t, T in enumerate(tasks):
        work = sum(sum(P[k]) for k in range(len(tasks)) if pos[k] <= pos[t])
        out.append(T.r + critical_path(T, P[t], D[t]) + F(work) / m)
    return out


def expected_cost(tasks, m, order, preemptive=True) -> F:
    tot = F(0)
    for prob, P, D in realizations(tasks):
        TC, _ = sim_machine(tasks, P, D, m, order, preemptive)
        tot += prob * sum(T.w * c for T, c in zip(tasks, TC))
    return tot


def lb_cp(tasks) -> F:
    """LB1 = sum_J w_J (r_J + E[CP_J]), valid for every policy, clairvoyant or not."""
    tot = F(0)
    for T in tasks:
        e = F(0)
        for prob, P, D in realizations([T]):
            e += prob * critical_path(T, P[0], D[0])
        tot += T.w * (T.r + e)
    return tot


def smith(ws: Sequence[F], sizes: Sequence[F]) -> F:
    """min over orders of sum_k w_k * (cumulative size up to k): Smith's rule (ratio order)."""
    idx = sorted(range(len(ws)), key=lambda i: (-(ws[i] / sizes[i]) if sizes[i] > 0 else -F(10**18), i))
    cum, tot = F(0), F(0)
    for i in idx:
        cum += sizes[i]
        tot += ws[i] * cum
    return tot


def order_value(ws, sizes, order) -> F:
    cum, tot = F(0), F(0)
    for i in order:
        cum += sizes[i]
        tot += ws[i] * cum
    return tot


def lb_smith_det(tasks, m) -> F:
    """Deterministic sizes: (1/m) * Smith(W) is a lower bound on every m-machine schedule (fast single machine)."""
    return smith([T.w for T in tasks], [sum(mean(d) for d in T.nodes) / m for T in tasks])


def msu_gamma(tasks, m, Delta=None) -> F:
    """gamma = min(0, 1/2 - (m-1)Delta/(2m)) of the task-level MSU bound (Delta defaults to the max node CV^2)."""
    if Delta is None:
        Delta = max(cv2(d) for T in tasks for d in T.nodes)
    b = F(1, 2) - F(m - 1) * Delta / (2 * m)
    return min(F(0), b)


def lb2_task_msu(tasks, m, Delta=None) -> F:
    """LB2: optimum of min sum w_J C_J s.t. sum_{J in A} E[W_J] C_J >= g(A) for all task sets A, with
    g(A) = (1/2m) E[W(A)]^2 + gamma sum_{J in A} E[W_J]^2; equal to the greedy (Smith-order) vertex."""
    g = msu_gamma(tasks, m, Delta)
    ew = [sum(mean(d) for d in T.nodes) for T in tasks]
    idx = task_index_order(tasks)
    X, tot = F(0), F(0)
    for k in idx:
        tot += tasks[k].w * (X / m + (F(1, 2 * m) + g) * ew[k])
        X += ew[k]
    return tot


def lb2_task_lp_z3(tasks, m, Delta=None) -> F:
    """The same task-level LP solved directly (all 2^n subset constraints) with Z3 over the rationals."""
    g = msu_gamma(tasks, m, Delta)
    ew = [sum(mean(d) for d in T.nodes) for T in tasks]
    opt = z3.Optimize()
    Cv = [z3.Real(f"C{k}") for k in range(len(tasks))]
    for r in range(1, len(tasks) + 1):
        for A in itertools.combinations(range(len(tasks)), r):
            s = sum(ew[k] for k in A)
            opt.add(z3.Sum([z3.RealVal(ew[k]) * Cv[k] for k in A]) >= z3.RealVal(s * s / (2 * m) + g * sum(ew[k] ** 2 for k in A)))
    h = opt.minimize(z3.Sum([z3.RealVal(T.w) * c for T, c in zip(tasks, Cv)]))
    assert opt.check() == z3.sat
    return _zfrac(opt.model().eval(z3.Sum([z3.RealVal(T.w) * c for T, c in zip(tasks, Cv)])))


def _zfrac(v) -> F:
    v = z3.simplify(v)
    return F(v.numerator_as_long(), v.denominator_as_long())


def msu_lp(tasks, m, Delta=None, with_cp=True, drop_variance=False):
    """Node-level MSU LP with lag-precedence and (optionally) critical-path constraints, solved exactly with Z3.

    Variables: C_x per node, C_J per task. Constraints:
      sum_{x in A} E[p_x] C_x >= f(A) for all node sets A (MSU load inequalities, SU2005 (3.1)-(3.2)),
      C_v >= C_u + E[D_uv] + E[p_v] (lag precedence),  C_x >= r_J + E[p_x],  C_J >= C_x,
      C_J >= r_J + E[CP_J] (critical path; valid pointwise).
    drop_variance=True removes the -(m-1)(Delta-1)/(2m) correction (an INVALID 'LP', used to show it is needed).
    Returns (LP value, {node: C_x}, [C_J])."""
    nodes = [(t, v) for t, T in enumerate(tasks) for v in range(len(T.nodes))]
    mu = {(t, v): mean(tasks[t].nodes[v]) for (t, v) in nodes}
    if Delta is None:
        Delta = max(cv2(tasks[t].nodes[v]) for (t, v) in nodes)
    opt = z3.Optimize()
    Cx = {n: z3.Real(f"x{n[0]}_{n[1]}") for n in nodes}
    CJ = [z3.Real(f"J{t}") for t in range(len(tasks))]
    for r in range(1, len(nodes) + 1):
        for A in itertools.combinations(nodes, r):
            s = sum(mu[n] for n in A)
            s2 = sum(mu[n] ** 2 for n in A)
            f = (s * s + s2) / (2 * m) - (F(0) if drop_variance else F(m - 1) * (Delta - 1) / (2 * m) * s2)
            opt.add(z3.Sum([z3.RealVal(mu[n]) * Cx[n] for n in A]) >= z3.RealVal(f))
    for t, T in enumerate(tasks):
        for v in range(len(T.nodes)):
            opt.add(Cx[(t, v)] >= z3.RealVal(T.r + mu[(t, v)]))
            opt.add(CJ[t] >= Cx[(t, v)])
        for (u, v, d) in T.arcs:
            opt.add(Cx[(t, v)] >= Cx[(t, u)] + z3.RealVal(mean(d) + mu[(t, v)]))
        if with_cp:
            e = sum(pr * critical_path(T, P[0], D[0]) for pr, P, D in realizations([T]))
            opt.add(CJ[t] >= z3.RealVal(T.r + e))
    obj = z3.Sum([z3.RealVal(T.w) * CJ[t] for t, T in enumerate(tasks)])
    opt.minimize(obj)
    assert opt.check() == z3.sat
    mdl = opt.model()
    return (_zfrac(mdl.eval(obj)), {n: _zfrac(mdl.eval(Cx[n])) for n in nodes},
            [_zfrac(mdl.eval(c)) for c in CJ])


def lp_order(tasks, m, Delta=None) -> list[int]:
    """Offline policy order: tasks by increasing LP value of C_J (ties by id)."""
    _, _, CJ = msu_lp(tasks, m, Delta)
    return sorted(range(len(tasks)), key=lambda t: (CJ[t], t))


# ------------------------------------------------------------------------------------------- exact DPs (tiny)
def _hazard(d: Dist, e: int) -> F:
    """P(X = e | X >= e) for an integer-valued law."""
    tail = sum(p for v, p in d if v >= e)
    return (sum(p for v, p in d if v == e) / tail) if tail else F(1)


def _arcs_in(T: MTask, v):
    return [k for k, (_, x, _) in enumerate(T.arcs) if x == v]


def _check_int(tasks, min_p=1):
    for T in tasks:
        assert T.r.denominator == 1
        for d in T.nodes:
            assert all(v.denominator == 1 and v >= min_p for v, _ in d)
        for (_, _, d) in T.arcs:
            assert all(v.denominator == 1 and v >= 0 for v, _ in d)


def _resolve_lags(tasks, done, lags, expired):
    """Branch over lag expirations at the current instant. lags: frozenset of ((t,k), elapsed). Yields
    (prob, lags', expired') where every lag in lags' did NOT expire now."""
    items = sorted(lags)
    outs = [(F(1), [], set(expired))]
    for (tk, e) in items:
        t, k = tk
        h = _hazard(tasks[t].arcs[k][2], e)
        nxt = []
        for pr, keep, ex in outs:
            if h:
                ex2 = set(ex)
                ex2.add(tk)
                nxt.append((pr * h, keep, ex2))
            if h != 1:
                nxt.append((pr * (1 - h), keep + [(tk, e)], ex))
        outs = nxt
    for pr, keep, ex in outs:
        yield pr, frozenset(keep), frozenset(ex)


def _available(tasks, t_now, done, busy, expired):
    out = []
    for t, T in enumerate(tasks):
        if t_now < T.r:
            continue
        for v in range(len(T.nodes)):
            n = (t, v)
            if n in done or n in busy:
                continue
            if all((t, k) in expired for k in _arcs_in(T, v)):
                out.append(n)
    return out


def opt_np_dp(tasks: Sequence[MTask], m: int) -> F:
    """Exact optimum over non-anticipatory NON-preemptive policies that decide at integer times (integer data,
    processing times >= 1). Idling allowed. Value = E[sum w_J C_J]."""
    _check_int(tasks)
    tasks = tuple(tasks)
    rmax = max(T.r for T in tasks)
    allnodes = frozenset((t, v) for t, T in enumerate(tasks) for v in range(len(T.nodes)))

    @lru_cache(maxsize=None)
    def V(tc, done, running, lags, expired):
        # state at an integer instant, lags already resolved for this instant
        if done == allnodes:
            return F(0)
        busy = {n for n, _ in running}
        avail = _available(tasks, tc, done, busy, expired)
        free = m - len(running)
        cost_unit = sum(T.w for t, T in enumerate(tasks)
                        if any((t, v) not in done for v in range(len(T.nodes))))
        best = None
        for k in range(0, min(free, len(avail)) + 1):
            for U in itertools.combinations(avail, k):
                run = tuple(sorted(running + tuple((n, 0) for n in U)))
                if not run and not lags and tc >= rmax:
                    continue  # idling forever is never optimal; also avoids a self-loop
                val = cost_unit + _advance(tc, done, run, lags, expired)
                if best is None or val < best:
                    best = val
        return best

    def _advance(tc, done, run, lags, expired):
        # one unit passes: running nodes may complete, lags age by one
        outs = [(F(1), set(done), [])]
        for (n, e) in run:
            t, v = n
            h = _hazard(tasks[t].nodes[v], e + 1)
            nxt = []
            for pr, dn, rr in outs:
                if h:
                    d2 = set(dn)
                    d2.add(n)
                    nxt.append((pr * h, d2, rr))
                if h != 1:
                    nxt.append((pr * (1 - h), dn, rr + [(n, e + 1)]))
            outs = nxt
        tot = F(0)
        for pr, dn, rr in outs:
            newl = set((tk, e + 1) for tk, e in lags)
            for n in dn - set(done):
                t, v = n
                for k, (u, x, _) in enumerate(tasks[t].arcs):
                    if u == v:
                        newl.add(((t, k), 0))
            tc2 = min(tc + 1, rmax)
            for pr2, l2, ex2 in _resolve_lags(tasks, dn, frozenset(newl), expired):
                tot += pr * pr2 * V(tc2, frozenset(dn), tuple(sorted(rr)), l2, ex2)
        return tot

    init = frozenset()
    tot = F(0)
    for pr, l2, ex2 in _resolve_lags(tasks, frozenset(), frozenset(), frozenset()):
        tot += pr * V(min(F(0), rmax), frozenset(), (), l2, ex2)
    return tot


def opt_pmtn_dp(tasks: Sequence[MTask], m: int) -> F:
    """Exact optimum over non-anticipatory PREEMPTIVE policies on the integer time grid (each unit, choose up to m
    available nodes to process for one unit). This is an UPPER bound on the true preemptive optimum (finer
    preemption may help when m >= 2); for m = 1 with integer data it is exact."""
    _check_int(tasks)
    tasks = tuple(tasks)
    rmax = max(T.r for T in tasks)
    allnodes = frozenset((t, v) for t, T in enumerate(tasks) for v in range(len(T.nodes)))

    @lru_cache(maxsize=None)
    def V(tc, done, att, lags, expired):
        if done == allnodes:
            return F(0)
        avail = _available(tasks, tc, done, set(), expired)
        cost_unit = sum(T.w for t, T in enumerate(tasks)
                        if any((t, v) not in done for v in range(len(T.nodes))))
        attd = dict(att)
        best = None
        for k in range(0, min(m, len(avail)) + 1):
            for U in itertools.combinations(avail, k):
                if not U and not lags and tc >= rmax:
                    continue
                outs = [(F(1), set(done), dict(attd))]
                for n in U:
                    t, v = n
                    e = attd.get(n, 0)
                    h = _hazard(tasks[t].nodes[v], e + 1)
                    nxt = []
                    for pr, dn, at in outs:
                        if h:
                            d2 = set(dn)
                            d2.add(n)
                            a2 = dict(at)
                            a2.pop(n, None)
                            nxt.append((pr * h, d2, a2))
                        if h != 1:
                            a2 = dict(at)
                            a2[n] = e + 1
                            nxt.append((pr * (1 - h), dn, a2))
                    outs = nxt
                val = cost_unit
                for pr, dn, at in outs:
                    newl = set((tk, e + 1) for tk, e in lags)
                    for n in dn - set(done):
                        t, v = n
                        for kk, (u, x, _) in enumerate(tasks[t].arcs):
                            if u == v:
                                newl.add(((t, kk), 0))
                    for pr2, l2, ex2 in _resolve_lags(tasks, dn, frozenset(newl), expired):
                        val += pr * pr2 * V(min(tc + 1, rmax), frozenset(dn), tuple(sorted(at.items())), l2, ex2)
                if best is None or val < best:
                    best = val
        return best

    tot = F(0)
    for pr, l2, ex2 in _resolve_lags(tasks, frozenset(), frozenset(), frozenset()):
        tot += pr * V(F(0), frozenset(), (), l2, ex2)
    return tot


def opt_clair_1m(tasks: Sequence[MTask]) -> F:
    """E over realizations of the clairvoyant optimum on ONE machine (preemptive, integer grid; exact for m = 1
    with integer data): for each realization, min sum w_J C_J by DP over remaining work."""
    _check_int(tasks)
    tot = F(0)
    for prob, P, D in realizations(tasks):
        det_tasks = tuple(MTask(T.r, T.w, tuple(det(P[t][v]) for v in range(len(T.nodes))),
                                tuple((u, v, det(D[t][k])) for k, (u, v, _) in enumerate(T.arcs)))
                          for t, T in enumerate(tasks))
        tot += prob * opt_pmtn_dp(det_tasks, 1)
    return tot


# ============================================================================================ 3. hybrid model H
@dataclass(frozen=True)
class HNode:
    """An LLM node of the hybrid model. Tokens: rho = parent-context tokens (recomputed unless the parent's KV is
    held), q = new prompt tokens, X = decode tokens (>= 1). If `pieces` is non-empty the node has one parent arc
    whose lag is the max of the piece lags (a tool fan-out) and the pieces' tokens are part of q (returned results
    appended in return order). Footprints: l_minus = rho + q (prompt), l_plus = rho + q + X (reserved)."""

    q: int
    X: int
    rho: int = 0
    pieces: tuple = ()  # tuple[(Dist lag, int tokens), ...]

    @property
    def l_minus(self):
        return self.rho + self.q

    @property
    def l_plus(self):
        return self.rho + self.q + self.X


@dataclass(frozen=True)
class HTask:
    r: F
    w: F
    nodes: tuple  # tuple[HNode, ...] topologically ordered
    arcs: tuple = ()  # (u, v, Dist); for a node with pieces the arc lag is ignored (use det(0))

    def preds(self, v):
        return [u for (u, x, _) in self.arcs if x == v]


@dataclass(frozen=True)
class HCfg:
    alpha: F  # prefill seconds per token (whole GPU)
    a: F  # decode seconds per token on one decode machine (batch-1 step time)
    m: int  # number of decode machines, m = a/c
    Mp: int | None = None  # memory budget M' for admitted reservations (None = unlimited)
    hold: str = "none"  # 'none' (evict-always) | 'lru' | 'fifo' | 'rand' (hold everything, evict held by order)
    preload: bool = False
    seed: int = 0

    @property
    def c(self):
        return self.a / self.m


def h_realizations(tasks: Sequence[HTask]):
    slots = []
    for t, T in enumerate(tasks):
        for k, (u, v, d) in enumerate(T.arcs):
            slots.append(("a", t, k, d))
        for v, nd in enumerate(T.nodes):
            for i, (d, _) in enumerate(nd.pieces):
                slots.append(("p", t, (v, i), d))
    for combo in itertools.product(*[s[3] for s in slots]):
        prob = F(1)
        lag = [dict() for _ in tasks]
        for s, (val, pr) in zip(slots, combo):
            prob *= pr
            lag[s[1]][(s[0], s[2])] = val
        yield prob, lag


def _h_arc_lag(T: HTask, lag, k):
    u, v, _ = T.arcs[k]
    nd = T.nodes[v]
    if nd.pieces:
        return max(lag[("p", (v, i))] for i in range(len(nd.pieces)))
    return lag[("a", k)]


def h_cp(T: HTask, lag, cfg: HCfg, kind: str) -> F:
    """Critical path of one realization. kind='alg': node length alpha(q+rho) + a X (evict-always sizes);
    kind='opt': alpha q + (a + c) X (the engine optimum's minimum per-node time; parent KV may be kept)."""
    fin = [F(0)] * len(T.nodes)
    for v, nd in enumerate(T.nodes):
        ins = [k for k, (_, x, _) in enumerate(T.arcs) if x == v]
        st = max([F(0)] + [fin[T.arcs[k][0]] + _h_arc_lag(T, lag, k) for k in ins])
        if kind == "alg":
            ln = cfg.alpha * (nd.q + nd.rho) + cfg.a * nd.X
        else:
            ln = cfg.alpha * nd.q + (cfg.a + cfg.c) * nd.X
        fin[v] = st + ln
    return max(fin)


def h_Lmax(tasks):
    return max(nd.l_plus for T in tasks for nd in T.nodes)


def h_sizes(tasks, cfg: HCfg):
    """Per-task sizes used by the policy index and the bound: Z'_K = sum_y alpha(q+rho) + kappa + c X + a X l+/M,
    with M = M' - L_max and kappa = alpha (l+ + L_max) when memory is bounded. Also returns sigma_K (GPU seconds,
    alpha q + c X) and area_K / M (l- (a+c) X / M) used by the lower bounds."""
    Lmax = h_Lmax(tasks)
    M = None if cfg.Mp is None else cfg.Mp - Lmax
    Z, S, A = [], [], []
    for T in tasks:
        z = s = ar = F(0)
        for nd in T.nodes:
            z += cfg.alpha * (nd.q + nd.rho) + cfg.c * nd.X
            s += cfg.alpha * nd.q + cfg.c * nd.X
            if M is not None:
                z += cfg.alpha * (nd.l_plus + Lmax) + cfg.a * nd.X * nd.l_plus / M
                ar += F(nd.l_minus) * (cfg.a + cfg.c) * nd.X / M
        Z.append(z)
        S.append(s)
        A.append(ar)
    return Z, S, A


def h_order(tasks, cfg):
    Z, _, _ = h_sizes(tasks, cfg)
    return sorted(range(len(tasks)), key=lambda t: (-(tasks[t].w / Z[t]), t))


def sim_hybrid(tasks: Sequence[HTask], lag, cfg: HCfg, order=None, kicks: bool = True, M_admit=None):
    """Exact fluid simulation of policy P (Section 4.2 of the doc) in the hybrid model.

    Admission: the admitted set is the longest priority prefix of the available nodes whose reservations l+ fit in
    M' (head-of-line rule).  A node falling out of the prefix while holding KV is kicked: its KV is lost and it
    must recompute prompt + decoded tokens.  kicks=False (a mutation): admitted nodes are never removed, the
    prefix is extended greedily only.  Prefill server: the highest-priority admitted prefill-phase node, preemptive
    (a paused partial prefill keeps its KV).  Decode machines: the top m admitted decode-phase nodes.  Held KV (a
    child's prefix: parent context and preloaded pieces) occupies only memory not reserved by admitted nodes and is
    evicted on demand (lowest class).  Preload chunks run on the server only when no admitted node is prefilling.
    Returns dict with task completion times, recompute tokens, peak memory, kicks, held hits."""
    import random as _r
    rng = _r.Random(cfg.seed)
    if order is None:
        order = h_order(tasks, cfg)
    rank = {t: i for i, t in enumerate(order)}
    Mp = cfg.Mp if M_admit is None else M_admit
    nodes = [(t, v) for t in range(len(tasks)) for v in range(len(tasks[t].nodes))]
    nd = {(t, v): tasks[t].nodes[v] for (t, v) in nodes}
    key = lambda n: (rank[n[0]], n[1])
    avail_at, piece_at, C = {}, {}, {}
    pre_rem = {n: F(nd[n].q + nd[n].rho) for n in nodes}
    pre_done = {n: F(0) for n in nodes}
    dec_rem = {n: F(nd[n].X) for n in nodes}
    first_adm = set()
    admitted: list = []
    held: dict = {}  # child node -> held prefix tokens (F)
    held_t: dict = {}
    stats = dict(recompute=F(0), kicks=0, held_hits=F(0), peak=F(0), preload_tokens=F(0))
    now = F(0)

    for t, T in enumerate(tasks):
        for v in range(len(T.nodes)):
            if not T.preds(v):
                avail_at[(t, v)] = T.r

    def on_complete(n):
        t, v = n
        T = tasks[t]
        for k, (u, x, _) in enumerate(T.arcs):
            if u == v:
                ch = (t, x)
                if nd[ch].pieces:
                    for i, (_, tok) in enumerate(nd[ch].pieces):
                        piece_at[(ch, i)] = C[n] + lag[t][("p", (x, i))]
                if cfg.hold != "none":
                    held[ch] = F(nd[ch].rho)
                    held_t[ch] = C[n]
        for v2 in range(len(T.nodes)):
            ch = (t, v2)
            if ch in avail_at:
                continue
            ins = [k for k, (_, x, _) in enumerate(T.arcs) if x == v2]
            if all((t, T.arcs[k][0]) in C for k in ins):
                avail_at[ch] = max([T.r] + [C[(t, T.arcs[k][0])] + _h_arc_lag(T, lag[t], k) for k in ins])

    def kv_of(n):
        return pre_done[n] + (nd[n].X - dec_rem[n])

    def recompute_admitted():
        nonlocal admitted
        av = sorted([n for n in nodes if n not in C and n in avail_at and avail_at[n] <= now], key=key)
        if Mp is None:
            new = av
        elif kicks:
            new, used = [], 0
            for n in av:
                if used + nd[n].l_plus > Mp:
                    break
                new.append(n)
                used += nd[n].l_plus
        else:
            new = [n for n in admitted if n not in C]
            used = sum(nd[n].l_plus for n in new)
            for n in av:
                if n in new:
                    continue
                if used + nd[n].l_plus > Mp:
                    break
                new.append(n)
                used += nd[n].l_plus
            new = sorted(new, key=key)
        for n in admitted:
            if n not in C and n not in new and kv_of(n) > 0:
                stats["kicks"] += 1
                lost = kv_of(n)
                stats["recompute"] += lost
                pre_rem[n] = F(nd[n].q + nd[n].rho) + (nd[n].X - dec_rem[n])
                pre_done[n] = F(0)
        for n in new:
            if n not in first_adm:
                first_adm.add(n)
                h = held.pop(n, F(0))
                held_t.pop(n, None)
                if h:
                    stats["held_hits"] += h
                    pre_rem[n] -= h
                    pre_done[n] += h
        admitted = new
        evict_held()

    def free_mem():
        if cfg.Mp is None:
            return None
        return cfg.Mp - sum(nd[n].l_plus for n in admitted) - sum(held.values())

    def evict_held():
        if cfg.Mp is None:
            return
        while free_mem() < 0 and held:
            if cfg.hold == "lru" or cfg.hold == "fifo":
                victim = min(held, key=lambda n: (held_t[n], n))
            else:
                victim = rng.choice(sorted(held))
            held.pop(victim)
            held_t.pop(victim, None)

    def preload_target():
        # lowest-priority work: the highest-priority child whose returned-prefix exceeds its held prefix
        best = None
        for (ch, i), tr in piece_at.items():
            pass
        cands = []
        for ch in {c for (c, _) in piece_at}:
            if ch in first_adm or ch in C:
                continue
            ret = sorted([(piece_at[(ch, i)], i) for i in range(len(nd[ch].pieces)) if (ch, i) in piece_at])
            limit = F(nd[ch].rho)
            for (tr, i) in ret:
                if tr <= now:
                    limit += nd[ch].pieces[i][1]
                else:
                    break
            h = held.get(ch, F(0))
            if limit > h:
                cands.append((key(ch), ch, limit))
        if not cands:
            return None
        cands.sort()
        return cands[0][1], cands[0][2]

    recompute_admitted()
    while len(C) < len(nodes):
        recompute_admitted()
        if cfg.Mp is not None:
            stats["peak"] = max(stats["peak"], sum(nd[n].l_plus for n in admitted))
            assert sum(nd[n].l_plus for n in admitted) <= cfg.Mp
        pf = [n for n in admitted if pre_rem[n] > 0]
        server = pf[0] if pf else None
        dec = [n for n in admitted if pre_rem[n] == 0][: cfg.m]
        pl = None
        if server is None and cfg.preload:
            pl = preload_target()
        cand = []
        if server is not None:
            cand.append(now + pre_rem[server] * cfg.alpha)
        for n in dec:
            cand.append(now + dec_rem[n] * cfg.a)
        if pl is not None:
            ch, limit = pl
            h = held.get(ch, F(0))
            room = free_mem()
            grow = limit - h if room is None else min(limit - h, room)
            if grow > 0:
                cand.append(now + grow * cfg.alpha)
            else:
                pl = None
        cand += [x for x in avail_at.values() if x > now]
        cand += [x for x in piece_at.values() if x > now]
        cand += [T.r for T in tasks if T.r > now]
        assert cand, "deadlock"
        t1 = min(cand)
        dt = t1 - now
        if server is not None:
            tok = dt / cfg.alpha
            pre_rem[server] -= tok
            pre_done[server] += tok
        for n in dec:
            dec_rem[n] -= dt / cfg.a
        if pl is not None:
            ch, limit = pl
            held[ch] = held.get(ch, F(0)) + dt / cfg.alpha
            held_t[ch] = t1
            stats["preload_tokens"] += dt / cfg.alpha
        now = t1
        for n in list(dec):
            if dec_rem[n] == 0:
                C[n] = now
                admitted.remove(n)
                on_complete(n)
    TC = [max(C[(t, v)] for v in range(len(T.nodes))) for t, T in enumerate(tasks)]
    stats["C"] = TC
    return stats


def h_bound(tasks, lag, cfg: HCfg, order=None) -> list[F]:
    """Theorem D pathwise bound: C_J <= r_J + CP'_J + sum_{K ranked <= J} Z'_K."""
    if order is None:
        order = h_order(tasks, cfg)
    Z, _, _ = h_sizes(tasks, cfg)
    pos = {t: i for i, t in enumerate(order)}
    return [T.r + h_cp(T, lag[t], cfg, "alg") + sum(Z[k] for k in range(len(tasks)) if pos[k] <= pos[t])
            for t, T in enumerate(tasks)]


def h_lower_bounds(tasks, cfg: HCfg):
    """(LB_cp, S(sigma + A/M)) for the engine optimum at speed 1 with memory M = M' - L_max (expected over lags).
    OPT >= LB_cp and OPT >= S(sigma + A/M)/2 (concurrent open shop argument)."""
    lbcp = F(0)
    for prob, lag in h_realizations(tasks):
        lbcp += prob * sum(T.w * (T.r + h_cp(T, lag[t], cfg, "opt")) for t, T in enumerate(tasks))
    _, S, A = h_sizes(tasks, cfg)
    return lbcp, smith([T.w for T in tasks], [s + a for s, a in zip(S, A)])


def h_constants(tasks, cfg: HCfg):
    """beta_cp, beta, theta of Theorem D."""
    Lmax = h_Lmax(tasks)
    bcp = max(cfg.alpha * nd.rho / (cfg.alpha * nd.q + cfg.a * nd.X) for T in tasks for nd in T.nodes)
    kap = (lambda nd: cfg.alpha * (nd.l_plus + Lmax)) if cfg.Mp is not None else (lambda nd: F(0))
    b = max((cfg.alpha * nd.rho + kap(nd)) / (cfg.alpha * nd.q + cfg.c * nd.X) for T in tasks for nd in T.nodes)
    th = max(F(nd.l_plus, nd.l_minus) for T in tasks for nd in T.nodes) if cfg.Mp is not None else F(1)
    return bcp, b, th


def sim_discrete(tasks: Sequence[HTask], lag, cfg: HCfg, s: F = F(3), Pbar: int | None = None, order=None):
    """Atomic-step engine running policy P: at each step boundary the admitted prefix is recomputed (kicks as in
    sim_hybrid), the highest-priority admitted prefill-phase node receives min(Pbar, remaining) prefill tokens,
    the top m admitted decode-phase nodes decode one token each, and the step lasts (a + c B + alpha P)/s.
    Events inside a step are seen at the next boundary. hold='none' only (evict-always).
    Returns dict(C=task completion times, completions=#prefill completions per node, kicks)."""
    assert cfg.hold == "none" and not cfg.preload
    if Pbar is None:
        Pbar = int(cfg.a / cfg.alpha)
    if order is None:
        order = h_order(tasks, cfg)
    rank = {t: i for i, t in enumerate(order)}
    nodes = [(t, v) for t in range(len(tasks)) for v in range(len(tasks[t].nodes))]
    nd = {(t, v): tasks[t].nodes[v] for (t, v) in nodes}
    key = lambda n: (rank[n[0]], n[1])
    avail_at, C = {}, {}
    pre_rem = {n: nd[n].q + nd[n].rho for n in nodes}
    pre_done = {n: 0 for n in nodes}
    dec_rem = {n: nd[n].X for n in nodes}
    compl = {n: 0 for n in nodes}
    admitted: list = []
    kicks = 0
    for t, T in enumerate(tasks):
        for v in range(len(T.nodes)):
            if not T.preds(v):
                avail_at[(t, v)] = T.r
    now = F(0)

    def on_complete(n):
        t, v = n
        T = tasks[t]
        for v2 in range(len(T.nodes)):
            ch = (t, v2)
            if ch in avail_at:
                continue
            ins = [k for k, (_, x, _) in enumerate(T.arcs) if x == v2]
            if all((t, T.arcs[k][0]) in C for k in ins):
                avail_at[ch] = max([T.r] + [C[(t, T.arcs[k][0])] + _h_arc_lag(T, lag[t], k) for k in ins])

    while len(C) < len(nodes):
        av = sorted([n for n in nodes if n not in C and n in avail_at and avail_at[n] <= now], key=key)
        new, used = [], 0
        for n in av:
            if cfg.Mp is not None and used + nd[n].l_plus > cfg.Mp:
                break
            new.append(n)
            used += nd[n].l_plus
        for n in admitted:
            if n not in new and n not in C and pre_done[n] + (nd[n].X - dec_rem[n]) > 0:
                kicks += 1
                pre_rem[n] = nd[n].q + nd[n].rho + (nd[n].X - dec_rem[n])
                pre_done[n] = 0
        admitted = new
        pf = [n for n in admitted if pre_rem[n] > 0]
        dec = [n for n in admitted if pre_rem[n] == 0][: cfg.m]
        if not pf and not dec:
            fut = [x for x in avail_at.values() if x > now] + [T.r for T in tasks if T.r > now]
            now = min(fut)
            continue
        P = min(Pbar, pre_rem[pf[0]]) if pf else 0
        tau = (cfg.a + cfg.c * len(dec) + cfg.alpha * P) / s
        now += tau
        if pf:
            n = pf[0]
            pre_rem[n] -= P
            pre_done[n] += P
            if pre_rem[n] == 0:
                compl[n] += 1
        for n in dec:
            dec_rem[n] -= 1
            if dec_rem[n] == 0:
                C[n] = now
                admitted.remove(n)
                on_complete(n)
    return dict(C=[max(C[(t, v)] for v in range(len(T.nodes))) for t, T in enumerate(tasks)], completions=compl,
                kicks=kicks)


def disc_bound(tasks, lag, cfg: HCfg, completions, order=None) -> list[F]:
    """Discrete-step bound (s = 3, Pbar = a/alpha): C_J <= hybrid bound + a (|nodes of J| + N_J), N_J = number of
    prefill completions of nodes of tasks ranked <= J."""
    if order is None:
        order = h_order(tasks, cfg)
    pos = {t: i for i, t in enumerate(order)}
    hb = h_bound(tasks, lag, cfg, order)
    out = []
    for t, T in enumerate(tasks):
        N = sum(completions[(k, v)] for k in range(len(tasks)) if pos[k] <= pos[t] for v in range(len(tasks[k].nodes)))
        out.append(hb[t] + cfg.a * (len(T.nodes) + N))
    return out


# ============================================================================================ 4. keep / evict
def ttl_cost(D: Dist, h: F, R: F, tau) -> F:
    """Expected cost of TTL tau for one held prefix: holding rate h (= price * size) until min(D, tau), plus
    recompute R if D > tau. tau=None means hold until use."""
    if tau is None:
        return h * mean(D)
    return sum(p * (h * min(v, tau) + (R if v > tau else 0)) for v, p in D)


def ttl_candidates(D: Dist):
    return [F(0)] + [v for v, _ in D]


def ttl_best(D: Dist, h: F, R: F):
    """Optimal non-anticipatory keep/evict rule for one prefix = best deterministic TTL (Prop R5a). Returns
    (tau*, cost). Candidates 0 and the support points suffice (the cost is non-decreasing between atoms)."""
    best = min(ttl_candidates(D), key=lambda t: (ttl_cost(D, h, R, t), t))
    return best, ttl_cost(D, h, R, best)


def ttl_clair(D: Dist, h: F, R: F) -> F:
    return sum(p * min(h * v, R) for v, p in D)


def ttl_grid_cost(D: Dist, h: F, R: F, grid: Sequence[F]) -> F:
    return min(ttl_cost(D, h, R, t) for t in grid)


def ski_worst_lp(n: int, B: F = F(1)):
    """Yao/LP: a distribution on the grid {B i/n : i = 1..n} (plus a mass at 2B, i.e. 'longer than B') that
    maximizes min over TTLs of E[TTL cost] / E[clairvoyant], h = 1, R = B. Solved exactly with Z3."""
    pts = [B * i / n for i in range(1, n + 1)] + [2 * B]
    opt = z3.Optimize()
    pv = [z3.Real(f"p{i}") for i in range(len(pts))]
    z = z3.Real("z")
    for x in pv:
        opt.add(x >= 0)
    # Charnes-Cooper: unnormalized masses with E[clair] = 1, so min_tau E[cost(tau)] is the ratio
    opt.add(z3.Sum([pv[i] * z3.RealVal(min(pts[i], B)) for i in range(len(pts))]) == 1)
    for tau in [F(0)] + pts:
        cost = z3.Sum([pv[i] * z3.RealVal(min(pts[i], tau) + (B if pts[i] > tau else 0)) for i in range(len(pts))])
        opt.add(cost >= z)
    opt.maximize(z)
    assert opt.check() == z3.sat
    mdl = opt.model()
    mass = [_zfrac(mdl.eval(pv[i])) for i in range(len(pts))]
    return _zfrac(mdl.eval(z)), dist(*[(pts[i], mass[i]) for i in range(len(pts)) if mass[i] > 0])


@dataclass(frozen=True)
class KeepClass:
    lam: F  # arrival rate of this preload opportunity
    P: F  # preload GPU work (s)
    L: F  # KV size (tokens)
    D: Dist  # remaining wait until use


def keep_lp_z3(classes: Sequence[KeepClass], nu_cp: F, nu_ld: F, Mh: F):
    """LP of R5(b): choose per class a mixture over {no preload} and {preload, TTL tau}. Linearized latency cost
    of (preload, tau): lam P [nu_ld (1 - sigma(tau)) - nu_cp sigma(tau)], sigma(tau) = P(D <= tau).
    Memory (Little): sum lam L E[min(D, tau)] <= Mh. Returns (value, solution dict)."""
    opt = z3.Optimize()
    obj, mem, sol = [], [], {}
    for i, k in enumerate(classes):
        none = z3.Real(f"n{i}")
        opt.add(none >= 0)
        row = [none]
        for tau in ttl_candidates(k.D):
            z_ = z3.Real(f"z{i}_{tau}")
            opt.add(z_ >= 0)
            row.append(z_)
            sig = sum(p for v, p in k.D if v <= tau)
            obj.append(z_ * z3.RealVal(k.lam * k.P * (nu_ld * (1 - sig) - nu_cp * sig)))
            mem.append(z_ * z3.RealVal(k.lam * k.L * sum(p * min(v, tau) for v, p in k.D)))
            sol[(i, tau)] = z_
        sol[(i, None)] = none
        opt.add(z3.Sum(row) == 1)
    opt.add(z3.Sum(mem) <= z3.RealVal(Mh))
    o = z3.Sum(obj)
    opt.minimize(o)
    assert opt.check() == z3.sat
    mdl = opt.model()
    return _zfrac(mdl.eval(o)), {kk: _zfrac(mdl.eval(v)) for kk, v in sol.items()}


def keep_lines(k: KeepClass, nu_cp, nu_ld):
    """Per-class Lagrangian lines: for each TTL, (intercept, slope) of lam[P(nu_ld - (nu_ld+nu_cp) sigma) +
    eta L E min(D, tau)] as a function of the memory price eta."""
    out = []
    for tau in ttl_candidates(k.D):
        sig = sum(p for v, p in k.D if v <= tau)
        out.append((tau, k.lam * k.P * (nu_ld - (nu_ld + nu_cp) * sig), k.lam * k.L * sum(p * min(v, tau) for v, p in k.D)))
    return out


def keep_dual(classes, nu_cp, nu_ld, Mh):
    """Lagrangian dual max_{eta >= 0} -eta Mh + sum_i min(0, min_tau line_i,tau(eta)); exact (breakpoints)."""
    lines = [keep_lines(k, nu_cp, nu_ld) for k in classes]
    cands = {F(0)}
    for L in lines:
        L2 = [(F(0), F(0))] + [(b, s) for _, b, s in L]
        for (b1, s1), (b2, s2) in itertools.combinations(L2, 2):
            if s1 != s2:
                e = (b2 - b1) / (s1 - s2)
                if e > 0:
                    cands.add(e)

    def g(eta):
        return -eta * Mh + sum(min([F(0)] + [b + eta * s for _, b, s in L]) for L in lines)

    eta = max(cands, key=lambda e: (g(e), -e))
    return g(eta), eta


def sigma_star_prices(u: F, rho_s: F):
    """nu_cp = 1/(1-u) (critical-path price), nu_ld = rho_s/(1-u)^2 (load price) and sigma* = nu_ld/(nu_ld+nu_cp)."""
    nu_cp = 1 / (1 - u)
    nu_ld = rho_s / (1 - u) ** 2
    return nu_cp, nu_ld, nu_ld / (nu_ld + nu_cp)


# ============================================================================================ 5. constructions
def lag_trap(L: int, eps: int = 1):
    """Theorem A lower-bound instance for P-WSPT-DAG (m = 1, deterministic sizes, integer data): task J = node of
    work eps -> lag L -> node of work L, weight 1; task K = one node of work L released at 0 with index just above
    J's (w_K = (L+1)/(L+eps)). The policy runs K first and J's lag starts late; the optimum starts J's tiny node
    first and hides K inside the lag. ALG/OPT -> 4/3 as L/eps -> infinity."""
    J = MTask(F(0), F(1), (det(eps), det(L)), ((0, 1, det(L)),))
    K = MTask(F(0), F(L + 1, L + eps), (det(L),))
    return [J, K]


def hold_only_lower(n: int, D: F, p: F = F(1)) -> F:
    """Any hold-only policy with memory for ONE task context: task i (plan p -> lag D -> final p) is in flight
    alone, so the i-th completion is >= i (2p + D)."""
    return sum(i * (2 * p + D) for i in range(1, n + 1))


def index_vs_pmtn_ratio(n: int, q: F) -> F:
    """n i.i.d. single-node jobs, P = 0 w.p. 1-q, L w.p. q, one machine. Any fixed-priority (attained-service
    oblivious) policy gives q L n(n+1)/2; probing (preemptive) gives L E[K(K+1)/2], K ~ Bin(n, q).
    Ratio (n+1)/(n q + 2 - q) -> 1/q = 1 + Delta."""
    return F(n + 1) / (n * q + 2 - q)
