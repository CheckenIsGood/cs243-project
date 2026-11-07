"""Independent verification of docs/stochastic_theory.md (commit e17bd7f).

Everything here is written from scratch: own instance types, own simulators, own exact optimal policies (DPs over
information states), own lower bounds. Nothing is imported from `stochastic_theory.py` except in the explicit
cross-check helpers at the end (marked CROSS-CHECK), which compare the author's numbers with ours.

All decisions use exact rational arithmetic (fractions.Fraction). Z3 is used for algebra and LPs.

Sections
  A. machine model M_m: P-WSPT-DAG simulator, Lemma A1, LB_cp, Smith, exact OPT DPs (clairvoyant / preemptive
     non-anticipatory / non-preemptive non-anticipatory), MSU load inequality on arbitrary non-preemptive
     non-anticipatory policies, LB2, Thm B, Thm B' LP.
  B. hybrid engine model H: own simulator of policy P (priority-prefix admission, kicks, evict-always, optional
     hold + preload), Lemma D0 / D1 checks, the preload counterexample to (L1), R6 DAG counterexample.
  C. keep/evict layer: TTL optimality, ski-rental constants, single-constraint keep LP solved by vertex
     enumeration (independent of Z3), sigma*.
  D. R7 constructions.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache

import z3

# ===================================================================================================== utilities


def law(*pairs):
    """Finite law: tuple of (value, prob) with Fraction entries, merged and sorted."""
    acc = {}
    for v, p in pairs:
        v, p = Q(v), Q(p)
        if p:
            acc[v] = acc.get(v, Q(0)) + p
    tot = sum(acc.values())
    return tuple(sorted((v, p / tot) for v, p in acc.items()))


def point(v):
    return ((Q(v), Q(1)),)


def E(d):
    return sum(v * p for v, p in d)


def V(d):
    mu = E(d)
    return sum(p * (v - mu) ** 2 for v, p in d)


def scv(d):
    mu = E(d)
    return V(d) / mu**2


# ============================================================================================ A. machine model


@dataclass(frozen=True)
class Job:
    """A task: release r, weight w, node size laws (topological order), arcs (u, v, lag law) with u < v."""

    r: Q
    w: Q
    sizes: tuple
    arcs: tuple = ()


def joint(tasks):
    """All joint realizations (prob, sizes[t][v], lags[t][k]) of independent sizes and lags."""
    slots = []
    for t, J in enumerate(tasks):
        for v, d in enumerate(J.sizes):
            slots.append((0, t, v, d))
        for k, a in enumerate(J.arcs):
            slots.append((1, t, k, a[2]))
    for combo in itertools.product(*[s[3] for s in slots]):
        pr = Q(1)
        P = [[None] * len(J.sizes) for J in tasks]
        D = [[None] * len(J.arcs) for J in tasks]
        for s, (val, p) in zip(slots, combo):
            pr *= p
            (P if s[0] == 0 else D)[s[1]][s[2]] = val
        yield pr, P, D


def cp_len(J, p, d):
    """Longest path (sizes + lags) of one realization, release-relative."""
    fin = []
    for v in range(len(J.sizes)):
        st = Q(0)
        for k, (u, x, _) in enumerate(J.arcs):
            if x == v:
                st = max(st, fin[u] + d[k])
        fin.append(st + p[v])
    return max(fin)


def wspt_rank(tasks, size_of=None):
    """Rank by decreasing w / size (size = E[W] by default); ties by id. Returns list of task ids."""
    sz = size_of or (lambda t: sum(E(d) for d in tasks[t].sizes))
    return sorted(range(len(tasks)), key=lambda t: (-tasks[t].w / sz(t), t))


def run_priority(tasks, P, D, m, rank, preemptive=True):
    """Exact event-driven simulation of a fixed-priority list policy on m unit-speed machines.

    Node priority = (position of its task in `rank`, node index). preemptive: at every instant the m highest-priority
    available nodes run (migration allowed). non-preemptive: Graham list scheduling (a free machine takes the
    highest-priority available unstarted node; started nodes run to completion).
    Returns (task completion times, node completion dict, node availability dict)."""
    pos = {t: i for i, t in enumerate(rank)}
    left = {}
    for t, J in enumerate(tasks):
        for v in range(len(J.sizes)):
            left[(t, v)] = Q(P[t][v])
    done, av = {}, {}
    started = set()

    def refresh():
        for t, J in enumerate(tasks):
            for v in range(len(J.sizes)):
                if (t, v) in av:
                    continue
                ins = [(u, k) for k, (u, x, _) in enumerate(J.arcs) if x == v]
                if all((t, u) in done for u, _ in ins):
                    av[(t, v)] = max([J.r] + [done[(t, u)] + D[t][k] for u, k in ins])

    refresh()
    now = Q(0)
    total = len(left)
    while len(done) < total:
        ready = sorted([n for n in left if n not in done and n in av and av[n] <= now],
                       key=lambda n: (pos[n[0]], n[1]))
        zero = [n for n in ready if left[n] == 0]
        if zero:
            for n in zero:
                done[n] = now
            refresh()
            continue
        if preemptive:
            run = ready[:m]
        else:
            run = [n for n in ready if n in started]
            for n in ready:
                if len(run) >= m:
                    break
                if n not in started:
                    started.add(n)
                    run.append(n)
        nxt = [now + left[n] for n in run] + [av[n] for n in left if n not in done and n in av and av[n] > now]
        t1 = min(nxt)
        for n in run:
            left[n] -= t1 - now
            if left[n] == 0:
                done[n] = t1
        now = t1
        refresh()
    C = [max(done[(t, v)] for v in range(len(J.sizes))) for t, J in enumerate(tasks)]
    return C, done, av


def lemmaA1_rhs(tasks, P, D, m, rank):
    pos = {t: i for i, t in enumerate(rank)}
    out = []
    for t, J in enumerate(tasks):
        W = sum(sum(P[k]) for k in range(len(tasks)) if pos[k] <= pos[t])
        out.append(J.r + cp_len(J, P[t], D[t]) + Q(W) / m)
    return out


def exp_cost(tasks, m, rank, preemptive=True):
    tot = Q(0)
    for pr, P, D in joint(tasks):
        C, _, _ = run_priority(tasks, P, D, m, rank, preemptive)
        tot += pr * sum(J.w * c for J, c in zip(tasks, C))
    return tot


def LBcp(tasks):
    tot = Q(0)
    for J in tasks:
        e = sum(pr * cp_len(J, P[0], D[0]) for pr, P, D in joint([J]))
        tot += J.w * (J.r + e)
    return tot


def smith_value(ws, ps):
    """min over orders of sum_k w_k * (prefix sum of p): ratio order (exchange argument)."""
    idx = sorted(range(len(ws)), key=lambda i: (-ws[i] / ps[i], i))
    s = tot = Q(0)
    for i in idx:
        s += ps[i]
        tot += ws[i] * s
    return tot


def smith_bruteforce(ws, ps):
    best = None
    for perm in itertools.permutations(range(len(ws))):
        s = tot = Q(0)
        for i in perm:
            s += ps[i]
            tot += ws[i] * s
        best = tot if best is None or tot < best else best
    return best


# ---------------------------------------------------------------------------- exact optimal policies (tiny)
def _ints(tasks, scale=1):
    """Scale all data by an integer factor (so a 1/scale time grid becomes the unit grid)."""
    out = []
    for J in tasks:
        out.append(Job(J.r * scale, J.w, tuple(tuple((v * scale, p) for v, p in d) for d in J.sizes),
                       tuple((u, x, tuple((v * scale, p) for v, p in d)) for (u, x, d) in J.arcs)))
    for J in out:
        assert J.r.denominator == 1
        for d in J.sizes:
            assert all(v.denominator == 1 and v >= 1 for v, _ in d)
        for a in J.arcs:
            assert all(v.denominator == 1 and v >= 0 for v, _ in a[2])
    return out


def _haz(d, e):
    """P(X = e | X >= e) for an integer law."""
    tail = sum(p for v, p in d if v >= e)
    if tail == 0:
        return Q(1)
    return sum(p for v, p in d if v == e) / tail


def opt_det_1m(tasks_det):
    """Clairvoyant optimum of ONE deterministic realization on one machine, preemption at integer times (exact
    for integer data under the integrality assumption checked by `grid_refinement_gap`). DP over (t, remaining,
    availability times)."""
    tasks = tasks_det
    nodes = [(t, v) for t, J in enumerate(tasks) for v in range(len(J.sizes))]
    p0 = tuple(int(tasks[t].sizes[v][0][0]) for (t, v) in nodes)
    rmax = int(max(J.r for J in tasks))
    idx = {n: i for i, n in enumerate(nodes)}

    def avail_times(rem, fin):
        out = {}
        for (t, v) in nodes:
            J = tasks[t]
            ins = [(u, a[2][0][0]) for (u, x, a) in [(a[0], a[1], a) for a in J.arcs] if x == v]
            if all(rem[idx[(t, u)]] == 0 for u, _ in ins):
                out[(t, v)] = max([J.r] + [fin[idx[(t, u)]] + lg for u, lg in ins])
        return out

    @lru_cache(maxsize=None)
    def best(t, rem, fin):
        if all(r == 0 for r in rem):
            return Q(0)
        av = avail_times(rem, fin)
        unit = sum(J.w for k, J in enumerate(tasks) if any(rem[idx[(k, v)]] for v in range(len(J.sizes))))
        cands = [n for n in nodes if rem[idx[n]] > 0 and n in av and av[n] <= t]
        res = None
        options = cands if cands else [None]
        for n in options:
            r2, f2 = list(rem), list(fin)
            if n is not None:
                r2[idx[n]] -= 1
                if r2[idx[n]] == 0:
                    f2[idx[n]] = t + 1
            val = unit + best(t + 1, tuple(r2), tuple(f2))
            res = val if res is None or val < res else res
        return res

    return best(0, p0, tuple([0] * len(nodes)))


def opt_clair_exp(tasks, scale=1):
    """E over realizations of the clairvoyant one-machine optimum (preemptive), on the 1/scale grid."""
    tot = Q(0)
    for pr, P, D in joint(tasks):
        det_tasks = _ints([Job(J.r, J.w, tuple(point(P[t][v]) for v in range(len(J.sizes))),
                               tuple((u, x, point(D[t][k])) for k, (u, x, _) in enumerate(J.arcs)))
                           for t, J in enumerate(tasks)], scale)
        tot += pr * opt_det_1m(det_tasks) / scale
    return tot


def _lag_resolve(tasks, pend, expired):
    """pend: frozenset of ((t,k), elapsed). Branch over which pending lags expire now."""
    outs = [(Q(1), [], set(expired))]
    for (tk, e) in sorted(pend):
        h = _haz(tasks[tk[0]].arcs[tk[1]][2], e)
        new = []
        for pr, keep, ex in outs:
            if h:
                new.append((pr * h, keep, ex | {tk}))
            if h != 1:
                new.append((pr * (1 - h), keep + [(tk, e)], ex))
        outs = new
    return [(pr, frozenset(k), frozenset(ex)) for pr, k, ex in outs]


def _ready(tasks, t, done, busy, expired):
    out = []
    for k, J in enumerate(tasks):
        if t < J.r:
            continue
        for v in range(len(J.sizes)):
            n = (k, v)
            if n in done or n in busy:
                continue
            if all((k, a) in expired for a, (u, x, _) in enumerate(J.arcs) if x == v):
                out.append(n)
    return out


def opt_na_pmtn_1m(tasks, scale=1):
    """Optimal NON-ANTICIPATORY PREEMPTIVE policy on one machine, deterministic or random integer sizes, random
    integer lags, decisions on the 1/scale grid. Information state: done nodes, attained service of started nodes,
    pending lags with elapsed time, expired lags."""
    tasks = tuple(_ints(tasks, scale))
    rmax = max(J.r for J in tasks)
    alln = frozenset((k, v) for k, J in enumerate(tasks) for v in range(len(J.sizes)))

    @lru_cache(maxsize=None)
    def val(t, done, att, pend, expired):
        if done == alln:
            return Q(0)
        attd = dict(att)
        unit = sum(J.w for k, J in enumerate(tasks) if any((k, v) not in done for v in range(len(J.sizes))))
        rd = _ready(tasks, t, done, set(), expired)
        best = None
        for n in rd + [None]:
            if n is None and not pend and t >= rmax and rd:
                continue  # idling with nothing to wait for is dominated
            if n is None and not pend and t >= rmax and not rd:
                raise AssertionError("deadlock")
            outs = []
            if n is None:
                outs = [(Q(1), done, attd)]
            else:
                e = attd.get(n, 0)
                h = _haz(tasks[n[0]].sizes[n[1]], e + 1)
                if h:
                    a2 = dict(attd)
                    a2.pop(n, None)
                    outs.append((h, done | {n}, a2))
                if h != 1:
                    a2 = dict(attd)
                    a2[n] = e + 1
                    outs.append((1 - h, done, a2))
            tot = Q(unit)
            for pr, dn, a2 in outs:
                newp = {(tk, e + 1) for tk, e in pend}
                for (k, v) in dn - done:
                    for a, (u, x, _) in enumerate(tasks[k].arcs):
                        if u == v:
                            newp.add(((k, a), 0))
                for pr2, p2, ex2 in _lag_resolve(tasks, frozenset(newp), expired):
                    tot += pr * pr2 * val(min(t + 1, rmax), dn, tuple(sorted(a2.items())), p2, ex2)
            best = tot if best is None or tot < best else best
        return best

    tot = Q(0)
    for pr, p2, ex2 in _lag_resolve(tasks, frozenset(), frozenset()):
        tot += pr * val(Q(0), frozenset(), (), p2, ex2)
    return tot / scale


def opt_np(tasks, m, scale=1):
    """Optimal NON-ANTICIPATORY NON-PREEMPTIVE policy on m machines with idling, decisions on the 1/scale grid
    (random integer sizes >= 1, random integer lags >= 0, integer releases)."""
    tasks = tuple(_ints(tasks, scale))
    rmax = max(J.r for J in tasks)
    alln = frozenset((k, v) for k, J in enumerate(tasks) for v in range(len(J.sizes)))

    @lru_cache(maxsize=None)
    def val(t, done, running, pend, expired):
        if done == alln:
            return Q(0)
        busy = {n for n, _ in running}
        rd = _ready(tasks, t, done, busy, expired)
        unit = sum(J.w for k, J in enumerate(tasks) if any((k, v) not in done for v in range(len(J.sizes))))
        best = None
        free = m - len(running)
        for kk in range(min(free, len(rd)) + 1):
            for S in itertools.combinations(rd, kk):
                run = list(running) + [(n, 0) for n in S]
                if not run and not pend and t >= rmax:
                    continue
                # one time unit
                outs = [(Q(1), set(done), [])]
                for (n, e) in run:
                    h = _haz(tasks[n[0]].sizes[n[1]], e + 1)
                    new = []
                    for pr, dn, rr in outs:
                        if h:
                            new.append((pr * h, dn | {n}, rr))
                        if h != 1:
                            new.append((pr * (1 - h), dn, rr + [(n, e + 1)]))
                    outs = new
                tot = Q(unit)
                for pr, dn, rr in outs:
                    newp = {(tk, e + 1) for tk, e in pend}
                    for (k, v) in dn - done:
                        for a, (u, x, _) in enumerate(tasks[k].arcs):
                            if u == v:
                                newp.add(((k, a), 0))
                    for pr2, p2, ex2 in _lag_resolve(tasks, frozenset(newp), expired):
                        tot += pr * pr2 * val(min(t + 1, rmax), frozenset(dn), tuple(sorted(rr)), p2, ex2)
                best = tot if best is None or tot < best else best
        return best

    tot = Q(0)
    for pr, p2, ex2 in _lag_resolve(tasks, frozenset(), frozenset()):
        tot += pr * val(Q(0), frozenset(), (), p2, ex2)
    return tot / scale


# ---------------------------------------------------------------- MSU inequality on arbitrary np policies
def run_np_policy(tasks, P, D, m, policy):
    """Simulate a non-preemptive NON-ANTICIPATORY policy given as a callback.

    policy(now, ready_nodes, free_machines, info) -> list of nodes to start now (subset of ready, len <= free);
    `info` contains only realized past data (completed nodes with their completion times and sizes, expired lags).
    The policy may also return a wake-up time (deliberate idling) via info['wake'].
    Sizes are revealed only at completion. Returns node completion dict and start dict."""
    done, start, av = {}, {}, {}
    running = {}
    now = Q(0)
    nodes = [(t, v) for t, J in enumerate(tasks) for v in range(len(J.sizes))]

    def refresh():
        for t, J in enumerate(tasks):
            for v in range(len(J.sizes)):
                if (t, v) in av:
                    continue
                ins = [(u, k) for k, (u, x, _) in enumerate(J.arcs) if x == v]
                if all((t, u) in done for u, _ in ins):
                    av[(t, v)] = max([J.r] + [done[(t, u)] + D[t][k] for u, k in ins])

    refresh()
    wake = []
    while len(done) < len(nodes):
        ready = [n for n in nodes if n not in done and n not in start and n in av and av[n] <= now]
        info = dict(now=now, done=dict(done), sizes={n: P[n[0]][n[1]] for n in done}, wake=wake)
        chosen = policy(now, ready, m - len(running), info) if ready and len(running) < m else []
        for n in chosen:
            assert n in ready and len(running) < m
            start[n] = now
            running[n] = now + P[n[0]][n[1]]
        evts = list(running.values()) + [av[n] for n in nodes if n in av and av[n] > now and n not in done] + \
            [x for x in wake if x > now]
        t1 = min(evts)
        now = t1
        for n in [n for n, c in running.items() if c == now]:
            done[n] = now
            del running[n]
        refresh()
    return done, start


def random_np_policy(rng, tasks):
    """A random non-anticipatory non-preemptive policy: random static priorities, random per-node earliest start
    thresholds (deliberate idling), and a coin that flips the priority of the two top candidates depending on
    the realized history (an adaptive, but non-anticipatory, rule)."""
    nodes = [(t, v) for t, J in enumerate(tasks) for v in range(len(J.sizes))]
    pr = {n: rng.random() for n in nodes}
    thr = {n: Q(rng.choice([0, 0, 0, 1, 2, 3])) for n in nodes}
    adapt = rng.random() < 0.5

    def pol(now, ready, free, info):
        cand = [n for n in ready if now >= thr[n]]
        for n in ready:
            if now < thr[n]:
                info["wake"].append(thr[n])
        cand.sort(key=lambda n: pr[n])
        if adapt and len(cand) >= 2 and (sum(info["sizes"].values()) % 2 == 1):
            cand[0], cand[1] = cand[1], cand[0]
        return cand[:free]

    return pol


def msu_f(mus, m, Delta):
    s = sum(mus)
    s2 = sum(x * x for x in mus)
    return (s * s + s2) / (2 * m) - Q(m - 1) * (Delta - 1) / (2 * m) * s2


def msu_check_policy(tasks, m, policy_maker, rng):
    """Exact E[C_x] of one random non-anticipatory non-preemptive policy (same policy object on every realization),
    then the minimum slack of the MSU inequality over all node sets A. Returns the min slack (>= 0 iff valid)."""
    nodes = [(t, v) for t, J in enumerate(tasks) for v in range(len(J.sizes))]
    seed = rng.random()
    EC = {n: Q(0) for n in nodes}
    for pr, P, D in joint(tasks):
        pol = policy_maker(random.Random(seed), tasks)
        done, _ = run_np_policy(tasks, P, D, m, pol)
        for n in nodes:
            EC[n] += pr * done[n]
    mu = {n: E(tasks[n[0]].sizes[n[1]]) for n in nodes}
    Delta = max(scv(tasks[n[0]].sizes[n[1]]) for n in nodes)
    worst = None
    for r in range(1, len(nodes) + 1):
        for A in itertools.combinations(nodes, r):
            sl = sum(mu[n] * EC[n] for n in A) - msu_f([mu[n] for n in A], m, Delta)
            worst = sl if worst is None or sl < worst else worst
    return worst


def lb2(tasks, m, Delta=None):
    """Task-level MSU polymatroid bound, greedy (Smith on w/E[W]) vertex."""
    if Delta is None:
        Delta = max(scv(d) for J in tasks for d in J.sizes)
    b = Q(1, 2) - Q(m - 1) * Delta / (2 * m)
    g = min(Q(0), b)
    EW = [sum(E(d) for d in J.sizes) for J in tasks]
    s = tot = Q(0)
    for t in wspt_rank(tasks):
        tot += tasks[t].w * (s / m + (Q(1, 2 * m) + g) * EW[t])
        s += EW[t]
    return tot, g


def lb2_dual_certificate(tasks, m, Delta=None):
    """Certificate that the greedy vertex is LP-optimal: the greedy point is primal feasible for all 2^n task-set
    constraints (checked by enumeration) and the chain dual y_{S_k} = c_k - c_{k+1} >= 0 (c = w/E[W] in greedy
    order) has the same objective. Returns (primal_value, dual_value, primal_feasible)."""
    if Delta is None:
        Delta = max(scv(d) for J in tasks for d in J.sizes)
    b = Q(1, 2) - Q(m - 1) * Delta / (2 * m)
    g = min(Q(0), b)
    EW = [sum(E(d) for d in J.sizes) for J in tasks]

    def gfun(S):
        s = sum((EW[k] for k in S), Q(0))
        return s * s / (2 * m) + g * sum((EW[k] ** 2 for k in S), Q(0))

    order = wspt_rank(tasks)
    C = {}
    pref = []
    for t in order:
        before = gfun(pref)
        pref = pref + [t]
        C[t] = (gfun(pref) - before) / EW[t]
    primal = sum(tasks[t].w * C[t] for t in order)
    feas = all(sum(EW[k] * C[k] for k in S) >= gfun(S)
               for r in range(1, len(tasks) + 1) for S in itertools.combinations(range(len(tasks)), r))
    c = [tasks[t].w / EW[t] for t in order] + [Q(0)]
    dual = sum((c[i] - c[i + 1]) * gfun(order[: i + 1]) for i in range(len(order)))
    assert all(c[i] >= c[i + 1] for i in range(len(order)))
    return primal, dual, feas


def thmB_rho(m, Delta, chain):
    x = max(Q(1), Q(m - 1) * (Delta - 1))
    return 2 + (x / (2 * m) if chain else x / 2)


def thmBp_rho(m, Delta):
    return 2 + max(Q(1), Q(m - 1) * Delta / m)


def msu_node_lp(tasks, m, Delta=None):
    """Thm B' LP (own formulation, Z3 over the rationals). Returns (value, C_J list)."""
    nodes = [(t, v) for t, J in enumerate(tasks) for v in range(len(J.sizes))]
    mu = {n: E(tasks[n[0]].sizes[n[1]]) for n in nodes}
    if Delta is None:
        Delta = max(scv(tasks[n[0]].sizes[n[1]]) for n in nodes)
    o = z3.Optimize()
    x = {n: z3.Real("x_%d_%d" % n) for n in nodes}
    y = [z3.Real("y_%d" % t) for t in range(len(tasks))]
    for r in range(1, len(nodes) + 1):
        for A in itertools.combinations(nodes, r):
            o.add(z3.Sum([z3.RealVal(mu[n]) * x[n] for n in A]) >= z3.RealVal(msu_f([mu[n] for n in A], m, Delta)))
    for t, J in enumerate(tasks):
        ecp = sum(pr * cp_len(J, P[0], D[0]) for pr, P, D in joint([J]))
        o.add(y[t] >= z3.RealVal(J.r + ecp))
        for v in range(len(J.sizes)):
            o.add(x[(t, v)] >= z3.RealVal(J.r + mu[(t, v)]))
            o.add(y[t] >= x[(t, v)])
        for (u, v, d) in J.arcs:
            o.add(x[(t, v)] >= x[(t, u)] + z3.RealVal(E(d) + mu[(t, v)]))
    obj = z3.Sum([z3.RealVal(J.w) * y[t] for t, J in enumerate(tasks)])
    o.minimize(obj)
    assert o.check() == z3.sat
    md = o.model()

    def fr(e):
        e = z3.simplify(md.eval(e))
        return Q(e.numerator_as_long(), e.denominator_as_long())

    return fr(obj), [fr(c) for c in y]


# ----------------------------------------------------------------------------- random machine instances
def rand_law(rng, lo, hi, k):
    vals = rng.sample(range(lo, hi + 1), k)
    return law(*[(v, rng.randint(1, 3)) for v in vals])


def rand_job(rng, det_sizes=True, shape=None, pmax=3, lagmax=3, rmax=2):
    shape = shape or rng.choice(["single", "chain2", "chain3", "fork", "forkjoin", "vee"])
    sz = (lambda: point(rng.randint(1, pmax))) if det_sizes else \
        (lambda: rand_law(rng, 1, pmax, rng.choice([1, 2, 2])))
    lg = lambda: rand_law(rng, 0, lagmax, rng.choice([1, 2]))
    r, w = Q(rng.randint(0, rmax)), Q(rng.randint(1, 4))
    if shape == "single":
        return Job(r, w, (sz(),))
    if shape == "chain2":
        return Job(r, w, (sz(), sz()), ((0, 1, lg()),))
    if shape == "chain3":
        return Job(r, w, (sz(), sz(), sz()), ((0, 1, lg()), (1, 2, lg())))
    if shape == "fork":  # 0 -> 1, 0 -> 2
        return Job(r, w, (sz(), sz(), sz()), ((0, 1, lg()), (0, 2, lg())))
    if shape == "vee":  # 0 -> 2, 1 -> 2 (two sources)
        return Job(r, w, (sz(), sz(), sz()), ((0, 2, lg()), (1, 2, lg())))
    return Job(r, w, (sz(), sz(), sz(), sz()), ((0, 1, lg()), (0, 2, lg()), (1, 3, lg()), (2, 3, lg())))


def is_chain(J):
    n = len(J.sizes)
    return len(J.arcs) == n - 1 and all(u == i and v == i + 1 for i, (u, v, _) in enumerate(J.arcs))


# ======================================================================================= B. hybrid model H


@dataclass(frozen=True)
class Node:
    q: int  # new prompt tokens
    X: int  # decode tokens
    rho: int = 0  # parent-context tokens
    pieces: tuple = ()  # ((lag law, tokens), ...): fan-out; arc lag = max piece lag; tokens are part of q

    @property
    def lm(self):
        return self.rho + self.q

    @property
    def lp(self):
        return self.rho + self.q + self.X


@dataclass(frozen=True)
class Agent:
    r: Q
    w: Q
    nodes: tuple
    arcs: tuple = ()  # (u, v, lag law); ignored lag (use point(0)) for nodes with pieces


@dataclass(frozen=True)
class Eng:
    alpha: Q
    a: Q
    m: int
    Mp: int | None = None
    hold: bool = False
    preload: bool = False

    @property
    def c(self):
        return self.a / self.m


def agent_reals(tasks):
    slots = []
    for t, T in enumerate(tasks):
        for k, (u, v, d) in enumerate(T.arcs):
            slots.append((t, ("arc", k), d))
        for v, nd in enumerate(T.nodes):
            for i, (d, _) in enumerate(nd.pieces):
                slots.append((t, ("pc", v, i), d))
    for combo in itertools.product(*[s[2] for s in slots]):
        pr = Q(1)
        lag = [dict() for _ in tasks]
        for s, (val, p) in zip(slots, combo):
            pr *= p
            lag[s[0]][s[1]] = val
        yield pr, lag


def arc_lag(T, lag, k):
    u, v, _ = T.arcs[k]
    if T.nodes[v].pieces:
        return max(lag[("pc", v, i)] for i in range(len(T.nodes[v].pieces)))
    return lag[("arc", k)]


def Lmax(tasks):
    return max(nd.lp for T in tasks for nd in T.nodes)


def zprime(tasks, cfg):
    """Z'_J = sum_y alpha(q+rho) + kappa_y + c X + a X l+/M (memory terms only when M' is finite)."""
    L = Lmax(tasks)
    out = []
    for T in tasks:
        z = Q(0)
        for nd in T.nodes:
            z += cfg.alpha * (nd.q + nd.rho) + cfg.c * nd.X
            if cfg.Mp is not None:
                M = cfg.Mp - L
                z += cfg.alpha * (nd.lp + L) + cfg.a * nd.X * Q(nd.lp) / M
        out.append(z)
    return out


def rank_P(tasks, cfg):
    Z = zprime(tasks, cfg)
    return sorted(range(len(tasks)), key=lambda t: (-tasks[t].w / Z[t], t))


def cp_path(T, lag, node_len):
    fin = []
    for v, nd in enumerate(T.nodes):
        st = Q(0)
        for k, (u, x, _) in enumerate(T.arcs):
            if x == v:
                st = max(st, fin[u] + arc_lag(T, lag, k))
        fin.append(st + node_len(nd))
    return max(fin)


def lemmaD1_rhs(tasks, lag, cfg, rank):
    Z = zprime(tasks, cfg)
    pos = {t: i for i, t in enumerate(rank)}
    out = []
    for t, T in enumerate(tasks):
        cp = cp_path(T, lag[t], lambda nd: cfg.alpha * (nd.q + nd.rho) + cfg.a * nd.X)
        out.append(T.r + cp + sum(Z[k] for k in range(len(tasks)) if pos[k] <= pos[t]))
    return out


def run_P(tasks, lag, cfg, rank=None, kicks=True):
    """Own exact fluid simulation of policy P (doc Section 4.2) in the hybrid model.

    State per node: phase ('pre' with remaining prefill tokens, 'dec' with remaining decode tokens), KV tokens
    currently materialized. Admission = longest priority prefix of available unfinished nodes with sum l+ <= M'.
    A previously admitted node that leaves the prefix while holding KV is kicked (KV lost; remaining prefill =
    rho + q + decoded so far). Server = top admitted node with remaining prefill (rate 1/alpha). Decoders = top m
    admitted decode-phase nodes (rate 1/a each). If cfg.hold: when a parent finishes, each child's parent context
    (rho tokens) is held; held KV uses only unreserved memory and is evicted LRU on demand; a node's first admission
    consumes its held tokens (prefill shrinks). If cfg.preload: when no admitted node prefills, the server grows the
    held prefix of the highest-priority not-yet-admitted child whose returned prefix (rho + returned pieces in return
    order) exceeds what is held, using only unreserved memory.

    Returns dict(C, kick_events=[(time, newly_available_l+_sum, kicked_l+_sum)], max_mem, held_hits)."""
    if rank is None:
        rank = rank_P(tasks, cfg)
    pos = {t: i for i, t in enumerate(rank)}
    nodes = [(t, v) for t in range(len(tasks)) for v in range(len(tasks[t].nodes))]
    ND = {(t, v): tasks[t].nodes[v] for (t, v) in nodes}
    pr_key = lambda n: (pos[n[0]], n[1])
    av, fin, pieces_at = {}, {}, {}
    pre = {n: Q(ND[n].q + ND[n].rho) for n in nodes}
    kv_pre = {n: Q(0) for n in nodes}
    dec = {n: Q(ND[n].X) for n in nodes}
    held, held_ts = {}, {}
    seen = set()
    adm = []
    res = dict(kick_events=[], max_mem=Q(0), held_hits=Q(0), kicked_tokens=Q(0))
    now = Q(0)
    for t, T in enumerate(tasks):
        for v in range(len(T.nodes)):
            if not any(x == v for (_, x, _) in T.arcs):
                av[(t, v)] = T.r

    def finish(n):
        t, v = n
        T = tasks[t]
        for k, (u, x, _) in enumerate(T.arcs):
            if u == v:
                ch = (t, x)
                for i in range(len(ND[ch].pieces)):
                    pieces_at[(ch, i)] = fin[n] + lag[t][("pc", x, i)]
                if cfg.hold and ND[ch].rho > 0:
                    held[ch] = Q(ND[ch].rho)
                    held_ts[ch] = now
        for x in range(len(T.nodes)):
            ch = (t, x)
            if ch in av:
                continue
            ins = [k for k, (u, y, _) in enumerate(T.arcs) if y == x]
            if all((t, T.arcs[k][0]) in fin for k in ins):
                av[ch] = max([T.r] + [fin[(t, T.arcs[k][0])] + arc_lag(T, lag[t], k) for k in ins])

    def kv(n):
        return kv_pre[n] + (ND[n].X - dec[n])

    def reserved():
        return sum(ND[n].lp for n in adm)

    def evict():
        if cfg.Mp is None:
            return
        while held and reserved() + sum(held.values()) > cfg.Mp:
            vic = min(held, key=lambda n: (held_ts[n], n))
            del held[vic]
            del held_ts[vic]

    def admit():
        nonlocal adm
        ready = sorted([n for n in nodes if n not in fin and n in av and av[n] <= now], key=pr_key)
        if cfg.Mp is None:
            new = ready
        elif kicks:
            new, s = [], 0
            for n in ready:
                if s + ND[n].lp > cfg.Mp:
                    break
                new.append(n)
                s += ND[n].lp
        else:
            new = [n for n in adm if n not in fin]
            s = sum(ND[n].lp for n in new)
            for n in ready:
                if n in new:
                    continue
                if s + ND[n].lp > cfg.Mp:
                    break
                new.append(n)
                s += ND[n].lp
            new.sort(key=pr_key)
        newly = [n for n in new if n not in seen]
        kicked = [n for n in adm if n not in new and n not in fin and kv(n) > 0]
        if kicked:
            arrivals = [n for n in ready if av[n] == now]
            res["kick_events"].append((now, sum(ND[n].lp for n in arrivals), sum(ND[n].lp for n in kicked),
                                       len(arrivals)))
        for n in kicked:
            res["kicked_tokens"] += kv(n)
            pre[n] = Q(ND[n].q + ND[n].rho) + (ND[n].X - dec[n])
            kv_pre[n] = Q(0)
        for n in newly:
            seen.add(n)
            h = held.pop(n, Q(0))
            held_ts.pop(n, None)
            h = min(h, pre[n])
            if h:
                res["held_hits"] += h
                pre[n] -= h
                kv_pre[n] += h
        adm = new
        evict()
        if cfg.Mp is not None:
            assert reserved() <= cfg.Mp
            assert reserved() + sum(held.values()) <= cfg.Mp
            res["max_mem"] = max(res["max_mem"], reserved() + sum(held.values()))

    def preload_choice():
        best = None
        for ch in sorted({c for (c, _) in pieces_at}, key=pr_key):
            if ch in seen or ch in fin:
                continue
            nd = ND[ch]
            ret = sorted((pieces_at[(ch, i)], i) for i in range(len(nd.pieces)))
            lim = Q(nd.rho)  # the held prefix is the parent context followed by returned pieces
            for (tr, i) in ret:
                if tr <= now:
                    lim += nd.pieces[i][1]
                else:
                    break
            if lim > held.get(ch, Q(0)):
                best = (ch, lim)
                break
        return best

    while len(fin) < len(nodes):
        admit()
        pf = [n for n in adm if pre[n] > 0]
        srv = pf[0] if pf else None
        decs = [n for n in adm if pre[n] == 0][: cfg.m]
        pl = None
        if srv is None and cfg.preload:
            pl = preload_choice()
        ev = []
        if srv is not None:
            ev.append(now + pre[srv] * cfg.alpha)
        for n in decs:
            ev.append(now + dec[n] * cfg.a)
        if pl is not None:
            ch, lim = pl
            room = None if cfg.Mp is None else cfg.Mp - reserved() - sum(held.values())
            grow = lim - held.get(ch, Q(0))
            if room is not None:
                grow = min(grow, room)
            if grow > 0:
                ev.append(now + grow * cfg.alpha)
            else:
                pl = None
        ev += [x for x in av.values() if x > now]
        ev += [x for x in pieces_at.values() if x > now]
        ev += [T.r for T in tasks if T.r > now]
        t1 = min(ev)
        dt = t1 - now
        if srv is not None:
            pre[srv] -= dt / cfg.alpha
            kv_pre[srv] += dt / cfg.alpha
        for n in decs:
            dec[n] -= dt / cfg.a
        if pl is not None:
            ch, _ = pl
            held[ch] = held.get(ch, Q(0)) + dt / cfg.alpha
            held_ts[ch] = t1
        now = t1
        for n in decs:
            if dec[n] == 0:
                fin[n] = now
                adm.remove(n)
                finish(n)
    res["C"] = [max(fin[(t, v)] for v in range(len(T.nodes))) for t, T in enumerate(tasks)]
    return res


def engine_lower_bounds(tasks, cfg, preload_aware=False):
    """(LB_cp, S(sigma + A/M)) of the doc's (L1)-(L3) for the speed-1 engine with memory M = M' - L_max.
    preload_aware=True replaces (L1)'s node length alpha q + (a+c) X by (a+c) X for nodes with pieces (valid when
    the benchmark may prefill returned pieces before the node is available)."""
    lb = Q(0)
    for pr, lag in agent_reals(tasks):
        for t, T in enumerate(tasks):
            f = (lambda nd: (cfg.a + cfg.c) * nd.X + (0 if (preload_aware and nd.pieces) else cfg.alpha * nd.q))
            lb += pr * T.w * (T.r + cp_path(T, lag[t], f))
    L = Lmax(tasks)
    sizes = []
    for T in tasks:
        s = Q(0)
        for nd in T.nodes:
            s += cfg.alpha * nd.q + cfg.c * nd.X
            if cfg.Mp is not None:
                s += Q(nd.lm) * (cfg.a + cfg.c) * nd.X / (cfg.Mp - L)
        sizes.append(s)
    return lb, smith_value([T.w for T in tasks], sizes)


def thmD_constants(tasks, cfg):
    L = Lmax(tasks)
    kap = (lambda nd: cfg.alpha * (nd.lp + L)) if cfg.Mp is not None else (lambda nd: Q(0))
    bcp = max(cfg.alpha * nd.rho / (cfg.alpha * nd.q + cfg.a * nd.X) for T in tasks for nd in T.nodes)
    b = max((cfg.alpha * nd.rho + kap(nd)) / (cfg.alpha * nd.q + cfg.c * nd.X) for T in tasks for nd in T.nodes)
    th = max(Q(nd.lp, nd.lm) for T in tasks for nd in T.nodes) if cfg.Mp is not None else Q(1)
    return bcp, b, th


def rand_agent(rng, qmax=8, dag=False):
    plan = Node(q=rng.randint(1, qmax), X=rng.randint(1, 4))
    k = rng.randint(1, 3)
    pcs = tuple((law(*[(v, 1) for v in rng.sample(range(0, 6), rng.choice([1, 2]))]), rng.randint(1, qmax))
                for _ in range(k))
    fin = Node(q=sum(t for _, t in pcs) + rng.randint(0, 2), X=rng.randint(1, 3), rho=plan.lp, pieces=pcs)
    if dag and rng.random() < 0.5:
        side = Node(q=rng.randint(1, qmax), X=rng.randint(1, 5), rho=plan.lp)
        lg = law(*[(v, 1) for v in rng.sample(range(0, 6), rng.choice([1, 2]))])
        return Agent(Q(rng.randint(0, 4)), Q(rng.randint(1, 4)), (plan, fin, side),
                     ((0, 1, point(0)), (0, 2, lg)))
    return Agent(Q(rng.randint(0, 4)), Q(rng.randint(1, 4)), (plan, fin), ((0, 1, point(0)),))


def run_discrete(tasks, lag, cfg, s=Q(3), rank=None):
    """Own atomic-step engine (evict-always): at each boundary recompute the admitted prefix (kicks as in run_P);
    the top admitted prefill-phase node gets min(Pbar, remaining) tokens, Pbar = a/alpha; the top m admitted
    decoders get one token each; the step lasts (a + c B + alpha P)/s. Events inside a step are seen at the next
    boundary. Returns (C, prefill completions per node)."""
    if rank is None:
        rank = rank_P(tasks, cfg)
    pos = {t: i for i, t in enumerate(rank)}
    Pbar = cfg.a / cfg.alpha
    assert Pbar.denominator == 1
    Pbar = int(Pbar)
    nodes = [(t, v) for t in range(len(tasks)) for v in range(len(tasks[t].nodes))]
    ND = {(t, v): tasks[t].nodes[v] for (t, v) in nodes}
    key = lambda n: (pos[n[0]], n[1])
    av, fin = {}, {}
    pre = {n: ND[n].q + ND[n].rho for n in nodes}
    kvp = {n: 0 for n in nodes}
    dec = {n: ND[n].X for n in nodes}
    ncomp = {n: 0 for n in nodes}
    adm = []
    for t, T in enumerate(tasks):
        for v in range(len(T.nodes)):
            if not any(x == v for (_, x, _) in T.arcs):
                av[(t, v)] = T.r
    now = Q(0)
    while len(fin) < len(nodes):
        ready = sorted([n for n in nodes if n not in fin and n in av and av[n] <= now], key=key)
        new, used = [], 0
        for n in ready:
            if cfg.Mp is not None and used + ND[n].lp > cfg.Mp:
                break
            new.append(n)
            used += ND[n].lp
        for n in adm:
            if n not in new and n not in fin and kvp[n] + (ND[n].X - dec[n]) > 0:
                pre[n] = ND[n].q + ND[n].rho + (ND[n].X - dec[n])
                kvp[n] = 0
        adm = new
        pf = [n for n in adm if pre[n] > 0]
        dc = [n for n in adm if pre[n] == 0][: cfg.m]
        if not pf and not dc:
            now = min([x for x in av.values() if x > now] + [T.r for T in tasks if T.r > now])
            continue
        P = min(Pbar, pre[pf[0]]) if pf else 0
        now += (cfg.a + cfg.c * len(dc) + cfg.alpha * P) / s
        if pf:
            pre[pf[0]] -= P
            kvp[pf[0]] += P
            if pre[pf[0]] == 0:
                ncomp[pf[0]] += 1
        for n in dc:
            dec[n] -= 1
            if dec[n] == 0:
                fin[n] = now
                adm.remove(n)
                t, v = n
                T = tasks[t]
                for x in range(len(T.nodes)):
                    ch = (t, x)
                    if ch in av:
                        continue
                    ins = [k for k, (u, y, _) in enumerate(T.arcs) if y == x]
                    if all((t, T.arcs[k][0]) in fin for k in ins):
                        av[ch] = max([T.r] + [fin[(t, T.arcs[k][0])] + arc_lag(T, lag[t], k) for k in ins])
    return [max(fin[(t, v)] for v in range(len(T.nodes))) for t, T in enumerate(tasks)], ncomp


# --------------------------------------------------------------------------- preload counterexample to (L1)
def preload_L1_counterexample():
    """One task: plan (q=1, X=1) -> final with two fan-out pieces: 10 tokens returning after lag 0, 1 token
    returning after lag 10. Engine at speed 1, alpha = a = 1, m = 1 (c = 1), unbounded memory.

    An explicit non-anticipatory engine schedule that prefills the first returned piece during the remaining tool
    wait (exactly what R6's preload does for ALG) finishes at 16, while the doc's (L1) critical-path bound is 26.
    Returns (explicit schedule cost, LB_cp of (L1), preload-aware LB, schedule trace)."""
    alpha, a, c = Q(1), Q(1), Q(1)
    trace = []
    t = Q(0)
    t += alpha * 1  # plan prefill (rate 1/alpha)
    trace.append(("plan prefill", Q(0), t))
    t0 = t
    t += (a + c) * 1  # plan decode at batch 1: step a + c per token
    trace.append(("plan decode", t0, t))
    t_plan = t
    # piece A (10 tokens) returns at t_plan + 0; prefill it right away (the node is not yet available)
    t0 = t
    t += alpha * 10
    trace.append(("prefill returned piece A (10 tok) before availability", t0, t))
    assert t <= t_plan + 10  # done before piece B returns
    t = t_plan + 10  # piece B returns, node becomes available
    t0 = t
    t += alpha * 1
    trace.append(("prefill piece B (1 tok)", t0, t))
    t0 = t
    t += (a + c) * 1
    trace.append(("final decode", t0, t))
    sched_cost = t
    T = Agent(Q(0), Q(1), (Node(q=1, X=1), Node(q=11, X=1, rho=0, pieces=((point(0), 10), (point(10), 1)))),
              ((0, 1, point(0)),))
    cfg = Eng(alpha=alpha, a=a, m=1)
    lb, _ = engine_lower_bounds([T], cfg)
    lb_fix, _ = engine_lower_bounds([T], cfg, preload_aware=True)
    return sched_cost, lb, lb_fix, trace, T, cfg


def r6_dag_counterexample():
    """R6 'isolated-task saving = alpha * held hits exactly' fails for DAG tasks: plan -> a (lag 5), plan -> b
    (lag 0), both with parent context rho = 2; b has a long decode and is the last node. Evict-always vs hold:
    the hit on a is off the critical path, so the saving is alpha*rho_b, not alpha*(rho_a + rho_b)."""
    plan = Node(q=1, X=1)
    na = Node(q=1, X=1, rho=2)
    nb = Node(q=1, X=100, rho=2)
    T = Agent(Q(0), Q(1), (plan, na, nb), ((0, 1, point(5)), (0, 2, point(0))))
    lag = [{("arc", 0): Q(5), ("arc", 1): Q(0)}]
    c0 = Eng(alpha=Q(1, 4), a=Q(1), m=2)
    c1 = Eng(alpha=Q(1, 4), a=Q(1), m=2, hold=True, preload=True)
    s0, s1 = run_P([T], lag, c0), run_P([T], lag, c1)
    return s0["C"][0] - s1["C"][0], c0.alpha * s1["held_hits"], T, lag


# ============================================================================================ C. keep / evict


def ttl_cost_(D, h, R, tau):
    return sum(p * (h * min(v, tau) + (R if v > tau else 0)) for v, p in D)


def ttl_opt_grid(D, h, R, grid):
    return min(ttl_cost_(D, h, R, t) for t in grid)


def karlin_cost(d, B):
    """Exact (float) expected cost of the randomized TTL with density e^{t/B}/(B(e-1)) on [0,B], h = 1, R = B,
    when the wait is d: integral_0^{min(d,B)} (t + B) p(t) dt + d * P(tau >= d)."""
    e = math.e
    x = min(d, B)
    # int_0^x t e^{t/B}/(B(e-1)) dt = [B e^{t/B}(t/B - 1)]_0^x / (e-1) ... computed in closed form:
    int_t = (B * math.exp(x / B) * (x / B - 1) + B) / (e - 1)
    int_1 = (math.exp(x / B) - 1) / (e - 1)
    tail = 1 - int_1
    return int_t + B * int_1 + (d * tail if d < B else 0.0)


def discrete_ski_ratio(n):
    return 1 / (1 - (1 - 1 / n) ** n)


def keep_lp_vertex(classes, nu_cp, nu_ld, Mh):
    """Exact optimum of the R5b keep LP by vertex enumeration (independent of Z3 and of the dual): with one
    coupling constraint, some optimal vertex has every class pure except at most one class that mixes two options.
    classes: list of (lam, P, L, D). Options per class: None (cost 0, mem 0) or each TTL in {0} U supp(D)."""
    opts = []
    for (lam, P, L, D) in classes:
        row = [(Q(0), Q(0))]
        for tau in sorted({Q(0)} | {v for v, _ in D}):
            sig = sum(p for v, p in D if v <= tau)
            row.append((lam * P * (nu_ld * (1 - sig) - nu_cp * sig), lam * L * sum(p * min(v, tau) for v, p in D)))
        opts.append(row)
    best = None
    n = len(classes)
    for choice in itertools.product(*[range(len(r)) for r in opts]):
        cost = sum(opts[i][choice[i]][0] for i in range(n))
        mem = sum(opts[i][choice[i]][1] for i in range(n))
        if mem <= Mh:
            best = cost if best is None or cost < best else best
        # one class i mixes choice[i] with another option j so that the memory constraint is tight
        for i in range(n):
            for j in range(len(opts[i])):
                if j == choice[i]:
                    continue
                c1, m1 = opts[i][choice[i]]
                c2, m2 = opts[i][j]
                rest_c, rest_m = cost - c1, mem - m1
                if m1 == m2:
                    continue
                lam_ = (Mh - rest_m - m2) / (m1 - m2)  # weight on choice[i]
                if 0 <= lam_ <= 1:
                    val = rest_c + lam_ * c1 + (1 - lam_) * c2
                    best = val if best is None or val < best else best
    return best


def keep_dual_scan(classes, nu_cp, nu_ld, Mh, etas):
    """max over the given eta grid of the Lagrangian dual (a lower bound on the LP)."""
    def g(eta):
        tot = -eta * Mh
        for (lam, P, L, D) in classes:
            best = Q(0)
            for tau in sorted({Q(0)} | {v for v, _ in D}):
                sig = sum(p for v, p in D if v <= tau)
                val = lam * P * (nu_ld * (1 - sig) - nu_cp * sig) + eta * lam * L * sum(p * min(v, tau) for v, p in D)
                best = min(best, val)
            tot += best
        return tot
    return max(g(e) for e in etas)


# ============================================================================================ D. R7 constructions


def index_vs_probe(n, q, L):
    """Exact values: any fixed-order policy on n i.i.d. {0 w.p. 1-q, L w.p. q} jobs (one machine) vs probing."""
    fixed = sum(i * q * L for i in range(1, n + 1))
    probe = sum(math.comb(n, K) * q**K * (1 - q) ** (n - K) * L * Q(K * (K + 1), 2) for K in range(n + 1))
    return fixed, probe


def hold_only_value_lb(n, k, t_in_flight):
    return sum(-(-i // k) * t_in_flight for i in range(1, n + 1))
