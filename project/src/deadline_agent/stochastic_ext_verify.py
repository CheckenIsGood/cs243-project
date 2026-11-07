"""Independent verification of docs/stochastic_theory_ext.md (commit eaa6958).

Everything below is written from scratch and imports NOTHING from `stochastic_theory.py` or
`stochastic_theory_ext.py`: own work-process model, own Gittins index (brute force over all stopping sets, plus an
independent Dinkelbach used only after being cross-checked against the brute force), own simulators (machine
model with lags; hybrid engine with unbounded memory; hybrid engine with KV memory, priority-prefix admission and
kicks), own exact DPs (single-machine optimum; m-machine preemptive non-anticipatory grid DP; clairvoyant grid DP),
own lower bounds and constants. Exact rational arithmetic (fractions.Fraction) everywhere except in the
adversarial searches, whose candidates are re-checked exactly. Z3 is used for algebra.

Sections
  A. laws and chain work processes; Gittins index; quanta
  B. machine model (Thm G): Gittins-quanta policy on m machines with releases and lags; V matrix; Lemma G3;
     single-machine identity G2; single-machine exact optimum; m-machine preemptive grid DP (upper bound on
     OPT_na, used to try to FALSIFY Lemma G1)
  C. Thm P: P-WSPT-DAG simulator (DAGs, random sizes, lags), Smith values, pair constant, Lemma P1, clairvoyant
     grid DP, the fixed-priority 1+Delta family
  D. Thm H: hybrid engine (unbounded memory) with Gittins quanta on the Z'-process; s-process; lambda; beta_cp
  E. bounded-memory hybrid engine (Thm D / DR / Prop K / kappa instance)
  F. Thm A lower bounds: the sqrt(2) family and the (known) Schulz-Skutella / Megow-Schulz family -> 2
  G. calibrated constants (Thm H, Thm DR, Prop K)
  H. adversarial searches
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from fractions import Fraction as Fr
from functools import lru_cache

DONE = ("done",)


# ===================================================================================================== A. basics
def law(*pairs):
    """Finite law as a sorted tuple of (value, prob); weights are normalized; equal values merged."""
    acc = {}
    for v, p in pairs:
        v, p = Fr(v), Fr(p)
        if p:
            acc[v] = acc.get(v, Fr(0)) + p
    tot = sum(acc.values())
    return tuple(sorted((v, p / tot) for v, p in acc.items()))


def point(v):
    return law((v, 1))


def lmean(d):
    return sum(v * p for v, p in d)


def lvar(d):
    m = lmean(d)
    return sum(p * (v - m) ** 2 for v, p in d)


def lcv2(d):
    m = lmean(d)
    return lvar(d) / (m * m)


@dataclass(frozen=True)
class Node:
    """A chain node: deterministic lump (plus ctx * parent's realized unit count), then N ~ law units of `unit` work
    each, N >= 1, revealed unit by unit."""

    law: tuple
    unit: Fr = Fr(1)
    lump: Fr = Fr(0)
    ctx: Fr = Fr(0)


@dataclass(frozen=True)
class Task:
    r: Fr
    w: Fr
    nodes: tuple
    lags: tuple = ()  # laws of the lag after node i (len = len(nodes) - 1)


def first_state(T: Task, i: int, px: int):
    nd = T.nodes[i]
    return ("L", i, px) if nd.lump + nd.ctx * px > 0 else ("U", i, 0, px)


def start(T: Task):
    return first_state(T, 0, 0)


def moves(T: Task, s):
    """Transitions (prob, work, next state) out of a live state."""
    if s[0] == "L":
        _, i, px = s
        nd = T.nodes[i]
        return [(Fr(1), nd.lump + nd.ctx * px, ("U", i, 0, px))]
    _, i, k, px = s
    nd = T.nodes[i]
    tail = sum(p for v, p in nd.law if v >= k + 1)
    h = sum(p for v, p in nd.law if v == k + 1) / tail
    out = []
    if h:
        out.append((h, nd.unit, DONE if i == len(T.nodes) - 1 else first_state(T, i + 1, k + 1)))
    if h != 1:
        out.append((1 - h, nd.unit, ("U", i, k + 1, px)))
    return out


def reachable(T: Task, x):
    seen, stack = [], [x]
    while stack:
        s = stack.pop()
        if s == DONE or s in seen:
            continue
        seen.append(s)
        stack.extend(nx for _, _, nx in moves(T, s))
    return seen


def _ratio_for_set(T: Task, x, cont: frozenset):
    memo = {}

    def ev(s, forced):
        key = (s, forced)
        if key in memo:
            return memo[key]
        if not forced and s not in cont:
            memo[key] = (Fr(0), Fr(0))
            return memo[key]
        R = W = Fr(0)
        for p, wk, nx in moves(T, s):
            W += p * wk
            if nx == DONE:
                R += p * T.w
            else:
                r2, w2 = ev(nx, False)
                R += p * r2
                W += p * w2
        memo[key] = (R, W)
        return R, W

    R, W = ev(x, True)
    return R / W


_GB = {}


def gittins_brute(T: Task, x) -> Fr:
    """Gittins index = max over ALL deterministic Markov continuation sets C (first step forced) of
    w P(complete before leaving C) / E[work until leaving C or completion]. Exponential; small states only."""
    key = (T, x)
    if key in _GB:
        return _GB[key]
    others = [s for s in reachable(T, x) if s != x]
    assert len(others) <= 14, "state space too large for brute force"
    best = None
    for k in range(len(others) + 1):
        for C in itertools.combinations(others, k):
            v = _ratio_for_set(T, x, frozenset(C))
            if best is None or v > best:
                best = v
    _GB[key] = best
    return best


_GD = {}


def gittins_dinkelbach(T: Task, x) -> Fr:
    """Independent fast path: Dinkelbach on lambda with exact optimal stopping (continue iff value > 0).
    Only used after cross-checking against gittins_brute (tests)."""
    key = (T, x)
    if key in _GD:
        return _GD[key]
    states = reachable(T, x)
    lam = _ratio_for_set(T, x, frozenset(states))  # start: never stop early
    while True:
        val = {}

        def V(s):
            if s in val:
                return val[s]
            v = Fr(0)
            for p, wk, nx in moves(T, s):
                v += p * (-lam * wk + (T.w if nx == DONE else max(Fr(0), V(nx))))
            val[s] = v
            return v

        cont = frozenset(s for s in states if s != x and V(s) > 0)
        new = _ratio_for_set(T, x, cont)
        if new <= lam:
            _GD[key] = lam
            return lam
        lam = new


def path(T: Task, Ns):
    """Realized path [(state, work)] for realized unit counts Ns (one per node)."""
    s, out = start(T), []
    while s != DONE:
        if s[0] == "L":
            p, wk, nx = moves(T, s)[0]
        else:
            _, i, k, px = s
            wk = T.nodes[i].unit
            if k + 1 == Ns[i]:
                nx = DONE if i == len(T.nodes) - 1 else first_state(T, i + 1, k + 1)
            else:
                nx = ("U", i, k + 1, px)
        out.append((s, wk))
        s = nx
    return out


def quanta(T: Task, Ns, G):
    """Megow-Vredeveld quanta along the realized path: priority = running minimum of G over visited states;
    a new quantum starts exactly when the index drops strictly below the current quantum's priority."""
    out = []
    for s, wk in path(T, Ns):
        g = G(T, s)
        if not out or g < out[-1][0]:
            out.append([g, Fr(0)])
        out[-1][1] += wk
    return [tuple(q) for q in out]


def realizations(tasks):
    """All joint realizations (prob, Ns[t], Ds[t]); all unit counts and lags independent."""
    slots = []
    for t, T in enumerate(tasks):
        for i, nd in enumerate(T.nodes):
            slots.append((0, t, i, nd.law))
        for i, d in enumerate(T.lags):
            slots.append((1, t, i, d))
    for combo in itertools.product(*[s[3] for s in slots]):
        pr = Fr(1)
        Ns = [[None] * len(T.nodes) for T in tasks]
        Ds = [[None] * len(T.lags) for T in tasks]
        for s, (v, p) in zip(slots, combo):
            pr *= p
            if s[0] == 0:
                Ns[s[1]][s[2]] = int(v)
            else:
                Ds[s[1]][s[2]] = v
        yield pr, Ns, Ds


def total_work(T, Ns):
    return sum(wk for _, wk in path(T, Ns))


def expected_total_work(T):
    return sum(pr * total_work(T, Ns[0]) for pr, Ns, _ in realizations([T]))


# ======================================================================================= B. machine model (Thm G)
def mtask(r, w, sizes, lags=()):
    return Task(Fr(r), Fr(w), tuple(Node(d) for d in sizes), tuple(lags))


def sim_machines(tasks, Ns, Ds, m, G=None, fixed=None):
    """Exact integer-time simulation on m machines (all data integral; every step is one unit of work).
    Policy: run the m available tasks of smallest key (-priority, id); priority = running min of G over the
    task's visited states (Gittins quanta), or a fixed number `fixed[t]` (fixed-priority mutation)."""
    n = len(tasks)
    P = [path(T, Ns[t]) for t, T in enumerate(tasks)]
    for t in range(n):
        assert all(wk == 1 for _, wk in P[t]), "machine model needs unit steps"
    pos, prio = [0] * n, [None] * n
    avail = [T.r for T in tasks]
    C = [None] * n
    now = Fr(0)
    while None in C:
        av = [t for t in range(n) if C[t] is None and avail[t] <= now]
        if not av:
            now = min(avail[t] for t in range(n) if C[t] is None)
            continue
        for t in av:
            if fixed is not None:
                prio[t] = fixed[t]
            else:
                g = G(tasks[t], P[t][pos[t]][0])
                prio[t] = g if prio[t] is None or g < prio[t] else prio[t]
        av.sort(key=lambda t: (-prio[t], t))
        now += 1
        for t in av[:m]:
            s = P[t][pos[t]][0]
            pos[t] += 1
            if pos[t] == len(P[t]):
                C[t] = now
            elif P[t][pos[t]][0][1] != s[1]:  # node index changed: lag
                avail[t] = now + Ds[t][s[1]]
    return C


def v_matrix(tasks, Q):
    n = len(tasks)
    last = [(-Q[j][-1][0], j) for j in range(n)]
    return [[sum((wk for g, wk in Q[k] if (-g, k) < last[j]), Fr(0)) if k != j else Fr(0) for j in range(n)]
            for k in range(n)]


def cp_chain(T, Ns, Ds):
    return total_work(T, Ns) + sum(Ds, Fr(0))


def lemma_G3_bounds(tasks, Ns, Ds, m, G):
    Q = [quanta(T, Ns[t], G) for t, T in enumerate(tasks)]
    V = v_matrix(tasks, Q)
    return [T.r + cp_chain(T, Ns[t], Ds[t]) + sum(V[k][t] for k in range(len(tasks))) / m
            for t, T in enumerate(tasks)]


def single_machine_quanta_C(tasks, Ns, G):
    """Lemma G2's schedule: all quanta in increasing key order; returns completion times."""
    Q = [quanta(T, Ns[t], G) for t, T in enumerate(tasks)]
    items = sorted(((-g, t), wk) for t in range(len(tasks)) for g, wk in Q[t])
    clock, C = Fr(0), [None] * len(tasks)
    for (key, wk) in items:
        clock += wk
        C[key[1]] = clock
    return C


def gitt1(tasks, G):
    """Gitt_1: expected value of the quanta order on one speed-1 machine (releases and lags ignored)."""
    return sum(pr * sum(T.w * c for T, c in zip(tasks, single_machine_quanta_C(tasks, Ns, G)))
               for pr, Ns, _ in realizations(tasks))


def single_machine_opt(tasks):
    """Exact optimum of E sum w C over non-anticipatory preemptive single-machine policies, no releases/lags,
    decisions at state transitions (a partially processed transition reveals nothing; interchange)."""
    tasks = tuple(tasks)

    @lru_cache(maxsize=None)
    def V(states):
        live = [t for t, s in enumerate(states) if s != DONE]
        if not live:
            return Fr(0)
        wl = sum(tasks[t].w for t in live)
        return min(sum(p * (wk * wl + V(states[:t] + (nx,) + states[t + 1:]))
                       for p, wk, nx in moves(tasks[t], states[t])) for t in live)

    return V(tuple(start(T) for T in tasks))


def grid_opt_machines(tasks, m, g=1):
    """UPPER bound on the m-machine preemptive non-anticipatory optimum with releases and lags (machine model).
    Time grid 1/g; each chosen task gets one machine for 1/g; every policy in this class is feasible, so the value
    is >= OPT_na. Lags are revealed when they START (extra information), so the value is even an upper bound on
    the optimum of policies that are clairvoyant about lags but not about sizes; Lemma G1 (OPT_na >= Gitt_1/m) and
    LB_cp hold for that larger class too, so 'grid value < Gitt_1/m' would refute G1."""
    tasks = tuple(tasks)
    d = Fr(1, g)
    n = len(tasks)

    @lru_cache(maxsize=None)
    def V(now, st):
        # st[t] = DONE | (avail_time, state, frac)
        live = [t for t in range(n) if st[t] != DONE]
        if not live:
            return Fr(0)
        av = [t for t in live if st[t][0] <= now]
        if not av:
            return V(min(st[t][0] for t in live), st)
        best = None
        for S in itertools.combinations(av, min(m, len(av))):
            # enumerate outcomes of the chosen tasks
            branches = [(Fr(1), list(st), Fr(0))]
            for t in S:
                a, s, f = st[t]
                nb = []
                for pr, cur, cost in branches:
                    if f + d < 1:
                        c2 = list(cur)
                        c2[t] = (a, s, f + d)
                        nb.append((pr, c2, cost))
                        continue
                    for p, wk, nx in moves(tasks[t], s):
                        if nx == DONE:
                            c2 = list(cur)
                            c2[t] = DONE
                            nb.append((pr * p, c2, cost + tasks[t].w * (now + d)))
                        elif nx[1] != s[1]:
                            for D, q in tasks[t].lags[s[1]]:
                                c2 = list(cur)
                                c2[t] = (now + d + D, nx, Fr(0))
                                nb.append((pr * p * q, c2, cost))
                        else:
                            c2 = list(cur)
                            c2[t] = (a, nx, Fr(0))
                            nb.append((pr * p, c2, cost))
                branches = nb
            v = sum(pr * (cost + V(now + d, tuple(c))) for pr, c, cost in branches)
            if best is None or v < best:
                best = v
        return best

    return V(Fr(0), tuple((T.r, start(T), Fr(0)) for T in tasks))


def lb_cp_machine(tasks):
    return sum(T.w * (T.r + sum(pr * cp_chain(T, Ns[0], Ds[0]) for pr, Ns, Ds in realizations([T])))
               for T in tasks)


def strip(tasks):
    return [Task(Fr(0), T.w, T.nodes, tuple(point(0) for _ in T.lags)) for T in tasks]


def expected_alg_machines(tasks, m, G=None, fixed=None):
    return sum(pr * sum(T.w * c for T, c in zip(tasks, sim_machines(tasks, Ns, Ds, m, G, fixed)))
               for pr, Ns, Ds in realizations(tasks))


def thmG_check(tasks, m, G=gittins_brute, grid=None):
    """Returns a dict with every link of Theorem G computed independently."""
    alg = expected_alg_machines(tasks, m, G)
    lb = lb_cp_machine(tasks)
    g1 = gitt1(strip(tasks), G)
    ew = sum(T.w * expected_total_work(T) for T in tasks)
    out = dict(alg=alg, lb_cp=lb, gitt1=g1, ew=ew, bound=lb + (g1 - ew) / m)
    if grid:
        out["grid_ub"] = grid_opt_machines(tasks, m, grid)
    return out


# ====================================================================================== C. Thm P (index vs clair)
@dataclass(frozen=True)
class DTask:
    """DAG task, machine model. joint: law over tuples of node sizes (allows within-task correlation);
    arcs: (u, v, lag law) with u < v (node index order is topological)."""

    r: Fr
    w: Fr
    joint: tuple
    arcs: tuple = ()

    @property
    def n(self):
        return len(self.joint[0][0])


def dtask_indep(r, w, sizes, arcs=()):
    joint = tuple((tuple(int(v) for v, _ in combo), math.prod((p for _, p in combo), start=Fr(1)))
                  for combo in itertools.product(*sizes))
    return DTask(Fr(r), Fr(w), joint, tuple(arcs))


def d_realizations(tasks):
    slots = []
    for t, T in enumerate(tasks):
        slots.append((0, t, None, T.joint))
        for a, (u, v, d) in enumerate(T.arcs):
            slots.append((1, t, a, d))
    for combo in itertools.product(*[s[3] for s in slots]):
        pr = Fr(1)
        P = [None] * len(tasks)
        D = [dict() for _ in tasks]
        for s, (val, p) in zip(slots, combo):
            pr *= p
            if s[0] == 0:
                P[s[1]] = val
            else:
                u, v, _ = tasks[s[1]].arcs[s[2]]
                D[s[1]][(u, v)] = val
        yield pr, P, D


def task_law_W(T):
    acc = {}
    for sz, p in T.joint:
        acc[sum(sz)] = acc.get(sum(sz), Fr(0)) + p
    return tuple(sorted(acc.items()))


def dag_cp(T, sz, D):
    fin = {}
    for v in range(T.n):
        st = max([fin[u] + D[(u, x)] for (u, x, _) in T.arcs if x == v] + [Fr(0)])
        fin[v] = st + sz[v]
    return max(fin.values())


def sim_pwspt_dag(tasks, P, D, m):
    """P-WSPT-DAG (fixed task index w/E[W], node priority (rank, node index)), m machines, integer data."""
    EW = [lmean(task_law_W(T)) for T in tasks]
    rank = {t: i for i, t in enumerate(sorted(range(len(tasks)), key=lambda t: (-(tasks[t].w / EW[t]), t)))}
    rem = {(t, v): Fr(P[t][v]) for t, T in enumerate(tasks) for v in range(T.n)}
    fin = {}
    C = [None] * len(tasks)
    now = Fr(0)

    def ready(t, v):
        T = tasks[t]
        if T.r > now:
            return None
        preds = [(u, x) for (u, x, _) in T.arcs if x == v]
        if any((t, u) not in fin for u, _ in preds):
            return None
        return max([fin[(t, u)] + D[t][(u, x)] for u, x in preds] + [T.r])

    while None in C:
        av = [(t, v) for (t, v), r_ in rem.items() if r_ > 0 and (ready(t, v) is not None and ready(t, v) <= now)]
        if not av:
            cands = [T.r for T in tasks if T.r > now]
            for (t, v), r_ in rem.items():
                if r_ > 0:
                    rt = ready(t, v) if tasks[t].r <= now else None
                    if rt is not None and rt > now:
                        cands.append(rt)
            now = min(cands)
            continue
        av.sort(key=lambda tv: (rank[tv[0]], tv[1]))
        now += 1
        for (t, v) in av[:m]:
            rem[(t, v)] -= 1
            if rem[(t, v)] == 0:
                fin[(t, v)] = now
                if all((t, u) in fin for u in range(tasks[t].n)):
                    C[t] = now
    return C


def smith_value(w, p):
    order = sorted(range(len(w)), key=lambda i: (-(Fr(w[i]) / p[i]) if p[i] else -math.inf, i))
    clock = tot = Fr(0)
    for i in order:
        clock += p[i]
        tot += w[i] * clock
    return tot


def thmP_parts(tasks, m):
    alg = lbcp = es = Fr(0)
    for pr, P, D in d_realizations(tasks):
        C = sim_pwspt_dag(tasks, P, D, m)
        alg += pr * sum(T.w * c for T, c in zip(tasks, C))
        lbcp += pr * sum(T.w * (T.r + dag_cp(T, P[t], D[t])) for t, T in enumerate(tasks))
        es += pr * smith_value([T.w for T in tasks], [Fr(sum(P[t])) for t in range(len(tasks))])
    laws = [task_law_W(T) for T in tasks]
    se = smith_value([T.w for T in tasks], [lmean(l) for l in laws])
    pair = Fr(1)
    for j, k in itertools.combinations(range(len(tasks)), 2):
        wj, wk = tasks[j].w, tasks[k].w
        a, b = wk * lmean(laws[j]), wj * lmean(laws[k])
        e = sum(p * q * min(wk * x, wj * y) for x, p in laws[j] for y, q in laws[k])
        pair = max(pair, min(a, b) / e)
    dW = max(lcv2(l) for l in laws)
    return dict(alg=alg, lb_cp=lbcp, smith_E=se, E_smith=es, pair=pair, delta_W=dW, m=m)


def rho_bound_ok(pair, D):
    """pair <= min{(27/16)(1+D), 1/(1-sqrt(D/2))} checked exactly (second branch only when D < 2)."""
    if pair <= Fr(27, 16) * (1 + D):
        return True
    if D < 2:
        # pair <= 1/(1 - s), s = sqrt(D/2)  <=>  1 - 1/pair <= s  <=> (1 - 1/pair)^2 <= D/2 (lhs base >= 0)
        return (1 - 1 / pair) ** 2 <= D / 2
    return False


def clair_grid_opt(tasks, P, D, m):
    """Exact-on-integer-grid clairvoyant optimum for ONE realization (deterministic data): an UPPER bound on
    OPT_clair(omega) (exact for m = 1 in all cases we compared against a finer grid)."""
    n = len(tasks)
    nodes = [(t, v) for t in range(n) for v in range(tasks[t].n)]

    @lru_cache(maxsize=None)
    def V(now, rem, finl):
        fin = dict(finl)
        if all(r_ == 0 for r_ in rem):
            return Fr(0)
        av, fut = [], []
        for idx, (t, v) in enumerate(nodes):
            if rem[idx] == 0:
                continue
            T = tasks[t]
            preds = [(u, x) for (u, x, _) in T.arcs if x == v]
            if any((t, u) not in fin for u, _ in preds):
                continue
            rt = max([fin[(t, u)] + D[t][(u, x)] for u, x in preds] + [T.r])
            (av if rt <= now else fut).append(idx if rt <= now else rt)
        if not av:
            return V(min(fut), rem, finl)
        best = None
        for S in itertools.combinations(av, min(m, len(av))):
            r2 = list(rem)
            f2 = dict(fin)
            cost = Fr(0)
            for idx in S:
                r2[idx] -= 1
                if r2[idx] == 0:
                    f2[nodes[idx]] = now + 1
            for t in range(n):
                idxs = [i for i, (tt, _) in enumerate(nodes) if tt == t]
                if any(rem[i] > 0 for i in idxs) and all(r2[i] == 0 for i in idxs):
                    cost += tasks[t].w * (now + 1)
            v = cost + V(now + 1, tuple(r2), tuple(sorted(f2.items())))
            if best is None or v < best:
                best = v
        return best

    rem0 = tuple(Fr(P[t][v]) for (t, v) in nodes)
    return V(Fr(0), rem0, ())


def expected_clair_ub(tasks, m):
    return sum(pr * clair_grid_opt(tasks, P, D, m) for pr, P, D in d_realizations(tasks))


def emin(X, Y):
    return sum(p * q * min(x, y) for x, p in X for y, q in Y)


def normalize(d):
    m = lmean(d)
    return tuple((v / m, p) for v, p in d)


def lemmaP1_holds(X, Y):
    X, Y = normalize(X), normalize(Y)
    D = max(lcv2(X), lcv2(Y))
    e = emin(X, Y)
    b1 = e >= Fr(16, 27) / (1 + D)
    b2 = e >= 1 or (1 - e) ** 2 <= D / 2
    return b1 and b2, e, D


def fixed_priority_family_ratio(n, q):
    """n i.i.d. jobs P in {0 w.p. 1-q, L w.p. q}, one machine: fixed order E = qL n(n+1)/2; probing (preemptive
    non-anticipatory, = clairvoyant here) E = L E[K(K+1)/2], K ~ Bin(n, q). Ratio is independent of L."""
    fixed = q * n * (n + 1) / 2
    EK = n * q
    EK2 = n * q * (1 - q) + (n * q) ** 2
    probe = (EK2 + EK) / 2
    return fixed / probe


# ============================================================================ D. Thm H: hybrid, unbounded memory
@dataclass(frozen=True)
class Cfg:
    alpha: Fr
    a: Fr
    m: int
    Mp: Fr | None = None  # memory M' (None = unbounded)

    @property
    def c(self):
        return self.a / self.m


@dataclass(frozen=True)
class RN:
    q: int
    law: tuple
    rho: int = 0
    ctx: int = 0  # recompute ctx * parent's realized X extra tokens


@dataclass(frozen=True)
class RT:
    r: Fr
    w: Fr
    nodes: tuple
    lags: tuple = ()


def zproc(T: RT, cfg: Cfg) -> Task:
    return Task(T.r, T.w, tuple(Node(nd.law, cfg.c, cfg.alpha * (nd.q + nd.rho), cfg.alpha * nd.ctx)
                                for nd in T.nodes), T.lags)


def sproc(T: RT, cfg: Cfg) -> Task:
    return Task(T.r, T.w, tuple(Node(nd.law, cfg.c, cfg.alpha * nd.q) for nd in T.nodes), T.lags)


def sim_hybrid_unbounded(tasks, Xs, Ds, cfg: Cfg, G=None, fixed=None):
    """Exact fluid simulation, hybrid model with unbounded memory, evict-always. Server (one prefill at a time,
    lump of the Z'-process in seconds) and m decoders (a seconds per token, fractional progress kept). Priority
    = running min of the Gittins index of the task's Z'-state (or a fixed number)."""
    n = len(tasks)
    Z = [zproc(T, cfg) for T in tasks]
    P = [path(Z[t], Xs[t]) for t in range(n)]
    pos, prio = [0] * n, [None] * n
    avail = [T.r for T in tasks]
    C = [None] * n
    rem = [None] * n

    def kind(t):
        return P[t][pos[t]][0][0]

    def dur(t):
        s, wk = P[t][pos[t]]
        return wk if s[0] == "L" else cfg.a

    for t in range(n):
        rem[t] = dur(t)
    now = Fr(0)
    while None in C:
        av = [t for t in range(n) if C[t] is None and avail[t] <= now]
        for t in av:
            if fixed is not None:
                prio[t] = fixed[t]
            else:
                gv = G(Z[t], P[t][pos[t]][0])
                prio[t] = gv if prio[t] is None or gv < prio[t] else prio[t]
        key = lambda t: (-prio[t], t)
        pre = sorted([t for t in av if kind(t) == "L"], key=key)
        dec = sorted([t for t in av if kind(t) == "U"], key=key)
        run = pre[:1] + dec[:cfg.m]
        nxt = [now + rem[t] for t in run] + [avail[t] for t in range(n) if C[t] is None and avail[t] > now]
        t1 = min(nxt)
        for t in run:
            rem[t] -= t1 - now
        now = t1
        for t in run:
            if rem[t] == 0:
                s = P[t][pos[t]][0]
                pos[t] += 1
                if pos[t] == len(P[t]):
                    C[t] = now
                    continue
                if P[t][pos[t]][0][1] != s[1]:
                    avail[t] = now + Ds[t][s[1]]
                rem[t] = dur(t)
    return C


def hyb_cp(T: RT, Xs, Ds, cfg, alg: bool):
    tot, px = Fr(0), 0
    for nd, x in zip(T.nodes, Xs):
        if alg:
            tot += cfg.alpha * (nd.q + nd.rho + nd.ctx * px) + cfg.a * x
        else:
            tot += cfg.alpha * nd.q + (cfg.a + cfg.c) * x
        px = x
    return tot + sum(Ds, Fr(0))


def rt_realizations(tasks):
    yield from realizations([Task(T.r, T.w, tuple(Node(nd.law) for nd in T.nodes), T.lags) for T in tasks])


def lemma_H2_bounds(tasks, Xs, Ds, cfg, G):
    Z = [zproc(T, cfg) for T in tasks]
    Q = [quanta(Z[t], Xs[t], G) for t in range(len(tasks))]
    V = v_matrix(Z, Q)
    return [T.r + hyb_cp(T, Xs[t], Ds[t], cfg, True) + sum(V[k][t] for k in range(len(tasks)))
            for t, T in enumerate(tasks)]


def cumulative_lambda(Zt: Task, St: Task):
    """max over all realizable paths and states of (Z' work so far)/(s work so far)."""
    best = Fr(0)
    for _, Ns, _ in realizations([Zt]):
        cz = cs = Fr(0)
        for (sz, wz), (ss, ws) in zip(path(Zt, Ns[0]), path(St, Ns[0])):
            assert (sz[0], sz[1]) == (ss[0], ss[1])
            cz += wz
            cs += ws
            best = max(best, cz / cs)
    return best


def beta_cp_h(tasks, cfg):
    b = Fr(0)
    for T in tasks:
        pxmax = 0
        for nd in T.nodes:
            xmin = min(v for v, _ in nd.law)
            b = max(b, cfg.alpha * (nd.rho + nd.ctx * pxmax) / (cfg.alpha * nd.q + cfg.a * xmin))
            pxmax = int(max(v for v, _ in nd.law))
    return b


def thmH_parts(tasks, cfg, G=gittins_brute):
    alg = sum(pr * sum(T.w * c for T, c in zip(tasks, sim_hybrid_unbounded(tasks, Xs, Ds, cfg, G)))
              for pr, Xs, Ds in rt_realizations(tasks))
    Z = strip([zproc(T, cfg) for T in tasks])
    S = strip([sproc(T, cfg) for T in tasks])
    gz, gs = gitt1(Z, G), gitt1(S, G)
    ez = sum(T.w * expected_total_work(z) for T, z in zip(tasks, Z))
    lam = max(cumulative_lambda(z, s) for z, s in zip(Z, S))
    bcp = beta_cp_h(tasks, cfg)
    lb = sum(T.w * (T.r + sum(pr * hyb_cp(T, Xs[0], Ds[0], cfg, False) for pr, Xs, Ds in rt_realizations([T])))
             for T in tasks)
    return dict(alg=alg, gz=gz, gs=gs, ez=ez, lam=lam, bcp=bcp, lb_cp=lb, Z=Z, S=S)


# ====================================================== E. hybrid engine with KV memory (Thm D / DR / Prop K / kappa)
@dataclass(frozen=True)
class MN:
    """Realized memory-model node: q new prompt tokens, rho recomputed context, X realized output, res =
    reservation (rho + q + cap)."""

    q: int
    rho: int
    X: int
    res: int


@dataclass(frozen=True)
class MT:
    r: Fr
    w: Fr
    nodes: tuple  # chain of MN
    lags: tuple = ()  # realized lag values


def sim_memory(tasks, cfg: Cfg, order):
    """Fluid hybrid model with memory M' (cfg.Mp): admitted set = longest priority prefix of available nodes whose
    reservations fit in M' (head-of-line stop); nodes falling out of the prefix are KICKED (KV lost; they recompute
    prompt + decoded tokens as prefill); server prefills the top admitted prefill-phase node at 1/alpha tokens/s;
    the top m admitted decode-phase nodes decode at 1/a tokens/s; evict-always. `order` = task ids by priority."""
    rank = {t: i for i, t in enumerate(order)}
    n = len(tasks)
    st = {}  # (t, i) -> dict
    for t, T in enumerate(tasks):
        for i, nd in enumerate(T.nodes):
            st[(t, i)] = dict(avail=(T.r if i == 0 else None), phase="pre", pre=Fr(nd.rho + nd.q), dec=Fr(0),
                              adm=False, kv=Fr(0), done=False)
    C = [None] * n
    kicks, recompute, kicked_res, kick_events = 0, Fr(0), [], []
    now = Fr(0)
    while None in C:
        label = "avail" if any(not x["done"] and x["avail"] == now for x in st.values()) else "completion"
        av = sorted([k for k, s in st.items() if not s["done"] and s["avail"] is not None and s["avail"] <= now],
                    key=lambda k: (rank[k[0]], k[1]))
        used, prefix = 0, []
        for k in av:
            if used + tasks[k[0]].nodes[k[1]].res > cfg.Mp:
                break
            used += tasks[k[0]].nodes[k[1]].res
            prefix.append(k)
        pset = set(prefix)
        kres = 0
        for k, s in st.items():
            if s["adm"] and k not in pset:
                nd = tasks[k[0]].nodes[k[1]]
                kicks += 1
                kres += nd.res
                recompute += s["kv"]
                s["adm"], s["kv"] = False, Fr(0)
                s["phase"], s["pre"] = "pre", nd.rho + nd.q + s["dec"]
                kick_events.append(label)
        if kres:
            kicked_res.append(kres)
        for k in prefix:
            st[k]["adm"] = True
        pre = [k for k in prefix if st[k]["phase"] == "pre"][:1]
        dec = [k for k in prefix if st[k]["phase"] == "dec"][:cfg.m]
        nxt = [now + st[k]["pre"] * cfg.alpha for k in pre]
        nxt += [now + (tasks[k[0]].nodes[k[1]].X - st[k]["dec"]) * cfg.a for k in dec]
        nxt += [s["avail"] for s in st.values() if not s["done"] and s["avail"] is not None and s["avail"] > now]
        t1 = min(nxt)
        dt = t1 - now
        for k in pre:
            st[k]["pre"] -= dt / cfg.alpha
            st[k]["kv"] += dt / cfg.alpha
        for k in dec:
            st[k]["dec"] += dt / cfg.a
            st[k]["kv"] += dt / cfg.a
        now = t1
        for k in pre:
            if st[k]["pre"] == 0:
                st[k]["phase"] = "dec"
        for k in dec:
            nd = tasks[k[0]].nodes[k[1]]
            if st[k]["dec"] == nd.X:
                s = st[k]
                s["done"], s["adm"], s["kv"] = True, False, Fr(0)
                t, i = k
                if i == len(tasks[t].nodes) - 1:
                    C[t] = now
                else:
                    st[(t, i + 1)]["avail"] = now + tasks[t].lags[i]
    return dict(C=C, kicks=kicks, recompute=recompute, kicked_res=kicked_res, kick_events=kick_events)


def mem_Lmax(tasks):
    return max(nd.res for T in tasks for nd in T.nodes)


def z_bar(T: MT, cfg: Cfg, Lmax):
    M = cfg.Mp - Lmax
    return sum(cfg.alpha * (nd.q + nd.rho) + cfg.alpha * (nd.res + Lmax) + cfg.c * nd.X + cfg.a * nd.X * Fr(nd.res) / M
               for nd in T.nodes)


def s_size(T: MT, cfg: Cfg, Lmax):
    """sigma + A/M with sigma = alpha q + c X, A = (rho + q)(a + c) X, M = M' - Lmax."""
    M = cfg.Mp - Lmax
    return sum(cfg.alpha * nd.q + cfg.c * nd.X + Fr(nd.rho + nd.q) * (cfg.a + cfg.c) * nd.X / M for nd in T.nodes)


def cp_alg_mem(T: MT, cfg):
    return sum(cfg.alpha * (nd.q + nd.rho) + cfg.a * nd.X for nd in T.nodes) + sum(T.lags, Fr(0))


def cp_opt_mem(T: MT, cfg):
    return sum(cfg.alpha * nd.q + (cfg.a + cfg.c) * nd.X for nd in T.nodes) + sum(T.lags, Fr(0))


def beta_cp_mem(T: MT, cfg):
    return max(cfg.alpha * nd.rho / (cfg.alpha * nd.q + cfg.a * nd.X) for nd in T.nodes)


def lemma_D1_bounds(tasks, cfg, order, Lmax):
    rank = {t: i for i, t in enumerate(order)}
    Zb = [z_bar(T, cfg, Lmax) for T in tasks]
    return [T.r + cp_alg_mem(T, cfg) + sum(Zb[k] for k in range(len(tasks)) if rank[k] <= rank[t])
            for t, T in enumerate(tasks)]


@dataclass(frozen=True)
class DRN:
    q: int
    law: tuple
    rho: int = 0

    @property
    def cap(self):
        return int(max(v for v, _ in self.law))


@dataclass(frozen=True)
class DRT:
    r: Fr
    w: Fr
    nodes: tuple
    lags: tuple = ()  # lag laws between consecutive nodes


def dr_realize(tasks):
    slots = []
    for t, T in enumerate(tasks):
        for i, nd in enumerate(T.nodes):
            slots.append((0, t, i, nd.law))
        for i, d in enumerate(T.lags):
            slots.append((1, t, i, d))
    for combo in itertools.product(*[s[3] for s in slots]):
        pr = Fr(1)
        X = [[None] * len(T.nodes) for T in tasks]
        L = [[None] * len(T.lags) for T in tasks]
        for s, (v, p) in zip(slots, combo):
            pr *= p
            (X if s[0] == 0 else L)[s[1]][s[2]] = (int(v) if s[0] == 0 else v)
        yield pr, [MT(T.r, T.w, tuple(MN(nd.q, nd.rho, X[t][i], nd.rho + nd.q + nd.cap)
                                      for i, nd in enumerate(T.nodes)), tuple(L[t]))
                   for t, T in enumerate(tasks)]


def thmDR_parts(tasks, cfg):
    reals = list(dr_realize(tasks))
    Lmax = mem_Lmax(reals[0][1])
    EZ = [sum(pr * z_bar(R[t], cfg, Lmax) for pr, R in reals) for t in range(len(tasks))]
    order = sorted(range(len(tasks)), key=lambda t: (-(tasks[t].w / EZ[t]), t))
    alg = lbcp = esm = Fr(0)
    bcp = Fr(0)
    slaw = [dict() for _ in tasks]
    path_ok = True
    checks = 0
    kick_events = []
    for pr, R in reals:
        out = sim_memory(R, cfg, order)
        kick_events += out["kick_events"]
        for c, b in zip(out["C"], lemma_D1_bounds(R, cfg, order, Lmax)):
            path_ok &= c <= b
            checks += 1
        alg += pr * sum(T.w * c for T, c in zip(R, out["C"]))
        lbcp += pr * sum(T.w * (T.r + cp_opt_mem(T, cfg)) for T in R)
        bcp = max(bcp, max(beta_cp_mem(T, cfg) for T in R))
        sk = [s_size(T, cfg, Lmax) for T in R]
        esm += pr * smith_value([T.w for T in R], sk)
        for t in range(len(tasks)):
            slaw[t][sk[t]] = slaw[t].get(sk[t], Fr(0)) + pr
    slaw = [tuple(sorted(d.items())) for d in slaw]
    Es = [lmean(l) for l in slaw]
    w = [T.w for T in tasks]
    pair = Fr(1)
    for j, k in itertools.combinations(range(len(tasks)), 2):
        a_, b_ = w[k] * Es[j], w[j] * Es[k]
        e = sum(p * q * min(w[k] * x, w[j] * y) for x, p in slaw[j] for y, q in slaw[k])
        pair = max(pair, min(a_, b_) / e)
    lamE = max(z / e for z, e in zip(EZ, Es))
    return dict(alg=alg, lb_cp=lbcp, bcp=bcp, EZ=EZ, Es=Es, smithEZ=smith_value(w, EZ), smithEs=smith_value(w, Es),
                E_smith_s=esm, pair=pair, lamE=lamE, delta_s=max(lcv2(l) for l in slaw), path_ok=path_ok,
                checks=checks, kick_events=kick_events)


def propK_parts(tasks, cfg):
    """Deterministic token counts (cap = X): Prop K's chain with lambda_D."""
    Lmax = mem_Lmax(tasks)
    Zp = [z_bar(T, cfg, Lmax) for T in tasks]
    S = [s_size(T, cfg, Lmax) for T in tasks]
    order = sorted(range(len(tasks)), key=lambda t: (-(tasks[t].w / Zp[t]), t))
    out = sim_memory(tasks, cfg, order)
    w = [T.w for T in tasks]
    alg = sum(T.w * c for T, c in zip(tasks, out["C"]))
    lamD = max(z / s for z, s in zip(Zp, S))
    bcp = max(beta_cp_mem(T, cfg) for T in tasks)
    lb = sum(T.w * (T.r + cp_opt_mem(T, cfg)) for T in tasks)
    # old Thm D constant pieces
    M = cfg.Mp - Lmax
    beta = max((cfg.alpha * nd.rho + cfg.alpha * (nd.res + Lmax)) / (cfg.alpha * nd.q + cfg.c * nd.X)
               for T in tasks for nd in T.nodes)
    theta = max(Fr(nd.res, nd.rho + nd.q) for T in tasks for nd in T.nodes)
    return dict(alg=alg, lamD=lamD, bcp=bcp, lb_cp=lb, smithZ=smith_value(w, Zp), smithS=smith_value(w, S),
                beta=beta, theta=theta, sim=out, path=all(c <= b for c, b in zip(out["C"],
                                                                                 lemma_D1_bounds(tasks, cfg, order, Lmax))))


# ===================================================================================== F. Thm A lower-bound families
def pwspt_single(jobs):
    """Preemptive WSPT on one machine, deterministic jobs (r, p, w); exact event simulation; ties by index."""
    rem = [Fr(p) for _, p, _ in jobs]
    C = [None] * len(jobs)
    now = Fr(0)
    while None in C:
        av = [j for j in range(len(jobs)) if C[j] is None and jobs[j][0] <= now]
        fut = [jobs[j][0] for j in range(len(jobs)) if jobs[j][0] > now]
        if not av:
            now = min(fut)
            continue
        j = min(av, key=lambda j: (-(Fr(jobs[j][2]) / jobs[j][1]), j))
        t1 = min([now + rem[j]] + fut)
        rem[j] -= t1 - now
        now = t1
        if rem[j] == 0:
            C[j] = now
    return C


def nonpreemptive_in_order(jobs, order):
    now, C = Fr(0), [None] * len(jobs)
    for j in order:
        now = max(now, jobs[j][0]) + jobs[j][1]
        C[j] = now
    return C


def sqrt2_instance(L, k, eps):
    return [(Fr(0), Fr(L), Fr(L))] + [(Fr(L - 1), Fr(1), 1 + eps)] * k


def schulz_skutella_instance(n):
    """Megow-Schulz (ORL 2004) Lemma 2.2, m = 1: w_j = 1, p_j = n - j/n, r_j = j n - j(j+1)/(2n), 0 <= j <= n."""
    return [(Fr(j * n) - Fr(j * (j + 1), 2 * n), Fr(n) - Fr(j, n), Fr(1)) for j in range(n + 1)]


def wval(jobs, C):
    return sum(Fr(w) * c for (_, _, w), c in zip(jobs, C))


def det_opt_single_int(jobs):
    """Exact optimum of 1|r_j, pmtn|sum w C on the integer grid (integer data); for tiny instances only."""
    n = len(jobs)

    @lru_cache(maxsize=None)
    def V(now, rem):
        if all(r_ == 0 for r_ in rem):
            return Fr(0)
        av = [j for j in range(n) if rem[j] > 0 and jobs[j][0] <= now]
        if not av:
            return V(min(jobs[j][0] for j in range(n) if rem[j] > 0), rem)
        best = None
        for j in av:
            r2 = list(rem)
            r2[j] -= 1
            v = (Fr(jobs[j][2]) * (now + 1) if r2[j] == 0 else 0) + V(now + 1, tuple(r2))
            best = v if best is None or v < best else best
        return best

    return V(Fr(0), tuple(int(p) for _, p, _ in jobs))


# =============================================================================== G. calibrated constants (exact)
ALPHA, A_DEC, M51 = Fr("0.000114"), Fr("0.027767"), 51


def calib_thmH(random_X: bool):
    """Plan q=2000 then final q=1500, rho=2200 (as the doc); X laws {50,200,200,800}/{100,300,300,1200} or 200/300.
    Returns (beta_cp, lambda, 1 + beta_cp + lambda) computed by closed form (no Gittins needed)."""
    c = A_DEC / M51
    xp = [50, 200, 800] if random_X else [200]
    xf_min = 100 if random_X else 300
    bcp = ALPHA * 2200 / (ALPHA * 1500 + A_DEC * xf_min)
    lam = Fr(1)
    for x in xp:  # cumulative ratio peaks right after the final node's lump (then decreases with tokens)
        z = ALPHA * 2000 + c * x + ALPHA * 3700
        s = ALPHA * 2000 + c * x + ALPHA * 1500
        lam = max(lam, z / s)
    return bcp, lam, 1 + bcp + lam


def calib_propK(Mp):
    cfg = Cfg(ALPHA, A_DEC, M51, Fr(Mp))
    T = MT(Fr(0), Fr(1), (MN(2000, 0, 200, 2200), MN(1500, 2200, 300, 4000)), (Fr(1),))
    P = propK_parts([T], cfg)
    old = (1 + P["bcp"]) + 2 * max(1 + P["beta"], P["theta"])
    new = (1 + P["bcp"]) + 2 * P["lamD"]
    return dict(old=old, new=new, lamD=P["lamD"], beta=P["beta"], theta=P["theta"], bcp=P["bcp"])


def calib_DR(Mp=64000):
    cfg = Cfg(ALPHA, A_DEC, M51, Fr(Mp))
    T = DRT(Fr(0), Fr(1), (DRN(2000, law((50, 1), (200, 2), (800, 1))),
                           DRN(1500, law((100, 1), (300, 2), (1200, 1)), 2200)), (point(1),))
    reals = list(dr_realize([T]))
    Lmax = mem_Lmax(reals[0][1])
    EZ = sum(pr * z_bar(R[0], cfg, Lmax) for pr, R in reals)
    sl = law(*[(s_size(R[0], cfg, Lmax), pr) for pr, R in reals])
    Es = lmean(sl)
    D = lcv2(sl)
    bcp = max(beta_cp_mem(R[0], cfg) for _, R in reals)
    lamE = EZ / Es
    r1 = Fr(27, 16) * (1 + D)
    r2 = 1 / (1 - math.sqrt(float(D) / 2))
    rho = min(float(r1), r2)
    return dict(lamE=lamE, delta_s=D, bcp=bcp, rho=rho, total=float(1 + bcp) + 2 * float(lamE) * rho,
                cv2_Xp=lcv2(T.nodes[0].law), cv2_Xf=lcv2(T.nodes[1].law))


# ======================================================================================= H. adversarial searches
def _emin_float(xs, ps, ys, qs):
    return sum(p * q * min(x, y) for x, p in zip(xs, ps) for y, q in zip(ys, qs))


def _norm_float(xs, ps):
    s = sum(ps)
    ps = [p / s for p in ps]
    m = sum(x * p for x, p in zip(xs, ps))
    return [x / m for x in xs], ps


def _cv2_float(xs, ps):
    return sum(p * x * x for x, p in zip(xs, ps)) - 1.0


def hill_climb_emin(seed, iters=4000, k=3):
    """Minimize (1+Delta) E min(X, Y) over pairs of k-point laws (float search); returns the best exact pair
    (rationalized) with its exact objective."""
    rng = random.Random(seed)

    def obj(v):
        if max(v[0]) <= 1e-9 or max(v[2]) <= 1e-9:
            return math.inf
        xs, ps = _norm_float(v[0], v[1])
        ys, qs = _norm_float(v[2], v[3])
        D = max(_cv2_float(xs, ps), _cv2_float(ys, qs))
        return (1 + D) * _emin_float(xs, ps, ys, qs)

    cur = [[rng.uniform(0, 5) for _ in range(k)], [rng.uniform(0.05, 1) for _ in range(k)],
           [rng.uniform(0, 5) for _ in range(k)], [rng.uniform(0.05, 1) for _ in range(k)]]
    cur[0][0] = 0.0
    best = obj(cur)
    step = 0.5
    for it in range(iters):
        cand = [list(v) for v in cur]
        a = rng.randrange(4)
        b = rng.randrange(k)
        cand[a][b] = max(1e-6 if a % 2 else 0.0, cand[a][b] + rng.gauss(0, step))
        o = obj(cand)
        if o < best:
            cur, best = cand, o
        if it % 1000 == 999:
            step *= 0.6
    xs, ps = _norm_float(cur[0], cur[1])
    ys, qs = _norm_float(cur[2], cur[3])
    X = law(*[(Fr(x).limit_denominator(10**6), Fr(p).limit_denominator(10**6)) for x, p in zip(xs, ps)])
    Y = law(*[(Fr(x).limit_denominator(10**6), Fr(p).limit_denominator(10**6)) for x, p in zip(ys, qs)])
    ok, e, D = lemmaP1_holds(X, Y)
    return X, Y, (1 + D) * e, ok
