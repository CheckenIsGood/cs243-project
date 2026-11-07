"""Exact machinery for docs/stochastic_theory_ext.md (extensions of docs/stochastic_theory.md).

1. Work processes and Gittins indices. A chain task is a sequence of nodes; node i is a lump of deterministic work
   (possibly depending on the previous node's realized length) followed by N_i units, each costing `unit` work;
   N_i is random (finite integer law, N_i >= 1) and is revealed unit by unit (hazards). Gittins indices are
   computed EXACTLY (Dinkelbach iteration over stopping policies, Fractions); quanta follow Megow-Vredeveld.
2. Single-machine optimum of work processes (exact DP), used to verify the Gittins identity / optimality.
3. Machine model with lags (Thm G): the Gittins-quanta policy (F-GIPP-style) on m machines with release dates and
   random arc lags; exact integer-time simulation.
4. Hybrid engine without memory limit, random output lengths (Thm H): Gittins-quanta policy on the Z'-process;
   exact fluid simulation; s-process lower bound; cumulative-domination constant lambda.
5. Sharpened Thm D constant (Prop K): per-task ratio lambda_D = max_K Z'_K / (sigma_K + A_K/M).
6. Lower-bound families for Thm A (preemptive WSPT, sqrt(2)).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from typing import Sequence

from deadline_agent.stochastic_theory import (
    MTask, HCfg, HNode, HTask, Dist, det, dist, mean, opt_pmtn_dp, smith, h_sizes, h_Lmax, h_constants,
)

DONE = "DONE"


# ============================================================================================ 1. work processes
@dataclass(frozen=True)
class PNode:
    """lump: deterministic work at node start; ctx: extra lump per unit of the previous node's realized N (context
    recompute); unit: work per unit; law: law of the number of units N >= 1."""

    lump: F
    unit: F
    law: Dist
    ctx: F = F(0)


@dataclass(frozen=True)
class PTask:
    r: F
    w: F
    nodes: tuple  # tuple[PNode, ...] (a chain)
    lags: tuple = ()  # tuple[Dist, ...], len = len(nodes) - 1 (lag between node i and i+1)


def _haz(d: Dist, e: int) -> F:
    tail = sum(p for v, p in d if v >= e)
    return (sum(p for v, p in d if v == e) / tail) if tail else F(1)


def start_state(T: PTask):
    return _node_start(T, 0, 0)


def _node_start(T: PTask, i: int, px: int):
    nd = T.nodes[i]
    if nd.lump + nd.ctx * px > 0:
        return (i, -1, px)
    return (i, 0, px)


def transitions(T: PTask, s):
    """List of (prob, work, next_state) from a non-terminal state s = (i, k, px); k = -1: lump pending."""
    i, k, px = s
    nd = T.nodes[i]
    if k == -1:
        return [(F(1), nd.lump + nd.ctx * px, (i, 0, px))]
    h = _haz(nd.law, k + 1)
    out = []
    if h:
        nxt = DONE if i == len(T.nodes) - 1 else _node_start(T, i + 1, k + 1)
        out.append((h, nd.unit, nxt))
    if h != 1:
        out.append((1 - h, nd.unit, (i, k + 1, px)))
    return out


def gittins(T: PTask, x) -> F:
    """Exact Gittins index sup_{tau >= 1 step} w P(done by tau) / E[work until tau or done] (Dinkelbach)."""
    return _gittins_cached(T, x)


@lru_cache(maxsize=None)
def _gittins_cached(T: PTask, x) -> F:
    lam = F(0)
    while True:
        Q = _q_values(T, lam)
        # evaluate the policy "continue iff Q > 0" (forced first step)
        num, den = _eval_policy(T, x, lam, Q, forced=True)
        new = num / den
        if new == lam:
            return lam
        assert new > lam
        lam = new


def _q_values(T: PTask, lam: F):
    memo = {}

    def Q(s):
        if s in memo:
            return memo[s]
        v = F(0)
        for p, wk, nx in transitions(T, s):
            v += p * (-lam * wk + (T.w if nx == DONE else max(F(0), Q(nx))))
        memo[s] = v
        return v

    return Q


def _eval_policy(T, x, lam, Q, forced):
    memo = {}

    def ev(s, first):
        key = (s, first)
        if key in memo:
            return memo[key]
        if not first and Q(s) <= 0:
            memo[key] = (F(0), F(0))
            return memo[key]
        n = d = F(0)
        for p, wk, nx in transitions(T, s):
            d += p * wk
            if nx == DONE:
                n += p * T.w
            else:
                n2, d2 = ev(nx, False)
                n += p * n2
                d += p * d2
        memo[key] = (n, d)
        return n, d

    return ev(x, forced)


def realized_path(T: PTask, Ns: Sequence[int]):
    """States visited and work per step for realized unit counts Ns (one per node)."""
    s = start_state(T)
    steps = []
    while s != DONE:
        i, k, px = s
        if k == -1:
            p, wk, nx = transitions(T, s)[0]
        else:
            wk = T.nodes[i].unit
            if k + 1 == Ns[i]:
                nx = DONE if i == len(T.nodes) - 1 else _node_start(T, i + 1, k + 1)
            else:
                nx = (i, k + 1, px)
        steps.append((s, wk, nx))
        s = nx
    return steps


def quanta(T: PTask, Ns, index=gittins):
    """Megow-Vredeveld quanta along the realized path: a quantum starts at state x0 with priority G(x0) and lasts
    until the first state y with G(y) < G(x0). Returns list of (priority, work, start_step_index)."""
    steps = realized_path(T, Ns)
    out = []
    g0 = None
    for j, (s, wk, nx) in enumerate(steps):
        g = index(T, s)
        if g0 is None or g < g0:
            out.append([g, F(0), j])
            g0 = g
        out[-1][1] += wk
    return [tuple(q) for q in out]


def wsept_index(T: PTask, s) -> F:
    """Mutation: a FIXED priority (index of the start state, i.e. w / E[total work] for the whole task)."""
    x0 = start_state(T)
    return T.w / expected_work(T)


def expected_work(T: PTask) -> F:
    tot = F(0)
    pxs = {0: F(1)}
    for nd in T.nodes:
        En = mean(nd.law)
        tot += nd.lump + nd.unit * En + nd.ctx * sum(px * p for px, p in pxs.items())
        pxs = {v: p for v, p in nd.law}
    return tot


def p_realizations(tasks: Sequence[PTask]):
    """Joint realizations (prob, Ns[t], lags[t]) of unit counts and lags (all independent)."""
    slots = []
    for t, T in enumerate(tasks):
        for i, nd in enumerate(T.nodes):
            slots.append(("n", t, i, nd.law))
        for i, d in enumerate(T.lags):
            slots.append(("d", t, i, d))
    for combo in itertools.product(*[s[3] for s in slots]):
        prob = F(1)
        Ns = [[None] * len(T.nodes) for T in tasks]
        Ds = [[None] * len(T.lags) for T in tasks]
        for s, (v, p) in zip(slots, combo):
            prob *= p
            (Ns if s[0] == "n" else Ds)[s[1]][s[2]] = int(v) if s[0] == "n" else v
        yield prob, Ns, Ds


# ======================================================================== 2. single machine: Gittins and exact DP
def single_machine_quanta_cost(tasks: Sequence[PTask], Ns, index=gittins):
    """Pathwise single-machine (speed 1, no releases, no lags) completion times of the quanta-priority policy:
    quanta are processed in order of key (-priority, task id, quantum number). Returns (C list, quanta lists)."""
    Q = [quanta(T, Ns[t], index) for t, T in enumerate(tasks)]
    items = sorted((-g, t, j, wk) for t in range(len(tasks)) for j, (g, wk, _) in enumerate(Q[t]))
    clock, C = F(0), [None] * len(tasks)
    for key in items:
        clock += key[3]
        C[key[1]] = clock  # last quantum of t processed last among t's quanta
    return C, Q


def gittins_single_value(tasks: Sequence[PTask], index=gittins) -> F:
    tot = F(0)
    for prob, Ns, _ in p_realizations(tasks):
        C, _ = single_machine_quanta_cost(tasks, Ns, index)
        tot += prob * sum(T.w * c for T, c in zip(tasks, C))
    return tot


def single_machine_opt(tasks: Sequence[PTask]) -> F:
    """Exact optimum of E[sum w C] over non-anticipatory preemptive single-machine policies (speed 1, no releases,
    no lags) for work processes; decisions at state transitions (switching inside a transition never helps on a
    single machine: nothing is learned and an interchange argument applies)."""
    tasks = tuple(tasks)

    @lru_cache(maxsize=None)
    def V(states):
        live = [t for t, s in enumerate(states) if s != DONE]
        if not live:
            return F(0)
        Wl = sum(tasks[t].w for t in live)
        best = None
        for t in live:
            v = F(0)
            for p, wk, nx in transitions(tasks[t], states[t]):
                ns = states[:t] + (nx,) + states[t + 1:]
                v += p * (wk * Wl + V(ns))
            if best is None or v < best:
                best = v
        return best

    return V(tuple(start_state(T) for T in tasks))


# ======================================================================== 3. machine model with lags (Thm G)
def machine_ptask(r, w, sizes: Sequence[Dist], lags: Sequence[Dist]) -> PTask:
    return PTask(F(r), F(w), tuple(PNode(F(0), F(1), d) for d in sizes), tuple(lags))


def to_mtask(T: PTask) -> MTask:
    assert all(nd.lump == 0 and nd.unit == 1 and nd.ctx == 0 for nd in T.nodes)
    return MTask(T.r, T.w, tuple(nd.law for nd in T.nodes), tuple((i, i + 1, d) for i, d in enumerate(T.lags)))


def sim_gittins_machines(tasks: Sequence[PTask], Ns, Ds, m: int, index=gittins):
    """Exact simulation (integer data: releases, lags, unit counts) of the quanta-priority policy on m machines:
    at each integer time run the m available tasks of smallest key (-current quantum priority, task id) for one
    unit. Returns task completion times."""
    for T in tasks:
        assert all(nd.lump == 0 and nd.unit == 1 and nd.ctx == 0 for nd in T.nodes)
    n = len(tasks)
    paths = [realized_path(T, Ns[t]) for t, T in enumerate(tasks)]
    pos = [0] * n  # step index along path
    qg = [None] * n
    avail = [tasks[t].r for t in range(n)]
    C = [None] * n
    t_now = F(0)

    def cur_priority(t):
        s = paths[t][pos[t]][0]
        g = index(tasks[t], s)
        if qg[t] is None or g < qg[t]:
            qg[t] = g
        return qg[t]

    while any(c is None for c in C):
        av = [t for t in range(n) if C[t] is None and avail[t] <= t_now]
        if not av:
            t_now = min(avail[t] for t in range(n) if C[t] is None)
            continue
        av.sort(key=lambda t: (-cur_priority(t), t))
        run = av[:m]
        t_now += 1
        for t in run:
            s, wk, nx = paths[t][pos[t]]
            pos[t] += 1
            if nx == DONE:
                C[t] = t_now
            else:
                i_old, i_new = s[0], nx[0]
                if i_new != i_old:  # node finished: next node after the lag
                    avail[t] = t_now + Ds[t][i_old]
    return C


def chain_cp(T: PTask, Ns, Ds) -> F:
    """Machine model: critical path = total realized work + total realized lags (chain)."""
    return sum(F(n) * nd.unit + nd.lump for n, nd in zip(Ns, T.nodes)) + sum(Ds)


def v_matrix(tasks, Q):
    """V[K][J] = work of K in quanta with key strictly before J's last quantum key."""
    n = len(tasks)
    last = [(-Q[j][-1][0], j) for j in range(n)]
    V = [[F(0)] * n for _ in range(n)]
    for k in range(n):
        for j in range(n):
            if k != j:
                V[k][j] = sum(wk for g, wk, _ in Q[k] if (-g, k) < last[j])
    return V


def thmG_pathwise_bound(tasks, Ns, Ds, m, index=gittins):
    Q = [quanta(T, Ns[t], index) for t, T in enumerate(tasks)]
    V = v_matrix(tasks, Q)
    return [T.r + chain_cp(T, Ns[t], Ds[t]) + sum(V[k][t] for k in range(len(tasks)) if k != t) / m
            for t, T in enumerate(tasks)]


def expected_gittins_machines(tasks, m, index=gittins) -> F:
    tot = F(0)
    for prob, Ns, Ds in p_realizations(tasks):
        C = sim_gittins_machines(tasks, Ns, Ds, m, index)
        tot += prob * sum(T.w * c for T, c in zip(tasks, C))
    return tot


def lb_cp_p(tasks) -> F:
    tot = F(0)
    for T in tasks:
        e = F(0)
        for prob, Ns, Ds in p_realizations([T]):
            e += prob * chain_cp(T, Ns[0], Ds[0])
        tot += T.w * (T.r + e)
    return tot


def strip_release_and_lags(tasks):
    return [PTask(F(0), T.w, T.nodes, tuple(det(0) for _ in T.lags)) for T in tasks]


# ======================================================================== 4. hybrid engine, random X (Thm H)
@dataclass(frozen=True)
class RNode:
    """Hybrid-model LLM node with RANDOM output length X ~ law (X >= 1, revealed token by token). q new prompt
    tokens, rho recomputed parent-context tokens (evict-always), plus ctx * (parent's realized X)."""

    q: int
    law: Dist
    rho: int = 0
    ctx: int = 0


@dataclass(frozen=True)
class RTask:
    r: F
    w: F
    nodes: tuple  # chain of RNode
    lags: tuple = ()


def z_process(T: RTask, cfg: HCfg) -> PTask:
    """ALG's Z'-process (no memory limit): lump alpha (q + rho + ctx * prevX), unit c per decoded token."""
    return PTask(T.r, T.w, tuple(PNode(cfg.alpha * (nd.q + nd.rho), cfg.c, nd.law, cfg.alpha * nd.ctx)
                                 for nd in T.nodes), T.lags)


def s_process(T: RTask, cfg: HCfg) -> PTask:
    """Lower-bound s-process (GPU-time accrual of any engine schedule at speed 1): lump alpha q, unit c."""
    return PTask(T.r, T.w, tuple(PNode(cfg.alpha * nd.q, cfg.c, nd.law) for nd in T.nodes), T.lags)


def hyb_cp(T: RTask, Xs, Ds, cfg: HCfg, kind: str) -> F:
    """Critical path. 'alg': alpha (q + rho + ctx prevX) + a X per node; 'opt': alpha q + (a + c) X."""
    tot, px = F(0), 0
    for nd, x in zip(T.nodes, Xs):
        if kind == "alg":
            tot += cfg.alpha * (nd.q + nd.rho + nd.ctx * px) + cfg.a * x
        else:
            tot += cfg.alpha * nd.q + (cfg.a + cfg.c) * x
        px = x
    return tot + sum(Ds)


def sim_gittins_hybrid(tasks: Sequence[RTask], Xs, Ds, cfg: HCfg, index=gittins):
    """Exact fluid simulation of the Gittins-quanta policy in the hybrid model with unbounded memory, evict-always.
    Priority of a task = its current Z'-quantum priority (ties by task id). Server: highest-priority task whose
    current node is in prefill; decode machines: the m highest-priority tasks in decode (rate 1/a tokens/s each,
    fluid preemption keeps fractional token progress). Returns task completion times."""
    assert cfg.Mp is None
    n = len(tasks)
    Z = [z_process(T, cfg) for T in tasks]
    paths = [realized_path(Z[t], Xs[t]) for t in range(n)]
    pos = [0] * n
    qg = [None] * n
    avail = [T.r for T in tasks]
    rem = [None] * n  # remaining time of the current step (server: lump seconds; decoder: a per token)
    C = [None] * n
    now = F(0)

    def step_kind(t):
        s = paths[t][pos[t]][0]
        return "pre" if s[1] == -1 else "dec"

    def step_time(t):
        s, wk, nx = paths[t][pos[t]]
        return wk if s[1] == -1 else cfg.a

    def prio(t):
        s = paths[t][pos[t]][0]
        g = index(Z[t], s)
        if qg[t] is None or g < qg[t]:
            qg[t] = g
        return qg[t]

    for t in range(n):
        rem[t] = step_time(t)
    while any(c is None for c in C):
        av = [t for t in range(n) if C[t] is None and avail[t] <= now]
        key = lambda t: (-prio(t), t)
        pre = sorted([t for t in av if step_kind(t) == "pre"], key=key)
        dec = sorted([t for t in av if step_kind(t) == "dec"], key=key)
        run = pre[:1] + dec[: cfg.m]
        cand = [now + rem[t] for t in run] + [avail[t] for t in range(n) if C[t] is None and avail[t] > now]
        t1 = min(cand)
        for t in run:
            rem[t] -= t1 - now
        now = t1
        for t in run:
            if rem[t] == 0:
                s, wk, nx = paths[t][pos[t]]
                pos[t] += 1
                if nx == DONE:
                    C[t] = now
                    continue
                if nx[0] != s[0]:
                    avail[t] = now + Ds[t][s[0]]
                rem[t] = step_time(t)
    return C


def r_realizations(tasks: Sequence[RTask]):
    ps = [PTask(T.r, T.w, tuple(PNode(F(0), F(1), nd.law) for nd in T.nodes), T.lags) for T in tasks]
    yield from p_realizations(ps)


def thmH_pathwise_bound(tasks, Xs, Ds, cfg, index=gittins):
    Z = [z_process(T, cfg) for T in tasks]
    Q = [quanta(Zt, Xs[t], index) for t, Zt in enumerate(Z)]
    V = v_matrix(Z, Q)
    return [T.r + hyb_cp(T, Xs[t], Ds[t], cfg, "alg") + sum(V[k][t] for k in range(len(tasks)) if k != t)
            for t, T in enumerate(tasks)]


def cumulative_domination(Zp: PTask, Sp: PTask) -> F:
    """lambda_T = max over reachable states x of Z'(work up to x) / S(work up to x) (paths of the joint process;
    both processes share the state space). Requires S > 0 after the first transition."""
    best = F(0)
    for prob, Ns, _ in p_realizations([Zp]):
        pz, ps = realized_path(Zp, Ns[0]), realized_path(Sp, Ns[0])
        cz = cs = F(0)
        for (sz, wz, _), (ss, ws, _) in zip(pz, ps):
            cz += wz
            cs += ws
            assert cs > 0
            best = max(best, cz / cs)
    return best


def thmH_constants(tasks: Sequence[RTask], cfg: HCfg):
    """beta_cp (node-wise, worst X) and lambda (cumulative domination of Z' by s)."""
    bcp = F(0)
    for T in tasks:
        xmax_prev = 0
        for nd in T.nodes:
            xmin = min(v for v, _ in nd.law)
            rho = nd.rho + nd.ctx * xmax_prev
            bcp = max(bcp, cfg.alpha * rho / (cfg.alpha * nd.q + cfg.a * xmin))
            xmax_prev = max(v for v, _ in nd.law)
    lam = max(cumulative_domination(z_process(T, cfg), s_process(T, cfg)) for T in tasks)
    return bcp, lam


def lb_cp_h(tasks, cfg) -> F:
    tot = F(0)
    for T in tasks:
        e = F(0)
        for prob, Xs, Ds in r_realizations([T]):
            e += prob * hyb_cp(T, Xs[0], Ds[0], cfg, "opt")
        tot += T.w * (T.r + e)
    return tot


# ======================================================================== 5. sharpened Thm D constant (Prop K)
def lambda_D(tasks: Sequence[HTask], cfg: HCfg) -> F:
    """Per-task ratio max_K Z'_K / (sigma_K + A_K / M) (Thm D sizes; memory bounded)."""
    Z, S, A = h_sizes(tasks, cfg)
    return max(z / (s + a) for z, s, a in zip(Z, S, A))


def thmD_old_constant(tasks, cfg) -> F:
    bcp, b, th = h_constants(tasks, cfg)
    return (1 + bcp) + 2 * max(1 + b, th)


def thmD_new_constant(tasks, cfg) -> F:
    bcp, _, _ = h_constants(tasks, cfg)
    return (1 + bcp) + 2 * lambda_D(tasks, cfg)


# ======================================================================== 6. Thm A lower-bound family
def wspt_sqrt2_instance(L: int, k: int, eps: F = F(1, 1000)):
    """One machine, deterministic: job A (p = L, w = L, ratio 1) at time 0; at time L - 1, k unit jobs of weight
    1 + eps (ratio > 1). P-WSPT preempts A for all k jobs; the schedule 'finish A first' is an upper bound on OPT.
    Returns (tasks, ALG value, UB on OPT)."""
    A = MTask(F(0), F(L), (det(L),))
    us = [MTask(F(L - 1), 1 + eps, (det(1),)) for _ in range(k)]
    alg = (1 + eps) * sum(L - 1 + i for i in range(1, k + 1)) + L * (L + k)
    ub = L * L + (1 + eps) * sum(L + i for i in range(1, k + 1))
    return [A] + us, alg, ub


# ======================================================================== 7. WSEPT vs clairvoyant (Thm P)
def emin(d1: Dist, d2: Dist) -> F:
    """E[min(X, Y)] for independent X ~ d1, Y ~ d2."""
    return sum(p * q * min(a, b) for a, p in d1 for b, q in d2)


def normalize(d: Dist) -> Dist:
    m = mean(d)
    return tuple((v / m, p) for v, p in d)


def emin_lower_bound_holds(d1: Dist, d2: Dist) -> bool:
    """Lemma P1 for mean-1 laws with CV^2 <= Delta: E min(X,Y) >= 16/(27(1+Delta)) and
    E min(X,Y) >= 1 - sqrt(Delta/2) (checked exactly by squaring)."""
    from deadline_agent.stochastic_theory import cv2
    X, Y = normalize(d1), normalize(d2)
    D = max(cv2(X), cv2(Y))
    e = emin(X, Y)
    ok1 = e >= F(16, 27) / (1 + D)
    ok2 = e >= 1 or (1 - e) ** 2 <= D / 2
    return ok1 and ok2


def smith_expected(tasks: Sequence[MTask]) -> F:
    """Smith(E W): the value of the fixed WSEPT task order on one machine (speed 1, no releases)."""
    return smith([T.w for T in tasks], [sum(mean(d) for d in T.nodes) for T in tasks])


def smith_clairvoyant(tasks: Sequence[MTask]) -> F:
    """E[Smith(W(omega))]: the clairvoyant single-machine optimum (speed 1, no releases), task works independent."""
    from deadline_agent.stochastic_theory import realizations
    tot = F(0)
    for prob, P, D in realizations(tasks):
        tot += prob * smith([T.w for T in tasks], [sum(P[t]) for t in range(len(tasks))])
    return tot


def pair_ratio_bound(tasks: Sequence[MTask]) -> F:
    """max over task pairs of min(a, b) / E min(a X, b Y) (a = w_k E W_j, b = w_j E W_k): the exact constant of the
    pairwise argument Smith(E W) <= max(1, this) * E Smith(W)."""
    from deadline_agent.stochastic_theory import realizations
    laws = []
    for T in tasks:
        acc: dict = {}
        for prob, P, D in realizations([T]):
            s = sum(P[0])
            acc[s] = acc.get(s, F(0)) + prob
        laws.append(tuple(sorted(acc.items())))
    best = F(1)
    for j, k in itertools.combinations(range(len(tasks)), 2):
        a, b = tasks[k].w * mean(laws[j]), tasks[j].w * mean(laws[k])
        e = sum(p * q * min(tasks[k].w * x, tasks[j].w * y) for x, p in laws[j] for y, q in laws[k])
        best = max(best, min(a, b) / e)
    return best


# ======================================================================== 8. Thm D with random outputs (Thm DR)
@dataclass(frozen=True)
class CapNode(HNode):
    """HNode whose admission reservation uses a cap xres >= X (reserve l- + xres; decode only X tokens)."""

    xres: int = 0

    @property
    def l_plus(self):
        return self.rho + self.q + max(self.X, self.xres)


@dataclass(frozen=True)
class DRNode:
    q: int
    law: Dist  # output length X, support <= xcap
    rho: int = 0

    @property
    def xcap(self):
        return int(max(v for v, _ in self.law))


@dataclass(frozen=True)
class DRTask:
    r: F
    w: F
    nodes: tuple
    lag: Dist = ((F(0), F(1)),)  # chain plan -> final lag (if 2 nodes)


def dr_realizations(tasks: Sequence[DRTask]):
    """Yield (prob, realized HTask list with CapNodes, lag dicts)."""
    slots = []
    for t, T in enumerate(tasks):
        for i, nd in enumerate(T.nodes):
            slots.append(("x", t, i, nd.law))
        if len(T.nodes) == 2:
            slots.append(("d", t, 0, T.lag))
    for combo in itertools.product(*[s[3] for s in slots]):
        prob = F(1)
        X = [[None] * len(T.nodes) for T in tasks]
        lag = [{} for _ in tasks]
        for s, (v, p) in zip(slots, combo):
            prob *= p
            if s[0] == "x":
                X[s[1]][s[2]] = int(v)
            else:
                lag[s[1]][("a", 0)] = v
        hts = []
        for t, T in enumerate(tasks):
            nodes = tuple(CapNode(q=nd.q, X=X[t][i], rho=nd.rho, xres=nd.xcap) for i, nd in enumerate(T.nodes))
            arcs = ((0, 1, det(0)),) if len(nodes) == 2 else ()
            hts.append(HTask(T.r, T.w, nodes, arcs))
        yield prob, hts, lag


def dr_expected_sizes(tasks: Sequence[DRTask], cfg: HCfg):
    """Returns (E[Z'_K] with cap reservations, marginal laws of s_K = sigma_K + A_K/M, E[Smith(s(omega))]).
    L_max (hence M) uses the cap reservations, so it is the same in every realization."""
    EZ = [F(0)] * len(tasks)
    s_laws = [dict() for _ in tasks]
    esmith = F(0)
    for prob, hts, lag in dr_realizations(tasks):
        Z, S, A = h_sizes(hts, cfg)
        sk = [x + y for x, y in zip(S, A)]
        for t in range(len(tasks)):
            EZ[t] += prob * Z[t]
            s_laws[t][sk[t]] = s_laws[t].get(sk[t], F(0)) + prob
        esmith += prob * smith([T.w for T in tasks], sk)
    return EZ, [tuple(sorted(d.items())) for d in s_laws], esmith


def dr_order(tasks, cfg):
    EZ = dr_expected_sizes(tasks, cfg)[0]
    return sorted(range(len(tasks)), key=lambda t: (-(tasks[t].w / EZ[t]), t))
