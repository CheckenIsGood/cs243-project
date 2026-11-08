"""Theory checks for KV preloading of tool results (docs/preload_theory.md).

Three models, from abstract to concrete:

1. Work model (Section 1 of the doc). One work-conserving server of rate 1 processes *work* (seconds).
   Task j: arrival r, plan work P, n tool calls with durations D_i (issued when the plan completes), result
   pieces w_i, tail work f. `simulate_work` is an exact event-driven simulator: with `fractions.Fraction`
   inputs every time it returns is exact. Disciplines: "fcfs" (preemptive priority by (class, key, seq)) and
   "ps" (egalitarian processor sharing inside the highest non-empty class). Preload modes: off / on / low / jit.
2. Closed forms and approximations (Pollaczek-Khinchine, M/G/1-PS), and the parameter estimates used for the
   data comparison (`Calib`).
3. A step-level continuous-batching engine (`simulate_engine`, floats): chunked prefill with a token budget,
   step time a + c*B + prefill cost, per-task prefix cache with optional LRU KV capacity. It runs the agent
   program of `multiagent.py` (fan-out: plan -> n parallel tools -> final) with preloading modes
   off / block (the version that produced results/real/preload_contention) / nb (current non-blocking code),
   at normal or lowest priority.

Nothing in this file imports the serving stack; it is self-contained so it can be checked independently.
"""

from __future__ import annotations

import csv
import heapq
import itertools
import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

INF = float("inf")


# =============================================================================================== 1. work model
@dataclass(frozen=True)
class TaskSpec:
    """r: arrival; P: plan work (server work for plan_mode work/top, a pure delay for exo); D: tool durations;
    w: result piece work, w[i] belongs to tool i; f: tail work (residual prompt + answer)."""

    r: object
    P: object
    D: tuple
    w: tuple
    f: object

    @property
    def n(self) -> int:
        return len(self.D)


@dataclass
class _Job:
    jid: int
    task: int
    kind: str  # plan | piece | final
    release: object
    work: object  # useful work (without overhead)
    rem: object  # remaining total (overhead is processed first)
    prio: tuple  # (class, key, seq): smaller runs first
    done_at: object = None


@dataclass
class WorkResult:
    completion: list  # C_j: completion of task j's final job
    release_final: list  # R_j: time the final became ready (last tool return)
    plan_done: list
    piece_release: list  # per task: list of (time, work) of preload pieces released
    jobs: list  # all jobs (for audits)
    breakpoints: list  # (t, cumulative work processed) at every event time
    arrivals: list  # (t, work incl. overhead) of every server-job release, in order
    unfinished_at_R: list  # per task: dict with V (total), own (own residual piece work), other_pre (others' pieces)


def _zero_like(x):
    return Fraction(0) if isinstance(x, Fraction) else 0.0


def simulate_work(tasks: Sequence[TaskSpec], mode: str = "off", disc: str = "fcfs", eps=0, plan_mode: str = "exo",
                  merge: bool = True, jit_key: str = "clairvoyant") -> WorkResult:
    """Exact event-driven simulation of the work model.

    mode: off (final work sum(w)+f released at R) | on (each non-last piece released at its return, same class as
    plans/finals) | low (pieces in a lower preemptive class) | jit (all jobs ordered by key: plans/finals by release
    time, pieces by their task's need time R (clairvoyant), FCFS among ties; only with disc='fcfs').
    eps: server overhead per preload piece (processed before the piece's useful work).
    plan_mode: exo (plan completes at r+P, no server work: exogenous releases) | work (plan is a class-0 job:
    endogenous) | top (plan is server work at preemptive top priority).
    merge: at R the task's unfinished piece jobs are removed and their remaining useful work is added to the final
    (the non-blocking code: the pending preload is cancelled; computed KV stays cached). If False, pieces stay and
    the task completes when its final and all its pieces are done.
    """
    if mode not in ("off", "on", "low", "jit") or disc not in ("fcfs", "ps"):
        raise ValueError((mode, disc))
    if mode == "jit" and disc != "fcfs":
        raise ValueError("jit is defined for priority (fcfs-by-key) service only")
    if plan_mode not in ("exo", "work", "top"):
        raise ValueError(plan_mode)
    z = _zero_like(tasks[0].f if tasks else 0)
    eps = Fraction(eps) if isinstance(z, Fraction) else float(eps)
    seq = itertools.count()
    ev: list = []  # (time, order, seq, action, payload)
    for j, tk in enumerate(tasks):
        heapq.heappush(ev, (tk.r, 0, next(seq), "arrive", j))
    active: list[_Job] = []
    jobs: list[_Job] = []
    nT = len(tasks)
    C = [None] * nT
    R = [None] * nT
    PD = [None] * nT
    piece_rel = [[] for _ in range(nT)]
    returns_left = [0] * nT
    final_job: list = [None] * nT
    pieces_of: list[list[_Job]] = [[] for _ in range(nT)]
    pending_need: list = [None] * nT
    unfinished = [None] * nT
    arrivals: list = []
    t = min((tk.r for tk in tasks), default=z)
    processed = z
    bps = [(t, processed)]

    def add_job(j, kind, work, ovh, prio):
        jb = _Job(len(jobs), j, kind, t, work, work + ovh, prio)
        jobs.append(jb)
        active.append(jb)  # zero-work jobs also wait for their turn (FCFS: until all work ahead is done)
        arrivals.append((t, work + ovh))
        return jb

    def task_done_check(j):
        fj = final_job[j]
        if fj is None or fj.done_at is None:
            return
        if not merge and any(p.done_at is None for p in pieces_of[j]):
            return
        if C[j] is None:
            C[j] = max([fj.done_at] + [p.done_at for p in pieces_of[j]]) if not merge else fj.done_at

    def plan_finished(j):
        PD[j] = t
        tk = tasks[j]
        order = sorted(range(tk.n), key=lambda i: (tk.D[i], i))
        returns_left[j] = tk.n
        for rank, i in enumerate(order):
            heapq.heappush(ev, (t + tk.D[i], 1, next(seq), "ret", (j, i, rank == tk.n - 1)))

    def on_event(action, payload):
        nonlocal active
        if action == "arrive":
            j = payload
            tk = tasks[j]
            if plan_mode == "exo":
                heapq.heappush(ev, (t + tk.P, 0, next(seq), "plan_exo", j))
            else:
                cls = -1 if plan_mode == "top" else 0
                add_job(j, "plan", tk.P, z, (cls, t, next(seq)))
        elif action == "plan_exo":
            plan_finished(payload)
        elif action == "ret":
            j, i, last = payload
            tk = tasks[j]
            if not last:
                if mode == "off":
                    return
                piece_rel[j].append((t, tk.w[i]))
                if mode == "on":
                    prio = (0, t, next(seq))
                elif mode == "low":
                    prio = (1, t, next(seq))
                else:  # jit: key = need time of this task (its last return), known to a clairvoyant scheduler
                    prio = (0, pending_need[j], next(seq))
                jb = add_job(j, "piece", tk.w[i], eps, prio)
                pieces_of[j].append(jb)
                task_done_check(j)
                return
            R[j] = t
            V = sum((jb.rem for jb in active), z)
            own = sum((min(jb.rem, jb.work) for jb in pieces_of[j] if jb.done_at is None), z)
            other_pre = sum((min(jb.rem, jb.work) for jb in active if jb.kind == "piece" and jb.task != j), z)
            unfinished[j] = dict(V=V, own=own, other_pre=other_pre)
            if mode == "off":
                work = sum(tk.w, z) + tk.f
            else:
                work = tk.w[i] + tk.f
                if merge:
                    for jb in pieces_of[j]:
                        if jb.done_at is None:
                            work += min(jb.rem, jb.work)
                            jb.done_at = t  # cancelled; its remaining useful work moves into the final
                            jb.rem = z
                    active = [jb for jb in active if jb.done_at is None]
            final_job[j] = add_job(j, "final", work, z, (0, t, next(seq)))
            task_done_check(j)

    def rates():
        if not active:
            return {}
        if disc == "fcfs":
            top = min(active, key=lambda jb: jb.prio)
            return {top.jid: 1}
        cls = min(jb.prio[0] for jb in active)
        grp = [jb for jb in active if jb.prio[0] == cls]
        share = Fraction(1, len(grp)) if isinstance(z, Fraction) else 1.0 / len(grp)
        return {jb.jid: share for jb in grp}

    # jit needs R_j as a key before R_j happens: precompute from plan completion (only exact for plan_mode exo/top
    # because R_j = plan_done + max D and plan_done does not depend on preloads there)
    if mode == "jit":
        if plan_mode == "work":
            raise ValueError("clairvoyant jit needs exogenous plan completion (plan_mode exo or top)")
        if plan_mode == "exo":
            for j, tk in enumerate(tasks):
                pending_need[j] = tk.r + tk.P + max(tk.D)
        else:  # top: plans see only plan work, so their completions can be computed first
            pd = simulate_work(tasks, "off", "fcfs", 0, "top").plan_done
            for j, tk in enumerate(tasks):
                pending_need[j] = pd[j] + max(tk.D)

    tol = 0 if isinstance(z, Fraction) else 1e-12
    while ev or active:
        rt = rates()
        dt_c = None
        for jb in active:
            rr = rt.get(jb.jid, 0)
            if rr:
                d = jb.rem / rr
                dt_c = d if dt_c is None or d < dt_c else dt_c
        t_ext = ev[0][0] if ev else None
        if dt_c is not None and (t_ext is None or t + dt_c <= t_ext):
            step = dt_c
        else:
            step = t_ext - t
        if step:
            for jb in active:
                rr = rt.get(jb.jid, 0)
                if rr:
                    jb.rem -= rr * step
                    if jb.rem <= tol:
                        jb.rem = z
            processed += step if rt else z
            t = t + step
        finished = [jb for jb in active if jb.rem <= tol and rt.get(jb.jid, 0)]
        fin_ids = {jb.jid for jb in finished}
        active = [jb for jb in active if jb.jid not in fin_ids]
        for jb in finished:
            jb.done_at = t
            if jb.kind == "plan":
                plan_finished(jb.task)
        for jb in finished:
            if jb.kind in ("final", "piece"):
                task_done_check(jb.task)
        while ev and ev[0][0] <= t:
            _, _, _, action, payload = heapq.heappop(ev)
            on_event(action, payload)
        bps.append((t, processed))
    return WorkResult(C, R, PD, piece_rel, jobs, bps, arrivals, unfinished)


def cumulative_at(bps: list, x):
    """Cumulative processed work at time x from the breakpoint list (piecewise linear)."""
    prev_t, prev_v = bps[0]
    if x <= prev_t:
        return prev_v
    for tt, vv in bps[1:]:
        if x <= tt:
            if tt == prev_t:
                return vv
            return prev_v + (vv - prev_v) * (x - prev_t) / (tt - prev_t)
        prev_t, prev_v = tt, vv
    return prev_v


def reflection_departures(arrivals: list, x):
    """D(x) = min( A(x), inf_{s<=x} [A(s-) + (x - s)] ) for a rate-1 work-conserving server started empty at
    time 0 (arrivals: (time >= 0, work)). The infimum is attained at s = 0 or at an arrival time."""
    A_x = sum(w for (s, w) in arrivals if s <= x)
    best = A_x
    cands = sorted({s for (s, _) in arrivals if s <= x} | {0 * x})
    for s in cands:
        A_before = sum(w for (u, w) in arrivals if u < s)
        best = min(best, A_before + (x - s))
    return best


def makespan(res: WorkResult):
    return max(c for c in res.completion)


def fcfs_harm_bound(tasks: Sequence[TaskSpec], res_on: WorkResult, j: int, eps=0):
    """Theorem 2 bound: H_j + eps * N(R_j), where H_j = useful work of preload pieces released no later than R_j by
    tasks whose final is behind j's in FCFS order, and N(R_j) = number of preloads (of any task) released no later
    than R_j."""
    Rj = res_on.release_final[j]
    fin = {jb.task: jb for jb in res_on.jobs if jb.kind == "final"}
    H = 0 * Rj
    N = 0
    for k in range(len(tasks)):
        for (tt, w) in res_on.piece_release[k]:
            if tt <= Rj:
                N += 1
                if k != j and fin[k].prio > fin[j].prio:
                    H += w
    return H + eps * N


# ------------------------------------------------------------------------------------- exhaustive / random search
def grid_tasks(r_vals, P_vals, D_vals, w_vals, f_vals, n_vals) -> list[TaskSpec]:
    out = []
    for n in n_vals:
        for r in r_vals:
            for P in P_vals:
                for D in itertools.product(D_vals, repeat=n):
                    for w in itertools.product(w_vals, repeat=n):
                        for f in f_vals:
                            out.append(TaskSpec(Fraction(r), Fraction(P), tuple(map(Fraction, D)),
                                                tuple(map(Fraction, w)), Fraction(f)))
    return out


def search_dominance(pool: Sequence[TaskSpec], k: int, compare: tuple, disc: str, plan_mode: str, eps=0,
                     merge: bool = True, limit: int | None = None):
    """Enumerate all k-tuples (with repetition, ordered by index) from `pool`; return the instance maximising
    max_j (C_j[mode_b] - C_j[mode_a]) and the maximal makespan increase. compare = (mode_a, mode_b): the claim
    tested is 'mode_b is never later than mode_a' (e.g. ('off','low'))."""
    a, b = compare
    worst_task = (None, None)
    worst_mk = (None, None)
    count = 0
    for combo in itertools.combinations_with_replacement(range(len(pool)), k):
        inst = [pool[i] for i in combo]
        ra = simulate_work(inst, a, disc, 0 if a == "off" else eps, plan_mode, merge)
        rb = simulate_work(inst, b, disc, 0 if b == "off" else eps, plan_mode, merge)
        d = max(cb - ca for ca, cb in zip(ra.completion, rb.completion))
        m = makespan(rb) - makespan(ra)
        if worst_task[0] is None or d > worst_task[0]:
            worst_task = (d, inst)
        if worst_mk[0] is None or m > worst_mk[0]:
            worst_mk = (m, inst)
        count += 1
        if limit and count >= limit:
            break
    return dict(count=count, worst_task=worst_task, worst_makespan=worst_mk)


# ----------------------------------------------------------------------------------- named counterexamples (exact)
def cex_endogenous_makespan() -> list[TaskSpec]:
    """Refutes aggregate dominance with endogenous plans (eps = 0, FCFS): A's piece (work 1, returned at 0) is
    ahead of B's plan (released at 1/10) and delays it by 9/10; B's long tool then shifts B's final, which is the
    last work in the system. Makespan off 121/10, on 13."""
    F = Fraction
    A = TaskSpec(F(0), F(0), (F(0), F(1)), (F(1), F(0)), F(1))
    B = TaskSpec(F(1, 10), F(1), (F(10),), (F(0),), F(1))
    return [A, B]


def cex_amplification(delta=Fraction(1, 100)) -> list[TaskSpec]:
    """Per-task harm is not bounded by any multiple of the preload work once plans are server work: a piece of work
    delta delays B's plan by ~delta, B's final is then released after C's plan instead of before it and waits for
    it: B is ~1 later (harm / preload work ~ 1/delta)."""
    F = Fraction
    eta = delta / 4
    A = TaskSpec(F(0), F(0), (F(0), F(50)), (delta, F(0)), F(0))
    B = TaskSpec(eta, F(1), (F(1),), (F(0),), F(10))
    Cc = TaskSpec(F(2) + (eta + delta) / 2, F(1), (F(0),), (F(0),), F(0))
    return [A, B, Cc]


def cex_low_endogenous() -> list[TaskSpec]:
    """Refutes per-task dominance of low-priority preloading when plans are server work (FCFS among plans and finals):
    A's final shrinks from 10 to 1, so B's plan finishes 9 earlier and B's final (work 5) now lands just before J's
    final, which J did not have to wait for without preloading."""
    F = Fraction
    A = TaskSpec(F(-20), F(0), (F(10), F(20)), (F(9), F(0)), F(1))  # piece 9 preloaded in idle time, final at 0
    B = TaskSpec(F(1), F(1), (F(20),), (F(0),), F(5))
    J = TaskSpec(F(45, 2), F(0), (F(0),), (F(0),), F(1))  # final released at 22.5
    return [A, B, J]


def cex_fcfs_on_per_task(delta=Fraction(1, 10)) -> list[TaskSpec]:
    """Tightness of Theorem 2: k's piece (work 2) returns delta before j's final; j has nothing to preload; k's
    final comes much later. Harm to j = 2 - delta (+ eps)."""
    F = Fraction
    j = TaskSpec(F(0), F(0), (F(5),), (F(0),), F(1))
    k = TaskSpec(F(0), F(0), (F(5) - delta, F(30)), (F(2), F(0)), F(1))
    return [j, k]


def cex_ps_on_per_task() -> list[TaskSpec]:
    """Per-task dominance also fails under processor sharing: k's piece shares the server with j's final."""
    F = Fraction
    j = TaskSpec(F(0), F(0), (F(5),), (F(0),), F(2))
    k = TaskSpec(F(0), F(0), (F(5), F(30)), (F(2), F(0)), F(1))
    return [j, k]


def cex_blocking_low() -> dict:
    """Blocking preload chain at lowest priority: the final waits for its own starved preload. Returned as a
    closed-form instance description (checked in tests with the engine simulator)."""
    return dict(note="see test_blocking_low_priority_starves")


# ============================================================================== 2. closed forms and calibration
def pk_wait(lam: float, ES: float, ES2: float) -> float:
    """Pollaczek-Khinchine mean waiting time of M/G/1-FCFS."""
    rho = lam * ES
    return INF if rho >= 1 else lam * ES2 / (2 * (1 - rho))


def ps_response(x: float, rho: float) -> float:
    """M/G/1-PS mean response time of a job of size x (insensitive to the service distribution)."""
    return INF if rho >= 1 else x / (1 - rho)


@dataclass
class Calib:
    """Parameters for the A40 / Qwen3-8B / vLLM 0.32 runs. 'measured' = from repo data; 'estimated' = fitted here
    (see docs/preload_theory.md Section 6 for the derivation); 'assumed' = not observable in the local data."""

    a: float = 0.027767  # measured: decode step fixed cost (batch_profile.json)
    c: float = 0.000543  # measured: per-sequence decode cost
    alpha: float = 1.140e-4  # estimated: prefill s/token (fit of 131 preloads in results/real/preload-load traces)
    kappa: float = 1.227e-8  # estimated: attention s/(new token x context token)
    delta: float = 0.049  # estimated: per-request non-GPU latency (fit intercept 0.0766 minus one step a)
    header: int = 110  # estimated: system prompt + question tokens of the evidence log
    block_hdr: int = 35  # estimated: per-result header tokens ("[c3] search(...)")
    suffix: int = 15  # estimated: chat-template tail that differs between preload and final requests
    plan_prompt: int = 89  # measured (traces)
    plan_out_wide: tuple = (127.2, 28.3)  # measured (traces, 8-query plans): mean, sd
    plan_out_narrow: int = 25  # ASSUMED: 1-query plans (not observable locally)
    final_out: int = 10  # measured (traces mean 9.6)
    result_cap: int = 1400  # preset "preload"
    delay: tuple = (0.5, 4.0, 0.3, 4.0, 12.0)  # DelayModel used by the run

    def prefill_cost(self, new: float, ctx: float) -> float:
        """GPU seconds to prefill `new` tokens after `ctx` cached tokens (no per-step overhead)."""
        return self.alpha * new + self.kappa * new * (ctx + new / 2)


# empirical result sizes (approx-token counts) of the 160 tool returns in results/real/preload-load (cap 1500).
# Loaded lazily from the traces when present; a fixed fallback keeps tests deterministic without the data.
_RESULT_TOKENS: list[int] | None = None


def result_token_samples(root: str | Path = ".") -> list[int]:
    global _RESULT_TOKENS
    if _RESULT_TOKENS is not None:
        return _RESULT_TOKENS
    import glob
    import json
    vals = []
    for f in sorted(glob.glob(str(Path(root) / "results/real/preload-load/traces/*__fold__*.jsonl"))):
        for line in open(f):
            d = json.loads(line)
            if d.get("event") == "tool_return":
                vals.append(int(d["tokens"]))
    _RESULT_TOKENS = vals or [1400]
    return _RESULT_TOKENS


def sample_delay(rng: random.Random, dm=Calib.delay) -> float:
    lo, hi, p, slo, shi = dm
    if p > 0 and rng.random() < p:
        return slo + rng.random() * (shi - slo)
    return lo + rng.random() * (hi - lo)


def delay_moments(dm=Calib.delay):
    lo, hi, p, slo, shi = dm
    m1 = (1 - p) * (lo + hi) / 2 + p * (slo + shi) / 2
    return m1


def expected_max_delay(n: int, dm=Calib.delay, grid: int = 20000) -> float:
    lo, hi, p, slo, shi = dm
    top = max(hi, shi)

    def F(x):
        u = 0.0 if x < lo else (1.0 if x >= hi else (x - lo) / (hi - lo))
        v = 0.0 if x < slo else (1.0 if x >= shi else (x - slo) / (shi - slo))
        return (1 - p) * u + p * v
    h = top / grid
    return sum((1 - F((k + 0.5) * h) ** n) * h for k in range(grid))


def task_work_draw(rng: random.Random, n: int, cal: Calib = Calib(), tokens: Sequence[int] | None = None):
    """One task's prefill work in GPU seconds: (pieces cost list in return order excluding the last, total final
    cost when nothing is preloaded, residual final cost when all but the last piece are preloaded)."""
    tokens = tokens or result_token_samples()
    sizes = [min(rng.choice(tokens), cal.result_cap) + cal.block_hdr for _ in range(n)]
    ctx = cal.header
    pieces = []
    for s in sizes[:-1]:
        pieces.append(cal.prefill_cost(s, ctx))
        ctx += s
    last = cal.prefill_cost(sizes[-1] + cal.suffix, ctx)
    W = cal.prefill_cost(cal.header + sum(sizes) + cal.suffix, 0)
    return pieces, W, last


def workload_moments(q_wide: float, n_wide: int = 8, cal: Calib = Calib(), samples: int = 4000, seed: int = 0,
                     plan_decode_work: bool = True):
    """Monte Carlo moments of the per-task GPU work (prefill seconds; plus c per decode token if plan_decode_work).
    Returns dict with E[S] (task work), E[W], E[W^2] of final bursts (off), mean preloaded work per wide task,
    E[w], E[w^2] of pieces, pieces per task."""
    rng = random.Random(seed)
    S, W1, W2, PW, w1, w2, npcs, last1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in range(samples):
        wide = rng.random() < q_wide
        n = n_wide if wide else 1
        pieces, W, last = task_work_draw(rng, n, cal)
        plan = cal.prefill_cost(cal.plan_prompt, 0)
        dec = cal.c * ((cal.plan_out_wide[0] if wide else cal.plan_out_narrow) + cal.final_out) if plan_decode_work \
            else 0.0
        S += W + plan + dec
        W1 += W
        W2 += W * W
        PW += sum(pieces) if wide else 0.0
        for p_ in pieces:
            w1 += p_
            w2 += p_ * p_
        npcs += len(pieces)
        last1 += last
    k = samples
    return dict(ES=S / k, EW=W1 / k, EW2=W2 / k, preloaded_per_task=PW / k, pieces_per_task=npcs / k,
                Ew=w1 / max(npcs, 1), Ew2=w2 / max(npcs, 1), Elast=last1 / k)


def predict_ps(lam: float, q_wide: float, cal: Calib = Calib(), eps_gpu: float = 0.0, **kw) -> dict:
    """M/G/1-PS approximation (requests treated as Poisson arrivals; preloads finish before R):
    saving(wide) = W/(1-rho_off) - last/(1-rho_on), saving(narrow) = x/(1-rho_off) - x/(1-rho_on) ~ 0."""
    m = workload_moments(q_wide, cal=cal, **kw)
    m_wide = workload_moments(1.0, cal=cal, **kw)
    m_n = workload_moments(0.0, cal=cal, **kw)
    rho_off = lam * m["ES"]
    rho_on = rho_off + lam * eps_gpu * m["pieces_per_task"]
    if rho_on >= 1:
        return dict(rho_off=rho_off, rho_on=rho_on, wide=INF, narrow=INF, mean=INF)
    wide = ps_response(m_wide["EW"], rho_off) - ps_response(m_wide["Elast"], rho_on)
    narrow = ps_response(m_n["EW"], rho_off) - ps_response(m_n["EW"], rho_on)
    return dict(rho_off=rho_off, rho_on=rho_on, wide=wide, narrow=narrow, mean=q_wide * wide + (1 - q_wide) * narrow)


def predict_fcfs(lam: float, q_wide: float, cal: Calib = Calib(), **kw) -> dict:
    """M/G/1-FCFS heuristic: off = PK with final bursts W (plans negligible); on = PK with pieces and finals as
    independent Poisson job streams (a heuristic: real pieces are clustered). saving(narrow) = Wq_off - Wq_on,
    saving(wide) = preloaded work + Wq_off - Wq_on."""
    m = workload_moments(q_wide, cal=cal, **kw)
    m_wide = workload_moments(1.0, cal=cal, **kw)
    ES = m["ES"]
    rho = lam * ES
    wq_off = pk_wait(lam, m["EW"], m["EW2"])
    # on: per task, pieces (count p) + final residual; the residual moments are approximated by the piece moments
    k = m["pieces_per_task"] + 1
    E_on = m["EW"] / k
    E2_on = (m["pieces_per_task"] * m["Ew2"] + m["Elast"] ** 2) / k
    wq_on = INF if rho >= 1 else lam * k * E2_on / (2 * (1 - rho))
    dq = wq_off - wq_on
    return dict(rho=rho, wide=m_wide["preloaded_per_task"] + dq, narrow=dq,
                mean=q_wide * (m_wide["preloaded_per_task"] + dq) + (1 - q_wide) * dq, E_on=E_on)


def breakeven_eps_ps(w: float, rho: float, lam: float, xcrit: float) -> float:
    """Theorem 6(b): under the PS approximation, removing one piece of work w from the critical path of a task whose
    mean critical-path work is xcrit, at the cost of eps GPU work, lowers mean latency iff eps < w(1-rho)/(lam*xcrit)
    (to first order: x/(1-rho) vs (x-w)/(1-rho-lam*eps))."""
    return w * (1 - rho) / (lam * xcrit)


def kv_littles_law(lam: float, kv_seconds_per_task: float) -> float:
    """Mean KV tokens held = lam * E[integral of a task's KV footprint over time] (Little's law)."""
    return lam * kv_seconds_per_task


def preload_kv_seconds(n: int = 8, cal: Calib = Calib(), samples: int = 20000, seed: int = 0) -> float:
    """E[integral over the tool wait of a task's preloaded-but-unused KV] in token-seconds, assuming each preload
    completes at its result's return (isolated task): after the k-th return the cached log holds header + k results
    until the last return T_(n)."""
    rng = random.Random(seed)
    toks = result_token_samples()
    acc = 0.0
    for _ in range(samples):
        d = sorted(sample_delay(rng, cal.delay) for _ in range(n))
        sizes = [min(rng.choice(toks), cal.result_cap) + cal.block_hdr for _ in range(n)]
        logl = cal.header
        for k in range(n - 1):
            logl += sizes[k]
            acc += logl * (d[k + 1] - d[k])
    return acc / samples


# ================================================================================ 3. step-level engine simulator
@dataclass
class EngineCfg:
    budget: int | None = 2048  # max tokens per step (decode tokens count against it); None = unchunked
    chunk: str = "fcfs"  # fcfs: prefills get the budget in priority/arrival order; fair: equal split
    kv_capacity: int | None = None  # tokens; None = unlimited
    block: int = 16


@dataclass
class _Req:
    rid: int
    task: int
    kind: str  # plan | preload | final
    prompt: int
    out: int
    prio: float
    t_arr: float
    cached: int = 0
    done_pf: int = 0  # prompt tokens computed (including cached)
    gen: int = 0
    finished: float | None = None
    cancelled: bool = False


@dataclass
class TaskIn:
    arrival: float
    n: int  # number of tool calls (width)
    delays: tuple
    sizes: tuple  # result tokens (with block header), per tool
    plan_out: int


@dataclass
class EngineOut:
    latency: list
    final_ready: list  # last tool return time (absolute)
    finish: list
    steps: int
    busy: float
    preloads: list  # per task: number of preload requests issued
    evicted_preloaded: int  # preloaded tokens evicted before the final used them (recomputed)
    kv_peak: int
    kv_violations: dict  # trajectory checks of Lemma K (necessary / LRU-sufficient condition)
    pi_at_R: list = field(default_factory=list)  # per task: other tasks' unprocessed preload prefill (GPU s) at R_j


def make_tasks(arrivals: Sequence[float], widths: Sequence[int], seed: int, cal: Calib = Calib()) -> list[TaskIn]:
    rng = random.Random(seed)
    toks = result_token_samples()
    out = []
    for t, n in zip(arrivals, widths):
        d = tuple(sample_delay(rng, cal.delay) for _ in range(n))
        s = tuple(min(rng.choice(toks), cal.result_cap) + cal.block_hdr for _ in range(n))
        po = cal.plan_out_narrow if n == 1 else max(40, int(round(rng.gauss(*cal.plan_out_wide))))
        out.append(TaskIn(t, n, d, s, po))
    return out


def simulate_engine(tasks: Sequence[TaskIn], mode: str = "off", low: bool = False, cal: Calib = Calib(),
                    cfg: EngineCfg = EngineCfg(), horizon: float = 1e6) -> EngineOut:
    """mode: off | block (preload k issued at max(k-th return, completion of preload k-1); final after the chain,
    as in the code that produced results/real/preload_contention) | nb (non-blocking: at most one preload in
    flight, cancelled when the final is ready: current multiagent.py). low: preloads at the lowest priority.

    Engine step: every running request past its prefill decodes one token; the token budget minus the decode
    tokens goes to prefilling requests in (priority, admission) order ('fcfs') or split equally ('fair');
    step time = a + c*(#decoding) + sum of prefill chunk costs. Requests reach the engine delta/2 after they are
    sent and their result reaches the agent delta/2 after the step that finishes them. Prefix cache: one cached
    evidence-log prefix per task (full blocks); with kv_capacity, unpinned caches are evicted LRU (tail first)."""
    assert mode in ("off", "block", "nb")
    B = cfg.block
    M = cfg.kv_capacity
    nT = len(tasks)
    rid = itertools.count()
    sq = itertools.count()
    ev: list = []  # (time, seq, kind, payload)
    running: list[_Req] = []
    K = [0] * nT  # cached prefix tokens of task j's evidence log (full blocks)
    K_access = [0.0] * nT
    pinned_by = [0] * nT  # running log requests of task j
    pre_total = [0] * nT  # tokens computed by preloads of task j (never reduced; for Lemma K (i))
    pre_live = [0] * nT  # preloaded tokens still cached
    stale: list = []  # [last_access, tokens] of finished plan caches
    evicted_pre = 0
    lat = [None] * nT
    fin = [None] * nT
    ready = [None] * nT
    npre = [0] * nT
    pi = [0.0] * nT
    st = [dict(returned=0, pending=None, chain_busy=False, next_k=1, final_sent=False) for _ in range(nT)]
    viol = dict(steps=0, necessary_flagged=0, necessary_without_eviction=0, evicted_while_sufficient=0)
    kv_peak = 0
    orders = [sorted(range(tk.n), key=lambda i: (tk.delays[i], i)) for tk in tasks]

    def log_len(j, k):
        return cal.header + sum(tasks[j].sizes[i] for i in orders[j][:k])

    def send(j, kind, prompt, out, prio, now):
        r = _Req(next(rid), j, kind, prompt, out, prio, now + cal.delta / 2)
        heapq.heappush(ev, (r.t_arr, next(sq), "admit", r))
        if kind == "preload":
            npre[j] += 1
        return r

    pre_prio = INF if low else 0.0

    def send_final(j, now):
        st[j]["final_sent"] = True
        send(j, "final", log_len(j, tasks[j].n) + cal.suffix, cal.final_out, 0.0, now)

    def advance_chain(j, now):
        s = st[j]
        n = tasks[j].n
        if s["chain_busy"] or s["final_sent"]:
            return
        k = s["next_k"]
        if k < n and s["returned"] >= k:
            s["chain_busy"] = True
            s["next_k"] = k + 1
            s["pending"] = send(j, "preload", log_len(j, k), 1, pre_prio, now)
        elif k >= n and s["returned"] == n:
            send_final(j, now)

    def on_task_event(kind, payload, now):
        if kind == "arrive":
            send(payload, "plan", cal.plan_prompt, tasks[payload].plan_out, 0.0, now)
        elif kind == "ret":
            j = payload
            s = st[j]
            s["returned"] += 1
            k, n = s["returned"], tasks[j].n
            if k == n:
                ready[j] = now
                pi[j] = sum(cal.prefill_cost(r.prompt - r.done_pf, r.done_pf) for r in running
                            if r.kind == "preload" and r.task != j and not r.cancelled)
            if mode == "off":
                if k == n:
                    send_final(j, now)
            elif mode == "nb":
                if k < n and (s["pending"] is None or s["pending"].finished is not None):
                    s["pending"] = send(j, "preload", log_len(j, k), 1, pre_prio, now)
                if k == n:
                    p = s["pending"]
                    if p is not None and p.finished is None:
                        p.cancelled = True
                    send_final(j, now)
            else:
                advance_chain(j, now)
        elif kind == "done":
            r = payload
            j = r.task
            if r.kind == "plan":
                for d in tasks[j].delays:
                    heapq.heappush(ev, (now + d, next(sq), "ret", j))
            elif r.kind == "final":
                fin[j] = now
                lat[j] = now - tasks[j].arrival
            elif mode == "block":
                st[j]["chain_busy"] = False
                advance_chain(j, now)

    for j, tk in enumerate(tasks):
        heapq.heappush(ev, (tk.arrival, next(sq), "arrive", j))

    def pinned_now(extra):
        u = extra
        for j in range(nT):
            if pinned_by[j]:
                u += K[j]
        for r in running:
            u += (r.done_pf + r.gen) if r.kind == "plan" else (r.gen + max(0, r.done_pf - K[r.task]))
        return u

    def kv_used():
        return sum(K[j] for j in range(nT) if not pinned_by[j]) + sum(x[1] for x in stale) + pinned_now(0)

    def waiting(j):
        return not pinned_by[j] and fin[j] is None and K[j] > 0

    def evict(need, demand):
        """LRU eviction of unpinned caches (tail blocks first) until `need` tokens are freed; `demand` is the
        step's full new-token demand (for the Lemma K (ii) check)."""
        nonlocal evicted_pre
        # LRU by last access; among equal timestamps, caches nobody waits for go first
        cands = [(K_access[j], 1 if waiting(j) else 0, 0, j) for j in range(nT) if K[j] > 0 and not pinned_by[j]]
        cands += [(x[0], 0, 1, i) for i, x in enumerate(stale) if x[1] > 0]
        cands.sort()
        # Lemma K (ii): pinned + waiting preloaded + unpinned caches touched after the oldest waiting one <= M
        oldest = min([K_access[j] for j in range(nT) if waiting(j)], default=INF)
        recent = sum(x[1] for x in stale if x[0] > oldest) + sum(
            K[j] for j in range(nT) if not pinned_by[j] and fin[j] is not None and K_access[j] > oldest)
        sufficient = pinned_now(demand) + sum(K[j] for j in range(nT) if waiting(j)) + recent <= M
        freed = 0
        lost = 0
        for _, _, typ, idx in cands:
            if freed >= need:
                break
            if typ == 0:
                newK = max(0, K[idx] - (need - freed)) // B * B
                freed += K[idx] - newK
                if fin[idx] is None and pre_live[idx] > newK:
                    lost += pre_live[idx] - newK
                    pre_live[idx] = newK
                K[idx] = newK
            else:
                take = min(stale[idx][1], need - freed)
                stale[idx][1] -= take
                freed += take
        evicted_pre += lost
        if lost and sufficient:
            viol["evicted_while_sufficient"] += 1
        return freed

    t = 0.0
    steps = 0
    busy = 0.0
    while ev or running:
        if t > horizon:
            break
        while ev and ev[0][0] <= t + 1e-12:
            et, _, kind, payload = heapq.heappop(ev)
            if kind == "admit":
                r = payload
                if r.cancelled:
                    continue
                if r.kind != "plan":
                    r.cached = min(K[r.task] // B * B, r.prompt - 1)
                    r.done_pf = r.cached
                    pinned_by[r.task] += 1
                    K_access[r.task] = t
                running.append(r)
            else:
                on_task_event(kind, payload, et)
        for r in [r for r in running if r.cancelled]:  # computed full blocks stay cached
            running.remove(r)
            pinned_by[r.task] -= 1
        if not running:
            if ev:
                t = max(t, ev[0][0])
                continue
            break
        dec = [r for r in running if r.done_pf >= r.prompt]
        pf = sorted([r for r in running if r.done_pf < r.prompt], key=lambda r: (r.prio, r.t_arr, r.rid))
        budget = INF if cfg.budget is None else max(0, cfg.budget - len(dec))
        alloc: dict = {}
        if cfg.chunk == "fcfs" or budget == INF:
            left = budget
            for r in pf:
                if left <= 0:
                    break
                x = min(r.prompt - r.done_pf, left)
                alloc[r.rid] = x
                left -= x
        else:  # fair: water-filling inside each priority class, best class first
            left = budget
            for _, grp in itertools.groupby(pf, key=lambda r: r.prio):
                g = list(grp)
                while left > 0 and g:
                    share = max(1, left // len(g))
                    nxt = []
                    for r in g:
                        need = r.prompt - r.done_pf - alloc.get(r.rid, 0)
                        x = min(need, share, left)
                        alloc[r.rid] = alloc.get(r.rid, 0) + x
                        left -= x
                        if need > x:
                            nxt.append(r)
                    g = nxt
        if M is not None:
            need = sum(alloc.values()) + len(dec)
            free = M - kv_used()
            if need > free:
                free += evict(need - free, need)
            if need > free:  # still short: shrink prefill chunks, latest first (requests stall)
                over = need - free
                for r in reversed(pf):
                    cut = min(alloc.get(r.rid, 0), over)
                    if cut:
                        alloc[r.rid] -= cut
                        over -= cut
                    if over <= 0:
                        break
                if not dec and not any(alloc.values()):
                    # no progress possible: preempt the latest prefilling request (vLLM recompute preemption);
                    # its computed full blocks stay cached (evictable) and it is re-admitted later
                    victim = pf[-1]
                    running.remove(victim)
                    if victim.kind != "plan":
                        pinned_by[victim.task] -= 1
                    victim.done_pf = 0
                    victim.t_arr = t + cal.a
                    heapq.heappush(ev, (victim.t_arr, next(sq), "admit", victim))
                    viol["preemptions"] = viol.get("preemptions", 0) + 1
                    t += cal.a
                    continue
            # Lemma K (i): pinned + all preloaded-and-waiting tokens > M  =>  some preloaded token was evicted
            viol["steps"] += 1
            need_now = pinned_now(sum(alloc.values()) + len(dec)) + sum(pre_total[j] for j in range(nT)
                                                                      if not pinned_by[j] and fin[j] is None)
            viol["peak_need"] = max(viol.get("peak_need", 0), need_now)
            viol["need_x_dt"] = viol.get("need_x_dt", 0.0)  # filled below (time average)
            if need_now > M:
                viol["necessary_flagged"] += 1
                if evicted_pre == 0:
                    viol["necessary_without_eviction"] += 1
        dt = cal.a + cal.c * len(dec)
        for r in pf:
            x = alloc.get(r.rid, 0)
            if x:
                dt += cal.prefill_cost(x, r.done_pf)
        t += dt
        steps += 1
        busy += dt
        if M is not None:
            viol["need_x_dt"] += need_now * dt
        for r in pf:
            x = alloc.get(r.rid, 0)
            if not x:
                continue
            r.done_pf += x
            if r.kind != "plan":
                newK = r.done_pf // B * B
                if newK > K[r.task]:
                    K[r.task] = newK
                K_access[r.task] = t
                if r.kind == "preload":
                    pre_total[r.task] = max(pre_total[r.task], newK)
                    pre_live[r.task] = max(pre_live[r.task], newK)
            if r.done_pf >= r.prompt:
                r.gen = 1  # the step that completes the prefill emits the first token
        for r in dec:
            r.gen += 1
        if M is not None:
            kv_peak = max(kv_peak, kv_used())
        for r in [r for r in running if r.gen >= r.out]:
            running.remove(r)
            r.finished = t
            if r.kind == "plan":
                stale.append([t, r.prompt + r.gen])
            else:
                pinned_by[r.task] -= 1
                K_access[r.task] = t
            heapq.heappush(ev, (t + cal.delta / 2, next(sq), "done", r))
    return EngineOut(lat, ready, fin, steps, busy, npre, evicted_pre, kv_peak, viol, pi)


# ------------------------------------------------------------------------------------------- data comparison
DATA_DIR = Path("results/real/preload_contention")
RATES = ("0.15", "0.3", "0.45", "0.6")


def load_contention(root: str | Path = ".") -> dict:
    """Per rate: task ids (sorted), arrivals, widths (from 'on' model_calls: n = calls - 1; 2 calls -> width 1),
    latencies per mode."""
    base = Path(root) / DATA_DIR
    out = {}
    for r in RATES:
        d = {}
        for m in ("off", "on", "low"):
            with open(base / f"tasks_r{r}_fcfs_pre-{m}.csv") as f:
                d[m] = {row["task_id"]: row for row in csv.DictReader(f)}
        ids = sorted(d["off"], key=lambda i: float(d["off"][i]["arrival"]))
        widths = [max(1, int(d["on"][i]["model_calls"]) - 1) for i in ids]
        out[r] = dict(ids=ids, arrival=[float(d["off"][i]["arrival"]) for i in ids], width=widths,
                      lat={m: [float(d[m][i]["latency"]) for i in ids] for m in d},
                      calls={m: [int(d[m][i]["model_calls"]) for i in ids] for m in d})
    return out


def bootstrap_ci(x: Sequence[float], reps: int = 4000, seed: int = 0, level: float = 0.95):
    rng = random.Random(seed)
    n = len(x)
    ms = sorted(sum(rng.choices(x, k=n)) / n for _ in range(reps))
    lo = ms[int((1 - level) / 2 * reps)]
    hi = ms[int((1 + level) / 2 * reps) - 1]
    return sum(x) / n, lo, hi


def measured_savings(data: dict, a: str = "off", b: str = "on") -> dict:
    out = {}
    for r, d in data.items():
        diff = [x - y for x, y in zip(d["lat"][a], d["lat"][b])]
        wide = [x for x, w in zip(diff, d["width"]) if w > 1]
        narrow = [x for x, w in zip(diff, d["width"]) if w == 1]
        out[r] = dict(all=bootstrap_ci(diff), wide=bootstrap_ci(wide), narrow=bootstrap_ci(narrow),
                      q_wide=len(wide) / len(diff), n=len(diff))
    return out


def gap_ci(data: dict, r: str, reps: int = 4000, seed: int = 0):
    """Bootstrap CI of mean saving(wide) - mean saving(narrow) (groups resampled independently)."""
    d = data[r]
    diff = [x - y for x, y in zip(d["lat"]["off"], d["lat"]["on"])]
    wide = [x for x, w in zip(diff, d["width"]) if w > 1]
    nar = [x for x, w in zip(diff, d["width"]) if w == 1]
    rng = random.Random(seed)
    g = sorted(sum(rng.choices(wide, k=len(wide))) / len(wide) - sum(rng.choices(nar, k=len(nar))) / len(nar)
               for _ in range(reps))
    return sum(wide) / len(wide) - sum(nar) / len(nar), g[int(0.025 * reps)], g[int(0.975 * reps) - 1]


def predict_engine(data: dict, rate: str, modes=(("off", False), ("block", False), ("block", True)), seeds=range(8),
                   cal: Calib = Calib(), cfg: EngineCfg = EngineCfg()) -> dict:
    """Run the step engine on the measured arrival times and widths, with tool delays and result sizes sampled from
    the run's distributions (seeds), and return mean paired savings vs off, overall and by width group."""
    d = data[rate]
    acc = {m: dict(all=[], wide=[], narrow=[]) for m in modes if m[0] != "off"}
    off_lat = []
    for s in seeds:
        tasks = make_tasks(d["arrival"], d["width"], seed=1000 + s, cal=cal)
        res = {m: simulate_engine(tasks, m[0], m[1], cal, cfg) for m in modes}
        off = res[("off", False)].latency
        for m in acc:
            diff = [x - y for x, y in zip(off, res[m].latency)]
            acc[m]["all"].append(sum(diff) / len(diff))
            w = [x for x, n in zip(diff, d["width"]) if n > 1]
            nn = [x for x, n in zip(diff, d["width"]) if n == 1]
            acc[m]["wide"].append(sum(w) / max(1, len(w)))
            acc[m]["narrow"].append(sum(nn) / max(1, len(nn)))
        off_lat.append(sum(off) / len(off))
    out = {"off_lat": sum(off_lat) / len(off_lat)}
    for m, v in acc.items():
        out[m] = {g: (sum(x) / len(x), min(x), max(x)) for g, x in v.items()}
    return out


def work_tasks_from(tasks: Sequence[TaskIn], cal: Calib = Calib()) -> list[TaskSpec]:
    """Map engine tasks to the work model: plan = pure delay (batch-1 plan latency), piece work = prefill cost at its
    context, tail = last piece + suffix + final decode work."""
    out = []
    for tk in tasks:
        order = sorted(range(tk.n), key=lambda i: (tk.delays[i], i))
        ctx = cal.header
        w = [0.0] * tk.n
        for i in order:
            w[i] = cal.prefill_cost(tk.sizes[i], ctx)
            ctx += tk.sizes[i]
        f = cal.prefill_cost(cal.suffix, ctx) + cal.c * cal.final_out
        plan = cal.delta + tk.plan_out * (cal.a + cal.c) + cal.prefill_cost(cal.plan_prompt, 0)
        out.append(TaskSpec(tk.arrival, plan, tuple(tk.delays), tuple(w), f))
    return out


def predict_work(data: dict, rate: str, disc: str, seeds=range(8), eps: float = 0.0, cal: Calib = Calib()) -> dict:
    d = data[rate]
    res = dict(all=[], wide=[], narrow=[], low_all=[])
    for s in seeds:
        tasks = work_tasks_from(make_tasks(d["arrival"], d["width"], seed=1000 + s, cal=cal), cal)
        off = simulate_work(tasks, "off", disc, 0, "exo").completion
        on = simulate_work(tasks, "on", disc, eps, "exo").completion
        diff = [x - y for x, y in zip(off, on)]
        res["all"].append(sum(diff) / len(diff))
        w = [x for x, n in zip(diff, d["width"]) if n > 1]
        nn = [x for x, n in zip(diff, d["width"]) if n == 1]
        res["wide"].append(sum(w) / max(1, len(w)))
        res["narrow"].append(sum(nn) / max(1, len(nn)))
    return {g: (sum(x) / len(x), min(x), max(x)) for g, x in res.items() if x}


def microbench_fit(root: str | Path = ".") -> dict:
    """Least-squares fit of the single-task microbenchmark: final latency saving (cold - preloaded, paired by k, rep)
    against the number of cached tokens: saving = slope * cached + intercept."""
    rows = list(csv.DictReader(open(Path(root) / "results/preload_bench/preload_bench.csv")))
    cold = {(r["k"], r["rep"]): float(r["final_latency_s"]) for r in rows if r["mode"] == "cold"}
    pts = []
    for r in rows:
        if r["mode"] == "preloaded" and (r["k"], r["rep"]) in cold:
            pts.append((float(r["final_cached_tokens"]), cold[(r["k"], r["rep"])] - float(r["final_latency_s"]),
                        int(r["preload_requests"]), float(r["preload_total_s"])))
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    slope = sxy / sxx
    return dict(n=n, slope=slope, intercept=my - slope * mx, points=pts)
