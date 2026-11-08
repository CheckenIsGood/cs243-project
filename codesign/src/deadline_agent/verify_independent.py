"""Independent verification machinery for docs/general_theory.md and docs/open_problem.md.

Written from the definitions only. Nothing here imports or reuses `general_theory.py` or `open_problem.py`
(tests that compare against those modules are separate and clearly labelled as cross-checks).
Every scheduling quantity is an exact `fractions.Fraction`; no floating point enters any decision.

Contents
- Model: `Task` = DAG of GPU nodes (work at speed 1) and TOOL nodes (fixed duration, off-GPU, never sped up,
  unlimited parallelism, started the instant they are enabled).
- `simulate`: exact event-driven simulator. A *rate policy* maps the ready GPU nodes to rates; helpers build
  global priority scheduling on m machines of speed s (EDF, CP-EDF, FIFO, static keys), processor sharing, and
  the fluid continuous-batching engine (rate lam/(a + c*B) per sequence). Tool durations may come from an
  adaptive oracle that only sees issue events.
- Two independent exact offline feasibility tests:
  * `offline_1m_by_orders`: one machine. Exhaustive over static priority orders of the GPU nodes. Exactness
    argument (proved in docs/verification_report.md, Sec. 3): if any feasible schedule exists, preemptive static
    priority by its node completion times is feasible.
  * `offline_mm_lp`: m machines, Z3 over the reals with an elementary-interval (time-indexed with rational
    variable breakpoints) encoding; every SAT answer is turned into an explicit McNaughton schedule and
    replayed by `validate_schedule`.
- Lemma 2 machinery: exact McNaughton wrap-around, exact fluid engine, brute-force discrete-step engine.
- Theorem 1 machinery: knapsack DP, brute force 1||sum w_j U_j, brute force engine makespan with batching.
- Instance generators (planted feasible instances with the release-order condition) and adversaries.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Callable, Iterable

import z3

GPU, TOOL = "G", "T"
ZERO, ONE = F(0), F(1)


def fr(x) -> F:
    if isinstance(x, F):
        return x
    if isinstance(x, float):
        raise TypeError("floats are not allowed; pass Fraction/int/str")
    return F(x)


# ============================================================================================== model
@dataclass(frozen=True)
class Node:
    kind: str
    size: F
    preds: tuple[int, ...] = ()


@dataclass
class Task:
    tid: int
    release: F
    deadline: F
    nodes: tuple[Node, ...]
    value: F = ONE

    def succ(self) -> list[list[int]]:
        out: list[list[int]] = [[] for _ in self.nodes]
        for v, nd in enumerate(self.nodes):
            for u in nd.preds:
                out[u].append(v)
        return out

    def topo(self) -> list[int]:
        indeg = [len(nd.preds) for nd in self.nodes]
        succ = self.succ()
        order, stack = [], [v for v in range(len(self.nodes)) if indeg[v] == 0]
        while stack:
            v = stack.pop()
            order.append(v)
            for w in succ[v]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    stack.append(w)
        if len(order) != len(self.nodes):
            raise ValueError("cycle")
        return order

    def copy(self, **kw) -> "Task":
        d = dict(tid=self.tid, release=self.release, deadline=self.deadline, nodes=self.nodes, value=self.value)
        d.update(kw)
        return Task(**d)


def mk_chain(tid, release, deadline, stages, value=1) -> Task:
    nodes = tuple(Node(k, fr(x), (i - 1,) if i else ()) for i, (k, x) in enumerate(stages))
    return Task(tid, fr(release), fr(deadline), nodes, fr(value))


def node_len(nd: Node, speed: F) -> F:
    return nd.size / speed if nd.kind == GPU else nd.size


def crit_path(task: Task, speed=ONE) -> F:
    speed = fr(speed)
    best: dict[int, F] = {}
    for v in task.topo():
        nd = task.nodes[v]
        best[v] = node_len(nd, speed) + max((best[u] for u in nd.preds), default=ZERO)
    return max(best.values(), default=ZERO)


def tails(task: Task, speed=ONE) -> list[F]:
    """tails[v] = longest path strictly after v (so the node's latest completion time is d - tails[v])."""
    speed = fr(speed)
    succ = task.succ()
    out = [ZERO] * len(task.nodes)
    for v in reversed(task.topo()):
        out[v] = max((node_len(task.nodes[w], speed) + out[w] for w in succ[v]), default=ZERO)
    return out


def slack_ok(task: Task, delta, speed=ONE) -> bool:
    return task.deadline - task.release >= (1 + fr(delta)) * crit_path(task, speed)


def gpu_work(task: Task) -> F:
    return sum((nd.size for nd in task.nodes if nd.kind == GPU), ZERO)


def with_tools(tasks: list[Task], durations: dict[tuple[int, int], F]) -> list[Task]:
    """Copy of tasks with the given tool durations (others keep their stored size)."""
    out = []
    for t in tasks:
        nodes = tuple(Node(nd.kind, durations.get((t.tid, v), nd.size) if nd.kind == TOOL else nd.size, nd.preds)
                      for v, nd in enumerate(t.nodes))
        out.append(t.copy(nodes=nodes))
    return out


# ============================================================================================== simulator
Key = tuple[int, int]  # (tid, node index)
RatePolicy = Callable[[F, list[Key], dict], dict]  # (time, ready GPU nodes, ctx) -> {node: rate}


@dataclass
class Run:
    node_end: dict[Key, F]
    task_end: dict[int, F]
    tool_dur: dict[Key, F]
    enabled_at: dict[Key, F]
    segments: list[tuple[F, F, dict]] = field(default_factory=list)  # (t0, t1, {node: rate})

    def met(self, tasks: list[Task]) -> set[int]:
        return {t.tid for t in tasks if t.tid in self.task_end and self.task_end[t.tid] <= t.deadline}

    def all_met(self, tasks: list[Task]) -> bool:
        return len(self.met(tasks)) == len(tasks)

    def value(self, tasks: list[Task]) -> F:
        ok = self.met(tasks)
        return sum((t.value for t in tasks if t.tid in ok), ZERO)

    def lateness(self, tasks: list[Task], horizon_pen=F(10 ** 6)) -> F:
        """max over tasks of (finish - deadline); unfinished tasks count as horizon_pen."""
        return max((self.task_end.get(t.tid, horizon_pen) - t.deadline for t in tasks), default=ZERO)


def simulate(tasks: list[Task], rate_policy: RatePolicy, tool_oracle=None, max_events: int = 100000,
             check: tuple[int, F] | None = None, record: bool = False) -> Run:
    """Exact event-driven simulation. `check=(m, s)` asserts every rate vector is feasible on m machines of
    speed s (each rate <= s, sum <= m*s)."""
    by = {t.tid: t for t in tasks}
    succs = {t.tid: t.succ() for t in tasks}
    arrivals = sorted(tasks, key=lambda t: (t.release, t.tid))
    ai = 0
    now = ZERO
    rem: dict[Key, F] = {}
    tool_end: dict[Key, F] = {}
    done: set[Key] = set()
    left: dict[int, int] = {}
    run = Run({}, {}, {}, {})
    ctx = {"tasks": by, "rem": rem, "done": done}

    def finish(k: Key, when: F, queue: list):
        tid, v = k
        done.add(k)
        run.node_end[k] = when
        left[tid] -= 1
        if left[tid] == 0:
            run.task_end[tid] = when
        for w in succs[tid][v]:
            if all((tid, u) in done for u in by[tid].nodes[w].preds):
                queue.append((tid, w))

    def enable_all(queue: list, when: F):
        while queue:
            k = queue.pop()
            tid, v = k
            nd = by[tid].nodes[v]
            run.enabled_at[k] = when
            if nd.kind == GPU:
                if nd.size == 0:
                    finish(k, when, queue)
                else:
                    rem[k] = nd.size
            else:
                dur = fr(tool_oracle(tid, v, when)) if tool_oracle is not None else nd.size
                if dur < 0:
                    raise ValueError("negative tool")
                run.tool_dur[k] = dur
                if dur == 0:
                    finish(k, when, queue)
                else:
                    tool_end[k] = when + dur

    for _ in range(max_events):
        queue = []
        while ai < len(arrivals) and arrivals[ai].release <= now:
            t = arrivals[ai]
            ai += 1
            left[t.tid] = len(t.nodes)
            if not t.nodes:
                run.task_end[t.tid] = now
            queue.extend((t.tid, v) for v, nd in enumerate(t.nodes) if not nd.preds)
        enable_all(queue, now)
        ready = sorted(rem)
        rates = {k: fr(r) for k, r in rate_policy(now, ready, ctx).items() if r != 0}
        assert all(k in rem for k in rates), "policy ran a node that is not ready"
        if check is not None:
            m, s = check
            assert all(0 <= r <= s for r in rates.values()) and sum(rates.values(), ZERO) <= m * s, rates
        cands = []
        if ai < len(arrivals):
            cands.append(arrivals[ai].release)
        cands.extend(tool_end.values())
        cands.extend(now + rem[k] / r for k, r in rates.items())
        extra = ctx.get("wake")  # policies may request a wake-up time
        if extra is not None and extra > now:
            cands.append(extra)
        cands = [c for c in cands if c > now]
        if not cands:
            break
        nxt = min(cands)
        dt = nxt - now
        if record and rates:
            run.segments.append((now, nxt, dict(rates)))
        for k, r in rates.items():
            rem[k] -= r * dt
            assert rem[k] >= 0
        now = nxt
        queue = []
        for k in sorted(k for k, x in rem.items() if x == 0):
            del rem[k]
            finish(k, now, queue)
        for k in sorted(k for k, e in tool_end.items() if e == now):
            del tool_end[k]
            finish(k, now, queue)
        enable_all(queue, now)
        if ai >= len(arrivals) and not rem and not tool_end:
            break
    else:
        raise RuntimeError("event limit")
    return run


# ---------------------------------------------------------------------------------- rate policies
def priority_policy(m: int, s, key: Callable[[Key, dict], object]) -> RatePolicy:
    """Global preemptive priority scheduling: the m ready nodes of smallest key run at speed s."""
    s = fr(s)

    def pol(now, ready, ctx):
        ranked = sorted(ready, key=lambda k: (key(k, ctx), k))
        return {k: s for k in ranked[:m]}
    return pol


def edf_key(k, ctx):
    return ctx["tasks"][k[0]].deadline


def fifo_key(k, ctx):
    return ctx["tasks"][k[0]].release


def lifo_key(k, ctx):
    return -k[0]


def make_cpedf_key(tasks: list[Task]):
    tl = {t.tid: tails(t) for t in tasks}

    def key(k, ctx):
        return ctx["tasks"][k[0]].deadline - tl[k[0]][k[1]]
    return key


def static_key(order: dict[Key, object]):
    def key(k, ctx):
        return order[k]
    return key


def srpt_key(k, ctx):
    return ctx["rem"][k]


def ps_policy(m: int, s) -> RatePolicy:
    s = fr(s)

    def pol(now, ready, ctx):
        if not ready:
            return {}
        r = s if len(ready) <= m else F(m) * s / len(ready)
        return {k: r for k in ready}
    return pol


def edf(m, s):
    return priority_policy(m, s, edf_key)


def engine_policy(a, c, lam=ONE, batch: Callable[[F, list[Key], dict], list[Key]] | None = None) -> RatePolicy:
    """Fluid continuous-batching engine: the chosen batch (default: every ready node) advances at
    lam/(a + c*B) work units per second each, work measured in tokens."""
    a, c, lam = fr(a), fr(c), fr(lam)

    def pol(now, ready, ctx):
        b = ready if batch is None else batch(now, ready, ctx)
        if not b:
            return {}
        r = lam / (a + c * len(b))
        return {k: r for k in b}
    return pol


# ============================================================================================== McNaughton
@dataclass(frozen=True)
class Piece:
    key: object
    machine: int
    start: F
    end: F


def mcnaughton(demand: dict, m: int, t0: F, t1: F) -> list[Piece]:
    """Wrap-around rule: give each job `demand[j]` machine-time inside [t0, t1] on m machines, one machine at a
    time per job. Requires demand[j] <= t1 - t0 and sum <= m (t1 - t0)."""
    L = t1 - t0
    if any(x < 0 or x > L for x in demand.values()) or sum(demand.values(), ZERO) > m * L:
        raise ValueError("infeasible demand vector")
    pieces, mach, pos = [], 0, ZERO
    for j in sorted(demand, key=repr):
        need = demand[j]
        while need > 0:
            take = min(need, L - pos)
            pieces.append(Piece(j, mach, t0 + pos, t0 + pos + take))
            need -= take
            pos += take
            if pos == L:
                mach, pos = mach + 1, ZERO
    assert all(p.machine < m for p in pieces)
    return pieces


def check_pieces(pieces: list[Piece]) -> None:
    """No machine runs two pieces at once; no job runs on two machines at once."""
    for attr in ("machine", "key"):
        groups: dict = {}
        for p in pieces:
            groups.setdefault(getattr(p, attr), []).append(p)
        for g in groups.values():
            g.sort(key=lambda p: (p.start, p.end))
            for x, y in zip(g, g[1:]):
                assert x.end <= y.start, (attr, x, y)


def validate_schedule(tasks: list[Task], m: int, speed, pieces: list[Piece]) -> bool:
    """Exact replay of an explicit m-machine schedule (pieces keyed by (tid, v)). Tools start eagerly.
    Checks machine/job overlap, that every GPU node gets exactly size/speed machine time, that it runs only
    after it is enabled, and every task's deadline."""
    speed = fr(speed)
    check_pieces(pieces)
    by_node: dict = {}
    for p in pieces:
        by_node.setdefault(p.key, []).append(p)
    for t in tasks:
        end: dict[int, F] = {}
        for v in t.topo():
            nd = t.nodes[v]
            en = max([t.release] + [end[u] for u in nd.preds])
            if nd.kind == TOOL:
                end[v] = en + nd.size
            else:
                ps = by_node.get((t.tid, v), [])
                if sum((p.end - p.start for p in ps), ZERO) != nd.size / speed:
                    return False
                if any(p.start < en for p in ps):
                    return False
                end[v] = max([en] + [p.end for p in ps])
        if max(end.values(), default=t.release) > t.deadline:
            return False
    return True


def run_to_pieces(run: Run, m: int, s: F) -> list[Piece]:
    """Turn a recorded run of a feasible rate policy into explicit pieces (McNaughton per segment)."""
    out = []
    for t0, t1, rates in run.segments:
        out.extend(mcnaughton({k: r * (t1 - t0) / s for k, r in rates.items()}, m, t0, t1))
    return out


# ============================================================================================== offline (exact)
def gpu_nodes(tasks: list[Task]) -> list[Key]:
    return [(t.tid, v) for t in tasks for v, nd in enumerate(t.nodes) if nd.kind == GPU and nd.size > 0]


def offline_1m_by_orders(tasks: list[Task], speed=ONE, limit: int = 50000):
    """Exact one-machine preemptive feasibility: try every static priority order of the GPU nodes (orders that
    differ only between a node and its own descendant are equivalent and skipped). Returns the order or None."""
    speed = fr(speed)
    nodes = gpu_nodes(tasks)
    anc: dict[Key, set] = {}
    for t in tasks:
        for v in t.topo():
            a = set()
            for u in t.nodes[v].preds:
                a |= anc[(t.tid, u)] | {(t.tid, u)}
            anc[(t.tid, v)] = a
    seen = 0
    for perm in itertools.permutations(nodes):
        pos = {k: i for i, k in enumerate(perm)}
        if any(pos[a] > pos[k] for k in nodes for a in anc[k] if a in pos):
            continue  # a node ranked above its ancestor: same schedule as some canonical order
        seen += 1
        if seen > limit:
            raise RuntimeError("too many orders")
        run = simulate(tasks, priority_policy(1, speed, static_key(pos)))
        if run.all_met(tasks):
            return pos
    return None


def _q(x):
    x = fr(x)
    return z3.RealVal(f"{x.numerator}/{x.denominator}")


def _val(model, e) -> F:
    v = model.eval(e, model_completion=True)
    return F(v.as_fraction())


def offline_mm_lp(tasks: list[Task], m: int = 1, speed=ONE, selected: Iterable[int] | None = None,
                  timeout_ms: int = 60000):
    """Exact m-machine preemptive (migratory) feasibility via Z3 over the reals.

    Variables: per GPU node an enabling time S_v and completion E_v; per tool a start S_u (end S_u + t_u);
    K = 2n sorted breakpoints tau_0 <= ... <= tau_{K-1}; per node and elementary interval a machine-time x >= 0
    with x <= interval length, sum over nodes <= m * length, and x > 0 only if the interval lies in [S_v, E_v].
    Complete (take the breakpoints = sorted {S_v, E_v} of a feasible schedule) and sound (McNaughton in each
    interval). Returns (True, pieces) with an explicit schedule, or (False, None)."""
    speed = fr(speed)
    sel = {t.tid for t in tasks} if selected is None else set(selected)
    ts = [t for t in tasks if t.tid in sel]
    g = gpu_nodes(ts)
    sol = z3.Solver()
    sol.set("timeout", timeout_ms)
    S, E = {}, {}
    for t in ts:
        for v, nd in enumerate(t.nodes):
            S[(t.tid, v)] = z3.Real(f"S_{t.tid}_{v}")
            E[(t.tid, v)] = z3.Real(f"E_{t.tid}_{v}")
    for t in ts:
        for v, nd in enumerate(t.nodes):
            k = (t.tid, v)
            sol.add(S[k] >= _q(t.release))
            for u in nd.preds:
                sol.add(S[k] >= E[(t.tid, u)])
            sol.add(E[k] <= _q(t.deadline))
            if nd.kind == TOOL:
                sol.add(E[k] == S[k] + _q(nd.size))
            elif nd.size == 0:
                sol.add(E[k] == S[k])
            else:
                sol.add(E[k] >= S[k] + _q(nd.size / speed))
    K = max(2, 2 * len(g))
    tau = [z3.Real(f"tau_{i}") for i in range(K)]
    for i in range(K - 1):
        sol.add(tau[i] <= tau[i + 1])
    x = {}
    for i in range(K - 1):
        L = tau[i + 1] - tau[i]
        tot = []
        for k in g:
            xv = z3.Real(f"x_{k[0]}_{k[1]}_{i}")
            x[(k, i)] = xv
            sol.add(xv >= 0, xv <= L, z3.Or(xv == 0, z3.And(tau[i] >= S[k], tau[i + 1] <= E[k])))
            tot.append(xv)
        if tot:
            sol.add(z3.Sum(tot) <= m * L)
    by = {t.tid: t for t in ts}
    for k in g:
        sol.add(z3.Sum([x[(k, i)] for i in range(K - 1)]) == _q(by[k[0]].nodes[k[1]].size / speed))
    res = sol.check()
    if res == z3.unknown:
        raise RuntimeError("z3 unknown")
    if res != z3.sat:
        return False, None
    mdl = sol.model()
    tv = [_val(mdl, e) for e in tau]
    pieces = []
    for i in range(K - 1):
        if tv[i + 1] > tv[i]:
            dem = {k: _val(mdl, x[(k, i)]) for k in g}
            dem = {k: d for k, d in dem.items() if d > 0}
            pieces.extend(mcnaughton(dem, m, tv[i], tv[i + 1]))
    assert validate_schedule(ts, m, speed, pieces), "Z3 witness failed exact replay"
    return True, pieces


def offline_max_value(tasks: list[Task], m: int = 1, speed=ONE):
    """Exact max on-time value: subsets by decreasing value (small n)."""
    best = (ZERO, ())
    ids = [t.tid for t in tasks]
    val = {t.tid: t.value for t in tasks}
    subs = sorted((sub for r in range(len(ids) + 1) for sub in itertools.combinations(ids, r)),
                  key=lambda sb: -sum((val[i] for i in sb), ZERO))
    for sub in subs:
        if offline_mm_lp(tasks, m, speed, selected=sub)[0]:
            return sum((val[i] for i in sub), ZERO), sub
    return best


# ============================================================================================== Theorem 1
def knapsack(weights: list[int], values: list[int], cap: int) -> int:
    best = {0: 0}
    for w, v in zip(weights, values):
        nxt = dict(best)
        for used, val in best.items():
            if used + w <= cap and nxt.get(used + w, -1) < val + v:
                nxt[used + w] = val + v
        best = nxt
    return max(best.values())


def one_machine_on_time_value(p, w, d) -> F:
    """Brute force 1 || sum w_j U_j (r_j = 0): a subset is schedulable iff EDD order meets it (Jackson)."""
    n = len(p)
    best = ZERO
    for mask in range(1 << n):
        sub = sorted((j for j in range(n) if mask >> j & 1), key=lambda j: d[j])
        t, ok = ZERO, True
        for j in sub:
            t += fr(p[j])
            if t > fr(d[j]):
                ok = False
                break
        if ok:
            best = max(best, sum((fr(w[j]) for j in sub), ZERO))
    return best


def engine_min_makespan(tokens: tuple[int, ...], a, c) -> F:
    """Brute force over every sequence of discrete engine steps (each step: a nonempty subset of the unfinished
    sequences, one token each, duration a + c*B). Returns the minimum time to finish all sequences."""
    a, c = fr(a), fr(c)
    memo: dict[tuple, F] = {}

    def go(state):
        if all(x == 0 for x in state):
            return ZERO
        if state in memo:
            return memo[state]
        live = [i for i, x in enumerate(state) if x > 0]
        best = None
        for r in range(1, len(live) + 1):
            for b in itertools.combinations(live, r):
                nxt = tuple(x - 1 if i in b else x for i, x in enumerate(state))
                val = a + c * r + go(nxt)
                if best is None or val < best:
                    best = val
        memo[state] = best
        return best
    return go(tuple(tokens))


def engine_on_time_value_common_deadline(tokens, values, a, c, D) -> F:
    """Max value of a subset the discrete batching engine can finish by common deadline D (release 0)."""
    n = len(tokens)
    best = ZERO
    for mask in range(1 << n):
        sub = [j for j in range(n) if mask >> j & 1]
        if engine_min_makespan(tuple(tokens[j] for j in sub), a, c) <= fr(D):
            best = max(best, sum((fr(values[j]) for j in sub), ZERO))
    return best


# ============================================================================================== Lemma 2
def machine_segments_completion(segments: list[tuple[F, F, frozenset]], work: dict, s: F) -> dict:
    """m-machine schedule given as segments (t0, t1, set of jobs each on its own machine at speed s)."""
    rem = dict(work)
    done = {}
    for t0, t1, js in segments:
        for j in sorted(js, key=repr):
            if j in rem:
                need = rem[j] / s
                if need <= t1 - t0:
                    done[j] = t0 + need
                    del rem[j]
                else:
                    rem[j] -= s * (t1 - t0)
    return done


def fluid_engine_following(segments, work: dict, a, c, lam=ONE) -> dict:
    """Fluid engine that, in each segment, batches the jobs the machine schedule runs there and has not yet
    finished; each advances at lam/(a + c*B). Exact (event-driven inside each segment)."""
    a, c, lam = fr(a), fr(c), fr(lam)
    rem = dict(work)
    done = {}
    for t0, t1, js in segments:
        t = t0
        while t < t1:
            b = [j for j in sorted(js, key=repr) if j in rem]
            if not b:
                break
            r = lam / (a + c * len(b))
            dt = min(t1 - t, min(rem[j] for j in b) / r)
            for j in b:
                rem[j] -= r * dt
            t += dt
            for j in b:
                if rem[j] == 0:
                    del rem[j]
                    done[j] = t
    return done


def discrete_engine_feasible(jobs: list[tuple[F, int, F]], a, c) -> bool:
    """Exhaustive search over discrete engine schedules. jobs: (release, tokens, deadline). A step is a nonempty
    set of released unfinished sequences; it starts at max(previous end, latest release in the set) (starting
    later never helps) and lasts a + c*B; each member gains one token."""
    a, c = fr(a), fr(c)
    n = len(jobs)

    def go(t, left, done_at):
        if all(x == 0 for x in left):
            return all(done_at[j] <= jobs[j][2] for j in range(n))
        live = [j for j in range(n) if left[j] > 0]
        for r in range(1, len(live) + 1):
            for b in itertools.combinations(live, r):
                start = max([t] + [jobs[j][0] for j in b])
                end = start + a + c * r
                nl = list(left)
                nd = dict(done_at)
                ok = True
                for j in b:
                    nl[j] -= 1
                    if nl[j] == 0:
                        nd[j] = end
                        if end > jobs[j][2]:
                            ok = False
                if ok and go(end, tuple(nl), nd):
                    return True
        return False
    return go(ZERO, tuple(j[1] for j in jobs), {})


def discrete_steps_to_machine(steps: list[tuple[F, frozenset]], a, c, m: int):
    """Lemma 2(ii) for the DISCRETE engine: each step (start, batch) lasting a + c*B is realised on m machines of
    speed 1/a (one token = a machine-seconds) inside the step's own interval by McNaughton. Returns pieces."""
    a, c = fr(a), fr(c)
    pieces = []
    t_prev = None
    for start, batch in steps:
        assert t_prev is None or start >= t_prev
        end = start + a + c * len(batch)
        pieces.extend(mcnaughton({(j, start): a for j in batch}, m, start, end))
        t_prev = end
    return pieces


# ============================================================================================== generators
def random_task(rng: random.Random, tid: int, release=ZERO, n=(1, 5), wmax=6, tmax=8, p_tool=0.4, den=2,
                zero_tools=False) -> Task:
    k = rng.randint(*n)
    nodes = []
    for i in range(k):
        kind = TOOL if (i > 0 and rng.random() < p_tool) else GPU
        lo = 0 if (kind == TOOL and zero_tools) else 1
        size = F(rng.randint(lo, (tmax if kind == TOOL else wmax) * den), den)
        preds = tuple(j for j in range(i) if rng.random() < 0.4)
        if i and not preds and rng.random() < 0.8:
            preds = (rng.randrange(i),)
        nodes.append(Node(kind, size, preds))
    return Task(tid, fr(release), ZERO, tuple(nodes))


def random_feasible_run(tasks: list[Task], m: int, rng: random.Random, speed=ONE, idle_prob=0.0) -> Run:
    """A random valid speed-`speed` m-machine schedule: at every event a random subset of at most m ready nodes
    runs (possibly idling some machines)."""
    speed = fr(speed)
    keys = {(t.tid, v): rng.random() for t in tasks for v in range(len(t.nodes))}

    def pol(now, ready, ctx):
        ranked = sorted(ready, key=lambda k: keys[k])
        cnt = m
        ctx["wake"] = None
        if idle_prob and rng.random() < idle_prob:
            cnt = rng.randint(0, m)
            ctx["wake"] = now + F(1, 2)  # idle for at most 1/2, then decide again
        return {k: speed for k in ranked[:cnt]}
    return simulate(tasks, pol, record=True, check=(m, speed))


def enforce_release_order(tasks: list[Task]) -> None:
    """Raise deadlines (never lower them) until r_i < r_j implies d_i > d_j."""
    rels = sorted({t.release for t in tasks}, reverse=True)
    floor = None
    for r in rels:
        grp = [t for t in tasks if t.release == r]
        if floor is not None:
            for t in grp:
                if t.deadline <= floor:
                    t.deadline = floor + F(1, 2)
        gmax = max(t.deadline for t in grp)
        floor = gmax if floor is None else max(floor, gmax)


def release_order_ok(tasks: list[Task]) -> bool:
    return all(not (a.release < b.release) or a.deadline > b.deadline for a in tasks for b in tasks)


def planted(rng: random.Random, n_tasks: int, m: int, delta, common=True, idle_prob=0.0, **kw):
    """Instance feasible at speed 1 on m machines by construction, with slack >= delta and (if not common)
    the release-order condition. Returns (tasks, explicit offline pieces)."""
    delta = fr(delta)
    tasks = [random_task(rng, i, ZERO if common else F(rng.randint(0, 16), 2), **kw) for i in range(n_tasks)]
    run = random_feasible_run(tasks, m, rng, idle_prob=idle_prob)
    for t in tasks:
        t.deadline = max(run.task_end[t.tid], t.release + (1 + delta) * crit_path(t))
    if not common:
        enforce_release_order(tasks)
    pieces = run_to_pieces(run, m, ONE)
    return tasks, pieces


# ============================================================================================== adversaries
def thm2_q(s, delta) -> int:
    """Smallest integer q >= 1 + delta with q (1/s - delta/(1+delta)) > 1, or ValueError at/above threshold."""
    s, delta = fr(s), fr(delta)
    gap = 1 / s - delta / (1 + delta)
    if gap <= 0:
        raise ValueError("s >= 1 + 1/delta: no adversary instance")
    q = 1
    while not (q * gap > 1 and q >= 1 + delta):
        q += 1
    return q


class LastIssuedLong:
    """Theorem 2 adversary: every tool gets 0 except the N-th one issued, which gets L."""

    def __init__(self, N: int, L: F):
        self.N, self.L, self.count = N, L, 0

    def __call__(self, tid, v, when):
        self.count += 1
        return self.L if self.count == self.N else ZERO


def thm2_instance(m: int, q: int, delta):
    delta = fr(delta)
    D = F(q)
    L = D / (1 + delta) - 1
    tasks = [mk_chain(i, 0, D, [(GPU, 1), (TOOL, L)]) for i in range(m * q)]
    return tasks, D, L


def thm2_offline_pieces(real: list[Task], m: int) -> list[Piece]:
    """Explicit offline schedule of the realised Theorem 2 instance: long task first on machine 0."""
    longs = [t.tid for t in real if t.nodes[1].size > 0]
    order = longs + [t.tid for t in real if t.tid not in longs]
    pieces = []
    for i, tid in enumerate(order):
        mach, slot = i % m, i // m
        pieces.append(Piece((tid, 0), mach, F(slot), F(slot + 1)))
    return pieces


class Thm3Adversary:
    def __init__(self, D: F):
        self.D = D

    def __call__(self, tid, v, when):
        return max(ZERO, self.D - when + F(1, 2))


# ============================================================================================== instances from the docs
def factA_instance(eps):
    eps = fr(eps)
    return [mk_chain(0, 0, 2, [(GPU, 1)]), mk_chain(1, 0, 10, [(GPU, eps), (TOOL, 10 - 2 * eps), (GPU, eps)])]


def r4_instance():
    """docs/open_problem.md R4, transcribed from the document text (not from the code)."""
    rK = F(1757, 45)
    return [mk_chain(0, 0, 54, [(GPU, 27)]),
            mk_chain(1, 0, 78, [(GPU, F(1, 10)), (TOOL, 27), (GPU, F(119, 10))]),
            mk_chain(2, rK, rK + 40, [(GPU, F(1, 10)), (TOOL, F(99, 5)), (GPU, F(1, 10))]),
            mk_chain(3, rK, rK + F(79, 2), [(GPU, F(39, 2))]),
            mk_chain(4, rK, rK + F(79, 2), [(GPU, F(39, 2))])]


def r4_family(s, y, eps, w, n_each, dA, dZ, dN_rel, D=F(40)):
    """The R4 family as described in the document (Z, A, K, N1, N2)."""
    s, y, eps, w, n_each, dA, dZ, dN_rel = map(fr, (s, y, eps, w, n_each, dA, dZ, dN_rel))
    LA = s * y
    rK = y + eps / s + LA
    return [mk_chain(0, 0, dZ, [(GPU, s * y)]),
            mk_chain(1, 0, dA, [(GPU, eps), (TOOL, LA), (GPU, w)]),
            mk_chain(2, rK, rK + D, [(GPU, eps), (TOOL, D / 2 - 2 * eps), (GPU, eps)]),
            mk_chain(3, rK, rK + dN_rel, [(GPU, n_each)]),
            mk_chain(4, rK, rK + dN_rel, [(GPU, n_each)])]


def r5_instance():
    return [mk_chain(0, 0, F(711, 100), [(GPU, 1), (TOOL, 5), (GPU, F(11, 10))]),
            mk_chain(1, 0, F(706, 100), [(GPU, F(1, 100)), (TOOL, F(49, 10)), (GPU, F(11, 10))])]


def evict_inflate(tasks: list[Task], beta) -> list[Task]:
    """Thm 7 transformation at the worst allowed recompute: GPU nodes with a TOOL predecessor get (1+beta)x."""
    beta = fr(beta)
    out = []
    for t in tasks:
        nodes = tuple(Node(nd.kind, nd.size * (1 + beta) if nd.kind == GPU and any(t.nodes[u].kind == TOOL
                           for u in nd.preds) else nd.size, nd.preds) for nd in t.nodes)
        out.append(t.copy(nodes=nodes))
    return out


def peak_held_kv(run: Run, tasks: list[Task]) -> int:
    """Number of tasks simultaneously holding KV under keep-KV, where a task holds from its first GPU node
    completion until it finishes (unit KV per task)."""
    ev = []
    for t in tasks:
        ends = [run.node_end[(t.tid, v)] for v, nd in enumerate(t.nodes) if nd.kind == GPU]
        if ends and t.tid in run.task_end:
            ev.append((min(ends), 1))
            ev.append((run.task_end[t.tid], -1))
    ev.sort(key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak
