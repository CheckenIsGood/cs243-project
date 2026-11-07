"""Exact machinery for docs/open_problem.md (online deadline scheduling of agent DAGs with tool waits).

Everything is exact rational arithmetic (fractions.Fraction); no floating point is used in any decision.

Contents
- Task model: a task is a DAG of GPU nodes (work, in machine-seconds at speed 1) and TOOL nodes (fixed duration,
  NOT sped up by a faster GPU, unlimited tool parallelism, issued as soon as enabled).
- `simulate`: event-driven simulator of an online policy on m identical machines of speed s with preemption and
  migration (the model Lemma 2 of docs/general_theory.md reduces the batched engine to). Policies: global EDF by
  task deadline, critical-path sub-deadline EDF (CP-EDF, needs known tool durations), FIFO, an arbitrary static
  key, and processor sharing. Tool durations can be supplied by an adaptive adversary that sees only the issue
  event (models "tool duration revealed at issue time or later").
- `offline_feasible_1m` / `offline_max_value_1m`: exact offline optimum on ONE machine via Z3 (linear real
  arithmetic). Encoding: a schedule exists iff there are node release times rho_v and completion times C_v that
  respect the DAG and tool durations and satisfy Horn's interval condition (for every [a, b] with a a release and
  b a completion, the work of nodes with window inside [a, b] is at most (b - a) * speed). Sufficiency is checked
  constructively: `witness_check` replays the model with EDF on the C_v and verifies every deadline.
- Constructions for the theorems and counterexamples in docs/open_problem.md.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Callable, Iterable

import z3

GPU, TOOL = "gpu", "tool"


def Fr(x) -> F:
    return x if isinstance(x, F) else F(x) if not isinstance(x, float) else F(str(x))


# ----------------------------------------------------------------------------------------------- model
@dataclass
class Node:
    name: str
    kind: str  # GPU or TOOL
    size: F  # GPU: work at speed 1; TOOL: duration
    preds: tuple[str, ...] = ()


@dataclass
class Task:
    tid: int
    release: F
    deadline: F
    nodes: dict[str, Node]
    value: F = F(1)

    def succs(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {n: [] for n in self.nodes}
        for n in self.nodes.values():
            for p in n.preds:
                out[p].append(n.name)
        return out

    def topo(self) -> list[str]:
        order, seen = [], set()

        def visit(n):
            if n in seen:
                return
            for p in self.nodes[n].preds:
                visit(p)
            seen.add(n)
            order.append(n)

        for n in self.nodes:
            visit(n)
        return order


def chain(tid: int, release, deadline, stages: Iterable[tuple[str, object]], value=1) -> Task:
    """Chain task from [(kind, size), ...], e.g. [(GPU, 1), (TOOL, 5), (GPU, 1)]."""
    nodes, prev = {}, ()
    for i, (kind, size) in enumerate(stages):
        name = f"{kind}{i}"
        nodes[name] = Node(name, kind, Fr(size), prev)
        prev = (name,)
    return Task(tid, Fr(release), Fr(deadline), nodes, Fr(value))


def node_time(n: Node, speed: F) -> F:
    return n.size / speed if n.kind == GPU else n.size


def critical_path(task: Task, speed=F(1)) -> F:
    """Longest path (GPU work / speed + tool durations)."""
    speed = Fr(speed)
    best: dict[str, F] = {}
    for n in task.topo():
        node = task.nodes[n]
        best[n] = node_time(node, speed) + max((best[p] for p in node.preds), default=F(0))
    return max(best.values(), default=F(0))


def tail(task: Task, speed=F(1)) -> dict[str, F]:
    """tail[v] = longest path strictly after v to a sink (so LCT(v) = deadline - tail[v])."""
    speed = Fr(speed)
    succ = task.succs()
    out: dict[str, F] = {}
    for n in reversed(task.topo()):
        out[n] = max((node_time(task.nodes[c], speed) + out[c] for c in succ[n]), default=F(0))
    return out


def has_slack(task: Task, delta, speed=F(1)) -> bool:
    return task.deadline - task.release >= (1 + Fr(delta)) * critical_path(task, speed)


def total_work(task: Task) -> F:
    return sum((n.size for n in task.nodes.values() if n.kind == GPU), F(0))


# ----------------------------------------------------------------------------------------------- simulator
@dataclass
class SimResult:
    finish: dict[int, F]  # task -> completion time (missing = never completed)
    node_finish: dict[tuple[int, str], F]
    tool_len: dict[tuple[int, str], F]  # realized tool durations (from the oracle)
    trace: list[tuple[F, F, tuple]] = field(default_factory=list)  # (start, end, running nodes)

    def on_time(self, tasks: list[Task]) -> set[int]:
        return {t.tid for t in tasks if t.tid in self.finish and self.finish[t.tid] <= t.deadline}

    def value(self, tasks: list[Task]) -> F:
        ok = self.on_time(tasks)
        return sum((t.value for t in tasks if t.tid in ok), F(0))

    def all_met(self, tasks: list[Task]) -> bool:
        return len(self.on_time(tasks)) == len(tasks)


ToolOracle = Callable[[int, str, F], F]  # (task id, node name, issue time) -> duration


def simulate(tasks: list[Task], policy="edf", m: int = 1, s=F(1), tool_oracle: ToolOracle | None = None,
             keys: dict[tuple[int, str], object] | None = None, admit: Callable[[Task, F], bool] | None = None,
             drop_late: bool = False, horizon=None, record_trace: bool = False) -> SimResult:
    """Exact event-driven simulation on m identical machines of speed s (preemption + migration).

    policy: 'edf'   key (deadline, tid)                     -- needs no tool or work information
            'cpedf' key (deadline - tail(v), tid)            -- latest-completion-time EDF; needs tool durations
            'fifo'  key (release, tid)
            'key'   key keys[(tid, node)] (static, any comparable)
            'ps'    processor sharing: all ready GPU nodes share m*s, each capped at s
    tool_oracle: if given, tool durations are decided at issue time by this callback (adaptive adversary);
            otherwise the durations stored in the task are used.
    admit: optional admission test called at release (rejected tasks never run).
    drop_late: abandon a task once its deadline has passed unfinished.
    """
    s = Fr(s)
    by_id = {t.tid: t for t in tasks}
    tails = {t.tid: tail(t) for t in tasks} if policy == "cpedf" else {}
    pending_arrivals = sorted(tasks, key=lambda t: (t.release, t.tid))
    t_now = F(0)
    remaining: dict[tuple[int, str], F] = {}  # ready GPU nodes -> remaining work
    tool_end: dict[tuple[int, str], F] = {}
    done: set[tuple[int, str]] = set()
    node_finish: dict[tuple[int, str], F] = {}
    tool_len: dict[tuple[int, str], F] = {}
    finish: dict[int, F] = {}
    active: set[int] = set()
    left_nodes: dict[int, int] = {}
    trace = []

    def key_of(nk):
        tid, n = nk
        tk = by_id[tid]
        if policy == "edf":
            return (tk.deadline, tid, n)
        if policy == "cpedf":
            return (tk.deadline - tails[tid][n], tid, n)
        if policy == "fifo":
            return (tk.release, tid, n)
        if policy == "key":
            return (keys[nk], tid, n)
        raise ValueError(policy)

    def complete(nk, when):
        tid, n = nk
        done.add(nk)
        node_finish[nk] = when
        left_nodes[tid] -= 1
        if left_nodes[tid] == 0:
            finish[tid] = when
            active.discard(tid)
            return
        for c in by_id[tid].succs()[n]:
            if all((tid, p) in done for p in by_id[tid].nodes[c].preds):
                enable((tid, c), when)

    def enable(nk, when):
        tid, n = nk
        node = by_id[tid].nodes[n]
        if node.kind == GPU:
            if node.size == 0:
                complete(nk, when)
            else:
                remaining[nk] = node.size
        else:
            dur = Fr(tool_oracle(tid, n, when)) if tool_oracle else node.size
            if dur < 0:
                raise ValueError("negative tool duration")
            tool_len[nk] = dur
            if dur == 0:
                complete(nk, when)
            else:
                tool_end[nk] = when + dur

    def drop(tid):
        active.discard(tid)
        for nk in [nk for nk in remaining if nk[0] == tid]:
            del remaining[nk]
        for nk in [nk for nk in tool_end if nk[0] == tid]:
            del tool_end[nk]

    while True:
        while pending_arrivals and pending_arrivals[0].release <= t_now:
            tk = pending_arrivals.pop(0)
            if admit is not None and not admit(tk, t_now):
                continue
            active.add(tk.tid)
            left_nodes[tk.tid] = len(tk.nodes)
            if not tk.nodes:
                finish[tk.tid] = t_now
                continue
            for n, node in tk.nodes.items():
                if not node.preds:
                    enable((tk.tid, n), t_now)
        if drop_late:
            for tid in [tid for tid in active if by_id[tid].deadline < t_now]:
                drop(tid)
        # choose running set and rates
        ready = list(remaining)
        if policy == "ps":
            rates = {}
            if ready:
                r = s if len(ready) <= m else F(m) * s / len(ready)
                rates = {nk: r for nk in ready}
        else:
            ready.sort(key=key_of)
            rates = {nk: s for nk in ready[:m]}
        cands = []
        if pending_arrivals:
            cands.append(pending_arrivals[0].release)
        if tool_end:
            cands.append(min(tool_end.values()))
        for nk, r in rates.items():
            cands.append(t_now + remaining[nk] / r)
        if drop_late:
            later = [by_id[tid].deadline for tid in active if by_id[tid].deadline > t_now]
            if later:
                cands.append(min(later))  # tasks are dropped at the first event strictly after their deadline
        if horizon is not None:
            cands.append(Fr(horizon))
        cands = [c for c in cands if c > t_now]
        if not cands:
            break
        t_next = min(cands)
        if horizon is not None and t_next > Fr(horizon):
            break
        dt = t_next - t_now
        if record_trace and rates and dt > 0:
            trace.append((t_now, t_next, tuple(sorted(rates))))
        for nk, r in rates.items():
            remaining[nk] -= r * dt
        t_now = t_next
        for nk in sorted([nk for nk, v in remaining.items() if v <= 0]):
            del remaining[nk]
            complete(nk, t_now)
        for nk in sorted([nk for nk, e in tool_end.items() if e <= t_now]):
            del tool_end[nk]
            complete(nk, t_now)
        if horizon is not None and t_now >= Fr(horizon):
            break
        if not pending_arrivals and not remaining and not tool_end:
            break
    return SimResult(finish, node_finish, tool_len, trace)


def realized(tasks: list[Task], tool_len: dict[tuple[int, str], F]) -> list[Task]:
    """Copy of `tasks` with tool durations replaced by the realized ones (tools never issued keep size 0)."""
    out = []
    for t in tasks:
        nodes = {}
        for n, node in t.nodes.items():
            size = tool_len.get((t.tid, n), F(0)) if node.kind == TOOL else node.size
            nodes[n] = Node(n, node.kind, size, node.preds)
        out.append(Task(t.tid, t.release, t.deadline, nodes, t.value))
    return out


# ----------------------------------------------------------------------------------------------- offline (1 machine)
def _q(x: F):
    x = Fr(x)
    return z3.Q(x.numerator, x.denominator)


def _to_frac(v) -> F:
    v = v.as_fraction() if hasattr(v, "as_fraction") else F(str(v))
    return F(v)


def offline_feasible_1m(tasks: list[Task], speed=F(1), selected: Iterable[int] | None = None):
    """Exact: can ONE machine of the given speed (preemptive) complete every selected task by its deadline?
    Returns (feasible, completion-time map C or None). Tool durations are the ones stored in the tasks."""
    speed = Fr(speed)
    sel = set(t.tid for t in tasks) if selected is None else set(selected)
    ts = [t for t in tasks if t.tid in sel]
    sol = z3.Solver()
    start, end = {}, {}
    gpu = []
    for t in ts:
        for n, node in t.nodes.items():
            start[(t.tid, n)] = z3.Real(f"st_{t.tid}_{n}")
            end[(t.tid, n)] = z3.Real(f"en_{t.tid}_{n}")
    for t in ts:
        for n, node in t.nodes.items():
            k = (t.tid, n)
            sol.add(start[k] >= _q(t.release))
            for p in node.preds:
                sol.add(start[k] >= end[(t.tid, p)])
            if node.kind == TOOL:
                sol.add(end[k] == start[k] + _q(node.size))
            else:
                sol.add(end[k] >= start[k] + _q(node.size / speed))
                gpu.append((k, node.size / speed))
            sol.add(end[k] <= _q(t.deadline))
    for (ka, _), (kb, _) in itertools.product(gpu, gpu):
        a, b = start[ka], end[kb]
        terms = [z3.If(z3.And(start[kw] >= a, end[kw] <= b), _q(p), _q(F(0))) for kw, p in gpu]
        sol.add(z3.Implies(a <= b, z3.Sum(terms) <= b - a))
    if sol.check() != z3.sat:
        return False, None
    mdl = sol.model()
    C = {k: _to_frac(mdl.eval(end[k], model_completion=True)) for k, _ in gpu}
    return True, C


def witness_check(tasks: list[Task], C: dict[tuple[int, str], F], speed=F(1), selected=None) -> bool:
    """Constructive check of a Z3 witness: EDF on the node completion targets C (static keys) at `speed`, run on
    the selected tasks, must complete every GPU node by its target and every task by its deadline."""
    sel = set(t.tid for t in tasks) if selected is None else set(selected)
    ts = [t for t in tasks if t.tid in sel]
    keys = {k: v for k, v in C.items() if k[0] in sel}
    res = simulate(ts, policy="key", m=1, s=speed, keys=keys)
    for k, c in keys.items():
        if res.node_finish.get(k, None) is None or res.node_finish[k] > c:
            return False
    return res.all_met(ts)


def offline_max_value_1m(tasks: list[Task], speed=F(1)):
    """Exact max on-time value on one machine: enumerate subsets by decreasing value (n <= ~8)."""
    ids = [t.tid for t in tasks]
    val = {t.tid: t.value for t in tasks}
    subsets = []
    for k in range(len(ids) + 1):
        for sub in itertools.combinations(ids, k):
            subsets.append((sum((val[i] for i in sub), F(0)), sub))
    subsets.sort(key=lambda x: -x[0])
    for v, sub in subsets:
        ok, _ = offline_feasible_1m(tasks, speed, selected=sub)
        if ok:
            return v, sub
    return F(0), ()


def static_order_feasible(tasks: list[Task], speed=F(1)) -> bool:
    """Sufficient test: does SOME static priority order over GPU nodes (single machine) meet all deadlines?"""
    gpu = [(t.tid, n) for t in tasks for n, node in t.nodes.items() if node.kind == GPU]
    for perm in itertools.permutations(gpu):
        keys = {k: i for i, k in enumerate(perm)}
        if simulate(tasks, policy="key", m=1, s=speed, keys=keys).all_met(tasks):
            return True
    return False


# ----------------------------------------------------------------------------------------------- constructions
def edf_tool_counterexample(eps) -> list[Task]:
    """Known fact 1 of the brief: A = 1 GPU, deadline 2; B = eps GPU -> tool 10-2eps -> eps GPU, deadline 10."""
    eps = Fr(eps)
    A = chain(0, 0, 2, [(GPU, 1)])
    B = chain(1, 0, 10, [(GPU, eps), (TOOL, 10 - 2 * eps), (GPU, eps)])
    return [A, B]


def ridouard_instance(k: int) -> list[Task]:
    """Ridouard, Richard, Cottet (RTSS 2004), proof of Theorem 6: EDF fails at speed k although speed 1 is feasible.
    tau1: C=2k, no suspension, D=4k-1;  tau2: 1 -> suspension 4k-2 -> 1, D=4k; both released at 0."""
    return [chain(0, 0, 4 * k - 1, [(GPU, 2 * k)]), chain(1, 0, 4 * k, [(GPU, 1), (TOOL, 4 * k - 2), (GPU, 1)])]


def r2_parameters(s, delta) -> int:
    """Smallest k for which the Theorem-2 adversary defeats every online algorithm at speed s with slack delta
    (requires s < (1+delta)/delta; for delta = 0 any s)."""
    s, delta = Fr(s), Fr(delta)
    gap = 1 / s - delta / (1 + delta)
    if gap <= 0:
        raise ValueError("no k exists: s >= (1+delta)/delta")
    k = max(2, int(1 / gap) + 1)
    while not (k * gap > 1 and k >= 1 + delta):
        k += 1
    return k


def r2_feasibility_adversary(k: int, delta, x=F(1)):
    """Theorem 2 instance family. k chain tasks GPU(x) -> TOOL(?), released at 0, common deadline D = k x.
    The tool length is revealed when the tool is issued: the LAST task to issue gets L_long = D/(1+delta) - x,
    every other task gets 0. Returns (tasks with placeholder tool sizes, oracle factory)."""
    x, delta = Fr(x), Fr(delta)
    D = k * x
    L_long = D / (1 + delta) - x
    tasks = [chain(i, 0, D, [(GPU, x), (TOOL, L_long)]) for i in range(k)]  # stored size = upper bound

    def make_oracle():
        issued = []

        def oracle(tid, node, when):
            issued.append(tid)
            return L_long if len(issued) == k else F(0)

        return oracle

    return tasks, make_oracle, L_long


def r2_offline_witness(tasks_realized: list[Task]) -> dict[tuple[int, str], int]:
    """Offline priority order for the realized Theorem-2 instance: the long-tool task's GPU node first."""
    long_tid = max(tasks_realized, key=lambda t: t.nodes["tool1"].size).tid
    order = [long_tid] + [t.tid for t in tasks_realized if t.tid != long_tid]
    return {(tid, "gpu0"): i for i, tid in enumerate(order)}


def r2_throughput_adversary(k: int, x=F(1)):
    """Theorem 3 (throughput) family: k unit-value chains GPU(x) -> TOOL(?), released at 0, deadline D = k x.
    When a task's tool is issued at time tau its length is set to D - tau + x/2, so it finishes after D."""
    x = Fr(x)
    D = k * x
    tasks = [chain(i, 0, D, [(GPU, x), (TOOL, D + x / 2)]) for i in range(k)]

    def oracle(tid, node, when):
        return max(F(0), D - when + x / 2)

    return tasks, oracle


def r2_throughput_offline_set(tasks_realized: list[Task], online_issue: dict[int, F], x=F(1)) -> list[int]:
    """Offline plan for Theorem 3: take the g tasks the online issued LAST, in online issue order, g = floor(k/s-1/2).
    Here we simply return the largest suffix (in online issue order) that a speed-1 machine can serve in order."""
    order = sorted(online_issue, key=lambda tid: (online_issue[tid], tid))
    best: list[int] = []
    for g in range(len(order), 0, -1):
        cand = order[len(order) - g:]
        ok = all((l + 1) * x + tasks_realized[tid].nodes["tool1"].size <= tasks_realized[tid].deadline
                 for l, tid in enumerate(cand))
        if ok:
            best = cand
            break
    return best


def r4_general_release_counterexample(s=F(9, 4), y=F(12), eps=F(1, 10), w=F(119, 10), n_each=F(39, 2),
                                      dA=F(78), dZ=F(54), dN_rel=F(79, 2)):
    """Refutes 'global EDF at speed 1 + 1/delta meets every feasible deadline for ARBITRARY release times'
    (delta = 1, so 1 + 1/delta = 2). Default parameters: EDF misses task K at speed 9/4 although a speed-1 schedule
    meets all tasks and every task has slack >= 1. `r4_at_5_2()` gives the speed-5/2 member of the family.
      Z: GPU(s*y), released 0, deadline dZ           (EDF runs it before A's first node: dZ < dA)
      A: GPU(eps) -> TOOL(s*y) -> GPU(w), released 0, deadline dA
      K: GPU(eps) -> TOOL(D/2 - 2eps) -> GPU(eps), released r_K = y + eps/s + s*y, deadline r_K + D, D = 40
      N1, N2: GPU(n_each), released r_K, deadline r_K + dN_rel
    Online, Z delays A's first node by y, so A's second node arrives exactly at r_K and, with N1, N2, blocks K's
    first node (EDF prefers the earlier deadlines dA, r_K + dN_rel < d_K). Offline, A runs first, its second node is
    done before r_K, and K's first node runs at r_K. See docs/open_problem.md, Claim R4."""
    LA = s * y
    rK = y + eps / s + LA
    D = F(40)
    LK = D / 2 - 2 * eps
    Z = chain(0, 0, dZ, [(GPU, s * y)])
    A = chain(1, 0, dA, [(GPU, eps), (TOOL, LA), (GPU, w)])
    K = chain(2, rK, rK + D, [(GPU, eps), (TOOL, LK), (GPU, eps)])
    N1 = chain(3, rK, rK + dN_rel, [(GPU, n_each)])
    N2 = chain(4, rK, rK + dN_rel, [(GPU, n_each)])
    return [Z, A, K, N1, N2], s, F(1)


def r4_at_5_2():
    return r4_general_release_counterexample(s=F(5, 2), y=F(11), eps=F(1, 100), w=F(1099, 100),
                                             n_each=F(197, 10), dA=F(77), dZ=F(55), dN_rel=F(79, 2))


def r4_offline_keys(tasks: list[Task]) -> dict[tuple[int, str], int]:
    """Explicit speed-1 schedule for r4 (static priorities): A.pre > K nodes > Z > A.post > N1 > N2."""
    order = [(1, "gpu0"), (2, "gpu0"), (2, "gpu2"), (0, "gpu0"), (1, "gpu2"), (3, "gpu0"), (4, "gpu0")]
    return {k: i for i, k in enumerate(order)}


def cpedf_two_task_family(a=F(1), c=F(11, 10), u=F(1, 10), eps=F(1, 100), t0=F(5)) -> list[Task]:
    """Claim R5: critical-path EDF (tool durations KNOWN) misses a deadline at every speed s < (1+a/(a+c+u))
    on this zero-slack, common-release instance although speed 1 is feasible (defaults: fails for s <= 29/20).
      T0: GPU(a) -> TOOL(t0) -> GPU(c),          deadline eps + a + t0 + c   (its first node has the earlier LCT)
      T1: GPU(eps) -> TOOL(t0-u) -> GPU(a+u),    deadline eps + 2a + t0 + (c-a)/2
    Offline runs T1's tiny first node first, so T1's second node is finished before T0's second node is ready.
    CP-EDF runs T0's first node first; T1's second node then arrives later and, having the earlier LCT,
    preempts T0's zero-slack second node."""
    t1 = t0 - u
    d0 = eps + a + t0 + c
    d1 = eps + 2 * a + t0 + (c - a) / 2
    return [chain(0, 0, d0, [(GPU, a), (TOOL, t0), (GPU, c)]), chain(1, 0, d1, [(GPU, eps), (TOOL, t1), (GPU, a + u)])]


def cpedf_search_instance() -> list[Task]:
    """A CP-EDF failure at speed 6/5 with general releases found by local search (zero slack, known tools)."""
    return [chain(0, F(5, 2), F(45, 4), [(GPU, 2), (TOOL, F(5, 2)), (GPU, F(15, 4))]),
            chain(1, 0, 1, [(GPU, 1)]),
            chain(2, 2, 12, [(GPU, 1), (TOOL, F(9, 4)), (GPU, 2), (TOOL, F(3, 2)), (GPU, F(1, 2))])]


# ----------------------------------------------------------------------------------------------- generators
def random_dag_task(rng, tid: int, release=F(0), n_nodes=(1, 5), max_work=6, max_tool=8, p_tool=0.4,
                    denom=2) -> Task:
    """Random DAG task (integer/denom sizes). Edges only from lower to higher index, so it is acyclic."""
    n = rng.randint(*n_nodes)
    nodes = {}
    for i in range(n):
        kind = TOOL if (rng.random() < p_tool and i > 0) else GPU
        size = F(rng.randint(1, (max_tool if kind == TOOL else max_work) * denom), denom)
        preds = tuple(f"v{j}" for j in range(i) if rng.random() < 0.5) if i else ()
        if i and not preds and rng.random() < 0.7:
            preds = (f"v{i-1}",)
        nodes[f"v{i}"] = Node(f"v{i}", kind, size, preds)
    return Task(tid, Fr(release), F(0), nodes)


def planted_instance(rng, n_tasks: int, m: int, delta, common_release=True, lifo=False, **kw) -> list[Task]:
    """Instance that is feasible for an offline speed-1 m-machine schedule BY CONSTRUCTION, with slack delta.
    A random static-priority schedule at speed 1 is simulated; each deadline is set to
    max(finish in that schedule, release + (1+delta)*critical path). Deadlines are never earlier than the planted
    finish, so the planted schedule is a feasible offline schedule. If lifo=True, releases are random and deadlines
    are then increased (never decreased) so that r_i < r_j implies d_i > d_j (the hypothesis of Theorem 4)."""
    delta = Fr(delta)
    tasks = []
    for i in range(n_tasks):
        rel = F(0) if common_release else F(rng.randint(0, 10 * 2), 2)
        tasks.append(random_dag_task(rng, i, rel, **kw))
    keys = {(t.tid, n): rng.random() for t in tasks for n in t.nodes}
    res = simulate(tasks, policy="key", m=m, s=F(1), keys=keys)
    for t in tasks:
        t.deadline = max(res.finish[t.tid], t.release + (1 + delta) * critical_path(t))
    if lifo:
        # enforce r_i < r_j  =>  d_i > d_j by processing releases from latest to earliest
        by_rel = sorted(tasks, key=lambda t: -t.release)
        bound = None
        cur_rel = None
        group_max = F(0)
        for t in by_rel:
            if cur_rel is None or t.release < cur_rel:
                if cur_rel is not None:
                    bound = group_max if bound is None else max(bound, group_max)
                cur_rel = t.release
            if bound is not None and t.deadline <= bound:
                t.deadline = bound + 1
            group_max = max(group_max, t.deadline)
    return tasks


# ----------------------------------------------------------------------------------------------- KV (P4)
def evict_on_suspend(tasks: list[Task], recompute: Callable[[Task, str], F]) -> list[Task]:
    """Transformed instance for the evict-on-suspend KV policy: a GPU node that has a TOOL predecessor (the task's
    KV was evicted while it waited) carries extra recompute work recompute(task, node)."""
    out = []
    for t in tasks:
        nodes = {}
        for n, node in t.nodes.items():
            extra = F(0)
            if node.kind == GPU and any(t.nodes[p].kind == TOOL for p in node.preds):
                extra = Fr(recompute(t, n))
            nodes[n] = Node(n, node.kind, node.size + extra, node.preds)
        out.append(Task(t.tid, t.release, t.deadline, nodes, t.value))
    return out


def peak_suspended_kv(res: SimResult, tasks: list[Task], kv: dict[int, F]) -> F:
    """Peak total KV held by tasks that are suspended in a tool, if the policy KEEPS KV during tool waits.
    A task is 'holding' from its first GPU completion to its finish."""
    events = []
    for t in tasks:
        gpu_done = [res.node_finish[(t.tid, n)] for n, nd in t.nodes.items() if nd.kind == GPU
                    and (t.tid, n) in res.node_finish]
        if gpu_done and t.tid in res.finish:
            events.append((min(gpu_done), kv[t.tid]))
            events.append((res.finish[t.tid], -kv[t.tid]))
    events.sort(key=lambda e: (e[0], e[1]))
    cur = peak = F(0)
    for _, dlt in events:
        cur += dlt
        peak = max(peak, cur)
    return peak
