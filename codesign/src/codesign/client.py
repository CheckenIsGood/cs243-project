"""Client side of the co-design (docs/vllm_codesign_spec.md Section 1, docs/experiment_plan.md Section 3).

CodesignLayer plugs into deadline_agent.multiagent.run_agent_task as its serving layer. Per model request it
sets the vLLM priority (lower runs first) and the `vllm_xargs` the CodesignScheduler reads
(cd_task, cd_preload, cd_ttl). Priority rules:

  fcfs       0 for every request (ties by request arrival: stock behaviour)
  zprime     ours: fixed task index Z'_J / w_J, computed once at release from token-count estimates (Thm D)
  task_fcfs  Continuum-like: task (program) arrival time
  plas       Autellix-like: attained service of the task so far (program-level least attained service)
  area       LAMPS-like: this request's memory-time area (prompt + max_tokens) x max_tokens x step time

Times are seconds; VLLMBackend converts priority to integer milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deadline_agent.clock import now
from deadline_agent.model_client import messages_tokens

RULES = ("fcfs", "zprime", "task_fcfs", "plas", "area")


@dataclass
class Calib:
    a: float = 0.027767  # s, step overhead
    c: float = 0.000543  # s per running sequence
    alpha: float = 1.14e-4  # s per prefill token


@dataclass
class ShapeEst:
    """Token-count estimates known at release (the theory's 'token counts known at release' assumption)."""
    plan_prompt: int = 160
    plan_out: int = 128
    result_tokens: int = 1000  # per tool result
    final_out: int = 64
    step_out: int = 48  # react step
    base_prompt: int = 120  # system + question


def task_nodes(shape: str, width: int, e: ShapeEst) -> list[tuple[int, int, int]]:
    """LLM nodes of a task as (rho, q, X): parent context recomputed under evict-always, new prompt, max output."""
    if shape == "fanout":
        return [(0, e.plan_prompt, e.plan_out), (0, e.base_prompt + width * e.result_tokens, e.final_out)]
    if shape == "react":
        out, ctx = [], 0
        for t in range(width):
            q = e.base_prompt if t == 0 else e.step_out + e.result_tokens
            out.append((ctx, q, e.step_out))
            ctx += q + (e.step_out if t else e.step_out)
        out.append((ctx, e.step_out + e.result_tokens + 40, e.final_out))
        return out
    raise ValueError(shape)


def zprime(nodes, cal: Calib, M_tokens: int | None, L_max: int) -> float:
    """Z'_J = sum_y alpha(q+rho) + kappa_y + cX + aX l+/M, kappa_y = alpha(l+ + L_max) (Thm D, Section 4.2)."""
    z = 0.0
    for rho, q, X in nodes:
        lp = rho + q + X
        z += cal.alpha * (q + rho) + cal.c * X
        if M_tokens:
            M = max(1, M_tokens - L_max)
            z += cal.alpha * (lp + L_max) + cal.a * X * lp / M
    return z


@dataclass
class CodesignLayer:
    rule: str = "fcfs"
    cal: Calib = field(default_factory=Calib)
    est: ShapeEst = field(default_factory=ShapeEst)
    M_tokens: int | None = None  # server KV capacity M' (tokens)
    L_max: int = 0
    ttl: float = 0.0  # cd_ttl for requests followed by a tool wait (0 = none)
    tag_preloads: bool = True
    service: dict = field(default_factory=dict)  # task -> attained service (s), for plas
    index: dict = field(default_factory=dict)  # task -> Z'/w, for zprime

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"unknown rule {self.rule!r}; one of {RULES}")

    # ServingLayer interface used by run_agent_task
    def priority(self, at, stage: str) -> float:
        return 0.0

    def keep(self, at, stage: str) -> bool:
        return True

    def task_index(self, at) -> float:
        if at.task_id not in self.index:
            nodes = task_nodes(at.shape, at.width, self.est)
            self.index[at.task_id] = zprime(nodes, self.cal, self.M_tokens, self.L_max) / max(at.value, 1e-9)
        return self.index[at.task_id]

    def request_extra(self, at, stage: str, phase: str, msgs, max_tokens: int, backend) -> dict:
        xa: dict = {"cd_task": at.task_id}
        if phase == "preload":
            prio = 1e9  # lowest; preloads are also gated in the engine when cd_preload is set
            if self.tag_preloads:
                xa["cd_preload"] = 1
        elif self.rule == "fcfs":
            prio = 0.0
        elif self.rule == "zprime":
            prio = self.task_index(at)
        elif self.rule == "task_fcfs":
            prio = at.arrival
        elif self.rule == "plas":
            prio = self.service.get(at.task_id, 0.0)
        else:  # area
            p = messages_tokens(backend, msgs)
            prio = (p + max_tokens) * max_tokens * (self.cal.a + self.cal.c) / 1000.0
        if self.ttl > 0 and phase in ("plan", "step"):  # a tool wait follows these calls
            xa["cd_ttl"] = self.ttl
        return {"priority": prio, "vllm_xargs": xa}

    def on_done(self, at, g) -> None:
        s = self.cal.alpha * g.prompt_tokens + (self.cal.a + self.cal.c) * g.completion_tokens
        self.service[at.task_id] = self.service.get(at.task_id, 0.0) + s


def delay_quantile(delay, p: float, dt: float = 0.01) -> float:
    pmf = delay.pmf(dt)
    acc = 0.0
    for i, w in enumerate(pmf):
        acc += w
        if acc >= p:
            return i * dt
    return len(pmf) * dt
