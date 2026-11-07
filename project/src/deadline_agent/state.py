"""Typed runtime state. The scheduler sees only `Observation`; hidden tool timing lives in tools."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class CallStatus(str, enum.Enum):
    RUNNING = "running"
    RETURNED = "returned"  # returned, raw, not yet consumed
    SUMMARIZED = "summarized"  # consumed by a completed evidence-update invocation
    FOLDED = "folded"  # consumed raw by the completed final invocation
    ERROR = "error"


@dataclass
class ToolCall:
    call_id: str
    query: str
    tool: str = "search"
    launch_t: float = 0.0  # seconds since task start (monotonic)
    return_t: float | None = None
    status: CallStatus = CallStatus.RUNNING
    result_text: str | None = None  # visible only after return
    result_tokens: int | None = None
    source_ids: list[str] = field(default_factory=list)
    consumed_by: str | None = None  # model invocation id
    retrieval_s: float | None = None  # real retrieval time (recorded by the tool worker)
    injected_delay_s: float | None = None  # recorded at return, never visible before


@dataclass(frozen=True)
class Observation:
    """Everything a policy may use. No future return times, no unreleased payloads."""

    elapsed_since_launch_s: float
    remaining_s: float
    n_admitted: int
    returned_raw_ids: tuple[str, ...]  # returned, unconsumed, in return order
    n_summarized: int
    n_running: int
    raw_tokens: tuple[int, ...]  # token counts of returned_raw_ids
    preloaded_ids: tuple[str, ...] = ()  # raw results already prefilled into the KV cache (evidence log)

    @property
    def n_returned(self) -> int:
        return self.n_admitted - self.n_running


@dataclass(frozen=True)
class Consume:
    call_ids: tuple[str, ...]


@dataclass(frozen=True)
class Preload:
    """Prefill raw results into the KV cache (prefix-only request, 1 output token) by appending them to the
    evidence log. The results stay raw: they are folded by the final answer, never summarized afterwards."""

    call_ids: tuple[str, ...]


@dataclass(frozen=True)
class Wait:
    max_s: float | None = None  # wake at the next tool return or after max_s


@dataclass(frozen=True)
class Finalize:
    pass


Action = Consume | Preload | Wait | Finalize


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    submit_t: float
    first_token_t: float | None
    end_t: float
    cached_tokens: int | None = None


@dataclass
class RunResult:
    run_id: str
    task_id: str
    policy: str
    deadline_s: float
    valid: bool  # complete parseable answer, all calls consumed, by the deadline
    miss_reason: str | None
    completion_s: float | None  # time of final completion (even if late), None if never
    answer: str | None
    f1: float = 0.0
    em: float = 0.0
    n_calls: int = 0
    n_summarized: int = 0
    n_folded: int = 0
    n_model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    controller_s: float = 0.0
    trace_path: str = ""

    @property
    def deadline_quality(self) -> float:
        return self.f1 if self.valid else 0.0
