"""Shared record types."""

from __future__ import annotations

from dataclasses import dataclass


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
