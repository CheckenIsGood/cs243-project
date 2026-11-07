"""Model backends: a deterministic fake (virtual clock, CPU tests) and a streaming vLLM client."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .clock import now
from .state import Generation
from .tools import approx_tokens


class ModelBackend(Protocol):
    name: str

    async def generate(self, messages: list[dict[str, str]], max_tokens: int, phase: str,
                       schema: dict | None = None, extra: dict | None = None) -> Generation: ...

    def count_tokens(self, text: str) -> int: ...


def messages_tokens(backend: ModelBackend, messages: list[dict[str, str]]) -> int:
    return sum(backend.count_tokens(m["content"]) + 4 for m in messages)


@dataclass(frozen=True)
class LatencyModel:
    """Batch-1 latency: overhead + prefill per prompt token + decode per output token (A40 + Qwen3-8B fit)."""

    overhead_s: float = 0.05
    prefill_s_per_tok: float = 1.8e-4
    decode_s_per_tok: float = 0.0293


# ---------------------------------------------------------------------------------------------
@dataclass
class FakeBackend:
    """Deterministic stand-in for correctness tests. Latency follows a batch-1 token cost model;
    outputs are mechanical extractions, so answer quality is NOT meaningful."""

    costs: LatencyModel = field(default_factory=LatencyModel)
    malformed_phases: set[str] = field(default_factory=set)
    prefix_cache: bool = False  # simulate vLLM automatic prefix caching (full 16-token blocks of a prior prompt)
    block_tokens: int = 16
    name: str = "fake"
    _seen: list[str] = field(default_factory=list)

    def count_tokens(self, text: str) -> int:
        return approx_tokens(text)

    def _output(self, messages: list[dict[str, str]], phase: str) -> str:
        user = messages[-1]["content"]
        if phase in self.malformed_phases:
            return '{"answer": "unterminated'
        if phase == "plan":
            q = user.removeprefix("Question: ")
            ents = re.findall(r"(?:the )?([A-Z][a-z]+(?: [A-Z][a-z]+)+)", q)
            queries = list(dict.fromkeys(ents))[:3] + [q]
            return json.dumps({"queries": queries[:4]})
        if phase == "preload":
            return "{"
        # final: a mechanical "answer" — first multi-word capitalized phrase in the evidence
        ev = user.split("\n", 1)[-1]
        m = re.search(r"([A-Z][a-z]+(?: [A-Z][a-z]+)+)", ev)
        return json.dumps({"answer": m.group(1) if m else "unknown"})

    async def generate(self, messages, max_tokens, phase, schema=None, extra=None) -> Generation:
        submit = now()
        text = self._output(messages, phase)
        prompt_toks = messages_tokens(self, messages)
        out_toks = approx_tokens(text)
        finish = "stop"
        if out_toks > max_tokens:
            text, out_toks, finish = text[: max_tokens * 4], max_tokens, "length"
        cached = self._cached_tokens(messages) if self.prefix_cache else None
        ttft = self.costs.overhead_s + self.costs.prefill_s_per_tok * (prompt_toks - (cached or 0))
        await asyncio.sleep(ttft)
        first = now()
        await asyncio.sleep(self.costs.decode_s_per_tok * out_toks)
        return Generation(text, prompt_toks, out_toks, finish, submit, first, now(), cached_tokens=cached)

    async def reset_prefix_cache(self) -> None:
        self._seen.clear()

    def _cached_tokens(self, messages) -> int:
        """Longest common prefix with any earlier prompt (serialized like a chat template), in whole blocks."""
        s = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        best = 0
        for prev in self._seen:
            n = 0
            for a, b in zip(prev, s):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        self._seen.append(s)
        toks = min(best // 4, messages_tokens(self, messages) - 1)
        return max(0, toks // self.block_tokens * self.block_tokens)


def _limits():
    import httpx

    # Idle keep-alive connections expire on the client (2 s) before uvicorn closes them (5 s): with equal
    # timeouts the client can reuse a connection the server is closing, which surfaces as httpx.ReadError.
    return httpx.Limits(max_connections=4096, max_keepalive_connections=512, keepalive_expiry=2.0)


# ---------------------------------------------------------------------------------------------
@dataclass
class VLLMBackend:
    """OpenAI-compatible streaming client for a local vLLM server.

    Cancellation (e.g. at the deadline) closes the HTTP stream; vLLM aborts requests on client disconnect,
    which the multi-agent runs rely on to stop serving late tasks.
    """

    base_url: str = "http://127.0.0.1:8000"
    model: str = "Qwen/Qwen3-8B"
    use_schema: bool = True
    timeout_s: float = 600.0
    name: str = "vllm"
    _client: Any = None

    def count_tokens(self, text: str) -> int:
        return approx_tokens(text)  # TODO(Gate 1): exact counts via the model tokenizer

    async def generate(self, messages, max_tokens, phase, schema=None, extra=None) -> Generation:
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s,
                                             limits=_limits())
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body.update(extra or {})
        if "priority" in body:  # vLLM's request priority is an int (lower runs first); keep order at 1 ms resolution
            body["priority"] = int(round(float(body["priority"]) * 1000))
        if self.use_schema and schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": phase, "schema": schema}}
        submit = now()
        for attempt in range(2):  # one retry if the connection drops before any output (stale pooled socket)
            first, chunks, usage, finish = None, [], {}, "unknown"
            try:
                await self._stream(body, chunks, usage)
                break
            except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError):
                if attempt == 1 or chunks:
                    raise
        first, finish = usage.pop("_first", None), usage.pop("_finish", "unknown")
        details = usage.get("prompt_tokens_details") or {}
        return Generation(
            "".join(chunks),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            finish,
            submit,
            first,
            now(),
            cached_tokens=details.get("cached_tokens"),
        )

    async def _stream(self, body: dict, chunks: list[str], usage: dict) -> None:
        """Stream one completion into `chunks`; usage fields, first-token time and finish reason go into `usage`."""
        async with self._client.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if ev.get("usage"):
                    usage.update(ev["usage"])
                for ch in ev.get("choices", []):
                    piece = ch.get("delta", {}).get("content")
                    if piece:
                        usage.setdefault("_first", now())
                        chunks.append(piece)
                    if ch.get("finish_reason"):
                        usage["_finish"] = ch["finish_reason"]

    async def reset_prefix_cache(self) -> None:
        """Requires the server to run with VLLM_SERVER_DEV_MODE=1 (exposes POST /reset_prefix_cache)."""
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s,
                                             limits=_limits())
        r = await self._client.post("/reset_prefix_cache")
        r.raise_for_status()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
