"""VLLMBackend against a mock OpenAI-compatible SSE stream (no GPU, no network)."""

import asyncio
import json

import httpx

from deadline_agent.model_client import VLLMBackend


def sse(*events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def test_streaming_parse_usage_and_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        body = sse(
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": '{"answer": '}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": '"Paris"}'}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 7,
                                       "prompt_tokens_details": {"cached_tokens": 16}}},
        )
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    async def go():
        b = VLLMBackend("http://mock", "m")
        b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock")
        g = await b.generate([{"role": "user", "content": "q"}], 32, "final", schema={"type": "object"},
                             extra={"min_tokens": 4})
        await b.aclose()
        return g

    g = asyncio.run(go())
    assert g.text == '{"answer": "Paris"}'
    assert (g.prompt_tokens, g.completion_tokens, g.cached_tokens, g.finish_reason) == (42, 7, 16, "stop")
    assert g.first_token_t is not None and g.submit_t <= g.first_token_t <= g.end_t
    b = seen["body"]
    assert seen["path"] == "/v1/chat/completions"
    assert b["chat_template_kwargs"] == {"enable_thinking": False}  # thinking explicitly disabled
    assert b["temperature"] == 0.0 and b["stream"] and b["stream_options"]["include_usage"]
    assert b["response_format"]["type"] == "json_schema" and b["min_tokens"] == 4


def test_http_error_raises():
    async def go():
        b = VLLMBackend("http://mock", "m")
        b._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad")),
                                      base_url="http://mock")
        try:
            await b.generate([{"role": "user", "content": "q"}], 8, "final")
        finally:
            await b.aclose()

    try:
        asyncio.run(go())
    except httpx.HTTPStatusError:
        return
    raise AssertionError("expected HTTPStatusError")


def test_priority_is_sent_as_integer_milliseconds_preserving_order():
    """vLLM's request `priority` is an int (lower runs first); the serving layer passes deadlines in seconds."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["priority"])
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "{}"}, "finish_reason": "stop"}]}).encode(),
                              headers={"content-type": "text/event-stream"})

    async def go(prios):
        b = VLLMBackend("http://mock", "m")
        b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock")
        for p in prios:
            await b.generate([{"role": "user", "content": "q"}], 4, "final", extra={"priority": p})
        await b.aclose()

    prios = [12.3456, 12.3449, 0.0, 1794034308.25, -3.5]
    asyncio.run(go(prios))
    assert all(isinstance(x, int) for x in sent)
    assert sent == [12346, 12345, 0, 1794034308250, -3500]
    assert sorted(range(len(prios)), key=lambda i: sent[i]) == sorted(range(len(prios)), key=lambda i: prios[i])


def test_retries_once_when_connection_drops_before_any_output():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadError("connection closed by peer")
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": '{"answer": "x"}'},
                                                             "finish_reason": "stop"}]}).encode(),
                              headers={"content-type": "text/event-stream"})

    async def go():
        b = VLLMBackend("http://mock", "m")
        b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock")
        g = await b.generate([{"role": "user", "content": "q"}], 4, "final")
        await b.aclose()
        return g

    g = asyncio.run(go())
    assert len(calls) == 2 and g.text == '{"answer": "x"}' and g.finish_reason == "stop"


def test_second_connection_failure_is_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("down")

    async def go():
        b = VLLMBackend("http://mock", "m")
        b._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock")
        try:
            await b.generate([{"role": "user", "content": "q"}], 4, "final")
        finally:
            await b.aclose()

    try:
        asyncio.run(go())
    except httpx.ReadError:
        return
    raise AssertionError("expected ReadError after one retry")
