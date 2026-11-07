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


def test_experiment_runs_consecutive_vllm_tasks_in_separate_event_loops(tmp_path, monkeypatch):
    """Regression: a shared VLLMBackend reused its HTTP pool across asyncio.run() loops ('Event loop is closed')."""
    import asyncio

    import httpx

    from deadline_agent import experiment as ex
    from deadline_agent.model_client import VLLMBackend

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        body = _json.loads(request.content)
        phase = body.get("response_format", {}).get("json_schema", {}).get("name", "")
        content = {"plan": '{"queries": ["Arden Valley Rowing Club", "Merrow Bay Sailing Society"]}',
                   "consume": '{"records": []}'}.get(phase, '{"answer": "Arden Valley Rowing Club"}')
        sse = (f'data: {_json.dumps({"choices": [{"delta": {"content": content}, "finish_reason": "stop"}]})}\n\n'
               f'data: {_json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})}\n\n'
               "data: [DONE]\n\n")
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    orig_generate = VLLMBackend.generate

    async def generate(self, *a, **kw):
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, transport=httpx.MockTransport(handler))
        return await orig_generate(self, *a, **kw)

    monkeypatch.setattr(VLLMBackend, "generate", generate)
    cfg = ex.ExperimentConfig(backend="vllm", split="fixture", policies=["fold", "rss"], deadlines_s=[30.0],
                              delay=dict(lo=0.01, hi=0.02, slow_p=0.0, slow_lo=0.0, slow_hi=0.0),
                              out_dir=str(tmp_path), dt=0.01)
    out = ex.run_experiment(cfg, "loops")
    rows = (out / "results.csv").read_text().strip().splitlines()
    assert len(rows) == 3  # header + 2 runs, each in its own asyncio.run() loop
    assert all(",True," in r for r in rows[1:])
