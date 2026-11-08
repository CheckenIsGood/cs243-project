"""Prompt construction and output parsing for the agent programs.

The answer prompt is an append-only evidence log (raw tool results in return order), so each KV-preload
request's prompt is a prefix of the final prompt and vLLM prefix caching reuses its KV blocks.
"""

from __future__ import annotations

import json
from typing import Any

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
    "required": ["queries"],
}


def plan_schema(n_calls: int) -> dict:
    return {"type": "object",
            "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": n_calls}},
            "required": ["queries"]}


FINAL_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

TOOL_DESCRIPTION = "search(query): returns the top passages from a small document collection, with source ids."


def plan_messages(question: str, max_queries: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You plan searches for a question-answering agent. "
         f"Return JSON {{\"queries\": [...]}} with at most {max_queries} short, distinct search queries, "
         "most useful first. Tool: " + TOOL_DESCRIPTION},
        {"role": "user", "content": f"Question: {question}"},
    ]


def raw_block(call_id: str, query: str, text: str) -> str:
    return f"[{call_id}] search(\"{query}\")\n{text}"


FINAL_SYSTEM = ("Answer the question using the evidence. Return JSON "
                "{\"answer\": ...} where answer is a short span (an entity, date, number, or yes/no).")


def evidence_log_messages(question: str, entries: list[str]) -> list[dict[str, str]]:
    """Append-only evidence log. For entries e1..ek, the user content is a strict string prefix of the content
    for e1..ek+1, so the KV blocks of a preload request are reused by later requests (vLLM prefix caching)."""
    body = "".join(e + "\n\n" for e in entries)
    return [{"role": "system", "content": FINAL_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nEvidence:\n\n{body}"}]


def _json_object(text: str) -> Any:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object")
    return json.loads(text[a : b + 1])


def parse_plan(text: str) -> list[str]:
    obj = _json_object(text)
    qs = obj.get("queries")
    if not isinstance(qs, list):
        raise ValueError("queries missing")
    return [q.strip() for q in qs if isinstance(q, str) and q.strip()]


def parse_final(text: str) -> str:
    ans = _json_object(text).get("answer")
    if not isinstance(ans, str) or not ans.strip():
        raise ValueError("answer missing")
    return ans.strip()
