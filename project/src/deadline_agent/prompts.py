"""Prompt construction and output parsing, shared by every policy.

Layout is deterministic and append-friendly: instructions, question, then evidence. Default final prompt:
summaries (sorted by call id), then raw results (in return order). Evidence-log final prompt (used with prefix
caching / preloading): append-only, preloaded raws in preload order, then summaries in summarization order, then
the remaining raws in return order, so every preload request's prompt is a prefix of the final prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any

MAX_FACTS = 3  # bounds summary length: Gate 1 saw unbounded summaries hit max_tokens and fail to parse

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
    "required": ["queries"],
}


def plan_schema(n_calls: int) -> dict:
    return {"type": "object",
            "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": n_calls}},
            "required": ["queries"]}


CONSUME_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "call_id": {"type": "string"},
                    "facts": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_FACTS},
                    "sources": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
                },
                "required": ["call_id", "facts"],
            },
        }
    },
    "required": ["records"],
}


def consume_schema(call_ids: list[str]) -> dict:
    """Per-call schema: exactly one record per result, call_id constrained to this batch's ids. Gate 1 showed the
    model writing "call_id": "1" for "c1" and dropping records, which silently emptied summaries."""
    import copy

    sch = copy.deepcopy(CONSUME_SCHEMA)
    rec = sch["properties"]["records"]
    rec["minItems"] = rec["maxItems"] = len(call_ids)
    rec["items"]["properties"]["call_id"] = {"type": "string", "enum": list(call_ids)}
    return sch


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


def consume_messages(question: str, items: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    """items: (call_id, query, raw text)."""
    body = "\n\n".join(raw_block(c, q, t) for c, q, t in items)
    return [
        {"role": "system", "content": "Extract only the facts relevant to the question from each search "
         "result. Return JSON {\"records\": [{\"call_id\": ..., \"facts\": [...], \"sources\": [...]}]} with "
         "exactly one record per result, in the order given. At most 3 facts per result, each under 15 words; "
         "quote names, dates and numbers exactly."},
        {"role": "user", "content": f"Question: {question}\n\nSearch results:\n{body}"},
    ]


def summary_block(call_id: str, record: dict[str, Any]) -> str:
    facts = "; ".join(record.get("facts", [])) or "(no relevant facts)"
    srcs = ", ".join(record.get("sources", []))
    return f"[{call_id}] facts: {facts}" + (f" (sources: {srcs})" if srcs else "")


def final_messages(question: str, summaries: list[tuple[str, dict]], raws: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    parts = []
    if summaries:
        parts.append("Evidence summaries:\n" + "\n".join(summary_block(c, r) for c, r in summaries))
    if raws:
        parts.append("Search results:\n" + "\n\n".join(raw_block(c, q, t) for c, q, t in raws))
    evidence = "\n\n".join(parts) if parts else "(no evidence)"
    return [
        {"role": "system", "content": FINAL_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\n{evidence}"},
    ]


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


def _norm_id(raw: Any, call_ids: list[str]) -> str | None:
    if not isinstance(raw, (str, int)):
        return None
    t = str(raw).strip().strip("[]").strip()
    if t in call_ids:
        return t
    if f"c{t}" in call_ids:  # the model often drops the "c" prefix
        return f"c{t}"
    return None


def parse_consume(text: str, call_ids: list[str]) -> dict[str, dict]:
    obj = _json_object(text)
    recs = obj.get("records")
    if not isinstance(recs, list):
        raise ValueError("records missing")
    out = {}
    unmatched = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        facts = [f for f in r.get("facts", []) if isinstance(f, str)]
        srcs = [s for s in r.get("sources", []) if isinstance(s, str)]
        cid = _norm_id(r.get("call_id"), call_ids)
        if cid and cid not in out:
            out[cid] = {"facts": facts, "sources": srcs}
        else:
            unmatched.append({"facts": facts, "sources": srcs})
    for c in call_ids:  # unmatched records fill the remaining ids in order (records are requested in order)
        if c not in out and unmatched:
            out[c] = unmatched.pop(0)
    for c in call_ids:  # exposure is what counts; a missing record means "no relevant facts"
        out.setdefault(c, {"facts": [], "sources": []})
    return out


_RECORD_RE = re.compile(r'\{\s*"call_id"\s*:\s*"\[?(c?\d+)\]?"\s*,\s*"facts"\s*:\s*(\[[^\[\]]*\])')


def fallback_summary(raw_text: str, max_chars: int = 184) -> dict:
    """Mechanical stand-in when the model's summary is unusable: the start of the raw result (~46 tokens)."""
    txt = re.sub(r"<[^>]+>", " ", raw_text)
    txt = re.sub(r"\s+", " ", txt).strip()
    return {"facts": [txt[:max_chars] + ("…" if len(txt) > max_chars else "")], "sources": [], "fallback": True}


def salvage_consume(text: str, call_ids: list[str], raw_texts: dict[str, str]) -> tuple[dict[str, dict], list[str]]:
    """Never fails: parse normally; else keep every complete record from truncated JSON; any result still
    without a record gets a mechanical fallback. Returns (records, ids that used the fallback). Guarantees
    progress, so a bad generation can never make the runtime re-summarize the same results."""
    try:
        return parse_consume(text, call_ids), []
    except ValueError:
        pass
    out: dict[str, dict] = {}
    for m in _RECORD_RE.finditer(text):
        cid = _norm_id(m.group(1), call_ids)
        if cid and cid not in out:
            try:
                facts = [f for f in json.loads(m.group(2)) if isinstance(f, str)]
            except ValueError:
                continue
            out[cid] = {"facts": facts, "sources": []}
    fb = [c for c in call_ids if c not in out]
    for c in fb:
        out[c] = fallback_summary(raw_texts.get(c, ""))
    return out, fb


def parse_final(text: str) -> str:
    ans = _json_object(text).get("answer")
    if not isinstance(ans, str) or not ans.strip():
        raise ValueError("answer missing")
    return ans.strip()


CALL_ID_RE = re.compile(r"\[(c\d+)\]")
