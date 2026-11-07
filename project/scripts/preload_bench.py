"""KV-preload microbenchmark on the live vLLM server (prefix caching ON, VLLM_SERVER_DEV_MODE=1).

For an evidence log of k tool results (~`chunk` tokens each, real HotpotQA paragraphs), measure:
  cold:      reset cache -> final request over the whole log                 (what fold pays after the last return)
  preloaded: reset cache -> k-1 preload requests (1 output token each)       (done while tools are still running)
             -> final request (only the last result + tail must be prefilled)
Reports final-request TTFT/latency, server cached_tokens, and the preload requests' total time (the work moved
off the critical path). Exact same prompts as the runtime (prompts.evidence_log_messages / raw_block).

Usage (on the pod):  python scripts/preload_bench.py --sizes 1 2 4 8 16 --reps 3 --out results/preload_bench
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import time
from pathlib import Path

from deadline_agent.datasets import load_hotpot
from deadline_agent.model_client import VLLMBackend
from deadline_agent.prompts import FINAL_SCHEMA, evidence_log_messages, raw_block
from deadline_agent.tools import approx_tokens


def make_chunks(examples, n: int, chunk_tokens: int, rng: random.Random) -> list[str]:
    paras = [(t, f"{t}: " + " ".join(x.strip() for x in s)) for e in examples for t, s in e["context"]]
    rng.shuffle(paras)
    out, cur, i = [], [], 0
    while len(out) < n:
        title, text = paras[i % len(paras)]
        i += 1
        cur.append(f'<source id="{title}">\n{text}\n</source>')
        if approx_tokens("\n".join(cur)) >= chunk_tokens:
            out.append(raw_block(f"c{len(out) + 1}", title, "\n".join(cur)))
            cur = []
    return out


async def one(be: VLLMBackend, question: str, chunks: list[str], mode: str, final_max: int) -> dict:
    await be.reset_prefix_cache()
    pre_s, pre_cached, pre_prompt = 0.0, 0, 0
    if mode == "preloaded":
        for j in range(1, len(chunks)):  # the last result returns last: it cannot be preloaded
            g = await be.generate(evidence_log_messages(question, chunks[:j]), 1, "preload", None)
            pre_s += g.end_t - g.submit_t
            pre_cached += g.cached_tokens or 0
            pre_prompt += g.prompt_tokens
    g = await be.generate(evidence_log_messages(question, chunks), final_max, "final", FINAL_SCHEMA)
    return dict(mode=mode, final_ttft_s=(g.first_token_t or g.end_t) - g.submit_t, final_latency_s=g.end_t - g.submit_t,
                final_prompt_tokens=g.prompt_tokens, final_cached_tokens=g.cached_tokens or 0,
                final_out_tokens=g.completion_tokens, preload_total_s=pre_s, preload_requests=max(0, len(chunks) - 1)
                if mode == "preloaded" else 0, preload_prompt_tokens=pre_prompt, preload_cached_tokens=pre_cached)


async def main_async(a) -> None:
    rng = random.Random(a.seed)
    examples = load_hotpot(a.data)
    be = VLLMBackend(a.base_url, a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        # warm-up (CUDA graphs, schema compilation) — not recorded
        warm = make_chunks(examples, 2, a.chunk, rng)
        for m in ("cold", "preloaded"):
            await one(be, "warm-up?", warm, m, a.final_max)
        jobs = [(k, r, m) for k in a.sizes for r in range(a.reps) for m in ("cold", "preloaded")]
        rng.shuffle(jobs)
        cache: dict[tuple[int, int], tuple[str, list[str]]] = {}
        for k, r, m in jobs:
            if (k, r) not in cache:
                ex = examples[rng.randrange(len(examples))]
                cache[(k, r)] = (ex["question"], make_chunks(examples, k, a.chunk, rng))
            q, ch = cache[(k, r)]
            res = await one(be, q, ch, m, a.final_max)
            res.update(k=k, rep=r, est_evidence_tokens=sum(approx_tokens(c) for c in ch))
            rows.append(res)
            print(json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in res.items()}), flush=True)
    finally:
        await be.aclose()
    with (out / "preload_bench.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "meta.json").write_text(json.dumps(vars(a) | {"date_utc": time.strftime("%FT%TZ", time.gmtime())}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--chunk", type=int, default=576, help="approx tokens per tool result (pilot result cap)")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--final-max", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data/raw/hotpot_distractor_validation.parquet")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--out", default="results/preload_bench")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
