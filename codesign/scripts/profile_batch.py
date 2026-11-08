"""Profile decode step time vs batch size on the live vLLM server: tau(B) = a + c*B (Lemma 2's parameters).
Sends B concurrent streaming requests (same prompt length, fixed output length, ignore_eos), measures each
request's decode time per token after its first token, and fits per-token time = a + c*B by least squares.
Usage (on the pod): python scripts/profile_batch.py --out results/calibration/batch_profile.json
"""

import argparse
import asyncio
import json
import statistics as st
import time

import httpx


async def one(client, model, prompt, out):
    body = {"model": model, "prompt": prompt, "max_tokens": out, "temperature": 0.0, "stream": True,
            "ignore_eos": True}
    t0, first, n = time.perf_counter(), None, 0
    async with client.stream("POST", "/v1/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                ev = json.loads(line[6:])
                if ev.get("choices") and ev["choices"][0].get("text"):
                    n += 1
                    if first is None:
                        first = time.perf_counter()
    end = time.perf_counter()
    # ignore_eos: exactly `out` tokens are generated; several may share one SSE event, so do not divide by n
    return (end - first) / max(1, out - 1)


async def main_async(a):
    words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 400).split()
    async with httpx.AsyncClient(base_url=a.base_url, timeout=600,
                                 limits=httpx.Limits(max_connections=4096)) as client:
        await one(client, a.model, " ".join(words[:a.prompt]), 16)  # warm-up
        rows = []
        for B in a.batches:
            for rep in range(a.reps):
                # distinct prompts so prefix caching cannot share work across the batch
                prompts = [f"[{B}-{rep}-{i}] " + " ".join(words[i % 7: i % 7 + a.prompt]) for i in range(B)]
                per_tok = await asyncio.gather(*(one(client, a.model, p, a.out) for p in prompts))
                rows.append({"B": B, "rep": rep, "per_token_s_median": st.median(per_tok), "per_token_s_max": max(per_tok)})
                print(json.dumps(rows[-1]), flush=True)
    xs = [r["B"] for r in rows]
    ys = [r["per_token_s_median"] for r in rows]
    mx, my = st.mean(xs), st.mean(ys)
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a0 = my - c * mx
    ss_res = sum((y - (a0 + c * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    fit = {"a": a0, "c": c, "r2": 1 - ss_res / ss_tot if ss_tot else 1.0, "m_a_over_c": a0 / c if c > 0 else None,
           "rows": rows, "prompt_tokens_approx": a.prompt, "out_tokens": a.out}
    with open(a.out_path, "w") as f:
        json.dump(fit, f, indent=1)
    print(json.dumps({k: v for k, v in fit.items() if k != "rows"}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--prompt", type=int, default=500, help="prompt length in words (~tokens)")
    p.add_argument("--out", type=int, default=128)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--out-path", default="results/calibration/batch_profile.json")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
