"""Batch-1 cost profiling against the live vLLM server: measure latency over a grid of prompt and output
lengths, then fit latency = overhead + prefill_per_tok * prompt + decode_per_tok * output by least squares.

Output lengths are forced with vLLM's `min_tokens` + `ignore_eos`, so each sample has exactly the requested
output. TTFT is recorded too, but it includes queueing/transport, so it is not an exact prefill measurement.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import numpy as np

from .model_client import VLLMBackend

FILLER = ("The river valley was settled by farmers who grew barley and kept sheep on the hills. "
          "Records from the parish describe floods, harvests, and the building of a stone bridge. ")


def filler_messages(target_tokens: int, nonce: int) -> list[dict[str, str]]:
    # ~4 chars per token; the nonce at the start defeats any prefix-cache reuse between samples
    text = f"[{nonce}] " + (FILLER * (1 + target_tokens * 4 // len(FILLER)))[: target_tokens * 4]
    return [{"role": "system", "content": "Continue the text."}, {"role": "user", "content": text}]


async def profile(backend: VLLMBackend, prompt_grid: list[int], out_grid: list[int], reps: int, seed: int = 0):
    rng = random.Random(seed)
    grid = [(p, o) for p in prompt_grid for o in out_grid for _ in range(reps)]
    rng.shuffle(grid)
    for _ in range(3):  # warmup, excluded
        await backend.generate(filler_messages(256, rng.randrange(10**9)), 16, "warmup",
                               extra={"min_tokens": 16, "ignore_eos": True})
    samples = []
    for i, (p, o) in enumerate(grid):
        g = await backend.generate(filler_messages(p, rng.randrange(10**9)), o, "profile",
                                   extra={"min_tokens": o, "ignore_eos": True})
        samples.append({"target_prompt": p, "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens,
                        "latency_s": g.end_t - g.submit_t,
                        "ttft_s": None if g.first_token_t is None else g.first_token_t - g.submit_t,
                        "cached_tokens": g.cached_tokens, "finish": g.finish_reason})
        print(f"[{i + 1}/{len(grid)}] prompt={g.prompt_tokens} out={g.completion_tokens} "
              f"latency={samples[-1]['latency_s']:.3f}s", flush=True)
    return samples


def fit_affine(samples: list[dict]) -> dict:
    X = np.array([[1.0, s["prompt_tokens"], s["completion_tokens"]] for s in samples])
    y = np.array([s["latency_s"] for s in samples])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return {"affine": {"overhead_s": float(coef[0]), "prefill_s_per_tok": float(coef[1]),
                       "decode_s_per_tok": float(coef[2])},
            "r2": float(r2), "max_abs_resid_s": float(np.abs(resid).max()),
            "median_rel_err": float(np.median(np.abs(resid) / y)), "n": len(samples)}


def run_profile(base_url: str, model: str, out_path: str, prompt_grid=(256, 1024, 2048, 3072),
                out_grid=(16, 64, 128, 192), reps: int = 3) -> dict:
    async def go():
        b = VLLMBackend(base_url, model, use_schema=False)
        try:
            return await profile(b, list(prompt_grid), list(out_grid), reps)
        finally:
            await b.aclose()

    samples = asyncio.run(go())
    fit = fit_affine(samples)
    fit["samples"] = samples
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(fit, indent=1))
    return fit


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fit latency = overhead + prefill*prompt + decode*output on the server")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="results/calibration/cost_profile.json")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    f = run_profile(a.base_url, a.model, a.out, reps=a.reps)
    print(json.dumps({k: v for k, v in f.items() if k != "samples"}, indent=1))
