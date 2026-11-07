"""Multi-tenant, multi-agent serving on the LIVE vLLM server (start it with SCHED=priority, MAXSEQS>1,
PREFIX_CACHE=1). Same workload generator, agent programs and serving layer as scripts/sim_multiagent.py.
FCFS is run on the same priority server by sending priority 0 to every request (ties break by arrival).

Two uses:
- policy sweep (default workload: 30 questions, fan-out + orchestrator, 4 searches, deadlines enforced):
    python scripts/real_multiagent.py --rates 1 2 3 4 --policies fcfs edf edf+ac llf+ac --duration 90
- KV preloading under contention (fan-out only, 8 searches, large results, many distinct questions, non-binding
  deadlines so every task completes and miss(D) can be computed afterwards for any D):
    python scripts/real_multiagent.py --preset preload --rates 0.15 0.3 0.45 0.6 --preload-modes off on low
"""

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import httpx

from deadline_agent.datasets import load_tasks
from deadline_agent.metrics import exact_match, f1_score
from deadline_agent.model_client import VLLMBackend
from deadline_agent.multiagent import CostEstimate, ServingLayer, expected_max_delay, make_workload, serve
from deadline_agent.tools import DelayModel

PRESETS = {
    "policy": dict(shapes=["fanout", "orchestrator"], width=4, top_k=4, result_cap=576, n_questions=30,
                   manifest="data/manifests/pilot.json", n_heldout=30, distractors=0, slack=[1.5, 4.0],
                   isolated_fanout=8.0, isolated_orchestrator=10.0, final_prompt_tokens=2500),
    "preload": dict(shapes=["fanout"], width=8, top_k=10, result_cap=1400, n_questions=400,
                    manifest="data/manifests/multitenant.json", n_heldout=400, distractors=90, slack=[1000.0, 1000.0],
                    isolated_fanout=15.0, isolated_orchestrator=15.0, final_prompt_tokens=11000),
}
PRELOAD_MODES = {"off": False, "on": True, "low": "low", "jit": "jit"}


class Counting:
    """Wraps a backend and counts in-flight requests (we are the only client, so this is the engine's load)."""

    def __init__(self, inner):
        self.inner, self.inflight, self.name = inner, 0, inner.name

    def count_tokens(self, text):
        return self.inner.count_tokens(text)

    async def generate(self, *args, **kw):
        self.inflight += 1
        try:
            return await self.inner.generate(*args, **kw)
        finally:
            self.inflight -= 1


async def one_run(tasks, policy, rate, cfg, seed, a, c, base_url, model, delay, preload):
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as cl:
        (await cl.post("/reset_prefix_cache")).raise_for_status()
    be = Counting(VLLMBackend(base_url, model))
    est = CostEstimate(decode_s_per_tok=a + c, tool_s=expected_max_delay(delay, cfg["width"]))
    est.prompt_tokens["final"] = cfg["final_prompt_tokens"]
    layer = ServingLayer(policy, est, slowdown=lambda: (a + c * be.inflight) / (a + c))
    wl = make_workload(tasks, rate, cfg["duration"], seed, shapes=tuple(cfg["shapes"]), width=cfg["width"],
                       slack=tuple(cfg["slack"]),
                       isolated={"fanout": cfg["isolated_fanout"], "orchestrator": cfg["isolated_orchestrator"]})
    t0 = time.perf_counter()
    res = await serve(wl, be, layer, delay, seed=1, preload=preload, inline_tools=False, top_k=cfg["top_k"],
                      result_cap=cfg["result_cap"])
    wall = time.perf_counter() - t0
    await be.inner.aclose()
    return res, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="policy")
    ap.add_argument("--rates", type=float, nargs="+", default=[1, 2, 4])
    ap.add_argument("--policies", nargs="+", default=["fcfs"])
    ap.add_argument("--preload-modes", nargs="+", choices=sorted(PRELOAD_MODES), default=["off"])
    ap.add_argument("--duration", type=float, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="results/calibration/batch_profile.json")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="results/real/multiagent")
    a = ap.parse_args()
    cfg = dict(PRESETS[a.preset], duration=a.duration)
    prof = json.loads(Path(a.profile).read_text())
    pairs = load_tasks("heldout", cfg["n_questions"], manifest_path=cfg["manifest"], n_heldout=cfg["n_heldout"],
                       distractor_passages=cfg["distractors"])
    tasks = [t for t, _ in pairs]
    gold = {g.task_id: g.answer for _, g in pairs}
    delay = DelayModel(0.5, 4.0, 0.3, 4.0, 12.0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(vars(a), **cfg), indent=1))
    summ = []
    for rate in a.rates:
        for pol in a.policies:
            for pm in a.preload_modes:
                res, wall = asyncio.run(one_run(tasks, pol, rate, cfg, a.seed, prof["a"], prof["c"], a.base_url,
                                                a.model, delay, PRELOAD_MODES[pm]))
                rows = []
                for r in res:
                    g = gold[r.task_id.split("#")[0]]
                    ans = r.answer if r.status == "ontime" else ""
                    rows.append(dict(vars(r), preload=pm, latency=None if r.end is None else r.end - r.arrival,
                                     gold=g, em=exact_match(ans or "", g), f1=f1_score(ans or "", g)))
                tag = f"r{rate:g}_{pol}" + ("" if pm == "off" and len(a.preload_modes) == 1 else f"_pre-{pm}")
                with (out / f"tasks_{tag}.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                n = len(rows)
                lat = sorted(x["latency"] for x in rows if x["latency"] is not None)
                s = dict(rate=rate, policy=pol, preload=pm, n=n, wall_s=round(wall, 1),
                         **{k: round(sum(x["status"] == k for x in rows) / n, 4)
                            for k in ("ontime", "late", "dropped", "error")},
                         lat_mean=round(sum(lat) / len(lat), 3) if lat else None,
                         lat_p50=round(lat[len(lat) // 2], 3) if lat else None,
                         lat_p90=round(lat[int(0.9 * (len(lat) - 1))], 3) if lat else None,
                         f1=round(sum(x["f1"] for x in rows) / n, 4), em=round(sum(x["em"] for x in rows) / n, 4))
                summ.append(s)
                print(json.dumps(s), flush=True)
                with (out / "summary.csv").open("w", newline="") as f:  # rewritten after every run
                    w = csv.DictWriter(f, fieldnames=list(summ[0]))
                    w.writeheader()
                    w.writerows(summ)


if __name__ == "__main__":
    main()
