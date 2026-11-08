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
from deadline_agent.preload_controller import PreloadController
from deadline_agent.tools import DelayModel
from codesign.client import RULES, Calib, CodesignLayer, ShapeEst, delay_quantile, task_nodes

PRESETS = {
    "policy": dict(shapes=["fanout", "orchestrator"], width=4, top_k=4, result_cap=576, n_questions=30,
                   manifest="data/manifests/pilot.json", n_heldout=30, distractors=0, slack=[1.5, 4.0],
                   isolated_fanout=8.0, isolated_orchestrator=10.0, final_prompt_tokens=2500),
    "preload": dict(shapes=["fanout"], width=8, top_k=10, result_cap=1400, n_questions=400,
                    manifest="data/manifests/multitenant.json", n_heldout=400, distractors=90, slack=[1000.0, 1000.0],
                    isolated_fanout=15.0, isolated_orchestrator=15.0, final_prompt_tokens=11000),
    # co-design sweep: heterogeneous agent tasks (fan-out of 2/4/8 searches, ReAct loops of 2/4/6 turns), so task
    # indices differ; results capped at 1000 tokens; non-binding deadlines (every task completes)
    "codesign": dict(shapes=["fanout", "react"], width={"fanout": [2, 4, 8], "react": [2, 4, 6]}, top_k=10,
                     result_cap=1000, n_questions=400, manifest="data/manifests/multitenant.json", n_heldout=400,
                     distractors=90, slack=[1000.0, 1000.0], isolated_fanout=15.0, isolated_orchestrator=15.0,
                     final_prompt_tokens=8200),
}
PRELOAD_MODES = {"off": False, "on": True, "low": "low", "jit": "jit", "chain": "chain", "ctrl": "ctrl"}


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


async def server_kv_tokens(base_url) -> int | None:
    """KV capacity in tokens from vLLM's /metrics (cache_config_info: num_gpu_blocks x block_size)."""
    import re
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as cl:
        txt = (await cl.get("/metrics")).text
    for line in txt.splitlines():
        if line.startswith("vllm:cache_config_info"):
            nb = re.search(r'num_gpu_blocks="(\d+)"', line)
            bs = re.search(r'block_size="(\d+)"', line)
            ov = re.search(r'num_gpu_blocks_override="(\d+)"', line)
            n = int(ov.group(1)) if ov else (int(nb.group(1)) if nb else None)
            if n and bs:
                return n * int(bs.group(1))
    return None


async def one_run(tasks, policy, rate, cfg, seed, a, c, base_url, model, delay, preload, kv_tokens=None, ctrl_out=None,
                  rule=None, ttl_q=0.0, alpha=1.14e-4, kappa=0.0):
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as cl:
        (await cl.post("/reset_prefix_cache")).raise_for_status()
    controller = None
    if preload == "ctrl":
        M = kv_tokens or await server_kv_tokens(base_url)
        if not M:
            raise SystemExit("ctrl mode needs the server KV capacity: pass --kv-tokens")
        controller = PreloadController(M=M, delay=(delay.lo, delay.hi, delay.slow_p, delay.slow_lo, delay.slow_hi))
    be = Counting(VLLMBackend(base_url, model))
    est = CostEstimate(decode_s_per_tok=a + c, tool_s=expected_max_delay(delay, max(max(v) for v in cfg["width"].values()) if isinstance(cfg["width"], dict) else cfg["width"]))
    est.prompt_tokens["final"] = cfg["final_prompt_tokens"]
    layer = ServingLayer(policy, est, slowdown=lambda: (a + c * be.inflight) / (a + c))
    if rule is not None:  # co-design client: priority rule + vllm_xargs for codesign.vllm_sched
        M = kv_tokens or await server_kv_tokens(base_url)
        e = ShapeEst(result_tokens=cfg["result_cap"])
        widths = cfg["width"] if isinstance(cfg["width"], dict) else {s: [cfg["width"]] for s in cfg["shapes"]}
        L_max = max(rho + q + X for s in cfg["shapes"] for w in widths[s] for rho, q, X in task_nodes(s, w, e))
        layer = CodesignLayer(rule, Calib(a, c, alpha), e, M, L_max,
                              ttl=delay_quantile(delay, ttl_q) if ttl_q > 0 else 0.0, kappa=kappa)
    wl = make_workload(tasks, rate, cfg["duration"], seed, shapes=tuple(cfg["shapes"]), width=cfg["width"],
                       slack=tuple(cfg["slack"]),
                       isolated={"fanout": cfg["isolated_fanout"], "orchestrator": cfg["isolated_orchestrator"]})
    t0 = time.perf_counter()
    res = await serve(wl, be, layer, delay, seed=1, preload=preload, inline_tools=False, top_k=cfg["top_k"],
                      result_cap=cfg["result_cap"], controller=controller)
    wall = time.perf_counter() - t0
    if controller is not None and ctrl_out is not None:
        Path(ctrl_out).write_text(json.dumps(controller.summary(), indent=1))
        print(json.dumps(dict(ctrl_M=controller.M, **controller.summary()["stats"],
                              sigma_hat=round(controller.sigma_hat()[0], 3))), flush=True)
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
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override preset fields, e.g. --set result_cap=200 width=4 (values parsed as JSON)")
    ap.add_argument("--kv-tokens", type=int, default=None,
                    help="server KV capacity for the ctrl mode (default: read from the server's /metrics)")
    ap.add_argument("--rules", nargs="+", choices=RULES, default=None,
                    help="co-design priority rules (codesign.client); one run per rule, policy fixed to fcfs")
    ap.add_argument("--age-kappa", type=float, default=0.0,
                    help="zprime_age rule: priority = task arrival + kappa * Z'/w (seconds)")
    ap.add_argument("--ttl-quantile", type=float, default=0.0,
                    help="cd_ttl = this quantile of the tool-delay law, on calls followed by a tool wait (0 = none)")
    ap.add_argument("--tag", default="", help="suffix for output file names (e.g. the engine config)")
    ap.add_argument("--delay", type=float, nargs="+", default=[0.5, 4.0, 0.3, 4.0, 12.0],
                    help="DelayModel arguments for the tool latency distribution")
    a = ap.parse_args()
    cfg = dict(PRESETS[a.preset], duration=a.duration)
    for kv in a.set:
        k, _, v = kv.partition("=")
        if k not in cfg:
            raise SystemExit(f"unknown preset field {k!r}; one of {sorted(cfg)}")
        cfg[k] = json.loads(v)
    prof = json.loads(Path(a.profile).read_text())
    pairs = load_tasks("heldout", cfg["n_questions"], manifest_path=cfg["manifest"], n_heldout=cfg["n_heldout"],
                       distractor_passages=cfg["distractors"])
    tasks = [t for t, _ in pairs]
    gold = {g.task_id: g.answer for _, g in pairs}
    delay = DelayModel(*a.delay)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(vars(a), **cfg), indent=1))
    summ = []
    rules = a.rules or [None]
    for rate in a.rates:
        for pol in (a.policies if a.rules is None else [f"rule-{r}" for r in rules]):
            for pm in a.preload_modes:
                rule = pol[5:] if pol.startswith("rule-") else None
                res, wall = asyncio.run(one_run(tasks, "fcfs" if rule else pol, rate, cfg, a.seed, prof["a"],
                                                prof["c"], a.base_url, a.model, delay, PRELOAD_MODES[pm], a.kv_tokens,
                                                out / f"ctrl_r{rate:g}_{pol}.json", rule=rule,
                                                ttl_q=a.ttl_quantile, kappa=a.age_kappa))
                rows = []
                for r in res:
                    g = gold[r.task_id.split("#")[0]]
                    ans = r.answer if r.status == "ontime" else ""
                    rows.append(dict(vars(r), preload=pm, latency=None if r.end is None else r.end - r.arrival,
                                     gold=g, em=exact_match(ans or "", g), f1=f1_score(ans or "", g)))
                tag = f"r{rate:g}_{pol}" + ("" if pm == "off" and len(a.preload_modes) == 1 else f"_pre-{pm}")
                tag += f"_{a.tag}" if a.tag else ""
                with (out / f"tasks_{tag}.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                n = len(rows)
                lat = sorted(x["latency"] for x in rows if x["latency"] is not None)
                wlat = [x["latency"] for x in rows if x["latency"] is not None]
                s = dict(rate=rate, policy=pol, preload=pm, tag=a.tag, n=n, wall_s=round(wall, 1),
                         **{k: round(sum(x["status"] == k for x in rows) / n, 4)
                            for k in ("ontime", "late", "dropped", "error")},
                         lat_mean=round(sum(lat) / len(lat), 3) if lat else None,
                         lat_p50=round(lat[len(lat) // 2], 3) if lat else None,
                         lat_p90=round(lat[int(0.9 * (len(lat) - 1))], 3) if lat else None,
                         f1=round(sum(x["f1"] for x in rows) / n, 4), em=round(sum(x["em"] for x in rows) / n, 4))
                summ.append(s)
                print(json.dumps(s), flush=True)
                with (out / f"summary{'_' + a.tag if a.tag else ''}.csv").open("w", newline="") as f:  # rewritten after every run
                    w = csv.DictWriter(f, fieldnames=list(summ[0]))
                    w.writeheader()
                    w.writerows(summ)


if __name__ == "__main__":
    main()
