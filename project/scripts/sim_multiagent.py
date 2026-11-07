"""Load sweep of serving-layer policies on the SIMULATED batching engine (virtual clock; no GPU).

Mixed workload of single-agent fan-out tasks and orchestrator+worker tasks over held-out HotpotQA questions,
Poisson arrivals, deadlines = arrival + U(1.5, 4) x isolated completion estimate. Same arrival trace and tool
delays for every policy. Engine: step = a + c*B + prefill; a + c = measured batch-1 decode time (29.3 ms);
c is NOT yet profiled, so results are reported for several c values.
Usage: python scripts/sim_multiagent.py --out results/sim/multiagent
"""

import argparse
import csv
import json
from pathlib import Path

from deadline_agent.clock import run_virtual
from deadline_agent.engine_sim import SimBackend, SimEngine
from deadline_agent.datasets import load_tasks
from deadline_agent.multiagent import CostEstimate, ServingLayer, expected_max_delay, make_workload, serve
from deadline_agent.tools import DelayModel


def run(tasks, policy, rate, c, max_seqs, seed, duration, preload, delay, kv=163904, a=None):
    a = 0.0293 - c if a is None else a

    async def go():
        e = SimEngine(a=a, c=c, max_seqs=max_seqs, kv_capacity=kv,
                      policy="fcfs" if policy.startswith("fcfs") else "priority")
        est = CostEstimate(tool_s=expected_max_delay(delay, 4))
        layer = ServingLayer(policy, est, slowdown=lambda: (e.a + e.c * len(e._running)) / (e.a + e.c))
        wl = make_workload(tasks, rate, duration, seed)
        res = await serve(wl, SimBackend(e), layer, delay, seed=1, preload=preload)
        await e.aclose()
        return res, e.stats
    res, st = run_virtual(go())
    n = len(res)
    by = lambda s: sum(r.status == s for r in res)
    return dict(policy=policy, rate=rate, c=c, max_seqs=max_seqs, kv=kv, seed=seed, preload=preload, n=n,
                ontime=by("ontime") / n, late=by("late") / n, dropped=by("dropped") / n,
                mean_batch=st.batch_token_sum / max(1, st.steps), max_batch=st.max_batch,
                gpu_busy_frac=st.busy_s / duration)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sim/multiagent")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 1, 1.5, 2, 3, 4])
    ap.add_argument("--cs", type=float, nargs="+", default=[0.0002, 0.0005, 0.001])
    ap.add_argument("--policies", nargs="+", default=["fcfs", "edf", "llf", "edf+ac", "llf+ac"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--a", type=float, default=None, help="fixed step overhead a (default: 0.0293 - c)")
    ap.add_argument("--kv", type=int, default=163904)
    ap.add_argument("--caps", type=int, nargs="*", default=None, help="max_seqs values (default: 256 and a/c)")
    ap.add_argument("--preload", action="store_true")
    a = ap.parse_args()
    tasks = [t for t, _ in load_tasks("heldout", 30)]
    delay = DelayModel(0.5, 4.0, 0.3, 4.0, 12.0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in a.cs:
        m = int((0.0293 - c if a.a is None else a.a) / c)  # Lemma 2's m = a/c
        for max_seqs in (a.caps or (256, m)):
            for rate in a.rates:
                for pol in a.policies:
                    for seed in a.seeds:
                        r = run(tasks, pol, rate, c, max_seqs, seed, a.duration, a.preload, delay, kv=a.kv, a=a.a)
                        rows.append(r)
                        print(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in r.items()}),
                              flush=True)
    with (out / "sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
