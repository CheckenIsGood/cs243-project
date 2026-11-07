"""Gate 1 regime check: model-predicted miss probability per policy, using MEASURED costs.

Inputs: an experiment config (delay model, context cap, token limits), the fitted latency coefficients
(`profile` output), and optionally a smoke run directory whose traces give measured token sizes and
planning time. Output: predicted miss rate vs. remaining-time-at-launch for each policy, the measured
calibration numbers, and suggested task deadlines (remaining + measured planning time).

These are model predictions from measured coefficients, not experiment results.

Usage: python scripts/regime_check.py --config configs/smoke.yaml --costs results/calibration/costs.json \
         [--run-dir results/real/smoke-gate1] [--raw-tokens N] [--n 4]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from dataclasses import replace
from pathlib import Path

from deadline_agent.experiment import ExperimentConfig
from deadline_agent.online_mdp import (ConsumptionMDP, required_summary_slack, static_batch, static_fold,
                                       static_incremental)
from deadline_agent.tools import DelayModel

POLS = {"optimal": None, "rss": required_summary_slack, "fold": static_fold,
        "batch2": lambda m: static_batch(m, 2), "incremental": static_incremental}


def trace_stats(run_dir: Path) -> dict:
    """Per-phase measured sizes. est = the runtime's chars/4 estimate; prompt_tokens = the server's count."""
    by_phase: dict[str, list[dict]] = {}
    plan_s, raw_toks = [], []
    for p in sorted((run_dir / "traces").glob("*.jsonl")):
        evs = [json.loads(line) for line in p.open()]
        t = {e["event"]: e["t"] for e in evs if e["event"] in ("planning_start", "planning_end")}
        if len(t) == 2:
            plan_s.append(t["planning_end"] - t["planning_start"])
        starts = {e["inv"]: e for e in evs if e["event"].endswith("_start") and "inv" in e}
        for e in evs:
            if e["event"].endswith("_end") and "prompt_tokens" in e:
                e = dict(e, est=starts.get(e["inv"], {}).get("est_prompt_tokens"), nres=len(e.get("call_ids") or []))
                by_phase.setdefault(e["event"][:-4], []).append(e)
            if e["event"] == "tool_return":
                raw_toks.append(e["tokens"])
    out = {"planning_s_median": st.median(plan_s) if plan_s else None,
           "planning_s_max": max(plan_s) if plan_s else None,
           "raw_result_tokens_median_est": st.median(raw_toks) if raw_toks else None}
    ratios = []
    for ph, es in by_phase.items():
        ratios += [e["prompt_tokens"] / e["est"] for e in es if e["est"]]
        out[ph] = {"n": len(es), "prompt_tokens_median": st.median(e["prompt_tokens"] for e in es),
                   "est_prompt_tokens_median": st.median(e["est"] or 0 for e in es),
                   "completion_tokens_median": st.median(e["completion_tokens"] for e in es),
                   "latency_s_median": st.median(e["latency_s"] for e in es),
                   "finish": sorted({e["finish"] for e in es})}
        if ph == "consume":
            out[ph]["completion_per_result_median"] = st.median(e["completion_tokens"] / max(1, e["nres"]) for e in es)
    out["server_over_est_token_ratio"] = st.median(ratios) if ratios else None
    return out


def calibrated(m, ts: dict):
    """Controller cost model in the runtime's token units (est), with measured output lengths."""
    upd = {}
    if ts.get("server_over_est_token_ratio"):
        upd["prefill_s_per_tok"] = m.prefill_s_per_tok * ts["server_over_est_token_ratio"]
    if ts.get("raw_result_tokens_median_est"):
        upd["raw_tokens"] = int(ts["raw_result_tokens_median_est"])
    if "final" in ts:
        upd["final_out"] = int(ts["final"]["completion_tokens_median"])
    if "consume" in ts:
        upd["consume_out_base"] = 0
        upd["consume_out_per_result"] = int(round(ts["consume"]["completion_per_result_median"]))
    return replace(m, **upd), upd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--costs", default=None, help="profile output JSON (overrides config costs_path)")
    ap.add_argument("--run-dir", default=None, help="smoke run dir for measured token sizes / planning time")
    ap.add_argument("--raw-tokens", type=int, default=None)
    ap.add_argument("--calibrate", action="store_true", help="use trace-measured sizes in the cost model")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--lo", type=float, default=4.0)
    ap.add_argument("--hi", type=float, default=30.0)
    ap.add_argument("--step", type=float, default=1.0)
    a = ap.parse_args()

    cfg = ExperimentConfig.load(a.config)
    if a.costs:
        cfg.costs_path = a.costs
    m = cfg.cost_model()
    if a.costs:
        fit = json.loads(Path(a.costs).read_text())
        print(f"latency fit: {fit['affine']}  r2={fit['r2']:.4f}  median_rel_err={fit['median_rel_err']:.3f}  n={fit['n']}")
    plan = 0.0
    if a.run_dir:
        ts = trace_stats(Path(a.run_dir))
        print("measured from traces:", json.dumps(ts, indent=1))
        plan = ts["planning_s_median"] or 0.0
        if a.calibrate:
            m, upd = calibrated(m, ts)
            print("calibrated cost overrides (paste into the pilot config `costs:`):", json.dumps(upd))
    if a.raw_tokens:
        m = replace(m, raw_tokens=a.raw_tokens)
    n = a.n or cfg.n_calls
    need = m.min_summaries_needed(n)
    print(f"\ncost model: {m}\nn={n}  min summaries needed to fit final context: {need}  "
          f"fold saving per summary delta={m.fold_saving_per_summary:.3f}s")
    pmf = DelayModel(**cfg.delay).pmf(cfg.dt)

    print("\npredicted miss probability (model, measured coefficients)")
    print(f"{'remain':>7} {'task_D':>7} " + "".join(f"{p:>12}" for p in POLS))
    rows = []
    R = a.lo
    while R <= a.hi + 1e-9:
        mdp = ConsumptionMDP(n, m, pmf, cfg.dt, R, objective="miss")
        vals = {p: (mdp.solve() if f is None else mdp.evaluate(f(mdp))) for p, f in POLS.items()}
        rows.append((R, vals))
        print(f"{R:7.1f} {R + plan:7.1f} " + "".join(f"{vals[p]:12.3f}" for p in POLS))
        R += a.step
    gap = max(rows, key=lambda r: r[1]["fold"] - r[1]["optimal"])
    print(f"\nlargest fold-vs-optimal gap: remain={gap[0]:.1f}s (task D≈{gap[0] + plan:.1f}s): "
          f"fold {gap[1]['fold']:.3f} vs optimal {gap[1]['optimal']:.3f} vs rss {gap[1]['rss']:.3f}")
    tight = next((r for r in rows if r[1]["optimal"] < 0.6), None)
    loose = next((r for r in rows if r[1]["fold"] < 0.05), None)
    print("suggested task deadlines (low / mid / high): "
          f"{(tight[0] + plan) if tight else 'n/a'} / {gap[0] + plan:.1f} / {(loose[0] + plan) if loose else 'n/a'}")


if __name__ == "__main__":
    main()
