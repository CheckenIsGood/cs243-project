"""Critical-path time breakdown per policy, from traces, over ALL runs (the same questions for every policy).

Each run is cut at its end time T = completion time if on time, else the deadline (a missed run is charged the
whole deadline). Then  T = planning + tool_wait + post_summarize + post_preload + final, where
  planning        plan request (before tools launch)
  tool_wait       tool launch -> last tool return before T (model may work in parallel; see 'hidden_*')
  post_summarize  summarization time after the last return (on the critical path)
  post_preload    preload (prefill-only) time after the last return
  final           final answer generation (all attempts) and any idle gaps after the last return
hidden_summarize / hidden_preload: model work done while tools were still running (off the critical path).
Averaging over all runs avoids comparing policies on different subsets (on-time-only averages favour a policy's
lucky fast-tool runs).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .plot import COLORS, LABELS  # noqa: E402

PARTS = ["planning", "tool_wait", "post_summarize", "post_preload", "final"]
PART_COLORS = {"planning": "#bdbdbd", "tool_wait": "#9ecae1", "post_summarize": "#fd8d3c",
               "post_preload": "#74c476", "final": "#756bb1"}
PART_LABELS = {"planning": "planning", "tool_wait": "waiting for tools", "post_summarize": "summarizing after last return",
               "post_preload": "preloading after last return", "final": "final answer / idle after last return"}


def _intervals(ev: list[dict], phase: str, t_end: float) -> list[tuple[float, float]]:
    """Model-call intervals clipped to [0, t_end]; a call still running at t_end (cut off) ends at t_end."""
    ends = {e["inv"]: e["t"] for e in ev if e["event"] == f"{phase}_end"}
    out = []
    for e in ev:
        if e["event"] == f"{phase}_start" and e["t"] < t_end:
            out.append((e["t"], min(ends.get(e["inv"], t_end), t_end)))
    return out


def _split(iv: list[tuple[float, float]], cut: float) -> tuple[float, float]:
    before = sum(max(0.0, min(b, cut) - a) for a, b in iv)
    after = sum(max(0.0, b - max(a, cut)) for a, b in iv)
    return before, after


def run_breakdown(trace_path: str | Path, deadline: float | None = None) -> dict | None:
    ev = [json.loads(line) for line in open(trace_path)]
    end = next((e for e in ev if e["event"] == "task_end"), None)
    if not end:
        return None
    valid = bool(end.get("valid"))
    if valid:
        t_end = end.get("completion_s") or end["t"]
    else:
        start = next((e for e in ev if e["event"] == "task_start"), {})
        t_end = deadline if deadline is not None else start.get("deadline_s", end["t"])
    launches = [e["t"] for e in ev if e["event"] == "tool_launch" and e["t"] <= t_end]
    if not launches:  # planning itself overran the deadline
        return dict(planning=t_end, tool_wait=0.0, post_summarize=0.0, post_preload=0.0, final=0.0,
                    hidden_summarize=0.0, hidden_preload=0.0, completion=t_end, valid=valid)
    t_launch = min(launches)
    # A return is timestamped when the event loop drains it, which is late if the model was busy generating.
    # Use the tool's actual finish time (launch + retrieval + injected delay) when the trace has it.
    def ret_t(e):
        if e.get("retrieval_s") is not None and e.get("injected_delay_s") is not None:
            return min(e["t"], t_launch + e["retrieval_s"] + e["injected_delay_s"])
        return e["t"]
    rets = [ret_t(e) for e in ev if e["event"] == "tool_return" and ret_t(e) <= t_end]
    all_back = len(rets) == len(launches)
    t_last = max(rets) if all_back else t_end  # tools still running at a miss: waiting until the end
    hid_s, post_s = _split(_intervals(ev, "consume", t_end), t_last)
    hid_p, post_p = _split(_intervals(ev, "preload", t_end), t_last)
    return dict(planning=t_launch, tool_wait=t_last - t_launch, post_summarize=post_s, post_preload=post_p,
                final=max(0.0, (t_end - t_last) - post_s - post_p), hidden_summarize=hid_s, hidden_preload=hid_p,
                completion=t_end, valid=valid)


def breakdown(run_dir: str | Path, deadline: float | None = None) -> tuple[dict, Path]:
    run_dir = Path(run_dir)
    rows = list(csv.DictReader((run_dir / "results.csv").open()))
    if deadline is None:
        deadline = max(float(r["deadline_s"]) for r in rows)
    per: dict[str, list[dict]] = defaultdict(list)
    n_all: dict[str, int] = defaultdict(int)
    for r in rows:
        if float(r["deadline_s"]) != deadline:
            continue
        n_all[r["policy"]] += 1
        tp = r["trace_path"] if Path(r["trace_path"]).exists() else run_dir / "traces" / Path(r["trace_path"]).name
        b = run_breakdown(tp, deadline)
        if b:
            per[r["policy"]].append(b)
    keys = PARTS + ["hidden_summarize", "hidden_preload", "completion"]
    mean = {p: {k: sum(b[k] for b in bs) / len(bs) for k in keys}
            | {"n": len(bs), "n_all": n_all[p], "n_miss": sum(not b["valid"] for b in bs)}
            for p, bs in per.items() if bs}
    order = [p for p in COLORS if p in mean] + [p for p in mean if p not in COLORS]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(order) + 1.4), facecolor="white")
    for i, p in enumerate(order):
        left = 0.0
        for part in PARTS:
            w = mean[p][part]
            if w > 1e-6:
                ax.barh(i, w, left=left, color=PART_COLORS[part], edgecolor="white",
                        label=PART_LABELS[part] if i == 0 or part not in ax.get_legend_handles_labels()[1] else None)
            left += w
        hid = mean[p]["hidden_summarize"] + mean[p]["hidden_preload"]
        if hid > 1e-6:  # work overlapped with tool waiting, drawn inside the tool-wait segment
            x0 = mean[p]["planning"]
            ax.barh(i, hid, left=x0, height=0.25, color="#08519c", hatch="///", edgecolor="white", lw=0,
                    label="model work hidden under tool waiting" if "model work hidden under tool waiting"
                    not in ax.get_legend_handles_labels()[1] else None)
        ax.text(deadline + 0.25, i, f"{left:.1f} s  ({mean[p]['n_miss']}/{mean[p]['n']} missed)", va="center", fontsize=8)
    ax.set_yticks(range(len(order)), [LABELS.get(p, p) for p in order])
    ax.invert_yaxis()
    ax.set_xlabel(f"mean time on the critical path at D = {deadline:g} s, all runs (missed runs charged D)")
    ax.axvline(deadline, color="black", lw=0.8, ls=":")
    ax.set_xlim(0, deadline + 4.2)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(run_dir / f"breakdown_D{deadline:g}.{ext}", dpi=200)
    with (run_dir / f"breakdown_D{deadline:g}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "n_runs", "n_missed", *keys])
        for p in order:
            w.writerow([p, mean[p]["n"], mean[p]["n_miss"], *(round(mean[p][k], 4) for k in keys)])
    return mean, run_dir / f"breakdown_D{deadline:g}.png"


# ---------------------------------------------------------------------------------------------------------------
PHASES = ("plan", "consume", "preload", "final")


def run_serving_metrics(trace_path: str | Path, deadline: float | None = None) -> dict | None:
    """Serving-side view of one run, cut at T (completion if on time, else the deadline):
    GPU (model) busy time in total / during tool waits / after the last return, GPU idle time while tools run,
    prefill tokens computed vs reused from the prefix cache, decode tokens, number of model requests."""
    ev = [json.loads(line) for line in open(trace_path)]
    b = run_breakdown(trace_path, deadline)
    if b is None:
        return None
    T = b["completion"]
    t_launch = b["planning"]
    t_last = t_launch + b["tool_wait"]
    busy = before = after = 0.0
    for ph in PHASES:
        for a, z in _intervals(ev, ph, T):
            busy += z - a
            before += max(0.0, min(z, t_last) - max(a, t_launch))
            after += max(0.0, z - max(a, t_last))
    ends = [e for e in ev if e["event"].endswith("_end") and e["event"].split("_")[0] in PHASES and e["t"] <= T + 1e-9]
    prompt = sum(e.get("prompt_tokens") or 0 for e in ends)
    cached = sum(e.get("cached_tokens") or 0 for e in ends)
    tool_wait = b["tool_wait"]
    return dict(valid=b["valid"], end_s=T, gpu_busy_s=busy, gpu_busy_during_tools_s=before,
                gpu_idle_during_tools_s=max(0.0, tool_wait - before),
                gpu_util_during_tools=before / tool_wait if tool_wait > 0 else 0.0,
                model_s_after_last_return=after, prefill_tokens_computed=prompt - cached, prefill_tokens_reused=cached,
                decode_tokens=sum(e.get("completion_tokens") or 0 for e in ends), model_requests=len(ends))


def serving_metrics(run_dir: str | Path, deadline: float | None = None) -> tuple[dict, Path]:
    run_dir = Path(run_dir)
    rows = list(csv.DictReader((run_dir / "results.csv").open()))
    if deadline is None:
        deadline = max(float(r["deadline_s"]) for r in rows)
    per: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if float(r["deadline_s"]) != deadline:
            continue
        tp = r["trace_path"] if Path(r["trace_path"]).exists() else run_dir / "traces" / Path(r["trace_path"]).name
        m = run_serving_metrics(tp, deadline)
        if m:
            per[r["policy"]].append(m)
    keys = [k for k in next(iter(per.values()))[0] if k != "valid"]
    mean = {p: {k: sum(m[k] for m in ms) / len(ms) for k in keys} | {"n": len(ms), "n_miss": sum(not m["valid"] for m in ms)}
            for p, ms in per.items()}
    order = [p for p in COLORS if p in mean] + [p for p in mean if p not in COLORS]
    path = run_dir / f"serving_D{deadline:g}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "n_runs", "n_missed", *keys])
        for p in order:
            w.writerow([p, mean[p]["n"], mean[p]["n_miss"], *(round(mean[p][k], 4) for k in keys)])
    return {p: mean[p] for p in order}, path
