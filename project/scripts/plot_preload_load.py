"""Preload deadline experiment (configs/preload_load.yaml): static policies never read the clock, so one run with a
non-binding deadline gives each run's completion time T, and miss(D) = P(T > D) exactly for every deadline D.
Plots miss rate vs deadline per policy (95% bootstrap band over questions) and reports the paired per-question
completion-time difference fold - fold_preload.
Usage: python scripts/plot_preload_load.py results/real/preload-load
"""

import csv
import json
import random
import statistics as st
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from deadline_agent.breakdown import run_serving_metrics  # noqa: E402
from deadline_agent.plot import COLORS, LABELS  # noqa: E402

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/real/preload-load")
rows = list(csv.DictReader((d / "results.csv").open()))
T = defaultdict(dict)  # policy -> task -> completion (inf if invalid)
F1 = defaultdict(dict)
for r in rows:
    T[r["policy"]][r["task_id"]] = float(r["completion_s"]) if r["valid"] == "True" else float("inf")
    F1[r["policy"]][r["task_id"]] = float(r["f1"])
pols = [p for p in COLORS if p in T]
qs = sorted(set.intersection(*(set(T[p]) for p in pols)))
rng = random.Random(0)
boots = [[rng.choice(qs) for _ in qs] for _ in range(2000)]
lo_d = min(min(v for v in T[p].values() if v < 1e9) for p in pols) - 1
hi_d = max(max(v for v in T[p].values() if v < 1e9) for p in pols) + 1
grid = [lo_d + i * 0.1 for i in range(int((hi_d - lo_d) / 0.1) + 1)]

fig, ax = plt.subplots(1, 2, figsize=(9, 3.3), facecolor="white")
for p in pols:
    miss = [sum(T[p][q] > D for q in qs) / len(qs) for D in grid]
    bands = [sorted(sum(T[p][q] > D for q in b) / len(b) for b in boots) for D in grid]
    ax[0].plot(grid, miss, color=COLORS[p], label=LABELS.get(p, p), lw=1.6)
    ax[0].fill_between(grid, [b[50] for b in bands], [b[1949] for b in bands], color=COLORS[p], alpha=0.15, lw=0)
ax[0].set_xlabel("completion deadline D (s)")
ax[0].set_ylabel("deadline miss rate  P(T > D)")
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=7)

# paired difference fold - fold_preload, and time after the last return
diff = [T["fold"][q] - T["fold_preload"][q] for q in qs]
after = defaultdict(list)
srv = defaultdict(list)
for p in pols:
    for q in qs:
        tp = next(r["trace_path"] for r in rows if r["policy"] == p and r["task_id"] == q)
        tp = tp if Path(tp).exists() else d / "traces" / Path(tp).name
        m = run_serving_metrics(tp)
        srv[p].append(m)
        after[p].append(m["model_s_after_last_return"])
order = sorted(range(len(qs)), key=lambda i: diff[i])
ax[1].bar(range(len(qs)), [diff[i] for i in order], color=COLORS["fold_preload"], alpha=0.8)
ax[1].axhline(st.mean(diff), color="black", lw=0.8, ls="--", label=f"mean {st.mean(diff):.2f} s")
ax[1].set_xticks([])
ax[1].set_xlabel("held-out question (sorted)")
ax[1].set_ylabel("completion time saved by preloading (s)")
ax[1].set_title("paired: fold minus fold + KV preload, same question and delays", fontsize=8)
ax[1].grid(alpha=0.3, axis="y")
ax[1].legend(fontsize=7)
if not os.environ.get("REPORT_FIGS"):
    fig.suptitle(f"KV preloading under deadlines: 8 searches, ~1.4k tokens/result ({len(qs)} held-out questions)",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(d / f"preload_load.{ext}", dpi=200)

bd = sorted(st.mean(diff[qs.index(q)] for q in b) for b in boots)
summary = {
    "n_questions": len(qs),
    "mean_completion_s": {p: st.mean(T[p][q] for q in qs) for p in pols},
    "median_completion_s": {p: st.median(T[p][q] for q in qs) for p in pols},
    "paired_fold_minus_preload_s": {"mean": st.mean(diff), "ci95": [bd[50], bd[1949]], "min": min(diff),
                                    "max": max(diff), "n_preload_faster": sum(x > 0 for x in diff)},
    "model_s_after_last_return_median": {p: st.median(after[p]) for p in pols},
    "mean_f1": {p: st.mean(F1[p][q] for q in qs) for p in pols},
    "n_identical_answers_fold_vs_preload": sum(
        next(r["answer"] for r in rows if r["policy"] == "fold" and r["task_id"] == q)
        == next(r["answer"] for r in rows if r["policy"] == "fold_preload" and r["task_id"] == q) for q in qs),
    "serving_mean": {p: {k: st.mean(m[k] for m in srv[p]) for k in srv[p][0] if k != "valid"} for p in pols},
    "miss_at": {f"{D:g}": {p: sum(T[p][q] > D for q in qs) / len(qs) for p in pols}
                for D in (12, 14, 16, 18, 20, 22, 25, 30)},
}
(d / "preload_load_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
