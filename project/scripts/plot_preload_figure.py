"""Figure: deadline miss rate with vs without KV preloading (single task on vLLM, measured 2026-11-06).

The two policies compared never read the clock, so one run per question with a non-binding deadline gives its
completion time T, and miss(D) = P(T > D) exactly for every deadline D on the same runs.
- Left: miss rate vs deadline, 95% bootstrap band over questions.
- Right: paired per-question completion-time saving of preloading (same question, same tool delays).
Usage: python scripts/plot_preload_figure.py results/real/preload-load report/figs/preload_load.pdf
"""

import csv
import random
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POLICIES = {"fold": ("wait for all results, then answer", "#2b7bba"),
            "fold_preload": ("+ KV preload during tool waits", "#6a51a3")}


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/real/preload-load")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "report/figs/preload_load.pdf")
    T: dict[str, dict[str, float]] = {p: {} for p in POLICIES}
    for r in csv.DictReader((d / "results.csv").open()):
        if r["policy"] in T:
            T[r["policy"]][r["task_id"]] = float(r["completion_s"]) if r["valid"] == "True" else float("inf")
    qs = sorted(set.intersection(*(set(v) for v in T.values())))
    fin = [t for p in T for t in T[p].values() if t != float("inf")]
    grid = [min(fin) - 1 + i * 0.05 for i in range(int((max(fin) - min(fin) + 3) / 0.05))]
    rng = random.Random(0)
    boots = [[rng.choice(qs) for _ in qs] for _ in range(2000)]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.2, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
    for p, (label, col) in POLICIES.items():
        miss = [sum(T[p][q] > D for q in qs) / len(qs) for D in grid]
        lo, hi = [], []
        for D in grid:
            vals = sorted(sum(T[p][q] > D for q in b) / len(b) for b in boots)
            lo.append(vals[int(0.025 * len(vals))])
            hi.append(vals[int(0.975 * len(vals)) - 1])
        ax.fill_between(grid, lo, hi, color=col, alpha=0.15, lw=0, step="post")
        ax.step(grid, miss, where="post", color=col, lw=1.8, label=label)
    ax.set_xlabel("completion deadline D (s)")
    ax.set_ylabel("deadline miss rate  P(T > D)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    sav = sorted(T["fold"][q] - T["fold_preload"][q] for q in qs)
    bx.bar(range(len(sav)), sav, color=POLICIES["fold_preload"][1], alpha=0.8)
    m = st.mean(sav)
    bx.axhline(m, ls="--", color="k", lw=1, label=f"mean {m:.2f} s")
    bx.set_xticks([])
    bx.set_xlabel("held-out question (sorted)")
    bx.set_ylabel("completion time saved (s)")
    bx.set_title("paired: same question, same tool delays", fontsize=8)
    bx.legend(fontsize=7, loc="upper left")
    bx.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"n={len(qs)} mean saving {m:.3f}s min {sav[0]:.2f} max {sav[-1]:.2f}; wrote {out}")


if __name__ == "__main__":
    main()
