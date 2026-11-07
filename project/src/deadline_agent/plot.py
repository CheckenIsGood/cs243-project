"""Midterm figure: deadline vs mean deadline-qualified F1 (left) and miss rate (right), per policy.

Uncertainty: 95% bootstrap intervals resampling *questions* (all deadlines/policies/seeds of a sampled
question stay together), so repeated runs of one question are never treated as independent.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"incremental": "#9e9e9e", "batch2": "#8c6d31", "batch3": "#bd9e39", "fold": "#3182bd",
          "rss": "#e6550d", "optimal": "#31a354", "fold_preload": "#6a51a3", "rss_preload": "#d6616b"}
LABELS = {"incremental": "static incremental", "batch2": "static batch (b=2)", "batch3": "static batch (b=3)",
          "fold": "static fold", "rss": "RSS heuristic", "optimal": "optimal (MDP)",
          "fold_preload": "fold + KV preload", "rss_preload": "RSS + KV preload"}


# Policies that often coincide (RSS == optimal in 29/30 pilot cells) get a small horizontal offset and distinct
# marker styles so neither hides the other. The offset is cosmetic; x positions are the same deadlines.
STYLE = {"optimal": dict(dx=-0.12, marker="o", mfc=None, ls="-"),
         "rss": dict(dx=+0.12, marker="D", mfc="white", ls="-"),
         "fold_preload": dict(dx=+0.12, marker="s", mfc="white", ls="-")}


def load(run_dir: Path) -> list[dict]:
    return list(csv.DictReader((run_dir / "results.csv").open()))


def aggregate(rows, n_boot: int = 2000, seed: int = 0):
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["task_id"]].append(r)
    qids = sorted(by_q)
    keys = sorted({(float(r["deadline_s"]), r["policy"]) for r in rows})

    def stats(sample_q):
        acc = defaultdict(lambda: [0.0, 0.0, 0])
        for q in sample_q:
            for r in by_q[q]:
                a = acc[(float(r["deadline_s"]), r["policy"])]
                a[0] += float(r["deadline_quality"])
                a[1] += r["valid"] != "True"
                a[2] += 1
        return {k: (v[0] / v[2], v[1] / v[2]) for k, v in acc.items() if v[2]}

    point = stats(qids)
    rng = random.Random(seed)
    boots = [stats([rng.choice(qids) for _ in qids]) for _ in range(n_boot)]
    out = {}
    for k in keys:
        dq = sorted(b[k][0] for b in boots if k in b)
        ms = sorted(b[k][1] for b in boots if k in b)
        lo, hi = int(0.025 * len(dq)), int(0.975 * len(dq)) - 1
        out[k] = dict(dq=point[k][0], dq_lo=dq[lo], dq_hi=dq[hi], miss=point[k][1], miss_lo=ms[lo], miss_hi=ms[hi])
    return out, len(qids)


def model_predictions(run_dir: Path, planning_s: float, step: float = 0.25) -> dict[str, list[tuple[float, float]]]:
    """Predicted miss probability per policy from the run's FROZEN config (calibration costs + delay model).
    Task deadline D maps to remaining-at-launch D - planning_s (planning time from calibration, not this run)."""
    import yaml

    from .experiment import ExperimentConfig
    from .online_mdp import ConsumptionMDP, static_batch, static_fold, static_incremental, required_summary_slack
    from .tools import DelayModel

    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg = ExperimentConfig(**raw)
    costs, pmf = cfg.cost_model(), DelayModel(**cfg.delay).pmf(cfg.dt)
    pols = {"incremental": static_incremental, "batch2": lambda m: static_batch(m, 2),
            "batch3": lambda m: static_batch(m, 3), "fold": static_fold, "rss": required_summary_slack,
            "optimal": None}
    ds = [float(d) for d in cfg.deadlines_s]
    lo, hi = min(ds) - 1.0, max(ds) + 1.0
    out: dict[str, list[tuple[float, float]]] = {p: [] for p in cfg.policies if p in pols}
    D = lo
    while D <= hi + 1e-9:
        mdp = ConsumptionMDP(cfg.n_calls, costs, pmf, cfg.dt, D - planning_s, objective="miss")
        for p in out:
            v = mdp.solve() if pols[p] is None else mdp.evaluate(pols[p](mdp))
            out[p].append((D, min(1.0, max(0.0, v))))
        D += step
    return out


def plot(run_dir: str | Path, title: str | None = None, overlay_planning_s: float | None = None) -> Path:
    run_dir = Path(run_dir)
    agg, nq = aggregate(load(run_dir))
    policies = [p for p in COLORS if any(k[1] == p for k in agg)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), facecolor="white")
    for p in policies:
        ds = sorted(d for d, q in agg if q == p)
        for ax, m in zip(axes, ("dq", "miss")):
            y = [agg[(d, p)][m] for d in ds]
            lo = [agg[(d, p)][m] - agg[(d, p)][m + "_lo"] for d in ds]
            hi = [agg[(d, p)][m + "_hi"] - agg[(d, p)][m] for d in ds]
            st = STYLE.get(p, {})
            ax.errorbar([d + st.get("dx", 0.0) for d in ds], y, yerr=[lo, hi], marker=st.get("marker", "o"), ms=4.5,
                        mfc=st.get("mfc") or COLORS[p], mec=COLORS[p], mew=1.3, capsize=3, lw=1.6,
                        ls=st.get("ls", "-"), color=COLORS[p], label=LABELS.get(p, p))
    if overlay_planning_s is not None:
        for p, pts in model_predictions(run_dir, overlay_planning_s).items():
            if p in COLORS and pts:
                axes[1].plot([d for d, _ in pts], [m for _, m in pts], ls="--", lw=1.0, color=COLORS[p], alpha=0.8)
        axes[1].plot([], [], ls="--", color="black", lw=1.0, label="model prediction (calibration costs)")
    axes[0].set_ylabel("mean deadline-qualified F1")
    axes[1].set_ylabel("deadline miss rate")
    for ax in axes:
        ax.set_xlabel("completion deadline (s)")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.03, 1.03)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=8, frameon=False)
    if title != "":
        fig.suptitle(title or f"{run_dir.name} ({nq} questions; 95% bootstrap CI over questions; "
                     f"points offset ±0.12 s for visibility)", fontsize=10)
    fig.tight_layout(rect=(0, 0.12 if len(labels) <= 4 else 0.18, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(run_dir / f"figure.{ext}", dpi=200)
    with (run_dir / "figure_data.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["deadline_s", "policy", "dq", "dq_lo", "dq_hi", "miss", "miss_lo", "miss_hi"])
        for (d, p), v in sorted(agg.items()):
            w.writerow([d, p, *(round(v[k], 4) for k in ("dq", "dq_lo", "dq_hi", "miss", "miss_lo", "miss_hi"))])
    return run_dir / "figure.png"
