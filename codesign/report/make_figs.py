"""Figures and numbers for the midterm report, from results/sweep (run from codesign/).

  ../project/.venv/bin/python report/make_figs.py

Configs: C0a = stock vLLM (FCFS), C1 = Continuum-like, C2 = Autellix-like (PLAS), C0b = ours (task index Z').
All numbers pool every seed run at a memory size; tasks are paired by (seed, task_id).
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dominance import load  # noqa: E402

OUT = ROOT / "report" / "figs"
VLLM, CONT, AUT, OURS = "C0a", "C1", "C2", "C0b"
COL = {VLLM: "#6b7280", CONT: "#eb6834", AUT: "#1baf7a", OURS: "#2a78d6"}
NAME = {VLLM: "vLLM (FCFS)", CONT: "Continuum-like", AUT: "Autellix-like", OURS: "Ours"}
CAPS = [("kv1000", "16k"), ("kv1500", "24k"), ("kv1875", "30k"), ("kvfull", "Full\n(167k)")]

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

lat, _ = load(ROOT / "results" / "sweep")


def seeds(kv):
    return sorted({s for k, s, _ in lat if k == kv}, key=lambda s: int(s[1:]))


def paired(kv, a, b):
    """Per-seed arrays of paired latencies for configs a and b."""
    out = []
    for s in seeds(kv):
        da, db = lat[(kv, s, a)], lat[(kv, s, b)]
        t = sorted(set(da) & set(db))
        out.append((np.array([da[x] for x in t]), np.array([db[x] for x in t])))
    return out


def pooled(kv, c):
    return np.concatenate([np.array(list(lat[(kv, s, c)].values())) for s in seeds(kv)])


def reduction_ci(kv, base, alg, stat, n_boot=4000, rng=np.random.default_rng(0)):
    """% reduction of stat(latency) from base to alg; two-level bootstrap (seeds, then tasks in each seed)."""
    per = paired(kv, base, alg)
    a_all = np.concatenate([p[0] for p in per]); b_all = np.concatenate([p[1] for p in per])
    point = 100 * (1 - stat(b_all) / stat(a_all))
    k, reps = len(per), []
    for _ in range(n_boot):
        A, B = [], []
        for j in rng.integers(0, k, k):
            i = rng.integers(0, len(per[j][0]), len(per[j][0]))
            A.append(per[j][0][i]); B.append(per[j][1][i])
        A, B = np.concatenate(A), np.concatenate(B)
        reps.append(100 * (1 - stat(B) / stat(A)))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, lo, hi


def fig_headline():
    kv = "kv1000"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.35), gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) CDF of task latency
    for c, ls in [(VLLM, (0, (4, 2))), (CONT, "-"), (OURS, "-")]:
        x = np.sort(pooled(kv, c))
        ax1.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COL[c], lw=1.8 if c == OURS else 1.3,
                 ls=ls, label=NAME[c])
    mv, mo = np.median(pooled(kv, VLLM)), np.median(pooled(kv, OURS))
    ax1.annotate("", xy=(mo, 0.5), xytext=(mv, 0.5), arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
    ax1.plot([mo, mv], [0.5, 0.5], "o", ms=3, color="black")
    ax1.text(mv + 6, 0.5, f"median {mv:.0f}s → {mo:.0f}s\n(−{100 * (1 - mo / mv):.0f}%)", ha="left", va="center",
             fontsize=7.5, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    ax1.set_xlim(0, 180); ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("Task completion time (s)"); ax1.set_ylabel("Fraction of tasks finished")
    ax1.set_yticks([0, .25, .5, .75, 1]); ax1.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax1.grid(alpha=0.25, lw=0.5)
    ax1.legend(loc="lower right", fontsize=7.5, handlelength=2.2)
    ax1.set_title("(a) All tasks, 7 seeds (653 tasks)", fontsize=8.5, loc="left")
    # (b) per-seed median, vLLM -> ours
    ss = seeds(kv)
    for i, s in enumerate(ss):
        a, b = paired(kv, VLLM, OURS)[i]
        ma, mb = np.median(a), np.median(b)
        ax2.plot([i, i], [mb, ma], color="#c7cbd1", lw=3, solid_capstyle="round", zorder=1)
        ax2.scatter([i], [ma], color=COL[VLLM], s=22, zorder=2, label=NAME[VLLM] if i == 0 else None)
        ax2.scatter([i], [mb], color=COL[OURS], s=22, zorder=3, label=NAME[OURS] if i == 0 else None)
        ax2.text(i, ma + 1.5, f"−{100 * (1 - mb / ma):.0f}%", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(range(len(ss))); ax2.set_xticklabels([s[1:] for s in ss])
    ax2.set_xlabel("Workload seed (different arrivals and tool delays)")
    ax2.set_ylabel("Median task completion time (s)")
    ax2.set_ylim(0, 60); ax2.grid(axis="y", alpha=0.25, lw=0.5)
    ax2.legend(loc="lower right", fontsize=7.5, ncol=2, handletextpad=0.2, columnspacing=0.8)
    ax2.set_title("(b) Every seed improves", fontsize=8.5, loc="left")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "headline.pdf", bbox_inches="tight")


def fig_memory():
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    x = np.arange(len(CAPS)); w = 0.36
    rows = []
    for j, (stat, lab, alpha) in enumerate([(np.median, "Median", 1.0), (np.mean, "Mean", 0.55)]):
        vals = [reduction_ci(kv, VLLM, OURS, stat) for kv, _ in CAPS]
        rows.append(vals)
        p = np.array([v[0] for v in vals]); lo = np.array([v[1] for v in vals]); hi = np.array([v[2] for v in vals])
        ax.bar(x + (j - 0.5) * w, p, w, color=COL[OURS], alpha=alpha, label=f"{lab} latency",
               yerr=[p - lo, hi - p], error_kw=dict(lw=0.7, capsize=2, capthick=0.7, ecolor="#374151"))
        for xi, pi, hii in zip(x, p, hi):
            if abs(pi) >= 1:
                ax.text(xi + (j - 0.5) * w, max(hii, 0) + 1.2, f"{pi:.0f}%", ha="center", va="bottom", fontsize=6.8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([c for _, c in CAPS])
    ax.set_xlabel("GPU memory for KV cache (tokens)  ← more pressure")
    ax.set_ylabel("Reduction vs vLLM")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(-6, 50); ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "memory.pdf", bbox_inches="tight")
    return rows


def numbers():
    kv = "kv1000"
    print("== 16k table (pooled over seeds)", seeds(kv))
    for c in [VLLM, CONT, AUT, OURS]:
        x = pooled(kv, c)
        per = paired(kv, VLLM, c)
        faster = np.mean(np.concatenate([p[1] < p[0] for p in per])) if c != VLLM else float("nan")
        print(f"  {NAME[c]:15} mean {x.mean():6.1f}  median {np.median(x):6.1f}  p25 {np.percentile(x, 25):5.1f}"
              f"  faster-than-vLLM {100 * faster:5.1f}%  n {len(x)}")
    for c in [CONT, AUT]:
        print(f"  ours vs {NAME[c]}: mean {reduction_ci(kv, c, OURS, np.mean)}, median {reduction_ci(kv, c, OURS, np.median)}")
    for s, (a, b) in zip(seeds(kv), paired(kv, VLLM, OURS)):
        print(f"  seed {s}: mean {100 * (1 - b.mean() / a.mean()):5.1f}%  median {100 * (1 - np.median(b) / np.median(a)):5.1f}%")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_headline()
    rows = fig_memory()
    for (kv, lab), med, mean in zip(CAPS, rows[0], rows[1]):
        print(f"{kv}: median {med[0]:.1f}% [{med[1]:.1f}, {med[2]:.1f}]  mean {mean[0]:.1f}% [{mean[1]:.1f}, {mean[2]:.1f}]")
    numbers()
