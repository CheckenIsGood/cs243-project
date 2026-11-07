"""Plot results/preload_bench/preload_bench.csv: final-request latency after the last tool return, cold vs
preloaded, vs evidence size; plus the preload work moved off the critical path, and cached-token fraction.
Usage: python scripts/plot_preload_bench.py results/preload_bench
"""

import csv
import statistics as st
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/preload_bench")
rows = list(csv.DictReader((d / "preload_bench.csv").open()))
g = defaultdict(list)
for r in rows:
    g[(int(r["k"]), r["mode"])].append(r)
ks = sorted({k for k, _ in g})


def med(k, m, f):
    return st.median(float(r[f]) for r in g[(k, m)])


tok = [med(k, "cold", "final_prompt_tokens") for k in ks]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3), facecolor="white")
for m, c, lab in (("cold", "#3182bd", "fold (cold prefill after last return)"),
                  ("preloaded", "#6a51a3", "fold + KV preload")):
    ax[0].plot(tok, [med(k, m, "final_latency_s") for k in ks], "o-", color=c, label=lab)
ax[0].plot(tok, [med(k, "preloaded", "preload_total_s") for k in ks], "s--", color="#6a51a3", alpha=0.5,
           label="preload requests (hidden under tool waits)")
ax[0].set_xlabel("final prompt tokens (server)")
ax[0].set_ylabel("seconds (median)")
ax[0].set_title("final-answer latency after the last return", fontsize=9)
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=7)
ax[1].plot(tok, [med(k, "preloaded", "final_cached_tokens") / med(k, "preloaded", "final_prompt_tokens") for k in ks],
           "o-", color="#6a51a3", label="preloaded")
ax[1].plot(tok, [med(k, "cold", "final_cached_tokens") / med(k, "cold", "final_prompt_tokens") for k in ks],
           "o-", color="#3182bd", label="cold")
ax[1].set_ylim(-0.03, 1.03)
ax[1].set_xlabel("final prompt tokens (server)")
ax[1].set_ylabel("cached / prompt tokens (final request)")
ax[1].grid(alpha=0.3)
ax[1].legend(fontsize=7)
if not os.environ.get("REPORT_FIGS"):
    fig.suptitle(f"KV preloading microbenchmark ({len(rows)} requests; Qwen3-8B, vLLM prefix caching, A40)", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(d / f"preload_bench.{ext}", dpi=200)
print("k  tokens  cold_s  preloaded_s  saved_s  preload_total_s  cached_frac")
for k, t in zip(ks, tok):
    c, p = med(k, "cold", "final_latency_s"), med(k, "preloaded", "final_latency_s")
    print(f"{k:2d} {t:7.0f} {c:7.3f} {p:11.3f} {c - p:8.3f} {med(k, 'preloaded', 'preload_total_s'):15.3f} "
          f"{med(k, 'preloaded', 'final_cached_tokens') / med(k, 'preloaded', 'final_prompt_tokens'):11.2f}")
