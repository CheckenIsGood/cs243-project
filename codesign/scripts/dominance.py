"""Stochastic-dominance analysis of the co-design sweep (results/sweep/kv*/s*/tasks_*_<CFG>.csv).

For every KV cap, task class and pair of configs (A, B), it compares the latency distributions by their quantile
difference D(p) = Q_B(p) - Q_A(p) (seconds A saves at percentile p) on p = 0.10, 0.15, ..., 0.90, pooled over the
seeds that are complete for all configs at that cap. Tasks are paired by (seed, task_id): same arrivals.

Uncertainty: two-level bootstrap with the seed as the unit (resample seeds, then tasks within each seed, keeping
the A/B pairing), and a simultaneous (sup-t) band over the p grid, so "the band is >= 0 everywhere" is a 95%
statement about the whole curve, not one percentile at a time.

Verdict for A vs B (A is the row):
  dominates      band >= 0 at every p and > 0 at some p (A first-order dominates B on [p10, p90])
  better_part    band > 0 at some p, and B is never significantly faster at any p
  crossing       A significantly faster at some p and significantly slower at others
  none           no significant difference at any p
  (and the mirror images when B is the faster one)

Usage: python scripts/dominance.py results/sweep out.json [--boot 1500]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

CONFIGS = ["C0a", "C0b", "C1", "C2", "C3", "C5", "C6", "C7"]
P = np.round(np.arange(0.10, 0.9001, 0.05), 2)
CLASSES = ["all", "fanout", "react", "fanout-2", "fanout-4", "fanout-8", "react-2", "react-4", "react-6"]


def task_class(row) -> str:
    if row["shape"] == "fanout":  # width from the final prompt: 2 -> < 2.7k tokens, 4 -> < 5.5k, 8 -> above
        t = int(row["final_prompt_tokens"])
        return f"fanout-{2 if t < 2700 else 4 if t < 5500 else 8}"
    return f"react-{int(row['model_calls']) - 1}"


def load(root: Path):
    lat = defaultdict(dict)  # (kv, seed, cfg) -> {task_id: latency}
    cls = {}  # (seed, task_id) -> class
    for f in root.glob("kv*/s*/tasks_*.csv"):
        kv, seed, cfg = f.parts[-3], f.parts[-2], f.stem.rsplit("_", 1)[-1]
        if cfg not in CONFIGS:
            continue
        for row in csv.DictReader(f.open()):
            if row["latency"] in ("", "None"):
                continue
            lat[(kv, seed, cfg)][row["task_id"]] = float(row["latency"])
            cls.setdefault((seed, row["task_id"]), task_class(row))
    return lat, cls


def in_class(c: str, name: str) -> bool:
    return name == "all" or c == name or c.startswith(name + "-")


def pair_stats(a_by_seed, b_by_seed, rng, n_boot):
    """a_by_seed, b_by_seed: lists (one per seed) of paired latency arrays."""
    k = len(a_by_seed)
    a_all, b_all = np.concatenate(a_by_seed), np.concatenate(b_by_seed)
    d_hat = np.quantile(b_all, P) - np.quantile(a_all, P)
    sizes = np.array([len(x) for x in a_by_seed])
    offs = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    nmax = sizes.max()
    idx_parts, mask_parts = [], []
    for _ in range(k):  # k seed draws per bootstrap replicate
        s = rng.integers(0, k, size=n_boot)
        u = rng.random((n_boot, nmax))
        idx_parts.append(offs[s][:, None] + np.floor(u * sizes[s][:, None]).astype(int))
        mask_parts.append(np.arange(nmax)[None, :] < sizes[s][:, None])
    idx, mask = np.concatenate(idx_parts, 1), np.concatenate(mask_parts, 1)
    idx = np.where(mask, idx, 0)
    A = np.where(mask, a_all[idx], np.nan)
    B = np.where(mask, b_all[idx], np.nan)
    d_boot = np.nanquantile(B, P, axis=1).T - np.nanquantile(A, P, axis=1).T  # (n_boot, len(P))
    sd = np.maximum(d_boot.std(0), 1e-6)
    tmax = (np.abs(d_boot - d_hat) / sd).max(1)
    c95, c99 = np.quantile(tmax, 0.95), np.quantile(tmax, 0.99)
    lo, hi = d_hat - c95 * sd, d_hat + c95 * sd
    lo99, hi99 = d_hat - c99 * sd, d_hat + c99 * sd
    diff = b_all - a_all
    m_boot = np.nanmean(B - A, 1)
    f_boot = np.nanmean(np.where(mask, (A < B).astype(float), np.nan), 1)
    return dict(
        d=d_hat, lo=lo, hi=hi, lo99=lo99, hi99=hi99,
        mean=float(diff.mean()), mean_ci=[float(np.quantile(m_boot, 0.025)), float(np.quantile(m_boot, 0.975))],
        faster=float((a_all < b_all).mean()), faster_ci=[float(np.quantile(f_boot, 0.025)),
                                                         float(np.quantile(f_boot, 0.975))],
        n=int(len(a_all)), seeds=k)


def verdict(lo, hi) -> str:
    if (lo >= 0).all() and (lo > 0).any():
        return "dominates"
    if (hi <= 0).all() and (hi < 0).any():
        return "dominated"
    up, down = (lo > 0).any(), (hi < 0).any()
    if up and down:
        return "crossing"
    if up:
        return "better_part"
    if down:
        return "worse_part"
    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--boot", type=int, default=1500)
    a = ap.parse_args()
    lat, cls = load(Path(a.root))
    rng = np.random.default_rng(0)
    kvs = sorted({k for k, _, _ in lat}, key=lambda s: (s == "kvfull", int(s[2:]) if s[2:].isdigit() else 0))
    out = dict(p=P.tolist(), configs=CONFIGS, classes=CLASSES, caps={})
    for kv in kvs:
        seeds = sorted({s for k, s, _ in lat if k == kv}, key=lambda s: int(s[1:]))
        seeds = [s for s in seeds if all((kv, s, c) in lat for c in CONFIGS)]
        if not seeds:
            continue
        # tasks present in every config for that seed
        tasks = {s: sorted(set.intersection(*(set(lat[(kv, s, c)]) for c in CONFIGS))) for s in seeds}
        raw = dict(seeds=seeds, tasks=[], lat={c: [] for c in CONFIGS})
        for s in seeds:
            for t in tasks[s]:
                raw["tasks"].append([s, cls[(s, t)]])
                for c in CONFIGS:
                    raw["lat"][c].append(round(lat[(kv, s, c)][t], 2))
        pairs = {}
        for name in CLASSES:
            sel = {s: [t for t in tasks[s] if in_class(cls[(s, t)], name)] for s in seeds}
            use = [s for s in seeds if sel[s]]
            if sum(len(sel[s]) for s in use) < 8:
                continue
            for A, B in combinations(CONFIGS, 2):
                st = pair_stats([np.array([lat[(kv, s, A)][t] for t in sel[s]]) for s in use],
                                [np.array([lat[(kv, s, B)][t] for t in sel[s]]) for s in use], rng, a.boot)
                rec = {k: (np.round(v, 2).tolist() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
                rec["verdict"] = verdict(st["lo"], st["hi"])
                rec["verdict99"] = verdict(st["lo99"], st["hi99"])
                pairs[f"{name}|{A}|{B}"] = rec
        out["caps"][kv] = dict(raw=raw, pairs=pairs)
        print(kv, seeds, len(raw["tasks"]), "tasks", len(pairs), "pairs")
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
