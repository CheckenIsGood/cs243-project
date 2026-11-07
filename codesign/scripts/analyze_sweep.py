"""Paired analysis of the co-design sweep (results/sweep/kv*/s*/tasks_*_<CFG>.csv).

For each KV cap: mean task latency per config (averaged over seeds), and for each config vs a baseline the paired
difference (baseline - config, seconds saved per task) with a bootstrap CI over (seed, task) pairs. Tasks are
paired by task_id within a seed (same arrivals). Usage: python scripts/analyze_sweep.py results/sweep [--base C0a]
"""

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load(root: Path):
    data = defaultdict(dict)  # (kv, cfg) -> {(seed, task_id): latency}
    for f in root.glob("kv*/s*/tasks_*.csv"):
        kv, seed = f.parts[-3], f.parts[-2]
        cfg = f.stem.rsplit("_", 1)[-1]
        with f.open() as fh:
            for row in csv.DictReader(fh):
                if row["latency"] not in ("", "None"):
                    data[(kv, cfg)][(seed, row["task_id"])] = float(row["latency"])
    return data


def boot_ci(xs, n=4000, seed=0):
    rng = random.Random(seed)
    m = len(xs)
    means = sorted(sum(xs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--base", default="C0a")
    a = ap.parse_args()
    data = load(Path(a.root))
    kvs = sorted({k for k, _ in data}, key=lambda s: (s == "kvfull", int(s[2:]) if s[2:].isdigit() else 0))
    for kv in kvs:
        cfgs = sorted(c for k, c in data if k == kv)
        base = data.get((kv, a.base), {})
        print(f"\n== {kv}  (paired vs {a.base}; + = faster than {a.base})")
        print(f"{'cfg':6} {'seeds':>5} {'n':>4} {'mean':>7} {'p50':>7} {'p90':>7} {'saved':>7}  95% CI")
        for c in cfgs:
            d = data[(kv, c)]
            lat = sorted(d.values())
            seeds = len({s for s, _ in d})
            pairs = [base[k] - d[k] for k in d if k in base]
            if pairs and c != a.base:
                lo, hi = boot_ci(pairs)
                sig = "*" if lo > 0 or hi < 0 else " "
                diff = f"{statistics.mean(pairs):+7.2f}  [{lo:+.2f}, {hi:+.2f}] {sig}"
            else:
                diff = "      -"
            print(f"{c:6} {seeds:5d} {len(lat):4d} {statistics.mean(lat):7.2f} {lat[len(lat) // 2]:7.2f} "
                  f"{lat[int(0.9 * (len(lat) - 1))]:7.2f} {diff}")


if __name__ == "__main__":
    main()
