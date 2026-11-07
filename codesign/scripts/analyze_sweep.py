"""Paired analysis of the co-design sweep (results/sweep/kv*/s*/tasks_*_<CFG>.csv).

For each KV cap: mean task latency per config (averaged over seeds), and for each config vs a baseline the paired
difference (baseline - config, seconds saved per task). Tasks are paired by task_id within a seed (same arrivals).
The CI is a two-level bootstrap with the seed as the unit: resample seeds, then tasks within each resampled seed, so
run-to-run noise (about 2.7 s on the mean for a repeated seed) is in the interval once there are several seeds.
With fewer than 3 seeds the CI only reflects task-to-task variation and is marked "~".
--unit task gives the old flat bootstrap over (seed, task) pairs.
Usage: python scripts/analyze_sweep.py results/sweep [--base C0a] [--unit seed|task]
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


def boot_ci_seeds(by_seed, n=4000, seed=0):
    """by_seed: list of per-seed lists of paired differences. Resample seeds, then tasks within each seed."""
    rng = random.Random(seed)
    k = len(by_seed)
    means = []
    for _ in range(n):
        tot = cnt = 0
        for _ in range(k):
            xs = by_seed[rng.randrange(k)]
            tot += sum(xs[rng.randrange(len(xs))] for _ in range(len(xs)))
            cnt += len(xs)
        means.append(tot / cnt)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--base", default="C0a")
    ap.add_argument("--unit", choices=("seed", "task"), default="seed")
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
            by_seed = defaultdict(list)
            for k in d:
                if k in base:
                    by_seed[k[0]].append(base[k] - d[k])
            if pairs and c != a.base:
                lo, hi = boot_ci(pairs) if a.unit == "task" else boot_ci_seeds(list(by_seed.values()))
                sig = "*" if lo > 0 or hi < 0 else " "
                if a.unit == "seed" and len(by_seed) < 3:
                    sig += "~"
                diff = f"{statistics.mean(pairs):+7.2f}  [{lo:+.2f}, {hi:+.2f}] {sig}"
            else:
                diff = "      -"
            print(f"{c:6} {seeds:5d} {len(lat):4d} {statistics.mean(lat):7.2f} {lat[len(lat) // 2]:7.2f} "
                  f"{lat[int(0.9 * (len(lat) - 1))]:7.2f} {diff}")


if __name__ == "__main__":
    main()
