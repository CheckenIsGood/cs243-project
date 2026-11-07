"""Paired per-task analysis of the non-blocking preload modes (on / low / jit) vs off, per seed and rate.
Seed 0 'off' comes from results/real/preload_contention (same seed, workload and code path for 'off').
Usage: python scripts/analyze_preload_jit.py > results/real/preload_jit/paired.txt
"""
import csv, random, statistics as st
from pathlib import Path

def load(p):
    return {r["task_id"]: r for r in csv.DictReader(Path(p).open())}

def boot_ci(xs, n=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choice(xs) for _ in xs) for _ in range(n))
    return ms[int(.025 * n)], ms[int(.975 * n) - 1]

SRC = {0: {"off": "results/real/preload_contention", "*": "results/real/preload_jit/seed0"},
       1: {"off": "results/real/preload_jit/seed1", "*": "results/real/preload_jit/seed1"}}
print("seed rate mode  n  mean_off mean_mode saving [95% CI]  frac_faster  p90_off p90_mode  same_answer  dF1")
allrows = []
for seed, src in SRC.items():
    for rate in (0.45, 0.6):
        off = load(f"{src['off']}/tasks_r{rate:g}_fcfs_pre-off.csv")
        for mode in ("on", "low", "jit"):
            m = load(f"{src['*']}/tasks_r{rate:g}_fcfs_pre-{mode}.csv")
            ids = sorted(set(off) & set(m))
            d = [float(off[i]["latency"]) - float(m[i]["latency"]) for i in ids]
            lo, hi = boot_ci(d)
            q = lambda xs: sorted(xs)[int(.9 * (len(xs) - 1))]
            lo_ = [float(off[i]["latency"]) for i in ids]; lm = [float(m[i]["latency"]) for i in ids]
            same = sum(off[i]["answer"] == m[i]["answer"] for i in ids) / len(ids)
            df1 = st.mean(float(m[i]["f1"]) - float(off[i]["f1"]) for i in ids)
            print(f"{seed} {rate:<4} {mode:<4} {len(ids):3d} {st.mean(lo_):7.2f} {st.mean(lm):7.2f} "
                  f"{st.mean(d):6.2f} [{lo:5.2f},{hi:5.2f}] {sum(x > 0 for x in d)/len(d):6.2f} "
                  f"{q(lo_):7.2f} {q(lm):7.2f} {same:8.2f} {df1:+.3f}")
            allrows.append((seed, rate, mode, ids, m))
print("\nPairwise between preload modes (same seed, rate): mean latency diff, 95% CI")
for seed in SRC:
    for rate in (0.45, 0.6):
        R = {mode: m for s, r, mode, ids, m in allrows if s == seed and r == rate}
        for a_, b_ in (("on", "low"), ("on", "jit"), ("low", "jit")):
            ids = sorted(set(R[a_]) & set(R[b_]))
            d = [float(R[a_][i]["latency"]) - float(R[b_][i]["latency"]) for i in ids]
            lo, hi = boot_ci(d)
            print(f"{seed} {rate:<4} {a_}-{b_:<4} n={len(ids)} {st.mean(d):+6.2f} [{lo:+.2f},{hi:+.2f}]")
