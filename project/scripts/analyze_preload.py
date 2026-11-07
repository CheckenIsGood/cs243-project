"""Paired per-task comparison of two preload configurations on the same arrival trace (same seed => same tasks,
arrivals and tool delays), from scripts/real_multiagent.py task CSVs.

Usage: python scripts/analyze_preload.py BASE_DIR:MODE TREAT_DIR:MODE [--rates 0.15 0.3 0.45 0.6]
  e.g. python scripts/analyze_preload.py results/real/preload_seeds/seed2:off results/real/preload_seeds/seed2_chain:chain
Columns: mean latency base/treat, paired saving with 95% bootstrap CI, fraction of tasks faster, p90 base/treat, and
(when the CSVs have the instrumentation fields) model time after the last tool return, final-call TTFT, fraction of
the final prompt served from the prefix cache, preloads per task; answer agreement and mean F1 difference.
"""

import argparse
import csv
import random
import statistics as st
from pathlib import Path


def load(d: str, mode: str, rate: float) -> dict[str, dict]:
    p = Path(d) / f"tasks_r{rate:g}_fcfs_pre-{mode}.csv"
    if not p.exists():  # single-mode runs are written without the _pre- suffix
        p = Path(d) / f"tasks_r{rate:g}_fcfs.csv"
    return {r["task_id"]: r for r in csv.DictReader(p.open())}


def ci(xs, n=4000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choice(xs) for _ in xs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def p90(xs):
    return sorted(xs)[int(0.9 * (len(xs) - 1))]


def instr(R, ids):
    if not ids or R[ids[0]].get("t_last_tool") in (None, ""):
        return None
    f = float
    return (st.mean(f(R[i]["end"]) - f(R[i]["t_last_tool"]) for i in ids),
            st.mean(f(R[i]["final_first"]) - f(R[i]["final_submit"]) for i in ids),
            st.mean(f(R[i]["final_cached_tokens"] or 0) / f(R[i]["final_prompt_tokens"]) for i in ids),
            st.mean(int(R[i]["n_preloads"]) for i in ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("treat")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.15, 0.3, 0.45, 0.6])
    a = ap.parse_args()
    (bd, bm), (td, tm) = a.base.rsplit(":", 1), a.treat.rsplit(":", 1)
    print(f"base={bd}:{bm}  treat={td}:{tm}")
    print("rate   n  base  treat  saving [95% CI]   faster  p90 b/t      | after-last-tool b/t  ttft b/t  "
          "cache-hit b/t  preloads b/t | same-ans  dF1")
    for rate in a.rates:
        try:
            B, T = load(bd, bm, rate), load(td, tm, rate)
        except FileNotFoundError as e:
            print(f"{rate:<5} missing: {e.filename}")
            continue
        ids = sorted(i for i in set(B) & set(T) if B[i]["latency"] and T[i]["latency"])
        lb = [float(B[i]["latency"]) for i in ids]
        lt = [float(T[i]["latency"]) for i in ids]
        d = [x - y for x, y in zip(lb, lt)]
        lo, hi = ci(d)
        line = (f"{rate:<5} {len(ids):3d} {st.mean(lb):5.2f} {st.mean(lt):6.2f} {st.mean(d):6.2f} [{lo:5.2f},{hi:5.2f}]"
                f"  {sum(x > 0 for x in d) / len(d):5.2f}  {p90(lb):5.1f}/{p90(lt):5.1f} |")
        ib, it = instr(B, ids), instr(T, ids)
        if ib and it:
            line += (f" {ib[0]:5.2f}/{it[0]:5.2f}        {ib[1]:4.2f}/{it[1]:4.2f}  {ib[2]:5.2f}/{it[2]:5.2f}"
                     f"    {ib[3]:3.1f}/{it[3]:3.1f}    |")
        else:
            line += " (no instrumentation)                                         |"
        same = sum(B[i]["answer"] == T[i]["answer"] for i in ids) / len(ids)
        df1 = st.mean(float(T[i]["f1"]) - float(B[i]["f1"]) for i in ids)
        print(line + f"  {same:5.2f}  {df1:+.3f}")


if __name__ == "__main__":
    main()
