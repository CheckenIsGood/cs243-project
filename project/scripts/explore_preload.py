"""MODEL PREDICTIONS FROM GUESSED COSTS (not measurements): Extension B, KV pre-loading.

Guessed costs of docs/theory.md Section 6; evidence-log prompt layout (docs/theory.md Section 8.2);
preload overhead alpha' = alpha = 0.08 s, 16 template-tail tokens never cached, no eviction.
"+pre" = the same policy, but whenever it would WAIT it pre-loads P0, pending summaries and the returned
raws it will not summarize (preload_mdp.with_idle_preload). "optimal+pre" = exact PreloadMDP optimum
(removal-free policies); "opt+pre,k<=m" = the same optimum restricted to at most m = min_summaries_needed
summaries (m = 0: never summarize). Replace with Gate 1 profiles (incl. measured cached-token savings) before citing.
"""
from deadline_agent.affine import AffineCostModel, prefix_dp
from deadline_agent.cache_model import PreloadCostModel, earliest_first_optimum, solve_preload_dp
from deadline_agent.online_mdp import (ConsumptionMDP, bounded_delay_pmf, required_summary_slack,
                                       static_fold, static_incremental)
from deadline_agent.preload_mdp import PreloadMDP, with_idle_preload
from deadline_agent.speculation import speculative_costs

dt, n = 0.1, 4
regimes = {"variable": bounded_delay_pmf(dt, 0.5, 4.0, slow_p=0.3, slow_lo_s=4.0, slow_hi_s=12.0),
           "predictable": bounded_delay_pmf(dt, 2.0, 3.0)}
contexts = {"ctx_binds(L=1500,cap=4096)": dict(raw_tokens=1500, context_limit=4096),
            "ctx_free(L=768,cap=8192)": dict(raw_tokens=768, context_limit=8192)}
deadlines = {"variable": [12, 14, 16, 18, 20, 22], "predictable": [10.5, 11.0, 16.5, 16.85, 17.0, 17.1]}
heur = {"rss": required_summary_slack, "fold": static_fold, "incremental": static_incremental}
print(__doc__)


def table(m: AffineCostModel, rname: str, pmf, Ds, label: str) -> None:
    print(f"\n== {label}, {rname} delays; need>={m.min_summaries_needed(n)} summaries")
    cols = ["optimal", "optimal+pre", "opt+pre,k<=m"] + [f"{h}{s}" for h in heur for s in ("", "+pre")]
    print("deadline " + "".join(f"{c:>16}" for c in cols))
    for D in Ds:
        bm = ConsumptionMDP(n, m, pmf, dt, D, objective="miss", tie_eps=0.0)
        pm = PreloadMDP(n, m, pmf, dt, D, objective="miss", tie_eps=0.0)
        cap = PreloadMDP(n, m, pmf, dt, D, objective="miss", tie_eps=0.0, max_summaries=m.min_summaries_needed(n))
        vals = [bm.solve(), pm.solve(), cap.solve()]
        for mk in heur.values():
            vals += [bm.evaluate(mk(bm)), pm.evaluate(with_idle_preload(pm, mk(pm)))]
        print(f"{D:>6}s  " + "".join(f"{v:>16.3f}" for v in vals))


for cname, ckw in contexts.items():
    m = AffineCostModel(overhead_s=0.08, prefill_s_per_tok=1 / 3000, decode_s_per_tok=1 / 30, **ckw)
    for rname, pmf in regimes.items():
        Ds = deadlines[rname] if not (rname == "predictable" and "free" in cname) else [9.5, 10.0, 10.25, 10.5]
        table(m, rname, pmf, Ds, cname)

# expected completion time (objective "time"), variable delays
print("\n== expected completion time E[C] (s), variable delays")
for cname, ckw in contexts.items():
    m = AffineCostModel(overhead_s=0.08, prefill_s_per_tok=1 / 3000, decode_s_per_tok=1 / 30, **ckw)
    pmf = regimes["variable"]
    bm = ConsumptionMDP(n, m, pmf, dt, 18.0, objective="time")
    pm = PreloadMDP(n, m, pmf, dt, 18.0, objective="time")
    row = {"optimal": bm.solve(), "optimal+pre": pm.solve()}
    for h, mk in heur.items():
        row[h], row[h + "+pre"] = bm.evaluate(mk(bm)), pm.evaluate(with_idle_preload(pm, mk(pm)))
    print(f"{cname}: " + ", ".join(f"{k}={v:.2f}" for k, v in row.items()))

# A + B together: 3x faster summary decode with pre-loading
m3 = speculative_costs(AffineCostModel(overhead_s=0.08, prefill_s_per_tok=1 / 3000, decode_s_per_tok=1 / 30,
                                       raw_tokens=1500, context_limit=4096), 3.0)
table(m3, "variable", regimes["variable"], [12, 14, 16, 18], "ctx_binds + summary decode 3x (A+B)")

# offline (deterministic) illustrations
print("\n== offline deterministic examples (ctx_binds costs): completion (s)")
cm = PreloadCostModel(AffineCostModel(overhead_s=0.08, prefill_s_per_tok=1 / 3000, decode_s_per_tok=1 / 30,
                                      raw_tokens=1500, context_limit=4096))
for rel in [(0.0, 0.7, 0.7, 0.7), (0.0, 1.0, 2.0, 9.0), (0.0, 3.0, 6.0, 12.0), (2.0, 2.0, 2.0, 2.0)]:
    opt = solve_preload_dp(rel, cm)
    print(f"r={rel}: no-preload {prefix_dp(rel, cm.base).completion:.3f}, preload opt {opt.completion:.3f} "
          f"(summarizes {opt.summarized}), earliest-first+preload {earliest_first_optimum(rel, cm).completion:.3f}")
    print("    plan: " + "; ".join(f"{k}{ids}@{s:.2f}-{e:.2f}" for k, ids, s, e in opt.actions))
