"""Model-based (NOT measured) exploration: predicted miss probability by policy.
Costs are rough guesses for an 8B model on a 24GB-class GPU; replace with Gate 1 measurements."""
from deadline_agent.affine import AffineCostModel
from deadline_agent.online_mdp import (ConsumptionMDP, bounded_delay_pmf, required_summary_slack,
                                       static_batch, static_fold, static_incremental)

dt = 0.1
regimes = {
    "variable": bounded_delay_pmf(dt, 0.5, 4.0, slow_p=0.3, slow_lo_s=4.0, slow_hi_s=12.0),
    "predictable": bounded_delay_pmf(dt, 2.0, 3.0),
}
contexts = {"ctx_binds(L=1500,cap=4096)": dict(raw_tokens=1500, context_limit=4096),
            "ctx_binds(L=1500,cap=5000)": dict(raw_tokens=1500, context_limit=5000),
            "ctx_free(L=768,cap=8192)": dict(raw_tokens=768, context_limit=8192)}
pols = {"optimal": None, "rss_heur": required_summary_slack, "fold": static_fold,
        "batch2": lambda x: static_batch(x, 2), "incremental": static_incremental}
n = 4
for cname, ckw in contexts.items():
    m = AffineCostModel(overhead_s=0.08, prefill_s_per_tok=1/3000, decode_s_per_tok=1/30, **ckw)
    for rname, pmf in regimes.items():
        print(f"\n== {cname}, {rname} delays; need>={m.min_summaries_needed(n)} summaries")
        print("deadline " + "".join(f"{p:>12}" for p in pols))
        for D in [12, 14, 16, 18, 20, 22, 24]:
            mdp = ConsumptionMDP(n, m, pmf, dt, D, objective="miss", tie_eps=0.0)
            vals = [mdp.solve() if p is None else mdp.evaluate(p(mdp)) for p in pols.values()]
            print(f"{D:>6}s  " + "".join(f"{v:>12.3f}" for v in vals))

# Predictable delays at deadlines between integers (docs/theory.md Section 6 rows): fold's completion lies
# in an interval of width 1 s, so integer deadlines give 0/1 for every policy.
print("\n== predictable delays, fine deadlines")
for cname, ckw, Ds in [("ctx_binds(L=1500,cap=4096)", contexts["ctx_binds(L=1500,cap=4096)"], [16.5, 16.85, 17.0]),
                       ("ctx_free(L=768,cap=8192)", contexts["ctx_free(L=768,cap=8192)"],
                        [9.0 + 0.0625 * i for i in range(29)])]:
    m = AffineCostModel(overhead_s=0.08, prefill_s_per_tok=1/3000, decode_s_per_tok=1/30, **ckw)
    gap = 0.0
    for D in Ds:
        mdp = ConsumptionMDP(n, m, regimes["predictable"], dt, D, objective="miss", tie_eps=0.0)
        vals = [mdp.solve() if p is None else mdp.evaluate(p(mdp)) for p in pols.values()]
        gap = max(gap, vals[2] - vals[0])
        if len(Ds) <= 3:
            print(f"{cname} {D:>6}s  " + "".join(f"{v:>12.3f}" for v in vals))
    print(f"{cname}: max (fold - optimal) miss gap over D in [{Ds[0]}, {Ds[-1]}] = {gap:.3f}")
