"""MODEL PREDICTIONS FROM GUESSED COSTS (not measurements): Extension A, speculative decoding.

Same guessed costs as scripts/explore_regimes.py (docs/theory.md Section 6). Prompt-lookup speculation is
modeled as a decode speedup factor: summaries (which copy from their prompt) get 2x or 3x, the final answer
1x (or 1.5x in the last setting). Prefill is unchanged. Replace with Gate 1 profiles before citing.
"""
from deadline_agent.affine import AffineCostModel
from deadline_agent.online_mdp import (ConsumptionMDP, bounded_delay_pmf, required_summary_slack,
                                       static_batch, static_fold, static_incremental)
from deadline_agent.speculation import (StochasticDurationMDP, final_summary_count_dist, mean_speedup_costs,
                                        speculative_costs)

dt, n = 0.1, 4
pmf = bounded_delay_pmf(dt, 0.5, 4.0, slow_p=0.3, slow_lo_s=4.0, slow_hi_s=12.0)
contexts = {"ctx_binds(L=1500,cap=4096)": dict(raw_tokens=1500, context_limit=4096),
            "ctx_free(L=768,cap=8192)": dict(raw_tokens=768, context_limit=8192)}
speedups = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (3.0, 1.5)]  # (summary decode x, final decode x)
pols = {"optimal": None, "rss_heur": required_summary_slack, "fold": static_fold,
        "batch2": lambda x: static_batch(x, 2), "incremental": static_incremental}
deadlines = [10, 12, 14, 16, 18, 20]
print(__doc__)
for cname, ckw in contexts.items():
    base = AffineCostModel(overhead_s=0.08, prefill_s_per_tok=1 / 3000, decode_s_per_tok=1 / 30, **ckw)
    need = base.min_summaries_needed(n)
    for sx, fx in speedups:
        m = speculative_costs(base, sx, fx)
        print(f"\n== {cname}, variable delays, summary decode {sx:g}x, final decode {fx:g}x; "
              f"c(0,1)={m.consume(0, 1):.2f}s, f(m)={m.final(need, n):.2f}s, need>={need}")
        print("deadline " + "".join(f"{p:>12}" for p in pols) + "   P(k>m|opt)  E[k|opt]")
        for D in deadlines:
            mdp = ConsumptionMDP(n, m, pmf, dt, D, objective="miss", tie_eps=0.0)
            vals = [mdp.solve() if p is None else mdp.evaluate(p(mdp)) for p in pols.values()]
            # structure of the optimal table, with the Section 5 tie-break toward earlier completion
            tb = ConsumptionMDP(n, m, pmf, dt, D, objective="miss", tie_eps=1e-3)
            tb.solve()
            dist = final_summary_count_dist(tb, tb.optimal_policy())
            extra = sum(p for k, p in dist.items() if k > need)
            ek = sum(k * p for k, p in dist.items())
            print(f"{D:>6}s  " + "".join(f"{v:>12.3f}" for v in vals) + f"   {extra:>10.3f}  {ek:>8.2f}")

# random speedups: exact stochastic-duration MDP vs the deterministic-mean approximation
print("\n== Random summary speedup {1x:0.25, 2x:0.35, 3.5x:0.40} (mean-equivalent 1.85x), final 1x;"
      " ctx_binds, variable delays")
sp = [(1.0, 0.25), (2.0, 0.35), (3.5, 0.40)]
base = AffineCostModel(overhead_s=0.08, prefill_s_per_tok=1 / 3000, decode_s_per_tok=1 / 30,
                       raw_tokens=1500, context_limit=4096)
print("deadline  exact-opt  mean-model-pred  mean-policy-true  rss-true  fold-true")
for D in [12, 14, 16, 18]:
    ex = StochasticDurationMDP(n, base, pmf, dt, D, objective="miss", tie_eps=0.0, consume_speedups=sp)
    mean = ConsumptionMDP(n, mean_speedup_costs(base, sp, [(1.0, 1.0)]), pmf, dt, D, objective="miss", tie_eps=0.0)
    v_ex, v_mean = ex.solve(), mean.solve()
    print(f"{D:>6}s  {v_ex:>9.3f}  {v_mean:>15.3f}  {ex.evaluate(mean.optimal_policy()):>16.3f}"
          f"  {ex.evaluate(required_summary_slack(ex)):>8.3f}  {ex.evaluate(static_fold(ex)):>9.3f}")
