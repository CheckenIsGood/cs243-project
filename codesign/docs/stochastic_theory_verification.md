# Independent verification of `docs/stochastic_theory.md` (commit e17bd7f)

Scope: the version of `docs/stochastic_theory.md`, `src/codesign/stochastic_theory.py` and
`tests/test_stochastic_theory.py` at commit e17bd7f. Line numbers refer to `git show e17bd7f:project/docs/stochastic_theory.md`.

How this was checked:
- I audited every proof line by line.
- I wrote my own code in `src/codesign/stochastic_verify.py`, with its own instance types, simulators, exact DPs, bounds and LP. Nothing from the author's module is used to compute OPT, ALG or any bound. The only exception is cross-checks run outside the test file.
- `tests/test_stochastic_verify.py` has 31 tests and all pass (`.venv/bin/python -m pytest -q tests/test_stochastic_verify.py`, about 55 s). The author's 27 tests also still pass.
- The algebra is checked with Z3 5.3.3. sympy and scipy are not installed, so every LP is solved either with Z3 or by enumerating vertices.

## Summary

| Result | Doc lines | Verdict | Evidence |
|---|---|---|---|
| Thm A: Lemma A1 (pathwise) | 160–181 | **VERIFIED** | Manual audit. Own simulator: 3,656 (task, realization) checks with DAGs, release dates, m ∈ {1,2,3} and random sizes, all holding; the largest (C−r)/(bound−r) was 0.94. |
| Thm A: LB_cp, (1/m)Smith(W) ≤ OPT_clair ≤ OPT_na, ratio 2 | 183–203 | **VERIFIED** | Manual audit. Own exact clairvoyant and non-anticipatory DPs (m = 1): 60 instances, all bounds hold, worst ALG/OPT_clair = 1.132. The integer grid is exact (refining it to 1/2 changes nothing). Hill-climbing for 200 s found at most 1.117. Smith rule matches brute force. Megow–Schulz checked. |
| Thm A lower bound 4/3 (lag trap) | 205–217 | **VERIFIED** | Exact: ALG = 4L+1 and OPT_clair = OPT_na = 3L+2 for L = 4, 10, 30. |
| MSU load inequality with lags/tasks | 228–240 | **VERIFIED** | Source checked (Skutella–Uetz (3.1), Thm 3.1, which is MSU Cor 3.1). Re-derived by hand. Checked exactly on 173 random non-preemptive non-anticipatory policies (static priorities, deliberate idling, adaptive rules; lags, release dates, DAGs; m ≤ 3): minimum slack 0, never negative. Shown to fail for preemptive policies (213/16 < 33/2). |
| LB2 (task polymatroid, greedy) | 242–251 | **VERIFIED** | Manual audit (supermodularity and Edmonds greedy with no sign constraints). Primal-feasible greedy point plus a matching chain dual certificate on 60 instances. |
| Thm B (online, vs OPT_np) | 253–272 | **VERIFIED** | Manual audit. Z3 for the 1/(2m) − γ identity and the chain/DAG constants. Own exact OPT_np DP: 35,327 random instances (m ≤ 3, chains, forks, vees, releases), zero violations, worst ALG/OPT_np = 1.286. One instance in the γ < 0 regime. |
| Thm B′ (offline LP) | 274–300 | **VERIFIED** | Manual audit (the LP is a relaxation of OPT_np; the per-task chain of inequalities). Skutella–Uetz Lemma 3.3 checked in the source. Own LP (Z3): 40 instances, LP ≤ OPT_np, and E[C_J] ≤ (2+max)·C_J^LP for every task. |
| Thm D: Lemma D0 (kicks) | 350–361 | **VERIFIED** (two small text fixes) | Z3 for both cases. Own simulator: 117,139 kick events, all at availability events; kicked l⁺ < l⁺(arrivals) + L_max every time (largest ratio 0.98, even with simultaneous arrivals). |
| Thm D: Lemma D1 (pathwise, any keep policy, preloads) | 363–397 | **VERIFIED** (two small text fixes) | Manual audit of the four-way partition. Own simulator: 592,077 checks, with and without memory limits, hold/LRU, preload, DAGs; largest (C−r)/(bound−r) 0.85. Without kicks or without κ the bound fails (own checker). |
| Thm D: (L1) critical-path lower bound | 315–318 | **GAP / false as stated for nodes with fan-out pieces** | Exact counterexample: an engine schedule at speed 1 that prefills an already-returned piece during the remaining tool wait costs 16, but LB_cp = 26 (§3.4). |
| Thm D: (L2), (L3), (L2+L3) | 319–330 | **VERIFIED** | Manual audit. Feng et al. Lemma 1 confirmed in the source. |
| Thm D (hybrid and engine) final constant | 399–427 | **VERIFIED WITH FIX** | The proof is correct if the benchmark may not prefill a node before it is available, or if no node has staggered pieces. Otherwise a corrected constant (2+β) + 2·max(1+β, θ) is proved (§3.5) and checked on 120 instances. Calibrated table recomputed exactly. |
| Thm D assumption wording | 306–309, 700–701 | **VERIFIED WITH FIX** | The index Z′_J is computed at release, so token counts must be known at release, not only "at availability". |
| R0a (speed-3 lift, tight, converse) | 449–456 | **VERIFIED** | Own Z3, including that time-sharing realizes exactly the hybrid rates (λ = (a+cB)/(2a)). The engine simulating the hybrid is legitimate in the fluid model. |
| R0b (discrete, additive) | 458–474 | **VERIFIED (proof is a sketch)** | Own discrete-step simulator: 354,067 checks, none violated. In no case did the discrete engine exceed the hybrid bound alone, consistent with the doc's remark. |
| R0c (machine cap loses m/m_eff) | 476–479 | **VERIFIED WITH FIX** | 1286/287 reproduced, but it compares m = 1 with m = 8 at unbounded memory, so it is not an instance of the m_eff rule as described. A consistent instance is given (§4.3). |
| R1 | 481–483 | **VERIFIED** (inside Thm D) | As for Thm D. |
| R5a (best TTL optimal, ≤ e/(e−1)) | 489–509 | **VERIFIED** | Candidate set checked against a 1/8 grid on 300 laws. Karlin density cost in closed form equals e/(e−1)·min(d,B). Grid LP values equal the discrete ski-rental 1/(1−(1−1/n)^n). Exponential rule checked in Z3. |
| R5b (keep LP, σ\*) | 511–536 | **VERIFIED** (linearized LP) | Primal solved by own vertex enumeration equals the dual on 25 instances. σ\* formula, η = 0 issue rule, BE′ closed form and first-order term checked in Z3. Matches `controller_theory.md` l. 246. |
| R5c (safety; per-task dominance false) | 538–554 | **VERIFIED** | Safety follows from Lemma D1 (which allows any keep policy). Counterexample reproduced with own simulator: 24 → 27, weighted total improves, bounds hold. |
| R6 (preload safety; isolated saving exact) | 556–564 | **VERIFIED WITH FIX** | Safety: VERIFIED. "Saving = α·held hits exactly" is **false for DAG tasks** (saving 1/2 vs α·hits = 1); it holds for chains (60 instances). |
| R7.2 hold-only unbounded | 571–577 | **VERIFIED** | Counting argument checked; Σ⌈i/2⌉/12 = 7/2. |
| R7.5(i) index vs probing → 1+Δ | 582–588 | **VERIFIED** | Exact enumeration; 1+Δ = 1/q. |
| R7.5(ii) EFMM Ω(Δ^{1/4}) | 589–590 | **VERIFIED** (source) | EFMM Thm 1: "Any index policy has an approximation factor Ω(Δ^{1/4}) for P‖E[ΣC_j]". |
| R7.1, R7.4 (existing results in `open_problem.md`) | 570, 580 | not re-checked | Outside this commit. |
| §9 citations | 620–682 | **VERIFIED, one inaccurate** | "later 2 + 2ln2 + ε" for precedence *delays* (l. 662–663) is not supported: Li's 2+2ln2+ε is for P\|prec\|ΣwC, and the paper does not mention delays. |

Nothing claimed as a theorem turned out to be false in its final ratio. Two statements are false as written:
1. (L1) as a lower bound on the stated benchmark, when nodes have fan-out pieces.
2. R6's "exact" saving, for DAG tasks.

The rest are text fixes.

---

## 1. Theorem A (doc §2, l. 149–217)

**Hypotheses as used.**
- m identical speed-1 machines.
- Deterministic node sizes.
- Lags form an arbitrary random vector that does not depend on the policy.
- DAG tasks, release dates, online arrival.
- The policy ranks tasks by w_J/W_J, fixed at release. Node priority is (task rank, topological index). The m highest-priority available nodes run, with preemption.

**Lemma A1 (l. 162–173), step by step.**
- (1) The backward path is well defined. Because a_{v_i} ≥ C_{v_{i−1}}, the node intervals [a_v, C_v] are pairwise disjoint and [r_J, C_J] is covered by lag intervals and node intervals.
- (2) If v is available and not running, then m available nodes of strictly higher priority are running. These are nodes of tasks ranked above J, or of J itself with a smaller index. Correct.
- (3) Disjointness means the all-busy instants carry at most Σ_{K⪯J} W_K units of work at rate m. The instants when v runs total at most p_v.
- (4) Lags plus node sizes along the path are at most CP_J.
- The bound overcounts J's path work twice (once in CP_J and once in W_J). That only weakens it. Correct.
- The order of the ranking is never used, so the lemma also holds for random sizes. Correct.

**Lower bounds (l. 185–195).**
- LB_cp holds pathwise for every policy. Correct.
- Smith bound: a speed-m single machine simulates any m-machine schedule. Every task's work finishes by C_J, and relaxing to one machine (dropping precedence, lags and release dates) gives a cost at least Smith(W/m) = (1/m)Smith(W). This holds per realization and is correct.
- OPT_clair ≤ OPT_na (an expected pathwise minimum is at most the cost of any policy). Correct.
- The ranking used is the Smith order of W, so the weighted sum of (1/m)Σ_{K⪯J}W_K is exactly (1/m)Smith(W), and ties do not change it. Correct.
- The ranking is deterministic because W and the release dates are deterministic. Correct.
- No E[max] versus max E confusion: only linearity of expectation is used.

**Machine evidence.**
- `test_lemmaA1_pathwise_own_simulator`.
- `test_thmA_vs_exact_clairvoyant_and_nonanticipatory_m1`: own clairvoyant DP and own non-anticipatory preemptive DP over information states (attained service, elapsed lag ages, hazards). In 14 of 60 instances OPT_clair < OPT_na strictly.
- `test_unit_grid_is_exact_for_m1_dps`: refining the grid to 1/2 leaves both m = 1 optima unchanged. This supports the author's statement that the m = 1 grid DPs are exact.
- `test_lag_trap_exact`.
- A 200-second hill-climb over chains and forks with 2–3 tasks found ALG/OPT_clair at most 1.117. The author's reported 1.128 and 1.842 (l. 199–200) reproduce exactly from their seeds.

**Source.** Megow–Schulz, ORL 32 (2004) 485–490, Thm 2.1: P-WSPT is at most twice the optimal value for P|r_j, pmtn|ΣwC (checked via the abstract and excerpt; full PDF not opened). Thm A is the same argument with lags added to the path term. **VERIFIED.**

## 2. Theorems B and B′ (doc §3, l. 221–300)

**MSU inequality (l. 228–240).** Re-derived by hand.
- Per machine, Σ p_j S_j ≥ ½[(Σp)² − Σp²]. Cauchy–Schwarz gives Σ p_j S_j ≥ (1/(2m))(Σp)² − ½Σp².
- Take expectations, using that S_x is independent of p_x, and E[(Σp)²] = (ΣEp)² + ΣVar.
- This gives (1/(2m))(ΣEp)² + [½ − (m−1)Δ/(2m)]Σ(Ep)². That equals the doc's f(A), and b is the coefficient of Σ(Ep)² (also Z3: `test_thmB_algebra_z3`).
- Independence of S_x from p_x needs three things: a non-anticipatory policy, independent sizes, and lags independent of sizes. All three are hypotheses of Thm B. The per-machine inequality needs a non-preemptive schedule, which is why the benchmark is OPT_np.

Source: Skutella–Uetz, SICOMP 34(4) 2005. I extracted the text from the PDF.
- (3.1) defines f(W) = (1/(2m))[(ΣE P_j)² + ΣE[P_j]²] − ((m−1)(Δ−1)/(2m))ΣE[P_j]², with Δ an upper bound on Var/E².
- Thm 3.1 ("see [15, Cor. 3.1]") says the load inequalities are valid "for all W ⊆ V and any nonanticipatory scheduling policy".
- Their policies are non-preemptive in the MSU sense.

Machine evidence:
- `test_msu_inequality_on_random_nonanticipatory_nonpreemptive_policies`: exact E[C_x] for random non-anticipatory non-preemptive policies (random priorities, deliberate idling thresholds, and a priority swap that depends on the parity of realized completed sizes), checked over all node subsets.
- `test_msu_fails_for_preemptive_policies`: 213/16 < 33/2.
- `test_msu_correction_needed`: OPT_np = 2358/125 = 18.864 < the uncorrected LP. This matches l. 239–240.

**LB2 (l. 242–251).**
- C_J ≥ C_x for x ∈ J gives Σ_J E W_J E C_J ≥ Σ_x E p_x E C_x. Correct.
- For b < 0, the step Q ≤ Σ(E W_J)² is correct (a sum of squares is at most the square of the sum, for nonnegative terms).
- g is supermodular with g(∅) = 0. The LP is min Σ(w_J/EW_J)z_J subject to z(S) ≥ g(S), with z free. Edmonds' greedy is optimal without monotonicity or sign constraints; this matters because 1/(2m)+γ can be negative.
- Verified by a primal-feasibility check over all 2^n sets plus the chain dual y_{S_k} = c_k − c_{k+1} ≥ 0 with equal objective (`test_lb2_greedy_has_matching_dual_certificate`).

**Theorem B (l. 253–264).**
- Step 1 is linearity: (1/m)Σ_{K⪯J} = (1/m)Σ_{K≺J} + (1/m)EW_J, so the excess over LB2 is (1/(2m) − γ)Σ w EW.
- Chains: W_J ≤ CP_J pathwise.
- DAGs: C_J ≥ r_J + W_J/m pathwise for any schedule.
- The identity 1/(2m) − γ = max(1,(m−1)(Δ−1))/(2m) is Z3-proved (own encoding).
- The ranking uses w/E[W], which is deterministic, so no random priority order is treated as fixed.
- The values on l. 266–268 are correct.

Evidence: `test_thmB_vs_exact_nonpreemptive_optimum` (150 instances in the test, 35,327 in the exploratory sweep) and `test_thmB_high_variance_gamma_negative`. The author's 1.068 and 1.098 (l. 272) reproduce. **VERIFIED.**

**Theorem B′ (l. 274–300).**
- The LP is a relaxation of OPT_np. Every added constraint holds pathwise or by linearity:
  - C_v ≥ C_u + D_uv + p_v pathwise;
  - C_J ≥ C_x;
  - C_J ≥ r_J + CP_J.
- Proof steps 1–3 are correct: E p_x ≤ C_x^LP ≤ C_J^LP for x ∈ A_J, because K ⪯ J in LP order.
- Step 4: for b < 0, C(1−b) ≥ μ/(2m), so μ/m ≤ (1 + (m−1)Δ/m)C. For b ≥ 0, μ/m ≤ 2C. Z3-proved.
- This is exactly Skutella–Uetz Lemma 3.3 (= MSU Lemma 4.2): "(1/m)Σ_{k≤j}E[P_k] ≤ (1+max{1,(m−1)Δ/m})C_j^LP" for points satisfying the load inequalities and C_j ≥ E[P_j]. I confirmed this in the extracted text.
- The remark "B′ beats B for DAGs when m ≥ 3 and Δ is large" is correct; precisely, it needs Δ > m/(m−2).
- Evidence: `test_thmBprime_lp_policy` (own Z3 LP, per-task inequality). **VERIFIED.**

## 3. Theorem D (doc §4, l. 304–443)

### 3.1 Lemma D0 (l. 350–361): VERIFIED, with two text fixes
- The proof says "the new prefix (which contains y)". If y itself does not fit, the head-of-line stop at y kicks everything below y without admitting y. The bound still holds: kicked = S_old − S_above < ℓ⁺_y (Z3, second case in `test_thmD_constant_algebra_z3`).
- With k simultaneous arrivals (a release with several sources, or several lags expiring at once), the bound is Σℓ⁺_{y_i} + L_max. Charging κ per node covers this.
- My simulator logs every kick event. In 117,139 events, all occur at availability instants and kicked l⁺ < l⁺(arrivals) + L_max.

### 3.2 Lemma D1 (l. 363–397): VERIFIED, with two text fixes

I audited each case.

- (C): server time spent on higher-priority prefill, including recompute.
  - The recompute of a kicked node z ≻ v is caused by an arrival y′ ≻ z, so y′ belongs to a task ⪯ J.
  - Lemma D0 bounds the total recompute per arrival by α(ℓ⁺_{y′} + L_max).
  - v's own post-kick recompute and the (C) instants share the same server pool, and they are disjoint in time.
- (A): (a/m)ΣX = cΣX. Kicked decoders resume, so each node decodes X tokens in total. Correct.
- (B): if v is not admitted, the prefix stops at some u ⪰ v, so the admitted nodes, all above v, reserve more than M′ − ℓ⁺_u ≥ M.
  - **Text fix:** the doc writes M′ − ℓ⁺_v (l. 375). It should be ℓ⁺_u. The conclusion is unchanged.
  - **Text fix:** "more than m are decoding" should be "at least m".
  - The remaining sub-case (all admitted nodes decoding, total reservation > M) is bounded by the area Σ aXℓ⁺/M. Correct.
- Held KV and preloads never take memory from admission and never preempt an admitted prefill (§4.2). Correct.

Evidence: `test_lemmaD0_D1_own_hybrid_simulator`. My simulator implements §4.2 from the text, and on evict-always runs its completion times agree exactly with the author's `sim_hybrid` (0 mismatches). `test_lemmaD1_mutations_detected_by_own_checker` shows that removing kicks or κ breaks the bound.

### 3.3 Lower bounds (L2), (L3), (L2+L3): VERIFIED
- (L2): in the fluid speed-1 engine, α·(prefill rate) + c·(decode rate) ≤ u + (1−u)·cB/(a+cB) ≤ 1.
- (L3): during decoding the full prompt KV ℓ⁻ is resident, and decoding takes at least (a+c)X.
- (L2+L3): per realization, Σ_{K completed by C_J}(σ_K + A_K/M) ≤ 2C_J, so S(σ + A/M) ≤ 2·OPT. This is valid even against the clairvoyant optimum.
- Source: Feng, Liu, Yang, Zhang, arXiv 2607.09248 v1 (HTML read). Lemma 1 is the "Area-order lower bound" ("the system can supply at most M units of memory-time area per round"). Their model:
  - rounds, unweighted total completion time;
  - memory s_i + u_{i,t} + 1;
  - kill-restart;
  - prompt lengths known, response lengths revealed at completion;
  - compared with the clairvoyant optimum;
  - Prop. 1: no single priority rule achieves a constant ratio;
  - a 16-approximation (Thm 6) and 3+o(1) for large memory (Cor 2).

  All of this matches doc l. 625–630.

### 3.4 (L1) is not a valid lower bound when the benchmark may preload fan-out pieces: GAP

**The claim (l. 316–318).** "Node y needs at least αq_y + (a+c)X_y seconds". It is used as a critical-path length on top of the lags, so it implicitly assumes all αq_y prefill happens *after* y becomes available.

**Why that fails.** In the model, a node with `pieces` (a tool fan-out) becomes available when the *last* piece returns. The pieces' tokens are part of q_y.
- R6 and `sim_hybrid` let ALG prefill already-returned pieces before y is available.
- The benchmark is "OPT_na in the engine at speed 1 with memory M (preemptive, any keep policy, may keep parent KV)" (l. 143). It is non-anticipatory and sees returned pieces, so nothing forbids it from doing the same.

**Counterexample** (`test_COUNTEREXAMPLE_L1_invalid_when_benchmark_may_preload_pieces`).
- Instance:
  - one task, w = 1, r = 0;
  - plan node (q = 1, X = 1) → final node with pieces {10 tokens, lag 0} and {1 token, lag 10}, ρ = 0;
  - engine at speed 1 with α = a = 1 and m = 1 (so c = 1), unbounded memory.
- Explicit non-anticipatory schedule:

  | Step | Interval |
  |---|---|
  | plan prefill | [0, 1] |
  | plan decode at B = 1 | [1, 3] |
  | piece A prefilled as soon as it returns, before the node is available | [3, 13] |
  | piece B returns at 13 and is prefilled | [13, 14] |
  | final decode | [14, 16] |

- Cost: 16.
- The doc's LB_cp is (1 + 2) + 10 + 11 + 2 = **26 > 16**. The author's `h_lower_bounds` also returns 26 on this instance.
- (For comparison, ALG in the hybrid gives 24 with evict-always and 14 with hold+preload.)

**Consequence.**
- Theorem D's chain E[ALG] ≤ … ≤ [(1+β_cp) + 2max(1+β,θ)]·OPT_na is unproven against the stated benchmark whenever some node has pieces that return at different times.
- Every instance in the author's Thm D tests has such nodes (`_rtask_h`).
- I did not find a violation of the final constant itself: the slack elsewhere is large.

### 3.5 Theorem D: VERIFIED WITH FIX

Either fix is sound.

**(a) Restricted benchmark.** State the benchmark as OPT_na over engine policies that start a node's prompt prefill only after the node is available (no preloading). Keeping the parent KV remains allowed. Alternatively, assume every node's q-tokens arrive at once, i.e. no staggered pieces. Under either, (L1) holds and Theorem D holds exactly as stated. I verified the proof steps l. 409–414:
- the node-length inequality;
- Z′ ≤ max(1+β,θ)(σ + A/M), Z3-proved;
- monotonicity of Smith;
- the R0a lift.

**(b) Unrestricted benchmark (preloading allowed).**
- Replace (L1) by LB⁰ with node length (a+c)X. Decoding cannot start before availability, so this is valid for any policy.
- Then CP′_J ≤ CP⁰_J + Σ_{y∈J}α(q_y + ρ_y) and α(q+ρ) ≤ (1+β)σ.
- Σ_J w_J σ_J ≤ OPT, because J's GPU work lies in [r_J, C_J].
- Hence:

> E[Σ w_J C_J] ≤ LB⁰ + (1+β)Σ_J w_J σ_J + max(1+β,θ)·S(σ + A/M) ≤ [(2+β) + 2·max(1+β, θ)]·OPT_na.

With unbounded memory this is 3 + 2β instead of 2 + β_cp + β. The first inequality is checked exactly in `test_thmD_corrected_bound_with_preload_aware_L1` (120 instances).

**Other wording fix (l. 306–309 and l. 700–701).**
- The index w_J/Z′_J is "computed at release" (l. 334), and Z′_J needs every node's q, ρ and X.
- So the token counts must be deterministic and known at release, as the summary table (l. 41) says.
- The sentence "token counts are known when a node becomes available" (l. 307, and §10.6 "known at node availability") is not enough for the policy as defined.

**Engine lift (l. 416–418).**
- Lifting the hybrid to the engine needs the speed-3 engine to reproduce the hybrid rates exactly, not merely to exceed them. Running faster under list scheduling can change the schedule (anomalies).
- Exact reproduction holds in the fluid model by time-sharing: a fraction λ = (a+cB)/(2a) ≤ 1 of the time in configuration (u = 1/3, all B decoders), and the rest in (u = 1/3, no decoders). Z3-proved in `test_R0a_z3_own`.
- Memory states are identical, so the use of R0a is legitimate.
- The augmentation is used consistently: ALG runs at speed 3 with memory M′, OPT at speed 1 with memory M, and M = M′ − L_max appears in Z′, A/M and Lemma D1 case (B).

**Calibrated table (l. 431–440).** Recomputed exactly (`test_thmD_calibrated_constants_table`): 0.03, 0.75, 2.78; 3.48, 1.1, 9.99. Correct.

## 4. R0, R1 (doc §5, l. 447–483)

### 4.1 R0a: VERIFIED
I proved four facts in Z3 with my own encoding:
- the lift (2/(a+cB) ≥ 1/a for cB ≤ a);
- tightness (s·u ≥ 1 and s(1−u) ≥ 2 force s ≥ 3);
- the time-sharing realization above;
- the converse (each speed-1 decoder at most 1/a, total decode at most 1/c = m/a, prefill at most 1/α).

"Tight" here means tight for per-instant rate domination, not for the theorem.

### 4.2 R0b: VERIFIED (proof sketch)
- The argument on l. 463–466 is a one-paragraph sketch. Its ingredients are correct:
  - steps last ≤ a;
  - one token per step for each decoder;
  - P̄ = a/α tokens per step on the server, except the last chunk;
  - one step of availability rounding per path node;
  - one wasted partial chunk per prefill completion, where recompute prefills also count in N_J.
- The Lemma D1 charging only needs rates at least the hybrid rates, which the discrete engine provides.
- Evidence: own discrete-step simulator, 354,067 checks, no violation. The hybrid bound alone was never exceeded, which matches the doc's remark that the additive term looks loose.
- A complete write-up should spell out the step-level version of each case.

### 4.3 R0c: VERIFIED WITH FIX
- "Running m_eff sequences is memory-safe, and the work term grows by m/m_eff" is immediate.
- The lower-bound instance as described (l. 477–479), "n = m = 8 equal decodes that all fit in memory, with m_eff = 1", is inconsistent. With ℓ⁺ = 11, m_eff = 1 needs M′ < 22, and then the 8 decodes do not all fit for policy P either.
- The test simply compares m = 1 with m = 8 at unbounded memory. I reproduce 1286/287.
- Consistent instance (`test_R0c_numbers_and_consistent_instance`):
  - add one low-weight long-context task (ℓ⁺ = 50) and set M′ = 99;
  - then m_eff = min(8, ⌊99/50⌋) = 1, while P admits all eight small decodes;
  - the small tasks' total completion time is more than 4 times worse under the cap.

### 4.4 R1: VERIFIED as part of Thm D.

## 5. R5 and R6 (doc §6, l. 487–564)

**R5a: VERIFIED.**
- The only information during a wait is the elapsed time, so deterministic rules are TTLs and the cost is linear in the mixture.
- The cost has slope h·P(D > τ) ≥ 0 between atoms, so the candidates {0} ∪ supp(D) suffice. Checked against a 1/8 grid on 300 random laws.
- The Karlin et al. density: my closed-form expected cost equals e/(e−1)·min(d, B) to 1e-12.
- The Yao-LP values 1.333/1.463/1.523/1.553 are exactly the discrete ski-rental ratios 1/(1−(1−1/n)^n).
- Exponential waits: cost(τ) = h/λ + (R − h/λ)e^{−λτ}, so the rule "hold iff hE[D] < R" is correct (Z3).

**R5b: VERIFIED** (for the linearized-price LP, as the doc states).
- Per class, after the λ_i factor cancels, the Lagrangian subproblem is cost(τ) with h = ηL_i and R = (ν_cp+ν_ld)P_i.
- Issue iff max_τ[σ(τ) − ηL E min(D,τ)/((ν_cp+ν_ld)P)] > ν_ld/(ν_ld+ν_cp) = ρ_s/(1−u+ρ_s). Z3-proved.
- Strong duality: the primal, solved by my own vertex enumeration (one coupling constraint, so at most one fractional class), equals the exact dual maximized over all breakpoints on 25 instances.
- The BE′ closed form and its first-order term σν_cp − (1−σ)ν_ld are Z3-proved. They match `controller_theory.md` l. 246.

**R5c: VERIFIED.**
- Safety is exactly Lemma D1, which is stated for any keep policy with held KV in the lowest class.
- Per-task counterexample reproduced with my simulator: task 2 goes 24 → 27, the weighted total improves, and the bounds hold.

**R6: VERIFIED WITH FIX.**
- Safety (Thm D unchanged with lowest-priority preloads) follows from Lemma D1 and is checked.
- The claim "for an isolated task, the saving is exactly α·(held hits)" is **false for DAG tasks**:
  - instance: plan → a (lag 5), plan → b (lag 0), both with ρ = 2; b has X = 100; α = 1/4, m = 2;
  - the saving is 1/2, but α·hits = 1, because the hit on a is off the critical path (`test_COUNTEREXAMPLE_R6_exact_saving_fails_for_DAG_tasks`).
- The claim holds for chain tasks, which is all the author's test covers (`test_R6_exact_saving_holds_for_chains`). For DAGs it should read "at most".

## 6. R7 (doc §7, l. 568–591)

- **R7.2:** VERIFIED.
  - Each task holds at least 4 tokens from plan start to final end, and ⌊11/4⌋ = 2, so at most 2 tasks are in flight.
  - Hence the i-th completion is at least ⌈i/2⌉·T, and Σ_{i≤12}⌈i/2⌉/12 = 7/2.
  - The ratio is against evict-always, which upper-bounds OPT, so it lower-bounds the ratio against OPT.
- **R7.5(i):** VERIFIED by exact enumeration. The ratio is (n+1)/(nq+2−q), and 1 + Δ = 1/q.
- **R7.5(ii):** VERIFIED in the source (see §7).
  - EFMM's lower bound is for index policies on P‖E[ΣC_j] with all jobs released at time 0.
  - There, P-WSPT-DAG never preempts, so it is a WSEPT list policy and the bound applies to it.
  - This is consistent with Thm B's O(Δ) upper bound.
- **R7.6:** lag trap, VERIFIED.

## 7. Cited sources

| Citation | Checked | Finding |
|---|---|---|
| Megow–Schulz, ORL 2004 | abstract + excerpt | Thm 2.1: P-WSPT ≤ 2·OPT for P\|r_j,pmtn\|ΣwC. Matches. |
| Möhring–Schulz–Uetz, JACM 1999 | via Skutella–Uetz's restatement (JACM PDF not opened) | Cor 3.1 / Lemma 4.2 as restated in SU Thm 3.1 / Lemma 3.3. Matches the doc's use. |
| Skutella–Uetz, SICOMP 2005 | full text (extracted from the PDF) | (3.1), Thm 3.1 (any non-anticipatory policy), Lemma 3.3, and the ratios (in-forest 2 − 1/m + max{…}; general with release dates (1+β)(1+1/β+max{…})). Match l. 640–643. |
| Megow–Uetz–Vredeveld, MOR 2006 | preprint snippet | 1 + max{1+δ/α, α+δ+(m−1)(Δ+1)/(2m)} for δ-NBUE. Matches l. 650. |
| Megow–Vredeveld | as EFMM ref. [18], "A tight 2-approximation for preemptive stochastic scheduling", MOR 39(4) 2014 | Matches l. 652. The "Θ(n) lower bound, jobs revealed at predecessor completion" claim (l. 651) is **not checked**; source not accessed. |
| Eberle–Fischer–Matuschke–Megow, ORL 2019 | full text (extracted) | Thm 1: index policies Ω(Δ^{1/4}) for P‖E[ΣC_j]. Ref [18] is the Megow–Vredeveld preemptive 2-approximation. Matches. |
| Feng–Liu–Yang–Zhang, arXiv 2607.09248 | HTML v1 (first 100k characters) | Lemma 1 area-order lower bound, Prop 1, Thm 6 (16), Cor 2 (3+o(1)), model details. Match l. 625–630. |
| Jaillet et al., arXiv 2502.07115 | abstract | "no deterministic online algorithm can achieve a constant competitive ratio when the arrival process is arbitrary", for total latency. Matches. |
| Queyranne–Schulz, SICOMP 2006 | abstract | 4-approximation with precedence delays, list scheduling in LP-midpoint order. Matches. The parenthetical "later 2 + 2ln2 + ε" is **not supported for delays**: Li (arXiv 1707.08039) proves it for P\|prec\|ΣwC, and the paper never mentions precedence delays. |
| Continuum 2511.02230 | not checked | — |

## 8. Code vs paper, and test strength

**Code matches the doc.**
- `sim_machine`, `pathwise_bound`, `lb2_task_msu`, `msu_lp`, `sim_hybrid`, `h_sizes`, `h_bound`, `h_lower_bounds`, `sim_discrete` and `disc_bound` implement what the doc states. My independent hybrid simulator reproduces `sim_hybrid`'s completion times exactly on evict-always runs.
- The author's DPs are restricted to grid decisions. They are exact for m = 1 (my grid-refinement test supports this). For m ≥ 2, `opt_pmtn_dp` is only an upper bound, as the doc says (l. 723).
- The reported worst ratios 1.128, 1.842, 1.068 and 1.098 reproduce from the author's seeds.
- Not tested anywhere: 1.45 and 29.3 (l. 426–427), and 1.153 and 1.244 (l. 294–295). I did not reproduce these.

**Mutation testing of `tests/test_stochastic_theory.py`.** I patched the module and re-ran the relevant test (script kept outside the repo).

| Mutation | Test | Result |
|---|---|---|
| Lemma A1 with 1/(m+1) instead of 1/m | thmA_pathwise_bound_every_realization | **fails (good)** |
| Lemma A1 without CP_J | thmA_pathwise_bound_every_realization | **fails (good)** |
| Lemma A1 with K ≺ J (own work dropped) | thmA_pathwise_bound_every_realization | **fails (good)** |
| Smith LB × 1.2 (invalid LB) | thmA_two_approx | **fails (good)** |
| Z′ halved | thmD_pathwise | **fails (good)** |
| ρ_B := 1 | thmB_online | **fails (good)** |
| Smith term × 0.8 in ALG ≤ LB_cp + Smith/m | thmA_parallel_machines | passes: test is loose |
| γ := 0 (variance correction dropped) | thmB_online | passes: the random instances rarely have γ < 0 |
| LB2 × 1.1 | thmB_online | passes: LB2 validity is not stressed |
| ρ_B := 1.1 | thmB_online | passes (expected: the ratio test is loose) |
| Z′ without κ, or without the area term | thmD_pathwise | passes: random instances do not stress these; only the dedicated instances in thmD_hypotheses_are_necessary do |
| R0b without the additive term | R0b | passes (consistent with the doc's remark) |
| Thm D constants set to (0, −1/2, 1/2) | thmD_ratio | passes: the test is very loose |
| e/(e−1) replaced by 1.4 | R5a | passes: the random laws never exceed 1.4; only the LP part (> 1.55) is near-tight |

So the pathwise-lemma tests are non-vacuous. The ratio tests mostly confirm loose inequalities and would not catch small constant errors. The necessity of κ, kicks and the L_max margin is shown only by the dedicated instances in `test_thmD_hypotheses_are_necessary`. My `test_lemmaD1_mutations_detected_by_own_checker` reproduces two of those three.

## 9. Recommended changes to the doc (not applied; the doc was not edited)

1. **(L1) / Thm D.** State the benchmark restriction "OPT_na does not prefill a node before it is available", or assume no staggered fan-out pieces. Otherwise use the corrected constant (2+β) + 2·max(1+β, θ) (3 + 2β with unbounded memory), with LB⁰ built from node lengths (a+c)X.
2. **Thm D assumptions** (l. 306–309, 700–701). Token counts must be known at release, because the index is computed at release.
3. **Lemma D0 proof** (l. 356–358). Add the case where y does not fit (head-of-line stop at y), and say that simultaneous arrivals are charged to each arrival.
4. **Lemma D1 case (B)** (l. 375–377). Use "> M′ − ℓ⁺_u, where u is the first node that does not fit", and "at least m decoding" instead of "more than m".
5. **R6** (l. 561–563). "Exactly" holds for chain tasks. For DAG tasks the saving is at most α·(held hits), and can be strictly less (instance above).
6. **R0c** (l. 477–479). Give a consistent instance with a long-context task fixing L_max (as above), or describe the test as "m = 1 versus m = 8".
7. **R0b**. Label it "proof sketch" or expand the step-level case analysis.
8. **§9** (l. 662–663). Drop "later 2 + 2ln2 + ε" for precedence delays, or cite a source that handles delays.

## 10. Not checked
- The Megow–Vredeveld "Θ(n) lower bound" model remark (l. 651) and Continuum (l. 665): sources not accessed.
- The MSU JACM paper itself: checked only through Skutella–Uetz's restatement.
- The rest of Feng et al. after the first 100k characters.
- R7.1 and R7.4 (they live in `open_problem.md`, outside this commit).
- Exact OPT for the engine model (Thm D). Its lower bounds were checked analytically, the preload counterexample by an explicit schedule. I did not compute an exact engine optimum, so I cannot say whether the original Thm D constant actually fails against a preloading OPT. I only show that its proof does not cover that case.
- Lean/Coq formalisation: not attempted. Z3 covers the algebraic lemmas.

---

## Re-verification (c46b893), 2026-11-07

This section covers the author's revision at commit c46b893:
- `codesign/docs/stochastic_theory.md` (907 lines);
- `src/codesign/stochastic_theory.py`;
- `tests/test_stochastic_theory.py` (32 tests).

These three files are byte-identical at c46b893, at HEAD (000df35) and in the working tree. Line numbers below refer to `git show c46b893:codesign/docs/stochastic_theory.md`.

What I did:
- Re-read every changed passage line by line, against the full diff from e17bd7f.
- Re-ran my independent suite: `tests/test_stochastic_verify.py`, now 41 tests, all passing (`../project/.venv/bin/python -m pytest -q tests/test_stochastic_verify.py`, about 60 s). The author's 32 tests also pass.
- Added 10 tests (`test_reverify_*` and one `test_CROSS_CHECK_*`) and the helpers they need (`Node.q0`, `beta_cp0`, `lb0_q0`, `thmD_unrestricted_constant`, `schulz_skutella`, `release_order_np`, `d0_tight_family`, `r6_isolated`).
- Mutation-tested the new and tightened tests (21 mutations, table below).

### Verdicts

| Item (new doc lines) | Verdict | Evidence |
|---|---|---|
| (L1) restricted to "no prefill before availability" (l. 381–386) | **VERIFIED** | The restricted (L1) is the original argument, now with the scope that makes it valid. The counterexample numbers 16 < 26 and LB⁰ = 15 are reproduced. |
| Arrival assumption (l. 370–373) and q⁰ (l. 387–393) | **VERIFIED** (one wording suggestion) | Under the stated assumption, a piece-free node's q tokens do not exist before availability, so αq⁰ + (a+c)X after availability is a valid per-node length for every engine schedule. q⁰ = 0 for nodes with pieces is conservative and always valid. Suggestion: say explicitly that a multi-parent node whose inputs come from different parents' tools at different times must be modelled with pieces. Otherwise the assumption is a real restriction of the model, not a property of it. |
| (L0) Σ w σ ≤ OPT_na (l. 394–395) | **VERIFIED** | J's prefill and decode tokens cannot be processed before r_J. By the (L2) rate inequality, α·prefill + c·decode ≤ elapsed time on [r_J, C_J]. |
| Thm D restricted: (1+β_cp) + 2max(1+β,θ) (l. 496–507) | **VERIFIED** | Same proof as before; the scope is now stated. Step 1 now also notes CP′ ≤ (1+β_cp)CP^opt (Z3 earlier; exact check in the author's test). |
| Thm D unrestricted: min(1+β_cp⁰, 2+β) + 2max(1+β,θ) (l. 509–524) | **VERIFIED** | Both branches: (a) the node-wise inequality α(q+ρ)+aX ≤ (1+β_cp⁰)(αq⁰+(a+c)X); (b) the additive α(q+ρ)+aX ≤ (a+c)X + (1+β)σ, with LB⁰ ≤ OPT, Σwσ ≤ OPT, S ≤ 2OPT. All Z3-proved in `test_reverify_thmD_unrestricted_algebra_z3`. The statement's use of LB⁰ (with q⁰) in the additive branch is valid, because LB⁰(q⁰) ≥ LB⁰(q⁰ ≡ 0). Exact check on 150 random instances (chains and DAGs, bounded and unbounded memory, hold and preload), including both pathwise step-1 forms: `test_reverify_thmD_unrestricted_both_branches`. Unbounded-memory form min(…) + (1+β) ≤ 3+2β: correct. |
| Sentence "ALG = 24 > (1+β_cp)·LB⁰ = 15, so the restricted constant cannot simply be reused against LB⁰" (l. 525–527) | **FIX STILL NEEDED** (wording only) | The numbers are right (ALG = CP′ = 24, LB⁰ = 15, β_cp⁰ = 11; reproduced). But the instance refutes only **step 1** with β_cp against CP⁰. The full restricted-constant inequality still holds here: (1+β_cp)LB⁰ + max(1+β,θ)·S = 15 + 14 = 29 ≥ 24. Suggested wording: "so step 1 cannot use β_cp against CP⁰". The next paragraph correctly says the restricted constant is open against the unrestricted benchmark. |
| Engine box (l. 533–546), "without pieces the versions coincide" | **VERIFIED** | With no pieces, q⁰ = q and β_cp⁰ = β_cp. Under the arrival assumption the two benchmark classes coincide, and the unrestricted constant is ≤ the restricted one. The time-sharing remark matches my Z3 check (λ = (a+cB)/(2a)). |
| Calibrated table (l. 572–578): 2.80 unbounded, 10.01 at 64k | **VERIFIED** | Recomputed from scratch with exact rationals: β_cp⁰ = 0.0506, 2.80, 10.01; restricted 9.99 (`test_reverify_calibrated_unrestricted_table`). |
| Lemma D0 rewritten proof (l. 432–444) | **VERIFIED** | Cases 1–3 checked in Z3 (y fits; y does not fit) plus 117k simulated kick events, including simultaneous arrivals. One trivial case is unstated: if the old prefix already stopped above y, nothing changes and nothing is kicked. Harmless. |
| D0 tightness family (l. 446–450; the extension verifier's family) | **VERIFIED** | Own simulator: one kick event with kicked reservation exactly ℓ⁺_y + L_max − 1, recompute > ℓ⁺_y + L_max − 3, Lemma D1 holds, for (L, s) = (10,3), (50,20), (200,50) (`test_reverify_lemmaD0_tight_family_own_simulator`). Using a fixed priority order (y, z1, z2) is legitimate, because Lemmas D0/D1 hold for any fixed order. |
| Lemma D1 case (B) fix (l. 464–467) | **VERIFIED** | "> M′ − ℓ⁺_u ≥ M" and "at least m": correct and complete. |
| Thm A tightness 2 via Schulz–Skutella / Megow–Schulz (l. 242–262) | **VERIFIED** | Own simulator plus own exact DP: n = 3 scaled by 6 gives ALG = 228 and OPT = 160 = release-order UB; ALG/UB = 1.622, 1.891, 1.9846 at n = 5, 20, 150. Checked structurally: each job is preempted with exactly 1/n left, and the release-order schedule never idles. Asymptotically ALG ≈ (n+1)n² while UB ≈ n³/2, so the ratio → 2. Source: Megow–Schulz states the P-WSPT bound 2 is tight (abstract/snippet) and Schulz–Skutella that the ratio is "precisely 2" (snippet). I could **not** open the full PDFs (redirects / 403), so the label "Lemma 2.2" and "for any number of machines" are not checked. |
| Lag-trap demoted to a remark; "tightness open" removed (§2.4, §7.6, §10.3) | **VERIFIED** | Consistent everywhere. |
| Smith-term sharpness (l. 237–238) | **VERIFIED** | 210 and 110 ≤ 125 reproduced independently. |
| LB2 sharpness (l. 326–330) | **VERIFIED** | Own code: the γ < 0 case, list schedule E = 69.97 (a non-preemptive policy, so ≥ OPT_np) < LB2(γ:=0) = 392/5, LB2 ≤ 69.97; 11 unit jobs: LB2 = 121/2, OPT_np = 66 < 1.1·LB2. |
| Token counts known at release (l. 365–368, §10.7) | **VERIFIED** | Fixed as recommended. The new §10.7 claim that bounds q̄, X̄ known at release "restore the theorem with those bounds in Z′ and in the constants" is plausible: every Lemma D1 charge only gets larger, and the lower bounds use actual sizes. But it is unproved, and the constants must be redefined as reserved-over-actual ratios. It sits in Limitations, so this is acceptable. Suggestion: write "we expect". |
| R0b relabelled "proof sketch" (l. 596–599, §10.6) | **VERIFIED** | Matches my verdict. |
| R0c both instances (l. 617–625) | **VERIFIED** | Matches my fix, reproduced. |
| R6 chains exact (l. 707–712) | **VERIFIED** | Reproduced on 60 chain instances. |
| R6 DAGs "no identity" (l. 714–726) | **VERIFIED; my earlier suggestion was wrong** | See ruling (a). |
| R7.5 wording (worst-case ratio ≥ 1+Δ as a supremum) | **VERIFIED** | (n+1)/(nq+2−q) increases to 1/q = 1+Δ. |
| §9 Queyranne–Schulz edit; new credits | **VERIFIED** | "later 2+2ln2+ε" dropped, with the reason. |
| Code: `HNode.q0`, `h_cp` 'opt0', `h_lower_bounds`/`h_constants` preload_aware, `thmD_constant`, `kicked_res` | **VERIFIED** | They match the doc. `test_CROSS_CHECK_author_new_code_matches_ours` compares them with my independent implementations on 80 random instances: q0, LB_cp, LB⁰, β_cp⁰, the unrestricted constant, completion times, and the per-event kicked reservation all agree exactly. |
| Unrelated changes in the diff | **none found** | The doc, code and test diffs (e17bd7f → c46b893) only touch the fixed items, the new verification-status table and cross-references to `stochastic_theory_ext.md`, which is out of scope. In the tests, the κ/margin instances were refactored into helpers (`_kappa_stress`, `_margin_stress`) without changing them. |

### Ruling (a): R6 for DAG tasks

**The author is right, and my e17bd7f recommendation ("for DAGs it should read 'at most α·(held hits)'") was false.** I reproduced both instances with my own simulator (`run_P`) and my own hold rule: hold the child's parent context ρ when the parent completes, LRU, lowest class; preload only returned pieces. Results are identical with and without preload, since there are no pieces (`test_reverify_R6_DAG_saving_both_directions_and_negative`):

- **Saving larger than α·hits.** α = 1/4, m = 1, plan (q1, X1) → a (q2, X1, ρ1, lag 1) and → b (q3, X1, ρ2, lag 0).
  - Evict-always completes at 5. Holding completes at 4: b's shorter prefill ends at 2, before a arrives at 9/4, so a no longer preempts it.
  - Saving 1 > α·hits = 3/4.
- **Negative saving.** α = 1/4, m = 2, plan (q3, X1) → a (q2, X1, lag 1), → b (q1, X2, ρ1, lag 1), → c (q3, X3, lag 0).
  - Evict-always completes at 6. Holding completes at 25/4.
  - I traced it by hand: b's earlier decode start preempts c's decode one quarter earlier, and c ends later.
  - Saving −1/4 < 0.

Lemma D1 holds in all runs. So for DAG tasks there is no identity and no inequality in either direction. Only the safety statement (Thm D with any keep/preload policy) survives, which is what the revised doc says (l. 714–728). Together with my earlier instance (saving 1/2 < α·hits = 1), all three directions are confirmed.

### Ruling (b): the unrestricted constant min{1+β_cp⁰, 2+β}

**Accepted.**
1. The assumption is stated explicitly (§4, l. 370–373): a piece-free node receives its q tokens at availability, and piece i's tokens arrive when piece i returns.
2. It is a valid modelling assumption, and it is exactly what makes (L1⁰) with q⁰ = q sound for piece-free nodes. The doc also says what to do if some tokens arrive early (shrink q⁰), and q⁰ ≡ 0 is always valid.
3. Both branches are proved; I checked each step:
   - the node-wise branch, with CP′ ≤ (1+β_cp⁰)CP⁰;
   - the additive branch (mine), with CP′ ≤ CP⁰ + (1+β)σ_J.
4. Each branch is an upper bound on the same quantity, so their minimum is too.
5. The result is never weaker than my (2+β) + 2max(1+β,θ), and it is strictly better when fan-out prompts are short relative to decodes. Calibrated, the author's form gives 2.80 versus 4.50 for my additive form alone with unbounded memory, and 10.01 versus 14.43 at M′ = 64k.

Remaining caveat: the wording suggestion for multi-parent nodes in the q⁰ row of the verdict table above.

### Mutation testing of the revised author tests

All 21 mutations make the targeted test fail (script kept outside the repo):

| Test | Mutations |
|---|---|
| thmA_two_approx, thmA_parallel | Smith × 0.8 |
| thmB_online | γ := 0 (two variants), LB2 × 1.1 |
| thmD_pathwise | Z′ without κ; area term normalized by M′ |
| thmD_ratio | constants (0, −1/2, 1/2); β 10% low |
| R5a | e/(e−1) → 1.4 |
| COUNTEREXAMPLE_L1 | LB⁰ := LB_cp |
| thmD_unrestricted | LB⁰ := LB_cp; β_cp⁰ := β_cp; β_cp⁰ halved; constant without the min; 'opt0' using q instead of q⁰ |
| lemmaD0_tight_family | no kicks; M′ + 1 |
| COUNTEREXAMPLE_R6 | hold ignored |
| thmA_tight | FIFO order instead of WSPT |
| R0c | P with m = 1 |

So every test whose weakness I reported for e17bd7f now catches its mutation.

One observation: in `test_thmD_ratio_against_engine_lower_bounds`, the new assertion `max(r_cp) == bcp` (and the same for β, θ) recomputes the same formula as `h_constants`, so it is a consistency check rather than evidence. The substantive new checks are Z′_K ≤ max(1+β,θ)(σ_K + A_K/M) and CP′ ≤ (1+β_cp)CP^opt.

### Still open or not checked
- Whether the *restricted* constant also holds against the unrestricted benchmark. The doc correctly calls this open (l. 529–531 and §10.4). No exact engine optimum has been computed.
- The full texts of Megow–Schulz (ORL 2004) and Schulz–Skutella (J. Sched. 2002), as noted in the Thm A tightness row.
- `stochastic_theory_ext.md` and its verification, which are out of scope for this re-verification.
