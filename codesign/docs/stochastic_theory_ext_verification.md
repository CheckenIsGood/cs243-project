# Independent verification of `docs/stochastic_theory_ext.md` (commit eaa6958)

**What was checked**

- **Scope:** Thm G, Thm P (with Lemma P1/P2), Thm H, Thm DR, Prop K, the κ lower bound, and the Thm A √2 lower bound.
- **Method:**
  - a line-by-line audit of every hypothesis, inequality and constant;
  - primary sources checked where they could be accessed;
  - my own exact code: `src/codesign/stochastic_ext_verify.py`, which imports nothing from the author's
    modules;
  - 23 tests: `tests/test_stochastic_ext_verify.py` (all pass, about 23 s).

**Notation**

- Line numbers refer to `docs/stochastic_theory_ext.md` at eaa6958 ("ext L.n"). The base doc is cited as "[base]".
- Test names are cited as **[V: name]**, without the `test_` prefix.
- The base theorems (A, B, B′, D, R0, R5) are audited by the other verifier and are **not** re-verified here. When an
  extension depends on one of them, the dependency is noted.

## Summary

| Result | Verdict | One-line reason |
|---|---|---|
| Lemma G2 + Gittins single-machine optimality for chain work processes | **VERIFIED** | Proof correct. On every realization, the identity equals direct simulation. Gitt₁ equals my exact DP on 30 instances. |
| **Thm G** (Gittins quanta, chains, lags, releases, m machines, ≤ 2·OPT_na) | **VERIFIED WITH FIX** | Proof correct, and it matches Megow–Vredeveld's F-GIPP proof step for step. "Random sizes with **any** law" (ext L.33, L.130) is not proved: MV, the doc's proof and the code all assume **discrete** laws. Restate for discrete (finite-support) laws. |
| Lemma P1 (E min ≥ max{16/(27(1+Δ)), 1−√(Δ/2)}) | **VERIFIED** | Every step checked; algebra by Z3. 3000 exact random law pairs plus annealing found no violation. Side finding: c\* ≤ **58/77 ≈ 0.7532**, sharper than the doc's 0.78 / "0.776". |
| Lemma P2 and **Thm P** (≤ (1+ρ(Δ_W))·OPT_clair, DAGs) | **VERIFIED** (minor wording fixes) | Proof correct. Δ_W ≤ max node CV² holds even for **correlated** nodes (Minkowski; Z3). The "1+Δ necessity" family is exact. The second branch of ρ needs Δ < 2. |
| **Thm H** (random X, unbounded memory, ≤ 1+β_cp+λ) | **VERIFIED WITH FIX** (scope) | H1, H2, H3 and the chain are correct, and λ really is Δ-free (λ ≤ 1 + max(ρ+ctx·X̄_par)/q). It inherits base (L1), which the base verifier refuted for prompts with fan-out pieces that return early. State that a node's q tokens cannot be prefilled before it is available, or use the preload-aware LB_cp (calibrated 2.67 → 2.74). |
| **Thm DR** (cap reservation, ≤ (1+β_cp)+2λ_E ρ(Δ_s) vs OPT_clair) | **VERIFIED WITH FIX** (scope) | Steps 1–4 correct. My own memory simulator confirms Lemma D1 pathwise with cap reservations (5908 checks; 28 instances with kicks). Same (L1) scope fix as Thm H (8.68 → 8.74 preload-aware). Base D0/D1/(L2+L3) are audited elsewhere. |
| **Prop K** (constant (1+β_cp)+2λ_D) | **VERIFIED** | Correct, with equality for one task. Calibrated 4.82 / 5.80 / 6.75 and old 9.985 all recomputed exactly. Same inherited (L1) caveat (5.80 → 5.82 preload-aware). |
| κ lower bound (9.49 recompute tokens) | **VERIFIED** | Recompute is exactly 949/100. I also give a general family where the kicked reservation is ℓ⁺_y + L_max − 1 for any ℓ⁺_y, so D0 is tight up to an additive 1. |
| Thm A √2 instance | **VERIFIED** (the ratio tends to √2 but never reaches it) | Z3 proves sup = √2 and that no finite instance reaches it. L = 1000, k = 1414 gives 1.41371. |
| Claims "√2 improves 4/3", "gap √2 vs 2", "Thm A tightness open" (ext L.49, L.379–380, L.391, L.405) | **FALSE** | A lower bound of **2** for P-WSPT on **one** machine (unit weights, deterministic, no lags) is already known: Schulz–Skutella (J. Sched. 2002), restated as Megow–Schulz (ORL 2004) Lemma 2.2. P-WSPT-DAG on single-node tasks *is* P-WSPT, so Thm A's constant 2 is **tight**. Exact: n = 150 gives a ratio of 1.9846. |
| Calibrated constants (2.52, 2.67, 8.68, 5.80, 4.82, 6.75, 9.99, λ_E 2.33, Δ_s 0.30, ρ 1.63, CV²(X) 0.85) | **VERIFIED** | All recomputed exactly from the stated inputs with independent closed forms and path enumeration. |
| Author's tests (20) | Pass, **mostly discriminating** | Mutations of the bounds are caught, with three weak spots (see §9). The P1 test passes with the **false** constant 0.76/(1+Δ). |

No FALSE mathematical theorem was found. The one FALSE item is a **literature-status claim**: the √2 bound is
correct but superseded, and Thm A is tight.

---

## 1. Lemma G2 and Gittins optimality (ext L.62–96)

**Hypotheses.**
- Chain work processes: lumps, plus N_i units revealed one at a time, with N_i independent.
- One machine, speed 1, no releases, no lags.
- Quanta: a quantum starts at x₀ and lasts until the first state y with G(y) < G(x₀).
- Key: (−G, id).

**Audit.**
- Priorities of successive quanta are strictly decreasing **by construction**: a new quantum starts only when G drops
  strictly. So keys of one task strictly increase, and the global key order is a total order that respects each
  task's own quantum sequence.
- J's last quantum is preceded exactly by J's other quanta and by quanta of other tasks with smaller keys, so
  C¹_J = W_J + Σ_{K≠J} V_KJ. Correct.
- Ordering quanta by key is non-anticipatory: at any time the smallest **current** key is the smallest remaining key.
- **Quantum definition vs MV.** The doc's rule "until the first state with a strictly smaller index" is the maximal
  optimal stopping rule. It matches MV's q_j(y) = max{q : R_j(q,y) = R_j(y)} and their Prop. 2(a–c). MV restrict to
  deterministic lengths q because a single-node job reveals only "not done". For chains, stopping over states is the
  correct generalization.
- **Gittins optimality.**
  - MV's Theorem 1 states optimality for single-node jobs with discrete laws, citing Konheim, Sevcik and Weiss 1995.
    MV note that Weiss covers more general job processes.
  - Chains are Markov job processes with state (node, units done, previous length). Optimality for them is the
    general Gittins "jobs" theorem. I could not access Gittins 1989 or Weiss 1995 directly.

**Evidence.**
- My Gittins index (brute force over **all** continuation sets) equals my separate Dinkelbach implementation on
  > 250 states [V: gittins_brute_equals_dinkelbach_and_quanta_strictly_decrease].
- G2 equals direct simulation equals W + ΣV on every realization, and Gitt₁ equals my exact single-machine DP on 30
  random 2–3-task chain instances. The doc's WSEPT example 15 > 57/4 is reproduced
  [V: G2_identity_and_gittins_single_machine_optimality].

**Verdict:** VERIFIED.

## 2. Theorem G (ext L.98–159)

**Hypotheses as used.**
- Chain tasks.
- Node unit counts independent across nodes and tasks, with **discrete** laws (see the fix below).
- Lags with any joint law, independent of sizes and of the policy.
- Integer release dates, online arrival, m identical machines at speed 1.
- Benchmark: OPT_na, preemptive and non-anticipatory.

**Lemma G1 (fast single machine).**
- The relaxation is valid with lags and chains. π's per-task rates are at most 1 and sum to at most m, so a speed-m
  single machine can reproduce them.
- The machine samples lags from their joint law with its own randomness. This needs lags independent of sizes, which
  is stated. The resulting randomized policy is non-anticipatory for the size processes, and task completions
  coincide.
- Dropping releases and lags only relaxes the problem. Scaling gives OPT₁(speed m) = Gitt₁/m.
- This is exactly MV Lemma 4 + Thm 5 + Cor. 6 (memorandum RM/09/054, §3), extended by the lag sampling. The argument
  also holds against policies that know lags in advance, which is what my DP exploits.
- Adversarial check. My m-machine grid DP (step 1/2, lags revealed at start) gives an **upper** bound on OPT_na.
  Gitt₁/m above that DP value would refute G1. Search over 3–4-task instances at m = 2 and 3 never got the ratio
  UB/(Gitt₁/m) below 1.25 [V: thmG_lemma_G1_adversarial_m2_m3].
- The doc's instance reproduces: grid OPT 59/8 ≥ Gitt₁/2 = 6, and 59/8 < Gitt₁ = 12, so speed m is needed.

**Lemma G3 (pathwise).**
- Instants in [r_J, C_J] fall into three types:
  - J in a lag (total = Σ lags);
  - J running (total = W_J);
  - J available but not running, so m machines run tasks whose current key is smaller than J's current key, which is
    ≤ J's last key.
- Work done in the third type belongs to quanta counted in V_·J, so the total is ≤ (1/m)Σ V_KJ.
- **Lags making a task unavailable in mid-quantum do not matter.** Quanta are defined on the task's own path; a lag
  changes no key, and lag time is in CP.
- This is MV Thm 7's proof (F-GIPP uses "rank R_j(y_{j,k+1}) of the current quantum", i.e. the running minimum, as
  the doc does). The only change is that lags enter CP.
- Evidence: 75,562 (task, realization) checks during development (15,000+ in [V: thmG_lemma_G3_pathwise_every_realization]),
  m ∈ {1,2,3}, chains of 1–3 nodes, random lags and releases. Zero violations, and over 40% of the checks are
  **tight**. The mutation that drops the task's own work fails.

**Theorem chain.**
- E[ALG] ≤ LB_cp + (Gitt₁ − Σ w E W)/m. Correct by G2 on the release-free, lag-free processes.
- LB_cp ≤ OPT_na pathwise. G1 gives Gitt₁/m ≤ OPT_na.
- On 25 random 2-task instances, m ∈ {1,2} [V: thmG_chain_and_lower_bounds_vs_grid_dp], with my own simulator, LB_cp,
  Gitt₁ and grid DP: every link holds. For m = 1 the step-1/2 grid DP always equals the step-1 grid DP (evidence that
  integer decisions are exact there). The worst ALG/OPT is 1.11 on my instances. The WSEPT mutation breaks the sharp
  bound [V: thmG_wsept_mutation_breaks_the_sharp_bound].

**Fix (needed).**
- ext L.33 and L.130 say "random sizes with **any** law".
- MV assume "random variables of processing times are stochastically independent and follow **discrete**
  probability distributions … integral" (RM/09/054 p. 2). Their Lemma 4 proof uses discreteness to define decision
  points. The doc's own model (ext L.64–72) has finite state spaces, and the code only handles finite laws.
- **Corrected statement:** Theorem G holds for unit-count (size) laws that are discrete with finite support,
  independent across nodes and tasks, with lags independent of sizes and of the policy. Continuous laws would need a
  continuous-time Gittins/quanta argument, which is not given.

**Verdict:** VERIFIED WITH FIX.

## 3. Lemma P1, Lemma P2 and Theorem P (ext L.161–222)

**Lemma P1, first bound.** Checked step by step.
- E min = ∫F̄Ḡ, and ∫₀^T F̄Ḡ ≥ (1/T)∫₀^T F̄ ∫₀^T Ḡ by Chebyshev's integral inequality (both survival functions are
  non-increasing). Correct direction.
- ∫₀^T F̄ = 1 − E(X−T)⁺ ≥ 1 − (1+Δ)/(4T), using (x−T)⁺ ≤ x²/(4T) (Z3) and E X² ≤ 1+Δ. The value is ≥ 2/3 > 0 at
  T = 3(1+Δ)/4, so squaring is valid.
- Z3 proves that T = 3c/4 is the **optimal** truncation: (1/T)(1−c/(4T))² ≤ 16/(27c) for all 4T ≥ c, with equality at
  3c/4 [V: lemmaP1_and_rho_algebra_z3].

**Lemma P1, second bound.**
- E|X−Y| ≤ √E(X−Y)² = √(Var X + Var Y) ≤ √(2Δ). Jensen is applied in the correct direction, and independence gives the
  variance sum.
- **Fix:** the bound and ρ's branch 1/(1−√(Δ/2)) are meaningful only for **Δ < 2**. Z3 confirms that the denominator
  is ≤ 0 at Δ ≥ 2. State "for Δ < 2, otherwise ∞".

**Lemma P1, exact checks.**
- 3000 random integer law pairs are checked exactly [V: lemmaP1_exact_random_and_witnesses].
- Simulated annealing over 2–4-point laws found no violation; the minimum found was 0.7534 ≥ 16/27.
- The doc's witness (Δ = 81/40, 187/240) is reproduced.
- **New witness:** X = {0 w.p. 19/20, 20 w.p. 1/20} and Y = {1/2 w.p. 76/77, 39 w.p. 1/77}. Both have mean 1 and
  CV² = 19, and (1+Δ)·E min = **58/77 ≈ 0.7532**. So c\* ∈ [16/27, 58/77]. This tightens ext L.183 and L.390 ("0.78",
  "local search found 0.776"). It is not an error, since 0.7532 < 0.78 keeps the stated bracket true.

**Lemma P2.**
- Both Smith values decompose as Σ w E W plus pair terms.
- WSEPT pair term: min(a, b). Clairvoyant pair term: E min(aX, bY) ≥ a·E min(X, Y) for a ≤ b.
- This needs only **pairwise independence of task totals**. Correct.

**Theorem P.**
- Lemma A1 (base, pathwise) plus the fixed ranking gives E[ALG] ≤ LB_cp + Smith(E W)/m.
- Then apply P2. (1/m)Smith(W(ω)) ≤ OPT_clair(ω) pathwise via the speed-m single machine; this holds for DAGs because
  that machine can run parallel nodes at total rate up to m.
- **The benchmark does not switch silently:** every link is against OPT_clair, and only the final step uses
  OPT_clair ≤ OPT_na.
- Lags may even be correlated with sizes; only the task totals must be independent across tasks.

**Δ_W ≤ max node CV².**
- For independent nodes, Var Σ = Σ Var ≤ Δ Σ(E p)² ≤ Δ(Σ E p)².
- **Stronger:** this holds for **arbitrarily correlated** nodes too, since sd(Σp) ≤ Σ sd(p) ≤ √Δ Σ E p (Minkowski).
  Z3 proves the two-node case, and it is checked exactly on random joint laws [V: thmP_chain_dags_correlated_nodes_m123].
- DAG totals are sums of node works, so the same holds.

**Exact chain checks.** [V: thmP_chain_dags_correlated_nodes_m123]
- 50 random instances: chains, forks and joins; correlated nodes on half of them; lags; releases; m ∈ {1,2,3}.
- Checked on each: E ALG ≤ LB_cp + Smith(EW)/m; Smith(EW) ≤ pair·E Smith(W); pair ≤ ρ(Δ_W), exactly.
- LB_cp and E Smith(W)/m stay below my clairvoyant grid upper bound. Worst ALG/UB_clair = 1.057.

**"Every fixed-priority policy loses 1+Δ" (ext L.215).**
- On the family n i.i.d. {0 w.p. 1−q, L w.p. q}, exact enumeration gives a ratio of exactly (n+1)/(nq+2−q) for n = 5, 6.
- Z3 shows this ratio is ≤ 1/q = 1+Δ, and it → 1+Δ as n → ∞ [V: thmP_necessity_family_one_plus_delta].
- On this family clairvoyant = probing, so the loss holds against OPT_clair and OPT_na alike.
- Wording fix: this is a *limit* (sup over instances), not a per-instance loss.
- "Tight up to the constant 27/16 (+1)" should read: the upper bound 1 + (27/16)(1+Δ) and the lower bound 1+Δ differ by
  a factor ≤ 27/16 + 1/(1+Δ) ≤ 43/16.

**Verdict:** VERIFIED (P1, P2, Thm P). The small wording fixes are listed above.

## 4. Theorem H (ext L.224–296)

**H1: OPT_na(engine, speed 1) ≥ Gitt₁(s).**
- Correct. A speed-1 fluid step accrues s-work at rate u + cB(1−u)/(a+cB) ≤ 1 (Z3). A discrete step accrues
  cB + αP ≤ a + cB + αP.
- The single-machine simulation sees the same states. Lags are sampled independently (independent of X, as stated).
- It survives OPT preloading prompt pieces during a tool wait. Such pieces exist only after the parent node finishes,
  and the lag-free s-machine may process the next lump any time after that.

**H2 (pathwise).**
- A waiting instant either has the server busy with a smaller-key prefill (Z′ accrual rate 1), or m smaller-key
  decoders running (rate m·c/a = 1). Correct.
- Evidence: over 15,000 checks [V: thmH_lemma_H2_pathwise_and_s_process_mutation]. These use random q, ρ, ctx ∈ {0,1,2}
  (context = parent's realized X), random X laws, lags, releases, and m ∈ {1,2,3}. Many checks are tight.
- The mutation that charges interference on the s-process fails, as the doc says.

**H3 (cumulative domination).**
- Correct: same states, same decisions, Z′ time ≤ λ·s time at every slice end; inside a lump the ratio is monotone.
- Checked: Gitt₁(Z′) ≤ λ·Gitt₁(s), with Gitt₁ equal to my exact DP for both processes [V: thmH_chain_domination_and_delta_free_lambda].

**"Δ-free" (ext L.40, L.272).**
- True. λ is a mediant of per-step ratios, so λ ≤ max_y (q_y + ρ_y + ctx_y·X̄_parent)/q_y. This is checked on every
  instance.
- λ and β_cp depend on the **support** (X_min, X̄_parent), not on CV². Nothing hides Δ.

**Chain.** E ALG ≤ (1+β_cp)LB_cp + Gitt₁(Z′) − Σ w E Z′ ≤ (1+β_cp+λ)·max(LB_cp, Gitt₁(s)) holds exactly on 35
instances. The fixed-priority mutation numbers 43.575 < 44.16 are reproduced [V: thmH_fixed_priority_mutation].

**Fix (scope, inherited).**
- (1+β_cp)·LB_cp uses base (L1), with node length αq + (a+c)X after the full lag.
- The base verifier refuted (L1) when a node's prompt is a fan-out of tool outputs returning at different times: OPT
  can prefill early pieces during the lag.
- Ext §2's scope note (L.152–153) "tool fan-out collapses to one lag" is valid for the machine model, but **not** for
  the engine lower bound.
- Either assume that a node's q tokens become usable only at its availability time, or use the preload-aware (L1),
  i.e. (a+c)X for fan-out nodes.
- With the latter, the calibrated constants become **2.54** (deterministic X) and **2.74** (random X), instead of
  2.52 and 2.67.
- H1, H2 and H3 are unaffected.

**Verdict:** VERIFIED WITH FIX (scope).

## 5. Theorem DR (ext L.298–330)

**Audit.**
- Step 1: base D1 with reservations ℓ̄⁺. The proof needs only actual KV ≤ reservation and
  M′ − ℓ̄⁺_v ≥ M′ − L̄_max = M. Correct.
- Step 2: the fixed ranking turns E Σ_{K⪯J} Z̄′_K into Σ E Z̄′_K. This is the Smith order of E Z̄′.
- Step 3: Smith is monotone and homogeneous in the sizes, so λ_E applies. P2 applies to s_K, which are independent
  across tasks because each depends only on its own X.
- Step 4: the concurrent open-shop bound holds pathwise against the clairvoyant optimum.
- No benchmark switch: OPT_clair throughout, with the same speed 3 and memory M′ = M + L̄_max for ALG as in base Thm D.

**Evidence.** [V: thmDR_pathwise_chain_random_outputs_cap_reservation]
- My own memory-engine simulator implements priority-prefix admission with head-of-line stop, kicks with
  recompute, evict-always, and cap reservations.
- 80 random 2–3-task instances: Lemma D1 holds on all 5908 pathwise checks, and 28 instances kick.
- Every kick happened at an availability event, as claimed.
- Every link of the chain holds exactly, including pair ≤ ρ(Δ_s).

**Calibration.**
- M′ = 64k, L̄_max = 4900, M = 59,100: λ_E = 2.3250, Δ_s = 0.30012, ρ = 1.63232, β_cp = 0.08508, total = **8.6755**.
- CV²(X) = 0.850 and 0.806 [V: calibrated_constants_recomputed].
- Note: the instance is a single task, for which the pair factor is actually 1. The 1.63 is the worst case for a
  workload of many such tasks, which is the right reading of a workload constant.

**Fix:** the same (L1) scope fix as Thm H. Preload-aware, the constant is 8.74.

**Verdict:** VERIFIED WITH FIX (scope). This depends on base D0/D1/(L2+L3), which the other verifier audits.

## 6. Prop K (ext L.332–351)

**Audit.**
- Z′_K ≤ λ_D(σ_K + A_K/M) per task, and Smith is monotone and homogeneous, so Smith(Z′) ≤ λ_D·S(σ + A/M) ≤ 2λ_D·OPT.
- λ_D ≤ max(1+β, θ) by the mediant over nodes.
- "Tight for one task" refers to equality in this step, not in the theorem.

**Evidence.** [V: propK_pathwise_chain_and_single_task_equality]
- 150 random instances with my simulator: Lemma D1 holds pathwise, ALG ≤ (1+β_cp)LB_cp + λ_D·S, and λ_D ≤ max(1+β, θ).
- Equality holds whenever there is one task.

**Calibrated, recomputed exactly.**

| M′ | λ_D | new constant | doc |
|---|---|---|---|
| 32k | 1.8959 | 4.8213 | 4.82 ✓ |
| 64k | 2.3847 | 5.7989 | 5.80 ✓ |
| 128k | 2.8623 | 6.7541 | 6.75 ✓ |

- Old constant: 9.9854 (doc 9.99 ✓), with β = 3.478 and θ = 1.1.
- The inherited (L1) caveat gives 5.82 at 64k.

**Verdict:** VERIFIED.

## 7. κ lower bound (ext L.361–364)

**The doc's instance.** z has q = 1, X = 9, ℓ⁺ = 10. y has q = 1, X = 1, w = 1000 and arrives at 17/2. Parameters:
M′ = 11, α = 1/100, a = 1, m = 2.

- In my simulator there is one kick, the kicked reservation is 10, and the recompute is exactly **949/100**
  (prompt 1 + 8.49 decoded tokens).
- Fractional tokens come from the fluid model.

**General family.** z1 (ℓ⁺ = L, decoding), z2 (ℓ⁺ = s, released at L − s), then y with ℓ⁺ = s + 1 and top priority,
arriving at L − 3/2. M′ = L + s.

- The head-of-line stop kicks **both**: the kicked reservation is ℓ⁺_y + L_max − 1.
- Recompute/charge → 1. For example, (L, s) = (200, 50) gives 248.998/251.
- So D0's charge ℓ⁺_y + L_max is tight up to an additive 1 even for large ℓ⁺_y, which is stronger than the doc's
  instance [V: kappa_instance_and_D0_tightness_family].

**Verdict:** VERIFIED.

## 8. Thm A lower bound (ext L.366–380)

**The √2 instance is correct.** [V: sqrt2_family_correct_but_never_reaches_sqrt2]

- Simulated ALG equals the closed form.
- "A first" is optimal for (L, k) = (3, 4) and (4, 5) by my exact DP. In any case, any feasible schedule upper-bounds
  OPT, so the ratio is a valid lower bound.
- L = 1000, k = 1414, ε = 10⁻⁶ gives 1.41371.
- Z3: sup over x of x/(1+x+x²/2) = √2 − 1, and ALG < √2·UB for **every** finite instance. So "ratio ≥ √2" means a
  supremum that is never attained.

**The literature claims are FALSE.** The doc says "improves 4/3", "the gap is now √2 vs 2" (ext L.49, L.379–380),
"Thm A tightness (√2 vs 2)" is open (L.391), and "√2 likely known" (L.405).

- Megow–Schulz, ORL 2004, Lemma 2.2: "The competitive ratio of the Algorithm P-WSPT is not better than 2 … for any
  given number of machines", building on Schulz–Skutella (J. Sched. 2002). It was read in the TU Berlin tech-report
  version 022-2003.
- The instance has m = 1 and unit weights: w_j = 1, p_j = n − j/n, r_j = jn − j(j+1)/(2n), 0 ≤ j ≤ n.
- P-WSPT preempts every job with 1/n left. The non-preemptive schedule upper-bounds OPT.
- The MV memorandum (Thm 8 and Introduction) also states that P-WSPT's 2 is tight [30].

**Exact counterexample.** [V: thmA_tight_lower_bound_two_is_known_schulz_skutella]

| n | ALG/UB(OPT) |
|---|---|
| 5 | 1.622 |
| 20 | 1.891 |
| 60 | 1.962 |
| **150** | **1.9846** |

The ratio tends to 2. Scaled n = 3: ALG = 228 and OPT = 160 (exact DP = non-preemptive schedule).

**Fix.** P-WSPT-DAG on single-node tasks with no lags *is* P-WSPT, so **Theorem A's constant 2 is tight** (m = 1,
deterministic, no lags). Replace §7 by this citation. Mark Thm A as tight in the ext table (L.49) and in [base]
(4/3 → 2). Remove §8 item 5 ("√2 vs 2").

## 9. Code vs paper, and are the author's tests discriminating?

**Code vs paper.**
- The author's `quanta` and `sim_gittins_machines` implement the running-minimum quantum priority. This matches
  ext L.79–81 and MV's F-GIPP.
- `v_matrix` uses a strict key comparison against J's last quantum, which matches L.86.
- `cumulative_lambda` and `thmH_constants` implement H3's λ and β_cp as stated.
- `lambda_D` matches Prop K.
- `dr_expected_sizes` uses a cap-based L̄_max and M, the same in every realization, as the theorem requires.
- I found no code/paper mismatch.

**Mutation testing of the author's 20 tests.** False claims were injected by monkeypatching; this is scripted in
[V: mutations_author_tests_detect_false_claims] and [V: mutations_author_lemmaP1_test_is_weak].

Detected (the test fails):
- G3 with 1/(m+1);
- G3 without the task's own work;
- Gitt₁ understated by 10%;
- pair constant forced to 1, at m = 1 and at m ≥ 2;
- λ understated by 5%;
- λ_D understated by 1%;
- the constant 1/(1+Δ) in P1.

Not detected (weak tests):
- **(a)** The P1 test passes with the false constants 0.80/(1+Δ) and 0.76/(1+Δ), which the 58/77 witness shows are
  false. The test does not certify 16/27; only the proof does. Recommendation: add the 58/77 witness as a mutation
  check.
- **(b)** The Thm H test passes with β_cp = 0, because the bound has slack there.
- **(c)** The m = 2 Thm G test passes when (Gitt₁ − ΣwEW)/2 is replaced by (Gitt₁ − ΣwEW)/4, so it does not exercise
  the 1/m factor sharply.

None of these weak spots affects a theorem: each claim is established by its proof, verified above.

## 10. Sources

| Source | Status |
|---|---|
| Megow–Vredeveld, *Approximating Preemptive Stochastic Scheduling*, METEOR RM/09/054 (2009), the precursor of MOR 39(4) 2014 | Read in full from the Maastricht repository: model, Def. of rank and quanta, Thm 1, Prop 2, Lemma 3, Lemma 4, Thm 5, Cor 6, Alg 3 F-GIPP, Thm 7, Thm 8 (tightness via [18]). The MOR 2014 PDF itself was not accessible (redirect / Cloudflare), and no arXiv version exists. |
| Megow–Schulz, *On-line scheduling to minimize average completion time revisited*, ORL 32 (2004); TU Berlin report 022-2003 | Read: Thm 2.1 (P-WSPT ≤ 2 via the speed-m relaxation) and Lemma 2.2 (lower bound 2, any m, from Schulz–Skutella). |
| Schulz–Skutella, J. Sched. 5 (2002) | Not read directly; cited through Megow–Schulz and MV. |
| Sevcik (JACM 1974), Konheim (1968), Weiss (AAP 1995), Gittins (1979/1989) | Not accessible. The single-machine optimality statement was confirmed as MV Thm 1. The chain-process generalization relies on the general Gittins job theorem, and my exact DP agrees on all tested instances. |

## 11. What could not be checked

- **Exact OPT_na for m ≥ 2.** Fractional sharing makes the exact optimum a continuous control problem. I used grid
  upper bounds (step 1/2, with extra lag information) to try to refute G1; I did not prove it numerically.
- **The engine OPT_na** (Thm H, DR, Prop K) was not computed. The lower bounds H1, (L1)–(L3) are checked by proof and
  Z3 rate algebra only.
- **Continuous size laws** for Thm G (see the fix).
- **Lemma P1** is proved for all laws. The numeric checks cover finite laws.
- **The primary Gittins, Weiss and Sevcik texts** could not be accessed.
