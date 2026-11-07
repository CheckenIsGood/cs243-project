# Stochastic theory: agent serving with random tool lags, bounded KV and batching

Companion to `docs/stochastic_plan.md` (items R0–R8). Code: `src/codesign/stochastic_theory.py`.
Machine checks: `tests/test_stochastic_theory.py`, cited below as **[T: name]** (name without the `test_` prefix).
All checks use exact rational arithmetic (`fractions.Fraction`). Z3 is used for LPs and for algebra.
Random instances use fixed seeds.

Status words:
- **proved**: full proof in this document plus machine checks;
- **proved (restricted)**: proved under the stated restriction only;
- **checked**: verified exactly on finite instances, no general proof;
- **refuted / counterexample**: an exact instance shows the statement as planned is false;
- **open**.

"New" and "known" are judged in §9. Most of the arguments here are short. Their value is the model (random
tool lags, KV memory, batching) and the explicit hypotheses, not new proof techniques.

## Verification status

An independent verifier audited commit e17bd7f (`docs/stochastic_theory_verification.md`, code
`src/codesign/stochastic_verify.py`). A second verifier audited the extensions
(`docs/stochastic_theory_ext_verification.md`) and found that Thm A's tightness is already known. This revision
applies their fixes. "Fixed" means the text, code and tests below were changed.

| Result | Verifier's verdict | Status in this revision |
|---|---|---|
| Lemma A1, Thm A (2 vs OPT_clair, random lags) | VERIFIED | unchanged; Smith-term test tightened (§2.3) |
| Thm A lower bound (lag trap 4/3; "tightness open") | lag trap VERIFIED; "tightness open" FALSE (ext verifier) | **fixed**: 2 is **tight**, by the known Schulz–Skutella / Megow–Schulz family (§2.4); the lag trap is kept as a remark |
| MSU inequality, LB2, Thm B, Thm B′ | VERIFIED | unchanged; tests tightened (γ is load-bearing, LB2 sharpness) (§3) |
| Lemma D0 | VERIFIED (two text fixes) | **fixed** (§4.3): the case where y does not fit; simultaneous arrivals. Tightness family added |
| Lemma D1 | VERIFIED (two text fixes) | **fixed** (§4.4): case (B) uses ℓ⁺_u and "at least m" |
| (L1) critical-path lower bound | GAP for nodes with fan-out pieces | **fixed** (§4.1): (L1) stated for the restricted benchmark; preload-aware LB⁰ for the unrestricted one |
| Thm D final constant | VERIFIED WITH FIX | **fixed** (§4.5): restricted benchmark keeps (1+β_cp) + 2max(1+β,θ); unrestricted benchmark gets min{1+β_cp⁰, 2+β} + 2max(1+β,θ) |
| Thm D assumption wording | VERIFIED WITH FIX | **fixed** (§4, §10.6): token counts known **at release** |
| (L2), (L3), (L2+L3), R0a, R1 | VERIFIED | unchanged (R0a: the time-sharing remark is added in §4.5) |
| R0b | VERIFIED (proof is a sketch) | **relabelled** "proof sketch" (§5) |
| R0c | VERIFIED WITH FIX (instance not an m_eff instance) | **fixed** (§5): m = 1 vs m = 8 stated as such, and a consistent m_eff instance added |
| R5a, R5b, R5c | VERIFIED | unchanged; R5a test tightened |
| R6 | safety VERIFIED; "exact saving" false for DAGs | **fixed** (§6.4): exact for chains only. For DAGs I go further than the verifier: the saving can be smaller **or larger** than α·(held hits), and even negative |
| R7 | VERIFIED | item 6 updated to the tight bound 2 |
| §9 citations | one inaccurate ("later 2+2ln2+ε" for delays) | **fixed**: dropped |

---

## 0. Results at a glance

Notation:
- OPT_na = the optimal **non-anticipatory** policy at speed 1 with memory M (the benchmark). The clairvoyant
  optimum OPT_clair is never the benchmark; it appears only as a lower bound (OPT_clair ≤ OPT_na).
- OPT_np = optimal non-anticipatory **non-preemptive** policy.
- Δ = max squared coefficient of variation of the node sizes.
- "Fluid" and "discrete" refer to the engine models of §1.3.

**R2 (offline core) and R4 (online)**

| Item | Statement (short) | Status | Restriction | Constant | Machine check |
|---|---|---|---|---|---|
| **R2/R4 – Thm A** | Online preemptive task-index policy P-WSPT-DAG. Pathwise C_J ≤ r_J + CP_J + (1/m)Σ_{K⪯J} W_K. Hence E[ALG] ≤ 2·OPT_clair ≤ 2·OPT_na. | **proved**; 2 is **tight** (known) | deterministic node sizes; arbitrary random (even correlated) lags; general DAGs; release dates; online; m machines; speed 1; no memory limit | 2, tight: the Schulz–Skutella family (m = 1, unit weights, no lags) gives 1.9846 at n = 150; worst random instance 1.128 | [T: thmA_pathwise_bound_every_realization], [T: thmA_pathwise_bound_mutations_fail], [T: thmA_two_approx_vs_exact_clairvoyant_m1], [T: thmA_parallel_machines_vs_lower_bounds], [T: thmA_constant_two_is_tight_schulz_skutella], [T: thmA_lag_trap_lower_bound_four_thirds], [T: thmA_clairvoyant_below_nonanticipatory_dp], [T: fanout_collapse_to_max_lag] |
| **R2/R4 – Thm B (online)** | Same policy with index w/E[W]. E[ALG] ≤ ρ·OPT_np. | **proved (restricted)** | random independent node sizes, CV² ≤ Δ; lags independent of sizes; benchmark **non-preemptive**; ALG preemptive | chains: 2 + max(1,(m−1)(Δ−1))/(2m); DAGs: 2 + max(1,(m−1)(Δ−1))/2 (worst checked 1.098) | [T: thmB_online_index_policy_vs_nonpreemptive_dp], [T: thmB_lb2_greedy_equals_task_lp], [T: thmB_constant_algebra_z3] |
| **R2 – Thm B′ (offline LP)** | MSU LP plus lag-precedence and critical-path constraints; list policy in LP order. | **proved (restricted)** | as Thm B; offline | 2 + max(1,(m−1)Δ/m), = 3 for Δ ≤ m/(m−1) (worst checked 1.153 vs OPT_np) | [T: thmB_offline_lp_policy], [T: msu_variance_correction_is_necessary], [T: nonpreemptive_benchmark_is_necessary] |

**R3 (engine guarantee)**

| Item | Statement (short) | Status | Restriction | Constant | Machine check |
|---|---|---|---|---|---|
| **R3 – Thm D** | Hybrid/fluid engine, KV memory, priority-prefix admission with kicks, evict-always. Against OPT_na restricted to engines that do not prefill a node before it is available: E[ALG] ≤ [(1+β_cp) + 2·max(1+β, θ)]·OPT_na. Against the unrestricted OPT_na (may prefill returned fan-out pieces early): E[ALG] ≤ [min(1+β_cp⁰, 2+β) + 2·max(1+β, θ)]·OPT_na. | **proved (restricted)** | deterministic token counts, **known at release**; random lags; DAGs; release dates; online; fluid engine at **speed 3**; memory **M′ = M + L_max ≤ 2M** | restricted: 3 when β = 0 and θ = 1; 2 with unbounded memory and no recompute; 9.99 calibrated with memory. Unrestricted: 10.01 calibrated (§4.6). Worst checked ALG/LB 1.45 | [T: thmD_pathwise_bound_any_keep_policy_and_preload], [T: thmD_ratio_against_engine_lower_bounds], [T: thmD_hypotheses_are_necessary], [T: COUNTEREXAMPLE_L1_needs_restricted_benchmark], [T: thmD_unrestricted_benchmark_corrected_bound], [T: lemmaD0_charge_tight_family] |

**R0 (engine reductions) and R1 (recompute)**

| Item | Statement (short) | Status | Restriction | Constant | Machine check |
|---|---|---|---|---|---|
| **R0a** | Fluid engine at speed 3 (prefill share 1/3) dominates the hybrid model, and no s < 3 does. Conversely, the speed-1 engine fits inside the hybrid rates. | **proved** | fluid; batch ≤ m = a/c | speed 3, tight | [T: R0a_engine_lift_and_converse_z3] |
| **R0b** | Discrete steps at speed 3, chunk P̄ = a/α: C_J ≤ (Thm D bound) + a(\|J\| + N_J). | **proof sketch** (additive) plus exact checks; the multiplicative form is open | evict-always; N_J = number of prefill completions | additive a per node and per prefill completion | [T: R0b_discrete_engine_bound] |
| **R0c** | Crude cap m_eff = min(a/c, ⌊M′/L_max⌋) is memory-safe but loses Θ(m/m_eff). | **proved**, with lower-bound instances | — | m = 1 vs m = 8 (unbounded memory): 1286/287 ≈ (n+1)/2; consistent m_eff = 1 instance (L_max = 50, M′ = 99): > 4 | [T: R0c_machine_cap_loses_m_over_meff] |
| **R1** | Evict-always is feasible with active-sequence memory only; the cost is the factors (1+β_cp), (1+β). | **proved** (inside Thm D) | as Thm D | β_cp = max αρ/(αq+aX), β = max(αρ+κ)/(αq+cX) | as Thm D |

**R5 (keep/evict) and R6 (preloading)**

| Item | Statement (short) | Status | Restriction | Constant | Machine check |
|---|---|---|---|---|---|
| **R5a** | Single prefix: the best deterministic TTL is the optimal non-anticipatory keep rule. Ratio vs clairvoyant ≤ e/(e−1), nearly tight. Exponential waits: hold iff h·E[D] < R. | **proved** (known type) | known wait law | e/(e−1) ≈ 1.582; Yao LP 1.553 on a 16-point grid | [T: R5a_best_ttl_is_optimal_and_within_e_over_e_minus_1], [T: R5a_exponential_wait_rule_z3], [T: R5a_randomized_ski_rental_density_numeric] |
| **R5b** | Many tenants: LP over (issue, TTL) with memory dual η. Each class becomes a ski rental. At η = 0 the issue rule is BE′ (σ > σ\* = ρ_s/(1−u+ρ_s)). | **proved** (for the linearized-price LP) | first-order (linearized) latency prices | — | [T: R5b_keep_lp_strong_duality_and_sigma_star] |
| **R5c** | Held KV in the lowest eviction class preserves Thm D for any keep policy. Per-task dominance is false. | **proved** + counterexample | priority order computed from evict-always sizes | — | [T: thmD_pathwise_bound_any_keep_policy_and_preload], [T: R5c_keep_policies_never_increase_recompute_bound_but_per_task_dominance_fails] |
| **R6** | Lowest-priority preload chunks preserve Thm D. For an isolated **chain** task the saving is α·(held prefix tokens used), exactly. For DAG tasks there is no such identity: the saving can be smaller or larger than that, or negative. | **proved** (safety; chain identity) + counterexamples (DAGs) | as Thm D | — | [T: thmD_pathwise_bound_any_keep_policy_and_preload], [T: R6_isolated_preload_saving_is_exact], [T: COUNTEREXAMPLE_R6_not_exact_for_DAGs_either_direction] |

**R7 (lower bounds) and R8 (engine checklist)**

| Item | Statement (short) | Status | Restriction | Constant | Machine check |
|---|---|---|---|---|---|
| **R7** | (1) clairvoyant comparison hopeless (Thm 2/3, existing); (2) hold-only is unbounded; (3) adversarial arrivals (Jaillet et al., known); (4) Prop I (existing); (5) fixed-priority policies have worst-case ratio ≥ 1+Δ vs preemptive OPT (a limit over instances); Ω(Δ^{1/4}) vs non-preemptive OPT (EFMM, known); (6) P-WSPT-DAG's constant 2 is tight (Schulz–Skutella, known). | **proved / known** | — | — | [T: R7_hold_only_is_unbounded], [T: R7_index_policies_lose_one_plus_delta_against_preemption], [T: thmA_constant_two_is_tight_schulz_skutella] |
| **R8** | Engine checklist P1–P5, with the property each proof uses (§8). P3 is corrected: priority-prefix admission **with** a head-of-line stop and kicks. | **stated** | — | — | — |

**Strongest statements (verbatim from the theorems below).**

- **Theorem A.** In the machine model M_m with deterministic node sizes, arbitrary random arc lags (any joint law,
  independent of the policy), DAG tasks, release dates and online arrival, the preemptive task-index policy
  P-WSPT-DAG satisfies, on every realization, C_J ≤ r_J + CP_J + (1/m)·Σ_{K ⪯ J} W_K. Consequently
  E[Σ w_J C_J] ≤ LB_cp + (1/m)·Smith(W) ≤ 2·E[OPT_clair] ≤ 2·OPT_na. This holds at speed 1 with no augmentation.
  The constant 2 is tight even for m = 1, unit weights and no lags (known; §2.4).
- **Theorem D.** In the fluid engine with step time τ = a + cB + αP, run at speed 3, with memory M′ = M + L_max
  for admitted reservations: policy P (§4.2) with evict-always, any keep policy for held KV in the lowest eviction
  class, and lowest-priority preloads satisfies
  E[Σ w_J C_J] ≤ [(1+β_cp) + 2·max(1+β, θ)]·OPT_na(engine, speed 1, memory M),
  where OPT_na is restricted to engines that do not prefill a node before it is available. Against the
  unrestricted OPT_na, which may prefill already-returned fan-out pieces during a tool wait, the constant is
  min(1+β_cp⁰, 2+β) + 2·max(1+β, θ) (§4.5).
  The hypotheses are deterministic token counts known at release and random tool lags independent of the schedule.

---

## 1. Models

### 1.1 Tasks, nodes, lags

Task J has:
- a release time r_J and a weight w_J > 0;
- a DAG G_J of **LLM nodes**;
- arcs u → v that carry a random **lag** D_uv ≥ 0.

A node becomes **available** when every predecessor u has completed and its lag D_uv has elapsed (and not before
r_J). J completes when all its nodes have completed. The objective is E[Σ_J w_J C_J].

**Tools are lags.** Tools have unlimited parallelism and are not sped up by GPU augmentation. In the machine model
M_m, a tool fan-out u → {t_1..t_k} → v therefore collapses to a single arc lag, max_i D_{t_i}
[T: fanout_collapse_to_max_lag]. This collapse is **not** valid for the engine's lower bound. There, the fan-out's
outputs are prompt tokens of v that return at different times (`pieces`), and a schedule may prefill the pieces
that have returned before v is available (§4.1). Tool chains give sums of lags along each path, and a DAG of tools gives the
max-plus path length. Lags may be correlated with each other. In Thms A and D they only need to be independent of
the policy. In Thm B they must also be independent of node sizes.

**Information.** The policy knows the laws, the DAG of each released task and all realized values so far. It does
not know the future (non-anticipatory). Tasks are revealed at r_J (online).

### 1.2 Machine model M_m (Theorems A, B)

- m identical machines, each with speed 1.
- Node x has processing time p_x. It is deterministic in Thm A and random in Thm B.
- Preemption and migration are allowed **for the policy**.
- W_J = Σ_{x∈J} p_x.
- CP_J = the longest path in G_J, counting node processing times plus arc lags (r_J-relative).

### 1.3 Engine models (Theorem D, R0)

Engine (from the plan):
- A **discrete** step decodes one token for each of the B running sequences and prefills P tokens.
- Step time is τ(B, P) = a + cB + αP.
- The **fluid** engine at speed s shares rates continuously. Given a prefill share u ∈ [0,1]:
  - prefill runs at s·u/α tokens/s;
  - each of B decoders runs at s(1−u)/(a+cB) tokens/s.

**Hybrid model H** (the analysis model):
- a prefill **server** at 1/α tokens/s, one node at a time, preemptive; a paused partial prefill keeps its KV;
- m = a/c **decode machines**, each at 1/a tokens/s;
- a memory budget M′.

Node y has:
- ρ_y = parent-context tokens (recomputed under evict-always unless held);
- q_y = new prompt tokens;
- X_y ≥ 1 decode tokens.

Its reservation is ℓ⁺_y = ρ_y + q_y + X_y and its prompt footprint is ℓ⁻_y = ρ_y + q_y. Write
L_max = max ℓ⁺ and M = M′ − L_max.

The hybrid model is the analysis vehicle. The theorem is stated for the engine via R0a: the hybrid model at
speed 1 is dominated by the fluid engine at speed 3, and every engine at speed 1 lies inside the hybrid rates.

### 1.4 Benchmarks, preemption and augmentation, stated once

- **Thm A:** benchmark OPT_na (any non-anticipatory policy, preemptive or not, m machines, speed 1). The proof
  compares against the stronger OPT_clair. No augmentation.
- **Thm B, B′:** benchmark OPT_np (non-anticipatory, **non-preemptive**, m machines, speed 1). The policy itself
  preempts. This restriction is necessary for the LP used: [T: nonpreemptive_benchmark_is_necessary] shows the
  MSU bound exceeds the preemptive optimum. No augmentation.
- **Thm D:** benchmark OPT_na in the engine at speed 1 with memory M (preemptive, any keep policy, may keep
  parent KV). Two versions:
  - *restricted*: OPT_na does not prefill a node's prompt before the node is available;
  - *unrestricted*: OPT_na may also prefill returned fan-out pieces during the remaining wait.

  The policy uses speed 3 (fluid engine) and memory M′ = M + L_max ≤ 2M (μ ≤ 2 when L_max ≤ M). Tools are not
  sped up.

---

## 2. Theorem A: deterministic sizes, random lags (R2/R4)

### 2.1 Policy P-WSPT-DAG (online)

When J is released, compute its index w_J / W_J and keep it fixed. Rank tasks by decreasing index, breaking ties by
id; K ⪯ J means K is ranked at or above J. A node's priority is (rank of its task, node index in topological
order). At every instant the policy runs the m highest-priority **available** nodes, with preemption and
migration.

The policy is online: a later arrival may get a higher rank. The policy never anticipates lags or future tasks.

### 2.2 Pathwise lemma

**Lemma A1.** On every realization, for every task J: C_J ≤ r_J + CP_J + (1/m)·Σ_{K⪯J} W_K.

*Proof.*
1. Build a path backwards. Let v_k be J's last-completing node. Its availability time a_{v_k} is either r_J or
   C_u + D_{u v_k} for some predecessor u. Choose the u that attains the maximum and recurse until a node v_1 with
   a_{v_1} = r_J. Then [r_J, C_J] splits into lag intervals [C_{v_{i−1}}, a_{v_i}] on path arcs and node intervals
   [a_{v_i}, C_{v_i}].
2. Inside [a_v, C_v], node v is available. At each instant either v runs, or all m machines run nodes of higher
   priority. Those nodes belong to tasks K ⪯ J (J's own nodes included).
3. The time v runs is at most p_v. The total time when all m machines are busy with nodes of tasks K ⪯ J is at most
   (1/m)·Σ_{K⪯J} W_K, over the whole union of node intervals.
4. The path's lags plus its node sizes are at most CP_J. ∎

The lemma uses nothing about how the order was chosen or about the size laws, so it also holds for random sizes
(Thm B). [T: thmA_pathwise_bound_every_realization] checks it on more than 1000 (task, realization) pairs with
DAGs, m ∈ {1,2,3}, and random sizes and lags.

Non-vacuity [T: thmA_pathwise_bound_mutations_fail]:
- dropping J's own work fails, because two parallel nodes on one machine give C = 2 > CP = 1;
- replacing 1/m by 1/(m+1) fails.

### 2.3 Lower bounds and the ratio

**LB_cp.** Every policy, clairvoyant or not, has C_J ≥ r_J + CP_J pathwise. So
LB_cp = Σ w_J(r_J + E[CP_J]) ≤ E[OPT_clair].

**Smith.** For deterministic sizes, simulate any m-machine schedule on one machine of speed m and drop the release
dates. The single-machine preemptive optimum of Σ w C is then Smith's rule, so (1/m)·Smith(W) ≤ OPT_clair for every
realization.

**Theorem A.** E[ALG] ≤ LB_cp + (1/m)·Smith(W) ≤ 2·E[OPT_clair] ≤ 2·OPT_na.

*Proof.* Take expectations in Lemma A1, using deterministic W. The ranking is the Smith order of W, so
Σ_J w_J (1/m)Σ_{K⪯J} W_K = (1/m)·Smith(W) exactly. ∎

**Checks.**
- [T: thmA_two_approx_vs_exact_clairvoyant_m1]: 50 random DAG/chain instances, m = 1, with an exact clairvoyant
  DP. All of E[ALG] ≤ LB_cp + Smith, LB_cp ≤ OPT and Smith ≤ OPT hold. Worst ALG/OPT_clair = 1.128; worst
  (LB_cp + Smith)/OPT = 1.842.
- [T: thmA_parallel_machines_vs_lower_bounds]: m = 2, 3.
- Both tests include 20 unit tasks released at 0, where the bound is attained (m = 1: ALG = 210 = LB_cp + Smith −
  ΣwW) or nearly attained (m = 2: ALG = 110 ≤ 125). A Smith term understated by 20% fails there.
- [T: thmA_clairvoyant_below_nonanticipatory_dp]: the benchmark chain LB ≤ OPT_clair ≤ OPT_na on exact DPs. The
  gap is strict on some instances, so information about lags has value.

### 2.4 Lower bounds for the policy: 2 is tight (known); the lag trap

**The constant 2 is tight (known).** On single-node tasks without lags, P-WSPT-DAG is exactly P-WSPT
(preemptive WSPT with releases). Megow–Schulz (ORL 2004, Lemma 2.2), building on Schulz–Skutella (J. Sched.
2002), show that P-WSPT's competitive ratio is not better than 2, for any number of machines. Their family has one
machine and unit weights: w_j = 1, p_j = n − j/n, r_j = jn − j(j+1)/(2n), for j = 0..n. P-WSPT preempts each job
when 1/n of it remains. The non-preemptive schedule in release order upper-bounds OPT.

[T: thmA_constant_two_is_tight_schulz_skutella] reproduces it exactly:
- scaled by 6 at n = 3: ALG = 228 and OPT = 160 (exact preemptive DP, equal to the release-order schedule);
- the ratio ALG/UB is 1.622, 1.891 and 1.9846 at n = 5, 20 and 150.

So Theorem A's constant cannot be improved for this policy, even against OPT_na, with m = 1, deterministic
sizes and no lags. This tightness result is not ours. The earlier version of this document called tightness open
and gave 4/3 as the best lower bound; that was wrong (found by the extension verifier).

**Remark: the lag trap** (m = 1, deterministic). It is kept because it shows a lag-specific failure mode, not
because it is a strong lower bound.
- J: a node of size ε, then a lag L, then a node of size L; weight 1.
- K: one node of size L; weight (L+1)/(L+ε), so its index is just above J's.

The policy runs K first, so J's lag starts late. The optimum runs J's tiny first node, then hides K inside J's lag.
For ε = 1: ALG = 4L+1 and OPT = 3L+2 exactly, so the ratio → 4/3
[T: thmA_lag_trap_lower_bound_four_thirds].

Index policies based on total work are blind to "start the cheap node that unlocks a long lag". A
critical-path-aware index (e.g., an LP order with the C_J ≥ r_J + E[CP_J] constraint) avoids this trap. Random
search found nothing above 1.18, far below the known family's 2.

---

## 3. Theorem B: random node sizes (R2/R4)

**Assumptions.**
- Node sizes p_x are independent, with E[p_x] > 0 and Var[p_x]/E[p_x]² ≤ Δ.
- They are independent of the lags.
- The benchmark is OPT_np. The policy is P-WSPT-DAG with index w_J/E[W_J] (WSEPT on tasks).

### 3.1 The MSU load inequality with tasks and lags

For every non-anticipatory non-preemptive policy on m machines (idling allowed, any release dates, precedence
and lags) and every node set A:

  Σ_{x∈A} E[p_x]·E[C_x] ≥ f(A) := (1/(2m))[(Σ_A E p)² + Σ_A (E p)²] − ((m−1)(Δ−1)/(2m))·Σ_A (E p)².

This is Möhring–Schulz–Uetz (JACM 1999, Cor. 3.1), as restated in Skutella–Uetz 2005 eq. (3.1). The proof only
needs the start time S_x to be independent of p_x. That holds because a non-anticipatory policy decides S_x from
information independent of p_x, and lags are independent of sizes. Lags and release dates only restrict the
policy class further, so validity is preserved. We re-derived the inequality and did not rely on it blindly.
[T: msu_variance_correction_is_necessary] shows the −(m−1)(Δ−1)/(2m) term cannot be dropped: four i.i.d. jobs
{1 w.p. 9/10, 30 w.p. 1/10} on m = 2 give an "LP" value of 19.5 > OPT_np = 18.864.

**Task-level bound LB2.**
1. Take a set 𝒜 of tasks and let A be the union of their nodes. Since C_J ≥ C_x for x ∈ J, we get
   Σ_{J∈𝒜} E[W_J] E[C_J] ≥ f(A) = (1/(2m))(Σ_{𝒜} E W)² + b·Q, where b = ½ − (m−1)Δ/(2m) and
   Q = Σ_{x∈A}(E p_x)².
2. If b ≥ 0, drop bQ. If b < 0, use Q ≤ Σ_{J∈𝒜}(E W_J)².
3. So with γ = min(0, b), the right-hand side is at least g(𝒜) = (1/(2m))(Σ E W)² + γ·Σ (E W)².
4. g is supermodular, so min Σ w_J C_J subject to these constraints is attained at the greedy vertex in Smith order
   (Edmonds). That gives LB2 = Σ_J w_J[(1/m)·Σ_{K≺J} E W_K + (1/(2m) + γ)·E W_J] ≤ OPT_np.

[T: thmB_lb2_greedy_equals_task_lp] checks that the greedy value equals the LP over all 2^n constraints (Z3).

### 3.2 Theorem B (online)

**Theorem B.** E[ALG] ≤ LB_cp + LB2 + (1/(2m) − γ)·Σ_J w_J E[W_J], and 1/(2m) − γ = max(1, (m−1)(Δ−1))/(2m).
Hence:
- for chain tasks: E[ALG] ≤ (2 + max(1,(m−1)(Δ−1))/(2m))·OPT_np;
- for general DAG tasks: E[ALG] ≤ (2 + max(1,(m−1)(Δ−1))/2)·OPT_np.

*Proof.*
1. Take expectations in Lemma A1. Then E[Σ w_J (1/m)Σ_{K⪯J} W_K] = LB2 + (1/m − 1/(2m) − γ)·Σ w_J E W_J.
2. Chains: E[W_J] ≤ E[CP_J], so Σ w E W ≤ LB_cp ≤ OPT_np.
3. DAGs: every schedule has C_J ≥ r_J + W_J/m, so Σ w E W ≤ m·OPT_np.
4. The identity for 1/(2m) − γ is [T: thmB_constant_algebra_z3]. ∎

Values:
- Δ ≤ 1 and m = 1: 2.5;
- Δ ≤ 1 + 1/(m−1): 2 + 1/(2m) (chains) and 2.5 (DAGs).

[T: thmB_online_index_policy_vs_nonpreemptive_dp] checks the inequality and the ratio against the exact
non-preemptive DP on 150 instances (m ∈ {1,2}, chains and fork-join DAGs, random sizes and lags). Worst
ALG/OPT_np = 1.068 at m = 1 and 1.098 at m = 2.

The same test checks that LB2's constants are sharp enough to matter (the verifier found that the random
instances alone did not stress them):
- *γ is needed.* Eight i.i.d. tasks {1 w.p. 9/10, 40 w.p. 1/10} on m = 2 have γ < 0. LB2 is below the expected
  cost of a non-preemptive list schedule (69.97 ≥ OPT_np). LB2 with γ := 0 is 392/5 = 78.4, above it.
- *No 10% slack.* On 11 unit jobs with m = 1, LB2 = 121/2 and OPT_np = 66, so 1.1·LB2 > OPT_np.

### 3.3 Theorem B′ (offline LP order)

LP:
- the MSU inequalities over all node sets;
- C_v ≥ C_u + E[D_uv] + E[p_v] (lag precedence), valid by linearity for each realization;
- C_x ≥ r_J + E[p_x] and C_J ≥ C_x;
- C_J ≥ r_J + E[CP_J], valid pointwise.

The policy is P-WSPT-DAG with tasks ranked by increasing LP value C_J^LP.

**Theorem B′.** E[ALG] ≤ (2 + max(1,(m−1)Δ/m))·LP ≤ (2 + max(1,(m−1)Δ/m))·OPT_np.

*Proof.*
1. Lemma A1 in expectation gives E C_J ≤ (r_J + E CP_J) + (1/m)·μ(A_J), where A_J = nodes of tasks K ⪯ J and
   μ = Σ E p. The first term is ≤ C_J^LP.
2. For x ∈ A_J, E p_x ≤ C_x^LP ≤ C_J^LP, so Q ≤ C_J^LP·μ(A_J).
3. The load inequality gives C_J^LP μ ≥ μ²/(2m) + bQ.
4. Then the Skutella–Uetz Lemma 3.3 form, μ/m ≤ (1 + max(1,(m−1)Δ/m))·C_J^LP, follows. It is Z3-proved in
   [T: thmB_constant_algebra_z3], which also checks that no constant below 2 works. ∎

[T: thmB_offline_lp_policy] checks LP ≤ OPT_np and the ratio on 60 instances. Worst ALG/OPT_np = 1.153;
worst ALG/LP = 1.244.

Remarks:
- B′ is offline. B′ beats B for DAGs when m ≥ 3 and Δ is large. Otherwise the online bound B is better.
- Both rely on the policy being preemptive. That is why no idle-time charging is needed, and why our constants
  are smaller than Skutella–Uetz's non-preemptive 3 + 2√2. This is **not** an improvement on their problem (§9).

---

## 4. Theorem D: the engine with KV memory (R3, R1, R5c, R6)

Assumptions:
- Token counts (q, ρ, X) of **every node of J are deterministic and known when J is released**. The index Z′_J
  is computed at release, so knowing them only at node availability is not enough (verifier's fix). A known
  upper bound X̄ also works if X̄ is reserved and used everywhere in place of X. Random X is handled in the
  extension (Thm H, Thm DR).
- Lags are random with any law independent of the schedule.
- Prompt tokens arrive as follows. A node without pieces receives its q tokens when it becomes available (they
  are the tool's output). A node with pieces receives piece i's tokens when piece i returns, and becomes
  available when the last piece returns. Prompt tokens are appended after the parent context, so no token of v
  can be prefilled before v's parent completes.

### 4.1 Lower bounds on OPT_na (engine, speed 1, memory M)

These are valid for every realization of the lags, so also in expectation.

- **(L1) Critical path, restricted benchmark.**
  - If the benchmark does not prefill a node before it is available, node y needs at least αq_y + (a+c)X_y
    seconds after its availability: OPT may keep the parent KV, and each decode step takes at least a + c.
  - Define CP^opt_J with these node lengths. Then LB_cp = Σ w_J(r_J + E CP^opt_J) ≤ OPT_na(restricted).
  - **(L1) is false for the unrestricted benchmark** when a node has fan-out pieces that return at different
    times (verifier's counterexample, [T: COUNTEREXAMPLE_L1_needs_restricted_benchmark]). One task: plan
    (q = 1, X = 1) → final with pieces {10 tokens, lag 0} and {1 token, lag 10}; α = a = c = 1. A speed-1 schedule
    that prefills the 10-token piece during the wait finishes at 16, but LB_cp = 26.
- **(L1⁰) Critical path, unrestricted benchmark.**
  - Let q⁰_y = q_y if y has no pieces and q⁰_y = 0 otherwise. Node y needs at least αq⁰_y + (a+c)X_y seconds after
    its availability. Decoding starts only after the full prompt is present. Without pieces the prompt arrives at
    availability.
  - Define CP⁰_J with these node lengths and LB⁰ = Σ w_J(r_J + E CP⁰_J) ≤ OPT_na (any engine schedule). On the
    counterexample LB⁰ = 15 ≤ 16. If some of a piece-free node's q tokens could arrive early, set q⁰ to the part
    that arrives at availability; q⁰ ≡ 0 (the verifier's choice) is always valid.
- **(L0) Own GPU work.** J's GPU work σ_J = Σ_{y∈J}(αq_y + cX_y) lies in [r_J, C_J] in any schedule, so
  Σ w_J σ_J ≤ OPT_na.
- **(L2) GPU time.**
  - Summing step times, time ≥ c·(decode tokens) + α·(prefill tokens). With σ_y = αq_y + cX_y, OPT dominates a
    single machine with sizes σ.
- **(L3) Memory area.**
  - During its X_y decode steps, node y holds at least ℓ⁻_y tokens for at least (a+c)X_y seconds.
  - With area A_y = ℓ⁻_y(a+c)X_y, total area by time t is at most M·t.
- **(L2+L3) Concurrent open shop.**
  - Order tasks by OPT's completion. By time C_J, OPT has done all σ and A/M of the tasks before J, so
    Σ_{K before J}(σ_K + A_K/M) ≤ 2C_J.
  - Hence S(σ + A/M) := Smith(σ + A/M) ≤ 2·OPT.
  - The area-order lower bound is known (Feng et al. 2607.09248, Lemma 1). Combining it with the GPU-time
    bound in this way is standard.

### 4.2 Policy P (online, fluid hybrid)

- **Index.** Each task gets the index w_J / Z′_J, computed at release, where Z′_J = Σ_{y∈J} Z′_y and
  Z′_y = α(q_y + ρ_y) + κ_y + cX_y + aX_y ℓ⁺_y/M, with κ_y = α(ℓ⁺_y + L_max). Without a memory limit, κ = 0 and the
  last term is absent.
- **Admission.** The admitted set is the **longest priority prefix** of the available nodes whose reservations
  fit in M′ (head-of-line stop).
- **Kicks.** If a newly available node enters the prefix ahead of admitted nodes, the tail that no longer fits is
  **kicked**: its KV is lost and it later recomputes prompt plus decoded tokens as prefill. Kicks happen only at
  availability events.
- **Server.** Prefill runs for the highest-priority admitted node in prefill phase.
- **Decoders.** The top m admitted decode-phase nodes decode.
- **Evict-always** is the baseline: a finished node's KV is freed.
- **Held KV** (a child's prefix: parent context and preloaded pieces) uses only memory not reserved by admitted
  nodes. It is evicted on demand (the lowest class), in any order (LRU, FIFO, random).
- **Preload chunks** (R6) run on the server only when no admitted node is prefilling. They use only unreserved
  memory.

### 4.3 Kick accounting

**Lemma D0.** When node y becomes available, the total reservation of the nodes kicked at that instant is less
than ℓ⁺_y + L_max. If several nodes y_1..y_k become available at the same instant, it is less than
Σ_i ℓ⁺_{y_i} + L_max.

*Proof.* Let S_above be the total reservation of the old admitted nodes ranked above y. The old prefix had total
S_old ≤ M′.
1. *y fits.* The new prefix contains y and stops at the first node that does not fit, of size
   ℓ_next ≤ L_max, so S_new + ℓ_next > M′. The kicked tail totals S_old − (S_new − ℓ⁺_y) < ℓ_next + ℓ⁺_y. (If
   nothing stops the new prefix, nothing is kicked.)
2. *y does not fit* (S_above + ℓ⁺_y > M′). The head-of-line stop is at y. Everything below y is kicked and y is
   not admitted. The kicked total is S_old − S_above ≤ M′ − S_above < ℓ⁺_y.
3. *Simultaneous arrivals.* Apply the same argument with the arrivals' total Σ_i ℓ⁺_{y_i} in place of ℓ⁺_y.
   Charging κ_{y_i} = α(ℓ⁺_{y_i} + L_max) to each arrival covers this, since k·L_max ≥ L_max.
4. Completions only extend the prefix, so no other event kicks. ∎

The recompute charged to each arrival y is therefore at most κ_y = α(ℓ⁺_y + L_max).

**Tightness** (verifier's family, [T: lemmaD0_charge_tight_family]). z1 (ℓ⁺ = L) is decoding. z2 (ℓ⁺ = s) is
released at L − s. A top-priority y with ℓ⁺_y = s + 1 arrives at L − 3/2, and M′ = L + s. The head-of-line stop
kicks both z1 and z2. The kicked reservation is exactly ℓ⁺_y + L_max − 1, and the recompute exceeds
ℓ⁺_y + L_max − 3; for (L, s) = (10, 3), (50, 20), (200, 50). So the charge is tight up to an additive 1, for
every ℓ⁺_y.

### 4.4 Pathwise bound

**Lemma D1.** On every realization, and for every keep policy and preload schedule as in §4.2:
C_J ≤ r_J + CP′_J + Σ_{K⪯J} Z′_K. Here CP′_J is the critical path with node lengths α(q+ρ) + aX.

*Proof.* Build the backward path as in Lemma A1. Within [a_v, C_v], each instant is one of four kinds.
- **(P) v progresses.** v prefills (total α(q_v+ρ_v), plus recompute after kicks, which is charged to the kicker)
  or decodes (total aX_v).
- **(C) v is admitted, in prefill phase, and the server is busy with a higher-priority prefill** (including
  recompute). The total over all such instants is ≤ Σ_{K⪯J} Σ_y [α(q_y+ρ_y) + κ_y], by Lemma D0.
- **(A) v is admitted, in decode phase, and m higher-priority decoders run.** They decode at total rate m/a, so
  the total is ≤ Σ_{K⪯J} Σ_y (a/m)X_y = Σ cX_y.
- **(B) v is not admitted.** Then the prefix stops at some node u ⪰ v (the first one that does not fit), so the
  admitted nodes, all above v, reserve more than M′ − ℓ⁺_u ≥ M. Three cases:
  - if one of them is prefilling, the instant is of type (C);
  - if at least m are decoding, it is of type (A);
  - otherwise all of them run, holding reservations ≥ M. Each running decoder y accrues area ℓ⁺_y over at most
    aX_y seconds, and kicked decoders resume decoding where they stopped. So the total of these instants is
    ≤ Σ_{K⪯J} Σ_y aX_y ℓ⁺_y / M.

Held KV and preload chunks never reduce memory available to admission and never occupy the server while an
admitted node prefills. A held hit only removes prefill work. Summing the four kinds proves the bound. ∎

Machine check: [T: thmD_pathwise_bound_any_keep_policy_and_preload].
- Setting: 200 instances; 4 configurations (evict-always, LRU, random eviction plus preload, LRU plus preload);
  m ∈ {1,2,3}; memory bounded or unbounded.
- Coverage: more than 1500 (task, realization) checks, with more than 10 kicking runs and more than 50 runs with
  held hits.
- The simulator asserts the memory invariant at every event.

**Necessity of each ingredient** [T: thmD_hypotheses_are_necessary], on exact instances:
1. Without kicks (non-preemptive memory), a high-priority arrival waits behind a long low-priority decode:
   41.2 > the bound 8.25.
2. Without the κ charge, five tiny high-priority arrivals each kick a 200-token prefill, and the uncharged bound
   fails.
3. Normalizing area by M′ instead of M = M′ − L_max fails.

### 4.5 Theorem D

Constants:
- β_cp = max_y αρ_y/(αq_y + aX_y);
- β = max_y (αρ_y + κ_y)/(αq_y + cX_y);
- θ = max_y ℓ⁺_y/ℓ⁻_y (θ = 1 with unbounded memory).

**Theorem D (hybrid, restricted benchmark).** Let OPT_na be restricted to engines that do not prefill a node
before it is available. Then
E[ALG] ≤ (1+β_cp)·LB_cp + max(1+β, θ)·S(σ + A/M) ≤ [(1+β_cp) + 2·max(1+β, θ)]·OPT_na.
With unbounded memory: E[ALG] ≤ (1+β_cp)·LB_cp + (1+β)·Smith(σ) ≤ (2 + β_cp + β)·OPT_na.

*Proof.*
1. Node length: α(q+ρ) + aX ≤ (1+β_cp)(αq + aX) ≤ (1+β_cp)(αq + (a+c)X), so CP′_J ≤ (1+β_cp)·CP^opt_J on every
   realization.
2. Z′_y ≤ (1+β)σ_y + θ·A_y/M ≤ max(1+β, θ)(σ_y + A_y/M), using αρ + κ ≤ βσ, a ≤ a+c and ℓ⁺ ≤ θℓ⁻.
3. The ranking is the Smith order of Z′, so Σ w_J Σ_{K⪯J} Z′_K = Smith(Z′) ≤ max(1+β,θ)·S(σ+A/M), since
   Smith is monotone in sizes.
4. Apply (L1) and (L2+L3). ∎

**Theorem D (hybrid, unrestricted benchmark).** Let β_cp⁰ = max_y α(q_y + ρ_y − q⁰_y)/(αq⁰_y + aX_y). Against
every non-anticipatory engine schedule at speed 1 with memory M,
E[ALG] ≤ min{(1+β_cp⁰)·LB⁰, LB⁰ + (1+β)·Σ_J w_J σ_J} + max(1+β, θ)·S(σ + A/M)
       ≤ [min(1+β_cp⁰, 2+β) + 2·max(1+β, θ)]·OPT_na.
With unbounded memory (A = 0, θ = 1, and Smith(σ) ≤ OPT_na by (L2)) the constant is
min(1+β_cp⁰, 2+β) + (1+β) ≤ 3 + 2β.

*Proof.* Lemma D1 and steps 2–3 above do not involve the benchmark. Only step 1 changes, in two alternative ways.
- *Node-wise* (ours): α(q+ρ) + aX = (αq⁰ + aX) + α(q+ρ−q⁰) ≤ (1+β_cp⁰)(αq⁰ + (a+c)X), so CP′_J ≤ (1+β_cp⁰)·CP⁰_J.
  Then use (L1⁰).
- *Additive* (the verifier's): α(q+ρ) + aX ≤ (a+c)X + α(q+ρ) and α(q+ρ) ≤ (1+β)σ, so
  CP′_J ≤ CP⁰_J + (1+β)σ_J. Then use (L1⁰) with q⁰ ≡ 0 and (L0).

Both are valid, so the minimum is. ∎

The node-wise form is sharper when prompts are short relative to decodes. The additive form is sharper when a
fan-out node's prompt dominates its decode, since β_cp⁰ then blows up. On the (L1) counterexample, β_cp = 0 but
β_cp⁰ = 11. There ALG = 24 > (1+β_cp)·LB⁰ = 15, so the restricted constant cannot simply be reused against LB⁰.

Whether the restricted constant (1+β_cp) + 2max(1+β,θ) still holds against the unrestricted OPT_na is **open**.
The proof does not cover it. No exact engine optimum was computed, by us or by the verifier, so it is not
refuted either.

**Engine version (R3).** By R0a, run P on the fluid engine at speed 3 with prefill share 1/3. Every hybrid rate
is dominated. The engine reproduces the hybrid schedule *exactly*, not merely faster: it time-shares a fraction
λ = (a+cB)/(2a) ≤ 1 of the time between (u = 1/3, all B decoders) and (u = 1/3, no decoders). The verifier noted
that this matters, since merely running faster under list scheduling could change the schedule. Memory states
are therefore identical, and completion times are equal. The lower bounds (L0–L3, L1⁰) are stated for the
speed-1 engine. Hence:

> **Theorem D (engine).** Fluid engine, speed 3, memory M′ = M + L_max (≤ 2M when L_max ≤ M), deterministic token
> counts known at release, random tool lags, DAG tasks, release dates, online:
> E[Σ w_J C_J] ≤ [(1+β_cp) + 2·max(1+β, θ)]·OPT_na(engine, speed 1, memory M, no prefill before availability),
> E[Σ w_J C_J] ≤ [min(1+β_cp⁰, 2+β) + 2·max(1+β, θ)]·OPT_na(engine, speed 1, memory M).
> In the restricted version, β = 0 and θ = 1 give 3. With unbounded memory and no parent context it is 2
> (speed 3, μ = 1). Without fan-out pieces, q⁰ = q, so β_cp⁰ = β_cp and the two versions coincide.

Checks:
- [T: thmD_ratio_against_engine_lower_bounds] checks the first inequality exactly over all lag realizations on
  200 instances. The worst ALG / max(LB_cp, S/2) = 1.45, while the proven constant on those instances reaches
  29.3, so the proven constant is very loose.
- On every instance the same test also checks the two proof steps with the exact constants:
  - the per-node maxima equal β_cp, β and θ;
  - Z′_K ≤ max(1+β,θ)(σ_K + A_K/M) for every task;
  - CP′ ≤ (1+β_cp)CP^opt on every realization.

  Understated constants therefore fail it. The verifier's mutation (0, −1/2, 1/2) previously passed.
- [T: thmD_unrestricted_benchmark_corrected_bound] checks the unrestricted version exactly on 120 instances:
  - both proof forms of step 1 on every realization;
  - LB⁰ ≤ LB_cp;
  - the counterexample mutation above;
  - the calibrated value.
- [T: thmD_pathwise_bound_any_keep_policy_and_preload] now also runs the κ and area stress instances. Z′ without
  κ, or without the area term, fails there (previously only [T: thmD_hypotheses_are_necessary] caught these).

### 4.6 Size of the constants (calibrated, honest)

Calibrated constants: a = 0.027767, c = 0.000543 (m = 51), α = 1.14·10⁻⁴.

Representative agent task:
- plan node: q = 2000, X = 200;
- final node: ρ = 2200, q = 1500, X = 300.

| Memory | β_cp | β | θ | Ratio (restricted) | β_cp⁰ (final prompt a fan-out) | Ratio (unrestricted) |
|---|---|---|---|---|---|---|
| unbounded | 0.03 | 0.75 | — | 2.78 | 0.051 | 2.80 |
| M′ = 64k tokens | 0.03 | 3.48 | 1.1 | 9.99 (dominated by κ = α(ℓ⁺ + L_max)) | 0.051 | 10.01 |

β_cp⁰ = α(1500 + 2200)/(a·300) = 0.0506 for the final node, and 0 for the plan node. The min picks
1 + β_cp⁰ over 2 + β. The 64k value is checked in [T: thmD_unrestricted_benchmark_corrected_bound].

The κ term is an artifact of charging worst-case kick recompute per availability event. It vanishes if kicks are
rare. Reducing it (e.g., by kick-free admission with a small reserve) is open.

---

## 5. R0 and R1: engine reductions and recompute

**R0a** [T: R0a_engine_lift_and_converse_z3], Z3-proved:
- *Lift.* With s = 3 and u = 1/3, prefill runs at 1/α and each decoder at 2/(a+cB) ≥ 1/a for B ≤ m = a/c.
- *No s < 3 works* at B = m, for any u.
- *Converse.* At speed 1, each decoder runs at ≤ 1/a, the total decode rate is ≤ m/a and prefill is ≤ 1/α. Any
  such rate vector is feasible on m preemptive machines plus the server (McNaughton). So the engine OPT at speed 1
  is no better than the hybrid OPT. This is not needed for Thm D, which uses direct lower bounds.

The plan's "factor 2 each way" becomes 3 one way (tight) and 1 the other way.

**R0b (discrete steps; proof sketch).** The argument below is a sketch: the step-level case analysis of Lemma D1
is not written out. The bound is supported by exact checks (ours and the verifier's independent simulator, with
354,067 checks and no violation), not by a complete proof. Engine at speed 3 with chunk P̄ = a/α and at most one
prefill operation per step:
- each step lasts (a + cB + αP)/3 ≤ a;
- each running sequence gains one token per step;
- the server gains P̄ = a/α tokens per step, except on the last partial chunk of a prefill.

Events are seen at the next boundary. Re-running Lemma D1 with step granularity loses at most one step per node
on J's path (availability rounding) and one step per prefill completion of a node ranked ⪯ J:

  C_J ≤ (Lemma D1 bound) + a·(|J| + N_J).

[T: R0b_discrete_engine_bound] checks this on 250 instances and over 400 checks. Comments:
- The additive term a|J| ≤ CP^opt_J is harmless.
- a·N_J is at most m/X_min times the GPU-time of those nodes. So the multiplicative form holds only when
  outputs are long (X ≥ m) or prompts are long (αq ≥ a).
- Bounding N_J ≤ Σ(1 + k_y) needs a count of kicks per node. That count is **open** for the discrete engine.
- We could not show the additive term is necessary; random mutations never violated the hybrid bound alone, so it
  is loose. Speed 3 is necessary for the per-step domination (Z3).

**R0c.** Running at most m_eff = min(a/c, ⌊M′/L_max⌋) sequences is trivially memory-safe. The work term of
Lemma A1 grows by m/m_eff. Lower-bound instances [T: R0c_machine_cap_loses_m_over_meff]:
- *m = 1 versus m = 8* (unbounded memory; this is a comparison of machine counts, not an instance of the m_eff
  rule): n = 8 equal decodes give the exact ratio 1286/287 ≈ 4.48 ~ (n+1)/2.
- *A consistent m_eff instance* (the verifier's). Add one low-weight long-context task (q = 40, X = 10,
  ℓ⁺ = 50, w = 1/1000) and take M′ = 99. Then L_max = 50 and m_eff = min(8, ⌊99/50⌋) = 1, while policy P fits all
  eight 11-token decodes. The eight short tasks' total completion time is more than 4 times larger under the cap.

The area term of Thm D is the plan's "better" alternative, and it avoids this loss.

**R1.** Evict-always never holds KV across a tool wait, so only active reservations count (ℓ⁺ of admitted
nodes). Its cost is inside Thm D: (1+β_cp) on the critical path and (1+β) on the load. This extends Thm 7 of
`docs/open_problem.md` to mixed steps, memory and lags.

---

## 6. R5 and R6: keep/evict and preload

### 6.1 R5a: one prefix

A prefix waits D ~ known law. Holding costs h per second; evicting costs R if the prefix is later reused.

1. While waiting, the only information is the elapsed time. So every deterministic non-anticipatory rule is a TTL
   τ, and every randomized rule is a mixture of TTLs.
2. The cost is cost(τ) = E[h·min(D,τ) + R·1{D>τ}]. This is linear in the mixture, so a deterministic TTL is
   optimal.
3. Between atoms of D, cost(τ) increases (slope h·P(D>τ) ≥ 0, indicator constant). So candidates {0} ∪ supp(D)
   suffice.

Against the clairvoyant E[min(hD, R)]:
- The Karlin et al. density e^{t/B}/(B(e−1)) on [0, B = R/h] costs e/(e−1)·min(hD, R) for **every** D
  ([T: R5a_randomized_ski_rental_density_numeric], numeric).
- Hence the best TTL is within e/(e−1) for every law.
- The Yao LP (Charnes–Cooper) forces 1.333, 1.463, 1.523, 1.553 on n = 2, 4, 8, 16 grid points.
- [T: R5a_best_ttl_is_optimal_and_within_e_over_e_minus_1] covers 300 random laws, a fine-grid optimality check,
  and the LP at n = 16 (> 1.55).

Exponential waits are memoryless, so τ ∈ {0, ∞}: hold iff h·E[D] < R [T: R5a_exponential_wait_rule_z3].
This is the known stochastic ski-rental structure.

### 6.2 R5b: many tenants and the memory price

Classes i have arrival rate λ_i, preload GPU work P_i, KV size L_i and remaining wait D_i. Per class, choose a
mixture over {no preload} ∪ {(preload, TTL τ)}. Write σ_i(τ) = P(D_i ≤ τ) for the probability the preload is used.

Latency is linearized at the operating point (`docs/controller_theory.md`):
- a used preload removes P from the critical path, at price ν_cp = 1/(1−u);
- a wasted preload adds load P, at price ν_ld = ρ_s/(1−u)².

LP:

  min Σ_i λ_i P_i Σ_τ z_{iτ}[ν_ld(1−σ_i(τ)) − ν_cp σ_i(τ)]  subject to  Σ_i λ_i L_i Σ_τ z_{iτ} E[min(D_i,τ)] ≤ M_h.

The constraint is Little's law for held memory.

**Proposition R5b.**
1. Strong duality holds. [T: R5b_keep_lp_strong_duality_and_sigma_star] computes the exact Lagrangian dual over
   all breakpoints of η, and it equals the Z3 LP optimum on 25 instances.
2. For a memory price η, class i's subproblem is a ski rental with holding rate ηL_i and recompute value
   (ν_cp+ν_ld)P_i. Its TTL is chosen as in R5a.
3. Issue iff max_τ [σ_i(τ) − ηL_i·E min(D_i,τ)/((ν_cp+ν_ld)P_i)] > σ\* = ν_ld/(ν_ld+ν_cp) = ρ_s/(1−u+ρ_s).
4. At η = 0, with σ read as the engine's exogenous survival probability, this is exactly BE′.
5. BE′'s exact saving is P(σ(1−u) − ρ_s(1−σ))/((1−u)(1−u−λ(1−σ)P)). Its first-order term is σν_cp − (1−σ)ν_ld
   (Z3).

So σ\* is the dual condition. This is proved only for the linearized prices, not for the exact queueing objective.

### 6.3 R5c: safety theorem

If held KV sits in the lowest eviction class, Lemma D1 holds for **any** keep policy. Held KV never reduces
admission memory, and hits only shrink works. The ranking stays the one computed from evict-always sizes Z′, so
the Thm D constant is unchanged and keeping is a free improvement **of the bound**.

It is not a per-task improvement. [T: R5c_keep_policies_never_increase_recompute_bound_but_per_task_dominance_fails]
gives an exact instance (m = 1, unbounded memory, held KV plus preload):
- task 2's completion goes from 24 to 27;
- the weighted total improves;
- all bounds still hold.

**Correction to the plan (R5c / §4).**
- *Lemma Q* (a queued request does not reference its cached prefix) does **not** violate P2. A queued request's
  prefix is unreferenced, so it is already in the lowest class. Lemma Q only destroys the *benefit* of holding.
- The *A1 counterexample* of `controller_theory.md` violates compute priority (P3/P4), not eviction priority.
- *Pins* violate P2: pinned KV is not lowest class, so it can block admission.

### 6.4 R6: preloading

Preload chunks run only when the server has no admitted prefill and use only unreserved memory. So they are
invisible to Lemma D1, and Thm D holds unchanged (same test, configurations with `preload=True`).

**Chain tasks.** For an isolated chain task, the saving is exactly α times the number of held prefix tokens used:
the evict-always completion minus the hold-plus-preload completion equals α·(held hits)
[T: R6_isolated_preload_saving_is_exact]. On a chain every node is on the critical path and nothing competes for
the server or decoders, so each held token removes exactly α seconds. On more than 20 of 60 instances the hits
exceed the parent context, so preloading strictly helps. This is the `preload_theory.md` P4 saving in the new
model.

**DAG tasks: no identity.** The earlier version claimed exactness for all tasks, and the verifier refuted it for
DAGs, with saving 1/2 < α·(held hits) = 1 (a hit off the critical path). The verifier's suggested replacement, "at
most α·(held hits)", is **also false**. Parallel branches of one task compete for the server and the decoders, so
removing prefill work can reorder them (a Graham-type anomaly). The exact instances, all with unbounded memory and
cross-checked with the verifier's simulator, are in [T: COUNTEREXAMPLE_R6_not_exact_for_DAGs_either_direction]:
1. *Smaller* (verifier): saving 1/2, α·hits = 1.
2. *Larger.* α = 1/4, m = 1; plan (q = 1, X = 1) → a (q = 2, X = 1, ρ = 1, lag 1) and → b (q = 3, X = 1, ρ = 2,
   lag 0). Holding
   lets b finish its prefill before a arrives. Completion goes 5 → 4: saving 1 > α·hits = 3/4.
3. *Negative.* α = 1/4, m = 2; plan (q = 3, X = 1) → a (q = 2, X = 1, lag 1), → b (q = 1, X = 2, ρ = 1, lag 1),
   → c (q = 3, X = 3, lag 0). Holding makes the isolated task *later*: 6 → 25/4.

What survives for DAGs is the safety statement: Thm D's bound holds with any keep and preload policy (§6.3).

---

## 7. R7: lower bounds and necessity

1. **Clairvoyant benchmark is hopeless.** Thm 2/3 of `docs/open_problem.md` (existing).
2. **Hold-only is unbounded** [T: R7_hold_only_is_unbounded].
   - Instance: a task holds at least 4 tokens from plan start to final end. With M′ = 11, at most k = 2 tasks are
     in flight under **any** hold-only policy.
   - So the i-th completion is ≥ ⌈i/k⌉(t_plan + D + t_final).
   - Evict-always overlaps all waits. Its ratio grows with D toward Σ⌈i/k⌉/n = 3.5 (n = 12) and ~ n/(2k) in
     general, which is unbounded in n.
   - Admitting more plans instead deadlocks memory.
3. **Adversarial arrivals.** Under adversarial arrivals, flow time has no deterministic constant ratio (Jaillet
   et al. 2502.07115, known). Hence the completion-time objective.
4. **Unpinned held KV can be flooded out.** Prop I (existing).
5. **Δ-dependence.**
   - (i) n i.i.d. jobs with P = 0 w.p. 1−q and L w.p. q, on one machine. Every fixed-priority (index) policy gets
     qL·n(n+1)/2, while probing (preemptive) gets L·E[K(K+1)/2] with K ~ Bin(n,q). The exact ratio is
     (n+1)/(nq+2−q) → 1/q = 1+Δ.
   - Also with P ≥ 1 via the exact preemptive DP (ratio > 6/5)
     [T: R7_index_policies_lose_one_plus_delta_against_preemption].
   - So against the **preemptive** benchmark, the worst-case ratio of index policies such as ours is at least
     1+Δ (a supremum over instances, approached as n → ∞). This is folklore-type;
     a Gittins/SEPT-with-probing policy is needed.
   - (ii) Against the non-preemptive benchmark, index policies lose Ω(Δ^{1/4}) (Eberle–Fischer–Matuschke–Megow,
     ORL 2019, Thm 1, known).
6. **P-WSPT-DAG's constant 2 is tight** (§2.4). This is known: Schulz–Skutella / Megow–Schulz Lemma 2.2 give a
   one-machine, unit-weight, lag-free family with ratio → 2
   [T: thmA_constant_two_is_tight_schulz_skutella]. The lag trap (4/3) is a remark.

---

## 8. Online algorithm and the R8 engine checklist

**Algorithm (what each theorem licenses).**
- *Index at release:* w_J / Z′_J (Thm D). In the machine model, w_J/E[W_J] (Thm A/B). Fixed for the task's
  lifetime.
- *Each engine step:*
  - admit the longest priority prefix that fits in M′, kicking the tail on new higher-priority arrivals;
  - prefill the top admitted prefill-phase node (chunk P̄ = a/α in the discrete engine);
  - decode the top m admitted decoders;
  - run preload chunks only when no admitted node prefills.
- *On each tool wait:* keep or evict by the R5a/R5b TTL rule. η is estimated from memory pressure, or σ̂ comes from
  `preload_controller.py`, as BE′ at η = 0. Held KV stays lowest class.

**Checklist (property → where used).**

| Property | Meaning | Used by |
|---|---|---|
| P1 | A queued request references its cached prefix | **No guarantee needs it.** Only the *benefit* of holding and preloading (R5b, R6 savings) does. |
| P2 | Eviction classes: held/unreferenced KV < admitted reservations | Lemma D1, Thm D, R5c, R6 (held KV never blocks admission). Pins violate it. |
| P3 (corrected) | Admission is a priority prefix of the policy's index, with a head-of-line stop, and **preemption (kicks) of lower-priority admitted requests** | Lemma D0/D1, case (B). Stock vLLM's running-first rule (never preempts a running request for a waiting higher-priority one) violates it ([T: thmD_hypotheses_are_necessary] (1)). The plan's "without the head-of-line break" is wrong: the stop is what Lemma D0 uses. |
| P4 | Prefill budget and choice set by the policy: the top admitted prefill, plus P̄ = a/α per step | Lemma D1 case (C); R0b. Lowest-priority preload (R6). |
| P5 | Swap as a timed delay (optional) | Not used. It could be modeled as an extra lag, which Thm D would absorb into CP′. |

---

## 9. Novelty and relation to the literature (honest)

**Literature checked.** Sources read directly were Skutella–Uetz, MUV, EFMM, Queyranne–Schulz, Feng et al.,
Jaillet et al. and Continuum. Megow–Schulz, Megow–Vredeveld and HSSW are from abstracts or secondary sources.

**Feng, Liu, Yang, Zhang (arXiv 2607.09248).** Their model:
- unit rounds, unweighted;
- KV = prompt + decoded, with kill-restart;
- O(1) non-clairvoyant vs clairvoyant OPT, 16 clairvoyant (Thm 6), 3 + o(1) (Cor. 2);
- area-order lower bound (Lemma 1); no global priority rule (Prop. 1);
- no agents, DAGs, tool lags or stochastic model.

Our Thm D is weighted, has DAG tasks with random tool lags, and uses a separate prefill server. Its benchmark is
the non-anticipatory OPT, and it states the speed and memory augmentation explicitly. Our area bound is their
Lemma 1 (known), and our kick charging is similar in spirit to kill-restart charging. We do not improve their
constants in their model.

**Jaillet et al. (2502.07115).** They prove no deterministic constant for adversarial arrivals under flow time.
That is consistent with our choice of objective; we use it only as a lower bound (R7.3).

**Skutella–Uetz (SICOMP 2005).** Non-preemptive list policies for P|r_j, prec|E Σ wC:
- general case: (1+β)(1+1/β+max{1,(m−1)Δ/m}), which equals 3 + 2√2 at Δ ≤ 1;
- in-forests: 2 − 1/m + max{1,…};
- no lags.

Our B′ uses their LP plus lag and critical-path constraints, and their Lemma 3.3. Our *policy preempts*, which
removes their idle-time charging. So B′ is an easier problem, not an improvement on theirs. The new part is
modest: random lags enter only the critical-path constraint and the pathwise lemma.

**Megow–Uetz–Vredeveld (MOR 2006) and Megow–Vredeveld.**
- MUV is online non-preemptive, with ratio 1 + max{1+δ/α, α+δ+(m−1)(Δ+1)/(2m)} for δ-NBUE.
- Megow–Vredeveld reveal jobs at predecessor completion and prove a Θ(n) lower bound (a different model).
- EFMM cite a preemptive index policy as a 2-approximation (their ref. 18, Megow–Vredeveld). That is the closest
  known analogue of Thm A/B.

Our Thm B is essentially WSEPT plus the MSU bound plus a pathwise bound, extended to tasks and lags. We expect a
specialist to find it unsurprising.

**Megow–Schulz (ORL 2004).** P-WSPT is 2-competitive for deterministic online preemptive scheduling. Their
Lemma 2.2, building on Schulz–Skutella (J. Sched. 2002), shows that 2 is tight. Thm A is its direct extension to
DAG tasks with random lags and is proved the same way, so its tightness is inherited (§2.4). The only new
observation is that lags contribute only to the critical-path term, so the ratio vs **clairvoyant** OPT stays 2
even with random lags.

**Queyranne–Schulz (SICOMP 2006).** Deterministic precedence *delays*, non-preemptive, ratio 4. (An earlier
version said "later 2 + 2ln2 + ε". That result (Li, arXiv 1707.08039) is for P|prec|ΣwC and does not cover delays,
so the remark is dropped.) Ours are random lags with a preemptive policy, and a better constant only because of
preemption.

**Continuum (2511.02230).** TTL pinning by predicted tool duration, empirical only. R5a gives the optimal TTL
under a known law and its e/(e−1) worst case, which is the known stochastic ski-rental structure. R5b, linking
the TTL and the issue decision to a memory dual and to BE′, appears new as a statement but is elementary LP
duality.

**Searches.** We found no paper combining random time lags (tools) with min-sum stochastic scheduling. That
combination, together with the KV-memory engine composition (Thm D) and the safety and dual results R5b/R5c,
is what we would call new.

**What is new.**
- (i) The model: DAG agent tasks with random tool lags on a batched engine with KV memory, prefill/decode split
  and explicit augmentation.
- (ii) Thm D and its proof (Lemma D0 kick accounting, three-way wait partition), with exact counterexamples
  showing kicks, κ and the L_max margin are each needed. The (L1) gap and its fix (restricted benchmark, or
  LB⁰ with the corrected constant) are due to the verifier, as are the additive form 2+β and the D0 tightness
  family. The node-wise form 1+β_cp⁰ is ours.
- (iii) R5b/R5c: σ\* as a memory-dual condition; the safety theorem and its per-task failure; the corrected P3.
- (iv) Exact machine checks of every claimed inequality on small instances.

**What is not new.** The MSU bound, Smith/WSEPT arguments, area lower bound, ski rental, EFMM, Jaillet, and the
tightness of Thm A's 2 (Schulz–Skutella, Megow–Schulz).

---

## 10. Limitations and open problems

1. **Random output lengths in Thm D.** Admission needs ℓ⁺, so X must be known or replaced by a reserved upper
   bound X̄; Thm D then holds with X̄ in the memory terms. A Δ-dependent engine theorem (Thm B lifted to the
   engine) is **open**. The obstacle is that S(σ + A/M) needs an MSU-type inequality for the memory area.
   (Partly addressed in `docs/stochastic_theory_ext.md`: Thm H, unbounded memory; Thm DR, cap reservations vs
   OPT_clair.)
2. **Preemptive benchmark with random sizes.** Our index policies have worst-case ratio ≥ 1+Δ (R7.5, a limit over instances). A Gittins/probing policy
   with a matching analysis (Megow–Vredeveld route) is the most promising next step for R2 vs OPT_na with random
   sizes. (Addressed for chains in `docs/stochastic_theory_ext.md`, Thm G.)
3. **Thm A is tight; beating it needs a different policy.** P-WSPT-DAG's 2 is tight (known, §2.4). Whether a
   different online policy beats 2 vs OPT_na with random lags is open. A critical-path-aware index should at least
   avoid the lag trap.
4. **Thm D against the unrestricted benchmark.** The proven constant is min(1+β_cp⁰, 2+β) + 2max(1+β,θ). Whether
   the restricted constant (1+β_cp) + 2max(1+β,θ) also holds is open. The proof does not cover it, and no exact
   engine optimum has been computed to test it.
5. **Large constants in Thm D with memory.** These come from κ and θ (≈ 10 at calibrated values, versus a worst
   checked instance ratio of 1.45). A kick-free admission rule with a reserve, or amortizing kicks against
   lower-priority work, could remove κ.
6. **Discrete engine.** R0b is a proof sketch. The kick count per node and the multiplicative form of R0b are
   open. The attention term
   κ·P·ctx is ignored; it could be folded into an effective α.
7. **Token counts known at release** (H-prompt). The index Z′_J is computed at release, so Thm D needs every node's
   q, ρ and X at release, including the token counts of tool outputs that have not returned yet. Knowing them only
   at node availability is **not** enough. That would be the realistic assumption, but the policy and proof do not
   cover it. Upper bounds known at release (reserved as X̄, q̄) restore the theorem with those bounds in Z′ and in
   the constants.
8. **Lag independence.** Thm A/D need lags independent of the schedule. Lags that depend on load (shared tool
   back-ends) are not covered.
9. **R5b uses linearized prices.** The exact queueing objective is not an LP.

---

## Appendix: machine checks index

32 tests in `tests/test_stochastic_theory.py`, about 25 s.

| Test | Section |
|---|---|
| thmA_* (7, including constant_two_is_tight_schulz_skutella), fanout_collapse_to_max_lag | §2 |
| thmB_* (4), msu_variance_correction_is_necessary, nonpreemptive_benchmark_is_necessary | §3 |
| thmD_* (4), COUNTEREXAMPLE_L1_needs_restricted_benchmark, lemmaD0_charge_tight_family, R5c_*, R6_*, COUNTEREXAMPLE_R6_not_exact_for_DAGs_either_direction | §4, §6 |
| R0a_*, R0b_*, R0c_* | §5 |
| R5a_* (3), R5b_* | §6 |
| R7_* (2) | §7 |

Exact optimal policies (`opt_np_dp`, `opt_pmtn_dp`, `opt_clair_1m`) are finite-horizon DPs on integer data:
- `opt_np_dp` decides at integer times;
- `opt_pmtn_dp` is exact for m = 1 and an upper bound on the true preemptive optimum for m ≥ 2;
- `opt_clair_1m` is exact for m = 1.
