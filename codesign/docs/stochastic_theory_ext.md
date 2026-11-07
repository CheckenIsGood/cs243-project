# Stochastic theory, extensions (R2–R5 pushed further)

This continues `docs/stochastic_theory.md` (cited as **[base]**; written against commit e17bd7f and revised
together with this document after verification). Notation, models and benchmarks are as in [base] §1:

- OPT_na: the optimal non-anticipatory policy at speed 1 (memory M where relevant).
- OPT_clair: the clairvoyant optimum, with OPT_clair ≤ OPT_na.
- Δ: a squared coefficient of variation.

Code: `src/codesign/stochastic_theory_ext.py`. Machine checks: `tests/test_stochastic_theory_ext.py`,
cited as **[X: name]** (without the `test_` prefix). There are 22 tests, about 18 s in total. All arithmetic is
exact (Fractions); Z3 is used for algebra.

Two pieces of background are taken from memory and were **not re-read in this session**:

- **Megow–Vredeveld (MOR 2014), "A tight 2-approximation for preemptive stochastic scheduling."** The Gittins
  index priority policy is a 2-approximation for P|r_j, pmtn|E Σ w_j C_j against the preemptive
  non-anticipatory optimum. Theorem G below is its extension to chain tasks with random lags, and its proof
  follows theirs.
- **Gittins/Sevcik/Weiss optimality.** For single-machine preemptive E Σ wC with independent job processes, the
  Gittins index policy is optimal.

We re-verified both facts exactly on small instances.

## Verification status

An independent verifier audited this document at commit eaa6958 (`docs/stochastic_theory_ext_verification.md`,
code `src/codesign/stochastic_ext_verify.py`). This revision applies the fixes.

| Result | Verifier's verdict | Status in this revision |
|---|---|---|
| Lemma G2, Gittins single-machine optimality | VERIFIED | unchanged |
| Thm G (≤ 2·OPT_na, chains, lags) | VERIFIED WITH FIX: "any law" not proved | **fixed** (§0, §1, §2): discrete finite-support laws, independent across nodes and tasks (the Megow–Vredeveld assumption). m = 2 test tightened |
| Lemma P1 | VERIFIED; side finding c\* ≤ 58/77 | **fixed** (§3): witness 58/77 added as its own test; second branch needs Δ < 2 |
| Lemma P2, Thm P | VERIFIED (wording fixes) | **fixed** (§3): Δ < 2 branch; "loses 1+Δ" is a limit; "within a factor ≤ 43/16"; Δ_W ≤ max node CV² for correlated nodes |
| Thm H | VERIFIED WITH FIX (inherits [base] (L1)) | **fixed** (§4): as stated for prompts without staggered pieces, or against the restricted benchmark; otherwise β_cp⁰ and LB⁰ (2.52 → 2.54, 2.67 → 2.74) |
| Thm DR | VERIFIED WITH FIX (same scope) | **fixed** (§5): 8.68 → 8.74 unrestricted |
| Prop K | VERIFIED (same caveat) | **fixed** (§6): 5.80 → 5.82 unrestricted |
| κ lower bound | VERIFIED; general family found | family added ([base] §4.3, `test_lemmaD0_charge_tight_family`) |
| √2 instance | VERIFIED (supremum, never attained) | kept as a **remark** only (§7) |
| "√2 improves 4/3", "gap √2 vs 2", "Thm A tightness open" | **FALSE**: 2 is known to be tight (Schulz–Skutella; Megow–Schulz Lemma 2.2) | **fixed** (§0, §7, §8, §9): Thm A is tight |
| Calibrated constants | VERIFIED | unchanged; preload-aware values added ([X: preload_aware_constants_H_DR_K]) |
| Tests | three weak spots | **fixed**: [X: lemmaP1_witness_58_77_refutes_larger_constants]; β_cp instance in [X: thmH_ratio_domination_and_gittins_optimality]; 1/m instance in [X: thmG_parallel_machines_consistency] |

---

## 0. New results at a glance

**R2/R4 (machine model)**

| Item | Statement (short) | Status | Restriction | Constant | Machine check |
|---|---|---|---|---|---|
| **Thm G** | Gittins-quanta policy. Pathwise C_J ≤ r_J + CP_J + (1/m)Σ_{K≠J} V_KJ. E[ALG] ≤ LB_cp + Gitt₁/m − (1/m)Σ w E W ≤ **2·OPT_na** | **proved** (extends a known result) | **chain** tasks (LLM nodes in sequence; tool fan-outs as lags, valid in the machine model); random sizes with **discrete finite-support** laws, independent across nodes and tasks; random lags independent of sizes; release dates; online; m machines; speed 1; **preemptive** benchmark | 2, no Δ. Worst checked 1.044 (m = 1, exact OPT_na) | [X: gittins_index_exact_and_quanta_strictly_decreasing], [X: gittins_single_machine_is_optimal_and_beats_wsept], [X: thmG_pathwise_bound_every_realization], [X: thmG_pathwise_mutations_fail], [X: thmG_two_approx_vs_exact_preemptive_optimum_m1], [X: thmG_parallel_machines_consistency], [X: thmG_hypotheses_are_necessary] |
| **Thm P** | Index policy P-WSPT-DAG (w/E[W]). E[ALG] ≤ (1 + ρ(Δ_W))·OPT_clair ≤ (1 + ρ(Δ_W))·OPT_na | **proved** | **general DAGs**; random sizes (only the task totals W_J need be independent across tasks); lags; releases; online; m machines | ρ(Δ) ≤ min{(27/16)(1+Δ), 1/(1−√(Δ/2))} (second branch only for Δ < 2), with ρ(0) = 1, which recovers Thm A. Index policies have ratios approaching 1+Δ (a supremum over instances), so the bounds are within a factor ≤ 43/16 | [X: lemmaP1_emin_lower_bound_and_witness], [X: lemmaP1_witness_58_77_refutes_larger_constants], [X: thmP_index_policy_vs_clairvoyant_m1], [X: thmP_parallel_machines_and_delta_dependence] |

**R3 (engine)**

| Item | Statement (short) | Status | Restriction | Constant | Machine check |
|---|---|---|---|---|---|
| **Thm H** | Engine with **random output lengths**, Gittins-quanta policy. E[ALG] ≤ (1+β_cp)·LB_cp + Gitt₁(Z′) − Σ w E Z′ ≤ (1 + β_cp + λ)·OPT_na | **proved (restricted)** | memory **unbounded** (or never binding); chain tasks; X with finite support, revealed token by token; fluid engine **speed 3** (R0a); evict-always; benchmark restricted as in [base] (no prefill before availability) **or** prompts without staggered pieces; else β_cp → β_cp⁰ | 1 + β_cp + λ, Δ-free. **2.67** at calibrated values with random X (2.52 with deterministic X, vs 2.78 for [base] Thm D). Unrestricted, fan-out final prompt: 2.74 (2.54) | [X: thmH_pathwise_bound_hybrid_random_outputs], [X: thmH_pathwise_mutation_s_process_fails], [X: thmH_ratio_domination_and_gittins_optimality], [X: thmH_fixed_priority_mutation_fails], [X: thmH_calibrated_constant], [X: preload_aware_constants_H_DR_K] |
| **Thm DR** | [base] Thm D with **random outputs and bounded KV memory**: reserve ℓ⁻ + X̄ (support cap) and use index w/E[Z′]. E[ALG] ≤ [(1+β_cp) + 2λ_E·ρ(Δ_s)]·OPT_clair | **proved (restricted)** | bounded support X ≤ X̄; memory M′ = M + L̄_max; speed 3; benchmark is the **clairvoyant** engine optimum (stronger than OPT_na), restricted as in [base] (L1), else β_cp → β_cp⁰ | **8.68** at calibrated values with variable X (Δ_s = 0.30); 5.80 with deterministic X. Unrestricted: 8.74 | [X: thmDR_random_outputs_with_cap_reservation], [X: thmDR_calibrated_constant], [X: preload_aware_constants_H_DR_K] |

**κ term (R3) and Thm A**

| Item | Statement (short) | Status | Restriction | Constant | Machine check |
|---|---|---|---|---|---|
| **Prop K** | Sharper [base] Thm D constant: (1+β_cp) + 2λ_D, with λ_D = max_K Z′_K/(σ_K + A_K/M) ≤ max(1+β, θ) | **proved**, tight for one task | as [base] Thm D (restricted benchmark; unrestricted: min(1+β_cp⁰, 2+β) in place of 1+β_cp) | **5.80** at calibrated values (was 9.99); 4.82 at M′ = 32k; 6.75 at 128k. Unrestricted: 5.82 (was 10.01) | [X: propK_sharpened_thmD_constant], [X: preload_aware_constants_H_DR_K] |
| **κ lower bound** | Under priority-prefix admission, one 2-token arrival can force recompute of a whole L_max context. Lemma D0's per-event charge ℓ⁺_y + L_max cannot be reduced without changing the admission rule. The verifier's family makes it tight up to an additive 1 for every ℓ⁺_y | **proved** (exact instances) | — | — | [X: D0_kick_charge_is_tight_per_event], [base: lemmaD0_charge_tight_family] |
| **Thm A lower bound** | Thm A's 2 is **tight**, by a known family (Schulz–Skutella; Megow–Schulz Lemma 2.2). Our √2 family is a remark only | known | deterministic, m = 1, unit weights, no lags | 2 (known); our family → √2 | [base: thmA_constant_two_is_tight_schulz_skutella], [X: thmA_lower_bound_sqrt2] |

**Still open** (§8):
- Gittins with **bounded memory**. The obstacle is that priority drops reorder the admitted prefix, which causes
  kicks at quantum boundaries.
- Gittins for **DAG** tasks with parallel LLM nodes.
- Quantile reservation with overflow.
- The exact pair constant: the worst case lies in [77/58, 27/16]·(1+Δ) ≈ [1.33, 1.69]·(1+Δ) (lower end witnessed at
  Δ = 19, the verifier's 58/77 witness).
- The multiplicative form of R0b.

---

## 1. Work processes and Gittins quanta

**Work process.** A chain task J is a sequence of nodes. Node i is:
- a deterministic **lump** of work. It may depend on the previous node's realized length, e.g. a context
  recompute α(q + ρ + X_parent);
- followed by N_i **units**, each of work u_i. N_i ≥ 1 is random with a **finite support**, independent across
  nodes and tasks, and revealed unit by unit: at each unit boundary we learn whether the node ends.

The state is (node, units done, previous length). The process is Markov and its state space is finite.

- Machine model: lump = 0, unit = 1, N = p (integer sizes).
- Engine: lump = prefill seconds, unit = per-token cost, N = output length X.

**Gittins index.** G(x) = sup over stopping times τ ≥ 1 step of w·P(done by τ) / E[work until τ or done]. We
compute it exactly by Dinkelbach iteration over stopping policies. [X: gittins_index_exact_and_quanta_strictly_decreasing]
checks it against brute force over all stopping sets.

**Quanta** (Megow–Vredeveld). From a state x₀, a quantum lasts until the first state y with G(y) < G(x₀). The next
quantum starts at y with priority G(y) < G(x₀). Along any realization, the priorities of a task's quanta are
**strictly decreasing** (checked). Ties between tasks are broken by task id, so key(quantum) = (−G, id).

**Lemma G2 (single-machine identity).** On one machine with no releases and no lags, process all quanta in
increasing key order. The completion time of J is then

  C¹_J = W_J + Σ_{K≠J} V_KJ, where V_KJ = the work of K in quanta with key < key(J's last quantum).

*Proof.*
1. A task's quanta have increasing keys.
2. So K's quantum ℓ is processed before J's last quantum iff its key is smaller.
3. In that case all of K's earlier quanta, which have smaller keys still, are processed too. ∎

This policy is the Gittins index policy, which is optimal for single-machine preemptive E Σ wC over
non-anticipatory policies. [X: gittins_single_machine_is_optimal_and_beats_wsept] checks the optimality on 40
instances with two independent exact DPs, `single_machine_opt` and `opt_pmtn_dp`. It also gives an instance where
WSEPT is strictly worse: 15 > 57/4. We write Gitt₁ for the optimal value.

## 2. Theorem G: chain tasks, random sizes and lags, preemptive benchmark (R2/R4)

**Policy.** At every instant, run the m available tasks with the smallest current-quantum key, preemptively. A
chain task has at most one available node at a time. The index of a task depends only on its own law, so the
policy is online.

**Lemma G1 (fast single machine).** OPT_na(m machines, releases, lags) ≥ Gitt₁/m, where Gitt₁ is computed at
speed 1 with no releases and no lags.

*Proof.*
1. Take any non-anticipatory policy π on m machines.
2. Run a single machine of speed m that gives each task, at each instant, the rate π gives it (at most 1 per task,
   at most m in total).
3. The single machine draws lags from their law using its own randomness, which is valid because lags are
   independent of sizes. It ignores releases.
4. Every task completes no later than under π. The simulated policy observes exactly the work-process states π
   observes, so it is non-anticipatory.
5. Hence OPT_na ≥ OPT₁(speed m) = Gitt₁/m. ∎

Mutation [X: thmG_hypotheses_are_necessary]: with speed 1 instead of m, the claim fails. OPT_na(m = 2) ≤ 59/8 by
grid DP, which is below Gitt₁ = 12.

**Lemma G3 (pathwise).** On every realization, C_J ≤ r_J + CP_J + (1/m)·Σ_{K≠J} V_KJ, where CP_J = W_J + the sum
of J's realized lags.

*Proof.* Classify each instant of [r_J, C_J]:
1. J is in a lag: total = Σ lags.
2. J runs: total = W_J.
3. J is available, in a quantum with key k, but not running. Then all m machines run tasks whose current quantum
   has key < k ≤ key(J's last quantum), since keys of J's quanta increase. That work belongs to quanta counted in
   V_·J, and each unit of work is counted once. So the total of such instants is ≤ (1/m)Σ V_KJ. ∎

**Theorem G.** For chain tasks with random sizes that have discrete laws with finite support, independent across
nodes and tasks (the Megow–Vredeveld setting), random lags independent of sizes, release dates, online arrival
and m machines:

  E[ALG] ≤ LB_cp + Gitt₁/m − (1/m)Σ_J w_J E[W_J] ≤ 2·OPT_na.

*Proof.* Take expectations in G3. By G2, Σ_J w_J Σ_{K≠J} V_KJ = Σ_J w_J (C¹_J − W_J). Then use LB_cp ≤ OPT_na and
G1. ∎

The earlier version said "any laws". The quanta, Lemma G2 and Gittins optimality are used here only for
finite-state Markov work processes. Megow–Vredeveld also assume discrete laws. Continuous or unbounded laws would
need a limit argument that is not given, so they are **not** claimed.

**Checks.**
- [X: thmG_pathwise_bound_every_realization]: G3 on more than 1500 (task, realization) pairs; m ∈ {1,2,3};
  chains of 1–3 nodes with random lags and releases.
- [X: thmG_two_approx_vs_exact_preemptive_optimum_m1]: 60 instances against the **exact** preemptive OPT_na (DP,
  m = 1). Worst ratio 1.044.
- [X: thmG_parallel_machines_consistency]: m = 2. It includes four unit jobs where ALG = 6 ≤ 4 + (10−4)/2. A
  halved interference term, 4 + (10−4)/4, fails there, so the 1/m factor is exercised (the verifier's weak spot
  (c)).

**Mutations** [X: thmG_pathwise_mutations_fail], [X: thmG_hypotheses_are_necessary]:
1. Using J's *first* quantum key instead of its last breaks G3.
2. Dropping the lags from CP breaks G3.
3. Replacing the Gittins quanta by a fixed WSEPT priority breaks the theorem's bound (sizes {1 w.p. 3/4, L w.p.
   1/4}).
4. Speed 1 instead of m in G1 fails.

**Scope.** Chains cover ReAct loops (LLM → tool → LLM → …) and our plan → {tools} → final workload, since a tool
fan-out collapses to one lag in the **machine** model ([base] §1.1). The collapse is not valid for the engine's
lower bound, where a fan-out's outputs are prompt pieces that can be prefilled before the node is available
([base] §4.1). That matters for Thm H, DR and Prop K (§4–§6), not for Thm G. DAGs whose LLM nodes run **in parallel** (multiple sub-agents) are
**open**: the single-machine relaxation then processes a non-sequential job, and the Gittins theorem no longer
applies.

**Novelty.** For single-node jobs with release dates this is Megow–Vredeveld's theorem, and the proof is theirs.
What is new is that it carries over unchanged to chains with **random time lags**: lags enter only the CP term.
This settles R2/R4 against OPT_na for the agent workload with constant 2 and no Δ dependence.

## 3. Theorem P: the index policy against the clairvoyant optimum, general DAGs (R2/R4)

**Lemma P1.** Let X, Y ≥ 0 be independent with E X = E Y = 1 and CV² ≤ Δ. Then

  E[min(X, Y)] ≥ max{ 16/(27(1+Δ)), 1 − √(Δ/2) }.

The second bound is informative only for Δ < 2. For Δ ≥ 2 it is ≤ 0, and the corresponding branch of ρ below is
read as ∞.

*Proof.*
- **First bound.**
  1. Write E min(X,Y) = ∫₀^∞ F̄(t)Ḡ(t) dt. Both survival functions are non-increasing, so Chebyshev's integral
     inequality gives ∫₀^T F̄Ḡ ≥ (1/T)(∫₀^T F̄)(∫₀^T Ḡ).
  2. Next, ∫₀^T F̄ = 1 − E(X−T)⁺ ≥ 1 − E X²/(4T), because (x−T)⁺ ≤ x²/(4T) (Z3).
  3. Take T = 3(1+Δ)/4. This gives (4/(3(1+Δ)))·(2/3)² = 16/(27(1+Δ)) (Z3).
- **Second bound.** min(X,Y) = (X + Y − |X−Y|)/2, and E|X−Y| ≤ √(Var X + Var Y) ≤ √(2Δ). ∎

[X: lemmaP1_emin_lower_bound_and_witness] checks the lemma exactly on 3000 random laws.

**The constant is not 1/(1+Δ).** For Bernoulli laws, E min = 1/(1+Δ) exactly, which suggests 1/(1+Δ) as the
right constant. That guess is **false**. Exact witness:
- X ∈ {0 w.p. 2/3, 3 w.p. 1/3};
- Y ∈ {11/20 w.p. 10/11, 11/2 w.p. 1/11};
- Δ = 81/40, and (1+Δ)·E min = 187/240 ≈ 0.779.

A sharper witness is due to the verifier ([X: lemmaP1_witness_58_77_refutes_larger_constants]):
- X ∈ {0 w.p. 19/20, 20 w.p. 1/20};
- Y ∈ {1/2 w.p. 76/77, 39 w.p. 1/77};
- both have mean 1 and CV² = 19, and (1+Δ)·E min = **58/77 ≈ 0.7532**.

So the best constant c\* in E min ≥ c\*/(1+Δ) lies in **[16/27, 58/77] ≈ [0.593, 0.753]**. Our earlier local
search (0.776) had not found this. The test computes the witness directly, so false constants such as 0.76 or 0.80
fail it. The verifier noted that the random-law check alone passes with them.

**Lemma P2 (pairwise reduction).** On one machine with independent task totals W_J:

  Smith(E W) ≤ ρ·E[Smith(W(ω))], with ρ = max(1, max_{J,K} min(a,b)/E min(aX, bY)) ≤ 1/c(Δ_W).

Here Smith(E W) is the value of the fixed WSEPT order, and E[Smith(W(ω))] is the **clairvoyant** single-machine
optimum.

*Proof.*
1. Both values equal Σ w E W plus a sum over pairs.
2. For WSEPT, the pair (J,K) costs min(w_K E W_J, w_J E W_K).
3. For the clairvoyant schedule, the pair costs E min(w_K W_J, w_J W_K).
4. With a = w_K E W_J ≤ b = w_J E W_K: E min(aX, bY) ≥ a·E min(X, Y) ≥ a·c(Δ). Apply P1. ∎

**Theorem P.** Take P-WSPT-DAG with index w_J/E[W_J] ([base] §2.1). Assume general DAG tasks, arbitrary lags
independent of the policy, release dates, online arrival and m machines, with task totals independent across
tasks. Then

  E[ALG] ≤ LB_cp + (1/m)Smith(E W) ≤ LB_cp + ρ(Δ_W)·E[(1/m)Smith(W(ω))] ≤ (1 + ρ(Δ_W))·OPT_clair ≤ (1 + ρ(Δ_W))·OPT_na,

where ρ(Δ) ≤ min{(27/16)(1+Δ), 1/(1−√(Δ/2))} (the second term only for Δ < 2) and Δ_W = max_J CV²(W_J) ≤ the
maximum node CV².

**Δ_W ≤ max node CV², even for correlated nodes.** By Minkowski, sd(Σ_x p_x) ≤ Σ_x sd(p_x) ≤ √Δ·Σ_x E p_x, so
CV²(W_J) ≤ Δ without any independence between the nodes of one task (verifier, Z3 and exact checks). Only the task
totals need to be independent across tasks.

*Proof.*
1. [base] Lemma A1 in expectation. The ranking is fixed, so E Σ_{K⪯J} W_K = Σ E W_K.
2. Apply P2.
3. Use (1/m)Smith(W(ω)) ≤ OPT_clair(ω) pathwise, as in [base] §2.3. ∎

**Remarks.**
- At Δ = 0, ρ = 1, which recovers Thm A's ratio of 2.
- The benchmark is the *clairvoyant* optimum, so this is stronger than [base] Thm B, which is only against
  OPT_np. It also holds for **general DAGs**, where Theorem G does not.
- **Necessity:** the ratio of every fixed-priority policy approaches 1+Δ against OPT_na on the family of
  [base] R7.5: (n+1)/(nq+2−q) → 1+Δ as n → ∞, and (n+1)/(nq+2−q) ≤ 1+Δ for each n. This is a supremum over
  instances, not a loss on every instance. On this family the clairvoyant optimum equals probing, so the same
  holds against OPT_clair. So linear dependence on Δ is unavoidable for index policies. The upper bound
  1 + (27/16)(1+Δ) and the lower bound 1+Δ differ by a factor of at most 27/16 + 1/(1+Δ) ≤ 43/16.

**Checks.**
- [X: thmP_index_policy_vs_clairvoyant_m1]: exact clairvoyant DP, with chains and DAGs.
- [X: thmP_parallel_machines_and_delta_dependence]: m = 2, 3, plus the Δ-necessity mutation.

Both tests check every link of the chain, including ρ ≤ (27/16)(1+Δ_W).

## 4. Theorem H: the engine with random output lengths, unbounded memory (R3)

**Hybrid model and engine lift.** These are as in [base] §1.3 and R0a: a fluid engine at speed 3 dominates the
hybrid model. Memory is unbounded, or large enough never to bind.

**Node and task assumptions.**
- Node y has q new prompt tokens and ρ_y recomputed context tokens. ρ_y may include the parent's realized output
  (`ctx`).
- The output length X_y has a discrete law with finite support in {1, 2, ...}, independent across nodes, and is
  revealed token by token (EOS).
- Tasks are chains; lags are independent of X.

**Two work processes per task.**
- ALG's **Z′-process**: lump α(q+ρ), and c per decoded token. c is the decode-machine time a/m per token, so m
  decoders accrue at total rate 1.
- The **s-process** (lower bound): lump αq, and c per token.

**Lemma H1 (lower bound).** OPT_na(engine, speed 1) ≥ Gitt₁(s).

*Proof.*
1. Every engine step of length τ = a + cB + αP accrues at most cB + αP ≤ τ of s-work (only first-time q tokens
   count).
2. Simulate OPT on one speed-1 machine processing each task's s-process at OPT's accrual rate. Lags are drawn
   independently. The simulation observes the same states (node, prefill progress, decoded tokens).
3. Each task completes no later than in OPT. Then apply Gittins optimality on the single machine. ∎

**Lemma H2 (pathwise).** On every realization of X and the lags, C_J ≤ r_J + CP′_J + Σ_{K≠J} V′_KJ. CP′ uses node
lengths α(q+ρ) + aX, and V′ is computed on the Z′-process.

*Proof.* As in G3. A waiting instant of J is one of:
- server busy with a smaller-key prefill: accrual rate 1;
- m smaller-key decoders running: accrual rate m·(c/a) = 1.

Both charge only quanta of keys < key(J's last). ∎

**Lemma H3 (domination).** Suppose that on every path and at every state, Z′(work so far) ≤ λ·S(work so far)
(cumulative). Then Gitt₁(Z′) ≤ λ·Gitt₁(s).

*Proof.*
1. Run the Z′ machine through the optimal s-schedule's slices in the same order.
2. The end of slice i on the Z′ machine is Σ_K Z′_K(state) ≤ λ·Σ_K S_K(state) ≤ λ·(the s-time at the end of
   slice i).
3. Inside a lump both processes accrue linearly, so the cumulative ratio peaks at states. The Z′ machine
   observes the same states, so it is non-anticipatory. ∎

**Theorem H.** With β_cp = max αρ/(αq + aX_min) and λ as in H3:

  E[ALG] ≤ (1+β_cp)·LB_cp + Gitt₁(Z′) − Σ w E Z′ ≤ (1 + β_cp + λ)·OPT_na(engine, speed 1),

with the policy run on the fluid engine at speed 3 and evict-always. **No Δ appears.**

**Scope ([base] (L1) fix).** The step LB_cp ≤ OPT_na uses [base] (L1). That holds when no node's prompt is a
staggered fan-out, which is the case in the model of this section (each node's q tokens arrive at its
availability), or when the benchmark does not prefill a node before it is available. If a node's prompt is a
fan-out whose pieces return at different times and the benchmark may prefill returned pieces early, replace LB_cp
by LB⁰ (q⁰ = 0 for such nodes) and β_cp by β_cp⁰ = max α(q + ρ − q⁰)/(αq⁰ + aX_min):

  E[ALG] ≤ (1+β_cp⁰)·LB⁰ + Gitt₁(Z′) − Σ w E Z′ ≤ (1 + β_cp⁰ + λ)·OPT_na.

H1, H2 and H3 are unaffected (H1's simulation also covers early piece prefill, since pieces exist only after the
parent finishes). [X: preload_aware_constants_H_DR_K] checks this inequality exactly on 30 random instances with
fan-out final nodes.

**Checks.**
- [X: thmH_pathwise_bound_hybrid_random_outputs]: H2 on more than 1500 checks.
- [X: thmH_ratio_domination_and_gittins_optimality]: Gitt₁ equals the exact single-machine DP for both the Z′ and
  s processes; H3; and the full chain.

**Mutations.**
- Charging with the s-process instead of Z′ (that is, ignoring recompute) breaks H2
  [X: thmH_pathwise_mutation_s_process_fails].
- A fixed priority (WSEPT on Z′) breaks the sharp bound on Bernoulli outputs (44.16 > 43.575)
  [X: thmH_fixed_priority_mutation_fails].
- The ratio of means E Z′/E S was never violated as a domination constant in 120 instances (the cumulative λ may
  be loose). This is not proved, so we do not use it.

**Calibrated constants** [X: thmH_calibrated_constant]: a = 0.027767, c = a/51, α = 1.14·10⁻⁴. The task has a plan
node with q = 2000 and a final node with q = 1500, ρ = 2200.

| Output lengths | β_cp | λ | Ratio | β_cp⁰ (final prompt a fan-out) | Ratio, unrestricted |
|---|---|---|---|---|---|
| deterministic X (200, 300) | 0.030 | 1.494 | **2.52** | 0.051 | 2.54 |
| random X ∈ {50, 200, 200, 800}, {100, 300, 300, 1200} | 0.085 | 1.588 | **2.67** | 0.152 | 2.74 |

[base] Thm D gives 2.78 here for deterministic X, so the cumulative domination is also tighter in the
deterministic case.

## 5. Theorem DR: bounded KV memory with random output lengths (R3)

**Policy.** This is [base] policy P with two changes:
- each node reserves ℓ̄⁺ = ρ + q + X̄ (X̄ = support cap) at admission;
- tasks are ranked by w/E[Z̄′], where Z̄′ is the [base] Z′ with reservations ℓ̄⁺, κ̄ = α(ℓ̄⁺ + L̄_max), and the
  realized decode length.

Kicks happen only at availability events, as before.

**Theorem DR.** Let s_K = σ_K + A_K/M (realized), λ_E = max_K E Z̄′_K/E s_K and Δ_s = max_K CV²(s_K). Then

  E[ALG] ≤ (1+β_cp)LB_cp + Smith(E Z̄′) ≤ [(1+β_cp) + 2λ_E·ρ(Δ_s)]·OPT_clair(engine, speed 1, memory M).

This uses memory M′ = M + L̄_max and speed 3.

*Proof.*
1. [base] Lemma D1 is pathwise and holds for realized X with reservations ℓ̄⁺. Its proof only needs the actual
   KV to stay within the reservation.
2. Take expectations with the fixed ranking.
3. Smith(E Z̄′) ≤ λ_E·Smith(E s), and Lemma P2 gives Smith(E s) ≤ ρ(Δ_s)·E[Smith(s(ω))].
4. Finally, E[Smith(s(ω))] ≤ 2·E[OPT_clair(ω)] by the pathwise concurrent-open-shop bound [base] §4.1. ∎

**Checks.** [X: thmDR_random_outputs_with_cap_reservation] covers 60 instances and more than 2000 pathwise
checks, plus every link of the chain. A `CapNode` makes the frozen simulator reserve the cap while decoding only
the realized X.

**Calibrated** [X: thmDR_calibrated_constant]: M′ = 64k, the variable X laws of §4. Δ_s is only 0.30 even though
CV²(X) ≈ 0.85, because the prompts are deterministic. The result is λ_E = 2.33, ρ ≤ 1.63, ratio **8.68**. With
deterministic X it is 5.80 (= Prop K).

**Scope ([base] (L1) fix).** As for Thm H: the ratio is against the restricted benchmark, or for prompts without
staggered pieces. Against the unrestricted clairvoyant benchmark, 1+β_cp becomes min(1+β_cp⁰, 2+β), giving
8.6755 − 0.0851 + 0.1519 = **8.74** at the calibrated values ([X: preload_aware_constants_H_DR_K]).

Restriction: X needs a bounded support cap X̄ that is reserved. **Quantile reservation with overflow is open.**
Doubling reservations work for the area term, but each growth event can kick up to L_max, which costs an α·L_max
per doubling.

## 6. Shrinking κ (R3)

**Prop K.** [base] Theorem D holds with the constant (1+β_cp) + 2λ_D, where λ_D = max_K Z′_K/(σ_K + A_K/M) ≤
max(1+β, θ).

*Proof.* Smith is monotone and homogeneous, so Smith(Z′) ≤ λ_D·Smith(σ + A/M). For a single task this is an
equality, so no smaller constant works for this step. ∎

[X: propK_sharpened_thmD_constant] checks this on 80 random instances: alg ≤ (1+β_cp)LB_cp + λ_D·S, the
equality for one task, and the calibrated values. The improvement comes from not maximizing the κ part and the
area part separately: κ ≈ α·8000 ≈ 0.91 s is comparable to the area term aXℓ⁺/M ≈ 0.56 s.

| M′ | Old constant | New constant | New, unrestricted benchmark |
|---|---|---|---|
| 32k | — | 4.82 | 4.84 |
| **64k** | 9.99 | **5.80** | 5.82 (old: 10.01) |
| 128k | — | 6.75 | 6.78 |

The unrestricted column replaces 1+β_cp by min(1+β_cp⁰, 2+β) ([base] §4.5), with the final prompt treated as a
fan-out ([X: preload_aware_constants_H_DR_K]).

λ_D grows with M′, because κ is fixed while A/M shrinks. When memory never binds, use the unbounded-memory
constants instead: Thm H gives 2.52.

**What does not work** (analysed, not adopted):
1. *Hysteresis* (admit above a soft limit up to a reserve R, then kick down to the soft limit). Low-priority
   insertions can fill the reserve and then be kicked, so the trigger still pays ℓ_y + L_max in the worst case.
2. *Parking* partially evicted KV in the lowest class. Readmissions then truncate other parked KV, and the total
   loss is again bounded only by the kicked KV.
3. *Charging kicks to later quanta or tokens of the kicker.* Their priority can fall below J's, which breaks the
   charging.

**Lower bound for this admission rule** [X: D0_kick_charge_is_tight_per_event]. One 2-token, high-priority arrival
late in a 10-token decode (M′ = 11) kicks the whole L_max context and forces 9.49 recompute tokens. So the
per-event charge ℓ⁺_y + L_max is tight up to O(1) under priority-prefix admission. The verifier's family ([base]
§4.3) is sharper: the kicked reservation is exactly ℓ⁺_y + L_max − 1 for every ℓ⁺_y. Shrinking κ further needs a
different rule, e.g. swap-out (P5) instead of recompute, which turns κ into a delay. That route is **open**.

## 7. Remark: a √2 family for P-WSPT (superseded)

**Thm A's 2 is tight, and this is known.** Megow–Schulz (ORL 2004, Lemma 2.2), building on Schulz–Skutella
(J. Sched. 2002), give a one-machine, unit-weight family without lags on which P-WSPT's ratio tends to 2. On
single-node tasks without lags, P-WSPT-DAG is P-WSPT. [base] §2.4 reproduces the family exactly (228/160 at n = 3;
1.9846 at n = 150). The earlier version of this section said that √2 "improves 4/3" and that "the gap is now √2
vs 2". That was false: the verifier found the known bound. The family below is kept only as a remark.

**Instance** [X: thmA_lower_bound_sqrt2]: one machine, deterministic, no lags.
- Job A has p = w = L (ratio 1) and is released at 0.
- k unit jobs with weight 1+ε are released at L−1.

P-WSPT preempts A for all k jobs:
- ALG = (1+ε)Σ_{i≤k}(L−1+i) + L(L+k);
- "A first" = L² + (1+ε)Σ_{i≤k}(L+i), and for small sizes the exact DP confirms this is OPT.

At k ≈ √2·L the ratio tends to (1+2x+x²/2)/(1+x+x²/2) at x = √2, which equals √2. The supremum is not attained
(verifier, Z3). The exact value at L = 1000, k = 1414 is 1.4137. The mutation k = 0 gives ratio 1. It is weaker
than the known 2.

## 8. Open problems and the most promising routes

1. **Gittins with bounded memory.** The obstacle: with Gittins priorities, a quantum boundary lowers a running
   node's priority, the prefix reorders, and newly admitted nodes kick it. The kicked amount per quantum boundary
   is not bounded by the boundary's own size. Route: admission by a coarse, monotone priority class (e.g. powers
   of 2 of the Gittins index), with kicks charged per class change.
2. **Gittins for DAG tasks with parallel LLM nodes.** Theorem P covers DAGs with a Δ-dependent constant.
3. **Quantile reservation with overflow** (Thm DR without a support cap).
4. **The exact pair constant c\*** ∈ [16/27, 58/77].
5. **R0b multiplicative**: not attempted further. (Thm A tightness is settled: 2, known; §7.)

## 9. Novelty (honest)

- **Theorem G.** Megow–Vredeveld's 2-approximation (known, for jobs) extended to chains with random lags. The
  extension is short; the content is the observation that lags enter only the critical path. This is the first
  O(1) result here against the **preemptive** non-anticipatory optimum with random sizes.
- **Theorem P.** The pairwise reduction is classical in style. The E min lemma, its constant 16/27 and the
  disproof of 1/(1+Δ) appear new to us but are elementary. The sharper witness 58/77 is the verifier's. The
  consequence — an index policy within 1 + O(1+Δ) of the **clairvoyant** optimum for DAGs with lags, linear in
  Δ, and within a factor ≤ 43/16 of the 1+Δ lower bound — we did not find stated.
- **Theorem H.** New composition. It is a Gittins analysis on engine work processes with a cumulative-domination
  lemma (H3). The engine lower bound (H1) is a direct simulation argument.
- **Theorem DR, Prop K.** Direct sharpenings of [base] Thm D. Prop K is a bookkeeping improvement, but it nearly
  halves the calibrated constant.
- **√2.** Not a contribution. Thm A's 2 is already known to be tight (Schulz–Skutella; Megow–Schulz Lemma 2.2; §7).
- **(L1) scope fix and the unrestricted constants** (2.54/2.74, 8.74, 5.82) follow the verifiers' findings.
