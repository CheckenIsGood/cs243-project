# Independent verification report: general_theory.md, open_problem.md, report/main.tex

Date: 2026-11-07. Scope: every formal claim in `docs/general_theory.md`, `docs/open_problem.md` and the
theorem/lemma statements of `report/main.tex`. No existing file was modified.

Independent artifacts:
- `src/deadline_agent/verify_independent.py`: model, exact simulator, two exact offline solvers, McNaughton,
  fluid and discrete engines, generators, adversaries. Written from the definitions. It does not import
  `general_theory.py` or `open_problem.py`. All scheduling arithmetic uses `fractions.Fraction`.
- `tests/test_independent_verification.py`: **2239 passed, 1 xfailed (strict), about 17 s**. The xfail states
  Lemma 2(i) literally for the discrete engine. The only tests that import the project modules are the two
  labelled `test_crosscheck_*`.
- Scratch (not part of the repo):
  - adversarial searches, `scratchpad/verify/search_thm4.py`;
  - mutation runs, `scratchpad/verify/mutate.py` and `mutate_mine.py`;
  - extracted literature text, `scratchpad/lit/`.

Verdict key:
- **VERIFIED**: the statement and its proof are correct as written.
- **VERIFIED WITH CORRECTIONS**: the statement is true, but the wording, hypotheses or proof need the stated fix.
- **FLAWED**: false as written. A corrected statement is given.
- **UNVERIFIABLE**: a conjecture, or it rests on sources that could not be read.

## 1. Verdict table

| # | Claim | Verdict | One-line reason |
|---|---|---|---|
| G1 | Thm 1 (NP-hardness by restriction) | VERIFIED WITH CORRECTIONS | "Batch size 1" restricts the solution space, not the instance. Fix it by making a batch cap an instance parameter, or use the strengthening in §2.1 (hardness survives unbounded batching). NP-hard in the weak sense only. |
| G2 | Remarks: LRKB 1977, Baruah 1/4, unbounded density | VERIFIED (secondary) | Consistent with the sources (§6). |
| G3 | Lemma 2(i) | **FLAWED as stated** (true in the fluid model) | It is false for the discrete-step engine of §1. Exact counterexamples are in §2.2. The proof silently uses the fluid relaxation. |
| G4 | Lemma 2(ii) | VERIFIED | It also holds for discrete steps. "Completion time is preserved" should read "is no later". |
| G5 | Non-integer m remark | VERIFIED | Correct, but a/c ≥ 1 suffices (not only a/c ≥ 2). |
| G6 | Consequence (C-competitive with 2σ speed) | VERIFIED WITH CORRECTIONS | Holds only in the fluid, decode-only model with unlimited KV. |
| G7 | Lemma 3 (eager issue) | VERIFIED | Monotonicity checked exactly on DAGs with 1 and 2 machines. The online caveat is confirmed by an explicit EDF anomaly. |
| O0 | §1 exact offline solver | VERIFIED | Agrees with two independent exact solvers. The Horn citation is not load-bearing. |
| O1 | Fact A | VERIFIED WITH CORRECTIONS | The "iff ε < 1/(2s)" is false. The exact condition is ε < 1/(2(s−1)). |
| O2 | Thm 2 | VERIFIED | Proof audited line by line. The adversary beats 8 policies at m ∈ {1,2,3}. The threshold is sharp. |
| O3 | Cor 2′ | VERIFIED | It implicitly needs k ≥ 1+δ (for L ≥ 0). |
| O4 | Thm 3 | VERIFIED | |
| O5 | Prop 0 | VERIFIED WITH CORRECTIONS | The complexity citation is misapplied: Ridouard Thm 1 is about *periodic* tasks. |
| O6 | Thm 4 | VERIFIED WITH CORRECTIONS | Step (iii)'s strict "<" is unjustified when k's last unfinished node is a tool. The repaired argument is in §3.4. The project's Z3 test assumes the unjustified premise. |
| O7 | Tightness s* = 1 + 1/δ | VERIFIED | Thm 2 plus the repaired Thm 4. |
| O8 | R4 | VERIFIED | Every claimed speed was reproduced exactly. |
| O9 | R5 | VERIFIED WITH CORRECTIONS | The instance has "near-zero" slack (T0: 1/710), not zero. The exact CP-EDF threshold is 321/221. |
| O10 | C4, C5 (family limits) | VERIFIED (algebra); correctly labelled conjecture | |
| O11 | C1, C2, C3 | UNVERIFIABLE (conjectures) | The evidence is weak, as the document says. |
| O12 | Prop 5 | VERIFIED WITH CORRECTIONS | Tool nodes incomparable to g add a GPU-independent condition. |
| O13 | Cor 5′ | VERIFIED (reduction) / secondary (cited results) | |
| O14 | Thm 6 | VERIFIED WITH CORRECTIONS | Fluid, decode-only, unlimited KV. Not "identical" completion times. The project tested only the algebra; the end-to-end check is new here. |
| O15 | Thm 6b | VERIFIED | Also transfers to the discrete-step engine. |
| O16 | Thm 7 | VERIFIED WITH CORRECTIONS | It is an upper bound. §0's "costs exactly (1+β)" overstates it: no tightness is shown. |
| O17 | Prop 8 | VERIFIED | The scope is EDF on that instance. |
| O18 | Prop 9 | VERIFIED WITH CORRECTIONS | The "cost-optimal caching policy" must be allowed to *delay* requests. Standard paging policies (LRU, Landlord) would evict A here and meet both deadlines. |
| O19 | §4 literature claims | Mostly VERIFIED; corrections in §6 | Two omissions matter for novelty: Chen et al. ECRTS 2019, and Ridouard's informal claim. |
| R1 | Report Thm 1 | VERIFIED WITH CORRECTIONS | Same as G1. |
| R2 | Report Lemma 2 | **FLAWED as stated** | Same as G3. It also omits "decode-only". |
| R3 | Report Lemma 3 | VERIFIED | |
| R4r | Report verification paragraph and abstract | FLAWED | "All proofs are machine-checked", "exhaustively" and "663 tests" are inaccurate (§8). |

## 2. docs/general_theory.md

### 2.1 Theorem 1

Proof audit:
1. Restricting to one LLM node per task with r_j = 0, unlimited KV and no tools gives jobs with fixed p_j. ✓
2. "Batch size 1" is a restriction on what the *scheduler may do*. The general problem lets the runtime batch,
   so the restricted problem is a special case only if the instance can force B ≤ 1. Examples are a batch-cap
   parameter, a KV capacity equal to one sequence (but the theorem also says "unlimited KV"), or a step time
   τ(B) = B·τ(1). As written, the "restriction" is not a restriction of instances. The proof is therefore
   incomplete for the model of §1, which has step time a + c·B.
3. With a common deadline, preemption is irrelevant and the problem is Knapsack (Karp 1972). ✓ This is weak
   NP-hardness: the problem has a pseudo-polynomial DP, and it must be encoded in binary.

Corrected (stronger) statement, now tested. Take the engine of §1 with unbounded batching and any given
rationals a ≥ 0, c > 0. All tasks are single nodes of n_j tokens, released at 0, with a common deadline.
1. Then the minimum time to finish a set S is exactly a·max_{j∈S} n_j + c·Σ_{j∈S} n_j. Brute force over every
   step sequence confirms it: `test_thm1_engine_makespan_with_batching_is_a_max_plus_c_sum`.
2. With c = 1, a = 1/(2Σn + 1) and deadline D + 1/2, S is on time iff Σ_{j∈S} n_j ≤ D.
3. So max on-time value equals Knapsack(D): `test_thm1_strengthening_hardness_survives_unbounded_batching`.
4. Hence the offline problem is NP-hard even with batching allowed. No batch-size restriction is needed.

Exact checks: `test_thm1_restricted_problem_is_knapsack` (120 cases, EDD brute force against a separately
written DP).

### 2.2 Lemma 2

**(i) is false for the discrete-step engine.** The engine of §1 runs atomic steps: a step with B sequences takes
a + c·B and gives each member exactly one token. A sequence cannot join a step in progress. The proof instead
treats each sequence as advancing continuously at rate 1/(a + cB) (the fluid relaxation), with batch changes at
arbitrary real times.

Counterexamples (exhaustive search over all discrete engine schedules; `discrete_engine_feasible`):
- **Common release.** Take a = 2, c = 1 (m = 2) and three 1-token jobs released at 0.
  - On M_2(1/(2a)) = M_2(1/4) each job needs 4 machine-seconds. McNaughton finishes them at (4, 6, 6).
  - No discrete engine schedule meets (4, 6, 6). A one-job step takes 3, a two-job step 4, a three-job step 5.
    Every ordered split of the jobs into steps misses (exhaustive search).
  - The deadlines (5, 7, 7) can be met.
  - The fluid engine does meet (4, 6, 6).
  - Tests: `test_lemma2_i_fails_for_the_discrete_step_engine_common_release`, and the strict xfail
    `test_lemma2_i_as_literally_stated_for_discrete_engine`.
- **With a release.** With the same a and c, take A (r 0, d 4) and B (r 1, d 5). Both are met on M_2(1/4). The
  discrete engine cannot meet both: either B waits for A's step to end, or A waits for B's release
  (`test_lemma2_i_fails_for_the_discrete_step_engine_with_release`).

Corrected statement:
- (i) holds for the **fluid** engine. Each running sequence advances at 1/(a + cB) tokens/s and the batch may
  change at any real time. This was checked exactly on 80 random machine schedules
  (`test_lemma2_i_fluid_engine_never_later_than_machines`).
- For the real step engine, an additive loss per job is unavoidable. In the counterexample it is a/2. A bound
  such as "at most one step, a + c·m ≤ 2a, per completion" is plausible but **not proved**. It needs its own
  argument, because rounding interacts with preemptions across jobs.

**(ii) is verified, including for discrete steps.** Within a step of length a + cB, the B tokens need a
machine-seconds each on M_m(1/a). Each a ≤ a + cB, and B·a ≤ (a/c)(a + cB). So McNaughton fits them inside the
step. Z3 confirms this. The exact construction was replayed on 60 random step schedules, including B > m
(`test_lemma2_ii_holds_for_discrete_steps`). Wording: completion times are "no later", not "preserved", because
McNaughton can finish a job's piece before the interval ends.

**Non-integer m.** Correct as stated. ⌊x⌋ ≥ x/2 already holds for every x ≥ 1, so "a/c ≥ 2" can be weakened to
"a/c ≥ 1". It fails for a/c < 1, where there are zero machines (Z3 with an integer floor,
`test_lemma2_floor_m_needs_only_a_over_c_at_least_one`). The measured m: `results/calibration/batch_profile.json`
gives a/c = 51.11, so ⌊a/c⌋ = 51. ✓

**Consequence (factor 2σ).** It holds in the fluid model. It inherits three unstated assumptions:
- decode-only steps;
- unlimited KV (B ≤ m sequences must fit, and suspended tasks' KV is not accounted for);
- results of early-finishing jobs held until their simulated time.

### 2.3 Lemma 3

Proof audit:
- Earlier issue means every tool ends no later. A fixed feasible schedule remains valid, since nothing forces a
  node to start when it is enabled. ✓
- Hidden assumptions, all stated in the doc: no tool capacity, durations and results independent of issue time,
  and the offline optimum free to delay.
- Two further assumptions:
  - Issuing costs neither GPU time nor KV memory.
  - A returned result can be held at no cost.

Exact checks:
- Feasibility is monotone in tool end times: 60 DAG instances on 1 machine and 15 on 2 machines
  (`test_lemma3_*`).
- The doc's "online need not be monotone" caveat is confirmed with an explicit instance. Shortening a tool from
  1/2 to 0 makes EDF at speed 1 miss (`test_lemma3_limit_online_edf_is_not_monotone`).

## 3. docs/open_problem.md

### 3.1 Independent exact offline solvers (cross-validation of §1)

- `offline_1m_by_orders` (one machine) tries every static priority order of the GPU nodes.
  - Exactness: take a feasible schedule with node windows [ρ_v, C_v] and run static priority by C_v. Induct on
    priority. A node's predecessors have smaller C, so by induction they finish by their C, and the node's
    replay release is ≤ ρ_v. The top-j nodes form a preemptive one-machine instance in which the original
    schedule is feasible, and on that instance EDF by C_v is optimal.
  - Orders that put a descendant above its ancestor are redundant and skipped.
- `offline_mm_lp` (m machines) is a Z3 linear-real-arithmetic encoding.
  - It uses 2n variable breakpoints and per-interval machine-time variables, with an
    "x = 0 ∨ interval ⊆ [S_v, E_v]" disjunction.
  - It is complete: take the breakpoints from a feasible schedule.
  - It is sound: every SAT model is turned into explicit McNaughton pieces, and `validate_schedule` replays them
    exactly. An assertion fails otherwise.
- The two solvers agree on 150 random instances (scratch). The project's Horn-condition solver agrees with them
  on 30 more (`test_crosscheck_project_offline_solver_and_simulator`). On those instances the project's simulator
  produces finish times identical to mine.

### 3.2 Fact A

The closed form (1+2ε)/s + 10 − 2ε is exact. But "> 10 **iff** ε < 1/(2s)" is false.
- The correct condition is ε < 1/(2(s−1)) for s > 1, and every ε for s = 1.
- Example: s = 2, ε = 1/3 ≥ 1/(2s). EDF still misses: B finishes at 61/6 > 10.
- Z3 proves the correct iff and refutes the stated one (`test_factA_iff_condition_in_doc_is_wrong`).
- The project test is named "iff" but checks only one direction.
- The conclusion (unbounded speedup) is unaffected.

### 3.3 Theorem 2, Cor 2′, Thm 3, Prop 0

**Thm 2. Audit:**
- The adversary is deterministic against a deterministic ALG, so it defines one fixed instance. The run is
  consistent because every tool before the last one is revealed as 0.
- Ties (several tools issued at once) are handled by issue count.
- An unfinished first node by D misses anyway. The realized instance then has all tools 0. It is feasible
  (mq unit jobs fit by q), and it has slack because q ≥ 1 + δ.
- Online: the last first node finishes at ≥ q/s. The miss condition is q(1/s − δ/(1+δ)) > 1. ✓
- Offline packing: (q − 1) + (m − 1)q = mq − 1. ✓
- Slack is exactly δ for the long task. ✓

Z3 (with hypothesis-necessity checks): `test_thm2_z3_algebra_with_hypothesis_necessity`.

Exact runs:
- 8 policies (EDF, FIFO, LIFO, SRPT, CP-EDF on the known upper bound, processor sharing, random static, and a
  non-work-conserving "lazy" policy) × m ∈ {1,2,3} × 8 (δ, s) pairs, including s at 0.95 of the threshold.
- Every run misses. The realized instance has slack and an explicit, validated offline schedule. When mq ≤ 6 it
  is also confirmed by the LP.

Sharpness: at s = 1 + 1/δ − 1/50, EDF is beaten. At s = 1 + 1/δ no instance exists, and EDF meets every instance
of the family for q ≤ 13 and every choice of long task (`test_thm2_threshold_is_sharp_both_ways`). This test
fails if the threshold is mutated to 1 + 1/(2δ).

**Cor 2′.** The coupling argument is correct. The index is independent of the seed (oblivious adversary), and
the run coincides with R until task i's tool is issued. It implicitly needs k ≥ 1 + δ, so that L ≥ 0 and the
short tasks have slack. The bound was verified exactly for 7 deterministic policies and their uniform mixture.

**Thm 3.** Correct. The l-th offline task's condition is monotone in l because s ≥ 1, so l = g is binding. Z3
shows that the g-bound is needed. Exact runs: 7 policies × 4 speeds, with offline subsets validated.

**Prop 0.** Correct: replay the offline optimum. But "feasibility is strongly NP-hard by Ridouard et al. Thm 1"
misapplies the source. Ridouard Thm 1 is stated for *periodic* tasks with implicit deadlines (3-PARTITION over
the hyperperiod). The one-shot, common-release model here is the master–slave problem of the OR literature,
which Chen et al. ECRTS 2019 §4 review. Cite that, or drop the complexity remark.

### 3.4 Theorem 4 (main positive result)

Line-by-line audit:
- (i) uses only r_i < r_j ⇒ d_i > d_j and the priority order. It gives W(S) ≤ m(d_k − r_k). ✓ Only this counting
  consequence of offline feasibility is used.
- (ii) Path lemma.
  - v_q exists (a minimal unfinished node), and the intervals partition [r_k, d_k]. ✓
  - A tool's interval is at most t, because tools start at enabling. ✓
  - A ready GPU node that is not running means m machines are busy with priority ≤ (d_k, k). This includes other
    nodes of k, which are in S. ✓
  - d_k − r_k ≤ T + P/s + |Y| ≤ C_k + |Y|. ✓
- (iii) **Gap.** "k is unfinished, so m·s·|Y| < W(S)" does not follow when k's unfinished node v_q is a TOOL. All
  of k's GPU work may then be done, and only "≤" holds. With "≤", the written algebra is satisfiable exactly at
  s = (1+δ)/δ. Z3 finds a model in `test_thm4_proof_counting_step_gap_and_repair`. The project's
  `test_thm4_z3_algebra` assumes `Wdone < Wtot`, the unjustified premise.
- **Repair** (Z3: both branches UNSAT; each needs its hypothesis):
  - If the path contains a GPU node, then P > 0 and s > 1 give P/s < P, so |Y| > (d_k − r_k) − C_k strictly.
    Then m·s·|Y| > m(d_k − r_k) ≥ W(S), a contradiction.
  - If the path is all tools, then d_k − r_k ≤ T ≤ C_k < (1+δ)C_k ≤ d_k − r_k. This uses δ > 0 and
    d_k − r_k > 0.
  - So the theorem holds at s ≥ (1+δ)/δ, including equality. It needs p_v > 0 for GPU nodes, as the model
    states.

Exact evidence:
- 1200 planted instances at exactly s = (1+δ)/δ:
  - m ∈ {1,2,3}; δ ∈ {1/4, 1/3, 1/2, 1, 2, 3};
  - common release or the release-order condition;
  - zero-length tools; tasks that end in a tool;
  - random offline schedules with idling;
  - every offline schedule replayed explicitly.
- 40 non-planted minimum-slack instances, decided by the exact LP (m ∈ {1,2}).
- Local search that maximizes lateness:
  - in tests: 15 seeds at the threshold, and a non-vacuity check at speed 1;
  - in scratch: 300 seeds × 300 steps at the threshold found **0** violations (worst score −0.0044). The same
    search at 0.95 × threshold found 24 failing seeds, so it can find real failures near the threshold.
- The release condition is needed: R4.

### 3.5 Refutations and family conjectures

- **R4.** Transcribed independently from the document text; it matches the code
  (`test_crosscheck_project_r4_r5_constructions_match_doc_text`).
  - Every task has slack ≥ 1.
  - EDF misses at 2, 21/10, 9/4, 23/10 and 12/5. K finishes at 734/9 at speed 9/4.
  - It meets at 5/2 and above.
  - It is feasible at speed 1 by both exact solvers. CP-EDF meets it at 9/4.
  - The 5/2 member misses at 2, 9/4, 12/5 and 5/2, and meets at 51/20 and 13/5.
  - C4 algebra ✓.
- **R5.** CP-EDF misses for every s < 321/221 ≈ 1.4525 and meets for s ≥ 321/221. The doc's tested speeds are all
  below this. EDF meets the instance at speeds 1, 6/5 and 29/20, and the offline schedule is feasible.
  Correction: "zero-slack" is approximate. T0 has slack (d − C)/C = 1/710 and T1 has 105/601. The
  general-release search instance at 6/5 was reproduced. C5 algebra ✓.

### 3.6 Prop 5 and Cor 5′

- The window reduction is correct.
- If a task has a tool node that is incomparable to its GPU node g, the condition r_j + t_u ≤ d_j must be added.
  It does not depend on the GPU, so the reduction becomes "feasible iff the windows are feasible and these side
  conditions hold". The doc's phrase "tool DAGs before and after" excludes such nodes, but the general model
  allows them.
- Checks:
  - On one machine, CP-EDF equals the exact optimum on 60 random instances, with side tools included.
  - On two machines, DAG feasibility equals window-job feasibility on 20 instances (LP).
- Cor 5′ algebra ✓. The hypothesis O ≥ 0 is needed: Z3 finds a model without it.

### 3.7 Theorems 6, 6b, 7 and Props 8, 9

- **Thm 6.** Correct in the fluid, decode-only model. The proof also assumes:
  - unlimited KV (suspended tasks keep KV, and the batch of up to m sequences must fit);
  - completion times only "no later" through Lemma 2(ii), not "identical".

  The project only checks the rate algebra. The new end-to-end test runs 40 instances:
  - a random unit-engine schedule, with batch sizes that may exceed m;
  - an exact M_m(1/a) schedule built from it;
  - EDF at (1+δ)/δ meeting every deadline;
  - the λ = 2s engine following the simulation, every node finishing no later
    (`test_thm6_engine_lift_end_to_end`).
- **Thm 6b.** Verified with 4 engine batching policies and λ ∈ {1, 2, 4}. Discrete-step schedules are a special
  case of fluid ones, and the offline engine schedule is integral, so the lower bound also holds for the real
  step engine. k = 209 at λ = 4 checks out.
- **Thm 7.** The proof is correct (scaling plus Thm 4 at offline speed 1 + β). Checks:
  - 300 planted instances at the worst allowed recompute;
  - slack preserved at speed 1 + β;
  - the inflated instance feasible at speed 1 + β by the exact LP (12 instances).

  Correction: §0 says the policy "costs exactly a (1+β) speed factor". Only an upper bound is shown.
- **Prop 8.** Verified.
- **Prop 9.** The arithmetic is right. The framing needs a fix. "Never evict" achieves zero recompute only by
  delaying B's request. In the paging model behind LRU and Landlord a request must be served on arrival, so those
  policies would evict A and meet both deadlines. The correct claim is that minimizing recompute cost, when
  requests may be deferred, is not aligned with deadlines. The project's test only evaluates constants. The new
  test simulates both executions.

## 4. Test-quality audit of the existing suites

Mutation runs (`scratchpad/verify/mutate.py`) used a mutated copy of `open_problem.py` or of the test file:

| Mutation | Failing tests (of 4666) | Caught? |
|---|---|---|
| M1 tools sped up by s | 111 | yes |
| M2 "EDF" key replaced by FIFO | 231 | yes |
| M3 offline solver without Horn constraints | 104 | yes |
| M4 `r2_parameters` without k ≥ 1+δ | 0 | **no** (never binding for the tested pairs) |
| M5 planted generator without the release-order fix | 1112 | yes |
| M6 Thm 4 test speed changed to 1 + 1/(2δ) | 2 of 3000 planted | **barely** |
| M7 Thm 4 test speed changed to 1 | 282 of 3000 | weakly |
| M8 Thm 7 test without the (1+β) factor | 7 of 1000 | **barely** |
| M9 random exact Thm 4 test at speed 1 | 18 of 300 | weakly |
| M10 Thm 2 long tool halved | 72 | yes |

Findings:
1. The random property tests of upper bounds (Thm 4, Thm 7) almost never detect an over-claimed speed. Evidence
   for the exact threshold comes only from the Thm 2 adversary. The new planted tests share this limitation (a
   1 + 1/(2δ) mutation passes them). `test_thm2_threshold_is_sharp_both_ways` catches it.
2. `test_thm4_z3_algebra` encodes the unjustified premise `Wdone < Wtot` (§3.4).
3. `test_factA_z3_edf_formula_exceeds_deadline_iff_eps_small` checks one direction of a false iff.
4. `test_thm6_engine_rate_and_lower_bound_algebra` is algebra only. No simulation of the Thm 6 policy existed.
5. `test_prop9_*` only evaluates constants. There is no memory model.
6. In `test_lemma2_ii_total_rate_below_m_machines_at_speed_1_over_a`, the second assertion (m = a/c ⇒ m/a = 1/c)
   is a tautology.
7. `test_lemma2_i_engine_finishes...` uses floats, a 2·10⁻³ time step and a 4·10⁻³ tolerance. It tests only the
   fluid model, which is why it missed §2.2. The report calls it "exhaustive".
8. Lemma 2(ii) had no test that engine schedules are realizable (only generic McNaughton). Lemma 3 was tested only
   on single-node, non-preemptive tool→LLM tasks.
9. The open-problem test count (4666, about 13 s) is accurate. General theory has 648 tests, not the report's 663.

Mutations of the independent suite (`mutate_mine.py`):
- tools sped up: 618 failures;
- EDF → FIFO: 89;
- LP without window constraints: 70;
- Thm 2 sharpness threshold mutated: 4;
- Thm 4 planted at 1 + 1/(2δ): 0, as explained above.

## 5. Z3 checks

Every check proves the negation UNSAT and shows at least one hypothesis is necessary (SAT when dropped):
- Lemma 2: (i), (ii), the discrete (ii), c > 0, and the integer floor.
- Fact A: the exact condition.
- Thm 2: online, offline and slack algebra; q exists iff sδ < 1 + δ.
- Thm 3.
- Thm 4: the gap and both repaired branches.
- C4 and C5 algebra.
- Cor 5′.
- Prop 9: "every speed".

## 6. Citation checks

| Source | Claim relied on | Finding |
|---|---|---|
| Karp 1972, [doi:10.1007/978-1-4684-2001-2_9](https://link.springer.com/content/pdf/10.1007/978-1-4684-2001-2_9.pdf) | Knapsack NP-complete | ✓. Karp's KNAPSACK is the subset-sum form. JOB SEQUENCING, which is exactly 1‖Σw_jU_j, is itself one of Karp's 21 and is the more direct citation. |
| Lenstra–Rinnooy Kan–Brucker 1977 (preprint BW 43/75, [CWI](https://ir.cwi.nl/pub/9725/9725D.pdf)) | 1\|r_j\|ΣU_j strongly NP-hard | The OCR'd preprint shows KNAPSACK reductions to n\|1\|r_j\|L_max and to the unit-weight ΣU_j problem with releases (ordinary NP-hardness). It also shows KNAPSACK → n\|1\|\|Σw_jU_j, which supports Thm 1. Strong NP-hardness (via 3-PARTITION for L_max with releases) is standard but was not seen in the readable text: **secondary**. |
| McNaughton 1959, [doi:10.1287/mnsc.6.1.1](https://pubsonline.informs.org/doi/fpi/10.1287/mnsc.6.1.1) | wrap-around rule | Bibliography ✓. Text not readable. Not load-bearing: the exact implementation and check here verify the needed property directly. |
| Baruah et al., RTS 4(2) 1992, [dblp](https://dblp.org/rec/journals/rts/BaruahKMMRRSW92.html) | online 1/4 | ✓ by the abstract. No online algorithm beats 1/4 (uniform value density), and TD1 achieves it. |
| Koren–Shasha, SIAM J. Comput. 24(2) 1995 | D^over 1/(1+√k)² | The paper is confirmed as optimally competitive. The formula was not seen in any accessible text (HAL blocked; [DTIC thesis](https://apps.dtic.mil/sti/html/tr/ADA598400/index.html)): **secondary**. |
| Lawler 1990, Ann. OR 26:125–133, [doi:10.1007/BF02248588](https://link.springer.com/doi/10.1007/BF02248588) | 1\|r_j,pmtn\|ΣU_j polynomial | ✓ by the abstract/review. The DP is pseudo-polynomial in the weights and polynomial for unit weights. |
| Ridouard–Richard–Cottet RTSS 2004, [PDF](https://www.lias-lab.fr/publications/7082/2004-IEEE-Ridouard.pdf) | Thm 1, 2, 6 | Thm 6 and its instance ✓ (τ1: C = 2k, D = 4k − 1; τ2: 1 → 4k − 2 → 1, D = 4k; integer k > 1). **Thm 1 is about periodic tasks**, so Prop 0's use of it is wrong. Section 4.3 and the conclusion *assert* that "using a k-speed processor cannot help" in general, but prove it only for EDF (relevant to novelty, §7). |
| Chen RTSS 2016, [PDF](https://ls12-www.cs.tu-dortmund.de/daes/media/documents/publications/downloads/2016-JJ-Suspension-Complexity.pdf) | Thm 5, §4.3, Thms 6–7 | ✓ verbatim. Thm 5 covers FP, EDF, SAEDF, EDZL, SAEDZL and SULLF (dynamic suspension). §4.3 says no lower bound could be built for suspension-aware LLF. Thm 6 (S_i ≤ λC_i) and Thm 7 (σ) are RM results. |
| Chen et al., RTS 2019 survey | open problem | Not re-read. The quotes in the doc are consistent with the ECRTS 2019 text below. |
| Eberle STACS 2024, [arXiv 2310.16697](https://arxiv.org/abs/2310.16697) | O(1/ε) weighted throughput with slack | ✓ by the abstract. It covers single and unrelated machines, slack (1+ε)p, and no speed augmentation. Non-migratory, compared against a migratory optimum. |
| Phillips–Stein–Torng–Wein STOC'97 / Algorithmica 2002 | EDF at speed 2 − 1/m | The paper is confirmed. "2 − 1/m" was seen only in a secondary summary ([RTSOPS 2018](https://rtsops2018.loria.fr/wp-content/uploads/2018/07/RTSOPS_10.pdf)): **secondary**. |
| JITServe, [arXiv 2504.20068v3](https://arxiv.org/html/2504.20068v3) | Thm 4.1 (1/8.56) and Thm D.3 | ✓ Thm 4.1 states 1/8.56. Thm D.3 ends with r ≈ 1/8.557, which is "solv[ed] … using numerical analysis under our experimental setting". So it is **not a universal constant**. The doc should say so. The proof does mention a batch size b (the doc says "no batching"). It has no step-time model, tools or KV. Preemption needs a (1+δ) gain ✓. Multi-slot "immediate" ✓. |
| Horn 1974 | interval condition | Not checked. Not load-bearing, since every feasible answer is certified by replay. |

## 7. Novelty assessment

- **Thm 2.** No source found states the all-deterministic-algorithms version with the 1 + 1/δ threshold for
  one-shot jobs. But two works are close.
  - Ridouard et al. 2004 (§4.3, conclusion) informally claim that speed does not help with self-suspensions, and
    prove it only for EDF. Thm 2 makes that claim rigorous for every deterministic algorithm.
  - **Chen, Hahn, Hoeksma, Megow, von der Brüggen, "Scheduling Self-Suspending Tasks: New and Old Results",
    ECRTS 2019** ([PDF](https://drops.dagstuhl.de/storage/00lipics/lipics-vol133-ecrts2019/LIPIcs.ECRTS.2019.16/LIPIcs.ECRTS.2019.16.pdf)).
    This paper is **not cited** in the doc.
    - It studies the *frame-based* setting (common release, common deadline), which is exactly the setting of
      Thm 2.
    - Thm 4.14 is an information-theoretic lower bound of the same adversary style: no deterministic algorithm
      gets below speed 2 in the hybrid model.
    - Thms 4.12 and 5.9 show that LSF (longest suspension first) has speedup 2 (uniprocessor) and 3 − 1/m
      (multiprocessor) when suspensions are known. That bears directly on the doc's I2 frontier: Prop 0, R5, C2
      and the claim that CP-EDF fails at 29/20.

  Verdict: a natural, probably folklore-level extension. The doc's own assessment is fair. It must cite
  ECRTS 2019.
- **Thm 4.** The proof is the standard critical-path / busy-interval argument used for global EDF on DAG tasks
  (Bonifaci et al., Li et al.: 2 − 1/m). Here non-sped-up tool nodes are absorbed by slack. The pairing with
  Thm 2 that gives the exact s* = 1 + 1/δ is clean but small. No prior statement of this exact pair was found
  (two targeted searches found none). This does not establish novelty.
- R4, R5, Props 5, 8, 9 and Thm 7 are straightforward. Thm 6 composes Lemma 2 with Thm 4.

## 8. Model validity (does the formal model describe vLLM?)

- **Discrete steps.** vLLM runs atomic iterations, so Lemma 2(i) and Thm 6 hold only for the fluid relaxation.
  Lower bounds (Thm 2, 3, 6b) transfer. Upper bounds need an additive per-completion rounding term that has not
  been proved.
- **Mixed prefill/decode.** With chunked prefill, step time depends on prefill tokens and on total context length
  (attention reads every KV token), not only on B. The measured a + cB fit uses about 500-word prompts and
  128-token outputs. Lemma 2 and Thm 6 are decode-only. Every agent LLM node starts with a prefill, which can be
  large (about 11k tokens in the report's workload), so the model does not cover the dominant final-call cost in
  the report's own experiment.
- **KV memory.** The calibration excludes B = 256 because it exceeds KV capacity (159,904 tokens), so memory
  binds in practice. Thms 4 and 6 assume unlimited memory. Thm 7 only inflates work. No result is
  memory-competitive (Prop 8).
- **Release-order hypothesis of Thm 4.** In online multi-tenant serving (the report's Poisson simulation, with
  deadlines proportional to unloaded time), later arrivals almost always have *later* deadlines. That violates
  r_i < r_j ⇒ d_i > d_j for nearly every pair. Thm 4 (and Thm 6, 7) therefore covers only batch, common-release
  workloads. R4 shows EDF needs more than 5/2 at δ = 1 once arrivals are allowed.
- **Implemented policies differ from the analysed ones.** The system sets vLLM request priorities. It does not
  simulate m-machine EDF, cap the batch at m, or hold results until simulated completion (Thm 6). The guarantees
  do not certify the built system.
- **Tools.** Off-GPU and parallel is realistic for search and API calls. Rate limits, cost and timing-dependent
  results break Lemma 3, as the doc notes.
- **What transfers.**
  - Transfers as stated: Thm 1 (and its strengthening), Lemma 3 under its assumptions, the lower bounds Thm 2,
    Cor 2′, Thm 3 and 6b, and the negative Props 8 and 9.
  - Transfers only after adding prefill, discrete-step rounding and memory: Lemma 2(i), Thm 4, Thm 6 and Thm 7.

## 9. Corrections required in report/main.tex

1. **Lemma 2 (Batching → parallel machines).**
   - Add "in the fluid model (each running sequence advances continuously at 1/(a + cB) tokens/s; decode-only
     steps; unlimited KV)". Part (i) is false for the discrete step engine (counterexample in §2.2).
   - Replace "no completion later" in (i) with the fluid qualifier.
   - (ii) may stay unqualified.
   - The abstract's "continuous batching is equivalent, up to a factor of 2 in speed, to … identical parallel
     machines" must carry the same qualifiers.
2. **Theorem 1.** "Batch size 1" is not an instance restriction in a model where the runtime chooses batches.
   Either add a batch-cap parameter to the model, or (better) state: "NP-hard even with unbounded batching (step
   time a + cB), no tools, unlimited KV, deterministic lengths, single-node tasks and r_j = 0", with the proof in
   §2.1. Say "NP-hard (in the ordinary sense)".
3. **"All proofs are machine-checked"** (abstract) and **"Every algebraic step … is proved with Z3 … Theorem 1,
   McNaughton's rule, Lemma 2(i) and Lemma 3 are checked exhaustively"**. Replace both with "the algebraic
   inequalities are checked with Z3, and the combinatorial claims are tested exactly on small random instances".
   - Nothing is formally verified.
   - The Lemma 2(i) test uses floating-point time stepping in the fluid model.
   - The Lemma 3 test covers single-node tasks only.
4. **"663 tests pass"**: `tests/test_general_theory.py` has 648. Regenerate the number.
5. The consequence sentence ("C-competitive … with augmentation 2σ") must inherit the fluid, decode-only and
   unlimited-KV qualifiers.
6. Plan item (2): "LRU-style and Landlord policies are 2-competitive with twice the memory" is correct as a caching
   statement. But the project's own Prop 9 shows that cost-competitiveness gives no deadline guarantee. The plan
   should say this, or drop the implication that caching competitiveness composes.
7. Lemma 3 statement: correct. Optionally add "issuing costs no GPU time or KV memory".
8. The other formal citations in the report (Lenstra et al., Baruah et al., Koren–Shasha, Kalyanasundaram–Pruhs)
   are consistent with the sources at abstract level.

## 10. Corrections required in docs (for the authors; not applied)

- general_theory.md:
  - Lemma 2: fluid qualifier on (i); "preserved" → "no later" in (ii); a/c ≥ 1 suffices.
  - Thm 1: batch-cap issue, plus the strengthening.
- open_problem.md:
  - Fact A: replace the iff with ε < 1/(2(s−1)).
  - Thm 4 step (iii): replace with the two-branch argument of §3.4, and fix `test_thm4_z3_algebra`.
  - Prop 0: replace the Ridouard Thm 1 citation.
  - R5: "near-zero slack (1/710)"; the CP-EDF threshold is exactly 321/221.
  - Cor 2′: add k ≥ 1 + δ.
  - Prop 5: side tools.
  - Thm 6: fluid and unlimited-KV qualifiers; "no later" instead of "identical"; add a simulation test.
  - §0: "at most (1+β)", not "exactly".
  - Prop 9: reframe.
  - §4: JITServe's constant is numerical, for their setting; cite Chen et al. ECRTS 2019.
