# When and why KV preloading of tool results helps: theory, checks, and comparison with data

Date: 2026-11-07. Scope: the fan-out agent of `src/deadline_agent/multiagent.py` (plan → n parallel tool calls →
final answer) served by one GPU. *Preloading* means: when a tool result returns and others are still outstanding,
send a 1-token request whose prompt is the evidence log so far, so the prefix cache holds its KV when the final
call arrives. No existing file was modified.

Artifacts:
- `src/deadline_agent/preload_theory.py`:
  - an exact event-driven work-model simulator (`simulate_work`; exact with `Fraction` inputs);
  - closed forms and approximations (`pk_wait`, `predict_ps`, `predict_fcfs`, `breakeven_eps_ps`);
  - a step-level continuous-batching engine simulator with chunked prefill, a prefix cache and LRU KV
    (`simulate_engine`);
  - loaders for the local data.
- `tests/test_preload_theory.py`: 35 tests, about 5 s. Hypothesis runs 300 examples per property. Each test is
  cited below as **[T: name]**.

Status labels:
- **proved**: full proof here, plus the named machine check;
- **known**: classical, with a citation verified by web search (title, venue and year confirmed; the content
  check is noted);
- **standard**: a folklore argument, stated plainly;
- **approximation**: a heuristic formula, not a theorem;
- **conjecture**;
- **refuted**: with a counterexample;
- **empirical-only**.

Wording: "exact check" means exact rational simulation, not a formal proof. Hypothesis tests use random integer
instances with:
- 1–4 tasks;
- 1–3 tool calls per task;
- r, D ∈ {0..6}, P ∈ {0,1,2}, w ∈ {0..3}, f ∈ {0,1,2};
- ε ∈ {0, 1/4, 1/2}.

"Exhaustive" is used only for the stated grids.

## 0. Results at a glance

| # | Statement (short) | Status |
|---|---|---|
| L1 | Reflection formula for a work-conserving server | known/standard; exact check [T: lemma1] |
| L2 | FCFS and PS completion times are monotone in job sizes | standard (proof given); exact check [T: lemma2] |
| T1 | Exogenous releases: preloading never lowers cumulative completed work; makespan ≤ off + ε·N | proved (restatement of reflection monotonicity); [T: thm1] |
| R1 | Same claim with plans as GPU work (endogenous releases): false even with ε = 0; per-task harm / preload work is unbounded | refuted, exact counterexamples [T: ref1_*] |
| T2 | FCFS, same priority: harm to task j ≤ H_j + ε·N(R_j), tight | proved; [T: thm2_*] |
| T3 | Preemptive lowest-priority preloads, merged into the final when it is ready: no task is ever later, any ε ≥ 0 (FCFS or PS among the rest) | proved; [T: thm3_*], exhaustive grid |
| R2 | T3 fails when plans are ordinary GPU work | refuted [T: ref2] |
| T3b | Clairvoyant JIT (priority = need time): task j is later by at most ε·(preloads of tasks needed no later than j) | proved for exogenous plans; [T: thm3b] |
| P4 | Isolated task: saving = preloaded work − own residual at the last return (never negative in the work model) | proved; [T: prop_isolated] |
| P5 | FCFS identity S_j = (W_j − F_j) + V_off(R_j) − V_on(R_j); width discriminant E[S∣wide] − E[S∣narrow] ≤ E[preloaded work] | proved (P5b under stationarity); [T: prop_fcfs_identity, z3] |
| A1 | M/G/1-PS approximation: saving ≈ W/(1−ρ) − F/(1−ρ′) for wide tasks, ≈ 0 for narrow ones | approximation; uses a known formula |
| T4 | on / low / JIT differ for task j by at most Π_j, the other tasks' unprocessed preload work at R_j | proved (exogenous plans, ε = 0, preemptive); [T: thm4_*] |
| K | Preloaded-KV eviction: necessary condition (any policy) and sufficient condition (LRU) | proved (elementary); trajectory-checked [T: lemma_k] |
| T6 | Overhead: (a) no isolated harm; (b) PS break-even ε < w(1−ρ)/(λx); (c) any ε_GPU > 0 shrinks the stable region | (a) proved; (b) algebra proved (Z3) inside approximation A1; (c) proved via Loynes |
| D | Data: FCFS work-queue mechanism rejected by P5b at 0.45 and 0.6/s; zero-parameter A1 inside all four measured CIs; mechanistic engine model right in magnitude but wrong in the wide/narrow split at 0.6 | empirical; [T: data checks] |

**Headline.** Three points:
1. With exogenous tool timings, two schedules make preloading safe per task, for any overhead:
   - lowest-priority preloads that are merged into the final at need (T3);
   - clairvoyant just-in-time preloads, which are exact when ε = 0 (T3b).
2. Same-priority preloading is safe only in aggregate (T1). Its per-task harm is bounded by T2.
3. In the data, the growth of the saving with load is concentrated in the tasks that have something to preload.
   That is the signature of the critical-path work being *stretched* by sharing (≈ 1/(1−ρ)), not of shorter
   queues from splitting bursts (P5b rejects the pure FCFS-queue mechanism).

## 1. Models

**M1 (work model).** One server of rate 1 does *work* (GPU seconds). Task j has:
- arrival r_j and plan work P_j;
- n_j tool calls with durations D_ji, issued when the plan completes;
- result pieces w_ji;
- tail f_j.

Order the returns by (D_ji, i). Write R_j for the last return. The n_j − 1 earlier results are the
*preloadable pieces*. **F_j** = (last result's piece) + f_j is the final's minimum work, and
**W_j** = Σ_i w_ji + f_j.

Modes:
- **off**: release W_j at R_j.
- **on**: release each preloadable piece at its return as a job of work w + ε, in the same class as plans and
  finals. Release F_j at R_j.
- **low**: as on, but pieces are in a lower preemptive class.
- **jit**: preemptive priority by a key. Plans and finals use their release time. A piece of task k uses R_k
  (the clairvoyant need time).

**Merge.** At R_j the task's unfinished pieces are cancelled. Their remaining *useful* work is added to the final
job, and their unspent overhead disappears. This is what the non-blocking code does: the pending preload is
cancelled and its computed blocks stay cached.

Plans:
- **exo**: the plan is a pure delay; releases are exogenous.
- **top**: plans are GPU work at preemptive top priority. Plan completions then do not depend on preloading.
- **work**: plans are ordinary class-0 work; releases are endogenous.

Disciplines: **fcfs** (preemptive priority by (class, key, seq); within a class this is FCFS) and **ps**
(egalitarian sharing within the top non-empty class). With ε = 0 the total work is identical in all modes. A
zero-work job also waits its turn.

**M2.** M1 with Poisson arrivals and i.i.d. task marks (used in §3).

**M3 (engine, `simulate_engine`).** Discrete steps of a continuous-batching engine:
- every request past its prefill decodes one token per step;
- the token budget b minus the decode tokens goes to prefilling requests, either in (priority, admission) order
  (`fcfs`) or split equally (`fair`);
- step time is a + c·B_decode + Σ chunk costs, where a chunk of x new tokens after ctx cached tokens costs
  α x + κ x (ctx + x/2);
- request latency is δ (half on the way in, half on the way out);
- each task has its own prefix cache with full 16-token blocks, and the final recomputes the partial block;
- optional KV capacity with LRU, tail-first eviction, and recompute-preemption when nothing can progress.

Agent modes:
- `off`;
- `block`: preload k is sent at max(k-th return, completion of preload k−1), and the final is sent after the
  chain. This is the version that produced `results/real/preload_contention`; its model_calls are 1 + (n−1) + 1.
- `nb`: at most one preload in flight, cancelled when the final is ready (current `multiagent.py`).

Either mode can run at normal or lowest priority.

## 2. Dominance

**Lemma 1 (reflection) [known/standard].** Take a rate-1 work-conserving server, empty at t0, with A(t) the
cumulative work released in [t0, t]. Then the completed work is
D(t) = min(A(t), inf_{t0≤s≤t} [A(s−) + t − s]),
for *every* work-conserving discipline.

This is the continuous form of Lindley's recursion (Lindley, "The theory of queues with a single server",
Math. Proc. Cambridge Phil. Soc. 48(2):277–289, 1952). In network calculus it is the statement that a
work-conserving rate-1 server has output A ⊗ β with β(t) = t (Le Boudec & Thiran, *Network Calculus*,
LNCS 2050, Springer 2001). Bibliographic details were verified; for Le Boudec–Thiran, the exact theorem number
was not.

*Proof sketch.* D(t) ≤ A(t). For any s, D(t) ≤ D(s) + (t − s) ≤ A(s−) + (t − s). Equality holds at s = the
start of the busy period containing t, since the server works at rate 1 on (s, t]. ∎

[T: lemma1_reflection_formula_equals_simulated_cumulative_work]

**Lemma 2 (size monotonicity) [standard; proof given].** Fix release times and a capacity process c(t) ∈ [0,1]
that is common to both systems. Under FCFS or PS, shrinking job sizes (x′ ≤ x pointwise) never delays any
completion.

*Proof (PS).* Suppose not. Among the jobs with C′_k > C_k, let j have the smallest C_j.
- For u < C_j, every job present in the primed system is also present in the unprimed one. A job present in
  the primed system but already gone in the unprimed one would be a violator with smaller C.
- So N′(u) ≤ N(u), and j's rate c(u)/N′(u) ≥ c(u)/N(u).
- Hence j's attained service by C_j is ≥ x_j ≥ x′_j, so C′_j ≤ C_j. Contradiction.

*Proof (FCFS).* C_j = inf{t ≥ R_j : ∫ c over [σ, t] covers the work ahead of and including j released since σ},
which is monotone in that work. ∎

[T: lemma2_completion_times_monotone_in_job_sizes]

**Theorem 1 (aggregate dominance, exogenous releases) [proved].** Assume plan_mode exo or top. Take any
work-conserving discipline (any of the modes, merge or not) and any ε ≥ 0. Then:
- (i) D_on(t) ≥ D_off(t) for all t, where D counts all completed work including overhead;
- (ii) the useful completed work satisfies D_on,useful(t) ≥ D_off(t) − ε·N(t), where N(t) is the number of
  preloads released by t;
- (iii) makespan_on ≤ makespan_off + ε·N[s*, ∞), where s* is the start of the last busy period of the off system
  attaining the sup below and N[s*, ∞) counts the preloads released from s* on. In particular
  makespan_on ≤ makespan_off when ε = 0.

*Proof.* Release times are schedule-independent. With "top", plans see a dedicated server, so their completions,
and hence all tool returns, are identical in both systems.

Let A_eff be the released work counting only overhead that is actually spent. Overhead removed by a merge at R_j
was never served, and it cannot change the busy/idle pattern before R_j (causality). So the trajectory of D_on
equals that of a work-conserving server fed A_eff.

Now A_eff,on(t) − A_off(t) = (useful piece work released by t of tasks with R_j > t) + (spent overhead) ≥ 0.
Lemma 1's right-hand side is monotone in A, which gives (i). Part (ii) follows because at most εN(t) of D_on is
overhead.

For (iii), the last completion of a work-conserving server is sup_s [s + A(∞) − A(s−)]. Then
A_on(∞) − A_on(s−) ≤ A_off(∞) − A_off(s−) + ε·N[s, ∞) − (useful piece work released before s) ≤
A_off(∞) − A_off(s−) + ε·N[s, ∞). ∎

Checks:
- [T: thm1_cumulative_work_and_makespan]: all breakpoints, FCFS and PS, exo and top, merge on and off;
- [T: thm1_makespan_bound_is_attained]: the ε·N term is attained.

**Refutation 1 (endogenous plans) [refuted].** With plans as ordinary GPU work, T1 fails even with ε = 0.

Makespan counterexample (`cex_endogenous_makespan`):
- A has a zero plan, a piece of work 1 returning at 0, and its last result at 1.
- B arrives at 1/10 with plan 1 and one tool of duration 10.
- Under FCFS the makespan goes 121/10 → 13. Under PS it goes 61/5 → 131/10.
- Mechanism: the piece is ahead of B's plan, so B's whole chain shifts.

Amplification (`cex_amplification(δ)`):
- A piece of work δ delays B's plan by about δ.
- B's final is then released just after C's plan instead of just before it.
- B ends 1 + 3δ/8 later, so harm / preload work ≈ 1/δ is unbounded. Checked for δ = 1/10, 1/100, 1/1000.

This is a timing anomaly in the spirit of Graham, "Bounds on multiprocessing timing anomalies", SIAM J. Appl.
Math. 17(2):416–429, 1969 (verified bibliographically). [T: ref1_*]

On the exhaustive grid of §2.1, 'on' with work plans shows a per-task harm of up to 4 but no makespan increase.
The makespan counterexample needs a long tool (D = 10), which is outside that grid.

**Theorem 2 (per-task harm, FCFS, same priority) [proved].** Assume exogenous releases, FCFS, ε ≥ 0, merge on or
off. Then

C_on,j − C_off,j ≤ H_j + ε·N(R_j),

where:
- H_j is the useful work of preload pieces released no later than R_j by tasks whose final is *behind* j's in
  FCFS order;
- N(R_j) is the number of preloads (of all tasks) released no later than R_j.

The bound is tight. In the tight instance (`cex_fcfs_on_per_task`), j has nothing to preload and k's piece of
work 2 returns δ before R_j. The harm is exactly 2 + ε − δ.

*Proof.* Jobs behind j never affect j, so C_j = sup_{s ≤ R_j} [s + A_⪯j[s, R_j]], where A_⪯j counts the work
released in [s, R_j] that precedes j's final. Without merges, compare on and off window by window:
- tasks k ≻ j contribute their pieces released in [s, R_j], each plus ε (that is H_j plus overhead);
- tasks k ⪯ j with R_k ≥ s contribute ε per piece released in [s, R_k], minus their useful pieces released
  before s;
- tasks with R_k < s contribute nothing in either system.

So the window difference is ≤ H_j + ε N(R_j). Take the sup.

Merges only move other tasks' residual work behind j or delete unspent overhead. Moving j's own residual into
j's final leaves C_j unchanged, because everything between them is ahead of j's final anyway. By Lemma 2, merges
never increase C_j. ∎

Checks:
- [T: thm2_fcfs_on_harm_bound], [T: thm2_tight_instance];
- an earlier draft of the bound that omitted the ε of tasks ahead of j was caught by this check.

Per-task dominance of 'on' fails under PS too: in `cex_ps_on_per_task`, j goes from 7 to 9.
[T: thm2_per_task_dominance_fails_under_ps_too]

**Theorem 3 (lowest-priority preloads are per-task safe) [proved].** Assume plan_mode exo or top, and that
pieces are in the lowest preemptive class with merge-at-need. Whether the discipline among the remaining work is
FCFS or PS, C_low,j ≤ C_off,j for every task j and every ε ≥ 0.

*Proof.*
- The low class never preempts, so the plans and finals of the low system evolve exactly as a system that
  contains only them.
- In that system the plans are identical to the off system, the finals have identical release times R_j, and
  the final sizes are F_j + U_j ≤ W_j, where U_j is the useful residual at R_j.
- The capacity left by top-class plans is the same in both systems.
- Lemma 2 gives C_low,j ≤ C_off,j.
- Overhead is spent only by the low class, or deleted by the merge, so it never reaches the other work. ∎

Checks:
- [T: thm3_low_priority_preloads_never_delay_any_task]: FCFS and PS, exo and top, ε ∈ {0, 1/4, 1/2}.
- [T: thm3_exhaustive_small_grid]: all 3240 unordered pairs from an 80-type grid; the same grid finds 'on'
  violations.

**Refutation 2 [refuted].** T3 fails when plans are ordinary GPU work (`cex_low_endogenous`):
- A's final shrinks from 10 to 1, so B's plan finishes 9 earlier.
- B's final (work 5) then lands just ahead of J's final.
- J is 9/2 later under FCFS and 1/2 later under PS.
- With top-priority plans the anomaly disappears.

[T: ref2_low_priority_with_endogenous_plans]

**Theorem 3b (clairvoyant JIT) [proved for exo; tested for top].** Use preemptive priority by key: plans and finals
by release time, a piece of task k by R_k, and assume distinct R's (or ties broken as in the off system). Then
C_jit,j ≤ C_off,j + ε·n_j, where n_j is the number of preloads of tasks with R_k ≤ R_j.

*Proof.* Only jobs with key ≤ R_j affect j. All of them are released by R_j, and they see a dedicated server.
- In the off system they are the finals W_k of the tasks with R_k ≤ R_j.
- In the jit system they are the same useful work released no later, plus at most ε n_j of overhead.
- By Lemma 1 applied to this subsystem, D_jit(t) ≥ D_off(t). At C_off,j at most ε n_j of the subsystem's jit
  work remains. ∎

[T: thm3b_clairvoyant_jit_dominance]. The real JIT uses *estimated* need times (expected remaining wait), so
priority inversions are possible and T3b does not apply to it directly.

**Per-discipline summary (exogenous releases):**

| Mode | Aggregate (T1) | Every task no later? |
|---|---|---|
| on, FCFS | yes (+εN) | no; harm ≤ H_j + εN(R_j), tight (T2) |
| on, PS | yes (+εN) | no (counterexample); no bound proved |
| low (preemptive, merge) | yes | **yes, any ε** (T3) |
| JIT (clairvoyant) | yes | yes up to ε·n_j (T3b) |

With endogenous plans every row fails (R1, R2).

### 2.1 Exhaustive grid (scratch run; reproducible with `search_dominance`)

The grid has r, P ∈ {0,1}; n ∈ {1,2}; D ∈ {0,1,2}; w ∈ {0,1,2}; f = 1. That is 360 task types and all 64,980
unordered pairs.

| Comparison | Worst per-task harm | Makespan increase |
|---|---|---|
| low vs off (FCFS exo, PS exo, FCFS top, FCFS exo with ε = 1/2) | 0 | 0 |
| JIT vs off (FCFS exo) | 0 | 0 |
| low vs off (FCFS work) | 0 | 0 |
| on vs off (FCFS exo; PS exo) | 2 | 0 |
| on vs off (FCFS work) | 4 | 0 |

The low-vs-off result with work plans on this grid does not contradict R2, which needs a longer chain.

## 3. Why the saving grows with load

**Proposition 4 (isolated task) [proved].** For one task alone in M1 (FCFS or PS, with merge),
S = C_off − C_on = (Σ of preloadable pieces) − U(R), where U(R) is the own useful residual at the last return.
So S ≥ 0 for any ε, and S equals all preloadable work when every piece finishes before R.

[T: prop_isolated_task_saving]. As λ → 0, the expected saving tends to the isolated value, since the probability
that another task overlaps tends to 0 and all supports are bounded (standard argument).

At the run's parameters, the isolated saving per 8-wide task is estimated from the cost model below at
**1.67 s** (cap 1400). The cap-1500 version predicts 1.77 s, against a measured single-task saving of
**1.61 s [1.49, 1.72]** (20/20 faster), so the model is about 10% high. The measured wide-task saving at
0.15/s is 1.60 s [0.68, 2.51].

**Proposition 5 (FCFS identity and width discriminant) [proved].**

(a) In M1 with FCFS, exogenous releases and no merge:

S_j = (W_j − F_j) + V_off(R_j) − V_on(R_j) exactly,

where V(R_j) is the unfinished work at the instant j's final is released.

(b) In M2 (Poisson arrivals, i.i.d. marks, exogenous plans, stationary regime ρ < 1), under FCFS without merge:

E[S | wide] − E[S | narrow] ≤ E[W − F | wide],

that is, at most the expected preloaded work per wide task. A narrow task is one with a single tool call.

*Proof.*
- (a) With FCFS and j present, the server is busy on [R_j, C_j]. Exactly the work present at R_j is ahead of
  j's final, so C_j = R_j + V(R_j) + (final size). Subtract the two systems.
- (b) Take E of (a) by type.
  - By the Slivnyak–Mecke property of Poisson processes, the other tasks seen from task j form the stationary
    process, independent of j's marks, and R_j is independent of them.
  - V_off(R_j) contains no work of j, so its law does not depend on j's type.
  - V_on(R_j) ≥ V_on,others(R_j), because adding arrivals cannot decrease FCFS backlog (Lemma 1), with
    equality for narrow tasks.
  - Subtracting gives the bound. The algebra step is checked by Z3. ∎

[T: prop_fcfs_identity_saving_decomposition], [T: z3_prop_fcfs_discriminant_algebra]

**Approximation A (FCFS, second moments) [approximation].**
- Off finals form a Poisson process: Poisson arrivals displaced by i.i.d. delays stay Poisson (Mirasol,
  "The output of an M/G/∞ queuing system is Poisson", Oper. Res. 11(2):282–284, 1963; verified).
- So the off wait is Pollaczek–Khinchine: λE[W²]/(2(1−ρ)).
- Treating on-pieces as independent Poisson jobs gives the "burst splitting" gain
  ≈ λ(E[W²] − E[w²]·k)/(2(1−ρ)) for *every* task, which is your hypothesis ∝ ρ/(1−ρ).
- `predict_fcfs` implements it.

**Approximation B (PS) [approximation].** Use the M/G/1-PS response time E[T | x] = x/(1−ρ), which does not
depend on the service distribution. Sources:
- Kleinrock, "Time-shared systems: a theoretical treatment", J. ACM 14(2):242–261, 1967, verified as the
  processor-sharing reference;
- Coffman, Muntz & Trotter, J. ACM 17(1):123–130, 1970, verified; M/M/1 conditional waiting-time law;
- attribution of the general-service form not verified; it is textbook material, e.g. Harchol-Balter, *Performance
  Modeling and Design of Computer Systems*, CUP 2013; the exact statement was not checked.

Assume preloads finish before R, and treat the requests as Poisson. Then:
- saving(wide) ≈ W/(1−ρ_off) − F/(1−ρ_on);
- saving(narrow) ≈ 0, by insensitivity: splitting other tasks' work does not change a PS job's mean
  response time.

So the gain from preloading is the preloaded work *stretched by* 1/(1−ρ), and it accrues only to the task that
preloads. `predict_ps` implements this.

The exact PS work-model simulation does **not** reproduce A1 at high load (§6). Pieces as separate egalitarian
jobs inflate the job count, and the arrivals are clustered. So A1 should be read as a model of "stretched
critical path", not as a consequence of M1-PS.

**What the data say (§6).** Under the FCFS work queue, P5b caps the wide-minus-narrow gap at about 1.7 s. The
measured gap is:
- 5.82 s [2.99, 8.53] at 0.45/s;
- 9.48 s [5.67, 13.23] at 0.6/s.

[T: fcfs_work_model_rejected_by_width_discriminant]

The narrow tasks' saving is never significantly different from 0: 0.30, 1.21, 0.57, 1.93 s, all CIs include 0.
So **the growth with load is not mainly cross-task queue reduction from splitting bursts.** It is the wide tasks'
own critical path, which shrinks by the preloaded work multiplied by a load-dependent stretch. That is A1's
structure. Caveats:
- P5b assumes stationarity, while the runs last 180 s and start empty;
- the classification of narrow tasks comes from the 'on' run's model_calls;
- the result is an inference about mechanism in one realization per rate.

## 4. Why preload priority barely matters

**Theorem 4 (priority insensitivity) [proved].** Assume M1 with exogenous (exo) plans, ε = 0, preemptive
priorities, FCFS within ties, and merge-at-need. For X ∈ {on, low, jit}:

R_j + V(R_j) − Π^X_j + F_j ≤ C^X_j ≤ R_j + V(R_j) + F_j,

where:
- V(R_j) is the total unfinished work at R_j, the same function of time in all three systems;
- Π^X_j is the other tasks' unprocessed preload work at R_j in system X.

Hence |C^X_j − C^Y_j| ≤ max(Π^X_j, Π^Y_j) ≤ V(R_j).

*Proof.*
- *V is common:* all three are work-conserving with identical arrivals, since with ε = 0 merges move work
  without changing it. Apply Lemma 1.
- *Upper bound:* no work released after R_j is ever ahead of j's final. On is FCFS. In low, the other work is
  FCFS and the low class is below it. In jit, later requests have keys ≥ their release > R_j. Merges only
  remove work.
- *Lower bound:* by C_j the server must finish all non-piece work present at R_j (released earlier, so ahead in
  all three), j's merged residual, and F_j. The only part of V that may be behind j is the other tasks' pieces,
  Π^X_j.
- *Difference:* follows from the bracket; Z3 checks it. ∎

[T: thm4_bracket_and_difference_bound], [T: z3_thm4_bracket_implies_difference_bound]

Plans at top priority break the upper bracket, because later plans preempt the final. That case is excluded
[T: thm4_needs_exogenous_plans].

*Size of Π in the run's regime (empirical, engine M3, nb, fcfs2048, 32 seeds).* E[Π] is 0.024, 0.052, 0.235 and
0.523 GPU-s per task at 0.15, 0.3, 0.45 and 0.6/s. Mean latencies are 14–33 s. The engine's nb/low latency
differences are 0.02, 0.02, 0.20 and 0.56 s. Π is ≈ 0 except near saturation, consistent with "on ≈ low ≈ JIT
within noise" for the non-blocking code. No non-blocking contention data are available locally, so this is not
checked against measurements here.

*The blocking version is different.* When the final waits for the agent's own preload chain, a lowest-priority
chain starves behind continuous normal-priority prefill, and the final cannot even be sent.
[T: blocking_low_priority_chain_starves_but_nonblocking_does_not]: a blocking low chain takes 20.9 s, a blocking
normal-priority chain 13.9 s, and non-blocking 8.4 s.

The measured blocking data show low ≈ on at 0.15, 0.3 and 0.45/s (differences 0.03, 0.33 and 0.13 s, CIs
include 0). At 0.6/s low is **2.92 s [2.23, 3.66] slower**. The engine model does *not* reproduce that penalty:
it predicts low slightly faster, by 0.4 s. The cause is open. Candidates are vLLM priority-queue head-of-line
behaviour (up to 7 requests sat in "Waiting" only in the low run at 0.6/s, per `results/logs_pod/serve.log`) and
unmodelled per-request CPU overhead.

## 5. Where it breaks

### 5a. KV capacity

Setting: KV capacity M tokens. At each engine step define:
- **Pin(t)**: the KV that running requests must hold, including their cached prefix and the new tokens of the
  step;
- **Q(t)** = Σ over waiting tasks (final not yet run, no running request) of their preloaded prefix K_j(t), as
  computed;
- **S_τ(t)**: unpinned caches that are not waiting-preloaded and were last used after τ(t), where τ(t) is the
  oldest last-use time of a waiting preloaded cache.

**Lemma K [proved, elementary].**
- (i) If Pin(t) + Q(t) > M at some step, then some preloaded token was evicted by then, under *any* policy.
  *Proof:* pinned KV must be resident, so the resident preloaded KV is < Q(t).
- (ii) Under LRU, with "not waited for" evicted first on ties: if Pin + Q + S_τ ≤ M at every allocation, no
  preloaded token is ever evicted before use. *Proof:* the free space plus the unpinned caches older than τ
  covers the demand, and LRU evicts those first.

[T: lemma_k_trajectory_checks_with_eviction]: in every engine run with a capacity, zero steps violate (i)
("necessary but no eviction") and zero evictions violate (ii).

*Consequence.* Evicted preloaded KV must be recomputed by the final. Its first computation (plus ε) was wasted,
so preloading then adds work without removing any from the critical path, which is pure harm. T1(iii) no longer
applies because total work grows.

*Checkable sizing criterion.* By Little's law, the mean waiting-preloaded KV is λ·q·E[Σ_{k<n} L_k (T_(k+1) −
T_(k))], where L_k is the log length after k results and T_(k) are the order statistics of the n tool delays.
For the run's delay law and sizes this is **57,560 token-seconds per wide task**, giving 6.0k, 11.1k, 18.5k and
23.0k tokens at the four rates (`preload_kv_seconds`, `kv_littles_law`).

The engine's peak of Pin + Q (the quantity in (i)) is 30.9k, 45.1k, 64.6k and 75.2k tokens, against
M = 163,904 (`serve.log`). So the run sits ≥ 2.2× inside the safe region at every rate. vLLM's reported "KV cache
usage" peaked at 41%, but that metric likely excludes cached-unreferenced blocks, so it is weaker evidence.

Engine sweep at 0.6/s (nb, 4 seeds):

| Capacity | Mean saving | Preloaded tokens evicted per run |
|---|---|---|
| 163,904 or 80,000 | 5.6 s | 0 |
| 50,000 | 2.3 s | 73k |
| 40,000 | −1.3 s (preloading hurts) | 130k |

At 0.3/s, 40k tokens costs only 0.1 s.

Rule: preloading is KV-safe while peak(Pin + Q) ≤ M. Q scales as λ × (result size) × (tool-wait spread) ×
(width²/2), so long contexts, many results per agent, long tool tails and many concurrent agents break it first.

### 5b. Overhead ε

**Theorem 6.**

(a) [proved] In isolation (M1, merge) preloading never hurts, for any ε (P4).

In the engine, the isolated harm is at most one step plus the recompute of a partial block (≈ a + 16α ≈ 0.03 s).
[T: engine_isolated_saving_matches_preloaded_prefill_cost]

(b) [algebra proved by Z3, inside approximation B] Removing work w from a task's critical path, whose mean is x,
at a cost of ε extra GPU work per piece, lowers the mean latency iff ε < w(1−ρ)/(λx), assuming
ρ + λε < 1. [T: z3_thm6b_ps_breakeven]. At 0.45/s (ρ ≈ 0.72, w ≈ 0.24 s, x ≈ 1.5 s) the threshold is
ε* ≈ 0.10 s.

(c) [proved] If ε_GPU > 0, preloading lowers the stable arrival rate from 1/s̄ to 1/(s̄ + k̄ε). For λ between the
two, the off system is stable and the on system unstable (Loynes, "The stability of a queue with non-independent
inter-arrival and service times", Math. Proc. Cambridge Phil. Soc. 58(3):497–520, 1962; verified). So preloading
with nonzero GPU overhead must lose near saturation. [T: z3_thm6c_capacity_loss]

*Against the microbenchmark (empirical).* The single-task final-latency saving is 1.71·10⁻⁴ s per cached token
− 0.04 s (15 paired points, k = 1…16 results; [T: microbenchmark_saving_per_cached_token]). Break-even in
isolation is therefore about 230 cached tokens. The run's 1,400-token results are about 6× above it.

The per-preload *wall* overhead is ≈ 0.077 s: the intercept of a fit to 131 isolated preloads, which includes one
step a = 0.028 s.

*Inference on ε_GPU (model-dependent).* Under approximation B, a positive saving at 0.6/s (ρ̂ ≈ 0.89, 4.7 preloads
per task) requires ε_GPU < 0.033 s per preload. So at most about 40% of the wall overhead can be GPU time. If all
0.077 s were GPU time, the on system would be overloaded (ρ ≈ 1.1) at 0.6/s while off is not, which contradicts
the measured 8.25 s gain.

## 6. Comparison with data

### Parameters

| Quantity | Value | Status | Source |
|---|---|---|---|
| a, c | 27.77 ms, 0.543 ms/seq | measured | `batch_profile.json` |
| Prefill cost | α = 1.14·10⁻⁴ s/token, κ = 1.227·10⁻⁸ s/(token·context) | estimated | least squares on 131 preloads in `results/real/preload-load` traces (residual sd 0.014 s); cold 10.8k-token TTFT predicted 2.14 s vs 1.99 s measured |
| Request latency δ | 0.049 s | estimated | fit intercept − a |
| Result size | min(t, 1400) + 35 header, t from 160 traced results (mean 1393 at cap 1500) | estimated | traces |
| Header, suffix | 110, 15 tokens | estimated | one trace |
| Tool delay | DelayModel(0.5, 4, 0.3, 4, 12): U[0.5, 4] w.p. 0.7, U[4, 12] w.p. 0.3; E D = 3.975 s, E max of 8 = 9.13 s | from code | `tools.py` |
| Plan | 89 prompt tokens; output N(127, 28) tokens for 8-query plans | measured | traces |
| Plan output for 1-query plans | 25 tokens | **assumed** | not observable |
| Final output | 10 tokens | measured | traces |
| Widths per task | from the 'on' run's model_calls (2 → n = 1, 8 → 7, 9 → 8); q_wide = 0.69, 0.64, 0.72, 0.67 | observed workload | CSVs |
| Arrival times | exact, from the CSVs | measured | — |
| ρ̂ (prefill + decode-marginal work × λ) | 0.23, 0.43, 0.72, 0.89 | estimated | `workload_moments` |
| Engine token budget, chunk order in vLLM 0.32 | unknown; not in `serve.log` non-default args | — | — |

No parameter was fitted to the contention data. The engine default (budget 2048, FCFS chunk order) was coded
before running the comparison. Other engine settings are reported as sensitivity. Tool delays and result sizes are
resampled per seed; the measured run is one realization.

### Predicted vs measured mean paired saving (off − on), seconds

Measured values are the blocking version, seed 0, with 95% bootstrap CIs over tasks. Engine rows are
blocking-mode means over 32 seeds; parentheses give the 2nd–31st order statistics over seeds. Work-model rows
are means over 4 seeds of M1 'on' (pieces released at return, exo plans, ε = 0), not blocking mode.

| Rate (tasks/s) | 0.15 | 0.3 | 0.45 | 0.6 |
|---|---|---|---|---|
| **Measured, all tasks** | **1.20** [0.50, 1.92] | **2.47** [1.45, 3.53] | **4.74** [3.44, 6.04] | **8.25** [6.09, 10.34] |
| A1 PS formula (zero-parameter) | 1.51 | 1.90 | 4.25 | 10.13 |
| A FCFS/PK formula | 1.39 | 1.66 | 3.02 | 5.62 |
| Engine M3, budget 2048, FCFS chunks (default) | 1.47 (0.8–2.2) | 3.04 (2.2–4.1) | 6.30 (4.8–7.6) | 6.48 (5.0–8.2) |
| Engine, budget 2048, fair chunks | 1.54 | 3.12 | 8.18 | 9.29 |
| Engine, budget 8192, FCFS chunks | 1.49 | 3.22 | 9.12 | 10.05 |
| Engine, unchunked | 1.46 | 3.21 | 9.70 | 12.05 |
| Work model M1-FCFS (simulated) | 1.37 | 1.37 | 2.75 | 2.32 |
| Work model M1-PS (simulated) | 1.50 | 1.45 | 2.97 | −0.56 |

### Split by width (the discriminating test)

| Rate | 0.15 | 0.3 | 0.45 | 0.6 |
|---|---|---|---|---|
| Measured, wide | 1.60 [0.68, 2.51] | 3.17 [2.18, 4.19] | 6.40 [5.03, 7.80] | 11.40 [8.93, 13.87] |
| Measured, narrow | 0.30 [−0.42, 0.95] | 1.21 [−0.85, 3.47] | 0.57 [−1.69, 3.14] | 1.93 [−1.07, 4.90] |
| A1 PS, wide / narrow | 2.19 / 0 | 2.96 / 0 | 5.93 / 0 | 15.2 / 0 |
| A FCFS, wide / narrow | 1.91 / 0.24 | 2.26 / 0.58 | 3.49 / 1.82 | 6.18 / 4.51 |
| Engine default, wide / narrow | 2.05 / 0.16 | 4.07 / 1.19 | 7.49 / 3.30 | 6.79 / 5.86 |
| Engine fair, wide / narrow | 2.17 / 0.11 | 4.31 / 0.97 | 10.65 / 1.97 | 12.91 / 2.04 |

Model validation on absolute latency: the measured off-mode mean latency is 13.71, 16.52, 23.40 and 32.90 s. The
default engine gives 13.8, 16.0, 25.8 and 32.9 s [T: engine_reproduces_off_latency]. The work models give 12–18 s,
because they ignore the stretching of decode by prefill.

### Honest reading

1. **Light load is explained.** At 0.15/s every model predicts 1.4–1.5 s against a measured 1.20 s (inside the
   CI). The wide-task value of 1.60 s matches the isolated saving (P4; single-task measurement 1.61 s). The
   models run 0.3–0.5 s high on wide tasks at light load, consistent with the ≈ 10% over-estimate of piece cost
   noted in §3.
2. **The zero-parameter PS formula A1 falls inside the measured CI at all four rates.** It also gets the
   structure right: wide tasks ≈ stretched preloaded work, narrow tasks ≈ 0. It overshoots wide tasks at 0.6/s
   (15.2 vs 11.4 [8.9, 13.9]); the 180-s, start-empty run cannot reach a ρ = 0.89 steady state. A1 is a
   heuristic, and its exact M1-PS counterpart does *not* reproduce it, so its agreement is evidence for the
   "stretched critical path" picture, not a derivation.
3. **The FCFS work-queue explanation (your hypothesis: gain from splitting one large burst) is rejected as the
   main mechanism.**
   - Its formula under-predicts at 0.45 and 0.6/s, by 1.7 and 2.6 s.
   - It predicts large narrow-task gains (1.8, 4.5 s) that are not seen.
   - P5b's bound on the wide–narrow gap (≈ 1.7 s) is exceeded with CI lower bounds of 2.99 and 5.67 s.
   - The mechanism still contributes: narrow-task point estimates are positive, though not significant.
4. **The mechanistic engine model gets the magnitude of the overall saving roughly right.**
   - At 0.15 and 0.3/s it is inside the CI.
   - At 0.45/s it is 1.6 s above; at 0.6/s it is 1.8 s below; its seed ranges overlap the CIs.
   - With the default FCFS chunk order it attributes too much gain to narrow tasks at high load (5.9 vs 1.9) and
     too little to wide tasks (6.8 vs 11.4; the seed range 4.8–8.6 excludes the CI).
   - Fair chunk sharing fixes the split at 0.6/s (12.9 / 2.0 vs 11.4 / 1.9) but over-predicts at 0.45/s.
   - None of the engine variants matches all four rates, and vLLM 0.32's chunk scheduling is not known locally.
   - We did not tune a variant to fit.
5. **Not explained:** the lowest-priority blocking penalty at 0.6/s (§4).

## 7. Prior work and positioning

**Queueing and scheduling.**
- **Lindley (1952) and the reflection map / network calculus** (Le Boudec–Thiran 2001): Lemma 1 and T1 are
  restatements of the monotonicity of the reflection map (earlier arrivals mean at least as much work done).
- **Loynes (1962)**: stability; T6(c).
- **Pollaczek–Khinchine** (textbook; not separately verified) and **processor sharing** (Kleinrock 1967;
  Coffman–Muntz–Trotter 1970): Approximations A and B.
- **Mirasol (1963)**: displaced Poisson arrivals stay Poisson, used in A.
- **Hajek**, "The proof of a folk theorem on queuing delay with applications to routing in networks", J. ACM
  30(4):834–851, 1983 (verified bibliographically; content not read): this folk theorem, that more regular input
  lowers delay, is the classical form of the burst-splitting intuition. §3 shows it is *not* the dominant effect
  here.
- **Fork–join queues** (Flatto & Hahn, "Two parallel queues created by arrivals with two demands I", SIAM J. Appl.
  Math. 44(5):1041–1053, 1984; verified): the n tool calls are a fork–join on infinite-server stations. Preloading
  adds a *partial join* (consume each branch on arrival), which we did not find analysed there.
- **Self-suspending tasks** (Chen et al., ECRTS 2019, verified in `docs/verification_report.md`): a tool wait is a
  suspension, and preloading moves part of the post-suspension computation *into* the suspension as soon as its
  input exists. The segmented-suspension literature fixes the segment structure, while here the segment is split
  online.
- **Graham (1969)**: timing anomalies, which are the R1/R2 phenomenon.

**LLM serving.**
- **vLLM / PagedAttention** (Kwon et al., SOSP 2023, verified) with automatic prefix caching, and **SGLang /
  RadixAttention** (Zheng et al., NeurIPS 2024, verified): the mechanism that makes a 1-token preload reusable.
- **Sarathi-Serve** (Agrawal et al., OSDI 2024, verified): chunked prefill, which model M3 follows.
- **InferCept** (Abhyankar et al., ICML 2024, PMLR 235:81–95, verified): what to do with a request's KV during an
  external interception (preserve, discard or swap) to minimise waste.
- **Continuum** (Li et al., arXiv 2511.02230, 2025; verified, version details differ): KV time-to-live during tool
  calls.
- **KVFlow** (arXiv 2507.07400, NeurIPS 2025; verified): workflow-aware eviction and prefetch for multi-agent
  graphs.
- **Autellix** (Luo et al., arXiv 2502.13965, 2025; verified): program-level scheduling.
- **Conveyor** (Xu et al., arXiv 2406.00059, 2024; verified): the dual idea, starting tool execution during LLM
  decoding. Here LLM prefill runs during tool execution.

Those works keep or reload KV *that already exists* across a pause. Preloading instead *creates* the KV of
arriving tool results during the pause. A targeted search found no paper describing incremental prefill of
parallel tool results as they arrive, and none analysing it. One search hit, "Learning Agent Execution for
KV-Cache Management in Agentic Serving" (arXiv 2608.14624), prefetches context before a call. It was not read and
is unverified.

## 8. What is new and what is folklore

**Folklore or standard, stated as such:**
- Lemma 1, Lemma 2 and T1 (reflection and size monotonicity);
- the PK and PS formulas and Loynes stability;
- Lemma K, which is elementary;
- the break-even algebra in T6(b);
- the light-load limit in P4.

**Small results not found stated for this setting (not claimed as deep):**
- T2: the tight per-task harm bound for same-priority preloading;
- T3: per-task safety of lowest-priority preloading *with merge-at-need*, and R2, showing it needs exogenous plan
  completions;
- T3b: clairvoyant JIT;
- T4: priority insensitivity bounded by the other tasks' unprocessed preload work;
- P5: the FCFS identity and the width discriminant, and its use to separate "burst splitting" from "stretched
  critical path" in the data.

Each proof is a few lines of standard scheduling reasoning. Novelty is limited to the formulation.

**Empirical, model-dependent:**
- the mechanism attribution in §3 and §6;
- the KV safety margin;
- the ε_GPU bound.

## 9. Limitations

1. Every theorem assumes exogenous tool timing: plans off the GPU, or at top priority. In the real system plans
   are decode-heavy GPU work, which is exactly the regime of R1 and R2.
2. Theorems 3, 3b and 4 assume preemptive priority. vLLM does not preempt running requests for priority (except
   under KV pressure), so a running preload chunk adds up to one chunk of blocking.
3. The data are one workload realization per rate (seed 0) from the *blocking* code. CIs are within-run bootstrap
   over tasks and understate run-to-run variance; the engine's seed spread at 0.45–0.6/s is ±1.5–2 s.
4. M3 is a calibrated model, not vLLM. Unknown items:
   - vLLM 0.32's token budget and chunk order;
   - whether decodes are scheduled before prefill chunks;
   - attention cost of decode at long context;
   - CPU-side overheads.
5. The plan length of narrow tasks is assumed.

Reproduce:
- `.venv/bin/python -m pytest -q tests/test_preload_theory.py` (35 tests, about 5 s).
- Engine table: `predict_engine(load_contention(), r, seeds=range(32), cfg=EngineCfg(...))`.
- Exhaustive grid: `search_dominance(grid_tasks([0,1],[0,1],[0,1,2],[0,1,2],[1],[1,2]), 2, (a, b), disc, pm, eps)`.
