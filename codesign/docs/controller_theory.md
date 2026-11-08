# A KV-aware preload controller: theory, specification, predictions

Companion code: `src/deadline_agent/controller_theory.py`. Machine checks: `tests/test_controller_theory.py`
(31 tests, about 13 s locally). A citation `[T: name]` points to a test function in that file. This document builds on
`docs/preload_theory.md` (models M1/M3, T1-T4, Lemma K). It corrects that document where the data contradicted it
(Section 2.4).

## Status labels

| label | meaning |
|---|---|
| **proved** | full proof given here; "+check" names the machine check and its exact instance bounds |
| **known** | a published result; the citation was checked against the source (or flagged where it was not) |
| **folklore** | standard argument, not checked against a specific source |
| **approximation** | derived inside an explicitly approximate model; the derivation is exact, the model is not |
| **empirical-only** | observed in the data or in the simulator M4, with no proof |
| **conjecture** | believed, not proved |
| **refuted** | false, with a counterexample |

No statement here is called "exhaustive" unless every instance in a stated finite grid was enumerated. "Z3" means an
SMT proof of an algebraic identity or implication over the reals, under the hypotheses stated.

## 0. Results at a glance

| # | statement | status |
|---|---|---|
| L | LRU residency: a cached free block survives iff its stack distance stays at most N-1 | proved; property test, 300 random traces, 3-12 blocks, 5-60 ops |
| L1 | tail-first freeing keeps each task's surviving prefix downward-closed | proved; property test |
| Q | a queued request's cached prefix is unreferenced, so it is evictable, until admission | known (vLLM v0.32.0 source); M4 ablation |
| I | an adversarial flood evicts any unpinned preload; only pinning certifies survival | proved; check |
| S | stretch identity: busy = steps*a + c*(decode slots) + prefill time; decode steps per busy second = (1-u)/(a+cB) | proved in M4 (exact); check |
| BE | processor-sharing case: preloading helps iff the survival fraction sigma > rho | approximation (PS); Z3 + exact rational test |
| BE' | stretch model: preloading helps iff sigma > sigma* = rho_s/(1-u+rho_s) | approximation; Z3 |
| BE'-data | the sign rule sigma vs sigma* matches the measured sign in 6 of 6 conditions | empirical-only |
| F | fluid TTL (Che) characteristic time, unique fixed point; a priori crossover near 55k tokens at 0.6/s | uniqueness proved; prediction is an approximation |
| A0 | under strict preemptive priority in GPU and KV, no preload delays any task | proved (T3 + Lemma 2 of preload_theory); property test |
| A1 | "a lowest-priority preload never delays another task" in vLLM-like model V | **refuted** (2-task counterexample in M4) |
| A-i | controller never worse than off | holds in expectation inside BE' only; empirically -0.12 s (worst of 5 conditions, M4 mean over 8 seeds) |
| A-ii | controller = always-preload ('low') when capacity suffices | proved about the rule; property test; M4 identical at full KV |
| A-iii | Cao-style constant factor | trivial (1+beta) bound for 'off', beta about 0.08 (approximation); no constant factor relative to the achievable saving for any online rule (follows from I and B3) |
| B1 | fluid relaxation: the density index rule is LP-optimal | known (Dantzig); Z3 property test |
| B2 | {skip, unpinned, pin} per task (multiple choice): hull-greedy is LP-optimal | known (Sinha-Zoltners); Z3 property test |
| B3 | the index rule in the time-indexed (interval) problem has a constant factor | **refuted**: ratio 2l-1 (preemptive), k (non-preemptive) |
| B3-grid | small grid: worst OPT/greedy = 3; LP/OPT at most 3/2 on a subsample | exhaustive over the stated grid for OPT/greedy; LP on every 7th instance |
| C | a pin pays iff (1-p) cost(L) > rho_s c h / a | approximation (BE' accounting + Prop S) |
| C' | refresh keep-alive does not beat pin/skip at 40k-50k | empirical-only (M4) |
| E | M4 reproduces measured signs and the crossover; computed prefill tokens within 12% of the vLLM logs | empirical-only (no fitted parameter) |

## 1. Data facts used (from the vLLM logs and CSVs)

Sources: `results/logs_pod/serve_kv{5000,3125,2500}.log` and the per-task CSVs under `results/real/preload_kv/`
(seed 2, 0.6/s unless stated otherwise; 40k at 0.3/s is `kv40k_r03`).

**Computed prefill tokens.** The vLLM "Avg prompt throughput" is a 10-s average, so 10 x the sum over a run's
window gives the computed (not cached) prompt tokens. The windows are in `LOG_WINDOWS`; the code is
`log_totals()`.

| condition | off | on | low |
|---|---|---|---|
| 80k | 716k | 716k | - |
| 50k | 716k | 947k | 793k |
| 40k | 730k | 932k | 842k |
| 40k at 0.3/s | 374k | 419k | - |

**Survival.** sigma = final-prompt cached tokens / prompt tokens computed by preloads. Preload tokens = log total -
final computed - plan prompts.

| run | sigma |
|---|---|
| 80k on | 1.08 (about 1; the excess is accounting error in the log windows) |
| 50k on | 0.48 |
| 50k low | 0.85 |
| 40k on | 0.39 |
| 40k low | 0.64 |
| 40k at 0.3/s on | 0.89 |

**Where the latency goes (50k on vs off).** Most of the harm is in arrival to last tool return (+9.1 s), i.e. the
plan stage is slowed by the extra prefill. The final TTFT adds +1.7 s and decode adds +0.5 s. So the harm is not the
task's own recompute, which is about 1.3 s. It is everyone's plans stretched by everyone's wasted prefill.

**Plan length of narrow tasks.** This matters for the M4 workload. At 0.15/s (light load,
`results/real/preload_seeds/seed2/tasks_r0.15_fcfs_pre-off.csv`), the estimated plan stage (arrival to last return
minus E[max tool delay | width]) is 3.40 s for wide tasks and 6.03 s for narrow ones. At batch size about 1, 128
generated tokens take about 3.6 s and 25 take about 0.7 s. So the `Calib.plan_out_narrow = 25` of preload_theory is
inconsistent with the data. M4 uses 128, and if anything that is low (empirical-only).

## 2. Models

### 2.1 Model V (vLLM v0.32.0 semantics, read from source)

Verified in the v0.32.0 `vllm/v1/core/sched/scheduler.py`, `kv_cache_manager.py` and `block_pool.py`:

- **Scheduling order.** Running requests are scheduled first, in list order, against the token budget
  (`max_num_batched_tokens` = 2048 by default). Waiting requests are considered only if no preemption happened in
  that step.
- **Admission.** With `scheduler_reserve_full_isl=True` (the default), a waiting request is admitted only if the
  blocks for its whole prompt can be allocated. If `allocate_slots` returns None, the loop `break`s (head-of-line
  blocking).
- **Preemption victim** under the PRIORITY policy: `max(running, key=(priority, arrival_time))`. Lowest-priority
  (largest value) requests go first.
- **Free queue.**
  - It contains every block with ref_cnt 0, including cached ones; usage = 1 - free/total.
  - `get_new_blocks` pops the head and evicts that block's hash.
  - `touch` removes a cached block from the queue.
  - On free, a request's blocks are released in reverse order (tail block first). Cached blocks are appended to the
    tail; uncached ones are prepended to the head.
- **"Deferred" in the logs.** This is `num_skipped_waiting_reqs` and is already included in "Waiting".
- **Not re-read (assumption).** The ordering of the waiting queue under PRIORITY, (priority, arrival).

### 2.2 M4: an event simulator of model V (`simulate_vllm`)

M4 models the following.

- **Token-level steps.** Step time is a + c*B + (prefill time of the chunks scheduled in the step), with the calibrated
  `Calib` of preload_theory: a = 27.77 ms, c = 0.543 ms/seq, prefill alpha L + kappa L^2/2.
- **Engine rules.** Running-first scheduling, the full-prompt admission gate with break, PRIORITY preemption, and an
  exact block-level LRU pool (`BlockPool`) with vLLM's append/prepend rules.
- **Client logic.** The client of `multiagent.py`:
  - plan, then n tool calls, then non-blocking preloads of the growing log at the lowest priority;
  - modes off / on / low / chain / pin / ctrl;
  - pins are open preloads held by `max_tokens`, cancelled when superseded or when the final's first token arrives.

Workload (`seed2_workload`):

- From the seed-2 CSVs: the exact arrival times and each task's final prompt size.
- Sampled from the calibrated law: tool delays, and result sizes rescaled so that they sum to the final prompt.
- Plan output: wide tasks use `Calib.plan_out_wide`; narrow tasks use 128 tokens (ASSUMPTION, supported by Section 1).

Not modelled:

- tokenization and chat-template details beyond fixed header/suffix sizes;
- CUDA-graph and padding effects;
- the measured per-call tool delays, which are not logged per call, so they are sampled.

No parameter of M4 was fitted to measured latencies.

### 2.3 Approximate models

- **PS.** Prefill is a processor-sharing server of load rho.
- **Stretch.** Section 3.
- **Fluid TTL.** Section 3.3.

### 2.4 Why preload_theory.md mispredicted the KV harm

1. **Lemma K sizing.** It used the necessary condition "pins + queue fit in M". It ignored that LRU retains younger
   dead blocks: the cached-but-dead volume behind a prefix grows like v_dead * tau, about 30k tokens at 0.6/s over a
   few seconds.
2. **Referencing in M3.** M3 referenced a request's cached prefix at arrival. vLLM references it only at admission
   (Lemma Q), so M3 under-evicted by about 3x. The ablation [T: test_M4_mechanism_unreferenced_while_queued] shows
   that referencing at arrival removes more than half of the extra computed prefill.
3. **What counted as harm.** It was counted as the task's own recompute. The real cost is the cross-tenant stretch
   (Theorem BE'): one wasted GPU-second per task costs about rho_s/(1-u)^2, about 27 s of summed latency at 50k,
   while one useful GPU-second gains only 1/(1-u), about 2.4 s.

## 3. Mechanism

### Lemma L (LRU stack-distance residency) - proved; [T: test_lemma_L_stack_predicate_matches_block_pool]

**Setting.** A pool of N blocks. Let b be a cached block whose ref count dropped to 0 at time t0. For a time t
after t0, let D_b(t) be the number of blocks other than b that are either referenced, or cached and unreferenced and
released (appended) after b.

**Claim.** b is still resident at time t iff D_b(s) <= N-1 for all s in [t0, t], where D_b is evaluated as if b
were resident.

**Proof.** While b is in the free queue, the queue is A, b, Z, where A is the part ahead of b (closer to the head).
Z consists exactly of the cached blocks appended after b and not yet touched: appends go to the tail, and pops take
the head, so nothing behind b leaves except by `touch`. Uncached frees are prepended, so they join A. Every block is
referenced, in A, b, or in Z, so N = |R| + |A| + 1 + |Z|, that is D_b = |R| + |Z| = N-1-|A|.

Each operation changes this as follows:
- `touch` of a block in Z moves it from Z to R and leaves D_b unchanged;
- `touch` of a block in A increases D_b by 1 and decreases |A| by 1;
- a free decreases |R| and adds to A or to Z;
- an allocation pops the head of the queue. This is a block of A if |A| > 0. Otherwise it is b itself: b is evicted,
  and the would-be D_b reaches N.

Hence b is evicted exactly at the first operation after which the would-be D_b exceeds N-1. QED

This is the Mattson et al. (1970) stack-distance criterion for LRU, adapted to vLLM's pool with its prepend rule and
reference counts.

**Check.** `stack_resident` implements the predicate; `BlockPool` replays the trace. On 300 hypothesis-generated
request-shaped traces (3-12 blocks, 5-60 operations, touches, uncached blocks), the two agree for every key.

### Corollary L1 - proved; [T: test_tail_first_freeing_keeps_task_prefixes_downward_closed]

A request frees its blocks tail-first, so block k of a prefix is appended before blocks 0..k-1. It is therefore
ahead of them, and evicted before them. So the surviving part of a cached prefix is always a prefix, which is the
only kind the prefix cache can reuse.

### Lemma Q (unreferenced while queued) - known (vLLM v0.32.0 source); M4 ablation

A waiting request holds no blocks. Its cached prefix is looked up (`get_computed_blocks`) and touched only when the
scheduler admits it. Until then, by Lemma L, that prefix keeps aging in the free queue. The effective re-reference
gap of a preload prefix is therefore the tool gap plus the next request's queueing delay q. At 50k on, "Waiting"
averages 4.1 requests.

### Proposition I (adversarial flood) - proved; [T: test_adversarial_flood_evicts_any_unpinned_preload]

Consider an unpinned preload of L blocks that completes at t0 and is next needed at t1. Any sequence of other
requests that allocates N or more blocks in (t0, t1) evicts all of it: by Lemma L, D_b reaches N. The controller is
the only client, but it does not control arrivals, so no online rule can certify that an unpinned preload survives.
A referenced (pinned) block is never evicted. A pin can be preempted, and under PRIORITY, LOW pins are preempted
first [T: test_M4_lowest_priority_pins_are_preempted_first]. Its blocks then become cached-free, so it degrades
to an unpinned preload.

### Proposition S (stretch identity) - proved for M4 (exact); [T: test_M4_accounting_and_stretch_identity]

In M4, each step costs a + c*B_t + p_t, where p_t is the prefill time of the step. Summing over the steps:

> busy = steps*a + c * sum_t B_t + prefill_time

With u = prefill_time/busy and B_bar = sum B_t / steps, this gives steps/busy = (1-u)/(a + c B_bar), as an
identity. Every decode token on a critical path therefore takes s_bar = (a + c B_bar)/(1-u) on average. Prefill
work stretches every decoding tenant's critical path by 1/(1-u), and extra decode sequences cost only c/a, about 2%
of a step each. The identity holds exactly in M4. For the real engine it is an approximation that is only as good as
the step-cost calibration.

### Theorem BE (PS case) - approximation; Z3 + [T: test_ps_saving_sign_rule_exact]

**Model.** Prefill is processor-sharing with load rho = lambda*Xbar. Preloading moves σP of each task's
critical-path work off the critical path and adds waste (1-σ)P.

**Saving.** Xbar/(1-rho) - (Xbar - σP)/(1-rho-lambda(1-σ)P).

**Claim.** If the denominators are positive, the saving is > 0 iff σ > rho.

**Proof.** The saving is a special case of BE' below, with rho_s = rho and u = rho. QED

**Check.** Exact `Fraction` evaluation of `ps_net_saving` on hypothesis-drawn rationals confirms the sign rule.

### Theorem BE' (stretch model) - approximation; Z3 [T: test_z3_theorem_BE_prime_break_even_survival]

**Model.**
- Each task's latency is a non-stretchable part (the tool waits) plus X/(1-U). X is the task's stretchable
  critical-path time in isolation (plan decode and prefill, final prefill and decode), and U is the prefill
  utilization (Prop S).
- Let u be the utilization without preloads, λ the arrival rate, and rho_s = λX. Note that rho_s can exceed 1: the
  critical paths overlap in a batch.
- Preloads that survive with fraction σ remove σP GPU-seconds from each task's X and add (1-σ)P of waste, so
  U = u + λ(1-σ)P.

**Claim.** If 1-u-λ(1-σ)P > 0 and P > 0, then the saving

> Delta = X/(1-u) - (X - σP)/(1-u-λ(1-σ)P) > 0  iff  σ > σ* = rho_s/(1-u+rho_s).

**Proof.** Multiply by the positive denominators. Delta > 0 iff X(1-u-λ(1-σ)P) > (X-σP)(1-u), that is
σP(1-u) > λX(1-σ)P. Dividing by P > 0 gives σ(1-u+rho_s) > rho_s. QED

The threshold does not depend on P. Z3 proves the equivalence under the positivity hypotheses.

**Interpretation.** At w = 0, the marginal cost of waste is -dDelta/dw = rho_s/(1-u)^2 per wasted GPU-second per
task. The marginal gain from a surviving GPU-second is 1/(1-u). At 50k, rho_s = 4.63 and u = 0.59, so the ratio is
rho_s/(1-u) = 11.3 and σ* = 0.92.

This high bar is the central quantitative fact. In a batched decode engine, wasted prefill hurts every tenant's
critical path at once, while a saved prefill helps one task.

**Data check (empirical-only)** [T: test_measured_break_even_rule_predicts_every_sign]. Inputs, with zero fitted
parameters:
- u and X come from the off run;
- σ comes from the treatment run's log;
- the magnitude uses the treatment run's measured waste, so it is a decomposition, not a forecast.

| condition | u | X (s) | rho_s | σ* | σ | predicted (s) | measured (s) |
|---|---|---|---|---|---|---|---|
| 80k on | 0.62 | 6.92 | 4.27 | 0.92 | 1.08 | +2.02 | +2.52 |
| 50k on | 0.59 | 7.51 | 4.63 | 0.92 | 0.48 | -12.08 | -11.28 |
| 50k low | 0.59 | 7.51 | 4.63 | 0.92 | 0.85 | -1.20 | -1.71 |
| 40k on | 0.63 | 6.54 | 4.03 | 0.92 | 0.39 | -17.78 | -11.12 |
| 40k low | 0.63 | 6.54 | 4.03 | 0.92 | 0.64 | -5.21 | -4.51 |
| 40k at 0.3/s on | 0.34 | 7.05 | 2.15 | 0.77 | 0.89 | +0.75 | +0.65 |

- **Sign.** sign(σ - σ*) equals the measured sign in all 6 conditions.
- **Magnitude** [T: test_measured_stretch_decomposition_magnitudes]. It is within 1 s in 5 of 6 conditions. 40k on
  sits near the pole (1-u-λw is about 0.18) and is over-predicted.
- **Caveats.** Each measured cell is one realization (one seed and one run). The 50k low and 0.3/s cells are close to
  σ* or to 0. No cell is used to set any parameter.

### 3.3 Approximation F (fluid TTL) - uniqueness proved; prediction approximation; [T: test_ttl_fixed_point_unique_and_increasing_in_capacity], [T: test_fluid_ttl_predicts_crossover_between_50k_and_80k]

This is the Che approximation (Che-Tung-Wang 2002; justified asymptotically by Fricker-Robert-Roberts 2012), applied
to the vLLM pool.

**Fixed point.** LRU behaves like a cache with a characteristic time T, where T solves

> M - Rbar = v_dead*T + live(T),  with live(T) = sum over live streams of rate * size * E[min(T, G)]

- Rbar is the mean referenced KV;
- v_dead is the rate at which dead cached tokens are released;
- G is a stream's re-reference gap.

The right side is continuous and strictly increasing in T (v_dead > 0), and it equals 0 at T = 0. So for M > Rbar
the root exists, is unique, and lies in [0, (M-Rbar)/v_dead]. `ttl_fixed_point` finds it by bisection.

**Prediction.** A preload prefix survives a gap G iff G + q <= T.

**Inputs.** Taken only from the off run of each condition and from the delay law (`fluid_table`):
- Rbar = the vLLM "GPU KV cache usage" times M;
- v_dead = computed tokens / wall time.

| condition | Rbar | v_dead (tok/s) | T (q=0) | σ (q=0) | σ (q=1 s) |
|---|---|---|---|---|---|
| 80k | 16.0k | 3.58k | 11.9 s | 1.00 | 0.94 |
| 50k | 12.8k | 3.41k | 5.0 s | 0.79 | 0.29 |
| 40k | 13.5k | 3.65k | 2.8 s | 0.46 | 0.08 |
| 40k at 0.3/s | 4.9k | 1.97k | 12.5 s | 1.00 | 0.98 |

**Crossover.** A scan at 0.6/s gives σ = 0.52 (40k), 0.81 (50k) and 0.99 (60k). With σ* = 0.92, the a priori
crossover is near 55k tokens. This is consistent with the measured sign change between 50k and 80k, and with the
positive saving at 40k / 0.3/s.

Neither q = 0 nor q = 1 is fitted. q = 0 ignores Lemma Q, and q = 1 s overstates it. The measured 'on' survival
(0.48 at 50k, 0.39 at 40k) lies between the two predictions (0.79/0.29 and 0.46/0.08). The model brackets the data;
it does not pin it down.

The realized LRU eviction ages in M4 'on' (median) are:

| condition | median eviction age |
|---|---|
| full | 31.6 s |
| 80k | 12.3 s |
| 50k | 5.0 s |
| 40k | 3.5 s |
| 40k at 0.3/s | 8.4 s |

## 4. Target A: safety and best-of-both

### A0 (strict-priority safety) - proved; [T: test_A0_lost_preloads_never_delay_anyone_under_strict_priority]

If preloads are served at strictly lower, preemptive priority in both GPU time and KV, then no task's latency is
larger than under off, whatever is evicted. This is T3 of preload_theory, together with its Lemma 2.

Plans and finals then see an "off" system in which each final's work is W_j minus the surviving preloaded work.
Under FCFS and PS, completion times are monotone in work sizes. The check runs hypothesis instances of at most 4
tasks with integer data, both disciplines, and arbitrary surviving amounts.

### A1 - refuted; [T: test_M4_counterexample_low_priority_preload_delays_another_task]

vLLM is not strictly preemptive within a step. A running request keeps its token budget ahead of waiting ones, and
KV for waiting requests is not obtained by preemption.

**Counterexample** (M4, capacity 16,000, budget 2048):
- task 0 arrives at 0, has 2 tools with delays 0.2 and 3.0 and results of 6000 and 400 tokens;
- task 1 arrives at 0.35, has 1 tool with delay 0.1 and a result of 6000 tokens.

Under 'low', task 1's latency rises from 1.62 s (off) to 2.54 s, while task 0 improves from 4.62 s to 3.61 s.
Task 0's running 6000-token preload holds the budget when task 1's plan arrives.

So a per-task, deterministic "never worse than off" is false for any rule that ever preloads.

### Strongest true versions

**(a) Direct interference** (proved, paper argument, no machine check). In model V, an unpinned preload of L
tokens delays a given other request directly by at most its own remaining prefill time plus one step. It takes
budget only while running, it frees its KV right after its single output token, and cached-free blocks do not count
against the admission gate. This is ε per preload, at most cost(L) + a + c (for example 0.72 s at L = 5000).

The bound excludes the indirect harm, which is evictions of other prefixes followed by recompute. That harm has no
per-preload bound: it is amplified by queueing (BE'), and when λ*w_bar >= 1-u, waste makes the prefill queue
unstable (Loynes' stability criterion; folklore, citation not re-checked).

**(b) In expectation** (approximation, inside BE'). The rule issues an unpinned preload only if its TTL survival
estimate exceeds σ*, and a pin only if pin_value > 0 (Prop C). Hence, in the stretch model, every issued action has
a non-negative expected saving, so the expected mean latency is at most off's. This guarantee is exactly as good as
the estimates û, rho_s_hat and T̂ (see the caveats).

**Empirically (M4, 8 seeds, mean saving vs off):**

| condition | ctrl saving | seed range |
|---|---|---|
| full | +3.27 | - |
| 80k | +3.01 | - |
| 50k | +1.02 | [-0.5, 2.2] |
| 40k | -0.12 | [-1.4, 0.9] |
| 40k at 0.3/s | +1.62 | - |

So (A-i) holds in M4 up to about 0.1 s in the mean, and not per seed.

### A-ii (equals always-preload when capacity suffices) - proved about the rule; [T: test_rule_equals_always_preload_when_capacity_suffices], [T: test_M4_controller_best_of_both]

**Claim.** If p_unpinned = 1 (LRU will keep every prefix), σ̂ = 1, and σ* < 1 (true whenever u < 1), then `decide`
returns "unpinned" at the lowest priority, which is the existing 'low' mode, or "pin" when pin_value > 0.

**Proof.** With p = 1, pin_value = -rho_s * keep <= 0, so no pin is chosen. Then unpinned_ok = (1 > σ*) and not
feedback_off; feedback_off is false because σ̂ = 1 > σ*. QED

**Check.** The property test covers hypothesis-drawn states and views. In M4 at full KV, ctrl equals low on every
seed (+3.27 s; the test's 2 seeds agree within 0.05 s).

Note: 'low' at full KV is better than 'on' in M4 (+3.27 vs +2.67 s). Neither 'low' nor ctrl was measured at full KV.

### A-iii (Cao-style constant factor)

The Cao-Felten-Karlin-Li (1995) bounds have three ingredients:
- the algorithm controls replacement (Rule 2, "optimal replacement": evict the block whose next reference is
  furthest);
- it obeys "do no harm" (Rule 3);
- the reference sequence is known.

Their results:
- aggressive prefetching has elapsed time at most (1+F/k) OPT for F <= k, and at most 2 OPT otherwise;
- conservative prefetching is at most 2 OPT;
- the aggressive bound is nearly tight, at 1+(F-2)/(k+1).

None of the three ingredients holds here. The replacement is the server's LRU, shared with all tenants, and the
reference times (the tool returns) are random.

- **Trivial constant** (approximation, stretch model). Preloads only move work: with σ = 1, U is unchanged. So
  OPT >= L_off - P/(1-u), and off/OPT <= 1 + β with β = (P/(1-u))/(L_off - P/(1-u)). With the measured 80k values
  (σP = 0.77 GPU-s per task, P/(1-u) about 2.0 s, L_off about 26 s), β is about 0.08. This bound is not reliable for
  the real system: measured chain at full KV saved 3.57 s, more than the stretch model's 2 s ceiling.
- **No constant factor relative to the achievable saving** (proved, by Prop I and B3). Against an adversarial
  tenant stream, no online rule can guarantee any constant fraction of OPT's saving:
  - an unpinned rule can be flooded (Prop I);
  - a pinning rule faces time-indexed knapsack instances on which index rules lose an unbounded factor (B3);
  - online knapsack has no constant competitive ratio (Marchetti-Spaccamela-Vercellis 1995).
- **Is the KV crossover just Cao's aggressive-prefetch harm?** Qualitatively yes. Prefetching too early forces
  evictions, which is exactly what Rules 3-4 exist to prevent. The difference is that the client cannot apply
  Rule 2. LRU evicts the prefetched prefix itself (or another tenant's prefix needed sooner) and keeps dead blocks.
  So the harm is not bounded by Cao's (1+F/k): it is multiplied by the shared-GPU stretch rho_s/(1-u)^2 (BE').
  Pinning is the only client lever that imposes a MIN-like keep-until-use order.

  What is new is quantitative: the σ* threshold, which comes from batched-decode stretch rather than from fetch
  counts; the TTL predictor for σ; and the vLLM-specific mechanism (Lemma Q). The phenomenon itself is not new.

## 5. Target B: which preloads to issue under scarcity

**B1 (fluid relaxation)** - known (Dantzig 1957; attributed via a secondary source); [T: test_fluid_relaxation_density_greedy_is_lp_optimal]

In a stationary fluid relaxation:
- each preload opportunity i offers value v_i = (1-p_i) cost(L_i), the expected GPU-seconds saved;
- it consumes KV-time c_i = L_i * w_i (tokens times hold);
- the opportunities share one aggregate budget, M times the window.

This is a fractional knapsack. Greedy by density v_i/c_i (the `preload_index`) is LP-optimal; the proof is the
standard exchange argument. The index is the GreedyDual-Size ratio cost/size, with size replaced by the KV-time
product.

**Check.** Exact rationals, at most 5 items, 40 hypothesis examples, compared with the Z3 LP optimum.

**B2 (options skip / unpinned / pin)** - known (Sinha-Zoltners 1979); [T: test_multiple_choice_skip_unpinned_pin_lp_greedy_is_optimal]

This is a multiple-choice knapsack LP. Greedy over the incremental slopes of each group's upper convex hull is
LP-optimal. The check covers at most 4 groups of at most 2 options, against the Z3 LP.

**B3 (integral, time-indexed)** - refuted; [T: test_index_rule_has_no_constant_factor_in_time_indexed_problem], [T: test_non_preemptive_admission_has_no_constant_factor]

When KV is a capacity at every instant (interval packing / unsplittable flow on a line), the index rule has no
constant factor:

| variant | instance | OPT | rule |
|---|---|---|---|
| preemptive index rule | X = [0, l), size 2, value 2l-1; Y = [0,1), size 1, value 1; capacity 2. Y has the higher density and displaces X. | 2l-1 | 1 |
| non-preemptive first-fit | X = [0,k), size 2, value 1, admitted first; k unit items [i,i+1) of size 2, value 1 | k | 1 |

The checks use l in {2, 5, 20, 100} and k in {2, 5, 10}, with brute-force OPT.

**B3-grid** [T: test_interval_lp_gap_small_grid]. Exhaustive over the grid: 2-3 items with shapes chosen with
replacement from starts {0,1} x lengths {1,2} x sizes {1,2}, values {1,2,3}, and capacity 2.
- The worst OPT/greedy ratio is exactly 3: [0,1) of size 1 and value 1, against [0,2) of size 2 and value 3.
- LP/OPT is at most 3/2 on every 7th instance of the grid. The LP was not run on all instances.

**Online integral version.**
- Known: there is no constant competitive ratio for online knapsack (Marchetti-Spaccamela-Vercellis 1995).
- With densities in [L,U], ratio ln(U/L)+1 is achievable (Zhou-Chakrabarty-Lukose 2008). This form was not seen in
  the source I could access, so it is unverified.
- For LLM inference with KV constraints, Jaillet et al. (2025) show no constant ratio under adversarial arrivals and
  constant regret under stochastic ones.

The defensible design target is therefore the stochastic fluid optimum, not a worst-case ratio. Conjecture, not
proved: the rule's index displacement among pins is near-optimal for the measured workload. Pins are rare (10-15 per
run), so B3 has little practical bite here.

## 6. Target C: the pinning lever

A pin is a preload request kept open with max_tokens about (remaining-wait quantile)/s_bar.

**KV cost.** The pin keeps L + generated tokens referenced. Referenced tokens count toward vLLM usage, so a pin
consumes admission-gate capacity. Because of head-of-line `break`, one blocked final stalls everyone behind it.
The rule therefore budgets pins with headroom = 110 + 8*(1400+35) + 15 = 11,605 tokens (the largest possible final
prompt), so a final always fits. Before submitting a plan or final, `yield_before_submit` cancels other tasks' pins,
lowest index first. Before the headroom was added, yields caused every pin-mode loss in M4.

**Batch-slot cost.** A pin takes one decode slot per step, which costs c = 0.543 ms per step. Over a hold h at step
time s_bar = a/(1-u) (Prop S), that is c*h*(1-u)/a GPU-seconds. In BE' units this is waste, weighted by rho_s.

**Proposition C** (approximation; [T: test_pin_break_even_threshold]). Pinning beats leaving the prefix to LRU
(survival p) iff

> (1-u)(1-p) cost(L) > rho_s * c*h*(1-u)/a,  equivalently  (1-p) cost(L) > rho_s c h / a.

Example: at 50k, rho_s c/a = 4.63 x 0.0196 = 0.090 per second of hold. For L = 5000, cost(L) = 0.72 GPU-s; a 5-s
hold pays only if 1-p > 0.62. This is why the controller rarely pins: 10 pins at 50k and 15 at 40k, against 123 and
63 unpinned preloads and 210 and 287 skips.

**Preemption.** Under PRIORITY, LOW pins are preempted first. A preempted pin's blocks become cached-free, so the
pin degrades to unpinned and never blocks a normal request through preemption.

**Pin-always** (empirical-only, M4, 8 seeds): +1.54 / +1.36 / -1.75 / -2.70 / +1.33 s at full / 80k / 50k / 40k /
0.3/s. It is worse than ctrl everywhere, because the slot cost and gate interference are paid on every task.

**Proposition C' (refresh lever)** (empirical-only). Re-send the cached prefix as a 1-token request every T̂/2. Each
refresh costs at most 31 recomputed tokens plus one decode slot, and refresh requests are themselves unreferenced
while queued (Lemma Q).

M4, ctrl with the refresh lever: +3.27 / +3.04 / +0.80 / -2.62 / +1.55 s. That is worse than the pin lever at 50k
(+1.02) and much worse at 40k (-0.12), so refresh is not recommended.

## 7. Target D: the controller specification

All inputs are client-observable. The client is the only tenant, so it knows every in-flight request.

```
CONSTANTS
  M          KV capacity (tokens): vLLM startup log "GPU KV cache size"            [observed]
  a, c, alpha, kappa, delay law                                                  [calibrated once, Calib]
  HEADROOM = header + W_max*(result_cap + block_hdr) + suffix = 11605             [derived]
  sigma_star(u, rs) = rs / (1 - u + rs)                                          [derived, Thm BE']
  s_bar(u) = a / (1 - u)                                                         [derived, Prop S]
  window_s = 30 s, min_obs = 5, pin_quantile = 0.99, 64 delay samples, last-50   [KNOBS, set a priori, not swept]

ESTIMATORS (updated on every client event)
  D       = sum over in-flight requests of prompt + generated (a task's shared prefix counted once)
  u       = sum over requests finished in the last window_s of prefill_cost(prompt - cached_tokens) / window_s
  v_dead  = tokens released (prompt + generated of finished requests) in the window / window_s
  T       = (M - D) / v_dead                       (fluid TTL with live(T) := 0: an optimistic upper bound)
  rho_s   = lambda_hat * X_hat,  X_hat = (1 - u) * mean(plan latency + final latency) over the last 50 tasks
            (prior before min_obs: 128 (a + c) + prefill_cost(7000))
  sigma   = sum cached_tokens(final) / sum completed-preload coverage, over the last 50 tasks with
            unpinned preloads (coverage = log tokens of the largest preload whose first token arrived,
            floored to 16-token blocks)

ON tool result of task j while others are outstanding:
  if a preload of j is still in prefill: return                   # at most one in flight per task
  W  = 64 samples of max over outstanding tools of (D - elapsed | D > elapsed)   # delay law only
  p  = fraction of W <= T;   h = mean(W);   hq = pin_quantile of W
  thr = sigma_star(u, rho_s)
  pv  = (1 - u)(1 - p) cost(L_j) - rho_s * c * h / s_bar(u)       # Prop C, L_j = current log tokens
  if pv > 0:
      if D + (L_j - pinned_j) + HEADROOM <= M:  PIN(j, max_tokens = hq / s_bar(u))
      else cancel pins of strictly lower index pv/(L*h), lowest first, while they free enough; if enough: PIN
      (a new pin of j supersedes j's previous pin when its first token arrives)
  elif p > thr and not (n_obs >= min_obs and sigma <= thr):  UNPINNED(j)   # max_tokens = 1, priority LOWEST
  else: SKIP

ON submit of plan or final of task j needing N tokens:
  if D + N > M: cancel other tasks' pins, lowest index first, until D + N <= M
ON first token of task j's final: cancel all of j's pins
ON final ready while j's preload is in prefill: cancel it, keep j's previous pin until the final's first token
```

**Fallback via cached_tokens.** σ̂ is the only direct measurement of survival. Once at least 5 observations show
σ̂ <= σ*, unpinned preloads stop whatever T̂ says. This guards against the two known biases of T̂: it ignores
live(T) and the queueing gap q (Lemma Q), and both make it optimistic.

Known limitation: σ̂ is fed only by tasks that had unpinned preloads. Once the fallback trips, it can recover only
if conditions change enough for pins to start. For the GPU version I recommend an exploration probe, such as 1 in
10 eligible opportunities (a new knob). It is not implemented or tested here.

**Honesty about design iteration.** The rule's structure was revised after seeing M4 outcomes:
- always-pin was replaced by pin_value;
- the headroom was added;
- σ̂'s denominator was changed from requested coverage to completed coverage;
- the PS threshold σ > u was replaced by BE'.

No parameter was fitted to measured latencies. The knobs were set once.

**Implementation notes for `multiagent.py`.**
- Unpinned preloads are the existing 'low' mode (priority = 1e9, max_tokens = 1).
- A pin is the same request with max_tokens = ceil(hq/s_bar), cancelled through the HTTP stream.
- cached_tokens comes from `usage.prompt_tokens_details.cached_tokens` of each final.
- u needs per-request cached_tokens and the cost model.

## 8. Target E: predictions vs measurements

Results are from M4 with seed-2 arrivals and sizes, 8 workload seeds, and no fitted parameter. The value is the mean
saving vs off in seconds; brackets give the range over seeds. Measured values are the paired means from
`results/real/preload_kv/paired.txt` (one realization).

| condition | on | low | chain | **ctrl** (pin lever) | ctrl vs on | measured |
|---|---|---|---|---|---|---|
| full 167k | +2.67 [1.1, 4.7] | +3.27 | +2.75 | **+3.27** | +0.60 | on +1.49, chain +3.57 |
| 80k | +2.56 | +3.06 | +2.51 | **+3.01** | +0.45 | on +2.52 [1.18, 3.86] |
| 50k | -5.34 [-7.5, -3.4] | -0.85 | -6.12 | **+1.02** [-0.5, 2.2] | +6.36 | on -11.28, low -1.71 |
| 40k (0.6/s) | -8.78 | -3.09 | -9.75 | **-0.12** [-1.4, 0.9] | +8.65 | on -11.12, low -4.51 |
| 40k (0.3/s) | +1.22 | +1.49 | +0.81 | **+1.62** | +0.40 | on +0.65 |

**Mean off latency.** M4 27.8 s at full KV; measured 26.3 s.

**Computed prefill tokens, M4 (one seed) vs vLLM log**
[T: test_M4_computed_prefill_matches_vllm_log_totals; all within 12%]:

| condition | M4 off / on / low | log off / on / low |
|---|---|---|
| 80k | 750 / 762 / - | 716 / 716 / - |
| 50k | - / 911 / 847 | 716 / 947 / 793 |
| 40k | - / 946 / 866 | 730 / 932 / 842 |
| 40k at 0.3/s | 378 / 406 / - | 374 / 419 / - |

At 50k and 40k, ctrl's computed prefill is 783k and 780k.

**Agreement.**
- Signs: M4 reproduces every measured sign and the low > on ordering
  [T: test_M4_reproduces_measured_signs_and_crossover].
- Magnitudes: M4 under-predicts the 50k 'on' harm by about 2x (-5.3 vs -11.3 s) and 40k by about 1.3x. Its
  computed-token excess is right (+195k vs +231k at 50k), so the shortfall is in how waste converts into latency.
  M4's stretch is milder than the real engine's, which BE' with measured waste captures better (Section 3).
- Variability: each measured cell is a single run, and M4's seed range for 'on' at full KV is [1.1, 4.7] s. So
  differences under about 1.5 s between measured and predicted are within the workload noise.

**Sensitivity (not used for the headline).**

| variant | condition | on | low | ctrl |
|---|---|---|---|---|
| narrow plan_out = 25 (4 seeds; off latency 24.3 s, closer to 25.0) | 80k | +4.56 | - | +4.83 |
| | 50k | -1.03 | +2.01 | +2.17 |
| | 40k | -3.84 | +0.32 | +1.62 |
| budget 4096 | 50k | -4.99 | - | +0.62 |
| | 40k | -10.61 | - | +0.27 |

With narrow plan_out = 25, the harms shrink and the crossover moves down, but Section 1 shows this value is
implausible.

Ablations (4 seeds):
- pin-waiting-prefix (M3-like): 50k on -0.3, low +1.76; 40k on -3.31, low -0.9 (Lemma Q matters);
- no full-ISL gate, or no HOL break: little change (50k on -5.14 / -4.67).

## 9. Prior work and novelty

**Verified citations.**
- Cao, Felten, Karlin, Li, "A study of integrated prefetching and caching strategies", SIGMETRICS 1995. Rules 1-4,
  aggressive (1+F/k) and conservative 2-approximations (read from the PDF).
- Albers, Garg, Leonardi, JACM 47(6):969-986, 2000. Optimal single-disk integrated prefetching/caching via LP.
- Kimbrel & Karlin, SIAM J. Comput. 29(4):1051-1082, 2000. Reverse aggressive, parallel disks.
- Sleator & Tarjan, CACM 28(2):202-208, 1985. LRU competitiveness (the k/(k-h+1) form was not seen in the snippets I
  could access).
- Young, "On-line file caching", Algorithmica 33:371-383, 2002. Landlord.
- Cao & Irani, USITS 1997. GreedyDual-Size.
- Mattson, Gecsei, Slutz, Traiger, IBM Sys. J. 9(2):78-117, 1970.
- Che, Tung, Wang, IEEE JSAC 20(7):1305-1314, 2002.
- Fricker, Robert, Roberts, ITC 24, 2012.
- Denning, "Thrashing", AFIPS FJCC 1968.
- Dantzig, Oper. Res. 5(2):266-288, 1957.
- Sinha & Zoltners, Oper. Res. 27(3):503-515, 1979.
- Marchetti-Spaccamela & Vercellis, Math. Prog. 68:73-104, 1995.
- Zhou, Chakrabarty, Lukose, WINE 2008.
- Jaillet, Jiang, Mellou, Molinaro, Podimata, Zhou, arXiv 2502.07115 (2025; summary from a secondary source).
- Loynes 1962 (stability): cited from memory, not re-checked.

**LLM-agent KV systems.**
- **Continuum** (Li et al., arXiv 2511.02230, 2025). Server-side KV TTL pinning across a tool call, set from the
  predicted tool duration, in vLLM. This is the closest analogue of our pin lever, but it is server-side and does not
  address incremental preloads or harm under pressure.
- **KVFlow** (Pan et al., NeurIPS 2025, arXiv 2507.07400). Steps-to-execution eviction plus CPU-to-GPU prefetch in
  SGLang. Server-side replacement, i.e. Cao's Rule 2, which a client cannot do.
- **InferCept** (Abhyankar et al., ICML 2024). Preserve / discard / swap KV during intercepts, choosing by a
  memory-waste cost. Server-side.
- **Autellix** (Luo et al., arXiv 2502.13965). Program-level scheduling, not KV preloading.
- **Sutradhara** (Biswas et al., arXiv 2601.12967, 2026). Prefills the tool-independent prefix during tool calls and
  uses server-side reuse-priority tags. Closest in spirit. It is unclear whether it does per-result incremental
  appends, and it gives no harm analysis under KV pressure.
- **SmoothAgent** (arXiv 2607.00151): not read, unverified.

**Honest novelty assessment.**
- Restatements: the mechanism (L, L1) restates LRU stack distance. The harm is Cao-style premature prefetching
  without control of replacement (Rules 3-4). The fluid predictor is the Che approximation. B1 and B2 are textbook
  knapsack results. B3 is an instance of known knapsack/UFP hardness.
- New, but modest:
  1. the vLLM-specific Lemma Q, which explains why survival collapses below the TTL estimate;
  2. Theorem BE', a break-even survival threshold σ* = rho_s/(1-u+rho_s) for a batched-decode engine, validated
     in sign on 6 of 6 measured conditions with zero fitted parameters, which explains why σ* is about 0.92 rather
     than u;
  3. Prop C: in batched decode, pinning costs mainly an externality rho_s*c/a and seldom pays;
  4. a client-only, server-unmodified admission rule with a cached_tokens fallback.
- I found no prior work that does client-only KV-aware admission of incremental preloads. This search was not
  systematic.

## 10. GPU experiments to run (decisive tests)

**Setup.** Same as `scripts/kv_pressure.sh`:
- restart vLLM per capacity (`--num-gpu-blocks-override` 10447 / 5000 / 3125 / 2500, i.e. full / 80k / 50k / 40k);
- seed 2, 180 s, 0.6/s, plus 40k at 0.3/s;
- policies off, low, ctrl, plus on as a reference.

Log, per request, `cached_tokens` and the prompt size, and the server's prompt throughput. Repeat each condition at
least 3 times (seeds 2, 3, 4) so the measured noise is known.

**Predictions to test** (M4, 8 seeds):

| condition | predicted ctrl saving vs off |
|---|---|
| full | +3.3 s |
| 80k | +3.0 s |
| 50k | +1.0 s |
| 40k at 0.6/s | -0.1 s |
| 40k at 0.3/s | +1.6 s |

- ctrl should be at least low at every capacity, and above on by at least 5 s at 50k and 40k.
- The sign rule sign(σ - σ*) should predict the sign of every low/on run.
- Measured σ for low should fall between the fluid q=0 and q=1 values.
- ctrl's realized σ̂ on its unpinned preloads should exceed σ* (otherwise the fallback should have tripped).

**Mechanism checks.**
- Count pins and skips: predicted 10-15 pins and 200-290 skips per run at 50k-40k.
- Rerun 50k with `--max-num-batched-tokens 4096`: M4 predicts that 'on' stays negative.
- Inspect the queue-time of finals at 40k for the HOL gate.

## 11. Limitations

- One measured realization per cell.
- M4 under-predicts the size of the harm.
- The narrow-plan length is an assumption.
- vLLM's PRIORITY waiting-queue order was assumed, not re-read.
- T̂ is optimistic.
- The fallback cannot recover without exploration.
- (A-i) is only an expectation statement inside an approximation.
- The calibration (a, c, alpha, kappa) is from preload_theory and is not re-validated at long contexts.

## 12. Reproduce

```
.venv/bin/python -m pytest -q tests/test_controller_theory.py      # 31 tests, about 13 s
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from deadline_agent.controller_theory import *; \
  print(predict_vllm('kv50k', ('off','on','low','ctrl'), range(8)))"
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from deadline_agent.controller_theory import *; \
  [print(r) for r in breakeven_table('.')]; [print(r) for r in fluid_table('.')]"
```
