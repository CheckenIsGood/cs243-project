# Progress report: everything since the pivot away from summarization

Written 2026-11-07; last updated 2026-11-07 evening (controller implementation, GPU test, prior-work review).
Covers commit `fc3566e` ("new direction and idea", 2026-11-07 03:44) and all uncommitted work after it. The old
summarization plan lives only in git history (commit `4e56d77`).

Labels used throughout:
- **[GPU]**: measured on the real serving stack.
- **[sim]**: a simulator, not the GPU.
- **[proved]**: written proof plus machine checks.
- **[approx]**: derived inside an approximate model.
- **[refuted]**: an exact counterexample exists.
- **[known]**: prior work.

---

## 0. Summary

1. **The pivot.** We dropped single-agent summarization. The project became deadline-aware serving of many LLM
   agents on one GPU: batching, KV keep/evict/preload, tool-call timing and admission, as one online scheduling
   problem.
2. **First theory pass on the general problem.** Results:
   - NP-hardness;
   - batching behaves like m ≈ 51 parallel machines (fluid model only);
   - eager tool issue is safe;
   - a tight speed bound for EDF, s = 1 + 1/δ, but only for tasks released together.

   An independent verifier confirmed most of this, found one flawed lemma and one proof gap, and repaired the
   gap. There is **no constant-factor guarantee for the full problem**.
3. **First multi-tenant GPU runs** [GPU]:
   - EDF and least-laxity priorities barely differ from FCFS;
   - admission control raises on-time completion at high load mainly by dropping tasks.
4. **New focus: KV preloading under multi-tenant contention.** A preload prefills tool results into the prefix
   cache while other tools are still running.
   - With the full KV cache, preloading cuts mean latency by about 1–7 s [GPU].
   - Under KV pressure it **flips sign and costs about 11 s** [GPU]. The harm comes from prefix-cache eviction,
     not from vLLM preemption.
5. **Theory of preloading, two rounds.**
   - Round 1 explains why the saving grows with load, but mispredicted the KV harm by about 10×.
   - Round 2 read vLLM's scheduler source and found the mechanism: a queued request does not hold its cached
     prefix (Lemma Q).
   - It derived a **break-even survival threshold σ\* ≈ 0.92**, which gets the sign right in 6/6 measured
     conditions.
   - It specified a **KV-aware preload controller**.
6. **The controller is implemented and GPU-tested** [GPU] (§6.9–6.10), on 3 seeds × 5 conditions. It runs entirely
   client-side on stock vLLM.
   - It keeps the full gain when memory is ample: +3.1 to +3.4 s on the three-seed mean.
   - Under KV pressure it has the best three-seed mean: +0.2 / +0.8 s at 50k / 40k, against −3.5 / −4.0 for
     always-preload and −0.5 / −1.3 for low-priority preload.
   - It beats always-preload by 7–9 s on the heaviest trace.
   - It is significantly worse than low-priority preload in 1 of 12 cells (−0.95 s at 80k), and worse than no
     preloading in 1 of 15 (−2.75 s at 50k).
7. **Novelty is modest** (§7).
   - The preload request itself is the known cache-warming trick.
   - The harm is the classical premature-prefetch effect (Cao, Felten, Karlin & Li 1995). CacheScout (2026)
     independently saw warm-ups hurt at small caches.
   - What is ours: the multi-tenant measurement of the sign flip, the vLLM queue-eviction mechanism, the σ\*
     threshold, and a controller that needs **no engine changes**. Continuum, KVFlow, Sutradhara and CacheScout
     all modify the engine.

---

## 1. The pivot (commit `fc3566e`)

`PLAN.md` was rewritten. The problem is now:
- Many agent tasks share one GPU.
- Each task has an arrival time, a deadline and a value.
- Each task is a DAG of LLM calls and off-GPU tool calls.
- The server decides online about batching, KV (keep / evict / recompute / preload), tool issue and admission.
- Objective: total value of tasks finished on time.
- Two settings: (a) independent tenants; (b) collaborating agents (orchestrator + workers).

The original goal was a constant-factor competitive online algorithm with resource augmentation:
1. reduce each resource to a classical online problem;
2. compose the reductions;
3. prove the combined guarantee;
4. implement it on vLLM.

**Code added in this commit:**
- `multiagent.py`: the serving layer and agent programs.
  - Request priorities: fcfs / edf / llf, plus admission control `+ac`.
  - Agent shapes: fan-out (plan → parallel searches → answer) and orchestrator + workers.
- `engine_sim.py`: a continuous-batching simulator (step = a + cB + prefill, KV capacity, priority queue).
- `general_theory.py`, `open_problem.py`: exact simulators and solvers for the proofs.
- `scripts/real_multiagent.py`, `scripts/sim_multiagent.py`, `scripts/profile_batch.py`.
- The single-task runtime and summarization code were deleted.

---

## 2. Theory of the general problem

Files: `docs/general_theory.md`, `docs/open_problem.md`, `docs/verification_report.md`.
Tests: `tests/test_general_theory.py`, `tests/test_open_problem.py`, `tests/test_independent_verification.py`.

### 2.1 Results

| Result | Status | Verifier verdict |
|---|---|---|
| Thm 1: NP-hard, even with no tools and one node per task (reduction from Knapsack) | proved | correct with a wording fix; NP-hard in the weak sense only |
| Lemma 2: an engine with step time a + cB acts like m = a/c identical machines, up to 2× speed | proved in the fluid model | **flawed as stated** for discrete steps; true in the fluid, decode-only model |
| Lemma 3: issuing tool calls as soon as their inputs exist never hurts the offline optimum (uncapacitated tools) | proved | verified |
| Thm 2: if tool durations are unknown until issue, no online algorithm meets all deadlines at speed below 1 + 1/δ (δ = slack) | proved | verified; the threshold is sharp |
| Thm 3: for on-time value, the ratio is unbounded at every speed | proved | verified |
| **Thm 4: global EDF at speed 1 + 1/δ meets every deadline a speed-1 offline schedule meets** | proved | proof gap in step (iii), **repaired** |
| Thm 6: Thm 4 lifted to the batched engine costs 2× speed | proved | fluid, decode-only, unlimited KV |
| Thm 7: evicting KV during every tool wait costs a (1+β) speed factor | proved | an upper bound only; "exactly" overstated |
| Props 8, 9: keeping KV under EDF can need k× memory; a recompute-cost-optimal cache misses at every speed | proved | Prop 9 needs a hypothesis fix |
| R4: Thm 4 with arbitrary arrivals | **refuted** | EDF misses at speeds 2, 9/4, 5/2 with δ = 1 |
| R5: critical-path EDF is speed-optimal with known tools | **refuted** | exact threshold 321/221 |

### 2.2 The Thm 4 model, and why it doesn't cover our workload

**The model:**
- m identical machines with free preemption and migration.
- Each task is a DAG of two kinds of node:
  - GPU nodes with known work;
  - tool nodes with adversarial, unknown durations, unlimited parallelism and no speedup.
- Slack δ: each deadline window is at least (1+δ) × the critical path.
- The benchmark is the offline optimum at speed 1, which knows every tool duration in advance.

**EDF** always runs the ready GPU work of the task with the soonest deadline. Thm 2 and Thm 4 together give the
tight constant s\* = 1 + 1/δ.

**Why it doesn't apply to us:**
- It needs "r_i < r_j ⇒ d_i > d_j", which in practice means all tasks released together. Poisson arrivals
  violate this for nearly every pair, and R4 shows the bound then fails.
- It assumes unlimited KV, no prefill and no preloading.
- What we built sets vLLM priorities. It is not the analysed m-machine EDF.

### 2.3 Status of the constant-factor goal

There is **no constant-factor guarantee for the full problem**:
- caching (4.4) and the composition (4.5) were never done;
- conjectures C1–C3 (constant speed for arbitrary arrivals, known tools, weighted throughput) are open.

Prior work points the same way. Jaillet et al. (arXiv 2502.07115) show that KV-constrained LLM scheduling has
no constant ratio under adversarial arrivals. A restricted version is a possible stretch goal: extra speed and
memory, fan-out tasks only, fluid model.

### 2.4 Corrections still pending in `report/main.tex` (verification report §9)

- Lemma 2 must be stated as fluid and decode-only.
- "All proofs are machine-checked", "exhaustively" and "663 tests" are inaccurate.
- Thm 1 wording.
- Other items as listed in §9.

---

## 3. Calibration and first multi-tenant runs

### 3.1 Serving-stack calibration [GPU]

**Setup:** A40, Qwen3-8B bf16, vLLM 0.32.0 (`results/calibration/`).

- **Decode step:** τ(B) = a + cB with a = 27.8 ms and c = 0.543 ms per sequence. So m = a/c ≈ 51.
  - R² = 0.9992 over B = 1…128; worst point 3% off.
  - B = 256 was excluded: it exceeds the KV cache (159,904 tokens) and forces preemption.
- **Prefill:** about 0.18 ms per token.

### 3.2 Policy sweep

**Setup:** fan-out + orchestrator mix; deadlines 1.5–4× the unloaded completion time; 90 s of Poisson
arrivals.

**Simulator** [sim], measured c, 3 seeds, max 256 sequences. Values are on-time fraction:

| tasks/s | FCFS | EDF | EDF + admission control |
|---|---|---|---|
| 2 | 0.93 | 0.93 | 0.92 |
| 3 | 0.61 | 0.61 | 0.73 |
| 4 | 0.32 | 0.32 | 0.58 |

Capping the batch at 51 sequences changes little: at 4 tasks/s, FCFS 0.33, EDF 0.36, EDF+AC 0.50.

**GPU** [GPU], one seed, uncapped (`results/real/multiagent/uncapped/`). Values are on-time fractions; dropped
tasks in parentheses:

| tasks/s | FCFS | EDF | EDF+AC | LLF+AC |
|---|---|---|---|---|
| 1 | 0.98 | 0.98 | 0.97 | 0.98 |
| 2 | 0.90 | 0.92 | 0.89 (6% dropped) | 0.88 |
| 3 | 0.76 | 0.77 | 0.73 (25% dropped) | 0.76 |
| 4 | 0.59 | 0.61 | 0.65 (34% dropped) | 0.64 |

**Takeaways:**
- Deadline ordering barely matters, because with up to 256 concurrent sequences requests rarely wait in vLLM's
  queue.
- Admission control helps only at 4 tasks/s, and does so by dropping about a third of tasks.
- The real GPU degrades more gently than the simulator: 0.59 vs 0.32 for FCFS at 4/s.

This weak lever is what moved the focus to preloading, where the single-task measurements had shown a large
effect.

---

## 4. KV preloading under contention [GPU]

### 4.1 Setup

**Agent:** a fan-out agent:
1. plan up to 8 searches over a pooled HotpotQA corpus (the question's 10 paragraphs plus 90 distractors);
2. run the searches, each returning its top 10 passages, about 1.4k tokens per result;
3. answer from an append-only evidence log, about 11k tokens.

**Tool delays:** Uniform[0.5, 4] s, or with probability 0.3 Uniform[4, 12] s. They are identical across
policies.

**Arrivals:** Poisson at 0.15 / 0.3 / 0.45 / 0.6 tasks per second for 180 s. Deadlines are effectively off
(slack 1000), so latency is the metric. Every comparison is **paired**: same seed, so the same tasks, arrivals
and delays.

**Preload modes:**
- **off**: no preloading.
- **on**: after each tool result, send a 1-token request on the log so far. At most one is in flight, and it is
  cancelled when the final call is ready.
- **low**: on, but at the lowest vLLM priority.
- **jit**: issued just before the predicted need.
- **chain** (new): whenever a preload finishes and more results have arrived, immediately preload again. The
  final call is sent the moment the last tool returns.

**Instrumentation added:**
- the final call's cached tokens and TTFT;
- the time of the last tool return;
- preloads per task.

The analysis script is `scripts/analyze_preload.py` (paired savings with bootstrap 95% CIs).

**Server setup:** `--scheduling-policy priority`, prefix caching on, max 256 sequences, max length 16,384.
KV is capped with `--num-gpu-blocks-override`: 5000 blocks = 80k tokens, 3125 = 50k, 2500 = 40k.

### 4.2 Full KV cache (about 160k tokens): preloading helps

Mean latency saving vs off, in seconds, with 95% CI:

| seed | rate | on | low | jit |
|---|---|---|---|---|
| 0 | 0.45 | 4.76 [3.51, 6.04] | 4.87 | 5.06 |
| 0 | 0.6 | 6.78 [4.80, 8.88] | 7.14 | 6.78 |
| 1 | 0.45 | 1.71 [0.75, 2.66] | 2.68 | 1.80 |
| 1 | 0.6 | 3.26 [2.12, 4.45] | 2.69 | 2.86 |

- **Modes:** on, low and jit are within noise of each other. Most pairwise CIs include 0.
- **Answers:** 92–98% of answers are identical, with |ΔF1| ≤ 0.024.
- **Seed 2** (`results/real/preload_seeds/`), saving vs off at 0.15 / 0.3 / 0.45 / 0.6 tasks/s:
  - on: 0.86 / 1.89 / 2.26 / 1.49;
  - chain: 1.00 / 2.69 / 3.23 / 3.57.
  - chain beats on by 0.12 / 0.33 / 0.77 / 1.37, growing with load.
  - A rerun of the same "on" configuration drifted by up to 0.7 s.

**Who gains:** about 35–41% of tasks issue only one search. This is probably because the plan's 128-token limit
truncates the JSON list of 8 queries; unconfirmed, to be fixed. These narrow tasks gain about nothing, so the
whole saving goes to wide tasks, a natural control.

**Cache hits on wide tasks:** the share of the final prompt served from cache falls with load, 0.84 → 0.72 for
on and 0.87 → 0.77 for chain. The ceiling is about 0.88.

### 4.3 KV pressure: the sign flips

Seed 2, 0.6 tasks/s unless noted (`results/real/preload_kv/paired.txt`):

| KV cache | off latency | on vs off | low vs off | final prompt cached (on) |
|---|---|---|---|---|
| 80k | 25.0 s | **+2.52** [1.18, 3.86] | — | 0.44 |
| 50k | 25.1 s | **−11.28** [−13.13, −9.37] | −1.71 [−3.19, −0.18] | 0.16 (low 0.28) |
| 40k | 24.5 s | **−11.12** [−12.95, −9.32] | −4.51 [−6.00, −3.03] | 0.12 (low 0.16) |
| 40k at 0.3/s | 17.6 s | +0.65 [−0.37, 1.72] | — | 0.45 |

- Without preloading, latency is about 25 s at every cache size. Only preloading is hurt.
- vLLM logged no preemptions. The harm is preloaded prefixes being evicted before the final call reuses them,
  after their prefill was already paid. At 50k, total prefill computed rose from 716k to 947k tokens.
- Low priority reduces the harm but does not remove it.

**Overhead check** at full KV, 0.45 tasks/s, with tool results capped at 40 or 200 tokens: +0.09 s and +0.15 s,
both CIs include 0. With small results there is little to preload, so the effect is neutral.

### 4.4 Measurement caveats

- The KV-pressure results come from **one seed**.
- Run-to-run drift is up to about 0.7 s.
- The 40k/0.3 and overhead CIs include 0.
- The pod (`duv6d7rvqgqk5q`, A40, $0.59/hr) is stopped.

---

## 5. Theory of preloading, round 1 (`docs/preload_theory.md`)

Code: `src/deadline_agent/preload_theory.py`. Tests: `tests/test_preload_theory.py` (35 pass).

| # | Statement | Status |
|---|---|---|
| T1 | With tool timings outside our control, preloading never lowers cumulative completed work; makespan ≤ off + ε·N | [proved] (reflection monotonicity) |
| R1 | Same claim when the plan calls are also GPU work: false even with zero overhead; per-task harm unbounded | [refuted] |
| T2 | FCFS, equal priority: harm to task j ≤ H_j + ε·N(R_j), tight | [proved] |
| T3 | Preemptive lowest-priority preloads, merged into the final at need, never make any task later | [proved], but vLLM is not strictly preemptive, so this does not apply as-is |
| T3b | A clairvoyant just-in-time preload delays a task by at most ε × (earlier-needed preloads) | [proved] |
| T4 | on / low / jit differ for a task by at most the other tasks' unprocessed preload work | [proved] (idealized); explains why the modes tie |
| P5 | FCFS identity and a wide-vs-narrow test | [proved]; the data rejects the pure "shorter queues" mechanism |
| A1 | Processor-sharing approximation: saving ≈ critical-path work stretched by 1/(1−ρ), for wide tasks only | [approx]; zero-parameter prediction inside all four measured CIs |
| T6 | Overhead break-even (about 230 tokens per result); any GPU overhead shrinks the stable region | [proved] / [approx] |
| K | KV eviction criterion | [proved], but **sized the risk wrongly**: predicted −1.3 s at 40k vs measured −11 s |

**Takeaway:** the saving grows with load because sharing stretches the critical path that preloading removes.
The burst-splitting hypothesis was rejected. Round 1's KV model was too optimistic.

---

## 6. Theory of preloading, round 2: the KV-aware controller (`docs/controller_theory.md`)

Code: `src/deadline_agent/controller_theory.py`, with **M4**, an event simulator that follows vLLM 0.32
semantics. Tests: `tests/test_controller_theory.py` (31 pass; full suite passes).

### 6.1 Model V: vLLM 0.32 behaviour, read from source

- Running requests are scheduled first.
- A waiting request is admitted only if its whole prompt fits. Otherwise the queue stops (head-of-line
  blocking).
- Under priority scheduling, the preemption victim is the request with the largest (priority, arrival).
- Freed blocks join an LRU list, and each request's blocks are freed tail-first.

### 6.2 Mechanism

- **Lemma L** [proved]: a cached free block survives iff its LRU stack distance stays below the free-pool size.
  This is classical stack-distance analysis (Mattson).
- **Lemma Q** [known, from source]: a queued request's cached prefix is unreferenced, so it can be evicted
  until the request is admitted.
  - Under load, the final call waits in the queue while other requests evict exactly the prefix it was
    preloaded for.
  - Round 1's model referenced the prefix at arrival, which explains most of the 10× misprediction.
- **Prop I** [proved]: any unpinned preload can be flooded out. Only pinning certifies survival.
- **Prop S** [proved in M4]: busy time = steps·a + c·(decode slots) + prefill time.

### 6.3 Break-even threshold (Theorem BE′) [approx]

Preloading lowers mean latency iff the surviving fraction of preloaded KV satisfies

  σ > σ\* = ρ_s / (1 − u + ρ_s)

- u is the prefill share of GPU busy time.
- ρ_s is the arrival rate × the slowable compute per task.
- **Here σ\* ≈ 0.92.**

Wasted prefill stretches every tenant, while a surviving preload helps only its own task. So preloads must
nearly always survive.

**Checks:**
- The algebra was checked with Z3.
- The sign is right in 6/6 measured conditions.
- **Caveat:** σ is read from the treatment runs' own waste, so this is a decomposition, not a forecast. One
  estimate (108% at 80k) is impossible and points to a measurement artifact.
- **Approximation F (fluid TTL, Che et al.)** predicts the crossover near 55k tokens, beforehand. That is
  consistent with the measured flip between 50k and 80k.

### 6.4 Safety

- **A0** [proved]: under strict preemptive priority in both GPU and KV, no preload delays anyone.
- **A1** [refuted]: "a lowest-priority preload never delays another task" is false in vLLM. In a 2-task M4
  counterexample, one task goes from 1.62 s to 2.54 s.
- **A-i** [approx]: "the controller is never worse than off" holds only in expectation inside BE′. The worst
  simulated case is −0.12 s.
- **A-iii**: there is no Cao-style constant factor relative to the achievable saving. This follows from Prop I
  and B3.

### 6.5 Which preloads to issue (Target B), and pinning (Target C)

- **B1, B2** [known]: in the fluid relaxation, ranking by saving per unit of memory is LP-optimal (Dantzig).
  The skip/unpinned/pin version is solved by Sinha–Zoltners.
- **B3** [refuted]: the same rule has no constant factor in the time-indexed version. The ratio is 2l−1
  (preemptive) or k (non-preemptive).
- **C** [approx]: pinning pays iff (1−p)·cost(L) > ρ_s·c·h/a. Holding a request open costs batch slots and KV,
  so pinning rarely pays.

### 6.6 Controller specification (implemented: §6.9)

On each tool result:
1. Skip if a preload is already in prefill.
2. Pin if pinning pays and KV demand plus about 11.6k tokens of headroom fits.
3. Otherwise send a low-priority unpinned preload, if P(remaining wait ≤ T̂) > σ\* and the running survival
   estimate σ̂ (after 5 observations) is above σ\*.
4. Otherwise skip.

Pins are released before plan or final calls. The knobs (30 s window, min 5 observations, 0.99 quantile) were
set once and never swept.

### 6.7 Predicted vs measured savings vs off, seconds

M4 used seed-2 arrivals, 8 workload seeds and no fitted parameters.

| condition | on (M4) | low (M4) | **ctrl (M4)** | measured |
|---|---|---|---|---|
| full KV | +2.67 | +3.27 | **+3.27** | on +1.49, chain +3.57 |
| 80k | +2.56 | +3.06 | **+3.01** | on +2.52 |
| 50k | −5.34 | −0.85 | **+1.02** | on −11.28, low −1.71 |
| 40k | −8.78 | −3.09 | **−0.12** | on −11.12, low −4.51 |
| 40k at 0.3/s | +1.22 | +1.49 | **+1.62** | on +0.65 |

**Agreement:**
- M4 reproduces every measured sign and the low > on ordering.
- Its computed prefill tokens are within 12% of vLLM's logs.

**Caveats:**
- M4 under-predicts the 50k harm by about 2×.
- The controller's structure was revised after seeing M4 results.
- The σ̂ fallback never recovers once tripped. It needs an exploration probe.

### 6.8 Proposed decisive GPU test

**Runs:**
- off / low / ctrl / on;
- KV of 160k / 80k / 50k / 40k (10447 / 5000 / 3125 / 2500 blocks) at 0.6/s, plus 40k at 0.3/s;
- seeds 2–4.

**Prediction:** ctrl ≥ low everywhere, and ≥ 5 s better than on at 50k and 40k.

**Cost:** about 3–4 GPU hours (about $2–2.50), plus about 1 h of implementation, including pin mechanics and the
exploration probe.

### 6.9 Implementation (`src/deadline_agent/preload_controller.py`)

- **One `PreloadController` per run, shared by all tasks.** It sees only client-observable data:
  - its own requests and their streamed first-token times;
  - `prompt_tokens` and `cached_tokens` from vLLM's usage fields;
  - the tool-latency law;
  - the server's KV capacity, read from `/metrics` (`cache_config_info`).

  It never modifies the engine.
- **Decision rule:** the theory doc's `decide()` from `controller_theory.py`, reused unchanged. Every estimator
  follows the M4 controller:
  - KV demand D̂;
  - prefill utilization û (cost model applied to computed tokens);
  - fluid TTL;
  - ρ_s;
  - σ̂ from the final calls' `cached_tokens`.
- **Pins:** a lowest-priority preload with `ignore_eos` and max_tokens = ⌈wait quantile / a⌉ + 1, held open on its
  HTTP stream. It is cancelled when superseded, when its task's final call emits its first token, or when another
  task's plan/final call needs the room (`yield_for`). Unpinned preloads are the 'low' mode's 1-token request.
- **Additions not in the theory doc's spec** (stated here because they are post-spec choices):
  1. **Exploration probe.** When only the σ̂ feedback blocks a preload, every 10th such opportunity is still
     preloaded, so σ̂ can recover. `explore_every = 10` is a new knob, set once.
  2. **Size guard.** Survival p = 0 if D̂ + L > M, i.e. the prefix cannot fit beside current demand.
  3. **Token-count calibration.** The client's approximate token counts are rescaled online by the server's
     reported prompt_tokens (ratio ≈ 0.97).
- **Wiring:**
  - `--preload-modes ctrl` in `multiagent.py` and `scripts/real_multiagent.py`.
  - A first-token callback in both backends.
  - Per-run decision logs in `ctrl_r<rate>_fcfs.json`.
- **Tests:** `tests/test_preload_controller.py`, 5 tests:
  - ample KV gives the 'low' behaviour;
  - tiny KV gives no preloads;
  - pins are held open and then released;
  - feedback switch-off plus the probe;
  - a multi-tenant run.

  The full suite passes (7646 tests).

### 6.10 GPU test results [GPU]

**Setup:** the §6.8 design on a new A40 pod (`9wtqy16fleq44n`; the old pod's host had no free GPU). Per KV cap,
one fresh server. Per seed, all four modes run on the same arrival trace.

**Paired savings vs off**, in seconds (positive = faster), with 95% bootstrap CIs on the ctrl column:

| seed (tasks, off mean) | cap | rate | on | low | **ctrl** | ctrl − low |
|---|---|---|---|---|---|---|
| 2 (111, ≈27 s) | full 167k | 0.6 | +3.40 | +2.38 | **+3.44** [2.1, 4.9] | +1.07 [0.5, 1.6] |
| | 80k | 0.6 | +3.09 | +3.28 | **+3.17** [1.9, 4.5] | −0.11 (n.s.) |
| | 50k | 0.6 | −9.81 | −3.37 | **−2.75** [−4.8, −0.7] | +0.62 (n.s.) |
| | 40k | 0.6 | −9.35 | −4.64 | **−0.06** [−2.1, 2.0] | **+4.59** [3.3, 6.0] |
| | 40k | 0.3 | — | — | **+3.27** [2.0, 4.6] | — |
| 3 (100, ≈20 s) | full | 0.6 | +2.94 | +3.19 | **+3.14** [2.0, 4.2] | −0.05 (n.s.) |
| | 80k | 0.6 | +2.82 | +2.56 | **+2.48** [1.3, 3.6] | −0.09 (n.s.) |
| | 50k | 0.6 | −0.20 | +1.38 | **+1.56** [0.5, 2.6] | +0.18 (n.s.) |
| | 40k | 0.6 | −1.30 | +0.15 | **+0.49** [−0.5, 1.5] | +0.34 (n.s.) |
| | 40k | 0.3 | — | — | **+0.51** [−0.3, 1.3] | — |
| 4 (103, ≈22 s) | full | 0.6 | +3.62 | +3.41 | **+3.67** [2.4, 4.9] | +0.27 (n.s.) |
| | 80k | 0.6 | +4.43 | +4.53 | **+3.58** [2.1, 5.0] | **−0.95** [−1.5, −0.4] |
| | 50k | 0.6 | −0.53 | +0.42 | **+1.82** [0.7, 3.0] | **+1.40** [0.5, 2.4] |
| | 40k | 0.6 | −1.36 | +0.52 | **+1.95** [0.75, 3.2] | **+1.43** [0.4, 2.5] |
| | 40k | 0.3 | — | — | **+1.71** [1.0, 2.4] | — |

**Mean over the three seeds**, saving vs off in seconds:

| cap | rate | on | low | **ctrl** | M4 prediction for ctrl |
|---|---|---|---|---|---|
| full | 0.6 | +3.32 | +2.99 | **+3.42** | +3.27 |
| 80k | 0.6 | +3.45 | +3.46 | **+3.08** | +3.01 |
| 50k | 0.6 | −3.51 | −0.52 | **+0.21** | +1.02 |
| 40k | 0.6 | −4.00 | −1.32 | **+0.79** | −0.12 |
| 40k | 0.3 | — | — | **+1.83** | +1.62 |

Complete: 3 seeds × 5 conditions = 15 paired cells, 51 runs, no failures. Pod `9wtqy16fleq44n` was up 3.6 h
(≈ $2.15) and is stopped.

**Against the §6.8 predictions:**
- **"ctrl ≥ low everywhere": holds in 11 of 12 cells.**
  - ctrl is significantly better in 4 cells: seed 2 full (+1.07) and 40k (+4.59); seed 4 50k (+1.40) and 40k
    (+1.43).
  - It is not significantly different in 7.
  - It is **significantly worse once**: seed 4 at 80k, −0.95 s [−1.5, −0.4]. There ctrl sent fewer preloads (1.6
    vs 2.1 per task) and had a lower final cache hit (0.38 vs 0.42). It was too conservative with ample memory.
    The likely causes are σ\* rising under load, which blocks preloads that would have survived, or pin overhead.
    The data does not separate the two.
- **"ctrl ≥ 5 s better than on at 50k/40k": holds only on the heavy trace.** It is 7.1 and 9.3 s on seed 2, but
  1.8–3.3 s on seeds 3 and 4, where "on" itself loses only 0.2–1.4 s. ctrl beats on in all 6 tight-memory cells.
- **"ctrl never worse than off" (A-i): fails once.** Seed 2 at 50k gives −2.75 s; the other 14 cells are ≥ −0.06.
  ctrl is significantly better than off in 11 of 15 cells.
- **Magnitudes:** the three-seed means track M4 closely:
  - full cache: predicted +3.27, measured +3.42;
  - 80k: predicted +3.01, measured +3.08;
  - 40k at 0.3/s: predicted +1.62, measured +1.83.

  The 50k mean (+0.21) is below the prediction (+1.02) because of seed 2. The 40k mean (+0.79) is above the
  prediction (−0.12).
- **Ranking by three-seed mean under memory pressure** (50k and 40k): ctrl > low > on, in both. With ample memory
  (full, 80k) all three are within 0.5 s.
- Answers are unaffected: 90–98% identical to off, |ΔF1| ≤ 0.045.

**Mechanism notes:**
- The controller pins far more than M4 predicted: 37–44 pins per run at 40–50k on seed 2, vs 10–15.
- In the one losing cell, the loss is spread evenly. Tasks the controller never preloaded slowed as much as
  pinned ones (−2.7 s each), and the final call's TTFT rose from 4.3 to 5.4 s. That is the cross-tenant stretch
  BE′ describes, plausibly from pins holding KV while other tasks' finals wait at the admission gate.
- A **pins-off variant** (survival-gated unpinned preloads only) is the obvious next ablation. It is not yet run.
- σ̂ stayed at 0.80–0.97. The fallback rarely binds: 1–2 exploration probes per run.

**Caveats:**
- **Drift across pods.** The same seed and cell gave off means 1–3 s apart across pods (e.g. seed 2 at 40k: 24.5 s
  earlier vs 27.5 s here). Only within-session pairs are compared.
- **One run per (seed, cell).** Seeds differ a lot in load, so the 50k/40k conclusions depend on the trace.

---

## 7. Prior work and novelty

- **The harm is classical.**
  - Aggressive prefetching evicts useful data (Cao, Felten, Karlin & Li, SIGMETRICS 1995: a conservative
    strategy is within 2× of optimal).
  - Albers–Garg–Leonardi (STOC'98).
  - LRU stack distance (Mattson); Greedy-Dual-Size and Landlord; Sleator–Tarjan.
- **LLM-specific work:**
  - Jaillet et al. 2025 (KV-constrained scheduling, no constant ratio);
  - Continuum, KVFlow, InferCept, Autellix, Sutradhara (closest systems work).

- **Closest work, checked 2026-11-07 against the papers themselves:**
  - **Continuum** (arXiv 2511.02230, ICLR 2026). Engine-side: about 1k lines added to vLLM 0.10.2's scheduler.
    - It *pins the previous turn's KV* during a tool call, for a TTL chosen to maximize
      P(tool done by τ)·benefit − (memory share)·τ. Memory cost is charged, but there is no break-even
      threshold and no analysis of harm to other tenants.
    - It does **not** prefill tool results ahead of need. Parallel tools are "natural" but not evaluated.
  - **KVFlow** (arXiv 2507.07400, NeurIPS 2025). Engine-side: SGLang frontend and backend changes.
    - Workflow-aware eviction by steps-to-execution, plus CPU→GPU *transfer* of already-computed KV.
    - No tool-latency model, and no new prefill ahead of need.
    - Prefetch is gated by a fixed "enough evictable memory" check and a concurrency cap. There is no harm
      analysis and no adaptive switch-off.
  - **Sutradhara** (arXiv 2601.12967, Microsoft, Jan 2026). The **closest**: an orchestrator–engine co-design on
    vLLM 0.11 with 5 new engine APIs.
    - It eagerly prefills the *tool-independent* part of the next prompt while tools run, pins those blocks at
      high reuse priority, and splices the tool outputs in when they arrive.
    - It does not preload each tool result as it arrives.
    - It does not study when early prefill hurts under KV pressure.
    - It states that its engine co-design is "strictly necessary" for its pinning.
  - **CacheScout** (arXiv 2608.14624, Aug 2026). An ~800-line vLLM 0.11 patch.
    - It predicts the next agent and sends **warm-up prefill requests** for that agent's reusable anchor
      (system prompt, tool definitions). It also uses a survival × recency × size eviction score.
    - Warm-ups are gated by a predictability threshold. With the gate disabled, warm-ups **raise per-turn latency
      by up to 22% at small cache sizes**. That is independent evidence for our harm finding, in a different
      setting.
    - No break-even analysis, no multi-tenant evaluation, no tool-result preloading.
  - **FlashAgents** (MLSys 2026). SGLang modifications.
    - Streams an upstream agent's tokens into the downstream agent's incremental prefill.
    - Measures compute overhead (up to 3× at small chunks), not KV-pressure harm.
  - **Prompt-cache warming** (folklore; e.g. Anthropic-cache warmers, vllm-mlx warm prompts). A `max_tokens=1`
    request so a prefix is cached before it is needed. **Our preload request is exactly this trick.**
  - **Less close:**
    - Conveyor (tool partial execution during decode);
    - PASTE / speculative tool execution (arXiv 2603.18897);
    - ThunderAgent (2602.13692, program-aware scheduler);
    - AgentServe ("resume prefill");
    - SmoothAgent.

    Of these, only abstracts or snippets were read.
  - Coverage: about 10 targeted searches, not a systematic review. 2026 arXiv moves fast.
- **What is ours:**
  - The others all modify the engine. Ours runs **on stock vLLM, client-side only**, with pins emulated as
    open low-priority requests.
  - the multi-tenant agent-serving measurement, including the sign flip and its mechanism;
  - Lemma Q as the dominant cause in vLLM;
  - the σ\* threshold for batched engines;
  - the pinning result;
  - a client-side, feedback-driven preload controller.
- From the general theory: the tight 1 + 1/δ pair (Thms 2 and 4), for batch releases only.
- **Do not claim** preloading or tool overlap as the idea. **Do claim:** "proactive KV work backfires in
  multi-tenant agent serving: when, why (Lemma Q), the σ\* threshold, and a stock-engine controller that avoids it
  while keeping the gain."

**Novelty grade relative to the course's example projects** (`example_projects/`). This is my judgment:

| project | core idea | main claimed contribution | grade |
|---|---|---|---|
| A: network-aware text / KV / compressed-KV transport between agents | decision already modeled (Dai et al. 2026; CacheGen) | real cross-machine measurement + adaptive controller | C+ / B− |
| B: predict KV-transfer success at prefill (activation routing + cross-model KV transfer) | fresh combination of two 2026 lines; self-described "fairly obvious" | characterization + predictor | B− |
| **Ours** | known trick (cache warming / eager prefill) | the harm under KV pressure, its mechanism, σ\*, and a client-only controller validated on GPU | **B− now; B / B+ if seed 4 and a pins-off ablation hold up** |

We are well ahead of both examples on execution and evidence (paired GPU runs, mechanism from source, theory
with an independent check). On novelty alone we are in the same band.

---

## 8. Code and data map

| What | Where |
|---|---|
| Serving layer and agents; preload modes off/on/low/jit/chain/ctrl; per-task timing and cache fields | `src/deadline_agent/multiagent.py` |
| Client-side KV-aware preload controller | `src/deadline_agent/preload_controller.py`, `tests/test_preload_controller.py` |
| GPU experiment driver (`--preset`, `--preload-modes`, `--set KEY=VALUE`, `--delay`, `--kv-tokens`) | `scripts/real_multiagent.py` |
| Server launcher (`KV_BLOCKS`, `GPU_UTIL`, `SCHED`, `PREFIX_CACHE`) | `scripts/serve_model.sh` |
| KV-pressure sweep; controller GPU test (both run on the pod) | `scripts/kv_pressure.sh`, `scripts/ctrl_test.sh` |
| Paired analysis | `scripts/analyze_preload.py`, `scripts/analyze_preload_jit.py`, `scripts/analyze_ctrl.sh` |
| Results | `results/real/{multiagent,preload_contention,preload_jit,preload_seeds,preload_kv,preload_ctrl}/`, `results/sim/`, `results/calibration/`, `results/logs_pod/` |
| Theory | `docs/{general_theory,open_problem,verification_report,preload_theory,controller_theory}.md` |
| Tests | `tests/` (full suite passes; run with `.venv/bin/python -m pytest -q`) |

---

## 9. Open items

1. **Midterm (due 2026-11-08, noon).**
   - `report/main.tex` still describes the single-task slice and the simulated policy sweep.
   - It needs refocusing on the KV-pressure result and the real GPU numbers.
   - It needs the verifier's §9 corrections and a citation of Cao et al.
   - The σ\* threshold should be labelled approximate.
   - The controller result (§6.10) can now go in as a measured result.
   - Cite CacheScout and Sutradhara.
2. ~~Finish seed 4 of the controller test, and stop pod `9wtqy16fleq44n`.~~ Done (§6.10); the pod is stopped.
3. **Pins-off ablation** of the controller at 50k/40k, the likely cause of the seed-2 50k loss. Add a 65k cap to
   locate the crossover.
4. **Fix plan truncation** (plan max_tokens 128) before the final evaluation.
5. **Update `PLAN.md`** to the preload/KV framing. Its §3–6 predate this work.
6. **Housekeeping.**
   - ~~Commit the uncommitted work.~~ Done (`93074d8`, `d8ad76d`).
   - Decide whether to delete the old stopped pods `apu73vvxwlk9qy` and `duv6d7rvqgqk5q`. Only their disks
     bill; `duv6d7rvqgqk5q`'s host had no free GPU on restart.
