# Plan: approximation algorithms for agent serving with stochastic tool calls, bounded KV and batching

Branch `stochastic-theory` (created 2026-11-07 from `preload-controller`, commit `93074d8`, which saves all the
preloading work: controller, GPU test, progress report). This document lays out the new direction:
- the model;
- every theorem we need, with how each could be proved and where it could fail;
- the online algorithm;
- the vLLM co-design;
- experiments;
- a literature check of what is new.

Status labels:
- **[target]**: to be proved.
- **[known]**: verified in a cited source.
- **[known?]**: recalled or seen only in an abstract or secondary source; check before relying on it.

---------------------------------------------------------------------------------------------------------------

## 0. Goal in one paragraph

Many LLM agents share one GPU.
- Each agent task is a DAG of LLM calls and tool calls. Tool durations and output lengths are random, with known
  (or estimated) distributions.
- The engine batches. A step that decodes B sequences and prefills P tokens takes τ = a + c·B + α·P.
- KV memory is bounded by M. A waiting agent's prefix can be kept (it holds memory), evicted (it must be
  recomputed later), or, as an extension, swapped.
- Objective: **E[Σ_j w_j C_j]**, i.e. weighted mean task latency.

We want:
1. an **offline (stochastic) approximation**: a non-anticipatory policy within a constant factor ρ(Δ) of the
   optimal non-anticipatory policy, with stated resource augmentation (speed, and possibly memory);
2. an **online version** for tasks that arrive over time;
3. a **deployable algorithm** for vLLM;
4. an **engine co-design**, where each engine change is exactly what makes a theorem's hypothesis true in the
   real engine;
5. a GPU evaluation.

Preloading becomes one action inside the model, not the headline.

## 1. Model (to be fixed precisely in `docs/stochastic_theory.md`)

**Tasks.** Task j has a release time r_j and a weight w_j. Its DAG G_j has three kinds of node:
- **LLM node v:**
  - prompt length L_v, known when v becomes ready: the parent context plus appended tool outputs;
  - output length X_v, random with known law, independent across nodes;
  - prefill work α·(L_v − cached) followed by X_v decode steps.
- **Tool node u:** duration D_u, random with known law, independent; unlimited parallelism; not sped up.
- **Prefix-sharing edges:** child LLM node v′ extends parent v's context, so v′ can reuse v's KV if v's KV is
  still resident.

**Engine.**
- Discrete steps. A step decodes one token for each of B running sequences and prefills P tokens (chunked
  prefill, P ≤ P_max). Step time τ(B, P) = a + c·B + α·P.
  - Measured: a = 27.8 ms, c = 0.543 ms, α ≈ 1.1–1.8 × 10⁻⁴ s per token. An attention term κ·P·ctx is optional.
- Fluid relaxation: decode slot-rate and prefill token-rate are shared continuously. Theorems state which model
  they hold in.

**Memory.**
- Blocks of 16 tokens; total capacity M tokens.
- A running sequence holds prompt + generated tokens.
- After an LLM node finishes, its KV is in one of three states:
  - **held**: occupies memory, reusable by the child;
  - **evicted**: frees memory; the child pays recompute of the missing prefix;
  - **swapped** (extension): frees GPU memory and pays a transfer delay modeled like a tool.

**Information.**
- The policy knows the distributions, the DAG structure revealed so far, and realized durations once they
  complete.
- **Benchmark:** OPT = the optimal non-anticipatory policy at speed 1 with memory M. This is the standard
  stochastic-scheduling benchmark.
- The clairvoyant optimum is **not** the benchmark: `docs/open_problem.md` Thm 2/3 show that comparison is
  hopeless.

**Variability parameter.** Δ ≥ max over random quantities of Var[X]/E[X]² (squared coefficient of variation), as
in Möhring–Schulz–Uetz.

**Resource augmentation** is allowed, and stated explicitly:
- speed s (all GPU work runs s times faster);
- memory μ·M.

## 2. Results we need

Each item gives the statement, how to prove it, building blocks, and risk. The theory agent's job is to prove,
refute, or narrow each one, and to machine-check what it can.

**R0. Engine reductions.**
- R0a [target], mixed-step Lemma 2′ in the fluid model:
  - an engine with τ = a + cB + αP, batch capped at m = a/c, is simulated by m identical machines plus a
    prefill "machine";
  - and conversely;
  - each direction loses at most a factor of 2 in speed.

  Lemma 2 of `docs/general_theory.md` is the decode-only fluid case. The verifier showed (verification report
  §2.2) that it is **false for discrete steps**. So also state:
- R0b [target]: the best discrete-step version (rounding loss), or a counterexample.
- R0c [target], memory → machine count: if every active sequence's context is ≤ L_max, any policy that runs at
  most m_eff = min(a/c, ⌊μM/L_max⌋) sequences never exceeds memory.
  - Bound the loss vs OPT, which may run more short sequences.
  - Better: use an area (memory × time) LP constraint as in the KV-scheduling literature (§6) instead of the
    crude cap.

**R1. Recompute charging.** [target]
- The evict-always policy (never hold KV across a tool call) is feasible with only active-sequence memory.
- It costs at most a (1+β) speed factor, where β bounds recompute work relative to node work.
- This extends Thm 7 of `docs/open_problem.md` to mixed steps and random sizes.
- Consequence: the memory constraint reduces to R0c plus speed (1+β). Keep/evict then becomes a cost
  optimization, not a feasibility question.

**R2. Offline stochastic approximation (the core).** [target]
- Problem: identical parallel machines, release dates, precedence DAG whose arcs carry **random time lags**
  (tool nodes), random processing times. Objective E[Σ w_j C_j], where a task completes when its sink completes.
  Prove that an LP-based list policy is a ρ(Δ)-approximation.
- **Building blocks:**
  - Möhring–Schulz–Uetz (JACM 1999), LP relaxation for stochastic scheduling (polymatroid-type inequalities in
    expected completion times) [known];
  - Skutella–Uetz (SICOMP 2005), constant-factor policies for precedence constraints and release dates on
    identical parallel machines, using LP order plus Chekuri–Motwani–Natarajan–Stein idle-time charging [known];
  - Megow–Vredeveld, stochastic online scheduling with precedence constraints [known?].
- **New step:** tool lags.
  - Idea: model each tool as a job on an extra machine with unlimited capacity (no GPU contention), so its delay
    enters only the precedence chain.
  - Check that the LP stays valid for policies that may idle GPUs during lags.
  - Check that idle-time charging still works when idle time is caused by lags rather than by unavailable
    machines.
  - The min-sum objective with time lags was **not** found in the literature search (§6). It needs a careful
    search ("generalized precedence", "minimal time lags", "delays") before claiming novelty.
- **Task-level weights:** only sinks carry weight. Use the standard trick of weight on the sink plus
  ε-weights, or the task-completion formulation of Skutella–Uetz if available.
- **Preemption:** the batched engine can pause sequences at token boundaries, so the benchmark may preempt.
  - Decide whether the LP is valid against preemptive non-anticipatory policies.
  - Megow–Vredeveld have preemptive stochastic results [known?]. Otherwise restrict OPT to non-preemptive
    policies and say so.
- **Risk: high.** This is the main proof.
- **Fallback:** fan-out DAGs only (plan → k parallel tools → final). That is our workload, and the structure is
  much simpler.

**R3. Composition: engine guarantee.** [target]
- R0 + R1 + R2 give a policy for the real engine model with
  E[Σ w C] ≤ ρ(Δ) · OPT(speed 1, memory M), at speed s = 2(1+β)·(R0 loss) and memory μM (μ = 1 or 2).
- State all hypotheses. Say exactly what transfers to discrete steps.

**R4. Online arrivals.** [target]
- Extend R2/R3 to tasks revealed at r_j (non-anticipatory about future tasks).
- Building block: Megow–Uetz–Vredeveld (MOR 2006), stochastic online scheduling on parallel machines with ratio
  depending on Δ [known?]. Megow–Vredeveld add precedence [known?].
- Target: ρ′(Δ) with the same augmentation.
- Also: a lower bound showing that some dependence on Δ is necessary [target], and the dependence on augmentation.

**R5. Keep / evict / swap layer: multi-tenant ski rental.** [target]
- Per prefix and per tool wait: holding costs λ·L per second (λ is the shadow price of memory); evicting costs
  recompute(L) on reuse.
- (a) With a known duration law, the optimal single-prefix rule is a threshold (TTL) [known for stochastic ski
  rental?]. State it and its ratio against a clairvoyant keep/evict.
- (b) **Many tenants:** λ is the dual variable of the memory constraint in an LP for the keep/evict layer. Show
  that the break-even survival σ\* of `docs/controller_theory.md` (Thm BE′) is this dual condition. That ties the
  old work in.
- (c) **Safety theorem:** if held KV is evicted strictly before any active sequence's memory (lowest eviction
  priority), then any keep policy preserves R3's guarantee. Holding becomes a free improvement.
  - This is A0 of `controller_theory.md` in the new model.
  - Its hypothesis is false in stock vLLM, by Lemma Q and the A1 counterexample. That is the theoretical reason
    for the engine co-design (§4).

**R6. Preloading as node splitting.** [target]
- A final node's prefill splits into chunks, and chunk i is ready when tool i returns.
- Under R5(c)'s eviction discipline, plus lowest scheduling priority for chunks that start early, the guarantee
  of R3 still holds. Connects to T3 of `preload_theory.md`.
- Also state when early chunks *strictly* help: the isolated-task saving P4 of `preload_theory.md`.

**R7. Lower bounds and necessity.** [target / known]
1. Without stochastic information, adversarial tool durations force unbounded speed (Thm 2/3, existing).
2. Without recompute (hold-only), no constant ratio: memory deadlock or blocking instance.
3. Under adversarial arrivals, flow time has no constant ratio (Jaillet et al. 2025 [known]). This is why the
   objective is completion time.
4. Unpinned held KV can be flooded out (Prop I, existing).
5. Dependence on Δ is necessary (from the stochastic scheduling literature, or a new construction).

**R8. What the engine must provide.** [target, as a checklist]
For each theorem, the exact engine property its proof uses:
- P1: a queued request references its cached prefix (fixes Lemma Q).
- P2: eviction priority classes (held KV < active KV).
- P3: admission by the policy's index, without the head-of-line `break` gate.
- P4: per-step prefill budget set by the policy.
- P5 (optional): swap as a timed delay.

## 3. The deployable online algorithm (sketch, to be fixed by R2/R4/R5)

**Index.** Each ready LLM node gets an LP-derived priority. The fallback is a weighted
shortest-expected-remaining-critical-path rule, using expected lags.

**Each engine step:**
- admit and run the top-index nodes, up to m_eff (R0c);
- set the prefill budget P by index (R0a);
- held KV sits in the lowest eviction class (R5c).

**On each tool wait:** keep, evict or swap by the R5 threshold against λ̂. λ̂ is estimated online from memory
pressure: the dual price, or the σ̂ feedback already built in `preload_controller.py`.

**Preload chunks** run at the lowest priority under R6.

**Deployment, in layers:**
1. A client-side layer: what we have now in `preload_controller.py` / `multiagent.py`.
2. A vLLM scheduler subclass (`--scheduler-cls`) plus a small block-pool patch.

## 4. vLLM co-design (each patch tied to a theorem hypothesis)

| patch | engine change (vLLM 0.32) | theorem hypothesis it makes true | measurable prediction |
|---|---|---|---|
| E1 | waiting requests take a soft reference on their longest cached prefix (kv_cache_manager / block_pool) | P1 (Lemma Q) | the −11 s preload harm at 40–50k shrinks sharply; σ rises toward 1 |
| E2 | free-block queue ordered by an index instead of LRU: (1 − survival)·recompute ÷ (memory × expected wait), with client hints via a request field | P2, R5 | higher cache hit at equal memory; ctrl ≥ low at every cap |
| E3 | admission by policy index, no head-of-line `break` | P3 | lower final-call queue time under pins |
| E4 | per-step prefill budget from the policy (chunked-prefill token budget) | P4, R0a | lower decode stretch at high load |
| E5 | swap via the KV offload connector as a third option | P5 | gains at very small M |

E1 is small and has the clearest prediction, so it comes first.

**Status (2026-11-07).** In `vllm_sched.py` the numbering follows `vllm_codesign_spec.md`, not this table. Done
there: prefix admission with kicks (this table's E3), queued-prefix protection (E1 here), the prefill budget
(E4 here), soft TTL eviction (a TTL-only version of E2 here) and the preload class. Not implemented: the value
index for the free-block queue (E2 here, beyond TTL), swap through the offload connector (E5 here), and the
online memory price. Results and next steps: `experiment_plan.md` §6–7.

## 5. Experiments

- **Simulator:** extend M4 (`controller_theory.simulate_vllm`) into the R-model, with policies A/B, to check every
  theorem's predicted direction before using GPU time.
- **GPU** (A40, Qwen3-8B, vLLM 0.32): the same paired design as `scripts/ctrl_test.sh`. Workloads:
  - fan-out;
  - orchestrator;
  - multi-turn (several LLM nodes per task).

  Arms:
  - stock vLLM + client controller (current);
  - + E1;
  - + E1 + E2;
  - + the full R-policy.

  Metrics:
  - E[Σ w C], mean and p90 latency;
  - cache hit;
  - recompute tokens;
  - final-call queue time.

## 6. Literature check: is this new? (searches 2026-11-07)

**LLM scheduling theory with KV-memory constraints.** All single LLM calls: no DAGs, no tool waits, no random
durations.

| work | model | result | gap vs us |
|---|---|---|---|
| Jaillet, Jiang, Mellou, Molinaro, Podimata, Zhou, arXiv 2502.07115 (2025) | batch rounds, per-token KV growth, memory M | no constant ratio for adversarial online arrivals (latency); semi-online optimal; MC-SF heuristic; constant regret for stochastic arrivals | single calls, no tools / DAG / batch-dependent step time |
| Feng, Liu, Yang, Zhang, arXiv 2607.09248 (Jul 2026) | unit-time rounds; KV = prompt + decoded; kill-and-restart | **O(1) non-clairvoyant for total completion time**; 16 clairvoyant; 3 + o(1) large memory; area-order lower bound; regime routing | single calls; step time independent of batch; no prefill cost; no tools; worst case, not stochastic; unweighted |
| Feng et al. 2026; Wang et al. 2025; Kong et al. 2026 (cited by 2607.09248) [known?] | similar | 61.92 / 32 non-clairvoyant; 48 + ε; 3 + o(1) | same gaps |
| Ao, Luo, Simchi-Levi, Wang, arXiv 2504.11320 (NeurIPS 2025) | fluid model, memory overflow | WAIT / Nested WAIT, near-optimal throughput in heavy traffic | throughput, single calls |
| Li, Dai, Peng, arXiv 2504.07347 (2025) | multi-class batched queueing networks, including DAG / fork-join agent routing | work-conserving policies are throughput-optimal for one engine; not enough for agent networks | stability / throughput, not completion-time approximation; no KV memory or tool waits |

**Stochastic scheduling** (the toolbox for R2/R4):
- Möhring–Schulz–Uetz (JACM 1999);
- Skutella–Uetz (SICOMP 2005), precedence constraints and release dates on parallel machines;
- Megow–Uetz–Vredeveld (MOR 2006), stochastic online scheduling;
- Megow–Vredeveld, stochastic online scheduling with precedence constraints;
- Skutella–Sviridenko–Uetz (2016), unrelated machines.

**Min-sum objective with time lags** (tool waits as precedence delays): not found in this search. Needs a
targeted check.

**Systems** (no approximation proofs):
- Continuum (TTL pinning during tool calls);
- SAGA (arXiv 2605.00528): workflow-aware LRU with a tool-latency TTL. Its "1.31× of Bélády" is empirical only.
- CacheScout, KVFlow, Sutradhara, TokenCake (arXiv 2510.18586), FlashAgents.

**Verdict (from targeted searches, not a systematic survey).** No work found combines:
1. agent DAGs with random tool lags;
2. KV held across tool waits with keep/evict/recompute;
3. a batch-dependent step time with prefill cost;
4. a stochastic approximation guarantee for E[Σ w C].

Each ingredient exists separately:
- KV-constrained single-call scheduling: Feng et al., Jaillet et al.;
- stochastic precedence scheduling: Skutella–Uetz;
- TTL keep/evict: Continuum, SAGA.

So the combination, the lag extension and the R5/R6 safety results would be new.

**Main novelty risks:**
- the KV-scheduling theory community (Feng et al., Jaillet et al.) is active in 2026 and could publish the agent
  version;
- the time-lag extension may already exist in the scheduling literature;
- the composed constant could be large. Feng et al.'s constants run 16–80, so a "constant" alone may not impress.

**Positioning:** "the first approximation guarantee for agent serving with tool waits", plus "engine properties
P1–P4 are exactly what the guarantee needs, and stock vLLM violates P1 (measured: −11 s)".

## 7. Timeline (final report due 2026-12-16), fallbacks and verification

| weeks | theory | systems |
|---|---|---|
| 1 | R0a/c, R1, model write-up; literature check on time lags | finish the preloading writeup (midterm); E1 patch prototype |
| 2–3 | R2: fan-out first, then general DAGs; R5 | E1 + E2 on GPU; M4 extended to the R-model |
| 3–4 | R3, R4, R6, R7 | R-policy as scheduler subclass; full GPU evaluation |
| 5 | write-up | write-up |

**Fallbacks, in order:**
1. fan-out DAGs only;
2. add memory augmentation μ = 2;
3. fluid model only (state the discrete gap);
4. non-preemptive benchmark.

**Verification standard** (same as `docs/verification_report.md`):
- every proof written out;
- algebra checked with Z3;
- exact small-instance checks: exact optimal non-anticipatory policy by dynamic programming on tiny discrete
  distributions; exact LP;
- randomized property tests;
- adversarial search for counterexamples near claimed constants;
- labels: proved / known (with verified citation) / approximation / conjecture / refuted.

Do not claim a ratio without an exact check on small instances.
