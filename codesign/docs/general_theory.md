# General theory: deadline-aware serving of agent workloads on a shared GPU

Status: DRAFT (started 2026-11-07). The project's theory. (The earlier single-agent, summarization-based theory
was removed on 2026-11-07; it remains in git history at commit `4e56d77`.)
Claims are marked **[proved]**, **[known; cite]** (classical result, citation to be verified before use),
or **[target]** (to be proved; not yet established). Nothing here is a measurement.

## 1. Model

**Engine.** One GPU running an iteration-level (continuous-batching) LLM engine such as vLLM.
- Each engine step processes a batch: decode one token for each of B running sequences, plus up to P prompt
  (prefill) tokens. Step time τ(P, B) is measured by profiling; for decode it is roughly a + c·B with a ≫ c
  at small B (memory-bandwidth bound), so batching raises throughput but slows each sequence.
- KV memory capacity M tokens (A40 + Qwen3-8B, measured: 159,904–163,904 depending on the host). A sequence occupies KV equal to its
  current length. Prefixes can be shared (prefix caching), kept across pauses, evicted, swapped, or
  recomputed. Preemption of a sequence is allowed at step granularity; resuming costs recompute or swap.

**Tasks** (setting a: many tasks share the GPU). Task j arrives at r_j with deadline d_j and value w_j.

**Agent programs** (setting b: collaborating agents inside one task). A task is a directed acyclic graph of
- LLM nodes: a model call by some agent (orchestrator, worker, verifier, ...), with prompt built from its
  agent's context and the outputs of predecessor nodes; output length is random;
- tool nodes: an off-GPU call (search, code execution, API, another service) with random duration drawn
  from a per-tool latency distribution estimated from history;
- edges: data dependencies. Every node's predecessors must finish before it starts, EXCEPT as allowed by
  asynchronous consumption (below). A task completes when its sink node finishes.

**Sync vs async tool calls.** A tool node t issued by agent A can be
- synchronous: A's next LLM node waits for t;
- asynchronous: A continues with LLM nodes that do not need t, and t's result is consumed at a later LLM
  node of A (its "consumption point"). The runtime chooses the consumption point, subject to the program's
  true data dependencies. Synchronous = consumption point is the next node.

**Runtime decisions** (the scheduling policy), made online without knowledge of future arrivals, tool
durations or output lengths:
1. batching: which ready LLM nodes (across all tasks and agents) run in each engine step, and how many
   prefill tokens to admit;
2. KV management: which prefixes to keep, evict, swap or preload (prefill ahead of need while waiting);
3. tool execution: sync vs async issue and consumption points; optionally which tool calls to launch at all
   (admission);
4. optional model work during idle GPU time, chiefly KV preloading (prefilling a prompt prefix before the
   node that needs it is ready).

**Objective.** Maximize the total value of tasks completed by their deadlines (weighted deadline goodput),
or minimize weighted tardiness. Single-task completion time and throughput are special cases.

## 2. What cannot be done in general

**Theorem 1 (NP-hardness) [proved, by restriction].** The offline problem is NP-hard even with: one GPU,
batch size 1 (no batching), no tools, unlimited KV, deterministic durations, every task a single LLM node
with known output length, all r_j = 0.
*Proof.* With these restrictions each task is one job of known processing time p_j (prefill + decode of its
node), the machine processes one job at a time, and the objective is to maximize Σ w_j over jobs finishing by
d_j, i.e. minimize the weighted number of late jobs, 1 || Σ w_j U_j. That problem is NP-hard by reduction from
Knapsack: give every job the common deadline d = knapsack capacity, p_j = item weight, w_j = item value; a set
of jobs can all finish by d iff its total processing time is ≤ d, so maximizing on-time value is exactly
Knapsack [known; cite Karp 1972]. Since our problem contains it as a special case, it is NP-hard. ∎

Remarks.
- With release dates and no preemption, even the unweighted version 1 | r_j | Σ U_j is strongly NP-hard
  [known; cite Lenstra, Rinnooy Kan, Brucker 1977]. LLM engines can preempt at step granularity, so the
  preemptive variants are the relevant ones; preemption under a KV budget has a cost (recompute or swap),
  which keeps the non-preemptive hardness relevant when memory binds.
- Consequence: an algorithm that is optimal on every instance and runs in polynomial time does not exist
  unless P = NP. Section 4 states what can be guaranteed.

**Online lower bounds [known; cite, to verify].** For online scheduling with deadlines on one preemptive
machine, no deterministic online algorithm achieves more than 1/4 of the optimal value even when value is
proportional to length (Baruah et al. 1992), and with arbitrary values no constant ratio is possible. So the
general online problem needs either restrictions (Section 3) or resource augmentation (Section 4).

## 3. Exact optimal algorithms for structured subclasses

| Subclass | Result | Status |
|---|---|---|
| One task, fork-join (n async tool calls, then one answer node), i.i.d. tool delays; KV preloading of each returned result while others are outstanding | Preload each result as it returns is optimal when the context fits | [target] (was proved in the removed single-task framework; re-prove in this model) |
| Many single-node tasks, preemptive, no batching, unit values | Polynomial exact algorithm exists for 1 \| r_j, pmtn \| Σ U_j | [known; cite Lawler 1990] |
| Many tasks, batching with step time a + c·B, common structure (e.g. agreeable deadlines) | Optimal batch formation rule | [target] |
| Single agent, sync vs async consumption point for each tool, deterministic durations | Choose consumption points by DP over the agent's node sequence | [target] |

## 4. General online algorithm: constant-factor guarantees (the main theory goal)

Decision (2026-11-07): the problem is solved online, so the target is a **constant-factor competitive online
algorithm**, not exact optimality. By Section 2, a constant factor against the clairvoyant optimum needs
either resource augmentation (the online algorithm gets a constant-factor faster GPU and/or more KV memory
than the optimum it is compared with) or bounded value densities. The plan is to reduce each resource to a
classical online problem with known constant-factor algorithms, then compose.

### 4.1 Batching reduces to identical parallel machines

**Lemma 2 (batching reduction) [proved, decode-only; prefill in mixed steps not yet covered].** Let a decode
step with B running sequences take τ(B) = a + c·B seconds (a, c > 0), so each sequence advances at rate
ρ(B) = 1/(a + c·B) tokens/s. Assume m := a/c is an integer ≥ 1. Let M_m(s) denote m identical machines of
speed s tokens/s with preemption and migration. Then
(i) any schedule feasible on M_m(1/(2a)) can be executed by the engine, and
(ii) any engine schedule can be executed on M_m(1/a).
*Proof.* (i) At any time M_m runs at most m jobs, so the engine batch is B ≤ m = a/c and every running
sequence advances at rate 1/(a + cB) ≥ 1/(2a), at least its rate on M_m(1/(2a)); a job that gets ahead can
idle, so every completion time is no later. (ii) An engine schedule running B sequences gives each rate
1/(a + cB) ≤ 1/a and total rate B/(a + cB) < 1/c = m·(1/a). A rate vector whose entries are ≤ s and whose sum is
≤ m·s is realizable on m identical machines of speed s with preemption and migration over any interval
(McNaughton's wrap-around rule [known; cite McNaughton 1959]). Applying this over each interval in which the
engine's batch is constant reproduces every job's progress, so every completion time is preserved. ∎

**Consequence.** If an online algorithm for m identical machines is C-competitive when given speed σ·s
against an optimum with speed s, running it on the engine as in (i) is C-competitive against the engine's
optimum with speed augmentation 2σ (the factor 2 is the gap between M_m(1/(2a)) and M_m(1/a)). If a/c is not
an integer, using m = ⌊a/c⌋ costs a further constant factor (≤ 2 when a/c ≥ 2). Measured on the A40
(Qwen3-8B, vLLM 0.32.0, 2026-11-07, `results/calibration/batch_profile.json`): a = 27.8 ms, c = 0.543 ms per
sequence, so m = ⌊a/c⌋ = 51; fit over B = 1…128 with R² 0.9992 and at most 3% error per point.

### 4.2 Tool calls: issue eagerly, consume as late as the program allows

**Lemma 3 (eager async issue) [proved].** Suppose tool calls run off the GPU with no shared capacity limit,
their durations do not depend on when they are issued, and their results do not depend on when they are
consumed. Then issuing every tool call as soon as its inputs are available never makes the offline optimum
worse.
*Proof.* Issuing a tool earlier makes every node that depends on it available no earlier than before
(release times only decrease). Any schedule that was feasible stays feasible, so the optimum cannot get
worse. ∎
Limits: false when tools are rate-limited, cost money per call, or results depend on timing; for an online
algorithm, earlier releases are not automatically better (online algorithms need not be monotone). With
eager issue, every task becomes a stream of LLM-node jobs released online as their inputs arrive, which is
the setting of 4.3. "Sync" is then the special case where nothing else is ready to run.

### 4.3 Deadlines on LLM-node jobs [target]

Each LLM node gets an intermediate deadline derived from its task's deadline and the remaining critical path
(deadline decomposition). Candidate algorithm: deadline-ordered admission into the batch with value-density
admission control, on the m-machine model of 4.1. Known building blocks [known; cite, to verify]: on one
preemptive machine, D^over (Koren and Shasha 1995) is optimally competitive, with ratio 1/(1+√k)² where k is
the ratio of largest to smallest value density (1/4 when values are proportional to work, matching Baruah et
al.); with speed augmentation, simple deadline-ordered policies become constant-competitive. To do: (a) pick
the multi-machine result we build on and verify it; (b) bound the loss from deadline decomposition.

### 4.4 KV keep / evict / preload is caching [target]

Keeping a paused agent's KV during a tool wait vs evicting it (and paying recompute or swap on resume) is an
online caching problem with variable sizes (sequence lengths) and costs (recompute time). Known [cite, to
verify]: for paging, LRU is k/(k−h+1)-competitive with cache k against an optimum with cache h (Sleator and
Tarjan 1985), i.e. 2-competitive with twice the memory; Young's Landlord algorithm extends this ratio to
variable sizes and costs (file caching). Our policy: evict by estimated time until reuse (the expected tool
return time, from the latency distribution), which approximates Belady's offline rule. Preloading during idle
GPU time is the single-task operator of Section 3 (Prop B2), generalized.

### 4.5 Composition [target]

Combine 4.1–4.4 into one policy and prove a constant competitive ratio under constant speed and memory
augmentation. Composition is not automatic: GPU time and KV memory interact (evicting saves memory but costs
GPU recompute). The proof must account for that coupling, e.g. by charging recompute to the speed
augmentation. Also check that on the Section 3 subclasses the policy matches or nearly matches the exact
optimal policies.

## 5. Relation to what is built and measured

- Built: the multi-tenant serving layer (`multiagent.py`: per-request priority fcfs/edf/llf, admission
  control, fan-out and orchestrator+worker agent programs, eager async tool issue, KV preloading during tool
  waits), a simulated continuous-batching engine (`engine_sim.py`), and the vLLM integration
  (`--scheduling-policy priority`). Plan and status: `PLAN.md`.
- Measured (A40, Qwen3-8B, vLLM 0.32.0): decode step time a + c·B with a = 27.8 ms, c = 0.543 ms/sequence,
  m = a/c ≈ 51 (B = 1…128, R² 0.9992), which instantiates Lemma 2; single-task KV preloading cuts final-call
  latency and deadline misses (`PLAN.md` Section 5).
- Not yet built: KV keep/evict policy, batch-size cap at m with prefill admission, a custom vLLM scheduler.
