# Theory-Guided Deadline Scheduling for Asynchronous LLM Agents

## 0. Instructions to the implementing agent

Treat this as an implementation brief. Build executable components rather than returning another plan. Read the repository and applicable AGENTS.md first; preserve unrelated work and reuse suitable code. Make routine implementation decisions autonomously and record them.

Use theory explicitly: define the restricted scheduling problem, implement an exact subset dynamic program, verify it against exhaustive enumeration, then build a causal online controller using measured cost estimates. Connect the controller to a real vLLM server and produce an initial quantitative result. A theory-only writeup or simulator alone does not satisfy the systems-building requirement. No model training is planned.

The first work session should produce the CPU-only exact solver, its assumptions/proof sketch, meaningful correctness checks, an example schedule, and the next executable command. Continue to the runtime and real-model smoke experiment if access exists. If a GPU or credentials are missing, finish CPU work and provide precise deployment commands; never invent measurements.

Owner context: systems/kernel and vLLM experience. Preserve the $80 working-cost ceiling in Section 3.

**Deadline and compute (updated 2026-11-06).** The midterm report is due 2026-11-08 at noon (about 40 working hours away); scope is cut to the Section 10 midterm slice. GPU work runs on **RunPod (primary) or Lambda Labs (fallback)**, paid by the owner. An AWS G-instance quota request (us-east-1, 8 vCPU) is pending and is only a backup. The expected midterm GPU spend is about $10–20.

**Progress:** Gate 0 theory is implemented and verified (`docs/theory.md`, `src/deadline_agent/{restricted,affine,online_mdp}.py`, `tests/test_theory.py`). It goes beyond the subset DP: a fold-gap bound, an O(n²) exact solver for token-count costs, the exact optimal *online* policy for Experiment A's setting, and a distilled heuristic (RSS). See Section 6.5a.

Use the pasted plan as the starting point. This revision makes adaptive result consumption the main experiment, supported by a restricted exact scheduling model. Full quality-optimal agent control and stochastic risk guarantees remain extensions.

## 1. Research question, contribution, and scope

**Overall research question:** When should an agent launch another asynchronous call, continue independent reasoning, or stop expanding the task and process all outstanding results, to maximize answer quality before a completion deadline?

**First paper-sized question:** For asynchronous tool results feeding one serialized model slot, when should the runtime process results individually, aggregate several in one invocation, or leave raw results for finalization? Can a controller informed by a restricted scheduling model improve deadline-qualified answer quality over well-tuned static choices?

Keep the broader question, but distinguish three deliverables:

| Deliverable | What is established | What is not established |
| --- | --- | --- |
| Restricted deterministic model and exact solver | Minimum modeled completion time for a fixed admitted call set under explicit assumptions | Optimal real-agent quality, launch selection, or stochastic online behavior |
| Practical online controller | A causal policy using observed results and predicted future costs | Knowledge of actual future tool returns or an unconditional deadline guarantee |
| Real vLLM evaluation | Measured timing, quality, prediction errors, and comparison with strong baselines | Universal dominance across workloads or inference configurations |

**Main hypothesis H1:** The best fixed processing mode varies with result-arrival pattern, input size, invocation overhead, and finalization cost. A small adaptive controller can exploit some of this variation.

**Secondary hypothesis H2:** Once processing choices are modeled, using the resulting completion forecast for further admissions improves outcomes compared with a tuned fixed reserve. Test this separately so improved processing and improved admission are not conflated.

Working name: `deadline-agent`. Default duration predictor: a lightweight Continuum-inspired empirical distribution with conditional residual estimates. Reuse established metareasoning and scheduling frameworks with attribution; prediction and slack checks are supporting components.

### Intended research contribution

The contribution to investigate is a measured model of the result-consumption tradeoff, an exact reference for a restricted model, and an online implementation that uses that reference to choose processing actions and forecast completion. The subset DP/proof may be a straightforward application of classical scheduling theory; do not market it as a new general scheduling theorem. No novelty or empirical benefit is assumed in advance.

Batching does not inherently enlarge final prefill. Costs depend on raw versus summarized evidence, prompt construction, output lengths, and prefix reuse. Measure them. Summarization can lose information, so timing optimization must be evaluated alongside answer quality.

### Execution contract

- One task and one active model generation at a time; external read-only tools can execute concurrently.
- In the first processing-only experiment, admit a fixed small set and launch it immediately after planning; use K=n, normally n<=4. CPU solver checks may use n<=6. A later admission experiment uses K=2 or K=4.
- Every admitted tool must return successfully, and its output must be presented to a completed evidence-processing or final-answer invocation. Folding raw results into the final answer satisfies this rule when the final invocation completes.
- Track each call's submission, actual execution start, return, and consumption by unique IDs. Exposure is an accounting condition, not proof of semantic understanding.
- The task deadline includes planning, model requests, dispatch, tools, waiting, ingestion, controller computation, retries, and complete final generation. TTFT is insufficient.
- Tool errors, cancellations, missing evidence, and unfinished/truncated answers cannot count as valid success. Cleanup after a deadline miss does not retroactively rescue the run.
- The all-admitted-results rule is a chosen workload contract. State where it is appropriate; abandonment is a separately labeled extension.
- Known duration bounds apply only when a genuine contract supports them. Empirical distributions and measured cost curves are estimates, not hard real-time guarantees.

### Scope control

First isolate processing with a fixed admitted set. Next couple processing to admission from a fixed ranked candidate list. Keep independent reasoning, value-aware query selection, RL, multiple user sessions, KV offload, and token-level preemption outside the first result. Implement a bounded THINK interface for future use without claiming the first solver optimizes it.

## 2. Model and vLLM strategy

### Recommended models

Primary: `Qwen/Qwen3-8B`, using its official weights and chat template, with thinking mode explicitly disabled. The official model card identifies an Apache 2.0 license and vLLM support. Use concise structured plans, evidence records, and answers.

Fallback/debug model: `Qwen/Qwen3-4B`. Use it if the 8B model cannot run reliably on the chosen GPU or setup is consuming the initial budget. Record the switch; do not mix model identities in an aggregate comparison.

Start with FP16 or BF16 as supported by the selected GPU/software combination. Avoid quantization during the first comparison unless memory requires it. If quantization is used, freeze the same checkpoint and quantization configuration for every policy.

An 8B model has approximately 16 GB of two-byte weights before runtime and KV-cache overhead. A single 24 GB-class GPU is a plausible starting point with a bounded context and one active sequence; fit is an engineering hypothesis to verify with a smoke run. Do not promise a particular throughput or memory footprint before measuring it.

Initial limits:

| Setting | Starting value |
| --- | --- |
| Context limit | 8192 tokens |
| Active model generations | 1 |
| Candidate searches | At most 6 |
| Planning output | At most 512 tokens |
| One tool output | At most 768 tokens, including metadata |
| Evidence batch output | Initially 32 + 64 tokens per result; calibrate |
| Optional independent-work chunk | At most 96 output tokens |
| Final answer | At most 192 output tokens |
| Thinking mode | Explicitly disabled |

These are configurable starting values. The batch output limit scales with result count, as defined in Section 5. Check rendered prompt size before every request. Apply the same deterministic truncation/context policy across schedulers, log truncations, and retain call IDs even when content is shortened. Never silently lose an unprocessed result from context.

### Stage A: stock vLLM plus the execution runtime

Use a pinned prebuilt vLLM package or container first. Record the exact package version/container digest, model revision, tokenizer revision, driver, PyTorch, CUDA, GPU, and launch flags. Resolve a compatible current version, smoke-test it, and then freeze it. Do not leave `latest` as the experimental dependency.

Use the OpenAI-compatible HTTP interface with streamed responses and usage accounting where supported. This is the protocol, not a requirement to buy an OpenAI API service. Run the model locally on the GPU instance.

An illustrative server command is below. Check the installed version's `--help`; adapt renamed flags and record the actual command. Do not copy older reasoning flags from unrelated examples.

```bash
vllm serve Qwen/Qwen3-8B \
  --dtype half \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.88 \
  --host 127.0.0.1 \
  --port 8000
```

Pass `chat_template_kwargs={"enable_thinking": false}` with each chat request, using the request shape supported by the pinned server. Prefer schema-constrained JSON outputs when available. Fix and record sampling parameters; start with deterministic decoding for the controlled non-thinking experiment and check for malformed/repetitive output in the pilot. If changing sampling is necessary, change it for all policies and rerun calibration.

For the first result, explicitly disable prefix caching using the supported flag, or reset it between independent task runs if disabling is unavailable. Record the policy. This avoids different policies benefiting from cache state left by earlier runs. Later evaluate prefix caching enabled as a separate configuration. Warm the model and kernels before experiments; do not warm it with evaluation answers. Setup/warmup is outside per-task deadlines but inside the AWS spending budget.

### Stage B: targeted vLLM integration, after an end-to-end trace

The owner is interested in modifying vLLM. Keep that path, but make each change address a measured need:

1. Inspect the pinned vLLM source and existing tracing/metrics facilities before adding code. Do not assume source paths or function names from another release.

2. Determine whether existing facilities expose per-generation queue/admission time, first output, completion, prompt/output token counts, and available cache metrics. Client-side TTFT alone is not an exact prefill measurement: it can include queueing and transport.

3. If required per-request telemetry is missing, implement a small opt-in patch associating a runtime task ID and phase with timing events. Keep the scheduling policy in the runtime because the engine does not know the outstanding external tools by itself.

4. If bounded-request cancellation is needed and the stock interface does not provide confirmed engine cleanup, add a narrow adapter/hook. Verify actual cleanup; closing an HTTP connection is not proof that GPU work stopped.

5. Record the base commit, patch, build method, and an enabled/disabled smoke comparison. Reuse precompiled components when supported instead of rebuilding CUDA kernels unnecessarily.

Do not inject a newly arrived tool result into an already executing prompt/KV state. Stage A integrates observations between bounded generation requests. Token-level interruption and resumable sessions are later research extensions with additional correctness and performance costs.

If existing vLLM interfaces are sufficient, document that finding and keep the runtime as the systems-building contribution. Do not invent a core patch just to increase the amount of modified code. A vLLM patch is not necessary for the first research result.

## 3. AWS budget and deployment

### Available resources and planning ceilings

### Provider decision (supersedes AWS-specific steps below for the midterm)

Use **RunPod** on-demand (not spot) pods: per-second billing, ready vLLM images, SSH. Fallback: **Lambda Labs** (A10 24 GB VMs, per-minute billing; availability varies). Pick one 24 GB-class or larger GPU (RTX A5000 / RTX 4090 / L4 at 24 GB; A40 / L40S at 48 GB for more KV headroom). Use the same GPU type for every compared run and record it. Check the live price before launching.

Owner steps (never paste API keys into chat): create an account, add about $25 credit, and add an SSH public key. Deploy a pod with ~60 GB container disk and a pinned `vllm/vllm-openai:<tag>` image (or PyTorch image + pinned `pip install vllm==<ver>`). Serve on `127.0.0.1:8000` and run the harness on the pod itself. Rsync `results/` back after each session. Stop the pod when idle and terminate it when done (stopped pods still bill disk). The ledger and session-timer rules below still apply. AWS-only items (quotas, EBS, billing alerts) apply only if the AWS backup is used.

### Original AWS budget notes

Only $100 is currently accessible. Target at most $80 of gross attributable AWS cost for the first experiment, leaving at least $20 headroom. Credit coverage, expiration, region, quotas, taxes, and excluded charges must be checked in the actual account; do not assume every charge is covered.

Use a single GPU instance. Suggested starting candidate: Linux `g5.2xlarge`, which AWS lists with one A10G GPU, 24 GB marketed GPU memory, 8 vCPUs, and 32 GiB host RAM. Its greater host RAM makes it a safer loading/building choice than the 16 GiB `g5.xlarge`. Compare a suitably priced single-L4 G6 option only if readily available; do not spend hours optimizing instance choice.

**Planning assumption, not an AWS price quote:** permit up to $1.50/hour compute. Verify the actual region-specific Linux On-Demand rate before launching. If it exceeds that ceiling, choose a cheaper suitable instance/region or report the budget conflict. Do not automatically move to a larger or multi-GPU machine.

| Allocation | Maximum planned spending |
| --- | ---: |
| GPU compute, including setup, downloads, idle time, and warmup | $60 |
| Storage, IP/network charges, and logging | $10 |
| Contingency within working budget | $10 |
| Uncommitted headroom from accessible $100 | $20 |

At the planning ceiling, $60 buys 40 instance-hours. This is a ceiling calculation, not a throughput prediction or a promise that 40 hours is required. Budget all running-instance time, not just timed experiment intervals.

Suggested compute-hour allocations: setup/smoke 4 h, calibration/pilot 8 h, first evaluation 16 h, reruns/analysis 8 h, reserve 4 h. Reduce the evaluation matrix based on measured runtime rather than exceed the cost cap.

### Operational requirements

- Develop CPU-only components before paying for GPU time.

- Use existing user-authorized AWS access. If account access or permission to provision is missing, provide a ready launch procedure and continue local implementation. Do not request or print secret keys.

- Check regional G-instance vCPU quota and available capacity early. Prepare the exact quota request if needed; continue CPU work while waiting.

- Use one instance, a minimal suitable root/data volume, and no paid marketplace software. Avoid NAT gateways, managed inference endpoints, load balancers, and always-on dashboards for this prototype.

- Bind inference to localhost and access it through SSH tunneling. Restrict SSH to the user's permitted source address.

- Maintain a local spending ledger using verified instance rate, launch/stop times, storage, and other known costs. Record sources and estimation uncertainty.

- Set AWS billing alerts if access permits, but do not treat alerts as a hard spending stop; billing data can lag. Use a pre-run runtime/cost check and a verified automatic instance-stop timer as well.

- Configure a bounded session, initially no longer than four hours. Checkpoint after each question. Stop the instance when work is complete or the session timer expires. Renew only within the remaining cost ledger.

- Export code, configs, logs, and completed results before stopping. Instance-store data is ephemeral. Stopping compute can leave EBS and other charges; document project-created resources and cleanup commands.

- Do not delete existing unrelated resources. Do not depend on inaccessible future credits.

- Use On-Demand for the first reproducible experiment. Spot is a later cost option after restart/checkpoint handling works; interruptions must be explicit failures/retries outside the original task measurement.

The estimated full batch plus a reasonable reserve must fit the remaining budget before execution. Stop scheduling new runs if the ledger reaches the working cap.

## 4. Repository, interfaces, and reproducibility

Use Python, asyncio, typed state, a simple HTTP model client, YAML configs, JSONL traces, pytest for scheduling correctness, and matplotlib for scientific plots. Pin versions after a working smoke test. Avoid a heavyweight orchestration framework initially.

```text
deadline-agent/
  pyproject.toml
  README.md
  configs/{smoke,pilot,midterm}.yaml
  src/deadline_agent/
    cli.py
    state.py
    runtime.py
    tools.py
    model_client.py
    cost_model.py
    predictors.py
    exact_dp.py
    exhaustive_reference.py
    schedule_forecast.py
    policies.py
    datasets.py
    metrics.py
    trace.py
  tests/
  docs/{theory,experiment_protocol,decisions,aws_setup,vllm_integration}.md
  scripts/{serve_model,run_pilot,run_midterm,plot_results}.sh
  data/manifests/
  results/
  patches/
```

`exact_dp.py` is pure CPU code over explicit release times and cost tables. `exhaustive_reference.py` is an independent small-instance checker. `cost_model.py` supplies canonical prompt-size calculations and measured latency estimates. `schedule_forecast.py` converts observable runtime state into a restricted model instance and maps the planned action back to execution.

Core types:

- `TaskState`: task/deadline, phase, admitted IDs, unlaunched candidates, running calls, returned raw results, per-ID summaries, consumed IDs, active generation, failure state.
- `ToolCall`: query/type, submission/start/return timestamps, public bounds/features, status, visible result, consumption invocation ID.
- `RestrictedInstance`: outstanding fixed set N, release-time vector, cost lookup c(A,B), finalization lookup f(S), feasible context/action masks, representation/cost-model version.
- `DPResult`: earliest completion by subset, backpointers, chosen summarized subset, folded subset, reconstructed schedule, solver runtime, assumptions/model ID.
- `Action`: LAUNCH, CONSUME(batch IDs), WAIT(until event/planned time), ENTER_DRAIN, FINALIZE(raw IDs), optional THINK.
- `Prediction`: total/residual target, sample count, fallback level, quantile/distribution snapshot, predicted costs, and feasibility reason.
- `RunResult`: final answer/score, validity, misses, latencies, quality/evidence accounting, counts/tokens, decision trace, config/environment hashes.

Keep evaluator-hidden future returns and unreleased payloads outside the scheduler's observation object. The production controller cannot call an oracle adapter. Do not commit credentials, model weights, or large dataset downloads. Export manifests, code, calibrated cost/predictor snapshots, and raw results durably.

## 5. Runtime and processing actions

Use one event loop for state transitions and one semaphore for model generation. Tool workers post immutable completion events; blocking retrieval runs in a thread/process. Tools continue running while the model consumes evidence. New results enter model context only at request boundaries.

Use task handles or ordinary structured model messages rather than an invalid native tool-call transcript. A call can have a runtime handle while other model work happens; do not leave native tool messages unresolved and append incompatible assistant turns. Native function calling is optional.

### Consumption choices

- `CONSUME({i})`: summarize one returned result into a per-ID evidence record.
- `CONSUME(B)`: summarize a nonempty batch of returned results in one invocation, preserving one record and source IDs per result.
- `WAIT`: allow more results to arrive before a planned batch or finalization. Bound waiting by an event/timer and deadline; no busy-polling model calls.
- `FINALIZE(U)`: generate the answer using existing summaries plus all still-raw results U. Every raw result must have returned before this invocation begins.

Defer is a choice to retain raw results and wait or do other valid work; it does not count them as processed. Drain stops future launches but can still choose consumption modes. Finalization is allowed when every admitted call has returned and every result is either already summarized or present raw in the final prompt.

### Prompt and evidence representation

Keep summaries as canonical per-ID records sorted by call ID. When a result is summarized, replace its raw text in the compact working context; retain the original in the trace. Do not accidentally append both a summary and the full historical raw output when modeling compression. Final prompts concatenate the question, ordered summaries, ordered remaining raw results, and a shared answer schema.

Use comparable per-result extraction capacity across single and batched consumption. Start with an output limit such as `32 + 64*len(B)` tokens for a batch, then calibrate. A one-result and four-result batch should not both get a 64-token total budget. Record actual lengths and formatting failures. Use the same per-ID schema across modes and report quality differences.

The exact model assumes canonical representation/cost state. Real generated summaries can differ across batches and histories; this is model mismatch to measure, not something a call-ID field fixes. The runtime DP remains a predictive heuristic when those assumptions do not hold exactly.

### Controller execution

At a model-free decision boundary, observe new completions, form a predicted restricted instance, solve it, and execute only the next causal action. A predicted batch may contain a running call: wait until its results really return, then replan; never fabricate the result at its predicted release time. Recompute after a completion or completed model request. If a timer fires without the predicted result, update its age/residual and replan.

Consume only visible returned IDs. Do not interrupt a generation to splice new observations into its KV state. Charge forecast construction, DP runtime, and request assembly to the real deadline. Cap decision loops and log pathological forecast oscillation.

### Deadlines and cleanup

Use a monotonic clock. Mark a miss at deadline expiry unless the entire valid answer and required evidence consumption completed. Formatting repairs and retries consume the same original budget. After a miss, abort/clean up remaining work and verify model cleanup before the next task. Cleanup time counts toward AWS usage even though it is outside the failed task's allowed deadline.

## 6. Explicit theory, exact reference, and online approximation

### 6.1 Overall objective and theoretical lineage

The application objective is expected `answer_F1 * I(valid completion by D)`. The restricted solver below minimizes modeled completion time for already admitted work; it does not optimize answer F1. Admission/value selection requires additional assumptions about information benefit that we do not know exactly.

Use rational metareasoning for the overall sequential-decision framing, real-time/imprecise computation for mandatory versus optional work, and single-machine release-time/setup scheduling for result consumption. Read the primary references in Section 13. Do not claim that adding a Bellman equation or Monte Carlo sampling is a new theoretical framework.

An eventual finite-horizon state can include remaining time, tool statuses/ages, raw results, evidence representations, candidate cursor, and cache state. Actions include launch/consume/wait/finalize. With a known transition law and terminal quality reward, Bellman recursion characterizes an optimal policy. Unknown semantic benefits, continuous state, and costly state enumeration prevent that characterization from being an immediately usable quality-optimal controller. A finite/discretized solver is optimal only for its specified model.

### 6.2 Restricted deterministic processing problem

At a decision boundary the model is idle. Translate current time to zero. Previously summarized evidence is a fixed background context. Let N be a fixed set of n outstanding admitted results. No new tools may be admitted inside this restricted optimization.

Inputs:

| Symbol | Definition |
| --- | --- |
| r_i >= 0 | Known release time of result i, relative to this boundary; zero if already returned |
| S subset N | Results summarized so far within this subproblem |
| c(A,B) | Known nonnegative duration of consuming nonempty batch B after set A was summarized; A and B disjoint |
| f(S) | Known nonnegative final invocation duration with summaries S and raw results N minus S |
| d | Remaining deadline, used to test the resulting completion time |

Assumptions required for exactness:

1. One nonpreemptive model slot; tools release results independently of model consumption.
2. A batch starts only when every member has returned. No streaming input ingestion during that batch.
3. For every summarized set S, the canonical representation/cost-relevant state is the same regardless of prior batching/order. The c(A,B) and f(S) tables capture all relevant cost/context effects.
4. Costs are independent of absolute start time and the detailed path to A. There is no cache eviction, load shift, changing queue contention, or semantic-history effect omitted from state.
5. Summarizing a result is optional, but consuming every result is mandatory: un-summarized outputs must be read by the final invocation. No result is discarded.
6. Mark context-infeasible or unsupported transitions as infinite cost. Costs/finalization are nonnegative and at least one complete path is feasible when reporting a finite optimum.

These are intentionally restrictive. Real batching can change summary lengths/content and cache behavior. Do not transfer an exactness claim to those cases without expanding the state or proving a valid dominance property.

### 6.3 Subset dynamic program

Define `R(B) = max(r_i for i in B)`, with `R(empty)=0`. Let `F[S]` be the earliest modeled time at which exactly S has been summarized, starting from the fixed background evidence.

```text
F[empty] = 0
F[S] = min over nonempty B subset S:
       max(F[S minus B], R(B)) + c(S minus B, B)
```

The complete answer can fold every remaining raw result into its final invocation:

```text
C_star = min over S subset N:
         max(F[S], R(N minus S)) + f(S)
```

Store backpointers for the minimizing batches and final subset. `S=empty` includes all-raw finalization; `S=N` includes full preprocessing. Intermediate subsets represent mixed summary/raw finalization. Feasibility under this model is `C_star <= d`. If every candidate is infinite, return explicit infeasibility.

Implementation sketch:

```python
F[0] = 0.0
for S in subsets_in_increasing_cardinality(N):
    if S == 0:
        continue
    for B in nonempty_subsets(S):
        A = S ^ B
        candidate = max(F[A], release_max[B]) + consume_cost[A, B]
        relax(F[S], candidate, predecessor=(A, B))
best_S = argmin_S(max(F[S], release_max[N ^ S]) + final_cost[S])
```

`relax` is pseudocode for updating the table/backpointer, not a provided implementation. Precompute release maxima and feasible cost masks. With constant-time table lookups, there are `3^n - 2^n` consume transitions, plus subset preprocessing/final checks: O(3^n + n*2^n) time and O(2^n) DP/backpointer space. Optional stored c(A,B) tables use O(3^n) space. At n<=6, exact enumeration of this restricted problem is small enough to attempt before complex heuristics. Report measured controller overhead.

### 6.4 Proposition and proof obligation

**Proposition under the stated assumptions:** The recurrence returns the minimum modeled final-completion time over all ordered disjoint summary batches followed by one final invocation consuming the remaining raw results.

Proof sketch to write and verify in `docs/theory.md`:

- Every schedule that summarizes S has a last nonempty summary batch B and a preceding summarized set A=S minus B.
- That batch cannot start before either the preceding processing finishes or its last result is released. Its finish is at least `max(F[A], R(B)) + c(A,B)` by induction.
- Conversely, the DP predecessor schedule followed by B at that earliest start attains this value.
- Keeping only the earliest finish per summarized set is valid because costs/state depend only on sets and an earlier finish can always wait to emulate a later one.
- For any final summarized set S, finalization cannot start before F[S] or the last raw result release; starting at their maximum attains the final expression.

This is a standard optimal-substructure argument applied to our restricted model, not a promised novel theorem. Validate both the recurrence and its modeling assumptions. If representation, cache, or quality state depends on the partition/order, add that state to an exhaustive solver or label the compressed DP approximate. Do not silently retain the proposition after relaxing its assumptions.

### 6.5 Verification and role of the oracle

Independently enumerate ordered disjoint batches, permit finalization at every node with all remaining outputs raw, and compare its optimum/backpointers with the DP for random small instances. Include c(A,B) varying with A, context-infeasible masks, simultaneous and staggered releases, empty/one-result cases, ties, mixed fold/summary optima, and impossible finalization. Check reconstructed timing and exactly-once evidence accounting.

CPU synthetic examples may illustrate why different modes win, but label them analytical/model examples rather than measured vLLM results. Do not assert that cheap prefill always makes incremental processing optimal or that a busy model always makes folding optimal.

Use two distinct labels:

- `offline_model_oracle`: exact DP with evaluator-known release times and the declared deterministic cost tables. It is a lower bound on feasible modeled completion times within those assumptions, not on arbitrary measured LLM executions or answer quality.
- `online_dp`: a causal controller that plugs predicted release times/costs into the same DP and replans after actual observations. This is a forecast-based heuristic; the offline proposition does not make it optimal online.

Compare modeled completion gap to the oracle on identical fixed admitted instances and cost tables. Separately measure prediction/model error and actual end-to-end quality. Do not report the modeled gap as an actual optimality certificate. An oracle with perfect duration inputs to a heuristic is also different from an exact scheduling oracle.

### 6.5a Structural results and the exact online policy (implemented, CPU-verified)

Full statements and proofs: `docs/theory.md`. Summary:

- **Token-count cost model.** latency = α + β_prefill·prompt + β_decode·output; raw results L tokens, summaries s < L. Let δ = β_prefill·(L − s).
- **Prop. 1.** Simultaneous returns and a fitting all-raw prompt ⇒ fold is optimal.
- **Prop. 2.** A fitting all-raw prompt ⇒ fold is within (n−1)·δ of optimal, for any release times. Summaries cost decode tokens; folding costs only prefill. **Consequence:** adaptive processing can only beat fold on time when the final-prompt context cap binds, or through quality effects. Experiment A's workload is therefore configured so the cap binds (Sections 7 and 10). This is a theory-driven design choice; state it as such.
- **Prop. 3.** Count-based costs ⇒ some optimal schedule summarizes a release-order prefix in contiguous batches ⇒ an O(n²) exact solver (`affine.prefix_dp`, matches the subset DP).
- **Prop. 4 (online optimum).** In Experiment A all n calls launch together with i.i.d. delays, costs are count-based, and decisions happen only when the model is idle. Under these assumptions, (elapsed time, #returned, #summarized) is a sufficient state: new returns during an action are Binomial with survivor-conditioned probability. Backward induction gives the **exact optimal nonanticipative policy** for any nondecreasing loss (miss probability, tardiness, completion time). It solves in under 0.1 s and is executed as a lookup table, so controller overhead is negligible (`online_mdp.ConsumptionMDP`).
- **RSS heuristic** (`online_mdp.required_summary_slack`), distilled from the optimal table. Summarize exactly the minimum number needed for the final prompt to fit, as soon as results are ready (using idle model time). Wait for one more result only when it is likely to arrive within the time a combined batch saves. Fold the rest. Under the model it is within 4 points of the optimal miss probability when the context binds. When it does not bind, RSS reduces to fold and can be up to 7 points worse. It generalizes to cases the MDP's assumptions do not cover.
- **Model-predicted separation (guessed costs, not measurements).** Context binding, variable delays, D = 20 s: optimal 0.01 miss vs best static (fold) 0.66. With a binding context and predictable delays the gain is confined to a narrow deadline band (e.g. 0.07 vs 0.72 at D = 17 s). Fold ≈ optimal only when delays are predictable and the context is free. Whether the real costs land in the separating regime is the Gate 1 regime check.

The certainty-equivalent `online_dp` (6.5) remains a secondary diagnostic; `online_mdp` and `rss` are the main proposed policies.

### 6.6 Cost model from the pinned vLLM server

Profile total latency by phase, input size, output limit, and cache configuration. Collect prefill/decode components only if instrumentation supports attribution; TTFT alone includes more than prefill. Avoid double-counting client overhead in a server-total measurement.

Create canonical prompt-length estimates for summaries A, raw batch B, and final raw subset N minus S. Predict unreleased output sizes from public caps or calibration data. Use measured cost surfaces rather than a fixed cost per result. For summary batches use a per-ID extraction budget, not one tiny fixed output limit for any batch size. Record actual prompt/output lengths and context failures.

The initial deterministic c/f tables are calibrated surrogates; observed runtimes/representations can differ. Report that error by action/mode. Constant-state theory experiments can use explicitly fixed representations and deterministic tables. Real-model experiments measure how well that abstraction transfers. Latency-only optimization may harm quality; do not assume equal semantic value across modes.

### 6.7 Continuum-inspired duration predictor

Implement `observe`, `predict_total`, `predict_remaining`, and `snapshot` in `predictors.py`. Continuum Section 4.2 motivates empirical per-tool duration distributions with pooled-history fallback. Its selected cache TTL is not a duration bound. Section 5.1's inter-model-request timing cannot identify concurrent tool durations; use direct per-call submission/start/return measurements instead. The inspected preview repository omits estimation from the paper; implement this small component directly and preserve attribution.

For pending age a, estimate remaining duration from survivor samples `T-a` for completed calibration durations satisfying `T>a`. Do not use only `mean(T)-a`. This residual adaptation is our choice rather than an asserted Continuum feature. For the first held-out experiment, freeze both tool and model-cost snapshots. In online extensions, unfinished/canceled calls are censored observations, not zero-time samples.

```yaml
predictor:
  name: continuum_empirical
  quantile: 0.90
  min_samples: 20
  min_survivor_samples: 5
  update_mode: frozen_after_calibration
```

These are initial calibration candidates, not validated reliability settings. Use per-tool then compatible pooled data. If tail data are inadequate, use a genuine public bound when available; never substitute the largest observed sample as a guarantee. A violated bound or unknown tail is explicit. Stop new admissions when needed and drain existing obligations; log fallback frequency. Never expose hidden future releases/payload sizes to the predictor.

Keep a simple mean predictor for CPU diagnostics. Compare predictor quality with the scheduling policy held fixed, and compare schedulers with one identical predictor snapshot. Conditional distribution estimates do not imply a task-level miss guarantee.

### 6.8 Coupling processing to admission

After the fixed-set processing experiment, evaluate admitting the next ranked candidate from a precomputed plan. Respect K tool slots and actual dispatch/proposal costs. Forecast the post-admission outstanding set with the DP and estimated costs; admit only if the chosen completion forecast fits the deadline with a calibrated margin. When rejecting further expansion, enter drain and continue adaptive consumption.

This is a feasibility heuristic with a fixed query order. It does not optimize query value or prove optimal admission. Compare it with a tuned fixed reserve: choose R on calibration data, stop new launches at D-R, then finish outstanding obligations. Tune static processing rules fairly too.

The original `max(tool residuals) + processing + finalization` estimate remains a simple baseline. Label it a conservative scheduling shape using estimated costs, not a rigorous upper bound unless the inputs truly are bounds.

### 6.9 Optional stochastic extension: nonanticipative risk rollouts

After the deterministic controller and main result work, sample conditional tool residuals and model-latency distributions to estimate completion risk. Evaluate each candidate first action followed by a specified causal continuation policy, with sampled future outcomes hidden from that policy until their simulated observation events. Use common random scenarios across candidate actions where appropriate.

Do not optimize each sampled future with a clairvoyant offline DP and call the resulting fraction an implementable policy's miss probability. That permits different future-informed decisions in each scenario and can be optimistically biased. The offline DP remains a reference; risk evaluation must follow a nonanticipative policy.

Model or test correlated latency/load where relevant; independently sampling marginals is an explicit approximation. Monte Carlo reduces sampling noise, not model error. Per-decision epsilon thresholds do not automatically imply an episode-level epsilon guarantee after replanning. Measure target versus realized task miss rates with uncertainty; 30 questions cannot substantiate tiny tail-risk claims. Cite MCPP for portfolio rollouts/replanning and distinguish our processing action/cost model rather than claiming Monte Carlo is new.

Risk control is a follow-up, not a prerequisite for the first midterm figure or an additional full GPU factorial matrix.

## 7. Workload

### Controlled initial task family

Use HotpotQA comparison questions, where independent lookups are plausible. Start from a deterministic manifest of question IDs. Partition calibration and evaluation before policy tuning. Include a few hand-constructed fixtures for correctness, but do not pass fixtures off as benchmark performance.

For the initial environment, expose each question's supplied distractor paragraphs through a local lexical/BM25 retrieval tool. Do not supply the whole document collection directly to the answer model. Give the planner only the question and public tool description. Preserve source IDs. Implement `search(query, top_k=2)` with a shared tokenizer-based result cap. Avoid gold-aware retrieval, gold-aware candidate selection, or using supporting-fact labels in policy features.

Describe this as an adapted HotpotQA distractor retrieval environment, not the official fullwiki task or open-web browsing. Pooling documents across questions or using a larger corpus is a later realism extension.

The planner produces at most six ordered candidate searches and optionally a concise question decomposition. Generate the plan under the measured deadline in every primary run. Freeze deterministic planning settings; detect and report plan variation. Fixed-plan timing replay may be used for debugging only, or must be separately labeled and charge plan cost consistently.

At each task, measure with real vLLM invocations and real retrieved content. Tool release timing may be controlled. Cap visible results consistently across policies. Do not cache final answers or evidence updates across policies in the primary timing experiment.

### Delay regimes

For Experiment A (per Prop. 2), use the **variable regime as primary** and make the final-prompt context cap bind. For example, use top-k passages with about 1500 tokens per result and a final-prompt cap of about 4096 tokens, so at least one summary is required. Calibrate both on calibration questions only. The predictable regime is a secondary condition: with a free context theory predicts no adaptive benefit, but with a binding context the adaptive policies still win in a narrow deadline band (docs/theory.md Section 6).

Support two initial regimes:

1. Predictable bounded delay: a fixed value or narrow bounded distribution.

2. Variable bounded delay: a mixture with occasional slower calls.

Choose scale from a pilot so tool waits and generation costs both matter. Freeze numeric distributions before held-out evaluation. Avoid selecting a regime after seeing test scores.

Derive delay samples reproducibly from `(task_id, normalized_query, occurrence_index, latency_seed)`, not policy name or launch order alone. Do not use Python's process-randomized built-in hash for persistent seeds. Keep release time and content hidden until the completion event. Record real retrieval time, injected additional delay, and observed total tool latency separately.

Also measure natural local retrieval latency with no injected delay. Expect it may be too short to expose this scheduling tradeoff; a null result is informative. Do not describe artificial delay as measured remote-service latency. A later real tool could be a bounded code/test execution job whose completion must be inspected.

### Model capability checks

On calibration questions, measure question-only answering and an all-evidence, generous-budget condition. The former detects pretraining shortcuts; the latter tests whether the chosen model can solve the workload at all. Do not select held-out questions based on their observed answers or drop hard questions to manufacture gain.

If both conditions score nearly the same, adjust the workload using calibration data. If all-evidence quality is near zero, fix retrieval/prompting or choose a suitable model before paying for a large timing matrix.

## 8. Baselines and fair comparisons

### Experiment A: processing only, fixed admitted set (primary)

After measured initial planning, admit the first up to four valid unique candidate queries and launch them without model-dependent staggering, K=n. Use the same generation settings and stable delay seeds across policies. This deliberately removes further admission decisions. The all-call obligation is shared, even under tight deadlines. Record any plan variation; do not use gold answers to choose queries.

| Policy | Processing behavior |
| --- | --- |
| `static_incremental` | Consume each returned result at the next model boundary |
| `static_batch` | A calibrated fixed batch-count and/or maximum-wait rule, with a calibrated finalization fallback |
| `static_fold` | Wait for all returns, then answer from raw results in one invocation |
| `online_mdp` | Exact optimal online policy table under the calibrated cost/delay model (Prop. 4) |
| `rss` | Required-summary-slack heuristic distilled from the optimal table |
| `online_dp` (optional) | Certainty-equivalent replanning with the restricted DP; secondary diagnostic |

When the context cap binds, `static_fold` summarizes only the minimum number of results needed to fit, in one batch after all returns.

All modes use the same evidence schema and comparable per-ID extraction budgets. Their processing rules intentionally differ; shared infrastructure, caps, tools, model, and timing accounting do not. Tune batch thresholds/timers on calibration data. Do not make batching a weak un-tuned straw baseline. `static_fold` cannot finalize until its admitted calls return.

The evaluator-only `offline_model_oracle` runs on fixed-set traces under the same deterministic model. It is not included as an implementable quality baseline and does not get silently substituted for online behavior.

### Experiment B: admission plus processing (secondary)

Keep the same ranked candidate list, concurrency cap, tool distributions, model, and deadlines. Select `best_static` on calibration data, then compare:

| Policy | Purpose |
| --- | --- |
| `best_static_fixed` | Best calibrated static processing mode plus tuned fixed reserve R |
| `best_static_pending` | Same static mode plus pending-work admission forecast; isolates admission |
| `adaptive_pending` | Adaptive DP-based processing plus pending-work admission |

Optionally add fixed-reserve plus adaptive processing to complete a small factorial study if cost permits. Serial execution, tool-only forecasts, pending-count reserves, and mean predictors are useful secondary ablations, not reasons to omit strong batching/fold comparators.

Freeze calibration snapshots and hyperparameters before evaluation; permit declared workload/deadline-specific static tuning. Randomize run order, isolate tasks from unrelated inference load, disable/reset cross-task prefix reuse consistently, and include controller/retry overhead. Use the same public information in each applicable comparison. Label descriptive baselines rather than implying a faithful reproduction of an unimplemented paper.

## 9. Metrics, theory diagnostics, and traces

Primary application outcome:

```text
deadline_quality = answer_F1 * I(valid_complete_run_by_deadline)
```

A valid run has a complete parseable final answer, no unresolved admitted calls, all required outputs consumed by completed invocation(s), and final completion by D. Wrong but valid answers score poorly; correct late answers score zero on this metric. Include all attempted runs/failures in denominators.

Use HotpotQA's official answer evaluation/normalization with adaptations documented. Also report raw EM/F1, miss rate, supporting evidence/source coverage where applicable, result count, prompt/output tokens, consumption/finalization latency, controller overhead, and quality by mode. No separate paid LLM judge is needed.

Theory/model diagnostics:

- DP versus independent exhaustive optimum under identical declared deterministic inputs.
- Reconstructed schedule validity and exactly-once accounting.
- Modeled finish-time gap between causal policies and `offline_model_oracle` on the same fixed set; label this as modeled, clairvoyant-reference analysis.
- Predicted versus observed action/drain/final costs, plus errors introduced by representation/caching assumptions.
- How often the adaptive controller chooses each mode or waits, and which observable conditions distinguish those decisions.
- Solver and total controller runtime, including table construction; no free planning overhead in the real deadline.

Log JSONL events with run/task ID, monotonic relative time, event type, policy, call/model ID, and config hashes. Required events include task/planning start/end, launch/return/error, decision/forecast/DP start/end, consume start/end, wait start/end, drain start, final start/first-token/end, deadline miss, and cleanup start/end.

Decisions record visible state, chosen batch/raw IDs, predicted releases/c/f, model assumptions/version, reconstructed next action, margin, solver time, and rejection reason. Predictions record age, sample/survivor counts, fallback level, quantile/distribution snapshot, and realized error once observable. Keep evaluator-only future/oracle fields physically outside scheduler inputs.

Archive actual visible raw results, concise evidence records, final outputs, usage/finish reasons, token/context sizes, truncation, repairs, and failures. Do not log secrets. Preserve enough information to distinguish scheduling error, timing-model error, and evidence-quality loss.

## 10. Experiment ladder and initial figure

### Gate 0: theory and executable CPU reference

Write `docs/theory.md` with the restricted assumptions, recurrence, proof sketch, and exactness limits. Implement DP and independent exhaustive enumeration. Test random n<=5 instances and targeted cases through n<=6, including state-dependent table costs c(A,B), infeasible masks, mixed folding, and schedule reconstruction. Add event-loop checks for hidden-future isolation, consumption accounting, elapsed-age prediction, waiting/deadline behavior, and no context mutation mid-generation.

Produce synthetic/model schedules showing distinct modes can win under different release/cost conditions. These are correctness/analytical examples, not measured performance. Do not launch a GPU to run subset DP checks.

### Gate 1: real model, workload capability, and cost calibration

Serve the pinned model on one GPU; run 3-5 questions and obtain one real auditable timeline. Confirm thinking is disabled, inputs fit, batches preserve per-ID records, cleanup works, and question-only/all-evidence controls show a meaningful task. Profile a modest grid of prompt lengths, batch counts, and output budgets; use interpolation rather than a GPU measurement for every subset.

**Regime check (required before the pilot):** fit the affine cost model to the profile. Solve the MDP with the fitted costs and the calibration delay distribution, and run `scripts/explore_regimes.py`-style predictions. Confirm that `online_mdp`/`rss` separate from the best static policy in some deadline band. If they do not, adjust the result size, context cap, or delay scale on calibration data only, and record the change. Pick the three deadlines inside that band.

Use disjoint calibration questions (roughly 10 initially) for tool duration samples, cost surfaces, static batch rules, and three deadlines. Freeze snapshots. Measure seconds per task and projected cost. Disable thinking or parsing failures before large runs. Do not select evaluation questions based on model correctness.

### Gate 2: smallest real quantitative pilot

Experiment A: 10 held-out questions x 3 deadlines x 5 processing policies (`static_incremental`, `static_batch`, `static_fold`, `online_mdp`, `rss`) x 1 delay seed = 150 real task runs in the variable regime. This is about an hour of GPU time and **is the midterm figure**. Overlay the MDP's model-predicted miss curves as dashed lines labeled "model prediction". Choose deadlines using calibration reference runtimes, for example around 0.6x/1.0x/1.4x a reference median; adjust only on calibration if all tasks trivially pass or fail, then freeze actual seconds.

On fixed-set traces, run the cheap modeled offline oracle analysis on CPU. If traces lack counterfactual model costs, use the declared calibrated table and label the analysis model-based; replay cannot invent unseen summaries, model answers, or timings. Any quality comparison requires actual model executions.

### Gate 3: deferred to the final report (not part of the midterm slice)

Also deferred: Experiment B, any vLLM patch, prefix-caching configuration, predictor comparison, and the stochastic extension.

If the projection fits: 20 calibration and 30 disjoint held-out questions x 3 deadlines x 4 processing policies x 2 seeds = 720 held-out task runs for Experiment A in one regime. A second complete regime doubles that count. Reduce sample size/repeats if necessary; preserve the four processing comparators.

Experiment B is a small follow-up, initially 10 held-out questions x 1 middle deadline x 3 policies x 1 seed = 30 real runs. Expand only after checking remaining budget and pilot findings. It is acceptable for the midterm's main result to establish the fixed-set processing tradeoff while admission remains preliminary.

Required midterm figure: one two-panel measured graph, deadline on x-axis, mean deadline-qualified F1 on the left and miss rate on the right, with the four Experiment A policies. Add a small table or inset for modeled oracle gap and controller overhead if space allows. Pure synthetic oracle gaps do not replace real system measurements.

Use white backgrounds, readable labels, consistent colors, question-level paired bootstrap intervals, and source CSV/plot scripts. Keep all repetitions from a question together when resampling. Small samples imply wide uncertainty, especially for rare misses. Label controlled delays, measured durations, model predictions, and oracle references distinctly.

Interpret null/negative results. If one calibrated static mode dominates, investigate whether the workload lacks variation or the adaptive forecast is inaccurate. Do not hide the winning static baseline or force inefficient per-result processing to manufacture a benefit.

## 11. Deliverables and acceptance criteria

Implement documented target interfaces equivalent to:

```bash
python -m deadline_agent.cli verify-theory --max-n 5
python -m deadline_agent.cli smoke --backend fake
python -m deadline_agent.cli prepare-data --config configs/pilot.yaml
python -m deadline_agent.cli smoke --backend vllm --config configs/smoke.yaml
python -m deadline_agent.cli calibrate --config configs/pilot.yaml
python -m deadline_agent.cli run --experiment processing --config configs/pilot.yaml
python -m deadline_agent.cli analyze-oracle --run-dir results/REAL_RUN_ID
python -m deadline_agent.cli plot --run-dir results/REAL_RUN_ID
```

These are desired interfaces to build, not existing commands. Support resumable completed-task checkpoints and separate output namespaces for fake/model/replay/real runs.

Acceptance criteria:

- CPU-only exact solver and independent exhaustive checker agree on meaningful small instances, with a documented proof and explicit assumptions.
- The solver reconstructs valid batch/wait/final schedules and labels infeasibility correctly.
- An online controller never receives evaluator-only futures; a favorable offline-oracle result does not prove online optimality.
- A real vLLM runtime runs concurrent retrieval and consumes evidence through each required mode with correct deadlines and accounting.
- Calibrated costs/predictors, static baselines, environment, and data splits are frozen and reproducible.
- A measured pilot includes all four processing comparators, valid per-task scores, raw traces, a figure, and an honest interpretation.
- Theory gap, prediction error, semantic quality, and actual latency are reported as different quantities.
- Every retry/controller invocation and paid session time is accounted for. The accessible $100/$80 working cap is respected.
- Any vLLM patch addresses a measured interface/telemetry need; stock-server integration is sufficient if no patch is needed.
- No measurements, theorem scope, or benchmark runs are invented.

Maintain `STATUS.md`: implemented components, commands run, verification outcomes, actual results, blockers, model assumptions currently satisfied, cost used/projected, and the immediate next command. Continue implementing through authorized milestones rather than ending at scaffolding.

## 12. Midterm report and remaining work

The course requires a systems-building component, an initial quantitative evaluation, and a 2-4 page report using the supplied ICML template. Produce a roughly three-page starting report after actual results exist.

| Space | Content |
| --- | --- |
| 0.5 page | Problem, processing tradeoff, execution contract, and related theory |
| 0.75 page | Restricted model, compact recurrence/proposition, and exactness limits |
| 0.75 page | Causal runtime, predictor/cost measurements, and implementation |
| 0.75 page | Measured figure, static comparisons, oracle/model diagnostics, and interpretation |
| 0.25 page | Limitations and next experiments |

Put the full proof, exhaustive checks, and implementation details in project documentation; include an appendix only if course rules permit. A standard DP proposition supports the systems story but is not automatically the main novel theorem. Do not claim quality-optimal control from a latency-only oracle.

After the midterm, prioritize coupling admission to processing, measured stochastic robustness/nonanticipative rollouts, dependent calls, and prefix-cache sensitivity. Value-aware search stopping and independent reasoning need additional quality modeling. Multi-session serving and token-level interruption are later scope expansions.

For three people, split restricted model/solver, runtime/vLLM integration, and workload/evaluation. For two, combine theory/solver with policy implementation and keep evaluation integrated. Group size does not change course scope expectations; integrate an executable early.

## 13. Related theory, attribution, and sources

The general ideas of metareasoning, optimal scheduling under explicit assumptions, mandatory/optional computation, batching, and Monte Carlo planning are established. State precisely what each source solves and which assumptions differ. Applying a known technique well can support a systems contribution; do not assert absence of all prior solutions.

| Primary theory | Relevant result/framework | Mapping limit |
| --- | --- | --- |
| Shih, Liu, Chung, SICOMP 1991, Algorithms for Scheduling Imprecise Computations with Timing Constraints | Mandatory work plus optional refinement; optimal algorithms for stated preemptive cases, harder all-or-nothing variant | Our admitted calls create obligations, costs/quality depend on processing mode, and invocations are nonpreemptive |
| Hay et al., UAI 2012, Selecting Computations: Theory and Applications | Bayesian metalevel computation selection and optimal-policy results in restricted cases | It does not by itself specify our tool, batching, context, and inference-cost model |
| Shperberg et al., ICAPS 2021, Situated Temporal Planning Using Deadline-aware Metareasoning | Deadline-aware computation allocation; a pseudo-polynomial optimal known-deadline special case | Different planning-process/action-expiration abstraction |
| Elboher et al., AAAI 2023, A Formal Metareasoning Model of Concurrent Planning and Execution | Concurrent planning/execution; general intractability and tractable special cases | Exact transfer requires matching its state/actions/objective assumptions |
| Schutten et al., Management Science 1996, Single-Machine Scheduling with Release Dates, Due Dates and Family Setup Times | Batching/setup versus timeliness; exact branch-and-bound for its NP-hard problem | Family setup scheduling differs from raw/summary finalization and subset-dependent LLM costs |

Read these sources before claiming a new theorem. Our restricted recurrence is presented as an explicit application to the declared processing model, with an independent verification requirement. Do not infer NP-hardness of our exact restricted problem merely because neighboring scheduling problems are NP-hard; a reduction would be required.

Primary theory links:

- https://epubs.siam.org/doi/10.1137/0220035
- https://proceedings.mlr.press/r10/hay12a.html
- https://www.cs.unh.edu/~ruml/papers/dda-icaps21.pdf
- https://arxiv.org/abs/2303.02664
- https://www.utwente.nl/en/bms/iebis/staff/schutten/setup.pdf

Agent/serving sources:

- Continuum, empirical duration distributions and TTL decisions, initially inspect v7 Sections 4.2/5.1: https://arxiv.org/html/2511.02230v7
- Continuum preview code; check its estimation-component limitation and actual reused code/license: https://github.com/Hanchenli/vllm-continuum
- Asynchronous Tool Usage for Real-Time Agents: https://arxiv.org/abs/2410.21620
- Speculative Interaction Agents: https://arxiv.org/abs/2605.13360
- Timely Machine: https://arxiv.org/abs/2601.16486
- Inference-Time Budget Control for LLM Search Agents: https://arxiv.org/abs/2605.05701
- On the Clock: https://arxiv.org/abs/2610.10833
- MCPP / On Time, Within Budget: https://arxiv.org/html/2605.06110v2 . Its supplied-workflow model allocates models/parallel samples using completion-probability rollouts. Use it as related planning methodology, not a straw baseline or evidence that Monte Carlo risk control is new.
- LADS, tool-result delivery scheduling coupled with prefix leases: https://arxiv.org/abs/2611.00881 . Its stated decision differs from our processing modes; avoid claiming first tool-result scheduling.
- Tool-Augmented LLM Serving Under Firm Deadlines: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6438169

Read relevant full papers before making strong exclusion claims. Time-aware agent work includes runtime/harness interventions as well as model-side changes; do not distinguish our project solely by saying it uses a runtime and no training. Our proposed distinction is the processing-mode model, measured costs, and demonstrated behavior under completion deadlines.

Official implementation/workload sources (verify the installed software and regional prices; links initially checked 2026-11-06):

- https://huggingface.co/Qwen/Qwen3-8B
- https://huggingface.co/Qwen/Qwen3-4B
- https://docs.vllm.ai/en/stable/features/reasoning_outputs/
- https://docs.vllm.ai/en/stable/features/structured_outputs/
- https://docs.vllm.ai/en/latest/features/tool_calling/
- https://docs.vllm.ai/en/stable/usage/metrics/
- https://docs.vllm.ai/en/latest/cli/serve/
- https://aws.amazon.com/ec2/instance-types/g5/
- https://aws.amazon.com/ec2/pricing/on-demand/
- https://hotpotqa.github.io/
- Course template: https://media.icml.cc/Conferences/ICML2025/Styles/icml2025.zip

## 14. Immediate execution checklist

1. ~~Inspect the repository and available resources.~~ Done.
2. ~~Implement the restricted cost/state types, exact DP, independent exhaustive solver, reconstruction, and proof document.~~ Done, plus the O(n²) solver, fold bound, online MDP, and RSS.
3. ~~Run CPU correctness checks and create clearly labeled model examples.~~ Done: `uv run pytest -q`, `uv run python scripts/explore_regimes.py`.
3a. Owner: set up a RunPod account, credit, and SSH key (in parallel).
4. Implement the event-driven runtime, actual retrieval, data split, canonical per-ID evidence format, and local vLLM client.
5. Implement incremental, tuned batch, fold, and online-DP processing modes behind one execution interface.
6. With authorized GPU access, serve the pinned model and record a real smoke trace plus capability controls.
7. Calibrate cost surfaces, Continuum-inspired total/residual distributions, static thresholds, and deadlines; freeze the manifest/snapshots.
8. Check projected cost, run the 120-run processing pilot, and perform CPU modeled-oracle diagnostics.
9. Produce the measured figure, quantify model/quality limitations, then decide whether to expand to 720 runs or prioritize a small admission follow-up.

**Start at step 1. Deliver theory that is executable and verified, then test its useful scope on the real system.**

