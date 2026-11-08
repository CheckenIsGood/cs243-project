# vLLM co-design spec: provable scheduling for agent tasks with tool waits and bounded KV

Target: vLLM **v0.32.0** (`vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/block_pool.py`,
`vllm/v1/core/kv_cache_manager.py`). Line numbers below are from the v0.32.0 tag.

The theory is in `docs/stochastic_theory.md` (commit e17bd7f). The independent verification is in
`docs/stochastic_theory_verification.md` (commit d334bac). Every rule below is tagged:
- **[proved]**: needed by a verified theorem;
- **[benefit]**: improves latency but no guarantee depends on it;
- **[unproven]**: a heuristic.

## 0. What the rules buy

**Theorem D (engine), verified.** Assumptions:
- agent tasks are DAGs of LLM calls;
- tool waits are random, with any law;
- tasks arrive online;
- token counts are known at release (or upper-bounded and reserved).

Under these, the policy below is within a constant factor of the best non-anticipatory policy for weighted mean
task completion time. It needs 3× engine speed and memory M′ = M + L_max. The constants:
- **3** in the ideal case (no parent-context recompute, ℓ⁺ ≈ ℓ⁻);
- **2.78** with unbounded memory, at our calibrated constants;
- **≈ 10** at a 64k KV budget. A sharpened, still-unverified version (Prop K) gives 5.80.

The verifier found that the benchmark may itself prefill already-returned tool results early. If it does, the
constant becomes (2+β) + 2·max(1+β, θ). **The rules are unchanged** either way.

Supporting results:
- **R1:** evict-always plus recompute is the safe baseline.
- **R5c:** any keep policy preserves the guarantee if held KV is in the lowest eviction class.
- **R6:** preloads are free for the guarantee if they run only in idle prefill capacity.

Speed 3 and M′ are proof devices. They are not knobs to implement.

## 1. Client side: the task index [proved]

Every request of task J carries the same fixed priority, computed once at release:

```
Z'_y = α(q_y + ρ_y) + κ_y + c·X_y + a·X_y·ℓ⁺_y / M        (per LLM node y of J)
κ_y  = α(ℓ⁺_y + L_max)                                     (kick charge; 0 with unbounded memory)
ℓ⁺_y = ρ_y + q_y + X_y       (reservation: parent context + new prompt + max output tokens)
Z'_J = Σ_y Z'_y
vLLM priority = int(1000 · Z'_J / w_J)                       (smaller value = served first; ties by arrival)
```

- Calibrated A40 / Qwen3-8B constants: a = 0.027767 s, c = 0.000543 s/seq, α = 1.14e-4 s/token, so m = a/c ≈ 51.
- M = M′ − L_max, where M′ is the KV capacity in tokens (from `/metrics` `cache_config_info`, as
  `preload_controller.py` already reads it).
- q, ρ and X for **future** nodes (the final call) must be estimated at release; use the workload profile. X is
  `max_tokens`, so it is a true upper bound.
- Sending an estimate is a deviation from the theorem. Theorems H/DR (random lengths, under verification) are
  the route to covering it.
- The priority is **fixed for the task's lifetime**. Do not use the Gittins index (Thm G) in the engine: Gittins
  with bounded memory is open.

## 2. Engine changes

### E1. Admission = priority prefix with head-of-line stop and kicks [proved: P3, Lemmas D0/D1]

**Stock behaviour.** `Scheduler.schedule()` (l. 591) schedules RUNNING requests first (l. 658). It admits WAITING
requests only if nothing was preempted this step (`if not preempted_reqs`, l. 900). It preempts by priority only
when a *running* request's `allocate_slots` fails (l. 796). A waiting high-priority request therefore never
displaces a running low-priority one. This "running-first" rule violates P3; the verifier's exact instance gives
41.2 against a bound of 8.25.

**Change.** At the start of `schedule()`:
1. Let `cand` = running ∪ waiting (excluding preload requests, §E4), sorted by `(priority, arrival_time)`.
2. Walk `cand`, summing reservations `res(r) = ceil((num_prompt_tokens + max_tokens) / block_size)` blocks.
   Count prefix-cache hits as reserved too; this is conservative and keeps the proof's accounting.
3. **Stop at the first request that does not fit** in `M′_blocks = total_gpu_blocks`. That request and every
   later one is outside the admitted prefix. Do not skip ahead to smaller requests: the stop is what Lemma D0
   uses.
4. Every RUNNING request outside the prefix is **kicked**: `self._preempt_request(req, ts)` (l. 1571), which
   frees its blocks and later recomputes its prompt plus generated tokens as prefill. Every WAITING request inside
   the prefix is scheduled this step. Drop the `not preempted_reqs` gate for them.
5. Since admitted reservations sum to ≤ M′, `allocate_slots` (`kv_cache_manager.py` l. 371) cannot fail for an
   admitted request. The old failure-preemption path (l. 796) should become dead code; assert it.

**Invariants to assert:**
- Σ res(admitted) ≤ M′.
- Kicks happen only in a step where a new request arrived or a request's tool result arrived (an availability
  event). Completions only extend the prefix.

### E2. Prefill choice and chunk size [proved: P4, Lemma D1 case (C), R0b]

**Stock behaviour.** Prefill tokens go to running requests in list order, up to `max_num_scheduled_tokens` and
`long_prefill_token_threshold` (l. 640–711). Several prefills can share a step.

**Change.**
- Iterate the running loop in `(priority, arrival_time)` order.
- Give prefill tokens only to the **highest-priority admitted request in prefill phase**, with a chunk of
  P̄ = a/α ≈ 244 tokens per step. Set `long_prefill_token_threshold = 244` and stop giving prefill tokens after
  the first prefill request.
- Decodes: the theory decodes the top m = ⌊a/c⌋ ≈ 51 admitted decoders. Expose a flag `--decode-top-m`.
  - Decoding all admitted requests (stock) is an **[unproven]** deviation. It likely helps throughput.
- Proved variant: at most one prefill per step. Letting leftover budget flow to the next prefill in priority order
  is **[unproven]** but work-conserving; keep it behind a flag.

### E3. Eviction classes: held KV is always evicted first [proved: P2, Lemma D1, R5c]

**Stock behaviour.** `BlockPool.free_blocks` (l. 774) puts a finished or preempted request's blocks with
ref_cnt 0 into `free_block_queue`, where they stay cached until `get_new_blocks` (l. 666) evicts them. Blocks
with ref_cnt > 0 are never evicted. So **P2 already holds in stock vLLM**, with one exception: `unpin_blocks`
(l. 708) and any pinning. Requirements:
- **No pins.** That covers block pinning, and also the client controller's "open low-priority ignore_eos
  request" pins: pinned KV is not lowest class and can block admission. The co-design version of the controller
  must disable pins.
- **Held KV is free for admission.** E1 measures fit against `total_gpu_blocks` minus admitted reservations. It
  never subtracts cached-free (held) blocks.

### E4. Preload requests are a background class [proved: R6 safety; benefit: R6 saving]

- Mark preload requests explicitly: the controller's `max_tokens=1` warm-up, plus a flag
  (`extra_args={"preload": true}`).
- Keep them out of `cand` in E1.
- Schedule a preload chunk **only in a step where no admitted request is prefilling**.
- Give it only blocks not reserved by the admitted prefix.
- Kick it first whenever admission needs the memory.
- When it finishes, its blocks become cached-free (held class), so the later real request hits them.

The guarantee is unchanged (Lemma D1 never sees preloads). For chain tasks the saving is exactly
α·(held prefix tokens used). For DAG tasks it is **at most** that (verifier counterexample).

### E5. Keep/evict order inside the held class [benefit: R5a/R5b; any order is safe by R5c]

Thm D allows any eviction order among held blocks, so this is purely about the benefit.

**R5a (one prefix):** the best keep rule is a TTL. With exponential tool waits it is keep iff
h·E[D] < R, where:
- h = holding cost per second;
- R = recompute cost.

The best TTL is within e/(e−1) of a clairvoyant keeper.

**R5b (many tenants):** class i's subproblem is a ski rental with:
- holding rate h_i = η·L_i (L_i = prefix size in tokens, η = memory price);
- recompute value R_i = (ν_cp + ν_ld)·P_i, with ν_cp = 1/(1−u) and ν_ld = ρ_s/(1−u)², as in
  `docs/controller_theory.md`.

Keep or preload iff max_τ [σ_i(τ) − η L_i E min(D_i, τ) / ((ν_cp+ν_ld) P_i)] > σ\* = ρ_s/(1−u+ρ_s). At η = 0
this is the current controller's rule. This is proved only for linearized latency prices.

**Change.**
- `free_blocks` takes a per-request TTL τ_i, supplied by the client in `extra_args`.
- A held block past its TTL moves to the **head** of `free_block_queue` (evicted first).
- Unexpired blocks stay in LRU order behind it.

**[unproven]** How to set η online. A first heuristic: η = 0 while the free fraction is > 10%, else
η = (Lagrangian price from the R5b LP at the current class mix).

### E6. Queued requests protect their cached prefix [benefit: P1, Lemma Q]

**Stock behaviour.** A WAITING request takes no block references until it is scheduled, so its cached prefix
can be evicted while it queues. This is Lemma Q, the dominant cause of preload harm we measured.

**Change.** In `add_request` (l. 2616), look up the prefix hit (`kv_cache_manager.get_computed_blocks`) and
**move those free blocks to the tail of `free_block_queue`** (most recently used).
- Do **not** take a reference (`touch`, l. 752). A reference would lift them out of the lowest class and break P2.
- No guarantee needs E6. It only makes keeping and preloading pay off.

## 3. Do not

- Keep the running-first admission (violates P3).
- Pin KV, client- or engine-side (violates P2).
- Change a task's priority after release (the proofs need a fixed index).
- Let preloads take prefill capacity from an admitted request (breaks R6).

## 4. Acceptance tests

1. **Unit invariants**, each run every step in a debug build:
   - Σ res(admitted) ≤ M′;
   - `allocate_slots` never fails for an admitted request;
   - kicks happen only on availability events;
   - no preload chunk runs in a step with an admitted prefill.
2. **Replay the exact counterexample instances** from `tests/test_stochastic_theory.py`
   (`thmD_hypotheses_are_necessary`) through the patched scheduler logic. Stock order should reproduce the
   violation; patched order should satisfy the bound.
3. **GPU A/B on the existing harness** (`scripts/real_multiagent.py`, paired seeds 2–4, KV caps 40k / 50k / 80k /
   full). Compare:
   - stock vLLM;
   - E1+E2+E3;
   - plus E4 and E6;
   - plus E5.

   Metric: weighted mean task completion time, with bootstrap CIs, as in `docs/progress_since_pivot.md` §6.10.

## 5. Not covered by a verified theorem yet

- Random output lengths with bounded memory: Thm DR, under verification. It needs X̄ = `max_tokens` reserved,
  which E1 already does.
- Gittins priorities with bounded memory: open.
- Decoding more than m sequences, multi-prefill steps, and the online η: these are the [unproven] items above.
- Discrete-step kick counts: R0b covers the additive form only.
