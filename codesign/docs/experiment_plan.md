# Experiment and implementation plan: vLLM co-design

This plan goes with `vllm_codesign_spec.md` (the engine rules) and `stochastic_theory.md` /
`stochastic_theory_ext.md` (the proofs). The verification reports are `stochastic_theory_verification.md` and
`stochastic_theory_ext_verification.md`.

## 1. Claims the experiments test

| # | Prediction | Source |
|---|---|---|
| H1 | Under KV pressure, our scheduler (priority prefix + kicks + fixed memory-aware index) beats stock vLLM and a Continuum-like scheduler on weighted mean task completion time | Thm D; the running-first counterexample |
| H2 | Soft hold (held KV always evicted first, TTL only orders eviction) beats hard TTL pins under pressure and ties them with ample memory | R5c; P2; our earlier pin measurements |
| H3 | Protecting a queued request's cached prefix (E6) removes the queued-eviction harm (Lemma Q) | progress_since_pivot §6 |
| H4 | The full index Z′ (GPU + area + kick charge) beats an area-only (LAMPS-like) ranking under pressure | Thm D, Prop K |
| H5 | Measured ALG / (realized lower bound) stays far below the proven constant | Thm D §4.6 |

## 2. Code

Everything plugs in through vLLM 0.32's `--scheduler-cls`, so there is no fork. One scheduler class with
switches produces every configuration.

| Step | File | Content | Size |
|---|---|---|---|
| 1 | `src/codesign/vllm_sched.py` | `CodesignScheduler(Scheduler)`: a pre-pass before `super().schedule()` (spec E1–E3). It sorts running ∪ waiting by priority and computes the admitted prefix with ℓ⁺ = prompt + max_tokens and a head-of-line stop. It kicks running requests outside the prefix with `_preempt_request`, hides waiting requests outside it, sorts `running` by priority, and holds back all but the top prefill (chunk a/α). It asserts the invariants each step. | ~300 lines |
| 2 | same + a `BlockPool` patch | E4: preload class. E6: `add_request` moves the queued prefix to the free-queue tail. E5: per-request TTL in `extra_args`, with expired blocks moved to the free-queue head. Hard-pin mode for the baseline. | ~150 lines |
| 3 | `src/deadline_agent/multiagent.py` (this folder's copy) | Client side. It computes Z′ per task at release and sends `priority`, `extra_args = {task_id, ttl, preload}`. It also has Continuum-like and Autellix-like client modes. | ~150 lines |
| 4 | `tests/test_vllm_sched_logic.py` | CPU tests. The admission logic is checked against `stochastic_theory.sim_hybrid`, the running-first counterexample is reproduced, and the invariants are checked. | ~150 lines |
| 5 | `tests/test_vllm_sched_engine.py` (pod) | Builds the vLLM `Scheduler` with mock configs, as in vLLM's own `tests/v1/core`. It checks kicks, the prefill choice, and preload gating. | ~150 lines |
| 6 | `scripts/profile_batch.py` | Recalibrates a, c and α if the GPU changes. | exists |

**Order.**
- Steps 1, 4 and 5 come first. They cover the proved rules, and H1 needs them.
- Then step 2 and step 3.
- Steps 4 and 5 must pass before any GPU sweep.

## 3. Configurations

| ID | Priority | Admission | Holding during tool waits | Preloads / E6 |
|---|---|---|---|---|
| C0a | stock FCFS | stock (running-first) | stock LRU | off |
| C0b | stock priority policy with our Z′ | stock (running-first) | stock LRU | off |
| C1 | Continuum-like: preempted > pinned > program FCFS | stock | hard pin with TTL, unpin on deadlock | off |
| C2 | Autellix-like: program attained service | kicks | none (evict) | off |
| C2′ | LAMPS-like: memory-area only | kicks | evict | off |
| C3 | ours: Z′ | prefix + kicks | evict-always | off |
| C4 | ours | prefix + kicks | evict-always | on |
| C5 | ours | prefix + kicks | soft hold, R5 TTL | on |

**Ablations.**
- C3 with FCFS priority isolates the index.
- C5 with hard pins isolates P2.

**Labeling.** C1, C2 and C2′ are our reimplementations of the published policies. Label them "-like".

## 4. Workload and matrix

**Workload.** Same harness and paired design as the preload study:
- `scripts/real_multiagent.py`, Qwen3-8B, fan-out agent tasks with random tool lags;
- 180 s arrival traces;
- seeds 2–4;
- KV caps 40k / 50k / 80k / full;
- rates 0.3 and 0.6.

**Size.** 10 configs × 4 caps × 3 seeds × ~1.2 rates ≈ 140 runs. At about 4.2 minutes each that is about 10
GPU-hours.

**Runtime is set by the trace, not by GPU speed.** The fix for wall time is parallelism, not a faster GPU.
- Use 4 A40s: either 4 pods, or one 4×A40 pod running 4 vLLM servers, one per seed/config group.
- That gives about 2.5–3 h of wall time, at roughly the same cost (about $6–8 at $0.59/GPU-hr).
- A different GPU type requires recalibrating a, c and α (step 6), plus a new rate that reproduces the same
  memory pressure.

**Metrics.**
- weighted mean and p90 task completion time;
- kicks per second and recomputed tokens;
- prefix hit rate;
- realized lower bounds (L1 with the preload-aware fix, L2+L3), computed from the logged token counts and lag
  realizations. This gives ALG/LB for H5.

**Analysis.** Paired bootstrap CIs, as in `progress_since_pivot.md` §6.10.

## 5. Budget and stop rules

- Before the sweep, run a 1-seed smoke test of C0a, C3 and C5 at 50k on one GPU, about 15 min. If the invariants
  fail or C3 is worse than C0a, stop and debug before scaling out.
- The preloading results (`../project`) remain the fallback for the report if the sweep isn't done in time.
