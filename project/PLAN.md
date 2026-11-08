# Plan: deadline-aware serving of multi-agent LLM workloads on one GPU

> **2026-11-07: the active direction is `../codesign`** (provable task-level scheduling on vLLM). Its midterm
> report is `codesign/report/main.pdf`; status and next steps are in `codesign/docs/experiment_plan.md` §6–7.
> This plan covers the preloading study, which is complete and is the fallback.

Last updated 2026-11-07. Replaces the old single-agent, summarization-centric plan (kept only in git history:
commit `4e56d77`). Summarization is no longer part of the project.

## 1. Problem

Many LLM agents share one GPU. Each task has an arrival time, a deadline and a value. Each task is an agent
program, possibly with several collaborating agents: a DAG of LLM calls and off-GPU tool calls (search, code,
APIs). The serving system decides online, without knowing future arrivals, tool durations or output
lengths:

1. **Batching**: which ready LLM calls, across all tasks and agents, run together in each engine step
   (single-request vs batched decoding; step time grows with batch size).
2. **KV cache**: which prefixes to keep, evict or recompute while an agent waits on a tool, and when to
   **preload** (prefill ahead of need during idle GPU time).
3. **Tool calls**: synchronous vs asynchronous issue, and when each result is consumed, interleaved with
   other agents' GPU work.
4. **Admission**: which tasks to keep serving when the GPU is overloaded.

Objective: maximize the total value of tasks finished by their deadlines.

Both settings are in scope:
- **(a) multi-tenant**: many independent agent tasks share the GPU;
- **(b) collaborating agents**: one task is several agents, e.g. an orchestrator plus workers.

## 2. Approach

The exact problem is NP-hard, and it must be solved online anyway, so the goal is a **constant-factor
competitive online algorithm**. That needs resource augmentation (a constant-factor faster GPU and/or more KV
memory than the optimum it is compared with). The plan:

1. reduce each resource to a classical online problem with known constant-factor algorithms;
2. compose the reductions into one policy;
3. prove the combined guarantee;
4. implement it as a layer on top of vLLM;
5. measure real speedups.

Every proof step gets a machine check: Z3 for algebra, exhaustive search and randomized property tests for
combinatorics.

## 3. Theory: status

Full statements and proofs: `docs/general_theory.md`. Verification: `tests/test_general_theory.py`.

| Result | Status |
|---|---|
| Theorem 1: NP-hard even with no batching, no tools, one node per task (reduces to Knapsack) | proved; checked against brute force on 300 instances |
| Online lower bounds without augmentation (Baruah et al. 1992; Koren and Shasha 1995) | known; citations to verify |
| Lemma 2: an engine with step time a + c·B behaves like m = a/c identical machines, up to a factor 2 in speed (decode steps) | proved; algebra by Z3, McNaughton step by property test, simulation on 40 instances |
| Lemma 3: issuing tool calls as soon as their inputs exist never hurts the offline optimum (uncapacitated tools) | proved; 300 instances, plus a counterexample when tools share capacity |
| 4.3: deadline-ordered (EDF) admission of LLM calls is constant-competitive with speed augmentation, for DAG tasks with tool delays | **to do** (main open proof) |
| 4.4: KV keep / evict / preload as online caching with variable sizes and recompute costs | **to do** |
| 4.5: composition of 4.1–4.4 into one competitive ratio (charge recompute to speed augmentation) | **to do** |
| Lemma 2 extended to mixed prefill + decode steps | **to do** |
| KV-preload optimality for one fork-join task | was proved in the removed single-task framework; **re-prove** in the new model |

Next theory step (4.3): prove that global EDF by task deadline, at speed about 2 − 1/m on m machines, meets
every deadline that any unit-speed schedule meets, when tasks are DAGs of GPU jobs and tool delays. With
Lemma 2 this becomes a GPU about 2(2 − 1/m) times faster. The hard part is work carried in from tasks that
arrived before the task that misses its deadline. If the general proof does not close, fall back to:
- independent jobs (Phillips, Stein, Torng, Wein 1997, citation to verify), plus
- a bounded loss from splitting task deadlines into per-node sub-deadlines.

## 4. System: status

Code in `src/deadline_agent/`:

| Component | File | Status |
|---|---|---|
| Serving layer: per-request priority (fcfs / edf / llf) plus admission control (`+ac`) | `multiagent.py` | built, tested |
| Agent programs: fan-out (plan → parallel async searches → answer), orchestrator + k workers | `multiagent.py` | built, tested |
| KV preloading during tool waits (prefill each returned result into the prefix cache) | `multiagent.py` (fan-out) | built; measured single-task (Section 5) |
| Simulated continuous-batching engine (step = a + c·B + prefill, KV capacity, priority queue) | `engine_sim.py` | built, tested |
| vLLM integration: `--scheduling-policy priority`, per-request integer `priority` | `model_client.py`, `scripts/serve_model.sh` | built; smoke run pending |
| Theory helpers (knapsack, McNaughton, engine/machine simulation) | `general_theory.py` | built, tested |
| KV keep/evict policy, reuse-time aware | — | to do |
| Batch-size cap at m = a/c, and prefill admission control | — | to do (the cap is a server flag today) |
| Custom vLLM scheduler (`scheduler_cls`), only if priorities plus admission are not enough | — | to do / maybe |

## 5. Measurements so far

All on an A40 with Qwen3-8B in bf16, vLLM 0.32.0.

- **Step-time profile** (2026-11-07, `results/calibration/batch_profile.json`):
  - per-token decode time = a + c·B with a = 27.8 ms and c = 0.543 ms per sequence, so **m = a/c ≈ 51**;
  - fit over B = 1…128: R² 0.9992, worst point 3% off;
  - B = 256 excluded: it exceeds the KV cache (159,904 tokens) and forces preemption.
- **KV preload, single task** (2026-11-06; runs with no summarization; data in `results/`):
  - microbenchmark: final-call latency 0.79 → 0.48 s at 2.8k tokens, 2.29 → 0.51 s at 10.8k;
  - deadline run (20 questions, 8 searches, ~11k-token final prompt): preload faster on 20/20, mean 1.61 s;
  - miss rate at a 16 s deadline 0.45 → 0.15; model time after the last tool return 2.34 → 0.73 s;
  - 87% of the final prompt served from cache;
  - these runs used the old single-task runtime, which is now removed; the fan-out agent in `multiagent.py`
    implements the same preload and is what gets rerun.
- **Simulated multi-tenant sweep** (630 runs, `results/sim/multiagent/`; simulator, not GPU). On-time
  fraction at c = 0.5 ms, close to the measured c:

  | Load (tasks/s) | FCFS | EDF | EDF + admission control |
  |---|---|---|---|
  | ≤ 1.5 | 0.96–0.98 | same | same |
  | 3 | 0.57 | 0.57 | 0.73 |
  | 4 | 0.29 | 0.29 | 0.55 |

  - Admission control is the big lever.
  - Deadline ordering alone changes nothing unless requests actually queue in the engine.

## 6. Next steps, in order

1. **Real multi-tenant runs** (pod running; server up with priority scheduling):
   - smoke test, confirming vLLM accepts our priorities;
   - `scripts/real_multiagent.py` with policies fcfs, edf, edf+ac, llf+ac at rates 1, 2, 3, 4 tasks/s;
   - the same with the server capped at `MAXSEQS=51` (= m);
   - the same with `--preload`.
2. **Rerun the simulator at the measured a and c**, then compare sim vs real per policy, to validate the
   simulator.
3. **Theory 4.3** (EDF with speed augmentation for DAG tasks with tool delays): proof plus machine checks:
   - an exact small-instance feasibility checker;
   - randomized tests of EDF at the claimed speed.
4. **KV policy**: keep/evict by expected time until the tool returns. Build it, add it to the layer, measure it
   under KV pressure (long contexts, many waiting agents).
5. **Theory 4.4 and 4.5** (caching reduction and composition).
6. **Final evaluation**:
   - workloads: fan-out, orchestrator, and a mix;
   - metrics: on-time value vs load, GPU utilization, cache hit rates;
   - an ablation of each component: batching cap, priorities, admission control, preload, KV policy.
7. Verify every bib entry against the source.

## 7. Deliverables and dates

| Date | Deliverable |
|---|---|
| 2026-11-08, noon | Midterm report: 2–4 pages, ICML template, one quantitative figure. Author names still placeholders; you review before submitting. |
| 2026-12-01 | Presentation |
| 2026-12-16 | Final report |

## 8. Infrastructure

- GPU pod `duv6d7rvqgqk5q`: A40, EU-SE-1, $0.59/hr.
  - SSH: direct, port 22050; the host key is pinned and matches the pod log.
  - Setup: `scripts/pod_setup.sh`, which installs vLLM 0.32.0 into `/opt/venv`.
  - Serve: `PREFIX_CACHE=1 MAXLEN=16384 MAXSEQS=256 SCHED=priority scripts/serve_model.sh`.
- Old pod `apu73vvxwlk9qy` (CA-MTL-1) is stopped and cannot restart: its host has no free GPU. Its 50 GB disk
  holds only re-downloadable files, so you can delete it.
- Local checks: `.venv/bin/python -m pytest -q`.

## 9. Rules for claims

- Label every number as measured (GPU) or simulated.
- Never report a benchmark that was not run.
- Mark theory claims as proved (with a machine check), known (with a verified citation) or target.
