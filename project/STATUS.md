# STATUS (updated 2026-11-07, ~03:00 UTC pod clock)

## Implemented
- Theory (Gate 0): `restricted.py` (subset DP + exhaustive checker), `affine.py` (token cost model, O(n^2)
  prefix DP, fold bound), `online_mdp.py` (exact online policy, static baselines, RSS). `docs/theory.md`
  independently reviewed; 3 false claims + gaps fixed (review log in its Section 9). A second agent is
  re-reviewing it and extending it for speculative decoding (A) and KV pre-loading (B).
- Runtime (Gate 0.5, CPU-verified): `clock.py` (virtual asyncio time), `state.py`, `trace.py` (JSONL),
  `tools.py` (BM25 + reproducible hidden delays), `datasets.py` (HotpotQA from the HF parquet mirror, sha256
  manifest, invented fixture), `prompts.py`, `model_client.py` (fake backend + streaming vLLM client, tested
  against a mock SSE server only), `policies.py` (incremental, batch2/3, fold, rss, optimal), `runtime.py`
  (event loop, single deadline, guards), `metrics.py` (HotpotQA-style F1/EM).
- Experiment pipeline: `experiment.py` (YAML config, randomized resumable runs, per-task checkpoint),
  `cli.py` (smoke / run / prepare-data / profile / plot / summarize), `profile.py` (latency grid + affine
  least-squares fit; recovers synthetic coefficients), `plot.py` (two-panel figure, question-level bootstrap).
- Pod scripts: `scripts/pod_setup.sh` (vllm==0.32.0), `scripts/serve_model.sh` (pinned flags, prefix caching
  off, max-num-seqs 1), `scripts/gate1.sh` (serve → health → smoke → profile).
- Configs: `configs/fake_hotpot.yaml`, `configs/smoke.yaml`, `configs/pilot.yaml` (deadlines TBD at Gate 1).

## Commands run
- `pytest -q` → 1277 passed, 231 skipped (incl. theory-agent extensions + regressions for RSS infeasible
  batches, per-loop vLLM client, consume salvage/no-retry, call-id normalization) (skips: cases where the all-raw prompt does not fit, so Props 1–2
  do not apply).
- `prepare-data` → 7405 dev examples; manifest 20 calibration / 30 held-out (39 bridge, 11 comparison).
- Retrieval sizing (BM25, top_k=4): median ≈ 593 tokens/result (p90 898, max 1156) → context binds via
  `final_context_cap` ≈ 2048.
- Fake HotpotQA run (75 runs) → `results/fake/fake-hotpot-20261106-195143` (plumbing check only).

## Measured results (Gate 1; RunPod A40 46 GB, CA-MTL-1, driver 595.91.07, vLLM 0.32.0, torch 2.13.0+cu130, Qwen3-8B bf16)
- Smoke (`results/real/smoke-gate1`, 3 calibration questions, D=60, fold vs rss): 6/6 valid. Mean completion
  fold 14.29 s vs rss 10.78 s, identical F1 (dq 0.841). Trace 5a85b2d955 shows the mechanism: rss summarizes
  c2,c3 (2.58 s) while c1,c4 are still running; fold does the same summary after the last return (9.68 s).
- Latency profile (`results/calibration/costs.json`, 48 samples, prompts 243-2534 tok, outputs 16-192):
  prefill 1.80e-4 s/tok (~5.5k tok/s), decode 2.934e-2 s/tok (~34 tok/s), overhead -0.04 s (~0, clamped);
  r2 = 0.99989, median rel err 0.4%.
- Token sizes (smoke traces): result median 571 est-tokens (range 262-879); server/est ratio 0.96;
  final output 11 tokens; consume output ~46 tokens/result; planning 1.6 s.
- Calibration attempts and bugs found (all fixed with regression tests; data kept):
  - `calib-gate1` (result cap 1200): 324-879-token results forced a 3rd summary after the last return →
    18/22 deadline misses. Fix: `result_token_cap: 576` (≈median) so 2 summaries always fit under 1600.
  - `calib-gate1b`: summaries hit max_tokens → JSON parse failure → results stayed raw → RSS re-summarized
    the same results until the deadline. Fix: 3-fact schema, 128 tok/result cap, salvage + logged fallback,
    never retry.
  - Summaries were silently EMPTY since the first smoke run: Qwen3 wrote "call_id": "1" for "c1", parser
    dropped it. Fix: per-call schema (call_id enum, exactly one record per result) + id normalization +
    `consume_records` trace event with fact counts. `smoke-gate1` sizes are therefore not used.
- Re-measured after fixes (`smoke-gate1c`): summary ≈102 output tok/result (≈3 s decode), ≈85 est tok in the
  final prompt, 3 facts each; consume base prompt 132; planning median 1.65 s (max 2.7 s).
- Calibration deadline check (`calib-gate1c`, 8 calibration q, measured miss / model-predicted miss):
  D=13: rss 0.38/0.19, fold 0.88/0.74, incremental 1.00/1.00; D=16: rss 0.00/0.00, fold 0.75/0.53,
  incremental 0.62/0.65; D=19: rss 0.00/0.00, fold 0.12/0.22, incremental 0.25/0.04. dq(F1) at D=16:
  rss 0.67, incremental 0.33, fold 0.10. No salvage/fallback events. Figure: results/real/calib-gate1c.
- Pilot deadlines frozen at 13/16/19 s (`configs/pilot.yaml`).

## Blockers
- None. Pod `apu73vvxwlk9qy` (A40, $0.59/hr) running since 01:00 UTC 2026-11-07. Setup lost ~35 min to a
  slow PyPI CDN on that host (1-2 MB/s); `scripts/pod_setup.sh` now pulls CUDA wheels from the PyTorch index and
  the sha256-verified vLLM wheel from GitHub.

## Budget
- Spent: ~$1.30 so far (pod since 01:00 UTC at $0.59/hr). Projected midterm: ~$3–5.

## Full plan (before midterm) — progress
1. Time breakdown: `breakdown.py` + `cli breakdown --run-dir --deadline` (critical path: planning / tool wait /
   summarizing after last return / preloading after last return / final; plus model work hidden under tool
   waits). Tested (tests/test_breakdown.py); calib-gate1c D=19: rss 11.8 s (0.8 s summarizing on the critical
   path, 5.4 s hidden), fold 16.1 s (6.0 s), incremental 16.3 s (4.0 s).
2. Normal-context control: `configs/normal.yaml` (result cap 1200, final cap 7000; else = pilot). Queued on pod.
3. KV preloading (CPU-verified, 6 tests in tests/test_preload.py):
   - `Preload` action; `RuntimeConfig.evidence_log` append-only final prompt (preloaded raws, summaries,
     remaining raws); preload = prefix-only request, max_tokens 1, charged to the deadline.
   - Guards: Preload requires evidence_log; never summarize a preloaded raw; preload/final context guards.
   - `<base>_preload` wrapper policies (fold_preload = theory B2); FakeBackend simulates block prefix caching.
   - `VLLMBackend.reset_prefix_cache()` (needs VLLM_SERVER_DEV_MODE=1); experiments reset before every run.
   - `serve_model.sh` PREFIX_CACHE=1; `scripts/preload_bench.py` (+ plot script) microbenchmark;
     `configs/preload_smoke.yaml` (live fold vs fold_preload); `distractor_passages` pooled corpus and
     `plan_schema(n_calls)` for larger evidence loads.
   - Pod pipeline after the pilot: normal control → restart with prefix caching (max-model-len 16384) →
     microbenchmark → live smoke. Cutoff for preloading: verified on the real server by ~6pm 11/7.
4. Report: only after the user reviews the pilot figure.

## Pilot result (`results/real/pilot-gate2`, 10 held-out q × {13,16,19} s × 5 policies, 150/150 runs, 0 errors)
- Miss rate (13/16/19 s): optimal 0.2/0.0/0.0, rss 0.2/0.0/0.0, fold 0.7/0.7/0.1, incremental 1.0/0.7/0.0,
  batch2 1.0/0.9/0.4. On-time F1 ≈0.77 for every policy (gain is timeliness, not answer quality).
  RSS = optimal in 29/30 question-deadline cells. Restricted-context stress setting (final cap 1600).
- Figures (after user review): `figure.png` (RSS/optimal offset ±0.12 s so neither hides the other);
  `breakdown_D{13,16,19}.png` now over ALL runs (missed runs charged D), using each tool's true finish time
  (trace `tool_return.t` is the drain time, late when the model was busy). D=16: identical 8.07 s tool wait
  for every policy; summarizing after the last return fold 4.2 s, incremental 5.4 s, rss 1.2 s.

## Normal-context control (`results/real/normal-gate2`, 90/90 runs, 0 errors; result cap 1200, final cap 7000)
- Miss (13/16/19 s): fold 0.4/0.0/0.0, rss 0.4/0.0/0.0 (rss made 0 summaries in every run = fold, as theory
  predicts when context is not binding), incremental 1.0/0.7/0.0. Mean completion at 19 s: fold 10.9 s vs
  incremental 16.8 s.
- Quality: at 19 s (no misses) incremental F1 0.83 vs fold 0.63 — the difference is 2/10 questions where the
  model answered from raw evidence with a related entity ("John Locke" for "Father of Liberalism", "Rusalka" for
  "water sprite"). Not significant at n=10; worth re-checking at n=30 post-midterm.

## KV preloading — VERIFIED on the real server (prefix caching on, max-model-len 16384, cache reset per run)
- Microbenchmark (`results/preload_bench`, 30 requests): final-answer latency after the last return, cold vs
  preloaded (median): 4 results (2.8k tok) 0.79 → 0.48 s; 8 (5.4k) 1.11 → 0.41 s; 16 (10.8k) 2.29 → 0.51 s.
  Cached fraction of the final prompt 0.72 / 0.88 / 0.94. Preload requests (hidden under tool waits) cost
  0.44 / 1.18 / 3.06 s in total (≈0.1 s per-request overhead + incremental prefill).
- Live agent smoke (`results/real/preload-smoke`, 4 calibration q, D=60): every preload hit the cache for the
  whole previous log (e.g. 675/0 → 1268/656 → 1830/1248 → final 2459/1808 prompt/cached tokens). Time after the
  last return: fold 0.60-0.72 s vs fold_preload 0.38-0.55 s; same answers (F1 0.88 both).
- At the pilot's evidence size (4 × 576 tok) the saving is only ~0.2-0.3 s; it matters at larger evidence loads.

## Serving metrics (`cli serving-metrics --run-dir --deadline`; tests in tests/test_breakdown.py)
- Pilot D=16: GPU utilization during tool waits fold 0.00, incremental 0.67, rss 0.58; model time after the
  last return fold 4.36 s, incremental 5.48 s, rss 1.71 s.

## Preload deadline experiment (`configs/preload_load.yaml`; running on pod as `preload-load`, 20 held-out q)
- 8 searches, pooled corpus (+90 distractor paragraphs), top_k 10, result cap 1500, final cap 15000, prefix
  caching on. Static policies ignore the clock, so one D=60 run gives miss(D) = P(T > D) for every D.
- Smoke (`preload-load-smoke`, 2 calibration q): fold 16.7 s vs fold_preload 15.2 s (paired diff 1.2-1.8 s), model
  time after the last return 2.49 s -> 0.97 s, same answers; incremental 29.2 s.
- Analysis: `python scripts/plot_preload_load.py results/real/preload-load`.

## Report draft
- `report/main.tex` (ICML 2025 style, `latexmk -pdf main.tex`): 3 pages + references. Figures: pilot figure,
  preload microbenchmark; Table 1 serving view. Preload deadline result pending. Author names are placeholders.
  Bib entries written from memory: verify before submission.
