# codesign: provable scheduling for agent serving with tool waits and bounded KV

This is the new direction, kept separate from the preloading study in `../project`, which stays intact as the
fallback.

- `docs/stochastic_plan.md`: the plan and literature check.
- `docs/stochastic_theory.md`, `docs/stochastic_theory_ext.md`: theorems and proofs.
- `docs/stochastic_theory_verification.md`, `docs/stochastic_theory_ext_verification.md`: independent
  verification.
- `docs/vllm_codesign_spec.md`: the engine rules for vLLM 0.32.
- `docs/experiment_plan.md`: code steps, configurations and the GPU matrix.
- `src/codesign/`: exact checkers for the theory, and later the vLLM scheduler.
- `src/deadline_agent/`, `scripts/`, `data/`: **copies** of the preloading harness, reused for GPU experiments.
- `docs/{controller_theory,open_problem,general_theory,preload_theory,verification_report,progress_since_pivot}.md`:
  copies of the earlier docs that the theory cites.

Tests: `../project/.venv/bin/python -m pytest -q` from this folder. `pythonpath = ["src"]` makes this folder's
copy of `deadline_agent` win over the editable install. The 13 skips are preloading-measurement checks whose
GPU data lives in `../project/results`.
