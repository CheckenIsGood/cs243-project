# codesign: provable scheduling for agent serving with tool waits and bounded KV

This is the new direction, kept separate from the preloading study in `../project`, which stays intact as the
fallback.

- `docs/stochastic_plan.md`: the plan and literature check.
- `docs/stochastic_theory.md`, `docs/stochastic_theory_ext.md`: theorems and proofs.
- `docs/stochastic_theory_verification.md`, `docs/stochastic_theory_ext_verification.md`: independent
  verification.
- `docs/vllm_codesign_spec.md`: the engine rules for vLLM 0.32.
- `docs/experiment_plan.md`: code steps, configurations and the GPU matrix.
- `src/codesign/`: exact checkers for the theory; `vllm_sched.py` (the vLLM scheduler plug-in, loaded with
  `--scheduler-cls codesign.vllm_sched.CodesignScheduler`); `client.py` (the task index Z′ and the baseline
  rules, including aging, `zprime_age`).
- `scripts/cd_sweep.sh`, `scripts/cd_run.sh`, `scripts/cd_configs.json`: the GPU sweep (configs C0a–C7, A1–A8).
  `scripts/analyze_sweep.py`, `scripts/dominance.py`: analysis. Results stay local in `results/` (gitignored).
- `report/`: the midterm report (ICML template). Build with `latexmk -pdf main.tex`; figures with
  `../project/.venv/bin/python report/make_figs.py` from this folder.
- **Status and next steps:** `docs/experiment_plan.md` §6–7.
- `src/deadline_agent/`, `scripts/`, `data/`: **copies** of the preloading harness, reused for GPU experiments.
- `docs/{controller_theory,open_problem,general_theory,preload_theory,verification_report,progress_since_pivot}.md`:
  copies of the earlier docs that the theory cites.

Tests: `../project/.venv/bin/python -m pytest -q` from this folder. `pythonpath = ["src"]` makes this folder's
copy of `deadline_agent` win over the editable install. The 13 skips are preloading-measurement checks whose
GPU data lives in `../project/results`.
