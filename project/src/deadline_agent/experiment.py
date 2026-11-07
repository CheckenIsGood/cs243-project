"""Experiment configuration and the run loop: questions × deadlines × policies × latency seeds.

Results go to <out_dir>/<run_id>/: config.yaml (frozen copy), manifest.json, results.csv (one row per task
run, resumable), traces/*.jsonl. Fake-backend runs use the virtual clock; vLLM runs use real time.
"""

from __future__ import annotations

import asyncio
import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .affine import AffineCostModel
from .clock import run_virtual
from .datasets import Gold, Task, fixture_task, from_hotpot, load_hotpot, make_manifest
from .metrics import exact_match, f1_score
from .model_client import FakeBackend, VLLMBackend
from .runtime import RuntimeConfig, run_task
from .tools import DelayModel, SearchTool


@dataclass
class ExperimentConfig:
    name: str = "smoke"
    backend: str = "fake"  # fake | vllm
    base_url: str = "http://127.0.0.1:8000"
    model: str = "Qwen/Qwen3-8B"
    data_path: str = "data/raw/hotpot_distractor_validation.parquet"
    manifest_path: str = "data/manifests/pilot.json"
    split: str = "fixture"  # fixture | calibration | heldout
    n_questions: int = 1
    n_calibration: int = 20
    n_heldout: int = 30
    policies: list[str] = field(default_factory=lambda: ["incremental", "batch2", "fold", "rss", "optimal"])
    deadlines_s: list[float] = field(default_factory=lambda: [30.0])
    latency_seeds: list[int] = field(default_factory=lambda: [1])
    order_seed: int = 0
    out_dir: str = "results/fake"
    # tool / workload
    n_calls: int = 4
    top_k: int = 4
    result_token_cap: int = 1200
    distractor_passages: int = 0  # pooled corpus: add this many paragraphs from other questions (deterministic)
    delay: dict[str, float] = field(default_factory=lambda: dict(lo=0.5, hi=4.0, slow_p=0.3, slow_lo=4.0, slow_hi=12.0))
    # runtime
    final_context_cap: int = 4096
    evidence_log: bool = False  # append-only final prompt layout (required by *_preload policies)
    prefix_caching: bool = False  # server runs with prefix caching; cache is reset before every task run
    plan_max_tokens: int = 128
    consume_out_base: int = 32
    consume_out_per_result: int = 64
    final_max_tokens: int = 192
    dt: float = 0.1
    # controller's frozen cost model (from `profile`), and the fake backend's latency model
    costs: dict[str, Any] = field(default_factory=dict)
    costs_path: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def cost_model(self) -> AffineCostModel:
        params = dict(self.costs)
        if self.costs_path:
            params.update(json.loads(Path(self.costs_path).read_text())["affine"])
        params.setdefault("context_limit", self.final_context_cap)
        params.setdefault("consume_out_base", self.consume_out_base)
        params.setdefault("consume_out_per_result", self.consume_out_per_result)
        params.setdefault("final_out", self.final_max_tokens)
        return AffineCostModel(**params)

    def runtime(self, deadline: float, virtual: bool) -> RuntimeConfig:
        return RuntimeConfig(deadline_s=deadline, n_calls=self.n_calls, plan_max_tokens=self.plan_max_tokens,
                             consume_out_base=self.consume_out_base, consume_out_per_result=self.consume_out_per_result,
                             final_max_tokens=self.final_max_tokens, final_context_cap=self.final_context_cap,
                             dt=self.dt, charge_controller=virtual, evidence_log=self.evidence_log)


def load_tasks(cfg: ExperimentConfig) -> list[tuple[Task, Gold]]:
    if cfg.split == "fixture":
        return [fixture_task()]
    examples = load_hotpot(cfg.data_path)
    mp = Path(cfg.manifest_path)
    if mp.exists():
        manifest = json.loads(mp.read_text())
    else:
        manifest = make_manifest(examples, cfg.n_calibration, cfg.n_heldout)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(manifest, indent=1))
    byid = {e["_id"]: e for e in examples}
    out = [from_hotpot(byid[i]) for i in manifest[cfg.split][: cfg.n_questions]]
    if cfg.distractor_passages:
        # Larger evidence load: each question's 10 paragraphs plus paragraphs of other questions, so different
        # queries return different passages. Seeded by task id, never includes manifest questions.
        import hashlib
        from dataclasses import replace as dc_replace

        used = set(manifest["calibration"]) | set(manifest["heldout"])
        pool = [e for e in examples if e["_id"] not in used]
        aug = []
        for task, gold in out:
            rng = random.Random(int(hashlib.sha256(task.task_id.encode()).hexdigest()[:12], 16))
            extra = [p for e in rng.sample(pool, -(-cfg.distractor_passages // 10)) for p in from_hotpot(e)[0].passages]
            aug.append((dc_replace(task, passages=task.passages + tuple(extra[: cfg.distractor_passages])), gold))
        out = aug
    return out


RESULT_FIELDS = ["run_id", "task_id", "policy", "deadline_s", "latency_seed", "valid", "miss_reason",
                 "completion_s", "answer", "gold", "f1", "em", "deadline_quality", "n_calls", "n_summarized",
                 "n_folded", "n_model_calls", "prompt_tokens", "completion_tokens", "controller_s", "trace_path"]


def run_experiment(cfg: ExperimentConfig, run_id: str | None = None) -> Path:
    run_id = run_id or f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(cfg.out_dir) / run_id
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    tasks = load_tasks(cfg)
    delay = DelayModel(**cfg.delay)
    pmf = delay.pmf(cfg.dt)
    costs = cfg.cost_model()
    virtual = cfg.backend == "fake"

    jobs = [(t, g, d, p, s) for t, g in tasks for d in cfg.deadlines_s for p in cfg.policies for s in cfg.latency_seeds]
    random.Random(cfg.order_seed).shuffle(jobs)  # randomized run order (PLAN.MD Section 8)
    csv_path = out / "results.csv"
    done = set()
    if csv_path.exists():
        with csv_path.open() as f:
            done = {(r["task_id"], float(r["deadline_s"]), r["policy"], int(r["latency_seed"])) for r in csv.DictReader(f)}
    new_file = not csv_path.exists()
    (out / "manifest.json").write_text(json.dumps({"run_id": run_id, "jobs": len(jobs), "tasks": [t.task_id for t, _ in tasks]}, indent=1))

    async def one(task: Task, deadline: float, policy: str, seed: int):
        # One backend per task run: each real-time run gets its own event loop (asyncio.run), and an HTTP
        # client's connection pool is bound to the loop that created it.
        backend = FakeBackend(costs, prefix_cache=cfg.prefix_caching) if virtual else VLLMBackend(cfg.base_url, cfg.model)
        if any(p.endswith("_preload") for p in cfg.policies) and not (cfg.evidence_log and cfg.prefix_caching):
            raise ValueError("*_preload policies need evidence_log: true and prefix_caching: true")
        tool = SearchTool(task.task_id, list(task.passages), delay, seed, top_k=cfg.top_k,
                          result_token_cap=cfg.result_token_cap, inline=virtual)
        trace = out / "traces" / f"{task.task_id}__{policy}__D{deadline:g}__s{seed}.jsonl"
        if trace.exists():
            trace.unlink()
        try:
            if cfg.prefix_caching:  # every run starts cold: no reuse across policies/questions (e.g. the plan prompt)
                await backend.reset_prefix_cache()
            return await run_task(task, policy, costs, pmf, backend, tool, cfg.runtime(deadline, virtual), run_id,
                                  str(trace))
        finally:
            if hasattr(backend, "aclose"):
                await backend.aclose()

    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            w.writeheader()
        for i, (task, gold, deadline, policy, seed) in enumerate(jobs):
            if (task.task_id, float(deadline), policy, seed) in done:
                continue
            coro = one(task, deadline, policy, seed)
            res, _ = run_virtual(coro) if virtual else asyncio.run(coro)
            ans = res.answer or ""
            f1, em = (f1_score(ans, gold.answer), exact_match(ans, gold.answer)) if res.answer else (0.0, 0.0)
            w.writerow({"run_id": run_id, "task_id": task.task_id, "policy": policy, "deadline_s": deadline,
                        "latency_seed": seed, "valid": res.valid, "miss_reason": res.miss_reason or "",
                        "completion_s": "" if res.completion_s is None else round(res.completion_s, 4),
                        "answer": ans, "gold": gold.answer, "f1": round(f1, 4), "em": em,
                        "deadline_quality": round(f1 if res.valid else 0.0, 4), "n_calls": res.n_calls,
                        "n_summarized": res.n_summarized, "n_folded": res.n_folded, "n_model_calls": res.n_model_calls,
                        "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                        "controller_s": round(res.controller_s, 6), "trace_path": res.trace_path})
            f.flush()  # checkpoint after every task
            print(f"[{i + 1}/{len(jobs)}] {task.task_id[:10]} D={deadline:g} {policy:11s} seed={seed} "
                  f"valid={res.valid} t={res.completion_s and round(res.completion_s, 2)} f1={f1:.2f}", flush=True)
    return out
