"""Command line: python -m deadline_agent.cli <command> [--config PATH]."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from .datasets import download_hotpot_dev, load_hotpot, make_manifest
from .experiment import ExperimentConfig, run_experiment


def summarize(run_dir: Path) -> None:
    rows = list(csv.DictReader((run_dir / "results.csv").open()))
    agg: dict = defaultdict(list)
    for r in rows:
        agg[(float(r["deadline_s"]), r["policy"])].append(r)
    print(f"\n{'deadline':>8} {'policy':12} {'n':>3} {'miss':>6} {'dq(F1)':>7} {'t_mean':>7} {'summ':>5}")
    for (d, p), rs in sorted(agg.items()):
        miss = sum(r["valid"] != "True" for r in rs) / len(rs)
        dq = sum(float(r["deadline_quality"]) for r in rs) / len(rs)
        ts = [float(r["completion_s"]) for r in rs if r["completion_s"]]
        summ = sum(int(r["n_summarized"]) for r in rs) / len(rs)
        print(f"{d:8g} {p:12} {len(rs):3d} {miss:6.2f} {dq:7.3f} {(sum(ts) / len(ts)) if ts else float('nan'):7.2f} {summ:5.2f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="deadline_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="run the configured experiment once (fake backend uses virtual time)")
    s.add_argument("--backend", choices=["fake", "vllm"], default="fake")
    s.add_argument("--config", default=None)
    r = sub.add_parser("run", help="run an experiment from a config")
    r.add_argument("--config", required=True)
    r.add_argument("--run-id", default=None)
    p = sub.add_parser("prepare-data", help="download HotpotQA dev (distractor) and write the manifest")
    p.add_argument("--config", required=True)
    pr = sub.add_parser("profile", help="Gate 1: profile vLLM latency and fit the affine cost model")
    pr.add_argument("--config", required=True)
    pr.add_argument("--out", default="results/calibration/costs.json")
    pr.add_argument("--reps", type=int, default=3)
    pl = sub.add_parser("plot", help="two-panel figure (PDF/PNG + source CSV) for a run directory")
    pl.add_argument("--run-dir", required=True)
    pl.add_argument("--overlay-planning-s", type=float, default=None,
                    help="overlay model-predicted miss curves; task deadline D -> remaining D - this (calibration)")
    bd = sub.add_parser("breakdown", help="critical-path time breakdown per policy (on-time runs at one deadline)")
    bd.add_argument("--run-dir", required=True)
    bd.add_argument("--deadline", type=float, default=None)
    sv = sub.add_parser("serving-metrics", help="GPU busy/idle during tool waits, prefill computed vs reused (per policy)")
    sv.add_argument("--run-dir", required=True)
    sv.add_argument("--deadline", type=float, default=None)
    sm = sub.add_parser("summarize", help="print a per-(deadline, policy) table for a run directory")
    sm.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "smoke":
        cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
        cfg.backend = args.backend
        if args.backend == "fake":
            cfg.out_dir = "results/fake"  # keep fake outputs separate from real results
        summarize(run_experiment(cfg))
    elif args.cmd == "run":
        cfg = ExperimentConfig.load(args.config)
        summarize(run_experiment(cfg, args.run_id))
    elif args.cmd == "prepare-data":
        cfg = ExperimentConfig.load(args.config)
        path = download_hotpot_dev(cfg.data_path)
        ex = load_hotpot(path)
        man = make_manifest(ex, cfg.n_calibration, cfg.n_heldout)
        Path(cfg.manifest_path).parent.mkdir(parents=True, exist_ok=True)
        import json

        Path(cfg.manifest_path).write_text(json.dumps(man, indent=1))
        print(f"{len(ex)} examples; manifest {cfg.manifest_path}: {len(man['calibration'])} calibration, "
              f"{len(man['heldout'])} held-out")
    elif args.cmd == "profile":
        from .profile import run_profile

        cfg = ExperimentConfig.load(args.config)
        fit = run_profile(cfg.base_url, cfg.model, args.out, reps=args.reps)
        print({k: v for k, v in fit.items() if k != "samples"})
    elif args.cmd == "plot":
        from .plot import plot

        print(plot(args.run_dir, overlay_planning_s=args.overlay_planning_s))
    elif args.cmd == "breakdown":
        from .breakdown import breakdown

        mean, path = breakdown(args.run_dir, args.deadline)
        for p, m in mean.items():
            print(p, {k: round(v, 2) if isinstance(v, float) else v for k, v in m.items()})
        print(path)
    elif args.cmd == "serving-metrics":
        from .breakdown import serving_metrics

        mean, path = serving_metrics(args.run_dir, args.deadline)
        cols = ["n_miss", "end_s", "gpu_busy_s", "gpu_busy_during_tools_s", "gpu_idle_during_tools_s",
                "gpu_util_during_tools", "model_s_after_last_return", "prefill_tokens_computed",
                "prefill_tokens_reused", "decode_tokens", "model_requests"]
        print("policy        " + " ".join(f"{c[:14]:>14s}" for c in cols))
        for p, m in mean.items():
            print(f"{p:13s} " + " ".join(f"{m[c]:14.2f}" for c in cols))
        print(path)
    elif args.cmd == "summarize":
        summarize(Path(args.run_dir))


if __name__ == "__main__":
    main()
