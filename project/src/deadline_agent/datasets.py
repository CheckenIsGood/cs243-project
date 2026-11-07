"""HotpotQA distractor tasks and a deterministic manifest. Gold answers stay with the evaluator."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .tools import Passage

# The original host (curtis.ml.cmu.edu) no longer resolves; the HotpotQA org's Hugging Face mirror serves
# the same distractor dev set as parquet.
HOTPOT_DEV_URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
                  "distractor/validation-00000-of-00001.parquet")


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    passages: tuple[Passage, ...]
    qtype: str = ""


@dataclass(frozen=True)
class Gold:
    task_id: str
    answer: str


def from_hotpot(ex: dict) -> tuple[Task, Gold]:
    passages = tuple(Passage(title, f"{title}: " + " ".join(s.strip() for s in sents)) for title, sents in ex["context"])
    return Task(ex["_id"], ex["question"], passages, ex.get("type", "")), Gold(ex["_id"], ex["answer"])


def load_hotpot(path: str | Path) -> list[dict]:
    """Load HotpotQA examples in the original JSON layout from either .json or the HF .parquet mirror."""
    path = Path(path)
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    import pyarrow.parquet as pq

    out = []
    for r in pq.read_table(path).to_pylist():
        ctx = r["context"]
        out.append({"_id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
                    "level": r["level"], "context": [[t, list(ss)] for t, ss in zip(ctx["title"], ctx["sentences"])]})
    return out


def download_hotpot_dev(dest: str | Path) -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HOTPOT_DEV_URL, dest)
    return dest


def _rank(task_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{task_id}".encode()).hexdigest()


def make_manifest(examples: list[dict], n_calibration: int, n_heldout: int, types=("comparison", "bridge"),
                  salt: str = "deadline-agent-v1") -> dict:
    """Deterministic split, fixed before any policy tuning. Selection uses only id hashes, never answers."""
    pool = sorted((e for e in examples if e.get("type") in types), key=lambda e: _rank(e["_id"], salt))
    cal = [e["_id"] for e in pool[:n_calibration]]
    held = [e["_id"] for e in pool[n_calibration : n_calibration + n_heldout]]
    return {"salt": salt, "types": list(types), "calibration": cal, "heldout": held}


def fixture_task() -> tuple[Task, Gold]:
    """Hand-written correctness fixture (invented entities). Not benchmark data."""
    paras = {
        "Arden Valley Rowing Club": "The Arden Valley Rowing Club is an amateur rowing club on the Arden River. "
        "It was founded in 1871 by a group of mill workers. The club hosts an annual regatta each June.",
        "Merrow Bay Sailing Society": "The Merrow Bay Sailing Society is a yacht club on the northern coast. "
        "It was established in 1894 and its first commodore was a retired harbour pilot.",
        "Arden River": "The Arden River flows for 140 kilometres through farmland before joining the estuary.",
        "Harbour pilots": "Harbour pilots guide ships through congested or dangerous waters near ports.",
        "Regatta": "A regatta is a series of boat races, usually for rowing or sailing boats.",
        "Mill workers": "Mill workers in the nineteenth century often formed sports and social clubs.",
    }
    task = Task(
        "fixture-0001",
        "Which was founded earlier, the Arden Valley Rowing Club or the Merrow Bay Sailing Society?",
        tuple(Passage(k, f"{k}: {v}") for k, v in paras.items()),
        "comparison",
    )
    return task, Gold(task.task_id, "Arden Valley Rowing Club")
