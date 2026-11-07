"""JSONL event traces. Times are seconds since task start on the loop's monotonic clock."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, path: str | Path | None, run_id: str, task_id: str, policy: str, t0: float):
        self.path = Path(path) if path else None
        self.run_id, self.task_id, self.policy, self.t0 = run_id, task_id, policy, t0
        self.events: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a")
        else:
            self._fh = None

    def emit(self, t_abs: float, event: str, **fields: Any) -> None:
        rec = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "policy": self.policy,
            "t": round(t_abs - self.t0, 6),
            "event": event,
            **fields,
        }
        self.events.append(rec)
        if self._fh:
            self._fh.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def wallclock_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
