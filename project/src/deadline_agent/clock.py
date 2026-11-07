"""Virtual time for asyncio, so the same runtime code runs instantly and deterministically in tests.

`run_virtual(coro)` runs `coro` on a fresh event loop whose `time()` is virtual: whenever the loop would
block waiting for its next timer, virtual time jumps straight to that timer. asyncio.sleep, wait_for
timeouts, etc. all work unchanged. Work done in other threads is NOT supported in virtual mode (it would
race with jumping time), so virtual-mode tools must run inline.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine


class _VirtualTime:
    def __init__(self) -> None:
        self.now = 0.0


def run_virtual(coro: Coroutine[Any, Any, Any]) -> Any:
    loop = asyncio.new_event_loop()
    vt = _VirtualTime()
    selector = loop._selector  # type: ignore[attr-defined]
    real_select = selector.select

    def virtual_select(timeout: float | None = None):
        events = real_select(0)
        if not events and timeout is not None and timeout > 0:
            vt.now += timeout
        return events

    loop.time = lambda: vt.now  # type: ignore[method-assign]
    selector.select = virtual_select
    try:
        return loop.run_until_complete(coro)
    finally:
        selector.select = real_select
        loop.close()


def now() -> float:
    """Monotonic seconds on the running loop's clock (virtual or real)."""
    return asyncio.get_running_loop().time()
