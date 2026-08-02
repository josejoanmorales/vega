"""Hard wall-clock caps on work that can HANG (WI-129).

WI-103 made a vendor call that RAISES non-fatal. The 2026-07-27 incident was
the other failure mode: a fetch that never returned and never raised, so the
process sat at 0%% CPU for 3d13h holding the run lock — no exception meant no
retry, no degrade, no alert.

Per-request timeouts on each adapter are necessary but NOT sufficient: they
only cover the calls we know about, in libraries whose internals we do not
control (the incident involved yfinance's own threads plus its SQLite tz
cache). A SIGALRM-based cap around the whole operation is the catch-all — it
interrupts a blocked syscall regardless of which library is stuck, and turns
"hangs forever" into an ordinary exception the existing degrade path already
knows how to handle.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager


class HardTimeout(TimeoutError):
    """Wall-clock cap exceeded — the operation hung rather than failing."""


@contextmanager
def hard_timeout(seconds: float, what: str = "operation") -> Iterator[None]:
    """Raise `HardTimeout` in the main thread if the block outruns `seconds`.

    No-ops (with the block still running normally) when it cannot arm a
    signal — off the main thread, or on a platform without SIGALRM. That is
    deliberate: a cap that cannot be armed must not break the caller, and the
    only current caller runs on the main thread of a CLI process.
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _fire(signum: int, frame: object) -> None:
        raise HardTimeout(f"{what} exceeded its {seconds:g}s wall-clock cap (hung, not failed)")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)  # cancel before restoring
        signal.signal(signal.SIGALRM, previous)
