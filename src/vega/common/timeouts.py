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
import time
from collections.abc import Iterator
from contextlib import contextmanager

MIN_REARM_S = 0.001  # setitimer(0) means "disarm", so an overdue outer cap uses this


class HardTimeout(BaseException):
    """Wall-clock cap exceeded — the operation hung rather than failing.

    Deliberately a BaseException, NOT an Exception (review finding): the
    briefing/execution path it guards contains eight separate
    `except Exception` handlers — "unreachable venue", "one bad order must not
    stop the batch", "unknown earnings is an acceptable answer" — every one of
    which would otherwise swallow the cap and quietly convert a hung pipeline
    into a benign-looking vendor miss. Like KeyboardInterrupt and SystemExit,
    a wall-clock abort is control flow, not an error the local code may
    handle. Cleanup still runs: BaseException unwinds `finally` normally.

    A handler that genuinely wants it (only `run.__main__._ingest_with_retry`,
    which degrades on a hung ingest) must name it explicitly."""


@contextmanager
def hard_timeout(seconds: float, what: str = "operation") -> Iterator[None]:
    """Raise `HardTimeout` in the main thread if the block outruns `seconds`.

    NESTS CORRECTLY, which matters more than it sounds: there is one process
    itimer, so a naive inner block's `setitimer(0)` on exit would cancel an
    enclosing cap and the outer deadline would be silently lost forever
    (review finding — the outer cap simply never fired again). An inner cap
    is therefore clamped to whatever the outer has left (the tighter deadline
    always wins — an inner block must never be able to EXTEND an outer
    budget), and the outer's remaining time is re-armed on the way out.

    No-ops (with the block still running normally) when it cannot arm a
    signal — off the main thread, or on a platform without SIGALRM. That is
    deliberate: a cap that cannot be armed must not break the caller.
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    outer_remaining = signal.getitimer(signal.ITIMER_REAL)[0]
    outer_handler = signal.getsignal(signal.SIGALRM)
    nested = outer_remaining > 0
    effective = min(seconds, outer_remaining) if nested else seconds

    def _fire(signum: int, frame: object) -> None:
        raise HardTimeout(f"{what} exceeded its {effective:g}s wall-clock cap (hung, not failed)")

    started = time.monotonic()
    signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, effective)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)  # disarm ours before restoring
        signal.signal(signal.SIGALRM, outer_handler)
        if nested:
            # Hand the outer cap back the time it still had. Already overdue
            # (its deadline passed inside our block) => fire as soon as the
            # caller yields control, rather than dropping it.
            left = outer_remaining - (time.monotonic() - started)
            signal.setitimer(signal.ITIMER_REAL, max(left, MIN_REARM_S))
