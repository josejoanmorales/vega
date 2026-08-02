"""Cross-process run lock (WI-088): the ONE gate between the scheduled
(launchd) and on-demand (web UI) pipeline triggers, so a second starter of
either kind can never run alongside the first.

Non-blocking by design (`LOCK_EX | LOCK_NB`): a caller wants to know
IMMEDIATELY whether a run is already in progress, not queue behind it — the
whole point is one honest refusal, never an interleaved pipeline. `flock` is
held only for the file descriptor's lifetime and is released automatically if
the holding process dies (crash, kill -9), so there is no stale-lock state to
clean up.

DEATH is not the only way a run stops making progress (WI-129). The
2026-07-27 incident held this lock for 3d13h from a process that was alive
but wedged — flock cannot distinguish that from healthy work, so every later
run refused with `RunInProgress` and exited EXIT_SKIPPED, which is by design
a silent no-op. `held_for_seconds()` gives callers the age of the current
hold so "a run is in progress" can be told apart from "a run is stuck".

That age is the lock file's mtime, which is why acquiring must NOT truncate
the file: `open("w")` truncates BEFORE flock is attempted, so every failed
probe — and `is_run_in_progress()` runs on every dashboard poll — would
otherwise stamp the mtime fresh and make a wedged run look eternally young.
Only a successful acquisition writes.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vega.common.paths import DATA_ROOT

DEFAULT_LOCK_PATH = DATA_ROOT / "run.lock"

# Pipeline exit codes, shared by the producer (vega.run.__main__) and the
# consumer (vega.web.runner) — WI-103 review: they were mirrored constants in
# two modules, so a change in one would silently misclassify run outcomes in
# the other. This module is the run-coordination home both already import.
EXIT_SKIPPED = 3  # lost the lock race — a correct no-op, not a failure
EXIT_DEGRADED = 4  # ingest down; briefing/exit monitoring still ran on stored data
EXIT_STUCK = 5  # the lock holder is wedged — NOT a benign skip (WI-129)

# A real run is minutes; the incident ran 3d13h. An hour is far outside
# normal and far inside "a human should know".
STUCK_RUN_AFTER_S = 3600.0


class RunInProgress(RuntimeError):
    """Another process already holds the run lock."""


@contextmanager
def _flock(path: Path) -> Iterator[Any]:
    """Take the exclusive lock, or raise `RunInProgress`. Opens with O_CREAT
    but NOT O_TRUNC so merely opening never touches mtime (see docstring)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    with os.fdopen(fd, "r+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunInProgress("a pipeline run is already in progress") from exc
        try:
            yield fh
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextmanager
def acquire_run_lock(path: Path = DEFAULT_LOCK_PATH) -> Iterator[None]:
    """Hold the run lock for the duration of the `with` block. Raises
    `RunInProgress` immediately if another process already holds it — never
    blocks waiting."""
    with _flock(path) as fh:
        # Stamp the hold: this write (and only this write) sets the mtime
        # `held_for_seconds` reads, and records who to go look for.
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps({"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()}))
        fh.flush()
        os.fsync(fh.fileno())
        yield


def is_run_in_progress(path: Path = DEFAULT_LOCK_PATH) -> bool:
    """Probe without holding the lock, and WITHOUT stamping it.

    A probe must be read-only: `/api/status` calls this on every dashboard
    poll, so stamping here would rewrite+fsync the lock on a seconds-long
    cycle and, worse, overwrite the holder identity with the web server's own
    pid — corrupting exactly the evidence stuck-detection depends on."""
    try:
        with _flock(path):
            return False
    except RunInProgress:
        return True


def held_for_seconds(path: Path = DEFAULT_LOCK_PATH) -> float | None:
    """How long the CURRENT holder has held the lock, or None if never held.

    Only a successful acquisition writes the file, so its mtime is the start
    of the current (or most recent) hold. Meaningful only while the lock is
    actually held — a caller that just got `RunInProgress` knows it is."""
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except FileNotFoundError:
        return None


def holder(path: Path = DEFAULT_LOCK_PATH) -> dict[str, Any] | None:
    """The current holder's {pid, started_at}, or None if unreadable — the
    identity a human needs to go look at (or kill) a wedged run."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
