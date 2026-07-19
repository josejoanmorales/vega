"""Cross-process run lock (WI-088): the ONE gate between the scheduled
(launchd) and on-demand (web UI) pipeline triggers, so a second starter of
either kind can never run alongside the first.

Non-blocking by design (`LOCK_EX | LOCK_NB`): a caller wants to know
IMMEDIATELY whether a run is already in progress, not queue behind it — the
whole point is one honest refusal, never an interleaved pipeline. `flock` is
held only for the file descriptor's lifetime and is released automatically if
the holding process dies (crash, kill -9), so there is no stale-lock state to
clean up.
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from vega.common.paths import DATA_ROOT

DEFAULT_LOCK_PATH = DATA_ROOT / "run.lock"


class RunInProgress(RuntimeError):
    """Another process already holds the run lock."""


@contextmanager
def acquire_run_lock(path: Path = DEFAULT_LOCK_PATH) -> Iterator[None]:
    """Hold the run lock for the duration of the `with` block. Raises
    `RunInProgress` immediately if another process already holds it — never
    blocks waiting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunInProgress("a pipeline run is already in progress") from exc
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def is_run_in_progress(path: Path = DEFAULT_LOCK_PATH) -> bool:
    """Probe without holding the lock — acquire-and-immediately-release."""
    try:
        with acquire_run_lock(path):
            return False
    except RunInProgress:
        return True
