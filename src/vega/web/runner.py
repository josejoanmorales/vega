"""Subprocess run tracking: spawns `python -m vega.run` as a subprocess so a
pipeline crash can never take the web server down, and exposes its live
status to the API. One `Runner` instance per server process; a run's
in-memory status is lost on server restart, but the log file and (if it got
that far) the briefing itself survive — and `status()` still reports an
`external` state whenever the cross-process lock is held by a run this
server doesn't own (an orphaned child from a previous server, or launchd's
scheduled run), so the UI never claims "idle" while the machine is
mid-pipeline (WI-088 review).
"""

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vega.common.paths import DATA_ROOT, PROJECT_ROOT
from vega.common.runlock import is_run_in_progress

RUNS_DIR = DATA_ROOT / "web-runs"
LOG_TAIL_LINES = 100
EXIT_SKIPPED = 3  # mirrors vega.run.__main__.EXIT_SKIPPED — a lost lock race is not a failure


class RunAlreadyInProgress(RuntimeError):
    pass


@dataclass
class _RunState:
    run_id: str
    state: str  # "running" | "succeeded" | "skipped" | "failed"
    started_at: str
    log_path: Path
    finished_at: str | None = None
    returncode: int | None = None


class Runner:
    def __init__(self, runs_dir: Path = RUNS_DIR) -> None:
        self._lock = threading.Lock()
        self._current: _RunState | None = None
        self._runs_dir = runs_dir

    def start(self) -> str:
        """Spawn the pipeline. Raises `RunAlreadyInProgress` if this server's
        own tracked run is still active, or if the cross-process run lock is
        already held (e.g. a launchd-triggered run) — the probe gives an
        immediate honest 409; the subprocess's own lock acquire remains the
        real arbiter for anything the probe missed (a lost race surfaces as
        `skipped`, never as a duplicate pipeline)."""
        with self._lock:
            if self._current is not None and self._current.state == "running":
                raise RunAlreadyInProgress(f"run {self._current.run_id} is still in progress")
            if is_run_in_progress():
                raise RunAlreadyInProgress("a pipeline run is already in progress (external)")

            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            self._runs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self._runs_dir / f"{run_id}.log"
            log_file = log_path.open("w")
            process = subprocess.Popen(  # noqa: S603 — fixed argv, no shell, no user input
                [sys.executable, "-m", "vega.run"],
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            self._current = _RunState(
                run_id=run_id,
                state="running",
                started_at=datetime.now(UTC).isoformat(),
                log_path=log_path,
            )
            threading.Thread(target=self._watch, args=(process, log_file), daemon=True).start()
            return run_id

    def _watch(self, process: subprocess.Popen[bytes], log_file: Any) -> None:
        returncode = process.wait()
        # The state transition MUST happen even if log-file cleanup raises —
        # a watcher death pre-transition left state 'running' forever and
        # locked out every future web run until restart (WI-088 review).
        try:
            with self._lock:
                assert self._current is not None  # noqa: S101 — set by start() before this thread
                self._current.returncode = returncode
                self._current.finished_at = datetime.now(UTC).isoformat()
                if returncode == 0:
                    self._current.state = "succeeded"
                elif returncode == EXIT_SKIPPED:
                    self._current.state = "skipped"  # lost the lock race — a no-op, not a failure
                else:
                    self._current.state = "failed"
        finally:
            try:
                log_file.close()
            except OSError:
                pass  # the log is best-effort capture; state honesty comes first

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._current is None or self._current.state != "running":
                # No run of OURS is active — but the machine may still be
                # mid-pipeline (orphaned child of a dead server, or launchd).
                if is_run_in_progress():
                    external = {"state": "external"}
                    if self._current is not None:
                        external["last_run"] = self._current.state
                    return external
            if self._current is None:
                return {"state": "idle"}
            tail = ""
            if self._current.log_path.exists():
                lines = self._current.log_path.read_text().splitlines()
                tail = "\n".join(lines[-LOG_TAIL_LINES:])
            return {
                "state": self._current.state,
                "run_id": self._current.run_id,
                "started_at": self._current.started_at,
                "finished_at": self._current.finished_at,
                "returncode": self._current.returncode,
                "log_tail": tail,
            }
