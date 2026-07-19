"""vega.run entry point: lock semantics + skip exit code (WI-088 review)."""

import subprocess
import sys
import time
from pathlib import Path

from vega.common.runlock import acquire_run_lock

# Patch the names AS BOUND inside vega.run.__main__ (it imports run_briefing
# and ingest at module load, so patching the source modules would be too late
# and would run the REAL pipeline against the live account).
_RUN_SNIPPET = (
    "import sys, vega.run.__main__ as m\n"
    "m.RETRY_DELAY_S = 0.1\n"
    "m.ingest = type('X', (), {'run': staticmethod(lambda days=7: "
    "  type('S', (), {'clean_rows':0,'quarantined_rows':0,'frozen_rows':0,"
    "'drift_rows':0,'dates':()})())})\n"
    "m.run_briefing = lambda: print('BRIEFING RAN')\n"
    "sys.argv = ['vega.run']\n"
    "m.main()\n"
)


def test_run_exits_skipped_when_lock_already_held(tmp_path: Path) -> None:
    # Hold the REAL run lock, then launch vega.run in a subprocess: it must
    # exhaust its retry, print the skip message, and exit EXIT_SKIPPED (3).
    with acquire_run_lock():
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _RUN_SNIPPET],
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert proc.returncode == 3
    assert "already in progress" in proc.stdout
    assert "BRIEFING RAN" not in proc.stdout


def test_run_succeeds_when_lock_free(tmp_path: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _RUN_SNIPPET],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "BRIEFING RAN" in proc.stdout


def test_run_retry_wins_after_transient_probe_hold(tmp_path: Path) -> None:
    # A probe-style hold released quickly must NOT cost the run — the first
    # attempt fails, the retry (0.1s later) succeeds.
    def _release_soon() -> None:
        time.sleep(0.05)

    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", _RUN_SNIPPET],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # briefly hold the lock, then release so the subprocess's retry wins
    with acquire_run_lock():
        time.sleep(0.15)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0
    assert "BRIEFING RAN" in out
