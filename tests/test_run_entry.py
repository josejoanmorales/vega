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


# ---- WI-103: ingest failure degrades, never aborts ------------------------

_DEGRADED_SNIPPET = (
    "import sys, vega.run.__main__ as m\n"
    "m.RETRY_DELAY_S = 0.1\n"
    "m.INGEST_RETRY_DELAY_S = 0.1\n"
    "m.notify = lambda title, message: print(f'NOTIFY: {title}')\n"
    "m.ingest = type('X', (), {'run': staticmethod(FAIL_BEHAVIOR)})\n"
    "m.run_briefing = lambda: print('BRIEFING RAN')\n"
    "sys.argv = ['vega.run']\n"
    "m.main()\n"
)

_ALWAYS_FAIL = "lambda days=7: (_ for _ in ()).throw(ConnectionError('vendor down'))"


def test_ingest_failure_still_runs_briefing_and_exits_degraded(tmp_path: Path) -> None:
    # The Jul 21-24 incident class: ingest dies on a vendor error every
    # attempt. The briefing (and with it exit evaluation) must STILL run,
    # and the exit code must be EXIT_DEGRADED (4) — not 1, not 0.
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _DEGRADED_SNIPPET.replace("FAIL_BEHAVIOR", _ALWAYS_FAIL)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 4
    assert "BRIEFING RAN" in proc.stdout
    assert "DEGRADED" in proc.stdout
    assert "NOTIFY: Vega pipeline DEGRADED" in proc.stdout
    assert "vendor down" in proc.stderr  # the failure is loud, not swallowed


def test_ingest_transient_failure_recovers_on_retry(tmp_path: Path) -> None:
    # First attempt raises, retry succeeds -> normal exit 0, no notification.
    snippet = (
        "import sys, vega.run.__main__ as m\n"
        "m.RETRY_DELAY_S = 0.1\n"
        "m.INGEST_RETRY_DELAY_S = 0.1\n"
        "m.notify = lambda title, message: print(f'NOTIFY: {title}')\n"
        "calls = {'n': 0}\n"
        "def _flaky(days=7):\n"
        "    calls['n'] += 1\n"
        "    if calls['n'] == 1:\n"
        "        raise ConnectionError('flap')\n"
        "    return type('S', (), {'clean_rows':0,'quarantined_rows':0,'frozen_rows':0,"
        "'drift_rows':0,'dates':()})()\n"
        "m.ingest = type('X', (), {'run': staticmethod(_flaky)})\n"
        "m.run_briefing = lambda: print('BRIEFING RAN')\n"
        "sys.argv = ['vega.run']\n"
        "m.main()\n"
        "assert calls['n'] == 2, calls\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "BRIEFING RAN" in proc.stdout
    assert "flap" in proc.stderr  # first failure is loud even when recovered
    assert "NOTIFY" not in proc.stdout


def test_briefing_crash_notifies_and_fails_hard(tmp_path: Path) -> None:
    # A crash AFTER ingest (in the briefing itself) is a real failure: exit
    # nonzero as before, but now with a notification so it can't run silent.
    ok_ingest = (
        "lambda days=7: type('S', (), {'clean_rows':0,'quarantined_rows':0,"
        "'frozen_rows':0,'drift_rows':0,'dates':()})()"
    )
    snippet = _DEGRADED_SNIPPET.replace("FAIL_BEHAVIOR", ok_ingest).replace(
        "m.run_briefing = lambda: print('BRIEFING RAN')",
        "m.run_briefing = lambda: (_ for _ in ()).throw(RuntimeError('briefing boom'))",
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    assert "NOTIFY: Vega pipeline FAILED" in proc.stdout
    assert "briefing boom" in proc.stderr
