"""vega.run entry point: lock semantics + skip exit code (WI-088 review)."""

import os
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
    "'drift_rows':0,'dates':(),'failed_sleeves':()})())})\n"
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
        "'drift_rows':0,'dates':(),'failed_sleeves':()})()\n"
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
        "'frozen_rows':0,'drift_rows':0,'dates':(),'failed_sleeves':()})()"
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


# ---- WI-129: a wedged predecessor is not a benign skip --------------------

_STUCK_SNIPPET = """
import functools, sys
from pathlib import Path
import vega.common.runlock as rl
import vega.run.__main__ as m

lock = Path(LOCK_PATH)
m.RETRY_DELAY_S = 0.1
m.acquire_run_lock = functools.partial(rl.acquire_run_lock, lock)
m.held_for_seconds = functools.partial(rl.held_for_seconds, lock)
m.holder = functools.partial(rl.holder, lock)
m.notify = lambda title, message: print(f"NOTIFY: {title} | {message}")
m.ingest = type("X", (), {"run": staticmethod(lambda days=7: 1 / 0)})
m.run_briefing = lambda: print("BRIEFING RAN")
sys.argv = ["vega.run"]
m.main()
"""


def _run_against_held_lock(lock: Path, age_s: float) -> subprocess.CompletedProcess[str]:
    """Hold the lock, backdate it to `age_s`, and let vega.run react."""
    with acquire_run_lock(lock):
        os.utime(lock, (time.time() - age_s, time.time() - age_s))
        return subprocess.run(  # noqa: S603
            [sys.executable, "-c", _STUCK_SNIPPET.replace("LOCK_PATH", repr(str(lock)))],
            capture_output=True,
            text=True,
            timeout=60,
        )


def test_a_long_held_lock_is_reported_stuck_not_skipped(tmp_path: Path) -> None:
    # The incident: the holder was alive but wedged for 3d13h, so every later
    # run exited EXIT_SKIPPED — silent by design. It must alert instead.
    proc = _run_against_held_lock(tmp_path / "run.lock", age_s=4 * 3600)
    assert proc.returncode == 5, proc.stdout + proc.stderr
    assert "NOTIFY: Vega pipeline STUCK" in proc.stdout
    assert "STUCK RUN" in proc.stderr
    assert "4.0h" in proc.stdout  # the age a human needs
    assert "BRIEFING RAN" not in proc.stdout  # nothing ran; it could not get the lock


def test_a_briefly_held_lock_still_skips_quietly(tmp_path: Path) -> None:
    # A genuine concurrent run must keep its WI-088 behaviour: quiet skip, no
    # alert, exit 3 — otherwise every real race would page a human.
    proc = _run_against_held_lock(tmp_path / "run.lock", age_s=5)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "already in progress" in proc.stdout
    assert "NOTIFY" not in proc.stdout


# ---- heartbeat wiring: which outcomes ping what -----------------------------

_HEARTBEAT_SNIPPET = """
import functools, sys
from pathlib import Path
import vega.common.runlock as rl
import vega.run.__main__ as m

lock = Path(LOCK_PATH)
m.RETRY_DELAY_S = 0.1
m.INGEST_RETRY_DELAY_S = 0.1
m.acquire_run_lock = functools.partial(rl.acquire_run_lock, lock)
m.held_for_seconds = functools.partial(rl.held_for_seconds, lock)
m.holder = functools.partial(rl.holder, lock)
m.notify = lambda title, message: True
m.heartbeat_ping = lambda kind="", detail="": print(f"PING[{kind or 'ok'}]")
m.ingest = type("X", (), {"run": staticmethod(INGEST_BEHAVIOR)})
m.run_briefing = lambda: print("BRIEFING RAN")
sys.argv = ["vega.run"]
m.main()
"""

_OK_INGEST = (
    "lambda days=7: type('S', (), {'clean_rows':0,'quarantined_rows':0,"
    "'frozen_rows':0,'drift_rows':0,'dates':(),'failed_sleeves':()})()"
)


def _heartbeat_run(lock: Path, ingest_behavior: str) -> subprocess.CompletedProcess[str]:
    snippet = _HEARTBEAT_SNIPPET.replace("LOCK_PATH", repr(str(lock))).replace(
        "INGEST_BEHAVIOR", ingest_behavior
    )
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", snippet], capture_output=True, text=True, timeout=60
    )


def test_success_pings_start_then_ok(tmp_path: Path) -> None:
    proc = _heartbeat_run(tmp_path / "run.lock", _OK_INGEST)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # start BEFORE the work: a start with no completion is what reveals a wedge
    assert proc.stdout.index("PING[start]") < proc.stdout.index("BRIEFING RAN")
    assert "PING[ok]" in proc.stdout


def test_degraded_pings_fail_not_ok(tmp_path: Path) -> None:
    # Exits ran, but the data is stale — the watchdog must not go green.
    behaviour = "lambda days=7: (_ for _ in ()).throw(ConnectionError('vendor down'))"
    proc = _heartbeat_run(tmp_path / "run.lock", behaviour)
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert "PING[start]" in proc.stdout
    assert "PING[fail]" in proc.stdout
    assert "PING[ok]" not in proc.stdout


def test_a_genuine_skip_pings_nothing(tmp_path: Path) -> None:
    """The other run owns this cycle and sends its own completion. Pinging
    here would let a skip stand in for work this process never did."""
    lock = tmp_path / "run.lock"
    with acquire_run_lock(lock):
        proc = _heartbeat_run(lock, _OK_INGEST)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "PING[" not in proc.stdout


def test_stuck_pings_fail(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    with acquire_run_lock(lock):
        os.utime(lock, (time.time() - 4 * 3600, time.time() - 4 * 3600))
        proc = _heartbeat_run(lock, _OK_INGEST)
    assert proc.returncode == 5, proc.stdout + proc.stderr
    assert "PING[fail]" in proc.stdout


# ---- the briefing (order-placing) path is capped too ------------------------


def test_a_hung_briefing_fails_hard_and_alerts(tmp_path: Path) -> None:
    """Unlike ingest there is no stored-data fallback: if the briefing hangs,
    nothing was monitored and no orders were placed, so it must fail LOUD
    rather than degrade. Regression for the review finding that this path —
    which places orders — had no cap at all."""
    snippet = (
        "import functools, sys\n"
        "from pathlib import Path\n"
        "import vega.common.runlock as rl\n"
        "import vega.run.__main__ as m\n"
        f"lock = Path({str(tmp_path / 'run.lock')!r})\n"
        "m.acquire_run_lock = functools.partial(rl.acquire_run_lock, lock)\n"
        "m.BRIEFING_CAP_S = 0.4\n"
        "m.notify = lambda t, msg: print(f'NOTIFY: {t}')\n"
        "m.heartbeat_ping = lambda kind='', detail='': print(f'PING[{kind or \"ok\"}]')\n"
        "m.ingest = type('X', (), {'run': staticmethod(lambda days=7: "
        "type('S', (), {'clean_rows':0,'quarantined_rows':0,'frozen_rows':0,"
        "'drift_rows':0,'dates':(),'failed_sleeves':()})())})\n"
        "import time as _t\n"
        "m.run_briefing = lambda: _t.sleep(60)\n"
        "sys.argv = ['vega.run']\n"
        "m.main()\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", snippet], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode != 0
    assert proc.returncode not in (3, 4), "a hung briefing is not a skip or a degrade"
    assert "NOTIFY: Vega pipeline FAILED" in proc.stdout
    assert "PING[fail]" in proc.stdout  # the external watchdog is told too
    assert "HardTimeout" in proc.stderr
