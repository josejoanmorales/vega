"""The locked daily pipeline: uv run python -m vega.run [days]

Runs ingest then the briefing under `runlock.acquire_run_lock` — this is what
launchd's plist and the web UI's `/api/run` both invoke, so a second starter
of either kind gets one honest refusal instead of an interleaved pipeline.

Skip semantics (WI-088 review): losing the lock race is the system WORKING —
it exits with EXIT_SKIPPED (3), distinct from real failure (1), so neither
launchd's log nor the web UI's watcher can mistake a correct no-op for a
broken pipeline. One short retry first: `is_run_in_progress` probes hold the
exclusive lock for a microsecond, and a probe colliding with our acquire must
not cost the day's scheduled run — a retry converts probe-collisions into
successes while a genuinely concurrent pipeline still holds the lock two
seconds later and is still honestly skipped.

Degraded semantics (WI-103, from the 2026-07-21..24 incident): a vendor
outage inside ingest must never cancel exit monitoring. Ingest gets one
retry after a backoff; if it still fails, the run CONTINUES to the briefing
on the already-stored data — `briefing.__main__` self-protects (stale store
gates NEW entries; exits are still evaluated, a late stop beats an unmanaged
one) — and exits with EXIT_DEGRADED (4): not success (data is missing), not
EXIT 1 (positions were monitored), not EXIT_SKIPPED (work happened). Any
non-success outcome also fires a local notification, because this incident
ran silent for three mornings before a human noticed.
"""

from __future__ import annotations

import subprocess
import sys
import time
import traceback

from vega.briefing.__main__ import main as run_briefing
from vega.common.runlock import RunInProgress, acquire_run_lock
from vega.data import ingest

EXIT_SKIPPED = 3
EXIT_DEGRADED = 4
RETRY_DELAY_S = 2.0
INGEST_RETRY_DELAY_S = 60.0


def notify(title: str, message: str) -> None:
    """Best-effort local notification (macOS only, stdlib only). Never raises:
    alerting must not be able to break the pipeline it reports on."""
    if sys.platform != "darwin":
        return
    script = f"display notification {_as_script_str(message)} with title {_as_script_str(title)}"
    try:
        subprocess.run(  # noqa: S603 — fixed binary, arguments quoted by _as_script_str
            ["/usr/bin/osascript", "-e", script], check=False, capture_output=True, timeout=10
        )
    except Exception:  # noqa: BLE001, S110 — best-effort by contract, nothing to log to
        pass


def _as_script_str(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ingest_with_retry(days: int) -> bool:
    """One retry after a backoff, then degrade. Returns True when ingest
    succeeded (either attempt)."""
    for attempt in (1, 2):
        try:
            summary = ingest.run(days)
        except Exception:  # noqa: BLE001 — any vendor/network error degrades, never aborts
            print(f"ingest attempt {attempt} FAILED:", file=sys.stderr)
            traceback.print_exc()
            if attempt == 1:
                time.sleep(INGEST_RETRY_DELAY_S)
                continue
            return False
        print(
            f"ingest ok — added: {summary.clean_rows} clean / "
            f"{summary.quarantined_rows} quarantined, frozen (already stored): "
            f"{summary.frozen_rows}, vendor drift on frozen rows: {summary.drift_rows}, "
            f"dates touched: {len(summary.dates)}"
        )
        return True
    raise AssertionError("unreachable")


def _pipeline(days: int) -> bool:
    ingest_ok = _ingest_with_retry(days)
    if not ingest_ok:
        print(
            "DEGRADED: ingest failed twice — continuing to the briefing on the "
            "stored data (exits still evaluated; new entries self-gate on staleness)"
        )
    run_briefing()
    return ingest_ok


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    for attempt in (1, 2):
        try:
            with acquire_run_lock():
                try:
                    ingest_ok = _pipeline(days)
                except Exception:
                    notify("Vega pipeline FAILED", "daily run crashed — check the run log")
                    raise
            if not ingest_ok:
                notify(
                    "Vega pipeline DEGRADED",
                    "ingest failed; briefing/exits ran on stored data",
                )
                sys.exit(EXIT_DEGRADED)
            return
        except RunInProgress:
            if attempt == 1:
                time.sleep(RETRY_DELAY_S)  # a status probe's microsecond hold self-heals
                continue
            print("a pipeline run is already in progress — skipping this trigger")
            sys.exit(EXIT_SKIPPED)


if __name__ == "__main__":
    main()
