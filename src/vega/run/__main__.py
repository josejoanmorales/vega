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
from vega.common.heartbeat import ping as heartbeat_ping
from vega.common.runlock import (
    EXIT_DEGRADED,
    EXIT_SKIPPED,
    EXIT_STUCK,
    STUCK_RUN_AFTER_S,
    RunInProgress,
    acquire_run_lock,
    held_for_seconds,
    holder,
)
from vega.common.timeouts import HardTimeout, hard_timeout
from vega.data import ingest

RETRY_DELAY_S = 2.0
INGEST_RETRY_DELAY_S = 60.0
# The briefing does per-candidate vendor lookups and broker calls; generous
# against a real run (seconds to low minutes), tight against a wedge.
BRIEFING_CAP_S = 900.0


def notify(title: str, message: str) -> bool:
    """Best-effort local notification (macOS only, stdlib only). Never raises:
    alerting must not be able to break the pipeline it reports on.

    Returns whether the notification was actually delivered, and says so on
    stderr when it was not (WI-129). Swallowing this silently was its own
    version of the bug under repair: an alerting path that fails invisibly is
    indistinguishable from one that never fires, and the run log is the only
    place a human can later find out which happened."""
    if sys.platform != "darwin":
        return False
    script = f"display notification {_as_script_str(message)} with title {_as_script_str(title)}"
    try:
        proc = subprocess.run(  # noqa: S603 — fixed binary, arguments quoted by _as_script_str
            ["/usr/bin/osascript", "-e", script], check=False, capture_output=True, timeout=10
        )
    except Exception as exc:  # noqa: BLE001 — best-effort by contract
        print(f"NOTIFY FAILED ({type(exc).__name__}: {exc}) — alert not delivered", file=sys.stderr)
        return False
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        print(f"NOTIFY FAILED (osascript exit {proc.returncode}: {err})", file=sys.stderr)
        return False
    return True


def _as_script_str(text: str) -> str:
    # Newlines flattened, not escaped: an embedded newline would end the
    # osascript -e statement mid-string (review: latent injection if an
    # exception message is ever interpolated into a notification).
    flat = text.replace("\n", " ").replace("\r", " ")
    return '"' + flat.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ingest_with_retry(days: int) -> bool:
    """One retry after a backoff, then degrade. Returns True when ingest
    succeeded (either attempt)."""
    for attempt in (1, 2):
        try:
            summary = ingest.run(days)
        except (Exception, HardTimeout):  # noqa: BLE001 — vendor error OR hang, both degrade
            # HardTimeout is named explicitly because it is a BaseException:
            # it must bypass the eight `except Exception` handlers downstream,
            # but THIS is the one place a hung ingest is legitimately
            # recoverable — the briefing still runs on stored data.
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
    # The briefing path is capped too (review finding): WI-129 bounded ingest
    # because that is where the incident happened, but this path makes the
    # SAME untimed vendor calls — regime.inputs.fetch_vix is a yf.download,
    # the identical call shape and library that wedged for 3d13h — and it is
    # the path that evaluates exits and PLACES ORDERS. A hang here is not
    # degradable: unlike ingest there is no stored-data fallback, nothing was
    # monitored, so it propagates as a hard failure.
    with hard_timeout(BRIEFING_CAP_S, "briefing"):
        run_briefing()
    return ingest_ok


def alert(title: str, detail: str, heartbeat: str = "fail") -> None:
    """Every bad-outcome announcement goes through here, so no path can grow
    an on-machine notification without also telling the external watcher —
    the local channel cannot report a wedged or offline machine, and the
    external one cannot pop a banner. They are not redundant."""
    notify(f"Vega pipeline {title}", detail)
    heartbeat_ping(heartbeat, f"{title}: {detail}")


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    for attempt in (1, 2):
        try:
            with acquire_run_lock():
                # Tell the watchdog we began BEFORE any work: a start with no
                # matching completion is precisely a wedge, and it is the only
                # signal that survives one (WI-129 review blocker — launchd
                # never starts a second run to notice locally).
                heartbeat_ping("start", f"run started (days={days})")
                try:
                    ingest_ok = _pipeline(days)
                except (Exception, HardTimeout) as exc:
                    # HardTimeout named explicitly: it is a BaseException by
                    # design, so a plain `except Exception` here would let a
                    # hung briefing exit with NO alert and NO fail ping —
                    # silently, which is the entire bug class under repair.
                    alert("FAILED", f"daily run crashed ({type(exc).__name__}) — check the log")
                    raise
            if not ingest_ok:
                # Deliberately a FAIL ping, not ok: exits were evaluated, but
                # the data is stale and a human should know that today.
                alert("DEGRADED", "ingest failed; briefing/exits ran on stored data")
                sys.exit(EXIT_DEGRADED)
            heartbeat_ping("", "run completed")
            return
        except RunInProgress:
            if attempt == 1:
                time.sleep(RETRY_DELAY_S)  # a status probe's microsecond hold self-heals
                continue
            # WI-129: "someone else is running" and "the last run wedged and
            # never let go" are the same RunInProgress. Treating both as a
            # benign skip is what let the 2026-07-27 hang disable exit
            # monitoring silently for 3d13h — a skip neither alerts nor
            # degrades. Age the hold to tell them apart.
            age = held_for_seconds()
            if age is not None and age > STUCK_RUN_AFTER_S:
                who = holder() or {}
                detail = (
                    f"run lock held {age / 3600:.1f}h by pid {who.get('pid', '?')} "
                    f"since {who.get('started_at', '?')} — pipeline is NOT running"
                )
                print(f"STUCK RUN: {detail}", file=sys.stderr)
                alert("STUCK", detail)
                sys.exit(EXIT_STUCK)
            # A genuine race: the OTHER run owns this cycle and will send its
            # own completion ping. Staying silent here is deliberate — pinging
            # would let a skip stand in for work this process did not do.
            print("a pipeline run is already in progress — skipping this trigger")
            sys.exit(EXIT_SKIPPED)


if __name__ == "__main__":
    main()
