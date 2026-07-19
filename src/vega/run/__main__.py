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
"""

from __future__ import annotations

import sys
import time

from vega.briefing.__main__ import main as run_briefing
from vega.common.runlock import RunInProgress, acquire_run_lock
from vega.data import ingest

EXIT_SKIPPED = 3
RETRY_DELAY_S = 2.0


def _pipeline(days: int) -> None:
    summary = ingest.run(days)
    print(
        f"ingest ok — added: {summary.clean_rows} clean / "
        f"{summary.quarantined_rows} quarantined, frozen (already stored): "
        f"{summary.frozen_rows}, vendor drift on frozen rows: {summary.drift_rows}, "
        f"dates touched: {len(summary.dates)}"
    )
    run_briefing()


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    for attempt in (1, 2):
        try:
            with acquire_run_lock():
                _pipeline(days)
            return
        except RunInProgress:
            if attempt == 1:
                time.sleep(RETRY_DELAY_S)  # a status probe's microsecond hold self-heals
                continue
            print("a pipeline run is already in progress — skipping this trigger")
            sys.exit(EXIT_SKIPPED)


if __name__ == "__main__":
    main()
