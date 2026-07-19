"""The locked daily pipeline: uv run python -m vega.run [days]

Runs ingest then the briefing under `runlock.acquire_run_lock` — this is what
launchd's plist and the web UI's `/api/run` both invoke, so a second starter
of either kind gets one honest refusal instead of an interleaved pipeline.
"""

from __future__ import annotations

import sys

from vega.briefing.__main__ import main as run_briefing
from vega.common.runlock import RunInProgress, acquire_run_lock
from vega.data import ingest


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    try:
        with acquire_run_lock():
            summary = ingest.run(days)
            print(
                f"ingest ok — added: {summary.clean_rows} clean / "
                f"{summary.quarantined_rows} quarantined, frozen (already stored): "
                f"{summary.frozen_rows}, vendor drift on frozen rows: {summary.drift_rows}, "
                f"dates touched: {len(summary.dates)}"
            )
            run_briefing()
    except RunInProgress:
        print("a pipeline run is already in progress — skipping this trigger")
        sys.exit(1)


if __name__ == "__main__":
    main()
