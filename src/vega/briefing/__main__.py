"""Generate today's pre-market briefing: uv run python -m vega.briefing"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from vega.briefing.engine import assemble
from vega.briefing.render import write_briefing


def main() -> None:
    data = assemble()
    stale_after = (datetime.now(UTC).date() - timedelta(days=4)).isoformat()
    if data.as_of < stale_after:
        print(f"⚠ store is stale (latest session {data.as_of}) — run the ingest first")
    path = write_briefing(data)
    print(f"briefing written: {path} (regime composite: {data.regime.composite})")


if __name__ == "__main__":
    main()
