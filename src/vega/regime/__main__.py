"""Live regime report: fetch+snapshot inputs, read the clean store, print state.

Run: uv run python -m vega.regime
"""

from __future__ import annotations

from datetime import UTC, datetime

from vega.common import db
from vega.data import snapshot
from vega.regime.calendar import macro_events_within
from vega.regime.regime import assemble_regime


def main() -> None:
    root = snapshot.DATA_ROOT
    with db.connect(root) as con:
        universe_bars = con.execute(
            "SELECT symbol, date, adj_close FROM bars WHERE source = 'yfinance'"
        ).df()
        spy = con.execute(
            "SELECT date, adj_close FROM bars WHERE symbol = 'SPY' ORDER BY date"
        ).df()

    state = assemble_regime(spy, universe_bars, root=root)
    print(state)
    upcoming = macro_events_within(datetime.now(UTC).date(), days_ahead=14)
    for event in upcoming:
        print(f"upcoming: {event.date} {event.event}")


if __name__ == "__main__":
    main()
