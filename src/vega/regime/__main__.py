"""Live regime report: fetch+snapshot inputs, read the clean store, print state.

Run: uv run python -m vega.regime
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb

from vega.data import snapshot
from vega.regime.calendar import macro_events_within
from vega.regime.inputs import fetch_fear_greed, fetch_vix
from vega.regime.regime import compute_regime


def main() -> None:
    vix = fetch_vix(days=300)
    fng = fetch_fear_greed(limit=30)

    con = duckdb.connect(str(snapshot.DATA_ROOT / "vega.duckdb"), read_only=True)
    try:
        universe_bars = con.execute(
            "SELECT symbol, date, adj_close FROM bars WHERE source = 'yfinance'"
        ).df()
        spy = con.execute(
            "SELECT date, adj_close FROM bars WHERE symbol = 'SPY' ORDER BY date"
        ).df()
    finally:
        con.close()

    state = compute_regime(spy, vix, universe_bars, crypto_fg=int(fng["value"].iloc[-1]))
    print(state)
    upcoming = macro_events_within(datetime.now(UTC).date(), days_ahead=14)
    for event in upcoming:
        print(f"upcoming: {event.date} {event.event}")


if __name__ == "__main__":
    main()
