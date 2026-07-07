"""Live smoke test: real equity, real regime, real store -> proposals that
round-trip into valid ledger.types.Recommendation objects (WI-064 DoD).

Run: uv run python -m vega.risk
"""

from __future__ import annotations

import os
from datetime import date
from typing import cast

import duckdb
from dotenv import load_dotenv

from vega.data import snapshot
from vega.regime.calendar import macro_events_within
from vega.regime.inputs import fetch_fear_greed, fetch_vix
from vega.regime.regime import compute_regime
from vega.risk.engine import propose, to_recommendation
from vega.risk.heat import OpenPositionHeat
from vega.risk.types import Rejection, SizedProposal

CANDIDATES = [("AAPL", "equity"), ("BTC", "crypto")]


def _live_equity() -> float:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.models import TradeAccount

    client = TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
    )
    account = cast(TradeAccount, client.get_account())
    if account.equity is None:
        raise RuntimeError("Alpaca paper account returned no equity value")
    return float(account.equity)


def main() -> None:
    load_dotenv()
    root = snapshot.DATA_ROOT
    con = duckdb.connect(str(root / "vega.duckdb"), read_only=True)
    try:
        frame = con.execute("SELECT * FROM bars").df()
        row = con.execute("SELECT max(date) FROM bars").fetchone()
        assert row is not None and row[0] is not None  # noqa: S101 — store is non-empty by now
        as_of: str = row[0]
    finally:
        con.close()

    equity = _live_equity()
    vix = fetch_vix(days=300)
    fng = fetch_fear_greed(limit=30)
    spy = frame[(frame["symbol"] == "SPY") & (frame["source"] == "yfinance")][["date", "adj_close"]]
    universe_bars = frame[frame["source"] == "yfinance"][["symbol", "date", "adj_close"]]
    regime = compute_regime(spy, vix, universe_bars, crypto_fg=int(fng["value"].iloc[-1]))

    print(f"equity: ${equity:,.2f} | regime: {regime.composite} (as_of {regime.as_of})")
    events = macro_events_within(date.fromisoformat(as_of), days_ahead=14)
    for e in events:
        print(f"  upcoming: {e.date} {e.event}")

    open_positions: list[OpenPositionHeat] = []
    for symbol, asset_class in CANDIDATES:
        source = "binance" if asset_class == "crypto" else "yfinance"
        sub = frame[(frame["symbol"] == symbol) & (frame["source"] == source)]
        if sub.empty:
            print(f"{symbol}: no store data, skipping")
            continue
        entry_ref = float(sub.sort_values("date")["adj_close"].iloc[-1])
        result = propose(
            symbol=symbol,
            asset_class=asset_class,
            entry_ref_price=entry_ref,
            frame=frame[frame["source"] == source],
            as_of=str(sub["date"].max()),
            equity=equity,
            regime=regime,
            open_positions=open_positions,
            invalidation=f"close reverses back through the {symbol} entry level",
        )
        if isinstance(result, Rejection):
            print(f"{symbol}: REJECTED — {result.reason}: {result.detail}")
            continue
        assert isinstance(result, SizedProposal)  # noqa: S101
        rec = to_recommendation(
            result,
            thesis="WI-064 live smoke test",
            confidence=0.5,
            signal_attribution=("smoke_test",),
            as_of=str(sub["date"].max()),
        )
        print(
            f"{symbol}: qty={rec.qty:.6f} entry={result.entry_ref_price:.2f} "
            f"stop={result.stop_price:.2f} R=${result.initial_r_dollars:.2f} "
            f"worst_case={result.worst_case_r_multiple:.2f}R cluster={result.cluster} "
            f"heat_after={result.heat_after}"
        )


if __name__ == "__main__":
    main()
