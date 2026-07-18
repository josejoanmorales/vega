"""Deterministic briefing assembly — reads the clean store, never live prices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from vega.briefing.calls import EligibleFamily, RenderedCall, RenderedRejection
from vega.data import snapshot
from vega.execution.executor import read_failures
from vega.execution.exits import ExitDecision
from vega.lifecycle.live_trades import DemotionOutcome
from vega.regime.calendar import MacroEvent, macro_events_within
from vega.regime.inputs import fetch_fear_greed, fetch_vix
from vega.regime.regime import RegimeState, compute_regime


@dataclass(frozen=True)
class BriefingData:
    as_of: str
    regime: RegimeState
    movers_equity: pd.DataFrame  # symbol, close, pct
    movers_crypto: pd.DataFrame
    events: list[MacroEvent]
    failures: list[dict[str, Any]]
    store_range: tuple[str, str]
    quarantined_today: int
    # WI-067, additive: absent (all-empty defaults) renders the v1 sections
    # byte-identically — ranked calls only appear once a family is eligible.
    calls: tuple[RenderedCall, ...] = ()
    rejections: tuple[RenderedRejection, ...] = ()
    eligible_families: tuple[EligibleFamily, ...] = ()
    no_trade_reason: str | None = None
    # A calls-path failure is PUBLISHED, not just printed — a briefing on a day
    # the call engine failed must be distinguishable from "no eligible families"
    # (WI-067 review: evidence integrity requires the miss on the record).
    calls_error: str | None = None
    # WI-087, additive: today's exit triggers and per-family live-track-record
    # verdicts (empty defaults keep pre-WI-087 rendering byte-identical).
    exits: tuple[ExitDecision, ...] = ()
    signal_health: tuple[DemotionOutcome, ...] = ()


def top_movers(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol % change, latest vs prior session, sorted descending.

    Only symbols present on both of the store's last two dates are judged —
    quarantined (symbol, date) rows never reach the store, so exclusion is
    by construction.
    """
    if bars.empty:
        return pd.DataFrame(columns=["symbol", "close", "pct"])
    dates = sorted(bars["date"].unique())
    if len(dates) < 2:
        return pd.DataFrame(columns=["symbol", "close", "pct"])
    last, prev = dates[-1], dates[-2]
    merged = bars[bars["date"] == last].merge(
        bars[bars["date"] == prev][["symbol", "adj_close"]],
        on="symbol",
        suffixes=("", "_prev"),
    )
    merged["pct"] = round(
        100.0 * (merged["adj_close"] - merged["adj_close_prev"]) / merged["adj_close_prev"], 2
    )
    out = merged.rename(columns={"adj_close": "close"})[["symbol", "close", "pct"]]
    result: pd.DataFrame = out.sort_values("pct", ascending=False).reset_index(drop=True)
    return result


def assemble(root: Path = snapshot.DATA_ROOT) -> BriefingData:
    con = duckdb.connect(str(root / "vega.duckdb"), read_only=True)
    try:
        equity_bars = con.execute(
            "SELECT symbol, date, adj_close FROM bars WHERE source = 'yfinance'"
        ).df()
        crypto_bars = con.execute(
            "SELECT symbol, date, adj_close FROM bars WHERE source = 'binance'"
        ).df()
        spy = con.execute(
            "SELECT date, adj_close FROM bars WHERE symbol = 'SPY' ORDER BY date"
        ).df()
        store_min, store_max = con.execute("SELECT min(date), max(date) FROM bars").fetchone()  # type: ignore[misc]
        quarantined_today = con.execute(
            "SELECT count(*) FROM quarantine WHERE date = (SELECT max(date) FROM bars)"
        ).fetchone()[0]  # type: ignore[index]
    finally:
        con.close()

    vix = fetch_vix(days=300, root=root)
    fng = fetch_fear_greed(limit=30, root=root)
    regime = compute_regime(spy, vix, equity_bars, crypto_fg=int(fng["value"].iloc[-1]))

    return BriefingData(
        as_of=str(store_max),
        regime=regime,
        movers_equity=top_movers(equity_bars),
        movers_crypto=top_movers(crypto_bars),
        events=macro_events_within(datetime.now(UTC).date(), days_ahead=14),
        failures=read_failures(),
        store_range=(str(store_min), str(store_max)),
        quarantined_today=int(quarantined_today),
    )
