"""Alpaca adapter — IEX-feed close cross-check for equities.

Alpaca free tier serves the IEX feed only (~2-3% of the consolidated tape):
closes are usable for cross-validation, but this volume is NOT representative
and must never be consumed downstream (STRATEGY.md §6, Pillar 2).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import cast

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from vega.data.types import BAR_COLUMNS


def fetch_daily(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    )
    # universe uses Yahoo class-share notation (BF-B); Alpaca wants BF.B
    request = StockBarsRequest(
        symbol_or_symbols=[s.replace("-", ".") for s in symbols],
        timeframe=TimeFrame.Day,
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        adjustment=Adjustment.SPLIT,  # match Yahoo's split-adjusted Close semantics
        feed=DataFeed.IEX,
    )
    bars = cast(BarSet, client.get_stock_bars(request))
    return normalize(bars.df)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    df = raw.reset_index()
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].str.replace(".", "-", regex=False),  # back to Yahoo notation
            "date": [t.strftime("%Y-%m-%d") for t in df["timestamp"]],
            "open": df["open"].astype("float64"),
            "high": df["high"].astype("float64"),
            "low": df["low"].astype("float64"),
            "close": df["close"].astype("float64"),
            "adj_close": df["close"].astype("float64"),
            "volume": df["volume"].astype("float64"),
            "source": "alpaca_iex",
        }
    )
    return out[list(BAR_COLUMNS)]
