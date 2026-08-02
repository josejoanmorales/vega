"""yfinance adapter — canonical equity/ETF daily bars (consolidated tape volume)."""

from __future__ import annotations

from typing import cast

import pandas as pd
import yfinance as yf

from vega.data.types import BAR_COLUMNS

# Per-request cap (WI-129). yfinance defaults to no timeout on this path: the
# 2026-07-27 wedge sat here for 3d13h. This bounds each HTTP request, NOT the
# whole call — yf.download fans out across its own thread pool, so the
# `hard_timeout` around ingest.run remains the actual guarantee.
REQUEST_TIMEOUT_S = 30


def fetch_daily(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers=" ".join(symbols),
        start=start,
        end=end,
        auto_adjust=False,  # keep raw Close AND Adj Close; adjustment stays explicit
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=REQUEST_TIMEOUT_S,
    )
    return normalize(raw, symbols)


def normalize(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        if isinstance(raw.columns, pd.MultiIndex):
            if sym not in raw.columns.get_level_values(0):
                continue
            sub = cast(pd.DataFrame, raw[sym])
        else:
            sub = raw
        sub = sub.dropna(subset=["Close"])
        if sub.empty:
            continue
        adj = sub["Adj Close"] if "Adj Close" in sub.columns else sub["Close"]
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": [d.strftime("%Y-%m-%d") for d in sub.index],
                    "open": sub["Open"].to_numpy(dtype="float64"),
                    "high": sub["High"].to_numpy(dtype="float64"),
                    "low": sub["Low"].to_numpy(dtype="float64"),
                    "close": sub["Close"].to_numpy(dtype="float64"),
                    "adj_close": adj.to_numpy(dtype="float64"),
                    "volume": sub["Volume"].to_numpy(dtype="float64"),
                    "source": "yfinance",
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    return pd.concat(frames, ignore_index=True)[list(BAR_COLUMNS)]
