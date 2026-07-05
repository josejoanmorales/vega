"""Binance public klines — canonical crypto daily OHLCV (single venue, no key).

Uses the official public market-data mirror (data-api.binance.vision), which is
not geo-restricted. Only completed UTC sessions are returned — the still-forming
day is excluded so bars are final.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

from vega.data.types import BAR_COLUMNS

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
TIMEOUT = 30

RawKlines = dict[str, list[list[Any]]]


def fetch_daily(pair_by_symbol: dict[str, str], days: int) -> tuple[pd.DataFrame, RawKlines]:
    raw: RawKlines = {}
    for symbol, pair in pair_by_symbol.items():
        resp = requests.get(
            BASE_URL,
            params={"symbol": pair, "interval": "1d", "limit": str(days + 1)},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        raw[symbol] = resp.json()
    return normalize(raw), raw


def normalize(raw: RawKlines) -> pd.DataFrame:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    rows: list[dict[str, object]] = []
    for symbol, klines in raw.items():
        for k in klines:
            date = datetime.fromtimestamp(int(k[0]) / 1000, tz=UTC).strftime("%Y-%m-%d")
            if date >= today:
                continue  # still-forming UTC session — bar is not final
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "adj_close": float(k[4]),
                    "volume": float(k[5]),
                    "source": "binance",
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    return pd.DataFrame(rows)[list(BAR_COLUMNS)]
