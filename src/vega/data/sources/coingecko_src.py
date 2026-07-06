"""CoinGecko market_chart — cross-check closes for crypto (venue-aggregated).

CoinGecko's daily series carries one price point per day at 00:00 UTC; the price
at 00:00 of day D+1 is the close of UTC session D, which is exactly Binance's
1d-kline close instant — so the two series are directly comparable. Only close
is used (cross-check role); OHLC/volume are not populated by this adapter.

Free tier is rate-limited (~10-30 req/min): calls are spaced and retried on 429.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from vega.data.types import BAR_COLUMNS

BASE_URL = "https://api.coingecko.com/api/v3"
TIMEOUT = 30
CALL_SPACING_S = 7.0
MAX_RETRIES = 6

RawCharts = dict[str, dict[str, Any]]


def _get_chart(coingecko_id: str, days: int) -> dict[str, Any]:
    url = f"{BASE_URL}/coins/{coingecko_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days + 1)}
    if days <= 90:
        params["interval"] = "daily"  # free tier rejects explicit interval beyond 90 days

    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 15 * 2**attempt))
            time.sleep(min(wait, 120.0))
            continue
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        return payload
    raise RuntimeError(f"coingecko rate limit persisted after {MAX_RETRIES} retries: {url}")


def fetch_daily(ids_by_symbol: dict[str, str], days: int) -> tuple[pd.DataFrame, RawCharts]:
    raw: RawCharts = {}
    for symbol, cid in ids_by_symbol.items():
        raw[symbol] = _get_chart(cid, days)
        time.sleep(CALL_SPACING_S)
    return normalize(raw), raw


def normalize(raw: RawCharts) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, chart in raw.items():
        for ms, price in chart.get("prices", []):
            at = datetime.fromtimestamp(int(ms) / 1000, tz=UTC)
            if at.hour != 0 or at.minute != 0:
                continue  # trailing "now" point, not a 00:00 UTC daily snapshot
            close_of = (at - timedelta(days=1)).strftime("%Y-%m-%d")
            rows.append(
                {
                    "symbol": symbol,
                    "date": close_of,
                    "open": float("nan"),
                    "high": float("nan"),
                    "low": float("nan"),
                    "close": float(price),
                    "adj_close": float(price),
                    "volume": float("nan"),
                    "source": "coingecko",
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    return pd.DataFrame(rows)[list(BAR_COLUMNS)]
