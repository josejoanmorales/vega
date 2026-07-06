"""Fetch + snapshot the regime inputs that live outside the validated bar store.

^VIX has no free cross-check source (Alpaca doesn't serve indices), so it is
stored as a labeled single-source series; fear/greed likewise. Both are
snapshotted raw before use — regime computation only ever reads stored data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests
import yfinance as yf

from vega.data import snapshot

FNG_URL = "https://api.alternative.me/fng/"
TIMEOUT = 30


def fetch_vix(days: int, root: Path = snapshot.DATA_ROOT) -> pd.DataFrame:
    """Daily ^VIX closes via yfinance; snapshotted raw; returns [date, close]."""
    raw = yf.download(tickers="^VIX", period=f"{days}d", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = cast(pd.DataFrame, raw.xs("^VIX", axis=1, level=1))
    frame = pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in raw.index],
            "close": raw["Close"].to_numpy(dtype="float64"),
        }
    ).dropna()
    snapshot.snapshot_raw_frame("yfinance_vix", "vix", frame, root)
    return frame


def fetch_fear_greed(limit: int, root: Path = snapshot.DATA_ROOT) -> pd.DataFrame:
    """Crypto Fear & Greed index (alternative.me, keyless); returns [date, value]."""
    resp = requests.get(FNG_URL, params={"limit": str(limit), "format": "json"}, timeout=TIMEOUT)
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    snapshot.snapshot_raw_json("alternative_me", "fng", payload, root)
    rows = [
        {
            "date": pd.Timestamp(int(item["timestamp"]), unit="s", tz="UTC").strftime("%Y-%m-%d"),
            "value": int(item["value"]),
        }
        for item in payload.get("data", [])
    ]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
