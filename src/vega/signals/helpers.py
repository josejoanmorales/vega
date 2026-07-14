"""Shared price/volume math for signal families — pure functions over an
already PIT-truncated bars frame (from MarketView.bars), nothing else."""

from __future__ import annotations

import pandas as pd

from vega.common.atr import compute_atr as _compute_atr

__all__ = ["is_new_high", "median_volume", "sma", "three_session_change", "atr14"]


def sma(closes: pd.Series, window: int) -> float | None:
    if len(closes) < window:
        return None
    value = closes.tail(window).mean()
    return None if pd.isna(value) else float(value)


def is_new_high(closes: pd.Series, window: int) -> bool:
    """True if the LAST value is the maximum over the trailing `window` sessions
    (today included) — i.e. today closed at a new `window`-session high."""
    if len(closes) < window:
        return False
    tail = closes.tail(window)
    return bool(tail.iloc[-1] >= tail.max())


def median_volume(volumes: pd.Series, window: int) -> float | None:
    if len(volumes) < window:
        return None
    value = volumes.tail(window).median()
    return None if pd.isna(value) else float(value)


def three_session_change(closes: pd.Series) -> float | None:
    if len(closes) < 4:
        return None
    return float(closes.iloc[-1] - closes.iloc[-4])


def atr14(bars: pd.DataFrame, symbol: str, as_of: str) -> float | None:
    """Reuses the ONE shared ATR implementation (vega.common.atr) — `bars` is
    already the PIT-safe per-symbol slice from MarketView.bars(), so filtering
    it again by symbol/as_of inside compute_atr is a safe no-op."""
    return _compute_atr(bars, symbol, as_of, period=14)
