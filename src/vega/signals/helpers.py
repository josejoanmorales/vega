"""Shared price/volume math for signal families — pure functions over an
already PIT-truncated bars frame (from MarketView.bars), nothing else.

NaN discipline (WI-066 review): pandas NaN is not None, and NaN comparisons
are silently False — a data-gap row must yield None/False here, explicitly,
or it slips straight through a signal's guards and fires a spurious entry.
"""

from __future__ import annotations

import pandas as pd

from vega.common.atr import compute_atr as _compute_atr

__all__ = ["adjusted_atr14", "is_new_high", "median_volume", "sma", "three_session_change"]


def sma(closes: pd.Series, window: int) -> float | None:
    if len(closes) < window:
        return None
    value = closes.tail(window).mean()
    return None if pd.isna(value) else float(value)


def is_new_high(closes: pd.Series, window: int) -> bool:
    """True if today's close STRICTLY exceeds the max of the `window` PRIOR
    sessions (today excluded) — conventional Donchian breakout semantics.

    WI-066 review fixed two defects here: the old window included today with
    `>=`, so a flat series 'broke out' every day and an 'N-session high' was
    really an (N-1)-session comparison. NaN today = False (bad data never fires).
    """
    if len(closes) < window + 1:
        return False
    today = closes.iloc[-1]
    prior_max = closes.iloc[-(window + 1) : -1].max()
    if pd.isna(today) or pd.isna(prior_max):
        return False
    return bool(today > prior_max)


def median_volume(volumes: pd.Series, window: int) -> float | None:
    if len(volumes) < window:
        return None
    value = volumes.tail(window).median()
    return None if pd.isna(value) else float(value)


def three_session_change(closes: pd.Series) -> float | None:
    if len(closes) < 4:
        return None
    now, then = closes.iloc[-1], closes.iloc[-4]
    if pd.isna(now) or pd.isna(then):
        return None  # NaN must surface as None, never leak into a threshold comparison
    return float(now - then)


def adjusted_atr14(bars: pd.DataFrame, symbol: str, as_of: str) -> float | None:
    """ATR14 in ADJUSTED price space — for signals whose thresholds compare
    against adj_close deltas (WI-066 review: comparing an adj-space move to a
    raw-space ATR mixes price spaces; dividends inside the window manufacture
    phantom shocks). Raw OHLC is scaled per-row by adj_close/close, then the
    ONE shared ATR implementation runs on the scaled frame.

    Note: risk/backtest STOP distances stay in RAW space (the space fills
    happen in) — this adjusted variant is for signal DECISION math only.
    """
    required = {"close", "high", "low", "adj_close"}
    if not required.issubset(bars.columns):
        return None
    scaled = bars[["symbol", "date", "close", "high", "low", "adj_close"]].copy()
    factor = scaled["adj_close"] / scaled["close"]
    if factor.isna().any() or (scaled["close"] <= 0).any():
        return None  # unscalable rows = unmeasurable, never guess
    scaled["high"] = scaled["high"] * factor
    scaled["low"] = scaled["low"] * factor
    scaled["close"] = scaled["adj_close"]
    return _compute_atr(scaled, symbol, as_of, period=14)
