"""Average True Range — shared by the backtest engine (WI-063) and the live
risk engine (WI-064) so stop-distance math can never silently diverge between
simulation and live sizing.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_PERIOD = 14


def _true_range(prev_close: float, high: float, low: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def compute_atr(
    frame: pd.DataFrame, symbol: str, as_of: str, period: int = DEFAULT_PERIOD
) -> float | None:
    """Raw-OHLC ATR through `as_of` inclusive. None if there isn't enough history."""
    bars = (
        frame[(frame["symbol"] == symbol) & (frame["date"] <= as_of)]
        .sort_values("date")
        .tail(period + 1)
    )
    if len(bars) < period + 1:
        return None
    trs = [
        _true_range(prev, high, low)
        for prev, high, low in zip(
            bars["close"].iloc[:-1], bars["high"].iloc[1:], bars["low"].iloc[1:], strict=True
        )
    ]
    return sum(trs) / len(trs)
