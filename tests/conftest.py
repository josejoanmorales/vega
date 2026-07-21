"""Shared test fixtures (WI-067 review: signal-shape frame builders were being
copied between test modules — they live here once; tune the shape in one place
and every consumer keeps testing a genuinely-firing scan path)."""

from __future__ import annotations

import pandas as pd


def make_ohlc_frame(
    closes: list[float],
    shocked: set[int],
    symbol: str = "AAA",
    volume: float | None = None,
) -> pd.DataFrame:
    """Raw OHLC (+ optional volume column) around each close; wider (more
    volatile) range on shocked indices."""
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D").strftime("%Y-%m-%d")
    rows = []
    for i, (d, c) in enumerate(zip(dates, closes, strict=True)):
        spread = 5.0 if i in shocked else 2.0
        row: dict[str, object] = {
            "symbol": symbol,
            "date": d,
            "adj_close": c,
            "open": c,  # no gap by default; tests needing one mutate this column directly
            "close": c,
            "high": c + spread,
            "low": c - spread,
        }
        if volume is not None:
            row["volume"] = volume
        rows.append(row)
    return pd.DataFrame(rows)


def steep_uptrend_then_shock(drop_total: float) -> list[float]:
    """100-bar steep uptrend (SMA100 lags well behind) then a 3-session shock —
    the canonical oversold_reversion trigger shape."""
    base = [100.0 + i * 1.0 for i in range(100)]
    peak = base[-1]
    return base + [peak - drop_total / 3, peak - 2 * drop_total / 3, peak - drop_total]


def flat_history(
    symbol: str,
    dates: list[str],
    o: float = 100.0,
    h: float = 101.0,
    low: float = 99.0,
    c: float = 100.0,
    volume: float = 1_000_000.0,
) -> list[dict]:
    """One row per date, every OHLC field pinned flat (WI-084: this exact
    zero-movement fixture was hand-copied, with drifting signatures, across
    test_common_atr.py, test_risk_engine.py, and test_backtest_simulate.py —
    ATR/heat/fill-timing tests only need constant-range bars, never a real
    price path). `open`/`volume` are extra columns most callers ignore."""
    return [
        {
            "symbol": symbol,
            "date": d,
            "open": o,
            "high": h,
            "low": low,
            "close": c,
            "adj_close": c,
            "volume": volume,
        }
        for d in dates
    ]


def flat_history_n(
    symbol: str, n: int, start_day: int = 1, month: str = "2026-05", **kwargs: float
) -> list[dict]:
    """`flat_history` for `n` sequential calendar-day dates starting `month-start_day`
    — the `_flat_history(symbol, n, ...)` shape test_common_atr.py and
    test_risk_engine.py used (test_backtest_simulate.py needs specific, often
    non-contiguous dates and calls `flat_history` directly with its own list)."""
    dates = [f"{month}-{d:02d}" for d in range(start_day, start_day + n)]
    return flat_history(symbol, dates, **kwargs)
