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
