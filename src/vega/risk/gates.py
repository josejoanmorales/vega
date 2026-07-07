"""Hard gates checked at proposal time — never advisory (STRATEGY.md §5.D).

Each gate returns a Rejection with a specific, auditable reason, or None.
Order matters only for which single reason surfaces first; all three are
independently enforced.
"""

from __future__ import annotations

from datetime import date, timedelta

from vega.regime.calendar import in_macro_window, next_earnings
from vega.regime.regime import RegimeState
from vega.risk.types import Rejection


def regime_gate(symbol: str, regime: RegimeState) -> Rejection | None:
    if regime.composite == "risk_off":
        return Rejection(
            symbol, "regime_risk_off", f"regime composite is risk_off as of {regime.as_of}"
        )
    return None


def macro_gate(symbol: str, on_date: date) -> Rejection | None:
    if in_macro_window(on_date, days_before=1):
        return Rejection(
            symbol,
            "macro_window",
            f"a scheduled FOMC/CPI event falls within T-1..T of {on_date.isoformat()}",
        )
    return None


def earnings_gate(symbol: str, on_date: date, horizon_days: int) -> Rejection | None:
    raw = next_earnings(symbol)
    if raw is None:
        return None
    try:
        earnings_date = date.fromisoformat(raw[:10])
    except ValueError:
        return None  # unparseable vendor response is not a fact worth rejecting on
    horizon_end = on_date + timedelta(days=horizon_days)
    if on_date <= earnings_date <= horizon_end:
        return Rejection(
            symbol,
            "earnings_in_horizon",
            f"earnings on {earnings_date.isoformat()} falls within the {horizon_days}-day horizon",
        )
    return None


def check_all_gates(
    symbol: str, on_date: date, horizon_days: int, regime: RegimeState
) -> Rejection | None:
    return (
        regime_gate(symbol, regime)
        or macro_gate(symbol, on_date)
        or earnings_gate(symbol, on_date, horizon_days)
    )
