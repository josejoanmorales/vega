"""Hard gates checked at proposal time — never advisory (STRATEGY.md §5.D).

Each gate returns a Rejection with a specific, auditable reason, or None.

The earnings gate consumes a caller-supplied EarningsFact instead of doing
network I/O itself (the engine's "no network" promise is real, not aspirational)
— and it FAILS CLOSED: an unavailable earnings lookup rejects the entry, because
"the vendor is down" must never mean "permission granted".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from vega.regime.calendar import in_macro_window, next_earnings
from vega.regime.regime import RegimeState
from vega.risk.types import Rejection


@dataclass(frozen=True)
class EarningsFact:
    """The caller's answer to "when does this symbol report next?".

    status: "date" (known, `date` set) | "none" (no earnings concept, e.g. crypto)
            | "unavailable" (lookup failed — gates fail CLOSED on this)
    """

    status: str
    date: str | None = None

    @classmethod
    def lookup(cls, symbol: str, asset_class: str) -> EarningsFact:
        """Resolve the fact OUTSIDE the pure engine (this does network I/O).

        Callers run this per candidate before propose(); crypto never hits the
        vendor (no earnings concept — also kills the yfinance 404 noise).
        """
        if asset_class == "crypto":
            return cls("none")
        raw = next_earnings(symbol)
        if raw is None:
            return cls("unavailable")
        try:
            return cls("date", date.fromisoformat(raw[:10]).isoformat())
        except ValueError:
            return cls("unavailable")


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


def earnings_gate(
    symbol: str, on_date: date, horizon_sessions: int, earnings: EarningsFact
) -> Rejection | None:
    if earnings.status == "none":
        return None
    if earnings.status == "unavailable":
        return Rejection(
            symbol,
            "earnings_unknown",
            "earnings date could not be resolved — failing closed, not open",
        )
    assert earnings.date is not None  # noqa: S101 — "date" status guarantees it
    earnings_date = date.fromisoformat(earnings.date)
    # horizon is in sessions; convert conservatively (sessions <= calendar days)
    horizon_end = on_date + timedelta(days=horizon_sessions * 7 // 5 + 1)
    if on_date <= earnings_date <= horizon_end:
        return Rejection(
            symbol,
            "earnings_in_horizon",
            f"earnings on {earnings.date} falls within the {horizon_sessions}-session horizon",
        )
    return None


def check_all_gates(
    symbol: str,
    on_date: date,
    horizon_sessions: int,
    regime: RegimeState,
    earnings: EarningsFact,
) -> Rejection | None:
    return (
        regime_gate(symbol, regime)
        or macro_gate(symbol, on_date)
        or earnings_gate(symbol, on_date, horizon_sessions, earnings)
    )
