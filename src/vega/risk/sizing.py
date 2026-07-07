"""ATR-scaled, gap-stressed position sizing (STRATEGY.md §5.A).

The min(base, gap) formula is what actually enforces "worst-case loss <= 2R"
— algebraically, for any G > 0 and k > 0, whichever term binds keeps nominal
risk <= 1R and gap-stressed risk <= 2R. The 1.5R clamp below is a defensive
invariant on top of that (catches a future caller bypassing the formula with
a manual qty override), not something today's defaults ever need to trigger.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_RISK_FRACTION = 0.0075  # 0.75% of equity per trade, mid of the 0.5-1% contract
STOP_ATR_MULT = {"equity": 2.0, "etf": 2.0, "crypto": 2.5}
GAP_STRESS_MULT = {"equity": 2.5, "etf": 2.5, "crypto": 2.0}
SINGLE_POSITION_CAP_R = 1.5  # fraction of `risk_fraction * equity` a position may ever risk


class SizingError(ValueError):
    pass


@dataclass(frozen=True)
class SizingResult:
    stop_price: float
    qty: float
    initial_r_dollars: float  # actual nominal risk at the stop, in dollars
    worst_case_r_dollars: float  # gap-stressed worst case, in dollars
    worst_case_r_multiple: float  # worst_case / (risk_fraction * equity) — should be <= ~2.0


def compute_stop(entry_price: float, atr: float, asset_class: str) -> float:
    if entry_price <= 0:
        raise SizingError("entry_price must be positive")
    if atr <= 0:
        raise SizingError("atr must be positive")
    k = STOP_ATR_MULT[asset_class]
    stop = entry_price - k * atr
    if stop <= 0:
        raise SizingError(f"computed stop {stop} is non-positive (entry={entry_price}, atr={atr})")
    return stop


def compute_qty(
    entry_price: float,
    stop_price: float,
    equity: float,
    asset_class: str,
    risk_fraction: float = DEFAULT_RISK_FRACTION,
) -> SizingResult:
    if equity <= 0:
        raise SizingError("equity must be positive")
    if not 0 < risk_fraction <= 0.05:
        raise SizingError("risk_fraction must be a small positive fraction (sanity: <=5%)")
    stop_distance = entry_price - stop_price
    if stop_distance <= 0:
        raise SizingError("stop_price must be below entry_price")

    r_dollars = risk_fraction * equity
    gap = GAP_STRESS_MULT[asset_class]

    qty_base = r_dollars / stop_distance
    qty_gap = (2.0 * r_dollars) / (gap * stop_distance)
    qty = min(qty_base, qty_gap)

    initial_r = qty * stop_distance
    worst_case_r = qty * gap * stop_distance

    cap = SINGLE_POSITION_CAP_R * r_dollars
    if initial_r > cap:
        qty = cap / stop_distance
        initial_r = qty * stop_distance
        worst_case_r = qty * gap * stop_distance

    return SizingResult(
        stop_price=stop_price,
        qty=qty,
        initial_r_dollars=round(initial_r, 6),
        worst_case_r_dollars=round(worst_case_r, 6),
        worst_case_r_multiple=round(worst_case_r / r_dollars, 4),
    )
