"""Paper P&L with a slippage haircut — paper fills are never taken at face value.

Stated assumption (WI-061): 10 bps per side for equities/ETFs, 30 bps per side
for crypto; configurable at call sites.
"""

from __future__ import annotations

SLIPPAGE_BPS = {"equity": 10.0, "etf": 10.0, "crypto": 30.0}


def haircut_prices(
    entry: float, exit_: float, asset_class: str, bps: dict[str, float] | None = None
) -> tuple[float, float]:
    rate = (bps or SLIPPAGE_BPS)[asset_class] / 10_000.0
    return entry * (1 + rate), exit_ * (1 - rate)


def paper_pnl(
    entry: float, exit_: float, qty: float, asset_class: str, bps: dict[str, float] | None = None
) -> float:
    eff_entry, eff_exit = haircut_prices(entry, exit_, asset_class, bps)
    return round((eff_exit - eff_entry) * qty, 2)
