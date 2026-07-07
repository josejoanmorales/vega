"""Cost model — called from inside the one fill function every trade uses.

No trade can be simulated without passing through here (backtest.simulate).
Rates sit deliberately harsher than execution/pnl.py's live paper haircuts
(10/30 bps) — a backtest must be the pessimistic estimate, never the optimistic one.
"""

from __future__ import annotations

from vega.execution.pnl import SLIPPAGE_BPS

# Every tier must sit at or above execution/pnl.py's live paper haircuts
# (equity/etf 10bps, crypto 30bps) — a backtest is the pessimistic estimate,
# never the optimistic one, by construction.
EQUITY_TIER_BPS = {"liquid": 12.0, "standard": 20.0}  # by median-dollar-volume tier
LIQUID_MEDIAN_DOLLAR_VOLUME = 50_000_000.0

CRYPTO_MAJOR_BPS = 35.0  # BTC, ETH
CRYPTO_OTHER_BPS = 50.0
CRYPTO_MAJORS = {"BTC", "ETH"}

assert min(EQUITY_TIER_BPS.values()) >= SLIPPAGE_BPS["equity"]  # noqa: S101
assert CRYPTO_MAJOR_BPS >= SLIPPAGE_BPS["crypto"]  # noqa: S101


def equity_tier(median_dollar_volume: float) -> str:
    return "liquid" if median_dollar_volume >= LIQUID_MEDIAN_DOLLAR_VOLUME else "standard"


def cost_bps(asset_class: str, symbol: str, median_dollar_volume: float | None = None) -> float:
    if asset_class == "crypto":
        return CRYPTO_MAJOR_BPS if symbol in CRYPTO_MAJORS else CRYPTO_OTHER_BPS
    tier = equity_tier(median_dollar_volume if median_dollar_volume is not None else 0.0)
    return EQUITY_TIER_BPS[tier]


def apply_cost(price: float, side: str, bps: float) -> float:
    """side='buy' raises the effective price, 'sell' lowers it — always pessimistic."""
    rate = bps / 10_000.0
    if side == "buy":
        return price * (1 + rate)
    if side == "sell":
        return price * (1 - rate)
    raise ValueError("side must be 'buy' or 'sell'")
