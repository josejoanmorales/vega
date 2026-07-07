"""Signal contract for the backtest engine.

A signal receives ONLY a MarketView (never raw data) and returns proposals to
enter. Exit mechanics (ATR stop, time stop, profit/trail) are computed by the
engine (simulate.py) from the params a proposal carries — this keeps every
signal's price-space handling identical and centralizes the PIT-safety rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vega.backtest.market_view import MarketView


@dataclass(frozen=True)
class EntryProposal:
    symbol: str
    signal_family: str
    signal_version: str
    thesis: str
    confidence: float
    invalidation: str
    time_stop_days: int = 15
    stop_atr_mult: float = 2.0
    profit_take_half_at_r: float = 2.0
    profit_trail_atr_mult: float = 2.5


class Signal(Protocol):
    family: str
    version: str
    promotable: bool

    def scan(self, view: MarketView, universe: list[str]) -> list[EntryProposal]: ...


class SmaCrossSignal:
    """Trivial placeholder: fast SMA crosses above slow SMA on adj_close.

    NON-PROMOTABLE by design — exists to smoke-test the engine end-to-end
    (WI-063 definition of done), not as a candidate trading signal.
    """

    family = "sma_cross_placeholder"
    version = "0.1"
    promotable = False

    def __init__(self, fast: int = 10, slow: int = 30, asset_class: str = "equity") -> None:
        self.fast = fast
        self.slow = slow
        self.asset_class = asset_class

    def scan(self, view: MarketView, universe: list[str]) -> list[EntryProposal]:
        proposals = []
        stop_mult = 2.5 if self.asset_class == "crypto" else 2.0
        for symbol in universe:
            bars = view.bars(symbol, lookback=self.slow + 5)
            if len(bars) < self.slow + 1:
                continue
            closes = bars["adj_close"]
            fast_now = closes.tail(self.fast).mean()
            slow_now = closes.tail(self.slow).mean()
            fast_prev = closes.iloc[:-1].tail(self.fast).mean()
            slow_prev = closes.iloc[:-1].tail(self.slow).mean()
            crossed_up = fast_prev <= slow_prev and fast_now > slow_now
            if not crossed_up:
                continue
            proposals.append(
                EntryProposal(
                    symbol=symbol,
                    signal_family=self.family,
                    signal_version=self.version,
                    thesis=f"{self.fast}/{self.slow} SMA cross (placeholder, non-promotable)",
                    confidence=0.5,
                    invalidation="fast SMA crosses back below slow SMA",
                    stop_atr_mult=stop_mult,
                )
            )
        return proposals
