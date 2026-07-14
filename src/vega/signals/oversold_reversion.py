"""oversold_reversion_v1 — vol-scaled 3-session shock inside an intact uptrend.

Economic rationale (recorded in the RationaleRegistry before this family's
first backtest): short-horizon reversal in liquid large caps: sharp multi-day
declines that are large relative to the stock's own volatility are
disproportionately liquidity-demand shocks (index flows, forced
deleveraging, stop cascades) rather than proportionate repricings; liquidity
providers earn a premium for absorbing them, realized as reversion over the
following days. Conditioning on an intact longer-term uptrend (close >
SMA100) filters structural declines where the shock IS the information.
Persists because providing liquidity into falling prices is capital- and
courage-constrained exactly when it pays most. Counterparty: forced/
mechanical sellers.

Falsified if: vol-scaled oversold entries above SMA100 show no net-of-cost
reversion within 7 sessions, or losses concentrate so heavily in crash
continuation that the distribution is untradeable at 2R gap-stress.

Exit override (within the doctrine's 5-20 session band, per WI-064's shared
exit-doctrine contract): reversion is fast or wrong — a 7-session time stop,
smaller profit-take target (+1.5R half), doctrine-default trail.
"""

from __future__ import annotations

from vega.backtest.market_view import MarketView
from vega.backtest.signals import EntryProposal
from vega.signals.helpers import atr14, sma, three_session_change

LOOKBACK = 115
SMA_WINDOW = 100
TIME_STOP_SESSIONS = 7
PROFIT_TAKE_HALF_AT_R = 1.5


class OversoldReversionSignal:
    family = "oversold_reversion_v1"
    version = "1"
    promotable = True

    def __init__(self, k: float) -> None:
        """k: ATR14 multiple defining the shock threshold (grid: 2.0, 2.5)."""
        self.k = k

    def scan(self, view: MarketView, universe: list[str]) -> list[EntryProposal]:
        proposals = []
        for symbol in universe:
            bars = view.bars(symbol, lookback=LOOKBACK)
            if len(bars) < SMA_WINDOW + 3:
                continue
            closes = bars["adj_close"]

            sma_now = sma(closes, SMA_WINDOW)
            if sma_now is None:
                continue
            close_now = float(closes.iloc[-1])
            if close_now <= sma_now:
                continue  # not in an intact uptrend

            change3 = three_session_change(closes)
            atr = atr14(bars, symbol, view.as_of)
            if change3 is None or atr is None or atr <= 0:
                continue
            if change3 > -self.k * atr:
                continue  # not a large-enough shock

            proposals.append(
                EntryProposal(
                    symbol=symbol,
                    signal_family=self.family,
                    signal_version=self.version,
                    thesis=(
                        f"3-session drop of {abs(change3) / atr:.1f}x ATR14 while still above "
                        f"the {SMA_WINDOW}-session SMA — liquidity-shock reversion setup"
                    ),
                    confidence=0.5,
                    invalidation=f"close falls below the {SMA_WINDOW}-session SMA",
                    time_stop_days=TIME_STOP_SESSIONS,
                    profit_take_half_at_r=PROFIT_TAKE_HALF_AT_R,
                )
            )
        return proposals
