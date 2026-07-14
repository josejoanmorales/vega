"""trend_pullback_v1 — buy the first up-close after a pullback inside a rising uptrend.

Economic rationale (recorded in the RationaleRegistry before this family's
first backtest, per the WI-065-enforced gate): time-series momentum is one of
the most persistent documented anomalies. Multi-day pullbacks within an
established uptrend are produced mainly by short-horizon profit-taking and
stop-runs — liquidity events, not information events — while the slower
informed flow that built the trend continues. Buying the first strength after
such a pullback harvests trend continuation at improved entry with definable
risk (the pullback low). Persists because buying recent weakness is
psychologically uncomfortable and career risk keeps institutions from
providing this liquidity systematically. Counterparty: profit-takers and
stopped-out momentum chasers.

Falsified if: pullback entries in confirmed uptrends show no positive drift
over 5-15 sessions versus a time-matched baseline across regimes.
"""

from __future__ import annotations

from vega.backtest.market_view import MarketView
from vega.backtest.signals import EntryProposal
from vega.signals.helpers import sma

LOOKBACK = 65
SMA_WINDOW = 50
TREND_CONFIRM_LAG = 10
HIGH_WINDOW = 10


class TrendPullbackSignal:
    family = "trend_pullback_v1"
    version = "1"
    promotable = True

    def __init__(self, depth: float) -> None:
        """depth: required drawdown from the prior 10-session closing high
        (grid: 3%, 5%)."""
        self.depth = depth

    def scan(self, view: MarketView, universe: list[str]) -> list[EntryProposal]:
        proposals = []
        for symbol in universe:
            bars = view.bars(symbol, lookback=LOOKBACK)
            if len(bars) < SMA_WINDOW + TREND_CONFIRM_LAG + 1:
                continue
            closes = bars["adj_close"]

            sma_now = sma(closes, SMA_WINDOW)
            sma_10ago = sma(closes.iloc[:-TREND_CONFIRM_LAG], SMA_WINDOW)
            if sma_now is None or sma_10ago is None:
                continue
            close_now = float(closes.iloc[-1])
            uptrend = close_now > sma_now and sma_now > sma_10ago

            # the 10-session high STRICTLY BEFORE today (today may itself be
            # the recovery day, which must not count toward its own high)
            prior_window = closes.iloc[-(HIGH_WINDOW + 1) : -1]
            if len(prior_window) < HIGH_WINDOW:
                continue
            high10 = float(prior_window.max())
            drawdown = (high10 - close_now) / high10 if high10 > 0 else 0.0

            first_up_day = close_now > float(closes.iloc[-2])

            if uptrend and drawdown >= self.depth and first_up_day:
                proposals.append(
                    EntryProposal(
                        symbol=symbol,
                        signal_family=self.family,
                        signal_version=self.version,
                        thesis=(
                            f"first up-close after a {self.depth:.0%}+ pullback from the "
                            f"{HIGH_WINDOW}-session high, inside a rising {SMA_WINDOW}-SMA uptrend"
                        ),
                        confidence=0.55,
                        invalidation=f"close falls back below the {SMA_WINDOW}-session SMA",
                    )
                )
        return proposals
