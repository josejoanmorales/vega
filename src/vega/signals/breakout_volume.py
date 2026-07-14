"""breakout_volume_v1 — a new N-session closing high on abnormal consolidated volume.

Economic rationale (recorded in the RationaleRegistry before this family's
first backtest): a close at a new N-session high on abnormally high
consolidated volume marks the absorption of overhead supply by informed
accumulation — a completed auction, not a poke. Post-breakout drift persists
because range-anchored holders distribute too early, under-positioned buyers
chase gradually, and information diffuses slowly. Low-volume breakouts lack
absorption evidence and are disproportionately failed auctions — hence the
volume gate, which REQUIRES consolidated volume (IEX's ~2-3% sample cannot
measure absorption; this family only ever runs against yfinance-sourced
equity/ETF bars). Counterparty: range-anchored sellers; short-covering
accelerates continuation.

Falsified if: high-volume N-high breakouts show no drift beyond what
low-volume breakouts show, or breakout entries mean-revert net of costs.
"""

from __future__ import annotations

from vega.backtest.market_view import MarketView
from vega.backtest.signals import EntryProposal
from vega.signals.helpers import is_new_high, median_volume

MEDIAN_VOLUME_WINDOW = 60
VOLUME_MULTIPLE = 1.5


class BreakoutVolumeSignal:
    family = "breakout_volume_v1"
    version = "1.1"  # 1.1: strict Donchian semantics — today > max of N PRIOR sessions (review fix)
    promotable = True

    def __init__(self, n_sessions: int) -> None:
        """n_sessions: breakout lookback window (grid: 40, 55)."""
        self.n_sessions = n_sessions
        self.params = {"n_sessions": n_sessions}  # recorded on every RunRecord
        self.lookback = max(n_sessions + 1, MEDIAN_VOLUME_WINDOW) + 5

    def scan(self, view: MarketView, universe: list[str]) -> list[EntryProposal]:
        proposals = []
        for symbol in universe:
            bars = view.bars(symbol, lookback=self.lookback)
            if len(bars) < max(self.n_sessions + 1, MEDIAN_VOLUME_WINDOW):
                continue
            closes = bars["adj_close"]
            volumes = bars["volume"]

            if not is_new_high(closes, self.n_sessions):
                continue
            med_vol = median_volume(volumes, MEDIAN_VOLUME_WINDOW)
            if med_vol is None or med_vol <= 0:
                continue
            today_vol = float(volumes.iloc[-1])
            if today_vol < VOLUME_MULTIPLE * med_vol:
                continue

            proposals.append(
                EntryProposal(
                    symbol=symbol,
                    signal_family=self.family,
                    signal_version=self.version,
                    thesis=(
                        f"new {self.n_sessions}-session closing high on "
                        f"{today_vol / med_vol:.1f}x median 60-session consolidated volume"
                    ),
                    confidence=0.55,
                    invalidation=f"close falls back below the pre-breakout {self.n_sessions}-high",
                )
            )
        return proposals
