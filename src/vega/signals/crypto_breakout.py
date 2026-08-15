"""crypto_breakout_v1 — a new N-session closing high on abnormal Binance volume.

Economic rationale (recorded in the RationaleRegistry before this family's
first backtest): a close at a new N-session high on abnormally high volume in
a liquid crypto pair marks genuine directional absorption, not noise — and
Binance can actually measure that, unlike `breakout_volume_v1`'s equity
volume gate, which exists specifically because Alpaca's IEX feed is only
~2-3% of the consolidated tape and cannot. Binance is the CANONICAL primary
source for these pairs (STRATEGY.md), so its volume genuinely reflects
market-wide participation.

Crypto breakouts should persist LONGER than the equivalent equity signal
(which failed — see breakout_volume.py's own falsification), for reasons
specific to crypto microstructure rather than a generic "momentum exists"
claim: (a) continuous 24/7 trading has no closing auction to force a session
back toward a "fair" print, so a breakout does not get mean-reverted by end-
of-day mechanics the way an equity one can be; (b) leveraged perpetual-
futures positioning on other venues is directly reflexive with spot —
crossing a widely-watched level triggers stop-outs and forced liquidations on
the losing side of that leverage, mechanically extending the move; (c)
information diffuses continuously rather than gapping at a discrete open, so
a volume-confirmed breakout reflects real-time absorption, not an overnight
gap artifact equities are prone to. Counterparty: leveraged shorts forced to
cover into the move; range-anchored spot holders distributing too early.

Falsified if: high-volume N-high breakouts show no net-of-cost continuation
over the following sessions, breakout entries mean-revert net of costs, or
losses concentrate so heavily in immediate whipsaw that the distribution
fails at the doctrine's crypto gap-stress multiple (GAP_STRESS_MULT["crypto"]
= 2.0, tighter than equity's 2.5 — crypto's shorter, sharper gaps get less
benefit of the doubt).

STOP_ATR_MULT IS EXPLICIT (not the EntryProposal default): the dataclass
default is STOP_ATR_MULT["equity"] (2.0) — asset-class-neutral by omission,
not by correctness. Any crypto signal that leaves it unset silently trades
crypto positions at the equity stop distance, understating simulated risk
against the doctrine's own crypto value (2.5) in exactly the mixed-space bug
class WI-064/066's reviews found repeatedly. `simulate.py` reads
`proposal.stop_atr_mult` directly and does not correct it by asset_class.
"""

from __future__ import annotations

from vega.backtest.market_view import MarketView
from vega.backtest.signals import EntryProposal
from vega.common.doctrine import STOP_ATR_MULT
from vega.signals.helpers import is_new_high, median_volume

MEDIAN_VOLUME_WINDOW = 60
VOLUME_MULTIPLE = 1.5
# Shorter than breakout_volume_v1's equity grid (40/55): the crypto store's
# CoinGecko-capped cross-check window bounds usable history to ~1yr (vs
# equity's ~2yr), and crypto's faster cycle time argues for a shorter,
# more-native lookback rather than forcing equity's window onto a 24/7
# market. A monthish and a six-week Donchian high, in daily (not weekly)
# sessions since crypto trades every day.
N_SESSIONS_GRID = (20, 30)


class CryptoBreakoutSignal:
    family = "crypto_breakout_v1"
    version = "1.0"
    promotable = True

    def __init__(self, n_sessions: int) -> None:
        """n_sessions: breakout lookback window (grid: 20, 30)."""
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
                        f"{today_vol / med_vol:.1f}x median 60-session Binance volume"
                    ),
                    confidence=0.5,
                    invalidation=f"close falls back below the pre-breakout {self.n_sessions}-high",
                    stop_atr_mult=STOP_ATR_MULT["crypto"],
                )
            )
        return proposals
