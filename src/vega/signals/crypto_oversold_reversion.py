"""crypto_oversold_reversion_v1 — vol-scaled 3-session shock inside an intact
uptrend, on Binance-sourced majors.

Economic rationale (recorded in the RationaleRegistry before this family's
first backtest): crypto's sharp short-horizon selloffs are disproportionately
driven by LEVERAGED PERPETUAL-FUTURES LIQUIDATION CASCADES on OTHER venues —
a decline triggers forced liquidation of over-leveraged longs, which
mechanically forces MORE selling beyond any fundamental repricing. Crypto has
no circuit breakers or market-maker pause mechanisms to dampen this the way
equities do, so the overshoot runs further before exhausting. The cascade
stops once over-leveraged positions are flushed, and funding-rate arbitrage
provides a natural reversion flow back: perpetual funding typically flips
negative during a crash, paying arbitrageurs to buy spot and short the perp,
which is net spot-buying pressure. Conditioning on an intact longer-term
uptrend (close > SMA100) filters structural declines where the shock IS the
information, same discipline as the equity family below. Counterparty:
forced/leveraged sellers being liquidated, not informed sellers.

THIS IS A DIFFERENT MECHANISM FROM `crypto_breakout_v1`, deliberately, not a
contradiction: that family hypothesized a volume-confirmed breakout reflects
absorption that liquidations then EXTEND (and it failed — dev Sharpe -2.92
and -1.68, sanity-checked against real trades, a genuine falsification). This
family targets a different setup — a violent, ATR-scaled DOWN shock, which
liquidation-cascade mechanics argue should OVERSHOOT and then revert, the
opposite dynamic. Both can be independently true or false; testing them
separately, rather than assuming one implies the other, is the point.

Mechanics deliberately mirror the equity family (`oversold_reversion_v1`) via
the SAME shared helpers (`signals/helpers.py`) rather than reinventing the
math — this is a test of whether the underlying "vol-scaled shock premium"
thesis transfers to crypto's different liquidation mechanism, not a test of
a different rule shape. The k grid (2.0, 2.5) and SMA100 trend filter are
unchanged from the equity version by design, so a pass or fail here isolates
the asset-class/mechanism question rather than also varying the rule.

Falsified if: vol-scaled oversold entries above SMA100 show no net-of-cost
reversion within 7 sessions, or losses concentrate so heavily in cascade
continuation (the liquidation keeps cascading past this entry too) that the
distribution is untradeable at the crypto doctrine's gap-stress multiple
(GAP_STRESS_MULT["crypto"] = 2.0).

Exit override (within the doctrine's 5-20 session band, per WI-064's shared
exit-doctrine contract): reversion is fast or wrong — a 7-session time stop
(7 calendar days for a 24/7 market, tighter than equity's ~9.8 calendar-day
equivalent — appropriate for crypto's faster cycle), smaller profit-take
target (+1.5R half), doctrine-default trail.

STOP_ATR_MULT IS EXPLICIT (WI-136's landmine, not re-earned here): the
EntryProposal dataclass default is STOP_ATR_MULT["equity"] (2.0), and
`simulate.py` reads `proposal.stop_atr_mult` directly with no asset_class
correction. Every proposal here sets it to STOP_ATR_MULT["crypto"] (2.5)
explicitly; a dedicated regression test locks this in, same as
`crypto_breakout_v1`.
"""

from __future__ import annotations

from vega.backtest.market_view import MarketView
from vega.backtest.signals import EntryProposal
from vega.common.doctrine import STOP_ATR_MULT
from vega.signals.helpers import adjusted_atr14, sma, three_session_change

LOOKBACK = 115
SMA_WINDOW = 100
TIME_STOP_SESSIONS = 7
PROFIT_TAKE_HALF_AT_R = 1.5


class CryptoOversoldReversionSignal:
    family = "crypto_oversold_reversion_v1"
    version = "1.0"
    promotable = True

    def __init__(self, k: float) -> None:
        """k: ATR14 multiple defining the shock threshold (grid: 2.0, 2.5)."""
        self.k = k
        self.params = {"k": k}  # recorded on every RunRecord (WI-066 attribution lesson)

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
            atr = adjusted_atr14(bars, symbol, view.as_of)  # SAME space as change3
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
                        f"the {SMA_WINDOW}-session SMA — leveraged-liquidation shock, "
                        "reversion setup"
                    ),
                    confidence=0.5,
                    invalidation=f"close falls below the {SMA_WINDOW}-session SMA",
                    time_stop_days=TIME_STOP_SESSIONS,
                    profit_take_half_at_r=PROFIT_TAKE_HALF_AT_R,
                    stop_atr_mult=STOP_ATR_MULT["crypto"],
                )
            )
        return proposals
