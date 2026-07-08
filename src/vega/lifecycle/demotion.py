"""Auto-demotion: live performance falls below the signal's backtest confidence
band (STRATEGY.md §6, Pillar 2).

Band = [worst, best] dev-fold Sharpe from the run that justified the family's
promotion — already recorded in the registry, so no new tunable constant is
invented. Evaluated only once >=30 live resolved trades exist (statistical
honesty, same MIN_TRADES_FOR_VERDICT threshold WI-063 uses for backtests —
below that, live Sharpe is noise, not evidence).

Reuses backtest.metrics.compute_fold_metrics for the live Sharpe calculation
(via a thin adapter to TradeRecord) so live and backtested Sharpe are computed
by the exact same formula — comparing them any other way risks a units bug of
the kind the WI-063/WI-064 reviews found repeatedly.

SCOPE NOTE: the live ledger does not yet record exit fills (WI-061 only
tracks entries; exit monitoring is WI-067's job) — there is no real data
source for `live_trades` today. This module is fully wired and tested against
synthetic trade data so it activates the moment WI-067 lands; until then any
real caller naturally sees zero live trades and reports insufficient_sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vega.backtest.engine import MIN_TRADES_FOR_VERDICT
from vega.backtest.metrics import compute_fold_metrics
from vega.backtest.simulate import TradeRecord

DEFAULT_STARTING_CAPITAL = 100_000.0  # stated default until a real capital source exists


@dataclass(frozen=True)
class LiveTrade:
    symbol: str
    asset_class: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    qty: float
    stop_price: float


@dataclass(frozen=True)
class DemotionVerdict:
    should_demote: bool
    reason: str
    live_sharpe: float | None
    band: tuple[float, float] | None
    n_trades: int


def confidence_band(justifying_run: dict[str, Any]) -> tuple[float, float] | None:
    """[worst, best] dev-fold Sharpe from the run — holdout fold excluded (it's
    a single evaluation, not a distribution the band should be built from)."""
    sharpes = [
        f["sharpe"]
        for f in justifying_run["fold_metrics"]
        if f.get("sharpe") is not None and not f.get("is_holdout")
    ]
    if not sharpes:
        return None
    return (min(sharpes), max(sharpes))


def _to_trade_record(t: LiveTrade) -> TradeRecord:
    initial_r = abs(t.entry_price - t.stop_price) or 1e-9
    return TradeRecord(
        symbol=t.symbol,
        asset_class=t.asset_class,
        signal_family="live",
        signal_version="live",
        entry_date=t.entry_date,
        entry_price=t.entry_price,
        initial_qty=t.qty,
        stop_price=t.stop_price,
        initial_r=initial_r,
        thesis="",
        confidence=0.0,
        invalidation="",
        exits=({"date": t.exit_date, "qty": t.qty, "price": t.exit_price, "reason": "live_exit"},),
        realized_pnl=round((t.exit_price - t.entry_price) * t.qty, 6),
        r_multiple=round((t.exit_price - t.entry_price) / initial_r, 4),
        unresolved_at_end=False,
    )


def check_auto_demotion(
    live_trades: list[LiveTrade],
    justifying_run: dict[str, Any],
    min_trades: int = MIN_TRADES_FOR_VERDICT,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
) -> DemotionVerdict:
    if len(live_trades) < min_trades:
        return DemotionVerdict(
            False,
            f"insufficient_sample: {len(live_trades)} < {min_trades}",
            None,
            None,
            len(live_trades),
        )
    band = confidence_band(justifying_run)
    if band is None:
        return DemotionVerdict(
            False,
            "no backtest fold Sharpe on record to compare against",
            None,
            None,
            len(live_trades),
        )
    dates = sorted({t.entry_date for t in live_trades} | {t.exit_date for t in live_trades})
    trade_records = [_to_trade_record(t) for t in live_trades]
    asset_class = live_trades[0].asset_class
    fm = compute_fold_metrics(trade_records, dates, starting_capital, asset_class)
    if fm.sharpe is None:
        return DemotionVerdict(
            False,
            "live Sharpe undefined (flat or single-session P&L)",
            None,
            band,
            len(live_trades),
        )
    if fm.sharpe < band[0]:
        return DemotionVerdict(
            True,
            f"live Sharpe {fm.sharpe} below band floor {band[0]}",
            fm.sharpe,
            band,
            len(live_trades),
        )
    return DemotionVerdict(False, "within band", fm.sharpe, band, len(live_trades))
