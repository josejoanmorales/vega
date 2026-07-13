"""Live-trade Sharpe, computed EXACTLY as backtest fold Sharpe is computed.

This module exists so governance (vega.lifecycle) never touches simulation
internals (TradeRecord, compute_fold_metrics) directly — backtest/ owns its
representations and exposes this one narrow service (a review finding: the
previous adapter lived in lifecycle/ and depended on three backtest modules).

Comparability rules (both were review findings):
- **Same sampling grid**: the caller must supply the full trading-session
  calendar covering the live window — NOT just trade-event days. Backtest fold
  Sharpe includes every flat session; dropping them inflates live Sharpe and
  makes auto-demotion under-fire. Session dates come from the clean store.
- **One asset class per computation**: annualization (√252 vs √365) differs
  by sleeve; a mixed list raises rather than silently annualizing a blended
  series by whichever trade happened to come first. A family trading both
  sleeves is evaluated per sleeve by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

# Re-exported for governance callers so lifecycle/ needs no engine import —
# and defined ONCE (in engine.py), never re-declared (review finding).
from vega.backtest.engine import DEFAULT_STARTING_CAPITAL, MIN_TRADES_FOR_VERDICT
from vega.backtest.metrics import compute_fold_metrics
from vega.backtest.simulate import TradeRecord

__all__ = [
    "DEFAULT_STARTING_CAPITAL",
    "MIN_TRADES_FOR_VERDICT",
    "LiveTrade",
    "live_sharpe",
]


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
        # compute_fold_metrics consumes ONLY exits/entry_price/entry_date/
        # unresolved_at_end; the two fields below are populated for record
        # completeness and must never be read as backtest-comparable output.
        realized_pnl=round((t.exit_price - t.entry_price) * t.qty, 6),
        r_multiple=round((t.exit_price - t.entry_price) / initial_r, 4),
        unresolved_at_end=False,
    )


def live_sharpe(
    live_trades: list[LiveTrade],
    session_dates: list[str],
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
) -> float | None:
    """Sharpe of realized live trades over the FULL session grid.

    `session_dates` must be the complete trading calendar spanning the live
    window (from the clean store) — every flat session counts, exactly as it
    does inside a backtest fold. Raises on an empty grid, a grid that doesn't
    cover the trades, or mixed asset classes.
    """
    if not live_trades:
        return None
    classes = {t.asset_class for t in live_trades}
    if len(classes) > 1:
        raise ValueError(
            f"live_sharpe requires a single asset class per computation (got {sorted(classes)}); "
            "evaluate a multi-sleeve family per sleeve"
        )
    if not session_dates:
        raise ValueError("session_dates must be the full trading calendar of the live window")
    first_event = min(t.entry_date for t in live_trades)
    last_event = max(t.exit_date for t in live_trades)
    grid = sorted(session_dates)
    if grid[0] > first_event or grid[-1] < last_event:
        raise ValueError(
            f"session_dates [{grid[0]}..{grid[-1]}] does not cover the live trade window "
            f"[{first_event}..{last_event}]"
        )
    records = [_to_trade_record(t) for t in live_trades]
    metrics = compute_fold_metrics(records, grid, starting_capital, next(iter(classes)))
    return metrics.sharpe
