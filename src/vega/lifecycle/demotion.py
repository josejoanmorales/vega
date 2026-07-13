"""Auto-demotion: live performance falls below the signal's backtest confidence
band (STRATEGY.md §6, Pillar 2).

Band = [worst, best] dev-fold Sharpe from the run that justified the family's
promotion — already recorded in the registry, so no new tunable constant is
invented. Evaluated only once >=30 live resolved trades exist (statistical
honesty — below that, live Sharpe is noise, not evidence).

Live Sharpe comes from vega.backtest.live_metrics.live_sharpe — the ONE
backtest-owned service guaranteeing the same formula AND the same session-grid
sampling as the band it's compared against (both were review findings: the
previous version reached into backtest internals directly and sampled only
trade-event days, which inflated live Sharpe and made demotion under-fire).

SCOPE NOTE: the live ledger does not yet record exit fills (WI-061 tracks
entries; exit monitoring is WI-067's job) — no real data source for
`live_trades` exists today. Fully wired and tested against synthetic data so
it activates the moment WI-067 lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vega.backtest.live_metrics import (
    DEFAULT_STARTING_CAPITAL,
    MIN_TRADES_FOR_VERDICT,
    LiveTrade,
    live_sharpe,
)

__all__ = ["LiveTrade", "DemotionVerdict", "confidence_band", "check_auto_demotion"]


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


def check_auto_demotion(
    live_trades: list[LiveTrade],
    justifying_run: dict[str, Any],
    session_dates: list[str],
    min_trades: int = MIN_TRADES_FOR_VERDICT,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
) -> DemotionVerdict:
    """`session_dates` = the FULL trading calendar covering the live window
    (from the clean store) — never just the trade-event days."""
    if len(live_trades) < min_trades:
        return DemotionVerdict(
            False,
            f"insufficient_sample: need >= {min_trades}",
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
    observed = live_sharpe(live_trades, session_dates, starting_capital)
    if observed is None:
        return DemotionVerdict(
            False,
            "live Sharpe undefined (flat or single-session P&L)",
            None,
            band,
            len(live_trades),
        )
    if observed < band[0]:
        return DemotionVerdict(
            True,
            f"live Sharpe {observed} below band floor {band[0]}",
            observed,
            band,
            len(live_trades),
        )
    return DemotionVerdict(False, "within band", observed, band, len(live_trades))
