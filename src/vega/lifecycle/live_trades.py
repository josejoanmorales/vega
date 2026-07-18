"""Bridges closed live round-trips (ledger exit fills, WI-087) into
`backtest.live_metrics.LiveTrade` rows and drives `LifecycleRegistry.demote()`
when `check_auto_demotion` says so — the wiring `demotion.py` was built and
tested against synthetic data for, awaiting real exit fills.

One `LiveTrade` row per exit LOT (a partial-take followed later by a
time-stop exit produces two rows for one position) — `LiveTrade` is
single-exit by design, matching `TradeRecord`'s per-trade shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vega.backtest.live_metrics import LiveTrade
from vega.backtest.registry import BacktestRegistry
from vega.execution.exits import trading_calendar
from vega.ledger.store import LedgerStore
from vega.lifecycle.demotion import DemotionVerdict, check_auto_demotion
from vega.lifecycle.lifecycle import LifecycleRegistry, is_eligible_state


def _family_of(rec: dict[str, object]) -> str:
    attribution = rec.get("signal_attribution") or [""]
    first = str(attribution[0]) if attribution else ""  # type: ignore[index]
    return first.split(":")[0] if first else "unknown"


def closed_round_trips(ledger: LedgerStore, frame: pd.DataFrame) -> dict[str, list[LiveTrade]]:
    """family -> every realized exit lot, across all positions. `entry_date`/
    `exit_date` are STORE SESSIONS (never wall-clock timestamps) — the same
    grid `live_sharpe` samples against, so live and backtest Sharpe are
    computed over comparable calendars."""
    calendar = trading_calendar(frame)
    trades_by_family: dict[str, list[LiveTrade]] = {}
    for rec, fills in ledger.latest_with_all_fills():
        if rec["direction"] != "long":
            continue
        buy_fills = [f for f in fills if f.get("side", "buy") == "buy"]
        if not buy_fills or buy_fills[-1].get("price") is None:
            continue
        entry_price = float(buy_fills[-1]["price"])
        rec_as_of = rec.get("as_of")
        if rec_as_of is None:
            continue
        later = [d for d in calendar if d > rec_as_of]
        if not later:
            continue
        entry_session = later[0]
        sell_fills = [f for f in fills if f.get("side") == "sell" and f.get("price") is not None]
        if not sell_fills:
            continue
        family = _family_of(rec)
        stop_price = float(rec["stop_price"])
        for f in sell_fills:
            exit_session = f.get("session") or str(f["at"])[:10]
            trades_by_family.setdefault(family, []).append(
                LiveTrade(
                    symbol=rec["symbol"],
                    asset_class=rec["asset_class"],
                    entry_date=entry_session,
                    entry_price=entry_price,
                    exit_date=str(exit_session),
                    exit_price=float(f["price"]),
                    qty=float(f["qty"]),
                    stop_price=stop_price,
                )
            )
    return trades_by_family


@dataclass(frozen=True)
class DemotionOutcome:
    family: str
    asset_class: str  # "" when there were no live trades to sleeve-split
    verdict: DemotionVerdict


def check_and_apply_demotions(
    ledger: LedgerStore,
    frame: pd.DataFrame,
    lifecycle: LifecycleRegistry,
    backtest_registry: BacktestRegistry,
    as_of: str,
    actor: str = "agent:exit-monitor",
) -> list[DemotionOutcome]:
    """Evaluate every paper-live+ family's real live track record against its
    justifying backtest band, and demote (agent-legal, `automatic=True` — the
    lifecycle contract requires only PROMOTION to be human-gated) the moment
    the evidence says so. A family trading more than one asset-class sleeve is
    evaluated per sleeve (`live_sharpe`'s own contract), demoted at most once
    per run even if multiple sleeves qualify (a second `demote()` call on an
    already-demoted family would raise — state changed under it)."""
    trades_by_family = closed_round_trips(ledger, frame)
    calendar = trading_calendar(frame)
    session_dates = [d for d in calendar if d <= as_of]

    outcomes: list[DemotionOutcome] = []
    for family in lifecycle.families():
        if not is_eligible_state(lifecycle.current_state(family)):
            continue
        run_id = lifecycle.justifying_run_id(family)
        if run_id is None:
            continue
        run = next((r for r in backtest_registry.runs(family) if r["run_id"] == run_id), None)
        if run is None:
            continue

        live_trades = trades_by_family.get(family, [])
        by_sleeve: dict[str, list[LiveTrade]] = {}
        for t in live_trades:
            by_sleeve.setdefault(t.asset_class, []).append(t)
        if not by_sleeve:
            by_sleeve = {"": []}  # still surface an insufficient_sample verdict

        demote_reasons = []
        for sleeve, sleeve_trades in by_sleeve.items():
            verdict = check_auto_demotion(sleeve_trades, run, session_dates)
            outcomes.append(DemotionOutcome(family=family, asset_class=sleeve, verdict=verdict))
            if verdict.should_demote:
                demote_reasons.append(f"{sleeve or 'n/a'}: {verdict.reason}")
        if demote_reasons:
            lifecycle.demote(family, actor=actor, reason="; ".join(demote_reasons), automatic=True)

    return outcomes
