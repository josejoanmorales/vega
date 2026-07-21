"""Bridges closed live round-trips (ledger exit fills, WI-087) into
`backtest.live_metrics.LiveTrade` rows and drives `LifecycleRegistry.demote()`
when `check_auto_demotion` says so — the wiring `demotion.py` was built and
tested against synthetic data for, awaiting real exit fills.

One `LiveTrade` row per exit LOT (a partial-take followed later by a
time-stop exit produces two rows for one position) — `LiveTrade` is
single-exit by design, matching `TradeRecord`'s per-trade shape.

Calendar contract (WI-087 review #5): the session calendar here must span the
FULL live history — every session from the oldest position's entry through
today — never the ~220-day signal-scan frame. A windowed calendar silently
clamps old entries to its floor and drops pre-floor exits from PnL, corrupting
the live Sharpe exactly when the 30-trade demotion gate opens. Callers load it
via `full_session_calendar()` (the whole store) or supply an equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vega.backtest.live_metrics import LiveTrade
from vega.backtest.registry import BacktestRegistry
from vega.common import db
from vega.data import snapshot
from vega.execution.exits import entry_session_for
from vega.ledger.store import LedgerStore
from vega.lifecycle.demotion import DemotionVerdict, check_auto_demotion
from vega.lifecycle.lifecycle import LifecycleRegistry, is_eligible_state


def full_session_calendar(root: Path = snapshot.DATA_ROOT) -> list[str]:
    """Every yfinance session in the ENTIRE store, sorted — the live track
    record's time axis. Deliberately unwindowed: it must cover the oldest
    round trip forever."""
    with db.connect(root) as con:
        rows = con.execute(
            "SELECT DISTINCT date FROM bars WHERE source = 'yfinance' ORDER BY date"
        ).fetchall()
    return [str(r[0]) for r in rows]


def _family_of(rec: dict[str, object]) -> str:
    attribution = rec.get("signal_attribution") or [""]
    first = str(attribution[0]) if attribution else ""  # type: ignore[index]
    return first.split(":")[0] if first else "unknown"


def closed_round_trips(ledger: LedgerStore, calendar: list[str]) -> dict[str, list[LiveTrade]]:
    """family -> every realized exit lot, across all positions. `entry_date`/
    `exit_date` are STORE SESSIONS (never wall-clock timestamps) — the same
    grid `live_sharpe` samples against, so live and backtest Sharpe are
    computed over comparable calendars. Entry sessions come from the shared
    `entry_session_for` (chain-origin `as_of`, with the legacy first-buy-fill
    fallback — WI-087 review #5: skipping `as_of=None` recs silently excluded
    the two real pre-`as_of` production positions from the track record
    forever)."""
    latest_session = calendar[-1] if calendar else ""
    trades_by_family: dict[str, list[LiveTrade]] = {}
    for rec, fills in ledger.latest_with_all_fills():
        if rec["direction"] != "long":
            continue
        buy_fills = [f for f in fills if f.get("side", "buy") == "buy"]
        if not buy_fills or buy_fills[-1].get("price") is None:
            continue
        entry_price = float(buy_fills[-1]["price"])
        entry_session = entry_session_for(rec, buy_fills, calendar, latest_session)
        if entry_session is None:
            continue
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


def evaluate_demotions(
    ledger: LedgerStore,
    calendar: list[str],
    lifecycle: LifecycleRegistry,
    backtest_registry: BacktestRegistry,
    as_of: str,
) -> list[DemotionOutcome]:
    """PURE: every paper-live+ family's real live track record against its
    justifying backtest band, per asset-class sleeve — verdicts only, no
    lifecycle write (WI-089: the dashboard needs to SHOW signal health on a
    page refresh without ever demoting anything; a page load is not a
    governance act). `calendar` must be the FULL live-history session grid
    (see module docstring). `check_and_apply_demotions` is the only caller
    that may act on `verdict.should_demote`."""
    trades_by_family = closed_round_trips(ledger, calendar)
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

        for sleeve, sleeve_trades in by_sleeve.items():
            verdict = check_auto_demotion(sleeve_trades, run, session_dates)
            outcomes.append(DemotionOutcome(family=family, asset_class=sleeve, verdict=verdict))

    return outcomes


def check_and_apply_demotions(
    ledger: LedgerStore,
    calendar: list[str],
    lifecycle: LifecycleRegistry,
    backtest_registry: BacktestRegistry,
    as_of: str,
    actor: str = "agent:exit-monitor",
) -> list[DemotionOutcome]:
    """Evaluate (via `evaluate_demotions`) and demote (agent-legal,
    `automatic=True` — the lifecycle contract requires only PROMOTION to be
    human-gated) the moment the evidence says so. A family trading more than
    one asset-class sleeve is demoted at most once per run even if multiple
    sleeves qualify (a second `demote()` call on an already-demoted family
    would raise — state changed under it)."""
    outcomes = evaluate_demotions(ledger, calendar, lifecycle, backtest_registry, as_of)

    demote_reasons_by_family: dict[str, list[str]] = {}
    for o in outcomes:
        if o.verdict.should_demote:
            demote_reasons_by_family.setdefault(o.family, []).append(
                f"{o.asset_class or 'n/a'}: {o.verdict.reason}"
            )
    for family, reasons in demote_reasons_by_family.items():
        lifecycle.demote(family, actor=actor, reason="; ".join(reasons), automatic=True)

    return outcomes
