"""Read-only dashboard data (WI-089): positions, signal health, failures, and
the symbol inspector. Every view reuses the EXACT production code path —
`exits.reconstruct_positions`, `briefing.calls.eligible_families`,
`risk.gates.check_all_gates` — never a parallel read-model, which is exactly
the two-reconstructions drift the WI-067/087 reviews kept killing. Nothing in
this module ever writes to the ledger or the lifecycle registry.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from vega.backtest.market_view import MarketView
from vega.backtest.registry import BacktestRegistry
from vega.briefing.calls import FAMILY_SIGNALS, eligible_families, load_signal_frame
from vega.briefing.engine import assemble
from vega.common.doctrine import DEFAULT_TIME_STOP_SESSIONS
from vega.data import snapshot
from vega.data.universe import load_universe
from vega.execution.executor import read_failures
from vega.execution.exits import reconstruct_positions
from vega.ledger.store import LedgerStore
from vega.lifecycle.lifecycle import LifecycleRegistry
from vega.lifecycle.live_trades import evaluate_demotions, full_session_calendar
from vega.risk.gates import EarningsFact, check_all_gates


class SymbolNotInUniverse(ValueError):
    """The requested symbol isn't in the committed tradable universe."""


def _max_session(root: Path = snapshot.DATA_ROOT) -> str:
    con = duckdb.connect(str(root / "vega.duckdb"), read_only=True)
    try:
        row = con.execute("SELECT max(date) FROM bars WHERE source = 'yfinance'").fetchone()
    finally:
        con.close()
    return str(row[0]) if row and row[0] is not None else ""


def positions(ledger: LedgerStore | None = None) -> list[dict[str, Any]]:
    """Open positions + same-session pending calls, via the SAME reconstruction
    the exit monitor uses — no parallel logic (WI-089-c1)."""
    ledger = ledger or LedgerStore()
    as_of = _max_session()
    if not as_of:
        return []
    frame = load_signal_frame(as_of)
    rows = []
    for pos in reconstruct_positions(ledger, frame, as_of):
        rows.append(
            {
                "symbol": pos.symbol,
                "asset_class": pos.asset_class,
                "qty": pos.remaining_qty,
                "in_flight_sell_qty": pos.in_flight_sell_qty,
                "entry_price": pos.entry_price,
                "current_stop_price": pos.current_stop_price,
                "sessions_held": pos.sessions_held,
                "time_stop_sessions": pos.time_stop_sessions,
                "sessions_until_time_stop": (
                    None if pos.is_pending else pos.time_stop_sessions - pos.sessions_held
                ),
                "is_pending": pos.is_pending,
                "entry_confirmed": pos.entry_confirmed,
            }
        )
    return rows


def signal_health(ledger: LedgerStore | None = None) -> list[dict[str, Any]]:
    """Per family/sleeve lifecycle state + live-trade count + band + the
    verdict evaluate_demotions WOULD reach — via the PURE evaluator, so a page
    load can never demote anything (WI-089-c2)."""
    ledger = ledger or LedgerStore()
    as_of = _max_session()
    if not as_of:
        return []
    lifecycle = LifecycleRegistry()
    registry = BacktestRegistry()
    outcomes = evaluate_demotions(ledger, full_session_calendar(), lifecycle, registry, as_of)
    return [
        {
            "family": o.family,
            "asset_class": o.asset_class,
            "state": lifecycle.current_state(o.family),
            "n_trades": o.verdict.n_trades,
            "live_sharpe": o.verdict.live_sharpe,
            "band": list(o.verdict.band) if o.verdict.band is not None else None,
            "would_demote": o.verdict.should_demote,
            "reason": o.verdict.reason,
        }
        for o in outcomes
    ]


def failures() -> list[dict[str, Any]]:
    """The exec-failures log, newest first (WI-089-c2)."""
    return list(reversed(read_failures()))


def inspect_symbol(symbol: str, ledger: LedgerStore | None = None) -> dict[str, Any]:
    """Signal status, position status, and gate status for one ticker —
    strictly read-only (WI-089-c4). Raises `SymbolNotInUniverse` for an
    unknown ticker (the caller 404s); `briefing.calls.CallsError` still
    propagates if eligibility bookkeeping is broken — the same loud-failure
    contract the daily run has, not silently swallowed here."""
    ledger = ledger or LedgerStore()
    entries = load_universe()
    entry = next((e for e in entries if e.symbol == symbol), None)
    if entry is None:
        raise SymbolNotInUniverse(symbol)

    briefing_data = assemble()  # same regime-assembly path the real briefing uses
    on_date = date.fromisoformat(briefing_data.as_of)

    as_of = _max_session()
    frame = load_signal_frame(as_of) if as_of else None

    position = None
    if frame is not None:
        for pos in reconstruct_positions(ledger, frame, as_of):
            if pos.symbol == symbol:
                position = {
                    "held": not pos.is_pending,
                    "pending": pos.is_pending,
                    "qty": pos.remaining_qty,
                    "in_flight_sell_qty": pos.in_flight_sell_qty,
                    "sessions_held": pos.sessions_held,
                    "time_stop_sessions": pos.time_stop_sessions,
                }
                break

    signals: list[dict[str, Any]] = []
    lifecycle = LifecycleRegistry()
    registry = BacktestRegistry()
    eligible = eligible_families(lifecycle, registry)  # CallsError propagates loudly, unswallowed
    if frame is not None and eligible:
        view = MarketView(frame, as_of)
        for fam in eligible:
            signal = FAMILY_SIGNALS[fam.family](**fam.justifying_params)
            proposals = signal.scan(view, [symbol])
            signals.append(
                {
                    "family": fam.family,
                    "state": fam.state,
                    "fires_today": bool(proposals),
                    "thesis": proposals[0].thesis if proposals else None,
                }
            )

    # Stated assumption: gates are checked against the doctrine default time
    # stop, not a specific signal's override — the inspector answers "is this
    # symbol tradable right now", not "would THIS exact call be accepted".
    earnings = EarningsFact.lookup(symbol, entry.asset_class)
    rejection = check_all_gates(
        symbol, on_date, DEFAULT_TIME_STOP_SESSIONS, briefing_data.regime, earnings
    )

    return {
        "symbol": symbol,
        "asset_class": entry.asset_class,
        "as_of": briefing_data.as_of,
        "position": position,
        "signals": signals,
        "gates": {
            "earnings_status": earnings.status,
            "blocked": rejection is not None,
            "reason": rejection.reason if rejection is not None else None,
            "detail": rejection.detail if rejection is not None else None,
        },
        "read_only_notice": "inspection is read-only — nothing was written to the ledger",
    }
