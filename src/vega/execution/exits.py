"""Exit monitor (WI-087): mirrors backtest/simulate.py's exit mechanics
against the live ledger, so live positions close the same way the backtest
that justified their promotion said they would.

Timing (necessarily different from the backtest, by acceptance — WI-067's
enrichment recorded this as a stated assumption): simulate.py evaluates stop/
profit triggers INTRADAY on the same session bar (full OHLC known, since it's
history). The live monitor runs pre-market and can only see the last
COMPLETED session's bar — so a triggered exit here submits a market sell that
fills at the NEXT session's open, one session later than simulate's intraday
fill. Time stops already have this lag in simulate.py itself (they queue and
fill at the next open), so they match exactly; only stop/profit fills carry
the documented one-session divergence.

State is never persisted separately from the ledger: every position's current
stop (trailed or not), remaining qty, and sessions-held are DERIVED fresh each
run from (entry fill, exit fills, exit_params, the clean store's session
calendar) — a pure function, safe to re-run any number of times.

STATED ASSUMPTION — R basis (WI-087 review #10): live R = actual fill price
minus the PUBLISHED ledger stop, so the profit target moves with the overnight
entry gap relative to the backtest (simulate.py re-derives its stop FROM the
fill, so its R is always stop_mult*ATR). This is deliberate: the published
stop is the binding contract the briefing printed and the stop the monitor
enforces — exits must be self-consistent with the position's REAL risk as
executed, even where that diverges from the simulated fill model. The
structural fix (one shared trigger function for simulate.py and this module)
is parked on the WI-084 cleanup item.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vega.common.doctrine import (
    DEFAULT_TIME_STOP_SESSIONS,
    PROFIT_TAKE_HALF_AT_R,
    PROFIT_TRAIL_ATR_MULT,
)
from vega.execution.executor import (
    FAILURES_PATH,
    TERMINAL_UNFILLED_STATUSES,
    TradingBackend,
    record_failure,
)
from vega.ledger.store import LedgerStore
from vega.risk.heat import OpenPositionHeat

# simulate.py's own exit-reason vocabulary — reused verbatim so live and
# backtest round-trips are directly comparable/groupable.
EXIT_REASONS = ("gap_stop", "stop", "profit_partial", "time_stop")


@dataclass(frozen=True)
class OpenPosition:
    """Everything with real or imminent exposure: a same-session pending call
    (`is_pending=True` — submitted, not yet filled, no trail possible) or a
    filled-and-still-open position (`is_pending=False` — trailed stop, qty net
    of any partial exits). The ONE reconstruction both heat accounting
    (`briefing/calls.py`) and exit evaluation (this module) read — two
    parallel reconstructions were exactly the kind of drift WI-067's review
    flagged."""

    ref_id: str
    symbol: str
    asset_class: str
    entry_price: float
    remaining_qty: float  # net of PRICED sells only — in-flight sells still count as held
    original_stop_price: float
    current_stop_price: float  # == original unless a partial has trailed it
    entry_session: str | None  # None only when is_pending
    sessions_held: int  # 0 when is_pending
    took_partial: bool
    atr_at_entry: float
    time_stop_sessions: int
    profit_take_half_at_r: float
    profit_trail_atr_mult: float
    invalidation: str
    thesis: str
    confidence: float
    signal_attribution: tuple[str, ...]
    spy_correlation: float | None
    is_pending: bool
    # WI-087 review findings #2/#4:
    # entry_confirmed — the buy has a real price. An accepted-but-unpriced buy
    # is presumed filled for HEAT (overstating = conservative) but must never
    # arm the SELL path (selling shares that may never have been bought is the
    # unsafe direction).
    entry_confirmed: bool = True
    # in_flight_sell_qty — submitted-but-unpriced sell qty. It does NOT reduce
    # remaining (the shares are ours until a priced fill confirms otherwise —
    # heat stays conservative) but exit decisions must not re-sell it.
    in_flight_sell_qty: float = 0.0


@dataclass(frozen=True)
class ExitDecision:
    ref_id: str
    symbol: str
    asset_class: str
    qty: float
    reason: str  # one of EXIT_REASONS
    detail: str


def to_heat(pos: OpenPosition) -> OpenPositionHeat:
    from vega.risk.clusters import contaminates_equity_beta

    return OpenPositionHeat(
        symbol=pos.symbol,
        asset_class=pos.asset_class,
        qty=pos.remaining_qty,
        entry_price=pos.entry_price,
        current_stop_price=pos.current_stop_price,
        contaminates_equity_beta=contaminates_equity_beta(pos.spy_correlation),
    )


def trading_calendar(frame: pd.DataFrame) -> list[str]:
    """Every distinct session in `frame`, sorted — the pooled equity/ETF
    calendar used for session counting (time stops) and the trail's
    high-water-close window. A single symbol's own thin/gappy history is
    never used for this — one delisted-adjacent bar must not stall or shift
    another symbol's time stop."""
    return sorted({str(d) for d in frame["date"].unique()})


def entry_session_for(
    rec: dict[str, Any],
    buy_fills: list[dict[str, Any]],
    calendar: list[str],
    as_of: str,
) -> str | None:
    """The session a position's clock starts on: the first session AFTER the
    chain-ORIGIN decision (`origin_as_of` — the surviving rec's own `as_of`
    would let a supersede correction appended days later reset the time-stop
    clock, WI-087 review #3). Legacy records predating the `as_of` field
    (WI-067's first live smoke) fall back to the ORIGINAL (first) buy fill's
    timestamp date — never a later reconciliation record, whose wall-clock
    stamp can land past the calendar's range and silently exclude the
    position from every check. Returns None when the entry session hasn't
    happened yet (or can't be derived). The ONE definition — exit
    reconstruction and live-trade round-trips both use it (WI-087 review #5)."""
    origin_as_of = rec.get("origin_as_of") or rec.get("as_of")
    if origin_as_of is not None:
        candidates = [d for d in calendar if d > origin_as_of]
    else:
        buy_date = str(buy_fills[0].get("at", ""))[:10] if buy_fills else ""
        candidates = [d for d in calendar if d >= buy_date] if buy_date else []
    if candidates and candidates[0] <= as_of:
        return candidates[0]
    return None


def _reconstruct_one(
    rec: dict[str, Any],
    fills: tuple[dict[str, Any], ...],
    frame: pd.DataFrame,
    calendar: list[str],
    as_of: str,
) -> OpenPosition | None:
    exit_params = rec.get("exit_params") or {}
    time_stop_sessions = int(exit_params.get("time_stop_sessions", DEFAULT_TIME_STOP_SESSIONS))
    profit_take_half_at_r = float(exit_params.get("take_half_at_r", PROFIT_TAKE_HALF_AT_R))
    profit_trail_atr_mult = float(exit_params.get("trail_atr_mult", PROFIT_TRAIL_ATR_MULT))
    atr_at_entry = float(exit_params.get("atr_at_proposal", 0.0))
    spy_correlation = exit_params.get("spy_correlation")
    original_stop = float(rec["stop_price"])
    common: dict[str, Any] = {
        "ref_id": rec["id"],
        "symbol": rec["symbol"],
        "asset_class": rec["asset_class"],
        "original_stop_price": original_stop,
        "time_stop_sessions": time_stop_sessions,
        "profit_take_half_at_r": profit_take_half_at_r,
        "profit_trail_atr_mult": profit_trail_atr_mult,
        "atr_at_entry": atr_at_entry,
        "invalidation": rec["invalidation"],
        "thesis": rec["thesis"],
        "confidence": rec["confidence"],
        "signal_attribution": tuple(rec.get("signal_attribution") or ()),
        "spy_correlation": spy_correlation,
    }

    buy_fills = [f for f in fills if f.get("side", "buy") == "buy"]
    if not buy_fills:
        # Same-session pending only — a stale pending call is the executor's
        # problem to expire, never treated as exposure here.
        if rec.get("as_of") != as_of:
            return None
        qty = rec.get("qty")
        if not qty:
            return None
        return OpenPosition(
            **common,
            entry_price=float(rec["entry_ref_price"]),
            remaining_qty=float(qty),
            current_stop_price=original_stop,
            entry_session=None,
            sessions_held=0,
            took_partial=False,
            is_pending=True,
            entry_confirmed=False,
        )

    buy = buy_fills[-1]
    if buy.get("price") is None and buy.get("status") in TERMINAL_UNFILLED_STATUSES:
        return None  # the entry itself never happened
    entry_confirmed = buy.get("price") is not None
    entry_price = float(buy["price"]) if entry_confirmed else float(rec["entry_ref_price"])
    entry_qty = float(buy.get("qty") or rec.get("qty") or 0.0)
    if entry_qty <= 0:
        return None

    live_sells = [
        f
        for f in fills
        if f.get("side") == "sell" and f.get("status") not in TERMINAL_UNFILLED_STATUSES
    ]
    # Only PRICED sells reduce the position — an in-flight (accepted-but-
    # unpriced) sell has not confirmed a fill, so its shares are still ours
    # for heat purposes; it is tracked separately so exit decisions never
    # re-sell qty already covered by a working order (WI-087 review #2).
    sold_qty = sum(float(f["qty"]) for f in live_sells if f.get("price") is not None)
    in_flight = sum(float(f["qty"]) for f in live_sells if f.get("price") is None)
    remaining = round(entry_qty - sold_qty, 6)
    if remaining <= 1e-9:
        return None  # fully closed

    entry_session = entry_session_for(rec, buy_fills, calendar, as_of)
    sessions_held = 0
    current_stop = original_stop
    took_partial = False
    if entry_session is not None:
        sessions_held = sum(1 for d in calendar if entry_session < d <= as_of)
        partial_fills = [f for f in live_sells if f.get("reason") == "profit_partial"]
        took_partial = bool(partial_fills)
        if took_partial and atr_at_entry > 0:
            # first session-tagged partial anchors the trail window; per-order
            # resolution upstream merges the tag onto reconciled records, and
            # this scan is the defense in depth if it's ever absent anyway
            partial_session = next(
                (f.get("session") for f in partial_fills if f.get("session")), None
            )
            if partial_session is not None:
                window = frame[
                    (frame["symbol"] == rec["symbol"])
                    & (frame["date"] >= partial_session)
                    & (frame["date"] <= as_of)
                ]
                if not window.empty:
                    high_water = float(window["close"].max())
                    trail = high_water - profit_trail_atr_mult * atr_at_entry
                    current_stop = max(current_stop, trail)

    return OpenPosition(
        **common,
        entry_price=entry_price,
        remaining_qty=remaining,
        current_stop_price=current_stop,
        entry_session=entry_session,
        sessions_held=sessions_held,
        took_partial=took_partial,
        is_pending=False,
        entry_confirmed=entry_confirmed,
        in_flight_sell_qty=round(in_flight, 6),
    )


def reconstruct_positions(
    ledger: LedgerStore, frame: pd.DataFrame, as_of: str
) -> list[OpenPosition]:
    calendar = trading_calendar(frame)
    positions = []
    for rec, fills in ledger.latest_with_all_fills():
        if rec["direction"] != "long":
            continue
        pos = _reconstruct_one(rec, fills, frame, calendar, as_of)
        if pos is not None:
            positions.append(pos)
    return positions


class ExitMonitorGapError(RuntimeError):
    """A held, confirmed position's symbol has NO bars anywhere in the frame —
    a structural data gap (e.g. a crypto position against the yfinance-only
    equity frame), not a one-day halt. Stops would silently never be
    evaluated; fail loudly instead (WI-087 review #8). A symbol merely
    missing TODAY's bar keeps the halt semantics simulate.py has (management
    skipped, time stop still counts)."""


def evaluate_exits(ledger: LedgerStore, frame: pd.DataFrame, as_of: str) -> list[ExitDecision]:
    """One decision per triggered event, in simulate.py's own priority order
    per position: gap-stop, then stop, then (if room) a profit partial, then
    (on whatever remains) a time stop — a partial and a same-session time
    stop on the remainder can both fire for one position in one run, exactly
    as simulate.py's step 2 (management) followed by step 3 (time-stop
    trigger) allows within a single simulated day.

    Only CONFIRMED entries (a priced buy fill) are ever evaluated — an
    accepted-but-unpriced buy is heat, not sellable inventory. Decisions
    cover at most `remaining - in_flight_sell_qty`: qty already covered by a
    working sell order is never re-sold."""
    decisions: list[ExitDecision] = []
    today = frame[frame["date"] == as_of]
    bars_by_symbol = {str(sym): g.iloc[0] for sym, g in today.groupby("symbol")}
    frame_symbols = set(frame["symbol"].unique())

    for pos in reconstruct_positions(ledger, frame, as_of):
        if pos.is_pending or pos.entry_session is None:
            continue  # nothing has actually been held yet
        if not pos.entry_confirmed:
            continue  # presumed-filled buys reserve heat but must never SELL (review #4)
        if pos.symbol not in frame_symbols:
            raise ExitMonitorGapError(
                f"{pos.symbol} ({pos.asset_class}) holds an open position but has NO bars "
                "in the monitoring frame — its stop/profit/time-stop would silently never "
                "be evaluated. Feed its bars into the frame before monitoring can resume."
            )
        remaining = round(pos.remaining_qty - pos.in_flight_sell_qty, 6)
        if remaining <= 1e-9:
            continue  # everything left is already covered by a working sell order
        row = bars_by_symbol.get(pos.symbol)
        if row is not None:
            if float(row["open"]) <= pos.current_stop_price:
                decisions.append(
                    ExitDecision(
                        pos.ref_id,
                        pos.symbol,
                        pos.asset_class,
                        remaining,
                        "gap_stop",
                        f"open {float(row['open']):.2f} <= stop {pos.current_stop_price:.2f}",
                    )
                )
                continue
            if float(row["low"]) <= pos.current_stop_price:
                decisions.append(
                    ExitDecision(
                        pos.ref_id,
                        pos.symbol,
                        pos.asset_class,
                        remaining,
                        "stop",
                        f"low {float(row['low']):.2f} <= stop {pos.current_stop_price:.2f}",
                    )
                )
                continue
            if not pos.took_partial:
                target = pos.entry_price + pos.profit_take_half_at_r * (
                    pos.entry_price - pos.original_stop_price
                )
                if float(row["high"]) >= target:
                    half = round(remaining / 2, 6)
                    decisions.append(
                        ExitDecision(
                            pos.ref_id,
                            pos.symbol,
                            pos.asset_class,
                            half,
                            "profit_partial",
                            f"high {float(row['high']):.2f} >= target {target:.2f}",
                        )
                    )
                    remaining = round(remaining - half, 6)
        if pos.sessions_held >= pos.time_stop_sessions and remaining > 1e-9:
            decisions.append(
                ExitDecision(
                    pos.ref_id,
                    pos.symbol,
                    pos.asset_class,
                    remaining,
                    "time_stop",
                    f"held {pos.sessions_held} sessions >= {pos.time_stop_sessions}",
                )
            )
    return decisions


def execute_exits(
    ledger: LedgerStore,
    backend: TradingBackend,
    decisions: list[ExitDecision],
    as_of: str,
    failures_path: Path = FAILURES_PATH,
) -> tuple[int, int]:
    """Submit every exit decision as a market sell; append the fill (or
    terminal-cancel record via `reconcile_fills`, unchanged) tagged
    `side="sell"` with the trigger reason and the deciding session — the
    session tag is what lets a later run's trail computation find the correct
    high-water-close window. One bad order never stops the batch."""
    submitted = 0
    failed = 0
    for d in decisions:
        try:
            result = backend.submit_market_sell(d.symbol, d.qty, d.asset_class)
            ledger.append_fill(
                ref_id=d.ref_id,
                order_id=result.order_id,
                qty=result.qty,
                price=result.filled_avg_price,
                status=result.status,
                side="sell",
                reason=d.reason,
                session=as_of,
            )
            submitted += 1
        except Exception as exc:  # noqa: BLE001 — one bad order must not stop the batch
            record_failure(d.ref_id, d.symbol, f"exit ({d.reason}) failed: {exc}", failures_path)
            failed += 1
    return submitted, failed
