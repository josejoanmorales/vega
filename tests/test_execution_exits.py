from pathlib import Path

import pandas as pd
import pytest

from vega.execution.executor import OrderResult
from vega.execution.exits import (
    evaluate_exits,
    execute_exits,
    reconstruct_positions,
    to_heat,
    trading_calendar,
)
from vega.ledger.store import LedgerStore
from vega.ledger.types import Recommendation

# D0..D17 — a plain session calendar; the tests don't care about weekends.
DATES = [f"2026-02-{d:02d}" for d in range(2, 20)]
ENTRY_STOP = 90.0
ENTRY_PRICE = 100.0
ATR_AT_ENTRY = 5.0
TRAIL_MULT = 2.5
TAKE_HALF_AT_R = 1.5
TIME_STOP_SESSIONS = 7


def _bars(rows: list[tuple[str, float, float, float, float]], symbol: str = "AAA") -> pd.DataFrame:
    """rows: (date, open, high, low, close)."""
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "date": d,
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "adj_close": c,
            }
            for d, o, h, low, c in rows
        ]
    )


def _flat_calendar(dates: list[str], close: float = 100.0, symbol: str = "AAA") -> pd.DataFrame:
    return _bars([(d, close, close + 1, close - 1, close) for d in dates], symbol=symbol)


def _rec(**overrides: object) -> Recommendation:
    base: dict[str, object] = {
        "symbol": "AAA",
        "asset_class": "equity",
        "direction": "long",
        "thesis": "fixture",
        "confidence": 0.5,
        "horizon_days": TIME_STOP_SESSIONS,
        "entry_ref_price": ENTRY_PRICE,
        "stop_price": ENTRY_STOP,
        "time_stop_date": "2026-03-01",
        "profit_rule": "half at +1.5R, trail 2.5xATR",
        "invalidation": "fixture",
        "signal_attribution": ("oversold_reversion_v1:1.1",),
        "as_of": DATES[0],
        "exit_params": {
            "atr_at_proposal": ATR_AT_ENTRY,
            "time_stop_sessions": TIME_STOP_SESSIONS,
            "take_half_at_r": TAKE_HALF_AT_R,
            "trail_atr_mult": TRAIL_MULT,
        },
    }
    base.update(overrides)
    return Recommendation(**base)  # type: ignore[arg-type]


# ---- reconstruction ---------------------------------------------------------


def test_pending_same_session_call_has_no_exit_state(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    ledger.append(_rec(as_of=DATES[3], qty=10.0))
    frame = _flat_calendar(DATES[:4])
    positions = reconstruct_positions(ledger, frame, DATES[3])
    assert len(positions) == 1
    pos = positions[0]
    assert pos.is_pending is True
    assert pos.remaining_qty == 10.0
    assert pos.current_stop_price == ENTRY_STOP
    assert pos.entry_session is None


def test_stale_pending_call_excluded(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    ledger.append(_rec(as_of=DATES[0], qty=10.0))  # decided days ago, never filled
    frame = _flat_calendar(DATES[:4])
    assert reconstruct_positions(ledger, frame, DATES[3]) == []


def test_filled_position_basic_reconstruction(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(as_of=DATES[0])
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, 101.0, "filled")
    frame = _flat_calendar(DATES[:4])
    positions = reconstruct_positions(ledger, frame, DATES[3])
    assert len(positions) == 1
    pos = positions[0]
    assert pos.is_pending is False
    assert pos.entry_price == 101.0  # the real fill price, not the ref price
    assert pos.remaining_qty == 10.0
    assert pos.current_stop_price == ENTRY_STOP  # no partial yet
    assert pos.entry_session == DATES[1]
    assert pos.sessions_held == 2  # DATES[2], DATES[3]
    assert pos.took_partial is False


def test_legacy_position_without_as_of_falls_back_to_fill_timestamp(tmp_path: Path) -> None:
    # Real production gap found via live smoke: WI-067's first live smoke ran
    # before the review fix added Recommendation.as_of, so real open positions
    # exist with as_of=None. Without a fallback they'd never get an
    # entry_session and would sit outside every exit check forever.
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(as_of=None)
    ledger.append(rec)
    ledger._log.append(  # append_fill always stamps "now" -- forge a dated fill
        {
            "type": "fill",
            "id": "fill-1",
            "at": f"{DATES[1]}T09:30:00+00:00",
            "ref_id": rec.id,
            "order_id": "ord-1",
            "qty": 10.0,
            "price": ENTRY_PRICE,
            "status": "filled",
            "side": "buy",
            "reason": None,
            "session": None,
        }
    )
    frame = _flat_calendar(DATES[:4])
    positions = reconstruct_positions(ledger, frame, DATES[3])
    assert len(positions) == 1
    pos = positions[0]
    assert pos.entry_session == DATES[1]  # derived from the fill's own date
    assert pos.sessions_held == 2


def test_legacy_fallback_uses_first_buy_fill_not_a_later_reconciliation(tmp_path: Path) -> None:
    # Real bug caught via live smoke: a later reconciliation event (updating
    # price/status once Alpaca resolves) is stamped with WALL-CLOCK time,
    # whenever that run happened to execute -- which can land arbitrarily far
    # past the calendar's last known date. Using buy_fills[-1] for the legacy
    # fallback silently excluded real positions from every check forever.
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(as_of=None)
    ledger.append(rec)
    ledger._log.append(
        {
            "type": "fill",
            "id": "fill-1",
            "at": f"{DATES[1]}T09:30:00+00:00",
            "ref_id": rec.id,
            "order_id": "ord-1",
            "qty": 10.0,
            "price": None,
            "status": "accepted",
            "side": "buy",
            "reason": None,
            "session": None,
        }
    )
    frame = _flat_calendar(DATES[:4])
    far_future = "2099-01-01T00:00:00+00:00"  # reconciliation ran long after the calendar's range
    ledger._log.append(
        {
            "type": "fill",
            "id": "fill-2",
            "at": far_future,
            "ref_id": rec.id,
            "order_id": "ord-1",
            "qty": 10.0,
            "price": ENTRY_PRICE,
            "status": "filled",
            "side": "buy",
            "reason": None,
            "session": None,
        }
    )
    positions = reconstruct_positions(ledger, frame, DATES[3])
    assert len(positions) == 1
    pos = positions[0]
    assert pos.entry_price == ENTRY_PRICE  # latest (priced) fill still wins for price
    assert pos.entry_session == DATES[1]  # but the FIRST fill's date anchors the session


def test_terminally_failed_entry_is_not_a_position(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(as_of=DATES[0])
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, None, "accepted")
    ledger.append_fill(rec.id, "ord-1", 0.0, None, "canceled")
    frame = _flat_calendar(DATES[:4])
    assert reconstruct_positions(ledger, frame, DATES[3]) == []


def test_partial_exit_trails_the_stop(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(as_of=DATES[0])
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, ENTRY_PRICE, "filled")
    ledger.append_fill(
        rec.id,
        "ord-2",
        5.0,
        115.0,
        "filled",
        side="sell",
        reason="profit_partial",
        session=DATES[2],
    )
    # closes from the partial's session through as_of: 110, 112, 108 -> high-water 112
    frame = _bars(
        [
            (DATES[0], 100.0, 101.0, 99.0, 100.0),
            (DATES[1], 100.0, 101.0, 99.0, 100.0),
            (DATES[2], 108.0, 111.0, 107.0, 110.0),
            (DATES[3], 110.0, 113.0, 109.0, 112.0),
            (DATES[4], 106.0, 109.0, 105.0, 108.0),
        ]
    )
    positions = reconstruct_positions(ledger, frame, DATES[4])
    assert len(positions) == 1
    pos = positions[0]
    assert pos.remaining_qty == 5.0
    assert pos.took_partial is True
    # trail = high_water(112) - 2.5*5.0 = 99.5, beats the original stop of 90
    assert pos.current_stop_price == pytest.approx(99.5)
    assert pos.sessions_held == 3  # DATES[2], DATES[3], DATES[4]


def test_fully_closed_position_excluded(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(as_of=DATES[0])
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, ENTRY_PRICE, "filled")
    ledger.append_fill(
        rec.id,
        "ord-2",
        5.0,
        115.0,
        "filled",
        side="sell",
        reason="profit_partial",
        session=DATES[2],
    )
    ledger.append_fill(
        rec.id, "ord-3", 5.0, 85.0, "filled", side="sell", reason="stop", session=DATES[4]
    )
    frame = _flat_calendar(DATES[:5])
    assert reconstruct_positions(ledger, frame, DATES[4]) == []


def test_to_heat_uses_remaining_qty_and_trailed_stop(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(as_of=DATES[0])
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, ENTRY_PRICE, "filled")
    (pos,) = reconstruct_positions(ledger, _flat_calendar(DATES[:4]), DATES[3])
    heat = to_heat(pos)
    assert heat.qty == pos.remaining_qty
    assert heat.current_stop_price == pos.current_stop_price


# ---- trigger evaluation ------------------------------------------------------


def _open_position(
    tmp_path: Path, entry_session_offset: int = 1
) -> tuple[LedgerStore, Recommendation]:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(as_of=DATES[0])
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, ENTRY_PRICE, "filled")
    return ledger, rec


def test_gap_stop_exits_full_remaining_qty(tmp_path: Path) -> None:
    ledger, rec = _open_position(tmp_path)
    frame = _bars(
        [
            (DATES[0], 100.0, 101.0, 99.0, 100.0),
            (DATES[1], 100.0, 101.0, 99.0, 100.0),
            (DATES[2], 100.0, 101.0, 99.0, 100.0),
            (DATES[3], 85.0, 86.0, 84.0, 85.5),  # gaps below stop at the open
        ]
    )
    decisions = evaluate_exits(ledger, frame, DATES[3])
    assert len(decisions) == 1
    d = decisions[0]
    assert d.ref_id == rec.id and d.reason == "gap_stop" and d.qty == 10.0


def test_stop_breach_without_gap(tmp_path: Path) -> None:
    ledger, rec = _open_position(tmp_path)
    frame = _bars(
        [
            (DATES[0], 100.0, 101.0, 99.0, 100.0),
            (DATES[1], 100.0, 101.0, 99.0, 100.0),
            (DATES[2], 100.0, 101.0, 99.0, 100.0),
            (DATES[3], 95.0, 96.0, 88.0, 93.0),  # open above stop, low breaches it
        ]
    )
    decisions = evaluate_exits(ledger, frame, DATES[3])
    assert len(decisions) == 1
    assert decisions[0].reason == "stop" and decisions[0].qty == 10.0


def test_profit_partial_sells_half(tmp_path: Path) -> None:
    ledger, rec = _open_position(tmp_path)
    target = ENTRY_PRICE + TAKE_HALF_AT_R * (ENTRY_PRICE - ENTRY_STOP)  # 115.0
    frame = _bars(
        [
            (DATES[0], 100.0, 101.0, 99.0, 100.0),
            (DATES[1], 100.0, 101.0, 99.0, 100.0),
            (DATES[2], 100.0, 101.0, 99.0, 100.0),
            (DATES[3], 105.0, target + 1.0, 104.0, 116.0),
        ]
    )
    decisions = evaluate_exits(ledger, frame, DATES[3])
    assert len(decisions) == 1
    assert decisions[0].reason == "profit_partial" and decisions[0].qty == 5.0


def test_time_stop_fires_after_enough_sessions(tmp_path: Path) -> None:
    ledger, rec = _open_position(tmp_path)
    # entry_session = DATES[1]; need sessions_held >= 7 -> as_of = DATES[8]
    quiet_rows = [
        (d, 100.0, 105.0, 98.0, 100.0)  # never breaches stop(90) or target(115)
        for d in DATES[:9]
    ]
    frame = _bars(quiet_rows)
    decisions = evaluate_exits(ledger, frame, DATES[8])
    assert len(decisions) == 1
    assert decisions[0].reason == "time_stop" and decisions[0].qty == 10.0


def test_partial_and_time_stop_can_both_fire_same_run(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(
        as_of=DATES[0],
        exit_params={
            "atr_at_proposal": ATR_AT_ENTRY,
            "time_stop_sessions": 2,
            "take_half_at_r": TAKE_HALF_AT_R,
            "trail_atr_mult": TRAIL_MULT,
        },
    )
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, ENTRY_PRICE, "filled")
    # entry_session=DATES[1]; as_of=DATES[3] -> sessions_held=2 (DATES[2],[3]) >= time_stop(2)
    target = ENTRY_PRICE + TAKE_HALF_AT_R * (ENTRY_PRICE - ENTRY_STOP)
    frame = _bars(
        [
            (DATES[0], 100.0, 101.0, 99.0, 100.0),
            (DATES[1], 100.0, 101.0, 99.0, 100.0),
            (DATES[2], 100.0, 101.0, 99.0, 100.0),
            (DATES[3], 105.0, target + 1.0, 104.0, 116.0),  # profit AND time-stop-eligible
        ]
    )
    decisions = evaluate_exits(ledger, frame, DATES[3])
    reasons = {d.reason: d for d in decisions}
    assert set(reasons) == {"profit_partial", "time_stop"}
    assert reasons["profit_partial"].qty == 5.0
    assert reasons["time_stop"].qty == 5.0  # the remainder after the partial


def test_pending_position_never_generates_an_exit(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    ledger.append(_rec(as_of=DATES[3], qty=10.0))  # same-session pending, no fill
    frame = _bars([(d, 50.0, 51.0, 49.0, 50.0) for d in DATES[:4]])  # would gap-stop if open
    assert evaluate_exits(ledger, frame, DATES[3]) == []


# ---- execution ---------------------------------------------------------------


class FakeBackend:
    def __init__(self, fail_symbols: set[str] | None = None) -> None:
        self.fail_symbols = fail_symbols or set()
        self.sells: list[tuple[str, float]] = []

    def submit_market_sell(self, symbol: str, qty: float, asset_class: str) -> OrderResult:
        if symbol in self.fail_symbols:
            raise RuntimeError(f"{symbol} not tradable")
        self.sells.append((symbol, qty))
        return OrderResult("ord-sell-1", symbol, qty, None, "accepted")


def test_execute_exits_appends_sell_fill_tagged_with_reason_and_session(tmp_path: Path) -> None:
    ledger, rec = _open_position(tmp_path)
    from vega.execution.exits import ExitDecision

    decision = ExitDecision(rec.id, "AAA", "equity", 10.0, "stop", "low <= stop")
    submitted, failed = execute_exits(ledger, FakeBackend(), [decision], DATES[3])
    assert (submitted, failed) == (1, 0)
    fill = ledger.fills()[-1]
    assert fill["side"] == "sell" and fill["reason"] == "stop" and fill["session"] == DATES[3]
    assert fill["ref_id"] == rec.id and fill["qty"] == 10.0


def test_execute_exits_failure_is_recorded_and_batch_continues(tmp_path: Path) -> None:
    ledger, rec = _open_position(tmp_path)
    from vega.execution.executor import read_failures
    from vega.execution.exits import ExitDecision

    decision = ExitDecision(rec.id, "AAA", "equity", 10.0, "stop", "low <= stop")
    submitted, failed = execute_exits(
        ledger, FakeBackend({"AAA"}), [decision], DATES[3], failures_path=tmp_path / "f.jsonl"
    )
    assert (submitted, failed) == (0, 1)
    failures = read_failures(tmp_path / "f.jsonl")
    assert len(failures) == 1 and "exit (stop) failed" in failures[0]["error"]


def test_trading_calendar_pools_all_symbols() -> None:
    frame = pd.concat(
        [
            _flat_calendar(["2026-01-01", "2026-01-02"], symbol="AAA"),
            _flat_calendar(["2026-01-02", "2026-01-03"], symbol="BBB"),
        ]
    )
    assert trading_calendar(frame) == ["2026-01-01", "2026-01-02", "2026-01-03"]


# ---- WI-087 review-fix regressions ------------------------------------------


def test_review1_reconcile_preserves_sell_identity(tmp_path: Path) -> None:
    # Review #1: reconciled records must keep side/reason/session — dropping
    # them re-labeled every reconciled SELL as a buy.
    from vega.execution.executor import OrderResult as OR
    from vega.execution.executor import reconcile_fills

    ledger, rec = _open_position(tmp_path)
    ledger.append_fill(
        rec.id,
        "ord-s1",
        5.0,
        None,
        "accepted",
        side="sell",
        reason="profit_partial",
        session=DATES[2],
    )

    class _B:
        def order_status(self, order_id: str) -> OR:
            return OR(order_id, "AAA", 5.0, 116.2, "filled")

    assert reconcile_fills(ledger, _B()) == 1
    resolved = ledger.latest_with_all_fills()[0][1]
    sell = next(f for f in resolved if f.get("side") == "sell")
    assert sell["price"] == 116.2
    assert sell["reason"] == "profit_partial" and sell["session"] == DATES[2]


def test_review9_acceptance_and_reconciliation_are_one_order(tmp_path: Path) -> None:
    # Review #9: same order_id resolves to ONE fill — sold_qty must not
    # double-count an acceptance plus its priced reconciliation.
    ledger, rec = _open_position(tmp_path)
    ledger.append_fill(
        rec.id,
        "ord-s1",
        5.0,
        None,
        "accepted",
        side="sell",
        reason="profit_partial",
        session=DATES[2],
    )
    ledger.append_fill(
        rec.id,
        "ord-s1",
        5.0,
        116.2,
        "filled",
        side="sell",
        reason="profit_partial",
        session=DATES[2],
    )
    (pos,) = reconstruct_positions(ledger, _flat_calendar(DATES[:4]), DATES[3])
    assert pos.remaining_qty == 5.0  # 10 entry - 5 sold ONCE
    assert pos.took_partial is True


def test_review2_inflight_sell_keeps_heat_but_blocks_resell(tmp_path: Path) -> None:
    # Review #2: an accepted-but-unpriced sell keeps its shares in heat
    # (remaining unchanged — conservative) while exit evaluation must not
    # re-sell the covered qty.
    ledger, rec = _open_position(tmp_path)
    ledger.append_fill(
        rec.id,
        "ord-s1",
        10.0,
        None,
        "accepted",
        side="sell",
        reason="time_stop",
        session=DATES[2],
    )
    frame = _bars([(d, 50.0, 51.0, 49.0, 50.0) for d in DATES[:4]])  # deep below stop
    (pos,) = reconstruct_positions(ledger, frame, DATES[3])
    assert pos.remaining_qty == 10.0  # unpriced sell does NOT reduce the position
    assert pos.in_flight_sell_qty == 10.0
    assert evaluate_exits(ledger, frame, DATES[3]) == []  # nothing left to re-sell


def test_review3_supersede_correction_does_not_reset_session_clock(tmp_path: Path) -> None:
    # Review #3: the clock anchors to the chain ORIGIN's as_of, not the
    # surviving correction's.
    ledger = LedgerStore(tmp_path / "l.jsonl")
    original = _rec(as_of=DATES[0])
    ledger.append(original)
    ledger.append_fill(original.id, "ord-1", 10.0, ENTRY_PRICE, "filled")
    ledger.append(_rec(as_of=DATES[3], stop_price=91.0, supersedes=original.id))
    (pos,) = reconstruct_positions(ledger, _flat_calendar(DATES[:5]), DATES[4])
    assert pos.entry_session == DATES[1]  # from the ORIGIN, not the correction
    assert pos.sessions_held == 3
    assert pos.current_stop_price == 91.0  # the corrected stop still governs


def test_review4_unconfirmed_entry_never_sells(tmp_path: Path) -> None:
    # Review #4: an accepted-but-unpriced buy reserves heat but must never
    # arm the sell path — even when its presumed bar breaches the stop.
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(as_of=DATES[0])
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 10.0, None, "accepted")  # never priced
    frame = _bars([(d, 50.0, 51.0, 49.0, 50.0) for d in DATES[:4]])  # deep below stop
    (pos,) = reconstruct_positions(ledger, frame, DATES[3])
    assert pos.entry_confirmed is False
    assert pos.remaining_qty == 10.0  # heat still reserved
    assert evaluate_exits(ledger, frame, DATES[3]) == []


def test_review8_missing_bars_fail_loudly(tmp_path: Path) -> None:
    # Review #8: a held, confirmed position whose symbol has NO bars in the
    # monitoring frame is a structural gap — raise, never silently skip stops.
    from vega.execution.exits import ExitMonitorGapError

    ledger, rec = _open_position(tmp_path)  # position in AAA
    frame = _flat_calendar(DATES[:4], symbol="OTHER")  # AAA absent entirely
    with pytest.raises(ExitMonitorGapError, match="AAA"):
        evaluate_exits(ledger, frame, DATES[3])
