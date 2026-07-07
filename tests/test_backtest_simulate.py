from __future__ import annotations

import pandas as pd
import pytest

from vega.backtest.costs import apply_cost
from vega.backtest.market_view import MarketView
from vega.backtest.signals import EntryProposal
from vega.backtest.simulate import compute_atr, simulate_signal

LIQUID_BPS = 12.0  # median-dollar-volume tier this fixture always lands in


class _FixedSignal:
    """Fires exactly on the given date(s) — decouples fill-timing tests from
    any real crossover logic, which is tested separately in test_backtest_signals.py."""

    family = "test_fixed"
    version = "0.1"
    promotable = False

    def __init__(self, fire_on: str, **exit_overrides: object) -> None:
        self.fire_on = fire_on
        self.exit_overrides = exit_overrides

    def scan(self, view: MarketView, universe: list[str]) -> list[EntryProposal]:
        if view.as_of != self.fire_on:
            return []
        return [
            EntryProposal(
                symbol=universe[0],
                signal_family=self.family,
                signal_version=self.version,
                thesis="fixture",
                confidence=0.5,
                invalidation="fixture",
                **self.exit_overrides,  # type: ignore[arg-type]
            )
        ]


def _dates(n: int, start_day: int = 1) -> list[str]:
    return [f"2026-02-{d:02d}" for d in range(start_day, start_day + n)]


def _flat_history(symbol: str, dates: list[str], o=100.0, h=101.0, low=99.0, c=100.0) -> list[dict]:
    return [
        {
            "symbol": symbol,
            "date": d,
            "open": o,
            "high": h,
            "low": low,
            "close": c,
            "adj_close": c,
            "volume": 1_000_000.0,
        }
        for d in dates
    ]


DECISION_DATE = "2026-02-20"  # the 20th flat-history day
FILL_DATE = "2026-02-21"  # next session's open


def test_entry_fills_at_next_session_open_not_decision_close() -> None:
    pre = _flat_history("TEST", _dates(20))  # Feb 01..20
    fill_day = _flat_history("TEST", [FILL_DATE])
    frame = pd.DataFrame(pre + fill_day)
    signal = _FixedSignal(fire_on=DECISION_DATE)

    trades = simulate_signal(frame, _dates(21), signal, ["TEST"], "equity")
    # nothing closes in this window (no stop/time-stop reached) — force-closed at the end
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_date == FILL_DATE
    assert t.entry_price == pytest.approx(apply_cost(100.0, "buy", LIQUID_BPS))


def test_atr_is_none_below_period_and_exact_on_constant_true_range() -> None:
    frame = pd.DataFrame(_flat_history("TEST", _dates(20)))
    assert compute_atr(frame, "TEST", as_of="2026-02-10") is None  # only 10 bars available
    atr = compute_atr(frame, "TEST", as_of=DECISION_DATE)
    assert atr == pytest.approx(2.0)  # TR = max(101-99, |101-100|, |99-100|) = 2 every day


def test_gap_through_stop_fills_at_open_not_stop_price() -> None:
    pre = _flat_history("TEST", _dates(20))
    fill_day = _flat_history("TEST", [FILL_DATE])
    gap_day = [
        {
            "symbol": "TEST",
            "date": "2026-02-22",
            "open": 90.0,
            "high": 90.0,
            "low": 88.0,
            "close": 89.0,
            "adj_close": 89.0,
            "volume": 1_000_000.0,
        }
    ]
    frame = pd.DataFrame(pre + fill_day + gap_day)
    signal = _FixedSignal(fire_on=DECISION_DATE)

    trades = simulate_signal(frame, _dates(22), signal, ["TEST"], "equity")
    assert len(trades) == 1
    exit_ = trades[0].exits[0]
    assert exit_["reason"] == "gap_stop"
    assert exit_["price"] == pytest.approx(apply_cost(90.0, "sell", LIQUID_BPS))


def test_intraday_stop_touch_fills_at_stop_price_not_low() -> None:
    pre = _flat_history("TEST", _dates(20))
    fill_day = _flat_history("TEST", [FILL_DATE])
    entry_price = apply_cost(100.0, "buy", LIQUID_BPS)
    stop_price = entry_price - 2.0 * 2.0  # default stop_atr_mult=2.0, atr=2.0
    touch_day = [
        {
            "symbol": "TEST",
            "date": "2026-02-22",
            "open": 98.0,
            "high": 99.0,
            "low": 95.0,
            "close": 97.0,
            "adj_close": 97.0,
            "volume": 1_000_000.0,
        }
    ]
    frame = pd.DataFrame(pre + fill_day + touch_day)
    signal = _FixedSignal(fire_on=DECISION_DATE)

    trades = simulate_signal(frame, _dates(22), signal, ["TEST"], "equity")
    exit_ = trades[0].exits[0]
    assert exit_["reason"] == "stop"
    assert exit_["price"] == pytest.approx(apply_cost(stop_price, "sell", LIQUID_BPS))


def test_time_stop_exits_at_open_of_the_session_after_expiry() -> None:
    pre = _flat_history("TEST", _dates(20))
    holding_days = _flat_history("TEST", ["2026-02-21", "2026-02-22", "2026-02-23"])
    exit_day = [
        {
            "symbol": "TEST",
            "date": "2026-02-24",
            "open": 105.0,
            "high": 106.0,
            "low": 104.0,
            "close": 105.0,
            "adj_close": 105.0,
            "volume": 1_000_000.0,
        }
    ]
    frame = pd.DataFrame(pre + holding_days + exit_day)
    signal = _FixedSignal(fire_on=DECISION_DATE, time_stop_days=3)

    trades = simulate_signal(frame, _dates(24), signal, ["TEST"], "equity")
    assert len(trades) == 1
    exit_ = trades[0].exits[0]
    assert exit_["date"] == "2026-02-24" and exit_["reason"] == "time_stop"
    assert exit_["price"] == pytest.approx(apply_cost(105.0, "sell", LIQUID_BPS))


def test_profit_partial_then_trail_only_ever_tightens() -> None:
    pre = _flat_history("TEST", _dates(20))
    entry_price = apply_cost(100.0, "buy", LIQUID_BPS)
    atr = 2.0
    # day 1: hits the (lowered, for test ease) profit target -> half exits, trail initializes
    d1 = [
        {
            "symbol": "TEST",
            "date": "2026-02-21",
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "adj_close": 103.0,
            "volume": 1_000_000.0,
        }
    ]
    # day 2: a down day, but still above the day-1 trail (98) -> trail must NOT loosen
    d2 = [
        {
            "symbol": "TEST",
            "date": "2026-02-22",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 99.0,
            "adj_close": 99.0,
            "volume": 1_000_000.0,
        }
    ]
    # day 3: a new high -> trail tightens further (up)
    d3 = [
        {
            "symbol": "TEST",
            "date": "2026-02-23",
            "open": 100.0,
            "high": 111.0,
            "low": 99.0,
            "close": 110.0,
            "adj_close": 110.0,
            "volume": 1_000_000.0,
        }
    ]
    # day 4: gap below the day-3 trail -> remaining half closes
    d4 = [
        {
            "symbol": "TEST",
            "date": "2026-02-24",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adj_close": 100.0,
            "volume": 1_000_000.0,
        }
    ]
    frame = pd.DataFrame(pre + d1 + d2 + d3 + d4)
    signal = _FixedSignal(fire_on=DECISION_DATE, profit_take_half_at_r=1.0)

    trades = simulate_signal(frame, _dates(24), signal, ["TEST"], "equity")
    assert len(trades) == 1
    t = trades[0]
    target = entry_price + 1.0 * (entry_price - (entry_price - 2.0 * atr))
    assert len(t.exits) == 2
    partial, final = t.exits
    assert partial["reason"] == "profit_partial"
    assert partial["price"] == pytest.approx(apply_cost(target, "sell", LIQUID_BPS))
    assert partial["qty"] == pytest.approx(t.initial_qty / 2)
    # trail after day1 (close=103): 103 - 2.5*2.0 = 98; day2 down-close=90 must not loosen it;
    # day3 up-close=110 tightens it to 110 - 2.5*2.0 = 105 -> day4 open=100 gaps under 105
    assert final["reason"] == "gap_stop"
    assert final["price"] == pytest.approx(apply_cost(100.0, "sell", LIQUID_BPS))


def test_missing_bar_is_skipped_not_crashed() -> None:
    """A symbol absent on a given date (holiday mismatch) must not raise."""
    pre = _flat_history("TEST", _dates(20))
    fill_day = _flat_history("TEST", [FILL_DATE])
    # 2026-02-22 has NO row for TEST at all
    frame = pd.DataFrame(pre + fill_day)
    signal = _FixedSignal(fire_on=DECISION_DATE)
    trades = simulate_signal(frame, _dates(22), signal, ["TEST"], "equity")
    assert len(trades) == 1  # force-closed at end of window, no crash


def test_unresolved_position_force_closed_at_end_of_window() -> None:
    pre = _flat_history("TEST", _dates(20))
    fill_day = _flat_history("TEST", [FILL_DATE])
    frame = pd.DataFrame(pre + fill_day)
    signal = _FixedSignal(fire_on=DECISION_DATE)
    trades = simulate_signal(frame, _dates(21), signal, ["TEST"], "equity")
    assert trades[0].unresolved_at_end is True
    assert trades[0].exits[0]["reason"] == "end_of_data"
