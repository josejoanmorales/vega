from pathlib import Path

from vega.execution.executor import (
    OrderResult,
    execute_pending,
    pending_longs,
    read_failures,
)
from vega.execution.pnl import haircut_prices, paper_pnl
from vega.ledger.store import LedgerStore
from vega.ledger.types import Recommendation


def _rec(**overrides: object) -> Recommendation:
    base: dict[str, object] = {
        "symbol": "AAPL",
        "asset_class": "equity",
        "direction": "long",
        "thesis": "trend pullback in a risk-on regime",
        "confidence": 0.62,
        "horizon_days": 10,
        "entry_ref_price": 230.0,
        "stop_price": 221.0,
        "time_stop_date": "2026-07-20",
        "profit_rule": "half at +2R, trail rest at 10DMA",
        "invalidation": "close below the June swing low on consolidated volume",
        "signal_attribution": ("trend_pullback_v1",),
    }
    base.update(overrides)
    return Recommendation(**base)  # type: ignore[arg-type]


class FakeBackend:
    def __init__(self, fail_symbols: set[str] | None = None) -> None:
        self.fail_symbols = fail_symbols or set()
        self.orders: list[str] = []

    def submit_market_buy(self, symbol: str, qty: float, asset_class: str) -> OrderResult:
        if symbol in self.fail_symbols:
            raise RuntimeError(f"{symbol} not tradable")
        self.orders.append(symbol)
        return OrderResult("ord-1", symbol, qty, 100.0, "filled")


def test_pending_skips_filled_and_superseded(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    first = _rec()
    ledger.append(first)
    corrected = _rec(stop_price=222.0, supersedes=first.id)
    ledger.append(corrected)
    other = _rec(symbol="MSFT")
    ledger.append(other)
    ledger.append_fill(other.id, "ord-9", 4.0, 500.0, "filled")
    pending = pending_longs(ledger)
    assert [r["id"] for r in pending] == [corrected.id]


def test_execute_records_fill_linked_to_rec(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec()
    ledger.append(rec)
    filled, failed = execute_pending(
        ledger, FakeBackend(), notional_usd=1000.0, failures_path=tmp_path / "f.jsonl"
    )
    assert (filled, failed) == (1, 0)
    fill = ledger.fills()[0]
    assert fill["ref_id"] == rec.id and fill["qty"] == round(1000.0 / 230.0, 6)
    assert pending_longs(ledger) == []


def test_failure_is_logged_and_batch_continues(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    bad = _rec(symbol="GRAM", asset_class="crypto")
    good = _rec(symbol="AAPL")
    ledger.append(bad)
    ledger.append(good)
    fpath = tmp_path / "f.jsonl"
    filled, failed = execute_pending(ledger, FakeBackend({"GRAM"}), failures_path=fpath)
    assert (filled, failed) == (1, 1)
    failures = read_failures(fpath)
    assert failures[0]["symbol"] == "GRAM" and failures[0]["ref_id"] == bad.id


def test_slippage_haircut_math() -> None:
    entry, exit_ = haircut_prices(100.0, 110.0, "equity")
    assert entry == 100.1 and exit_ == 109.89
    # crypto haircut is 3x wider
    assert paper_pnl(100.0, 110.0, 10.0, "crypto") < paper_pnl(100.0, 110.0, 10.0, "equity")
    # a flat round-trip is a small loss after slippage, never zero
    assert paper_pnl(100.0, 100.0, 10.0, "equity") < 0
