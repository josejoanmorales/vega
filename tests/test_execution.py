from pathlib import Path

from vega.execution.executor import (
    OrderResult,
    execute_pending,
    pending_longs,
    read_failures,
    reconcile_fills,
)
from vega.execution.pnl import haircut_prices, paper_pnl
from vega.ledger.store import LedgerStore
from vega.ledger.types import Recommendation

AS_OF = "2026-07-10"


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
        "qty": 4.2,
        "as_of": AS_OF,
    }
    base.update(overrides)
    return Recommendation(**base)  # type: ignore[arg-type]


class FakeBackend:
    def __init__(
        self,
        fail_symbols: set[str] | None = None,
        order_statuses: dict[str, OrderResult] | None = None,
    ) -> None:
        self.fail_symbols = fail_symbols or set()
        self.order_statuses = order_statuses or {}
        self.orders: list[str] = []

    def submit_market_buy(self, symbol: str, qty: float, asset_class: str) -> OrderResult:
        if symbol in self.fail_symbols:
            raise RuntimeError(f"{symbol} not tradable")
        self.orders.append(symbol)
        return OrderResult(f"ord-{len(self.orders)}", symbol, qty, 100.0, "filled")

    def order_status(self, order_id: str) -> OrderResult:
        if order_id not in self.order_statuses:
            raise RuntimeError(f"unknown order {order_id}")
        return self.order_statuses[order_id]


def _execute(ledger: LedgerStore, backend: FakeBackend, tmp_path: Path) -> tuple[int, int]:
    return execute_pending(
        ledger, backend, as_of=AS_OF, equity=100_000.0, failures_path=tmp_path / "f.jsonl"
    )


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


def test_filled_then_superseded_is_not_re_executed(tmp_path: Path) -> None:
    # WI-067 review: fills resolve through supersede chains — a correction to a
    # FILLED position must not make it look pending again.
    ledger = LedgerStore(tmp_path / "l.jsonl")
    original = _rec()
    ledger.append(original)
    ledger.append_fill(original.id, "ord-1", 4.2, 230.5, "filled")
    ledger.append(_rec(stop_price=222.0, supersedes=original.id))
    assert pending_longs(ledger) == []


def test_execute_records_fill_with_risk_engine_qty(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec(qty=4.2)
    ledger.append(rec)
    filled, failed = _execute(ledger, FakeBackend(), tmp_path)
    assert (filled, failed) == (1, 0)
    fill = ledger.fills()[0]
    assert fill["ref_id"] == rec.id and fill["qty"] == 4.2
    assert pending_longs(ledger) == []


def test_stale_pending_expires_instead_of_late_filling(tmp_path: Path) -> None:
    # A rec that missed its T+1 open is never submitted at unmodeled prices.
    ledger = LedgerStore(tmp_path / "l.jsonl")
    stale = _rec(symbol="STALE", as_of="2026-07-08")
    ledger.append(stale)
    backend = FakeBackend()
    filled, failed = _execute(ledger, backend, tmp_path)
    assert (filled, failed) == (0, 0)  # expiry is not a today-failure
    assert backend.orders == []
    failures = read_failures(tmp_path / "f.jsonl")
    assert len(failures) == 1 and "expired" in failures[0]["error"]
    # surfaced exactly once, not re-recorded every subsequent run
    _execute(ledger, backend, tmp_path)
    assert len(read_failures(tmp_path / "f.jsonl")) == 1


def test_qty_less_rec_is_refused_never_notional_sized(tmp_path: Path) -> None:
    # WI-067 review: a sizing decision the risk engine never made must not
    # enter the track record — the $1,000 notional fallback is gone.
    ledger = LedgerStore(tmp_path / "l.jsonl")
    ledger.append(_rec(qty=None))
    backend = FakeBackend()
    filled, failed = _execute(ledger, backend, tmp_path)
    assert (filled, failed) == (0, 1)
    assert backend.orders == []
    assert "risk-engine qty" in read_failures(tmp_path / "f.jsonl")[0]["error"]


def test_order_notional_ceiling_refuses_oversized_orders(tmp_path: Path) -> None:
    # 200 shares x $230 = $46k > 25% of $100k equity — refused, not clamped.
    ledger = LedgerStore(tmp_path / "l.jsonl")
    ledger.append(_rec(qty=200.0))
    backend = FakeBackend()
    filled, failed = _execute(ledger, backend, tmp_path)
    assert (filled, failed) == (0, 1)
    assert backend.orders == []
    assert "ceiling" in read_failures(tmp_path / "f.jsonl")[0]["error"]


def test_failure_is_logged_and_batch_continues(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    bad = _rec(symbol="GRAM", asset_class="crypto")
    good = _rec(symbol="AAPL")
    ledger.append(bad)
    ledger.append(good)
    filled, failed = _execute(ledger, FakeBackend({"GRAM"}), tmp_path)
    assert (filled, failed) == (1, 1)
    failures = read_failures(tmp_path / "f.jsonl")
    assert failures[0]["symbol"] == "GRAM" and failures[0]["ref_id"] == bad.id


def test_reconcile_prices_an_accepted_order_once_filled(tmp_path: Path) -> None:
    # Pre-market submissions record price=None/status='accepted'; reconciliation
    # appends the real fill once the venue reports it, and is idempotent.
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 4.2, None, "accepted")
    backend = FakeBackend(
        order_statuses={"ord-1": OrderResult("ord-1", "AAPL", 4.2, 231.1, "filled")}
    )
    assert reconcile_fills(ledger, backend) == 1
    pairs = {r["id"]: f for r, f in ledger.latest_with_fills()}
    assert pairs[rec.id] is not None and pairs[rec.id]["price"] == 231.1  # priced fill wins
    assert reconcile_fills(ledger, backend) == 0  # already resolved — idempotent


def test_reconcile_marks_terminally_dead_orders(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 4.2, None, "accepted")
    backend = FakeBackend(
        order_statuses={"ord-1": OrderResult("ord-1", "AAPL", 0.0, None, "canceled")}
    )
    assert reconcile_fills(ledger, backend) == 1
    pairs = {r["id"]: f for r, f in ledger.latest_with_fills()}
    assert pairs[rec.id] is not None and pairs[rec.id]["status"] == "canceled"
    assert reconcile_fills(ledger, backend) == 0


def test_reconcile_tolerates_lookup_failures(tmp_path: Path) -> None:
    ledger = LedgerStore(tmp_path / "l.jsonl")
    rec = _rec()
    ledger.append(rec)
    ledger.append_fill(rec.id, "ord-1", 4.2, None, "accepted")
    assert reconcile_fills(ledger, FakeBackend()) == 0  # unknown order → retry next run


def test_slippage_haircut_math() -> None:
    entry, exit_ = haircut_prices(100.0, 110.0, "equity")
    assert entry == 100.1 and exit_ == 109.89
    # crypto haircut is 3x wider
    assert paper_pnl(100.0, 110.0, 10.0, "crypto") < paper_pnl(100.0, 110.0, 10.0, "equity")
    # a flat round-trip is a small loss after slippage, never zero
    assert paper_pnl(100.0, 100.0, 10.0, "equity") < 0
