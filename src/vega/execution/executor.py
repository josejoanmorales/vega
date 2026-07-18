"""Paper-trading executor: pending ledger longs → Alpaca paper market orders.

Execution honors the backtest's fill model (WI-067 review):
- Only calls decided at the CURRENT session execute — a pending rec from an
  earlier session missed its T+1 open and EXPIRES with a failure record; it is
  never late-filled at prices the backtest never modeled.
- qty is exclusively the risk engine's. There is no notional fallback: a
  sizing decision the risk engine never made must not enter the paper track
  record. Orders breaching the notional ceiling are refused, not clamped.
- A submission records whatever Alpaca reported — often an ACCEPTANCE
  (price=None) for pre-market orders. `reconcile_fills` re-polls those orders
  on later runs and appends the real fill (or terminal cancel) once known;
  `LedgerStore.latest_with_fills` prefers priced fills.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from vega.common.paths import DATA_ROOT
from vega.ledger.store import LedgerStore

FAILURES_PATH = DATA_ROOT / "ledger/exec_failures.jsonl"
FILL_POLL_ATTEMPTS = 5
FILL_POLL_DELAY_S = 2.0
# Sanity ceiling per order, as a fraction of account equity. Risk-engine sizing
# lands well under this (~7-10% observed); the ceiling exists to stop a corrupt
# or hand-edited ledger qty from becoming an unbounded market order.
MAX_ORDER_EQUITY_FRACTION = 0.25
# Alpaca order states that mean "this order will never fill".
TERMINAL_UNFILLED_STATUSES = frozenset({"canceled", "expired", "rejected"})


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    symbol: str
    qty: float
    filled_avg_price: float | None
    status: str


class TradingBackend(Protocol):
    def submit_market_buy(self, symbol: str, qty: float, asset_class: str) -> OrderResult: ...

    def submit_market_sell(self, symbol: str, qty: float, asset_class: str) -> OrderResult: ...

    def order_status(self, order_id: str) -> OrderResult: ...


def live_account_equity() -> float:
    """Alpaca paper-account equity — the ONE copy (WI-067 review found this
    duplicated byte-for-byte across two __main__ entry points)."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.models import TradeAccount

    client = TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
    )
    account = cast(TradeAccount, client.get_account())
    if account.equity is None:
        raise RuntimeError("Alpaca paper account returned no equity value")
    return float(account.equity)


class AlpacaPaperBackend:
    """Real backend. TradingClient(paper=True) manages the endpoint itself."""

    def __init__(self) -> None:
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(
            os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
        )

    @staticmethod
    def _to_result(order: Any, symbol: str, qty: float) -> OrderResult:
        return OrderResult(
            order_id=str(order.id),
            symbol=symbol,
            qty=qty,
            filled_avg_price=(
                float(order.filled_avg_price) if order.filled_avg_price is not None else None
            ),
            status=str(order.status.value if hasattr(order.status, "value") else order.status),
        )

    def _submit_market_order(
        self, symbol: str, qty: float, asset_class: str, side: Any
    ) -> OrderResult:
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.models import Order
        from alpaca.trading.requests import MarketOrderRequest

        alpaca_symbol = f"{symbol}/USD" if asset_class == "crypto" else symbol
        tif = TimeInForce.GTC if asset_class == "crypto" else TimeInForce.DAY
        request = MarketOrderRequest(symbol=alpaca_symbol, qty=qty, side=side, time_in_force=tif)
        order = self._client.submit_order(request)
        assert isinstance(order, Order)  # noqa: S101 — narrows the union alpaca-py returns
        for _ in range(FILL_POLL_ATTEMPTS):
            if order.filled_avg_price is not None:
                break
            time.sleep(FILL_POLL_DELAY_S)
            refreshed = self._client.get_order_by_id(order.id)
            assert isinstance(refreshed, Order)  # noqa: S101
            order = refreshed
        return self._to_result(order, symbol, qty)

    def submit_market_buy(self, symbol: str, qty: float, asset_class: str) -> OrderResult:
        from alpaca.trading.enums import OrderSide

        return self._submit_market_order(symbol, qty, asset_class, OrderSide.BUY)

    def submit_market_sell(self, symbol: str, qty: float, asset_class: str) -> OrderResult:
        from alpaca.trading.enums import OrderSide

        return self._submit_market_order(symbol, qty, asset_class, OrderSide.SELL)

    def order_status(self, order_id: str) -> OrderResult:
        from alpaca.trading.models import Order

        order = self._client.get_order_by_id(order_id)
        assert isinstance(order, Order)  # noqa: S101
        qty = float(order.filled_qty or order.qty or 0.0)
        symbol = str(order.symbol).removesuffix("/USD")
        return self._to_result(order, symbol, qty)


def record_failure(ref_id: str, symbol: str, error: str, path: Path = FAILURES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "ref_id": ref_id,
                    "symbol": symbol,
                    "error": error,
                },
                sort_keys=True,
            )
            + "\n"
        )
        fh.flush()
        os.fsync(fh.fileno())


def read_failures(path: Path = FAILURES_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def pending_longs(ledger: LedgerStore) -> list[dict[str, Any]]:
    """Latest (non-superseded) long recommendations with no fill anywhere in
    their supersede chain (WI-067 review: id-based fill lookup re-executed
    filled-then-corrected positions)."""
    return [
        rec
        for rec, fill in ledger.latest_with_fills()
        if rec["direction"] == "long" and fill is None
    ]


def reconcile_fills(ledger: LedgerStore, backend: TradingBackend) -> int:
    """Re-poll orders whose ledger record is still an acceptance (price=None)
    and append the real outcome once Alpaca knows it: a priced fill, or a
    terminal cancel/reject marker (qty 0). Idempotent — chains that already
    carry a resolved record are skipped. Best-effort: a lookup failure leaves
    the record for the next run."""
    resolved_orders = {
        f["order_id"]
        for f in ledger.fills()
        if f.get("price") is not None or f.get("status") in TERMINAL_UNFILLED_STATUSES
    }
    reconciled = 0
    for fill in ledger.fills():
        order_id = fill["order_id"]
        if order_id in resolved_orders:
            continue
        try:
            result = backend.order_status(order_id)
        except Exception:  # noqa: BLE001, S112 — best-effort; unresolved records retry next run
            continue
        # The reconciliation record must carry the SAME identity as the
        # acceptance it resolves (WI-087 review finding #1: dropping side/
        # reason/session re-labeled every reconciled SELL as a buy, corrupting
        # reconstruction and silently starving auto-demotion of round trips).
        identity = {
            "side": fill.get("side", "buy"),
            "reason": fill.get("reason"),
            "session": fill.get("session"),
        }
        if result.filled_avg_price is not None:
            ledger.append_fill(
                fill["ref_id"],
                order_id,
                result.qty,
                result.filled_avg_price,
                result.status,
                **identity,
            )
        elif result.status in TERMINAL_UNFILLED_STATUSES:
            ledger.append_fill(fill["ref_id"], order_id, 0.0, None, result.status, **identity)
        else:
            continue  # still working at the venue — try again next run
        resolved_orders.add(order_id)
        reconciled += 1
    return reconciled


def execute_pending(
    ledger: LedgerStore,
    backend: TradingBackend,
    as_of: str,
    equity: float,
    failures_path: Path = FAILURES_PATH,
) -> tuple[int, int]:
    """Execute pending longs decided at `as_of`. Returns (submitted, failed).
    Failures never raise; each is recorded once. Stale pending calls expire
    (recorded once, never counted as today's failures, never late-filled)."""
    submitted = 0
    failed = 0
    max_notional = MAX_ORDER_EQUITY_FRACTION * equity
    already_recorded = {f["ref_id"] for f in read_failures(failures_path)}
    for rec in pending_longs(ledger):
        if rec.get("as_of") != as_of:
            if rec["id"] not in already_recorded:
                record_failure(
                    rec["id"],
                    rec["symbol"],
                    f"expired: decided at session {rec.get('as_of')!r}, missed its T+1 open "
                    "— never late-filled (backtest fill model)",
                    failures_path,
                )
            continue
        qty = rec.get("qty")
        if qty is None or float(qty) <= 0:
            record_failure(
                rec["id"],
                rec["symbol"],
                "no positive risk-engine qty on record — refusing to size outside the risk engine",
                failures_path,
            )
            failed += 1
            continue
        qty = round(float(qty), 6)
        notional = qty * float(rec["entry_ref_price"])
        if notional > max_notional:
            record_failure(
                rec["id"],
                rec["symbol"],
                f"order notional ${notional:,.0f} breaches the sanity ceiling "
                f"${max_notional:,.0f} ({MAX_ORDER_EQUITY_FRACTION:.0%} of equity) — refused",
                failures_path,
            )
            failed += 1
            continue
        try:
            result = backend.submit_market_buy(rec["symbol"], qty, rec["asset_class"])
            ledger.append_fill(
                ref_id=rec["id"],
                order_id=result.order_id,
                qty=result.qty,
                price=result.filled_avg_price,
                status=result.status,
            )
            submitted += 1
        except Exception as exc:  # noqa: BLE001 — one bad order must not stop the batch
            record_failure(rec["id"], rec["symbol"], str(exc), failures_path)
            failed += 1
    return submitted, failed
