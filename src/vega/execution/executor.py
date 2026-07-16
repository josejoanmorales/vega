"""Paper-trading executor: pending ledger longs → Alpaca paper market orders.

Sizing prefers the risk engine's `qty` (WI-064/WI-067: recommendations built
via `risk.engine.to_recommendation` always carry it); fixed notional is the
fallback only for recommendations that bypassed the risk engine entirely.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from vega.common.paths import DATA_ROOT
from vega.ledger.store import LedgerStore

DEFAULT_NOTIONAL_USD = 1_000.0
FAILURES_PATH = DATA_ROOT / "ledger/exec_failures.jsonl"
FILL_POLL_ATTEMPTS = 5
FILL_POLL_DELAY_S = 2.0


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    symbol: str
    qty: float
    filled_avg_price: float | None
    status: str


class TradingBackend(Protocol):
    def submit_market_buy(self, symbol: str, qty: float, asset_class: str) -> OrderResult: ...


class AlpacaPaperBackend:
    """Real backend. TradingClient(paper=True) manages the endpoint itself."""

    def __init__(self) -> None:
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(
            os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
        )

    def submit_market_buy(self, symbol: str, qty: float, asset_class: str) -> OrderResult:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.models import Order
        from alpaca.trading.requests import MarketOrderRequest

        alpaca_symbol = f"{symbol}/USD" if asset_class == "crypto" else symbol
        tif = TimeInForce.GTC if asset_class == "crypto" else TimeInForce.DAY
        request = MarketOrderRequest(
            symbol=alpaca_symbol, qty=qty, side=OrderSide.BUY, time_in_force=tif
        )
        order = self._client.submit_order(request)
        assert isinstance(order, Order)  # noqa: S101 — narrows the union alpaca-py returns
        for _ in range(FILL_POLL_ATTEMPTS):
            if order.filled_avg_price is not None:
                break
            time.sleep(FILL_POLL_DELAY_S)
            refreshed = self._client.get_order_by_id(order.id)
            assert isinstance(refreshed, Order)  # noqa: S101
            order = refreshed
        return OrderResult(
            order_id=str(order.id),
            symbol=symbol,
            qty=qty,
            filled_avg_price=(
                float(order.filled_avg_price) if order.filled_avg_price is not None else None
            ),
            status=str(order.status.value if hasattr(order.status, "value") else order.status),
        )


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
    """Latest (non-superseded) long recommendations without a linked fill."""
    filled = {f["ref_id"] for f in ledger.fills()}
    return [r for r in ledger.latest() if r["direction"] == "long" and r["id"] not in filled]


def execute_pending(
    ledger: LedgerStore,
    backend: TradingBackend,
    notional_usd: float = DEFAULT_NOTIONAL_USD,
    failures_path: Path = FAILURES_PATH,
) -> tuple[int, int]:
    """Execute all pending longs. Sizes from the recommendation's risk-engine
    `qty` when present (WI-067) — the risk-sized qty reflects the family's
    actual R-based sizing and must win over the notional placeholder. Falls
    back to fixed notional only for recommendations that never went through
    the risk engine (e.g. hand-entered ledger overrides). Returns (filled,
    failed). Failures never raise."""
    filled = 0
    failed = 0
    for rec in pending_longs(ledger):
        qty = (
            round(float(rec["qty"]), 6)
            if rec.get("qty")
            else round(notional_usd / float(rec["entry_ref_price"]), 6)
        )
        try:
            result = backend.submit_market_buy(rec["symbol"], qty, rec["asset_class"])
            ledger.append_fill(
                ref_id=rec["id"],
                order_id=result.order_id,
                qty=result.qty,
                price=result.filled_avg_price,
                status=result.status,
            )
            filled += 1
        except Exception as exc:  # noqa: BLE001 — one bad order must not stop the batch
            record_failure(rec["id"], rec["symbol"], str(exc), failures_path)
            failed += 1
    return filled, failed
