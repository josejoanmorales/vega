"""Generate today's pre-market briefing: uv run python -m vega.briefing"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

from dotenv import load_dotenv

from vega.briefing.calls import build_calls, load_signal_frame
from vega.briefing.engine import assemble
from vega.briefing.render import write_briefing
from vega.execution.executor import AlpacaPaperBackend, execute_pending
from vega.ledger.store import LedgerStore


def _live_equity() -> float:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.models import TradeAccount

    client = TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
    )
    account = cast(TradeAccount, client.get_account())
    if account.equity is None:
        raise RuntimeError("Alpaca paper account returned no equity value")
    return float(account.equity)


def main() -> None:
    load_dotenv()
    data = assemble()
    stale_after = (datetime.now(UTC).date() - timedelta(days=4)).isoformat()
    if data.as_of < stale_after:
        print(f"⚠ store is stale (latest session {data.as_of}) — run the ingest first")

    ledger = LedgerStore()
    equity: float | None = None
    try:
        equity = _live_equity()
        frame = load_signal_frame()
        result = build_calls(frame, data.as_of, equity, data.regime, ledger)
        data = replace(
            data,
            calls=result.calls,
            rejections=result.rejections,
            eligible_families=result.eligible_families,
            no_trade_reason=result.no_trade_reason,
        )
    except Exception as exc:  # noqa: BLE001 — fail closed: publish v1-only rather than crash the batch
        print(f"⚠ ranked calls unavailable this run ({exc}) — publishing v1 briefing only")

    path = write_briefing(data)
    print(f"briefing written: {path} (regime composite: {data.regime.composite})")

    if equity is not None:  # Alpaca reachable — safe to sweep pending longs (today's + stragglers)
        filled, failed = execute_pending(ledger, AlpacaPaperBackend())
        print(f"executed {filled} call(s), {failed} failure(s)")


if __name__ == "__main__":
    main()
