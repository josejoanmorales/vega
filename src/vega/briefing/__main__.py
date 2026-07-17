"""Generate today's pre-market briefing: uv run python -m vega.briefing

Stage ordering and failure semantics (WI-067 review):
- Staleness and Alpaca reachability are HARD gates for the calls path — a
  stale store or unreachable account skips both call generation and execution
  (never place orders on old data or unverified equity).
- A calls-generation failure fails CLOSED for execution too: partially
  appended recs are never submitted by the run that failed to publish them.
  The failure itself is rendered into the published briefing (calls_error).
- write_briefing keeps its write-once contract; on a same-session re-run the
  FIRST briefing wins and the conflict is tolerated, not fatal — build_calls'
  already_held dedup guarantees a re-run appended nothing new, and execution/
  reconciliation still proceed.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv

from vega.briefing.calls import build_calls, load_signal_frame
from vega.briefing.engine import assemble
from vega.briefing.render import write_briefing
from vega.data.types import SnapshotConflictError
from vega.execution.executor import (
    AlpacaPaperBackend,
    execute_pending,
    live_account_equity,
    reconcile_fills,
)
from vega.ledger.store import LedgerStore

STALE_AFTER_DAYS = 4


def main() -> None:
    load_dotenv()
    data = assemble()
    ledger = LedgerStore()

    calls_error: str | None = None
    equity: float | None = None

    stale_after = (datetime.now(UTC).date() - timedelta(days=STALE_AFTER_DAYS)).isoformat()
    if data.as_of < stale_after:
        calls_error = (
            f"store is stale (latest session {data.as_of}) — calls and execution "
            "skipped; run the ingest"
        )
    else:
        try:
            equity = live_account_equity()
        except Exception as exc:  # noqa: BLE001 — unreachable account = no calls, no orders
            calls_error = f"Alpaca account unavailable ({exc}) — calls and execution skipped"

    if calls_error is None and equity is not None:
        try:
            frame = load_signal_frame(as_of=data.as_of)
            result = build_calls(frame, data.as_of, equity, data.regime, ledger)
            data = replace(
                data,
                calls=result.calls,
                rejections=result.rejections,
                eligible_families=result.eligible_families,
                no_trade_reason=result.no_trade_reason,
            )
        except Exception as exc:  # noqa: BLE001 — publish the failure; never execute unpublished
            calls_error = f"ranked-calls generation failed ({exc}) — execution skipped this run"
            equity = None  # fail CLOSED: partial appends must not execute this run

    if calls_error is not None:
        data = replace(data, calls_error=calls_error)
        print(f"⚠ {calls_error}")

    try:
        path = write_briefing(data)
        print(f"briefing written: {path} (regime composite: {data.regime.composite})")
    except SnapshotConflictError:
        print(
            f"briefing for {data.as_of} already published — write-once keeps the original; "
            "continuing (re-runs append nothing: already_held dedup)"
        )

    if equity is not None:
        backend = AlpacaPaperBackend()
        reconciled = reconcile_fills(ledger, backend)
        if reconciled:
            print(f"reconciled {reconciled} pending fill record(s)")
        submitted, failed = execute_pending(ledger, backend, as_of=data.as_of, equity=equity)
        print(f"executed {submitted} call(s), {failed} failure(s)")


if __name__ == "__main__":
    main()
