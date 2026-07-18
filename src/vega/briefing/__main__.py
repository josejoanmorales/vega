"""Generate today's pre-market briefing: uv run python -m vega.briefing

Stage ordering and failure semantics (WI-067/WI-087 reviews):
- Exit monitoring is risk-REDUCING, so it runs whenever Alpaca is reachable —
  even on a stale store (a late stop beats an unmanaged one); only NEW ENTRIES
  require fresh data. An unreachable Alpaca skips everything and the published
  message says how many open positions went unmanaged this run.
- Exits attach to the published briefing THE MOMENT they execute — before
  demotions or entry generation run — so a later failure can never place
  sells that the published record omits (WI-087 review #6).
- A calls-generation failure fails CLOSED for entry execution: partially
  appended recs are never submitted by the run that failed to publish them.
- write_briefing keeps its write-once contract; on a same-session re-run the
  FIRST briefing wins and the conflict is tolerated, not fatal — build_calls'
  already_held dedup guarantees a re-run appended nothing new.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv

from vega.backtest.registry import BacktestRegistry
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
from vega.execution.exits import evaluate_exits, execute_exits, reconstruct_positions
from vega.ledger.store import LedgerStore
from vega.lifecycle.lifecycle import LifecycleRegistry
from vega.lifecycle.live_trades import check_and_apply_demotions, full_session_calendar

STALE_AFTER_DAYS = 4


def main() -> None:  # noqa: PLR0912, PLR0915 — the daily run's one orchestration seam
    load_dotenv()
    data = assemble()
    ledger = LedgerStore()

    calls_error: str | None = None
    equity: float | None = None
    backend: AlpacaPaperBackend | None = None

    stale_after = (datetime.now(UTC).date() - timedelta(days=STALE_AFTER_DAYS)).isoformat()
    stale = data.as_of < stale_after

    try:
        equity = live_account_equity()
        backend = AlpacaPaperBackend()
    except Exception as exc:  # noqa: BLE001 — unreachable venue = nothing may trade
        unmanaged = "open positions are UNMANAGED this run"
        try:
            frame = load_signal_frame(as_of=data.as_of)
            n = sum(1 for p in reconstruct_positions(ledger, frame, data.as_of) if not p.is_pending)
            unmanaged = f"{n} open position(s) are UNMANAGED this run"
        except Exception:  # noqa: BLE001, S110 — the count is best-effort context
            pass
        calls_error = (
            f"Alpaca account unavailable ({exc}) — entries AND exit monitoring skipped; {unmanaged}"
        )

    exited_today: frozenset[str] = frozenset()
    monitoring_ok = False
    lifecycle: LifecycleRegistry | None = None
    registry: BacktestRegistry | None = None
    if backend is not None:
        try:
            reconciled = reconcile_fills(ledger, backend)
            if reconciled:
                print(f"reconciled {reconciled} pending fill record(s)")

            frame = load_signal_frame(as_of=data.as_of)
            exit_decisions = evaluate_exits(ledger, frame, data.as_of)
            if exit_decisions:
                submitted, failed = execute_exits(ledger, backend, exit_decisions, data.as_of)
                print(f"exits: {submitted} submitted, {failed} failed")
            exited_today = frozenset(d.symbol for d in exit_decisions)
            # attach IMMEDIATELY — sells that executed must be on the published
            # record no matter what fails later this run (WI-087 review #6)
            data = replace(data, exits=tuple(exit_decisions))

            lifecycle = LifecycleRegistry()
            registry = BacktestRegistry()
            outcomes = check_and_apply_demotions(
                ledger, full_session_calendar(), lifecycle, registry, data.as_of
            )
            data = replace(data, signal_health=tuple(outcomes))
            monitoring_ok = True
        except Exception as exc:  # noqa: BLE001 — publish the failure; entries fail closed
            calls_error = (
                f"exit monitoring failed ({exc}) — entries skipped this run; any exits "
                "already submitted are on the ledger and in the Exits section above"
            )

    if calls_error is None and stale:
        calls_error = (
            f"store is stale (latest session {data.as_of}) — NEW ENTRIES skipped; exit "
            "monitoring still ran against the last-known session. Run the ingest."
        )

    entries_ok = False
    if (
        calls_error is None
        and monitoring_ok
        and equity is not None
        and lifecycle is not None
        and registry is not None
    ):
        try:
            result = build_calls(
                frame,
                data.as_of,
                equity,
                data.regime,
                ledger,
                lifecycle=lifecycle,
                backtest_registry=registry,
                exited_today=exited_today,
            )
            data = replace(
                data,
                calls=result.calls,
                rejections=result.rejections,
                eligible_families=result.eligible_families,
                no_trade_reason=result.no_trade_reason,
            )
            entries_ok = True
        except Exception as exc:  # noqa: BLE001 — never execute unpublished entries
            calls_error = f"ranked-calls generation failed ({exc}) — entry execution skipped"

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

    if backend is not None and equity is not None and entries_ok:
        submitted, failed = execute_pending(ledger, backend, as_of=data.as_of, equity=equity)
        print(f"executed {submitted} call(s), {failed} failure(s)")


if __name__ == "__main__":
    main()
