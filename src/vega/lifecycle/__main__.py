"""End-to-end lifecycle smoke test against the real store (WI-065 DoD).

Run: uv run python -m vega.lifecycle

Uses a 3-symbol universe (not the full 545) — this smoke test proves the
gate + state machine end to end, not backtest correctness at scale, so it
runs in seconds. The gate-blocked demonstration comes FIRST and cheaply,
since the rationale gate now fires before any walk-forward compute.
"""

from __future__ import annotations

from pathlib import Path

from vega.backtest.engine import run_backtest
from vega.backtest.registry import BacktestRegistry
from vega.backtest.signals import SmaCrossSignal
from vega.data.universe import load_universe, symbols
from vega.lifecycle.lifecycle import LifecycleError, LifecycleRegistry
from vega.lifecycle.rationale import RationaleRegistry

SMOKE_ROOT = Path("data/lifecycle_smoke")


def main() -> None:
    rationale = RationaleRegistry(SMOKE_ROOT / "rationale.jsonl")
    registry = BacktestRegistry(SMOKE_ROOT / "registry.jsonl")
    lifecycle = LifecycleRegistry(SMOKE_ROOT / "transitions.jsonl")
    family = SmaCrossSignal.family
    universe = symbols(load_universe(), "equity", "etf")[:3]  # tiny — this tests governance

    print(f"before rationale: has_rationale={rationale.has_rationale(family)}")
    try:
        run_backtest(
            signal=SmaCrossSignal(asset_class="equity"),
            universe=universe,
            asset_class="equity",
            rationale_registry=rationale,
        )
    except ValueError as exc:
        print(f"backtest blocked BEFORE any compute (no rationale yet): {exc}")

    rationale.record(
        family,
        "Fixture signal for pipeline smoke-testing only — non-promotable by design.",
        author="agent:sonnet",
    )
    report = run_backtest(
        signal=SmaCrossSignal(asset_class="equity"),
        universe=universe,
        asset_class="equity",
        rationale_registry=rationale,
    )
    print(f"backtest recorded: verdict={report.record.verdict}")

    try:
        lifecycle.promote_to_backtested(family, rationale, registry, actor="agent:sonnet")
    except LifecycleError as exc:
        print(f"promotion correctly blocked (non-promotable, no 'pass' run): {exc}")

    print(f"final state: {lifecycle.current_state(family)}")


if __name__ == "__main__":
    main()
