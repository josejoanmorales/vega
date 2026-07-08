"""End-to-end lifecycle smoke test against the real store (WI-065 DoD).

Run: uv run python -m vega.lifecycle

Uses the non-promotable SmaCrossSignal fixture (same as WI-063's smoke test)
to exercise the rationale gate and the state machine without ever actually
reaching paper-live — a fixture signal should never get that far.
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

    print(f"before rationale: has_rationale={rationale.has_rationale(family)}")
    universe = symbols(load_universe(), "equity", "etf")

    try:
        run_backtest(
            signal=SmaCrossSignal(asset_class="equity"),
            universe=universe,
            asset_class="equity",
            registry=registry,
            rationale_registry=rationale,
        )
    except ValueError as exc:
        print(f"backtest blocked as expected (no rationale yet): {exc}")

    rationale.record(
        family,
        "Fixture signal for pipeline smoke-testing only — non-promotable by design.",
        author="agent:sonnet",
    )
    report = run_backtest(
        signal=SmaCrossSignal(asset_class="equity"),
        universe=universe,
        asset_class="equity",
        registry=registry,
        rationale_registry=rationale,
    )
    print(f"backtest recorded: verdict={report.record.verdict}")

    try:
        lifecycle.promote_to_backtested(family, rationale, registry, actor="smoke_test")
    except LifecycleError as exc:
        print(f"promotion correctly blocked (non-promotable verdict, no 'pass'): {exc}")

    print(f"final state: {lifecycle.current_state(family)}")


if __name__ == "__main__":
    main()
