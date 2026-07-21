"""Smoke-test the engine end-to-end against the real store (WI-063 DoD).

Run: uv run python -m vega.backtest
Uses the non-promotable SmaCrossSignal placeholder — this proves the
walk-forward/registry pipeline works, it does not produce a trading signal.
"""

from __future__ import annotations

import json

from vega.backtest.engine import DEFAULT_BENCHMARK, run_backtest
from vega.backtest.signals import SmaCrossSignal
from vega.data.universe import load_universe, tradable_symbols
from vega.lifecycle.rationale import NullRationaleRegistry


def main() -> None:
    # SPY excluded (WI-084 item 7): same self-benchmarking hazard as signals/__main__.py.
    universe = tradable_symbols(
        load_universe(), "equity", "etf", exclude={DEFAULT_BENCHMARK["equity"]}
    )
    report = run_backtest(
        signal=SmaCrossSignal(asset_class="equity"),
        universe=universe,
        asset_class="equity",
        # explicitly opting out of the rationale gate — this smoke test exercises
        # the walk-forward mechanics, not governance (the gate has its own tests)
        rationale_registry=NullRationaleRegistry(),
    )
    r = report.record
    print(f"verdict: {r.verdict} (family={r.signal_family}, folds={r.n_folds})")
    print(f"aggregate: {json.dumps(r.aggregate_metrics, indent=2)}")
    print(f"notes: {r.notes}")


if __name__ == "__main__":
    main()
