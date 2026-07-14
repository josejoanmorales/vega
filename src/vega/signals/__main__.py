"""Run all 6 grid points across the 3 WI-066 signal families against the real
store, honestly. Rationale-first gate must already be satisfied (recorded via
RationaleRegistry before this ever runs) or run_backtest raises.

Run: uv run python -m vega.signals
"""

from __future__ import annotations

from vega.backtest.engine import run_backtest
from vega.backtest.registry import BacktestRegistry
from vega.backtest.signals import Signal
from vega.data.universe import load_universe, symbols
from vega.lifecycle.rationale import RationaleRegistry
from vega.signals.breakout_volume import BreakoutVolumeSignal
from vega.signals.oversold_reversion import OversoldReversionSignal
from vega.signals.trend_pullback import TrendPullbackSignal

GRID_SIZE_PER_FAMILY = 2  # total planned grid per family (for honest cumulative-trial accounting)


def main() -> None:
    universe = symbols(load_universe(), "equity", "etf")
    rationale = RationaleRegistry()
    registry = BacktestRegistry()

    signals: list[tuple[str, Signal]] = [
        ("trend_pullback_v1[depth=3%]", TrendPullbackSignal(depth=0.03)),
        ("trend_pullback_v1[depth=5%]", TrendPullbackSignal(depth=0.05)),
        ("breakout_volume_v1[N=40]", BreakoutVolumeSignal(n_sessions=40)),
        ("breakout_volume_v1[N=55]", BreakoutVolumeSignal(n_sessions=55)),
        ("oversold_reversion_v1[k=2.0]", OversoldReversionSignal(k=2.0)),
        ("oversold_reversion_v1[k=2.5]", OversoldReversionSignal(k=2.5)),
    ]

    for label, signal in signals:
        print(f"\n=== {label} ===")
        report = run_backtest(
            signal=signal,
            universe=universe,
            asset_class="equity",
            rationale_registry=rationale,
            registry=registry,
            param_grid_size=GRID_SIZE_PER_FAMILY,
        )
        r = report.record
        n_trades = r.aggregate_metrics.get("n_trades")
        sharpe = r.aggregate_metrics.get("sharpe")
        print(f"verdict: {r.verdict}")
        print(f"n_trades: {n_trades}  sharpe: {sharpe}")
        print(f"promotion_bar: {r.promotion_bar}")
        if r.verdict == "fail":
            fail_note = next((n for n in r.notes if n.startswith("sharpe=")), None)
            print(f"  {fail_note}")


if __name__ == "__main__":
    main()
