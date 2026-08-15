"""Run all grid points across every registered signal family against the real
store, honestly. Rationale-first gate must already be satisfied (recorded via
RationaleRegistry before this ever runs) or run_backtest raises.

Run: uv run python -m vega.signals
"""

from __future__ import annotations

from vega.backtest.engine import DEFAULT_BENCHMARK, run_backtest
from vega.backtest.registry import BacktestRegistry
from vega.backtest.signals import Signal
from vega.data.universe import load_universe, tradable_symbols
from vega.lifecycle.rationale import RationaleRegistry
from vega.signals.breakout_volume import BreakoutVolumeSignal
from vega.signals.crypto_breakout import N_SESSIONS_GRID, CryptoBreakoutSignal
from vega.signals.crypto_oversold_reversion import CryptoOversoldReversionSignal
from vega.signals.oversold_reversion import OversoldReversionSignal
from vega.signals.trend_pullback import TrendPullbackSignal

_Universe = list[str]

# One run = ONE grid point = param_grid_size 1 (review fix: passing the family's
# planned total on every run double-counted trials — the bar for run 2 was computed
# as if 4 hypotheses had been tried when only 2 params exist).
PARAM_GRID_SIZE_PER_RUN = 1


def _run(
    label: str,
    signal: Signal,
    universe: _Universe,
    asset_class: str,
    rationale: RationaleRegistry,
    registry: BacktestRegistry,
) -> None:
    print(f"\n=== {label} ({asset_class}) ===")
    report = run_backtest(
        signal=signal,
        universe=universe,
        asset_class=asset_class,
        rationale_registry=rationale,
        registry=registry,
        param_grid_size=PARAM_GRID_SIZE_PER_RUN,
    )
    r = report.record
    dev_sharpe = r.aggregate_metrics.get("sharpe")
    n_trades = r.aggregate_metrics.get("n_trades")
    print(f"data span: {r.data_span[0]} .. {r.data_span[1]}  ({r.n_folds} dev folds)")
    print(f"dev:     n_trades={n_trades}  sharpe={dev_sharpe}")
    print(f"holdout: sharpe={r.holdout_sharpe}  evaluated={r.holdout_evaluated}")
    print(f"verdict: {r.verdict}   promotion_bar: {r.promotion_bar}")
    if (
        r.verdict == "pass"
        and r.holdout_sharpe is not None
        and dev_sharpe is not None
        and r.holdout_sharpe < 0 <= dev_sharpe
    ):
        print(
            "  ⚠ DEV PASSES BUT HOLDOUT IS NEGATIVE — classic overfitting signature "
            "(trend_pullback_v1's exact shape). NOT eligible for promotion despite the "
            "dev verdict; this needs a human decision, never an automatic promote."
        )
    if r.verdict == "fail":
        fail_note = next((n for n in r.notes if n.startswith("sharpe=")), None)
        print(f"  {fail_note}")


def main() -> None:
    # SPY excluded (WI-084 item 7): it is DEFAULT_BENCHMARK for equity/etf —
    # letting a signal fire on the same symbol the run is benchmarked against
    # would mix an actively-traded SPY position into the equity curve
    # compared to passive SPY buy-and-hold.
    equity_universe = tradable_symbols(
        load_universe(), "equity", "etf", exclude={DEFAULT_BENCHMARK["equity"]}
    )
    # BTC excluded on the same principle, for the crypto sleeve's own benchmark.
    crypto_universe = tradable_symbols(
        load_universe(), "crypto", exclude={DEFAULT_BENCHMARK["crypto"]}
    )
    rationale = RationaleRegistry()
    registry = BacktestRegistry()

    equity_signals: list[tuple[str, Signal]] = [
        ("trend_pullback_v1[depth=3%]", TrendPullbackSignal(depth=0.03)),
        ("trend_pullback_v1[depth=5%]", TrendPullbackSignal(depth=0.05)),
        ("breakout_volume_v1[N=40]", BreakoutVolumeSignal(n_sessions=40)),
        ("breakout_volume_v1[N=55]", BreakoutVolumeSignal(n_sessions=55)),
        ("oversold_reversion_v1[k=2.0]", OversoldReversionSignal(k=2.0)),
        ("oversold_reversion_v1[k=2.5]", OversoldReversionSignal(k=2.5)),
    ]
    for label, signal in equity_signals:
        _run(label, signal, equity_universe, "equity", rationale, registry)

    # WI-136/WI-145: the crypto sleeve's families. The store's crypto history
    # is CoinGecko-cross-check-capped to ~1yr, shorter than equity's ~2yr —
    # the printed data_span makes that visible on every run rather than
    # silently comparing a 1yr crypto verdict to a 2yr equity one.
    crypto_signals: list[tuple[str, Signal]] = [
        (f"crypto_breakout_v1[N={n}]", CryptoBreakoutSignal(n_sessions=n)) for n in N_SESSIONS_GRID
    ] + [
        (f"crypto_oversold_reversion_v1[k={k}]", CryptoOversoldReversionSignal(k=k))
        for k in (2.0, 2.5)
    ]
    for label, signal in crypto_signals:
        _run(label, signal, crypto_universe, "crypto", rationale, registry)


if __name__ == "__main__":
    main()
