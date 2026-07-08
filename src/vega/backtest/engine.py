"""Orchestrates one signal's full backtest: dev walk-forward -> verdict -> (maybe) holdout.

An unregistered run cannot promote — this is the only code path that writes
to the registry, and it always writes, regardless of verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from vega.backtest.folds import split_dev_holdout, walk_forward_folds
from vega.backtest.metrics import FoldMetrics, aggregate_metrics, compute_fold_metrics
from vega.backtest.registry import BacktestRegistry, RunRecord
from vega.backtest.signals import Signal
from vega.backtest.simulate import simulate_signal
from vega.data import snapshot
from vega.data.universe import universe_version
from vega.lifecycle.rationale import RationaleRegistry

DEFAULT_STARTING_CAPITAL = 100_000.0
MIN_TRADES_FOR_VERDICT = 30
MAX_DD_MULTIPLE = 1.5

SOURCE_BY_ASSET_CLASS = {"equity": "yfinance", "etf": "yfinance", "crypto": "binance"}
DEFAULT_BENCHMARK = {"equity": "SPY", "etf": "SPY", "crypto": "BTC"}

DOCTRINE_NOTES = (
    "dividends ignored in v1 (conservative bias, long returns understated ~1-2%/yr)",
    "survivorship-bound to the current universe artifact only",
    "ATR frozen at entry for the trailing stop (not recomputed daily)",
    "equity accounting is mark-to-trade, not mark-to-market (Sharpe is a touch optimistic)",
)


@dataclass(frozen=True)
class BacktestReport:
    record: RunRecord
    trades_by_fold: tuple[tuple[dict[str, object], ...], ...]


def _load_bars(symbols: list[str], source: str, root: Path) -> pd.DataFrame:
    con = duckdb.connect(str(root / "vega.duckdb"), read_only=True)
    try:
        placeholders = ",".join(f"'{s}'" for s in symbols)
        return con.execute(
            f"SELECT * FROM bars WHERE source = ? AND symbol IN ({placeholders})",  # noqa: S608
            [source],
        ).df()
    finally:
        con.close()


def run_backtest(
    signal: Signal,
    universe: list[str],
    asset_class: str,
    root: Path = snapshot.DATA_ROOT,
    holdout_frac: float = 0.2,
    test_size_sessions: int = 63,
    notional_usd: float = 1_000.0,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
    param_grid_size: int = 1,
    benchmark_symbol: str | None = None,
    registry: BacktestRegistry | None = None,
    rationale_registry: RationaleRegistry | None = None,
) -> BacktestReport:
    source = SOURCE_BY_ASSET_CLASS[asset_class]
    bench_symbol = benchmark_symbol or DEFAULT_BENCHMARK[asset_class]
    frame = _load_bars([*universe, bench_symbol], source, root)
    if frame.empty:
        raise ValueError(f"no {source} bars found for the requested universe/benchmark")

    all_dates = sorted(frame["date"].unique())
    dev_dates, holdout_dates = split_dev_holdout(all_dates, holdout_frac)
    folds = walk_forward_folds(dev_dates, test_size_sessions)

    def _bench_series(dates: list[str]) -> pd.Series:
        sub = frame[(frame["symbol"] == bench_symbol) & (frame["date"].isin(dates))]
        return sub.sort_values("date").set_index("date")["adj_close"]

    fold_metrics: list[FoldMetrics] = []
    fold_metrics_payload: list[dict[str, Any]] = []
    trades_by_fold: list[tuple[dict[str, object], ...]] = []
    for fold in folds:
        fold_frame = frame[frame["date"] <= fold.test_dates[-1]]
        trades = simulate_signal(
            fold_frame, list(fold.test_dates), signal, universe, asset_class, notional_usd
        )
        fm = compute_fold_metrics(
            trades,
            list(fold.test_dates),
            starting_capital,
            asset_class,
            _bench_series(list(fold.test_dates)),
        )
        fold_metrics.append(fm)
        fold_metrics_payload.append(
            {"test_span": (fold.test_dates[0], fold.test_dates[-1]), **asdict(fm)}
        )
        trades_by_fold.append(tuple(asdict(t) for t in trades))

    agg = aggregate_metrics(fold_metrics)
    reg = registry or BacktestRegistry()

    notes = list(DOCTRINE_NOTES)
    promotion_bar = None
    holdout_evaluated = False

    if not signal.promotable:
        verdict = "non_promotable_placeholder"
        notes.append("signal.promotable=False — smoke/fixture signal, never eligible to promote")
    elif agg.n_trades < MIN_TRADES_FOR_VERDICT:
        verdict = "insufficient_sample"
        notes.append(
            f"only {agg.n_trades} closed trades across folds (need >= {MIN_TRADES_FOR_VERDICT})"
        )
    else:
        prior_grid = reg.cumulative_grid_points(signal.family)
        promotion_bar = reg.promotion_bar(prior_grid + param_grid_size)
        sharpe_ok = agg.sharpe is not None and agg.sharpe >= promotion_bar
        # fail-closed: if the benchmark drawdown can't be computed, the cap cannot be
        # validated, so it does NOT pass — except a strategy with zero drawdown, which
        # trivially satisfies any cap.
        strat_dd = agg.max_drawdown
        bench_dd = agg.benchmark_max_drawdown
        if strat_dd is None or strat_dd == 0:
            dd_ok = True
        elif bench_dd is None or bench_dd == 0:
            dd_ok = False
            notes.append("benchmark drawdown unavailable/zero — drawdown cap fails closed")
        else:
            dd_ok = abs(strat_dd) <= MAX_DD_MULTIPLE * abs(bench_dd)
        if sharpe_ok and dd_ok:
            verdict = "pass"
            holdout_evaluated = True
        else:
            verdict = "fail"
            notes.append(
                f"sharpe={agg.sharpe} vs bar={promotion_bar}, dd_ok={dd_ok} "
                f"(dd={agg.max_drawdown}, benchmark_dd={agg.benchmark_max_drawdown})"
            )

    if holdout_evaluated:
        holdout_frame = frame[frame["date"] <= holdout_dates[-1]]
        holdout_trades = simulate_signal(
            holdout_frame, holdout_dates, signal, universe, asset_class, notional_usd
        )
        holdout_fm = compute_fold_metrics(
            holdout_trades,
            holdout_dates,
            starting_capital,
            asset_class,
            _bench_series(holdout_dates),
        )
        fold_metrics_payload.append(
            {
                "test_span": (holdout_dates[0], holdout_dates[-1]),
                "is_holdout": True,
                **asdict(holdout_fm),
            }
        )
        trades_by_fold.append(tuple(asdict(t) for t in holdout_trades))

    touches_after = reg.holdout_touch_count(signal.family) + (1 if holdout_evaluated else 0)
    if touches_after > 1:
        notes.append(f"WARNING: holdout touched {touches_after} times for this family")

    record = reg.record_run(
        signal_family=signal.family,
        signal_version=signal.version,
        param_grid_size=param_grid_size,
        universe_version=universe_version(root / "universe"),
        data_span=(all_dates[0], all_dates[-1]),
        n_folds=len(folds),
        fold_metrics=fold_metrics_payload,
        aggregate_metrics=asdict(agg),
        verdict=verdict,
        holdout_evaluated=holdout_evaluated,
        promotion_bar=promotion_bar,
        notes=notes,
        rationale_registry=rationale_registry,
    )
    return BacktestReport(record=record, trades_by_fold=tuple(trades_by_fold))
