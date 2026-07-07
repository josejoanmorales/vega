"""Per-fold and aggregate metrics.

Equity accounting is mark-to-trade, not mark-to-market: the curve only moves
when a trade closes (fully or partially), never from unrealized swings on
still-open positions. This is a stated v1 simplification (it understates
volatility, so Sharpe is a touch optimistic) — the same class of documented
bias as the dividend-ignored and survivorship-bound caveats.

Benchmark comparison: buy-and-hold on a reference symbol (SPY/BTC), scaled by
the strategy's own exposure % — cash is assumed to earn 0% in v1 (stated).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from vega.backtest.simulate import TradeRecord

TRADING_DAYS_PER_YEAR = {"equity": 252, "etf": 252, "crypto": 365}


@dataclass(frozen=True)
class FoldMetrics:
    n_trades: int
    n_closed_exits: int
    total_pnl: float
    cagr: float | None
    sharpe: float | None
    max_drawdown: float | None
    exposure_pct: float
    benchmark_return: float | None
    benchmark_max_drawdown: float | None


def _daily_pnl(trades: list[TradeRecord], dates: list[str]) -> pd.Series:
    pnl_by_date: dict[str, float] = dict.fromkeys(dates, 0.0)
    for t in trades:
        for e in t.exits:
            d = str(e["date"])
            if d in pnl_by_date:
                pnl_by_date[d] += (float(e["price"]) - t.entry_price) * float(e["qty"])
    return pd.Series([pnl_by_date[d] for d in dates], index=dates)


def _exposure_pct(trades: list[TradeRecord], dates: list[str]) -> float:
    if not dates:
        return 0.0
    covered: set[str] = set()
    for t in trades:
        last_exit_date = max(str(e["date"]) for e in t.exits) if t.exits else t.entry_date
        covered.update(d for d in dates if t.entry_date <= d <= last_exit_date)
    return float(round(100.0 * len(covered) / len(dates), 2))


def _max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak.replace(0, pd.NA)
    return float(drawdown.min()) if not drawdown.empty else 0.0


def compute_fold_metrics(
    trades: list[TradeRecord],
    dates: list[str],
    starting_capital: float,
    asset_class: str,
    benchmark_closes: pd.Series | None = None,
) -> FoldMetrics:
    if not dates:
        return FoldMetrics(0, 0, 0.0, None, None, None, 0.0, None, None)

    daily_pnl = _daily_pnl(trades, dates)
    equity = starting_capital + daily_pnl.cumsum()
    returns = equity.pct_change().dropna()

    annualization = math.sqrt(TRADING_DAYS_PER_YEAR.get(asset_class, 252))
    sharpe = (
        float(returns.mean() / returns.std() * annualization)
        if len(returns) > 1 and returns.std() not in (0, None)
        else None
    )

    span_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days or 1
    end_equity = float(equity.iloc[-1])
    cagr = (end_equity / starting_capital) ** (365.0 / span_days) - 1.0 if end_equity > 0 else None

    exposure = _exposure_pct(trades, dates)
    benchmark_return = None
    benchmark_dd = None
    if benchmark_closes is not None and len(benchmark_closes) > 1:
        bh_return = float(benchmark_closes.iloc[-1] / benchmark_closes.iloc[0] - 1.0)
        benchmark_return = round(bh_return * exposure / 100.0, 4)  # cash=0, exposure-scaled
        benchmark_dd = _max_drawdown(benchmark_closes)

    return FoldMetrics(
        n_trades=len(trades),
        n_closed_exits=sum(len(t.exits) for t in trades),
        total_pnl=round(float(daily_pnl.sum()), 2),
        cagr=round(cagr, 4) if cagr is not None else None,
        sharpe=round(sharpe, 3) if sharpe is not None else None,
        max_drawdown=round(_max_drawdown(equity), 4),
        exposure_pct=exposure,
        benchmark_return=benchmark_return,
        benchmark_max_drawdown=round(benchmark_dd, 4) if benchmark_dd is not None else None,
    )


def aggregate_metrics(fold_metrics: list[FoldMetrics]) -> FoldMetrics:
    """Trade-weighted aggregate across folds (used for the promotion verdict)."""
    total_trades = sum(f.n_trades for f in fold_metrics)
    if total_trades == 0:
        return FoldMetrics(0, 0, 0.0, None, None, None, 0.0, None, None)
    sharpes = [f.sharpe for f in fold_metrics if f.sharpe is not None]
    dds = [f.max_drawdown for f in fold_metrics if f.max_drawdown is not None]
    bench_returns = [f.benchmark_return for f in fold_metrics if f.benchmark_return is not None]
    bench_dds = [
        f.benchmark_max_drawdown for f in fold_metrics if f.benchmark_max_drawdown is not None
    ]
    return FoldMetrics(
        n_trades=total_trades,
        n_closed_exits=sum(f.n_closed_exits for f in fold_metrics),
        total_pnl=round(sum(f.total_pnl for f in fold_metrics), 2),
        cagr=None,  # not meaningful compounded blindly across non-contiguous folds
        sharpe=round(sum(sharpes) / len(sharpes), 3) if sharpes else None,
        max_drawdown=round(min(dds), 4) if dds else None,  # worst across folds
        exposure_pct=round(sum(f.exposure_pct for f in fold_metrics) / len(fold_metrics), 2),
        benchmark_return=round(sum(bench_returns) / len(bench_returns), 4)
        if bench_returns
        else None,
        benchmark_max_drawdown=round(min(bench_dds), 4) if bench_dds else None,
    )
