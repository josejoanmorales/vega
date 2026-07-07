import pandas as pd

from vega.backtest.metrics import aggregate_metrics, compute_fold_metrics
from vega.backtest.simulate import TradeRecord


def _trade(
    entry_date: str, exit_date: str, entry: float, exit_price: float, qty: float
) -> TradeRecord:
    return TradeRecord(
        symbol="TEST",
        asset_class="equity",
        signal_family="f",
        signal_version="1",
        entry_date=entry_date,
        entry_price=entry,
        initial_qty=qty,
        stop_price=entry - 5,
        initial_r=5.0,
        thesis="t",
        confidence=0.5,
        invalidation="x",
        exits=({"date": exit_date, "qty": qty, "price": exit_price, "reason": "stop"},),
        realized_pnl=round((exit_price - entry) * qty, 2),
        r_multiple=round((exit_price - entry) / 5.0, 4),
        unresolved_at_end=False,
    )


def _dates(n: int) -> list[str]:
    return [f"2026-03-{d:02d}" for d in range(1, n + 1)]


def test_exposure_and_pnl_from_a_single_winning_trade() -> None:
    dates = _dates(10)
    trades = [_trade("2026-03-02", "2026-03-05", 100.0, 110.0, 10.0)]
    m = compute_fold_metrics(trades, dates, starting_capital=10_000.0, asset_class="equity")
    assert m.n_trades == 1
    assert m.total_pnl == 100.0  # (110-100)*10
    assert 0 < m.exposure_pct <= 100
    assert m.max_drawdown is not None and m.max_drawdown <= 0


def test_no_trades_yields_zeroed_metrics_not_a_crash() -> None:
    m = compute_fold_metrics([], _dates(5), starting_capital=10_000.0, asset_class="equity")
    assert m.n_trades == 0 and m.total_pnl == 0.0 and m.sharpe is None


def test_benchmark_return_is_scaled_by_exposure() -> None:
    dates = _dates(10)
    trades = [_trade("2026-03-02", "2026-03-09", 100.0, 105.0, 1.0)]
    bench = pd.Series([100.0] * 5 + [110.0] * 5, index=dates)  # 10% buy-hold return
    m = compute_fold_metrics(trades, dates, 10_000.0, "equity", bench)
    assert m.benchmark_return is not None
    # exposure < 100% -> scaled benchmark return must be strictly less than the raw 10%
    assert m.benchmark_return < 0.10


def test_aggregate_sums_trades_and_pnl_across_folds() -> None:
    dates = _dates(10)
    f1 = compute_fold_metrics(
        [_trade("2026-03-02", "2026-03-03", 100.0, 105.0, 1.0)], dates, 10_000.0, "equity"
    )
    f2 = compute_fold_metrics(
        [_trade("2026-03-04", "2026-03-05", 100.0, 95.0, 1.0)], dates, 10_000.0, "equity"
    )
    agg = aggregate_metrics([f1, f2])
    assert agg.n_trades == 2
    assert agg.total_pnl == 0.0  # +5 and -5 net to zero


def test_aggregate_with_zero_trades_never_crashes() -> None:
    empty = compute_fold_metrics([], _dates(5), 10_000.0, "equity")
    agg = aggregate_metrics([empty, empty])
    assert agg.n_trades == 0 and agg.sharpe is None
