from pathlib import Path

import duckdb
import pandas as pd

from vega.backtest.engine import run_backtest
from vega.backtest.market_view import MarketView
from vega.backtest.registry import BacktestRegistry
from vega.backtest.signals import EntryProposal
from vega.lifecycle.rationale import NullRationaleRegistry

_NULL_RATIONALE = NullRationaleRegistry()


def _make_store(tmp_path: Path, frame: pd.DataFrame) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    con = duckdb.connect(str(root / "vega.duckdb"))
    con.execute("CREATE TABLE bars AS SELECT * FROM frame")
    con.close()
    return root


def _trend_frame(symbol: str, n: int, bench_symbol: str = "SPY") -> pd.DataFrame:
    """Strictly increasing daily closes, narrow intraday range (stop never touched)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    rows = []
    for sym in (symbol, bench_symbol):
        price = 100.0
        for d in dates:
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "open": price - 0.5,
                    "high": price + 0.5,
                    "low": price - 1.0,
                    "close": price,
                    "adj_close": price,
                    "volume": 5_000_000.0,
                    "source": "yfinance",
                }
            )
            price += 1.0
    return pd.DataFrame(rows)


class _AlwaysWinSignal:
    family = "always_win_test"
    version = "1"
    promotable = True

    def scan(self, view: MarketView, universe: list[str]) -> list[EntryProposal]:
        return [
            EntryProposal(
                symbol=s,
                signal_family=self.family,
                signal_version=self.version,
                thesis="fixture",
                confidence=0.9,
                invalidation="fixture",
                time_stop_days=1,
            )
            for s in universe
        ]


class _NonPromotableSignal(_AlwaysWinSignal):
    family = "non_promotable_test"
    promotable = False


def test_insufficient_sample_when_too_few_trades(tmp_path: Path) -> None:
    frame = _trend_frame("AAA", n=20)  # far short of a full 63-session fold
    root = _make_store(tmp_path, frame)
    report = run_backtest(
        signal=_AlwaysWinSignal(),
        universe=["AAA"],
        asset_class="equity",
        root=root,
        test_size_sessions=63,
        registry=BacktestRegistry(tmp_path / "reg.jsonl"),
        rationale_registry=_NULL_RATIONALE,
    )
    assert report.record.verdict == "insufficient_sample"
    assert report.record.holdout_evaluated is False


def test_non_promotable_signal_never_gets_a_pass_verdict(tmp_path: Path) -> None:
    frame = _trend_frame("AAA", n=200)
    root = _make_store(tmp_path, frame)
    report = run_backtest(
        signal=_NonPromotableSignal(),
        universe=["AAA"],
        asset_class="equity",
        root=root,
        test_size_sessions=30,
        registry=BacktestRegistry(tmp_path / "reg.jsonl"),
        rationale_registry=_NULL_RATIONALE,
    )
    assert report.record.verdict == "non_promotable_placeholder"
    assert report.record.holdout_evaluated is False


def test_holdout_is_never_touched_unless_dev_verdict_is_pass(tmp_path: Path) -> None:
    frame = _trend_frame("AAA", n=20)
    root = _make_store(tmp_path, frame)
    reg = BacktestRegistry(tmp_path / "reg.jsonl")
    report = run_backtest(
        signal=_AlwaysWinSignal(),
        universe=["AAA"],
        asset_class="equity",
        root=root,
        test_size_sessions=63,
        registry=reg,
        rationale_registry=_NULL_RATIONALE,
    )
    assert reg.holdout_touch_count("always_win_test") == 0
    assert not any("is_holdout" in fold for fold in report.record.fold_metrics)


def test_every_run_is_recorded_regardless_of_verdict(tmp_path: Path) -> None:
    frame = _trend_frame("AAA", n=20)
    root = _make_store(tmp_path, frame)
    reg = BacktestRegistry(tmp_path / "reg.jsonl")
    run_backtest(
        signal=_AlwaysWinSignal(),
        universe=["AAA"],
        asset_class="equity",
        root=root,
        test_size_sessions=63,
        registry=reg,
        rationale_registry=_NULL_RATIONALE,
    )
    assert len(reg.runs()) == 1
    # tmp root has no universe artifact — provenance must say so, never guess a version
    assert reg.runs()[0]["universe_version"] == "unknown"


def test_a_clear_winning_signal_with_enough_trades_passes_and_touches_holdout_once(
    tmp_path: Path,
) -> None:
    frame = _trend_frame("AAA", n=170)  # 2 dev folds @63 + a holdout slice
    root = _make_store(tmp_path, frame)
    reg = BacktestRegistry(tmp_path / "reg.jsonl")
    report = run_backtest(
        signal=_AlwaysWinSignal(),
        universe=["AAA"],
        asset_class="equity",
        root=root,
        test_size_sessions=63,
        registry=reg,
        rationale_registry=_NULL_RATIONALE,
    )
    assert report.record.n_folds == 2
    assert report.record.aggregate_metrics["n_trades"] >= 30
    assert report.record.verdict == "pass"
    assert report.record.holdout_evaluated is True
    assert reg.holdout_touch_count("always_win_test") == 1
    assert any(fold.get("is_holdout") for fold in report.record.fold_metrics)

    # a second run for the SAME family must raise the bar (more cumulative trials)
    second = run_backtest(
        signal=_AlwaysWinSignal(),
        universe=["AAA"],
        asset_class="equity",
        root=root,
        test_size_sessions=63,
        registry=reg,
        rationale_registry=_NULL_RATIONALE,
    )
    assert second.record.promotion_bar is not None and report.record.promotion_bar is not None
    assert second.record.promotion_bar > report.record.promotion_bar


def test_benchmark_symbol_is_never_traded_even_if_the_caller_passes_it(tmp_path: Path) -> None:
    """WI-084 item 7 defense-in-depth: run_backtest itself excludes
    bench_symbol from the tradable set, even if a caller forgets to filter it
    out of `universe` first (data.universe.tradable_symbols is the primary
    fix at the __main__ callers; this is the belt for the highest-stakes
    module in the codebase). _AlwaysWinSignal fires on every symbol handed to
    it, so if SPY reached simulate_signal it would generate SPY trades too."""
    frame = _trend_frame("AAA", n=170, bench_symbol="SPY")
    root = _make_store(tmp_path, frame)
    report = run_backtest(
        signal=_AlwaysWinSignal(),
        universe=["AAA", "SPY"],  # caller did NOT pre-exclude the benchmark
        asset_class="equity",
        root=root,
        test_size_sessions=63,
        registry=BacktestRegistry(tmp_path / "reg.jsonl"),
        rationale_registry=_NULL_RATIONALE,
    )
    traded_symbols = {trade["symbol"] for fold in report.trades_by_fold for trade in fold}
    assert traded_symbols == {"AAA"}
    assert "SPY" not in traded_symbols
