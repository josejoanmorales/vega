import math
from pathlib import Path

from vega.backtest.registry import BASE_SHARPE_BAR, LOG_SLOPE, BacktestRegistry


def _record(reg: BacktestRegistry, family: str, grid_size: int, holdout: bool) -> None:
    reg.record_run(
        signal_family=family,
        signal_version="1",
        param_grid_size=grid_size,
        universe_version="v1",
        data_span=("2026-01-01", "2026-06-01"),
        n_folds=2,
        fold_metrics=[{"sharpe": 1.0}],
        aggregate_metrics={"sharpe": 1.0},
        verdict="pass" if holdout else "fail",
        holdout_evaluated=holdout,
        promotion_bar=0.9,
        notes=["test"],
    )


def test_an_unregistered_run_simply_does_not_exist(tmp_path: Path) -> None:
    reg = BacktestRegistry(tmp_path / "registry.jsonl")
    assert reg.runs() == []
    _record(reg, "fam_a", grid_size=5, holdout=False)
    assert len(reg.runs()) == 1
    assert reg.runs()[0]["signal_family"] == "fam_a"


def test_cumulative_grid_points_sums_prior_runs_of_the_same_family(tmp_path: Path) -> None:
    reg = BacktestRegistry(tmp_path / "registry.jsonl")
    _record(reg, "fam_a", grid_size=5, holdout=False)
    _record(reg, "fam_a", grid_size=7, holdout=False)
    _record(reg, "fam_b", grid_size=100, holdout=False)  # a different family must not leak in
    assert reg.cumulative_grid_points("fam_a") == 12
    assert reg.cumulative_grid_points("fam_b") == 100


def test_holdout_touch_count_only_counts_evaluated_runs(tmp_path: Path) -> None:
    reg = BacktestRegistry(tmp_path / "registry.jsonl")
    _record(reg, "fam_a", grid_size=1, holdout=False)
    assert reg.holdout_touch_count("fam_a") == 0
    _record(reg, "fam_a", grid_size=1, holdout=True)
    assert reg.holdout_touch_count("fam_a") == 1
    _record(reg, "fam_a", grid_size=1, holdout=True)
    assert reg.holdout_touch_count("fam_a") == 2  # the "touched more than once" flag condition


def test_promotion_bar_rises_with_cumulative_trials(tmp_path: Path) -> None:
    reg = BacktestRegistry(tmp_path / "registry.jsonl")
    bar_1 = reg.promotion_bar(1)
    bar_100 = reg.promotion_bar(100)
    assert bar_1 == BASE_SHARPE_BAR  # log10(1) == 0
    assert bar_100 == round(BASE_SHARPE_BAR + LOG_SLOPE * math.log10(100), 4)
    assert bar_100 > bar_1  # more trials tried -> harder bar to clear


def test_promotion_bar_never_crashes_on_zero_or_negative_input(tmp_path: Path) -> None:
    reg = BacktestRegistry(tmp_path / "registry.jsonl")
    assert reg.promotion_bar(0) == reg.promotion_bar(1)  # floored at 1 trial
