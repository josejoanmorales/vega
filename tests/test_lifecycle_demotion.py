import pandas as pd

from vega.lifecycle.demotion import LiveTrade, check_auto_demotion, confidence_band


def _run(sharpes: list[float], holdout_sharpe: float | None = None) -> dict[str, object]:
    folds = [{"sharpe": s} for s in sharpes]
    if holdout_sharpe is not None:
        folds.append({"sharpe": holdout_sharpe, "is_holdout": True})
    return {"run_id": "r1", "fold_metrics": folds}


def _dates(n: int) -> list[str]:
    return list(pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d"))


def _trades(n: int, entry: float, exit_: float, asset_class: str = "equity") -> list[LiveTrade]:
    d = _dates(n + 1)
    return [
        LiveTrade(
            symbol="AAPL",
            asset_class=asset_class,
            entry_date=d[i],
            entry_price=entry,
            exit_date=d[i + 1],
            exit_price=exit_,
            qty=10.0,
            stop_price=entry - 5.0,
        )
        for i in range(n)
    ]


def test_confidence_band_excludes_holdout() -> None:
    run = _run([1.0, 2.0], holdout_sharpe=99.0)
    assert confidence_band(run) == (1.0, 2.0)


def test_confidence_band_none_when_no_fold_sharpe_recorded() -> None:
    assert confidence_band({"run_id": "r1", "fold_metrics": [{"sharpe": None}]}) is None


def test_insufficient_sample_below_min_trades() -> None:
    verdict = check_auto_demotion(_trades(5, 100.0, 110.0), _run([1.0, 2.0]))
    assert verdict.should_demote is False
    assert "insufficient_sample" in verdict.reason
    assert verdict.live_sharpe is None


def test_no_band_available_never_demotes() -> None:
    verdict = check_auto_demotion(_trades(35, 100.0, 110.0), _run([]))
    assert verdict.should_demote is False and verdict.band is None


def test_strong_winning_trades_stay_within_band() -> None:
    verdict = check_auto_demotion(_trades(35, 100.0, 110.0), _run([0.1, 0.2]))
    assert verdict.should_demote is False
    assert verdict.live_sharpe is not None and verdict.band == (0.1, 0.2)


def test_consistent_losses_breach_the_band_floor() -> None:
    # alternating tiny win/loss around a losing drift -> negative, low-variance-ish Sharpe
    d = _dates(36)
    trades = [
        LiveTrade(
            "AAPL",
            "equity",
            d[i],
            100.0,
            d[i + 1],
            100.0 - (1.0 if i % 2 == 0 else 0.5),
            10.0,
            95.0,
        )
        for i in range(35)
    ]
    verdict = check_auto_demotion(trades, _run([1.5, 2.0]))  # a strong backtest band
    assert verdict.should_demote is True
    assert verdict.live_sharpe is not None and verdict.live_sharpe < verdict.band[0]  # type: ignore[index]
    assert "below band floor" in verdict.reason


def test_min_trades_threshold_is_configurable() -> None:
    verdict = check_auto_demotion(_trades(5, 100.0, 110.0), _run([0.1, 0.2]), min_trades=3)
    assert verdict.n_trades == 5 and "insufficient_sample" not in verdict.reason
