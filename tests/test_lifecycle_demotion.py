import pandas as pd
import pytest

from vega.lifecycle.demotion import LiveTrade, check_auto_demotion, confidence_band


def _run(sharpes: list[float], holdout_sharpe: float | None = None) -> dict[str, object]:
    folds = [{"sharpe": s} for s in sharpes]
    if holdout_sharpe is not None:
        folds.append({"sharpe": holdout_sharpe, "is_holdout": True})
    return {"run_id": "r1", "fold_metrics": folds}


def _cal(n: int) -> list[str]:
    """A full trading-session calendar covering the trades (every session, flat days incl.)."""
    return list(pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d"))


def _trades(n: int, entry: float, exit_: float, asset_class: str = "equity") -> list[LiveTrade]:
    d = _cal(n + 1)
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
    assert confidence_band(_run([1.0, 2.0], holdout_sharpe=99.0)) == (1.0, 2.0)


def test_confidence_band_none_when_no_fold_sharpe() -> None:
    assert confidence_band({"run_id": "r1", "fold_metrics": [{"sharpe": None}]}) is None


def test_insufficient_sample_below_min_trades() -> None:
    verdict = check_auto_demotion(
        _trades(5, 100.0, 110.0), _run([1.0, 2.0]), session_dates=_cal(40)
    )
    assert verdict.should_demote is False and "insufficient_sample" in verdict.reason
    assert verdict.live_sharpe is None


def test_no_band_available_never_demotes() -> None:
    verdict = check_auto_demotion(_trades(35, 100.0, 110.0), _run([]), session_dates=_cal(60))
    assert verdict.should_demote is False and verdict.band is None


def test_strong_winning_trades_stay_within_band() -> None:
    verdict = check_auto_demotion(
        _trades(35, 100.0, 110.0), _run([0.1, 0.2]), session_dates=_cal(60)
    )
    assert verdict.should_demote is False
    assert verdict.live_sharpe is not None and verdict.band == (0.1, 0.2)


def test_consistent_losses_breach_the_band_floor() -> None:
    d = _cal(60)
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
    verdict = check_auto_demotion(trades, _run([1.5, 2.0]), session_dates=d)
    assert verdict.should_demote is True
    assert verdict.live_sharpe is not None and verdict.live_sharpe < verdict.band[0]  # type: ignore[index]
    assert "below band floor" in verdict.reason


def test_session_grid_must_cover_the_trade_window() -> None:
    # a grid that stops short of the trades is a caller bug — raise, don't silently mis-sample
    with pytest.raises(ValueError, match="does not cover"):
        check_auto_demotion(_trades(35, 100.0, 110.0), _run([0.1]), session_dates=_cal(5))


def test_full_grid_sampling_differs_from_event_only_sampling() -> None:
    # The comparability fix: Sharpe over the FULL session grid (flat days included)
    # is materially different from Sharpe over only trade-event days. This asserts the
    # grid is actually used — a sparse grid and a dense grid give different live Sharpe.
    trades = _trades(35, 100.0, 101.0)
    dense = check_auto_demotion(trades, _run([0.1]), session_dates=_cal(200))
    tight = check_auto_demotion(trades, _run([0.1]), session_dates=_cal(36))
    assert dense.live_sharpe != tight.live_sharpe


def test_mixed_asset_classes_raise() -> None:
    mixed = _trades(20, 100.0, 110.0, "equity") + _trades(20, 100.0, 110.0, "crypto")
    with pytest.raises(ValueError, match="single asset class"):
        check_auto_demotion(mixed, _run([0.1]), session_dates=_cal(60))
