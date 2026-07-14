import pandas as pd

from vega.backtest.market_view import MarketView
from vega.signals.breakout_volume import BreakoutVolumeSignal


def _frame(closes: list[float], volumes: list[float], symbol: str = "AAA") -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({"symbol": symbol, "date": dates, "adj_close": closes, "volume": volumes})


def test_fires_on_new_high_with_volume_spike() -> None:
    closes = [100.0] * 64 + [110.0]  # clean breakout on the last session
    volumes = [1_000_000.0] * 64 + [2_000_000.0]  # 2x the flat median
    frame = _frame(closes, volumes)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = BreakoutVolumeSignal(n_sessions=40)
    proposals = signal.scan(view, ["AAA"])
    assert len(proposals) == 1
    assert proposals[0].signal_family == "breakout_volume_v1"


def test_does_not_fire_on_new_high_without_volume_confirmation() -> None:
    closes = [100.0] * 64 + [110.0]
    volumes = [1_000_000.0] * 65  # no spike
    frame = _frame(closes, volumes)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = BreakoutVolumeSignal(n_sessions=40)
    assert signal.scan(view, ["AAA"]) == []


def test_does_not_fire_on_volume_spike_without_a_new_high() -> None:
    closes = [100.0] * 64 + [99.0]  # NOT a new high (below the flat prior range)
    volumes = [1_000_000.0] * 64 + [2_000_000.0]
    frame = _frame(closes, volumes)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = BreakoutVolumeSignal(n_sessions=40)
    assert signal.scan(view, ["AAA"]) == []


def test_insufficient_history_yields_no_proposals() -> None:
    frame = _frame([100.0] * 20, [1_000_000.0] * 20)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = BreakoutVolumeSignal(n_sessions=40)
    assert signal.scan(view, ["AAA"]) == []


def test_lookback_covers_both_the_breakout_window_and_the_volume_window() -> None:
    s40 = BreakoutVolumeSignal(n_sessions=40)
    s55 = BreakoutVolumeSignal(n_sessions=55)
    assert s40.n_sessions == 40 and s55.n_sessions == 55
    assert s40.lookback >= 60 and s55.lookback >= 55  # never shorter than either requirement


def test_is_marked_promotable() -> None:
    assert BreakoutVolumeSignal(n_sessions=40).promotable is True
