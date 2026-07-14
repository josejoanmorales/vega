import pandas as pd

from vega.backtest.market_view import MarketView
from vega.signals.trend_pullback import TrendPullbackSignal


def _frame(closes: list[float], symbol: str = "AAA") -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({"symbol": symbol, "date": dates, "adj_close": closes})


def _uptrend_then(extra: list[float]) -> list[float]:
    base = [100.0 + i * 0.5 for i in range(60)]  # steady rising trend, 60 sessions
    return base + extra


def test_fires_on_a_clear_pullback_and_first_up_day() -> None:
    peak = 100.0 + 59 * 0.5  # 129.5
    closes = _uptrend_then([peak - 12.0, peak - 11.0])  # ~9.3% drop, then first up-day
    frame = _frame(closes)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = TrendPullbackSignal(depth=0.05)
    proposals = signal.scan(view, ["AAA"])
    assert len(proposals) == 1
    assert proposals[0].symbol == "AAA" and proposals[0].signal_family == "trend_pullback_v1"


def test_does_not_fire_when_pullback_is_too_shallow() -> None:
    peak = 100.0 + 59 * 0.5
    closes = _uptrend_then([peak - 2.0, peak - 1.5])  # ~1.2% drop, first up-day
    frame = _frame(closes)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = TrendPullbackSignal(depth=0.03)
    assert signal.scan(view, ["AAA"]) == []


def test_does_not_fire_on_a_second_down_day() -> None:
    peak = 100.0 + 59 * 0.5
    closes = _uptrend_then([peak - 8.0, peak - 10.0])  # still falling, not the first up-day
    frame = _frame(closes)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = TrendPullbackSignal(depth=0.03)
    assert signal.scan(view, ["AAA"]) == []


def test_does_not_fire_outside_an_uptrend() -> None:
    # a flat/declining series never satisfies close > SMA50 with SMA50 rising
    closes = [100.0 - i * 0.3 for i in range(60)] + [90.0, 91.0]
    frame = _frame(closes)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = TrendPullbackSignal(depth=0.03)
    assert signal.scan(view, ["AAA"]) == []


def test_insufficient_history_yields_no_proposals() -> None:
    frame = _frame([100.0 + i for i in range(30)])  # far short of the 61-session requirement
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = TrendPullbackSignal(depth=0.03)
    assert signal.scan(view, ["AAA"]) == []


def test_is_marked_promotable() -> None:
    assert TrendPullbackSignal(depth=0.03).promotable is True
