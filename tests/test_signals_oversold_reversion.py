import pandas as pd

from vega.backtest.market_view import MarketView
from vega.signals.oversold_reversion import OversoldReversionSignal


def _ohlc_frame(closes: list[float], shocked: set[int], symbol: str = "AAA") -> pd.DataFrame:
    """Raw OHLC around each close; wider (more volatile) range on shocked indices."""
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D").strftime("%Y-%m-%d")
    rows = []
    for i, (d, c) in enumerate(zip(dates, closes, strict=True)):
        spread = 5.0 if i in shocked else 2.0
        rows.append(
            {
                "symbol": symbol,
                "date": d,
                "adj_close": c,
                "close": c,
                "high": c + spread,
                "low": c - spread,
            }
        )
    return pd.DataFrame(rows)


def _steep_uptrend_then_shock(drop_total: float) -> list[float]:
    base = [100.0 + i * 1.0 for i in range(100)]  # steep rising trend, SMA100 lags well behind
    peak = base[-1]
    return base + [peak - drop_total / 3, peak - 2 * drop_total / 3, peak - drop_total]


def test_fires_on_a_large_shock_while_still_above_sma100() -> None:
    closes = _steep_uptrend_then_shock(drop_total=39.0)
    frame = _ohlc_frame(closes, shocked={100, 101, 102})
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = OversoldReversionSignal(k=2.0)
    proposals = signal.scan(view, ["AAA"])
    assert len(proposals) == 1
    assert proposals[0].signal_family == "oversold_reversion_v1"
    assert proposals[0].time_stop_days == 7  # exit override applied
    assert proposals[0].profit_take_half_at_r == 1.5


def test_higher_k_is_a_stricter_threshold() -> None:
    closes = _steep_uptrend_then_shock(drop_total=39.0)
    frame = _ohlc_frame(closes, shocked={100, 101, 102})
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    loose = OversoldReversionSignal(k=2.0).scan(view, ["AAA"])
    strict = OversoldReversionSignal(k=100.0).scan(view, ["AAA"])
    assert len(loose) == 1
    assert strict == []


def test_does_not_fire_on_a_shallow_move() -> None:
    closes = _steep_uptrend_then_shock(drop_total=1.5)  # far too small a move
    frame = _ohlc_frame(closes, shocked=set())
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = OversoldReversionSignal(k=2.0)
    assert signal.scan(view, ["AAA"]) == []


def test_does_not_fire_below_sma100() -> None:
    base = [200.0 - i * 1.0 for i in range(100)]  # declining trend
    closes = base + [base[-1] - 13.0, base[-1] - 26.0, base[-1] - 39.0]  # same shock, no uptrend
    frame = _ohlc_frame(closes, shocked={100, 101, 102})
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = OversoldReversionSignal(k=2.0)
    assert signal.scan(view, ["AAA"]) == []


def test_insufficient_history_yields_no_proposals() -> None:
    frame = _ohlc_frame([100.0 + i for i in range(50)], shocked=set())
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = OversoldReversionSignal(k=2.0)
    assert signal.scan(view, ["AAA"]) == []


def test_is_marked_promotable() -> None:
    assert OversoldReversionSignal(k=2.0).promotable is True
