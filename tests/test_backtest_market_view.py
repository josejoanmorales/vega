import pandas as pd

from vega.backtest.market_view import MarketView


def _frame() -> pd.DataFrame:
    dates = [f"2026-01-{d:02d}" for d in range(1, 11)]
    return pd.DataFrame(
        {"symbol": ["AAA"] * 10, "date": dates, "adj_close": [float(i) for i in range(10)]}
    )


def test_bars_never_returns_rows_past_as_of() -> None:
    view = MarketView(_frame(), as_of="2026-01-05")
    bars = view.bars("AAA")
    assert bars["date"].max() == "2026-01-05"
    assert len(bars) == 5  # Jan 01..05 inclusive


def test_lookback_truncates_from_the_tail() -> None:
    view = MarketView(_frame(), as_of="2026-01-08")
    bars = view.bars("AAA", lookback=3)
    assert list(bars["date"]) == ["2026-01-06", "2026-01-07", "2026-01-08"]


def test_with_as_of_never_exposes_a_wider_frame() -> None:
    view = MarketView(_frame(), as_of="2026-01-03")
    later = view.with_as_of("2026-01-09")
    assert later.bars("AAA")["date"].max() == "2026-01-09"
    # the original view is untouched — no shared mutable state leaks dates forward
    assert view.bars("AAA")["date"].max() == "2026-01-03"


def test_symbols_excludes_names_with_no_history_yet() -> None:
    frame = pd.concat(
        [
            _frame(),
            pd.DataFrame({"symbol": ["BBB"], "date": ["2026-01-09"], "adj_close": [1.0]}),
        ]
    )
    view = MarketView(frame, as_of="2026-01-05")
    assert view.symbols() == ["AAA"]
