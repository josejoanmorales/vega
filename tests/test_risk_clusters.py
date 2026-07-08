import pandas as pd
import pytest

from vega.risk.clusters import classify, contaminates_equity_beta, spy_correlation


def test_classify_known_rates_and_commodities() -> None:
    assert classify("TLT", "etf") == "rates"
    assert classify("IEF", "etf") == "rates"
    assert classify("GLD", "etf") == "commodities"
    assert classify("XME", "etf") == "commodities"


def test_classify_crypto_always_crypto_beta() -> None:
    assert classify("BTC", "crypto") == "crypto_beta"


def test_classify_default_is_us_equity_beta() -> None:
    assert classify("AAPL", "equity") == "us_equity_beta"
    assert classify("HYG", "etf") == "us_equity_beta"  # not explicitly named -> stated default


def _returns_frame(
    symbol: str, bench_prices: list[float], target_prices: list[float]
) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(bench_prices), freq="D").strftime("%Y-%m-%d")
    rows = [
        {"symbol": "SPY", "date": d, "adj_close": p}
        for d, p in zip(dates, bench_prices, strict=True)
    ]
    rows += [
        {"symbol": symbol, "date": d, "adj_close": p}
        for d, p in zip(dates, target_prices, strict=True)
    ]
    return pd.DataFrame(rows)


def test_correlation_none_with_insufficient_history() -> None:
    frame = _returns_frame("BTC", [100.0] * 10, [50.0] * 10)
    assert spy_correlation(frame, "BTC", as_of="2026-01-10") is None


def test_spyless_frame_raises_instead_of_silently_not_contaminating() -> None:
    n = 95
    dates = pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    frame = pd.DataFrame({"symbol": ["BTC"] * n, "date": dates, "adj_close": range(100, 100 + n)})
    with pytest.raises(ValueError, match="no SPY rows"):
        spy_correlation(frame, "BTC", as_of=frame["date"].max())


def test_calendar_mismatch_crypto_7day_vs_spy_5day_still_measures() -> None:
    # crypto trades every day; SPY only weekdays. Merge-first windowing must
    # still find `window` shared sessions instead of under-firing to None.
    all_days = pd.date_range("2025-12-01", periods=200, freq="D")
    rows = []
    for i, d in enumerate(all_days):
        # varied but deterministic path; SPY is exactly half of BTC on the
        # weekdays it trades, so shared-session returns are identical
        price = 100.0 + i * 0.5 + (i % 7) * 3.0
        rows.append({"symbol": "BTC", "date": d.strftime("%Y-%m-%d"), "adj_close": price})
        if d.weekday() < 5:
            rows.append({"symbol": "SPY", "date": d.strftime("%Y-%m-%d"), "adj_close": price / 2})
    frame = pd.DataFrame(rows)
    corr = spy_correlation(frame, "BTC", as_of=all_days[-1].strftime("%Y-%m-%d"))
    assert corr is not None and corr > 0.9  # both monotonic up on shared sessions


def test_correlation_high_when_series_move_together() -> None:
    n = 95
    prices = [100.0 + i * 0.5 for i in range(n)]
    frame = _returns_frame("BTC", prices, [p * 2 for p in prices])  # perfectly co-moving
    corr = spy_correlation(frame, "BTC", as_of=frame["date"].max())
    assert corr is not None and corr > 0.99


def test_correlation_none_when_a_series_is_flat() -> None:
    n = 95
    frame = _returns_frame("BTC", [100.0 + i * 0.1 for i in range(n)], [50.0] * n)
    corr = spy_correlation(frame, "BTC", as_of=frame["date"].max())
    assert corr is None  # zero-variance series -> undefined correlation, not zero


@pytest.mark.parametrize(
    ("corr", "expected"), [(None, False), (0.3, False), (0.5, False), (0.51, True), (0.9, True)]
)
def test_contamination_threshold_and_unmeasurable_default(
    corr: float | None, expected: bool
) -> None:
    assert contaminates_equity_beta(corr) is expected


def test_cluster_frozensets_stay_reconciled_with_the_committed_universe() -> None:
    """Guard until cluster membership migrates onto the universe artifact
    (universe-v2, parked): every hardcoded cluster symbol must exist in the
    current universe, so a refresh can't silently orphan the frozensets."""
    from vega.data.universe import load_universe
    from vega.risk.clusters import COMMODITIES, RATES

    universe_symbols = {e.symbol for e in load_universe()}
    assert RATES <= universe_symbols
    assert COMMODITIES <= universe_symbols
