from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from vega.regime.calendar import in_macro_window, load_macro_calendar, macro_events_within
from vega.regime.regime import compute_regime


def _series(symbol: str, n: int, start_price: float, drift: float) -> pd.DataFrame:
    dates = pd.date_range("2025-06-01", periods=n, freq="D").strftime("%Y-%m-%d")
    prices = [start_price + drift * i for i in range(n)]
    return pd.DataFrame({"symbol": symbol, "date": dates, "adj_close": prices})


def _vix(level: float) -> pd.DataFrame:
    return pd.DataFrame({"date": ["2026-07-02"], "close": [level]})


def test_uptrend_calm_vix_is_risk_on() -> None:
    spy = _series("SPY", 260, 400.0, 0.5)[["date", "adj_close"]]
    universe = pd.concat([_series(s, 260, 100.0, 0.2) for s in ("AAA", "BBB", "CCC")])
    state = compute_regime(spy, _vix(13.0), universe, crypto_fg=60)
    assert state.trend == "risk_on" and state.vix_band == "calm"
    assert state.breadth_pct == 100.0 and state.composite == "risk_on"


def test_downtrend_or_crisis_vix_forces_risk_off() -> None:
    spy_down = _series("SPY", 260, 500.0, -0.5)[["date", "adj_close"]]
    universe = pd.concat([_series(s, 260, 100.0, -0.2) for s in ("AAA", "BBB")])
    assert compute_regime(spy_down, _vix(16.0), universe, crypto_fg=50).composite == "risk_off"
    spy_up = _series("SPY", 260, 400.0, 0.5)[["date", "adj_close"]]
    assert compute_regime(spy_up, _vix(35.0), universe, crypto_fg=50).composite == "risk_off"


def test_extreme_fear_degrades_to_caution() -> None:
    spy = _series("SPY", 260, 400.0, 0.5)[["date", "adj_close"]]
    universe = pd.concat([_series(s, 260, 100.0, 0.2) for s in ("AAA", "BBB")])
    state = compute_regime(spy, _vix(13.0), universe, crypto_fg=10)
    assert state.composite == "caution"


def test_insufficient_history_yields_no_breadth_and_neutral_trend() -> None:
    spy = _series("SPY", 30, 400.0, 0.5)[["date", "adj_close"]]
    universe = _series("AAA", 30, 100.0, 0.2)
    state = compute_regime(spy, _vix(16.0), universe, crypto_fg=50)
    assert state.breadth_pct is None and state.trend == "neutral"
    assert state.composite == "caution"


def test_macro_calendar_loads_and_windows(tmp_path: Path) -> None:
    art = tmp_path / "macro.csv"
    art.write_text("# test\ndate,event\n2026-07-14,CPI release\n2026-07-29,FOMC decision\n")
    events = load_macro_calendar(art)
    assert len(events) == 2
    assert in_macro_window(date(2026, 7, 13), days_before=1, path=art) is True
    assert in_macro_window(date(2026, 7, 10), days_before=1, path=art) is False
    assert [e.event for e in macro_events_within(date(2026, 7, 1), 31, path=art)] == [
        "CPI release",
        "FOMC decision",
    ]


def test_malformed_macro_artifact_fails_loudly(tmp_path: Path) -> None:
    art = tmp_path / "macro.csv"
    art.write_text("date,event\nnot-a-date,CPI release\n")
    with pytest.raises(ValueError):
        load_macro_calendar(art)


def test_committed_macro_artifact_is_valid() -> None:
    events = load_macro_calendar()  # the real committed data/calendar/macro-v1.csv
    assert len(events) == 20
    assert {e.event for e in events} == {"CPI release", "FOMC decision"}
