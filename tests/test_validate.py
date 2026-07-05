import pandas as pd

from vega.data.validate import cross_check


def _bars(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": s,
                "date": d,
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "adj_close": c,
                "volume": 1e6,
                "source": "test",
            }
            for s, d, c in rows
        ]
    )


def test_within_tolerance_passes() -> None:
    res = cross_check(
        _bars([("AAPL", "2026-07-02", 100.0)]), _bars([("AAPL", "2026-07-02", 100.3)])
    )
    assert len(res.clean) == 1 and res.quarantine.empty


def test_breach_quarantines_with_reason() -> None:
    res = cross_check(
        _bars([("AAPL", "2026-07-02", 100.0)]), _bars([("AAPL", "2026-07-02", 102.0)])
    )
    assert res.clean.empty and len(res.quarantine) == 1
    assert "diverges" in res.quarantine.iloc[0]["reason"]


def test_missing_from_check_source_quarantines() -> None:
    res = cross_check(
        _bars([("AAPL", "2026-07-02", 100.0), ("MSFT", "2026-07-02", 500.0)]),
        _bars([("AAPL", "2026-07-02", 100.0)]),
    )
    assert list(res.clean["symbol"]) == ["AAPL"]
    assert res.quarantine.iloc[0]["reason"] == "missing from cross-check source"


def test_empty_primary_is_safe() -> None:
    res = cross_check(_bars([]).head(0), _bars([("AAPL", "2026-07-02", 100.0)]))
    assert res.clean.empty and res.quarantine.empty
