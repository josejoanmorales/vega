from datetime import UTC, datetime, timedelta

import pandas as pd

from vega.data.sources import alpaca_src, binance_src, coingecko_src, yfinance_src
from vega.data.types import BAR_COLUMNS


def _utc_midnight_ms(days_ago: int) -> int:
    day = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((day - timedelta(days=days_ago)).timestamp() * 1000)


def test_binance_normalize_excludes_forming_session() -> None:
    raw = {
        "BTC": [
            [_utc_midnight_ms(2), "50000", "51000", "49500", "50500", "1234.5", 0, 0, 0, 0, 0, 0],
            [_utc_midnight_ms(0), "50500", "50900", "50100", "50700", "222.2", 0, 0, 0, 0, 0, 0],
        ]
    }
    out = binance_src.normalize(raw)
    assert list(out.columns) == list(BAR_COLUMNS)
    assert len(out) == 1  # today's forming bar excluded
    assert out.iloc[0]["close"] == 50500.0 and out.iloc[0]["source"] == "binance"


def test_coingecko_normalize_maps_midnight_to_prior_session_close() -> None:
    ts = _utc_midnight_ms(1)
    raw = {"BTC": {"prices": [[ts, 50500.0], [ts + 3_600_000, 50999.0]]}}  # 01:00 point dropped
    out = coingecko_src.normalize(raw)
    assert len(out) == 1
    expected_date = (datetime.fromtimestamp(ts / 1000, tz=UTC) - timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    assert out.iloc[0]["date"] == expected_date and out.iloc[0]["close"] == 50500.0


def test_yfinance_normalize_multiindex() -> None:
    idx = pd.to_datetime(["2026-07-01", "2026-07-02"])
    cols = pd.MultiIndex.from_product(
        [["AAPL"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    raw = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 100.4, 5e7], [101.0, 102.0, 100.0, 101.5, 101.4, 6e7]],
        index=idx,
        columns=cols,
    )
    out = yfinance_src.normalize(raw, ["AAPL", "MISSING"])
    assert list(out.columns) == list(BAR_COLUMNS)
    assert len(out) == 2 and set(out["symbol"]) == {"AAPL"}
    assert out.iloc[1]["volume"] == 6e7


def test_alpaca_normalize_flattens_multiindex() -> None:
    raw = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.6],
            "volume": [1e5],
        },
        index=pd.MultiIndex.from_tuples(
            [("AAPL", pd.Timestamp("2026-07-02 04:00", tz="UTC"))], names=["symbol", "timestamp"]
        ),
    )
    out = alpaca_src.normalize(raw)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["symbol"] == "AAPL" and row["date"] == "2026-07-02"
    assert row["adj_close"] == 100.6 and row["source"] == "alpaca_iex"
