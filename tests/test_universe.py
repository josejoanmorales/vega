from pathlib import Path

import pytest

from vega.data.universe import load_universe, symbols, tradable_symbols

VALID = """\
# universe-v1 test fixture
symbol,asset_class,name,coingecko_id,binance_symbol
AAPL,equity,Apple Inc.,,
SPY,etf,SPDR S&P 500,,
BTC,crypto,Bitcoin,bitcoin,BTCUSDT
"""

# v2-shaped fixture: explicit cluster column (WI-084 item 8)
VALID_V2 = """\
# universe-v2 test fixture
symbol,asset_class,name,coingecko_id,binance_symbol,cluster
AAPL,equity,Apple Inc.,,,us_equity_beta
TLT,etf,iShares 20+ Year Treasury,,,rates
GLD,etf,SPDR Gold Shares,,,commodities
BTC,crypto,Bitcoin,bitcoin,BTCUSDT,crypto_beta
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "universe.csv"
    p.write_text(body)
    return p


def test_loads_and_filters_by_class(tmp_path: Path) -> None:
    entries = load_universe(_write(tmp_path, VALID))
    assert len(entries) == 3
    assert symbols(entries, "equity", "etf") == ["AAPL", "SPY"]
    assert entries[2].binance_symbol == "BTCUSDT"


def test_rejects_unknown_asset_class(tmp_path: Path) -> None:
    bad = VALID.replace("etf", "fund")
    with pytest.raises(ValueError, match="unknown asset_class"):
        load_universe(_write(tmp_path, bad))


def test_rejects_crypto_without_mappings(tmp_path: Path) -> None:
    bad = VALID.replace("bitcoin,BTCUSDT", ",")
    with pytest.raises(ValueError, match="missing source mappings"):
        load_universe(_write(tmp_path, bad))


def test_missing_cluster_column_defaults_by_asset_class(tmp_path: Path) -> None:
    """Pre-v2 artifacts and hand-written fixtures (no `cluster` column at all)
    must still load, falling back the same way classify() used to hardcode."""
    entries = load_universe(_write(tmp_path, VALID))
    by_symbol = {e.symbol: e for e in entries}
    assert by_symbol["AAPL"].cluster == "us_equity_beta"
    assert by_symbol["SPY"].cluster == "us_equity_beta"
    assert by_symbol["BTC"].cluster == "crypto_beta"


def test_explicit_cluster_column_is_parsed(tmp_path: Path) -> None:
    entries = load_universe(_write(tmp_path, VALID_V2))
    by_symbol = {e.symbol: e for e in entries}
    assert by_symbol["TLT"].cluster == "rates"
    assert by_symbol["GLD"].cluster == "commodities"
    assert by_symbol["AAPL"].cluster == "us_equity_beta"


def test_rejects_unknown_cluster(tmp_path: Path) -> None:
    bad = VALID_V2.replace("us_equity_beta", "not_a_real_cluster")
    with pytest.raises(ValueError, match="unknown cluster"):
        load_universe(_write(tmp_path, bad))


def test_tradable_symbols_excludes_the_given_set(tmp_path: Path) -> None:
    """WI-084 item 7: SPY is both a regular universe ETF and the backtest
    benchmark — tradable_symbols() is how callers building a signal-candidate
    list keep the benchmark out of it without removing SPY from the artifact
    (ingest/regime still need SPY's bars via plain symbols())."""
    entries = load_universe(_write(tmp_path, VALID))
    assert tradable_symbols(entries, "equity", "etf") == ["AAPL", "SPY"]
    assert tradable_symbols(entries, "equity", "etf", exclude={"SPY"}) == ["AAPL"]
