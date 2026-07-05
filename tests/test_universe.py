from pathlib import Path

import pytest

from vega.data.universe import load_universe, symbols

VALID = """\
# universe-v1 test fixture
symbol,asset_class,name,coingecko_id,binance_symbol
AAPL,equity,Apple Inc.,,
SPY,etf,SPDR S&P 500,,
BTC,crypto,Bitcoin,bitcoin,BTCUSDT
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
