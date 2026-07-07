from vega.risk.heat import (
    CAPS_R,
    CAUTION_TOTAL_CAP_R,
    OpenPositionHeat,
    cluster_heat,
    first_breach,
    position_r_dollars,
)


def _pos(
    symbol: str, asset_class: str, qty: float, entry: float, stop: float, contaminates: bool = False
) -> OpenPositionHeat:
    return OpenPositionHeat(symbol, asset_class, qty, entry, stop, contaminates)


def test_position_heat_floored_at_zero_when_stop_trailed_past_breakeven() -> None:
    pos = _pos("AAPL", "equity", qty=10.0, entry=100.0, stop=105.0)  # stop above entry
    assert position_r_dollars(pos) == 0.0


def test_position_heat_is_qty_times_open_risk() -> None:
    pos = _pos("AAPL", "equity", qty=10.0, entry=100.0, stop=96.0)
    assert position_r_dollars(pos) == 40.0


def test_cluster_heat_buckets_correctly() -> None:
    positions = [
        _pos("AAPL", "equity", 10.0, 100.0, 96.0),  # 40 -> us_equity_beta
        _pos("TLT", "etf", 5.0, 90.0, 88.0),  # 10 -> rates
        _pos("GLD", "etf", 2.0, 200.0, 190.0),  # 20 -> commodities
        _pos("BTC", "crypto", 0.1, 60000.0, 58000.0),  # 200 -> crypto_beta
    ]
    heat = cluster_heat(positions)
    assert heat["us_equity_beta"] == 40.0
    assert heat["rates"] == 10.0
    assert heat["commodities"] == 20.0
    assert heat["crypto_beta"] == 200.0
    assert heat["total"] == 270.0


def test_correlated_crypto_contaminates_equity_beta() -> None:
    positions = [
        _pos("AAPL", "equity", 10.0, 100.0, 96.0),  # 40 -> us_equity_beta
        _pos(
            "BTC", "crypto", 0.1, 60000.0, 58000.0, contaminates=True
        ),  # 200 -> crypto_beta, 100 -> equity too
    ]
    heat = cluster_heat(positions)
    assert heat["crypto_beta"] == 200.0
    assert heat["us_equity_beta"] == 40.0 + 100.0  # 50% of the crypto R counted twice
    assert heat["total"] == 240.0  # total is NOT double-counted, only the display bucket is


def test_first_breach_none_under_caps() -> None:
    heat = {
        "total": 1.0,
        "us_equity_beta": 1.0,
        "crypto_beta": 0.0,
        "rates": 0.0,
        "commodities": 0.0,
    }
    assert first_breach(heat, r_dollar_unit=750.0, regime_caution=False) is None


def test_first_breach_detects_total_cap() -> None:
    r = 750.0
    heat = {
        "total": (CAPS_R["total"] + 0.1) * r,
        "us_equity_beta": 0.0,
        "crypto_beta": 0.0,
        "rates": 0.0,
        "commodities": 0.0,
    }
    assert first_breach(heat, r, regime_caution=False) == "total"


def test_first_breach_detects_cluster_cap_before_total() -> None:
    r = 750.0
    heat = {
        "total": 3.0 * r,  # under the 6R total cap
        "crypto_beta": (CAPS_R["crypto_beta"] + 0.1) * r,  # but over the 2.5R crypto cap
        "us_equity_beta": 0.0,
        "rates": 0.0,
        "commodities": 0.0,
    }
    assert first_breach(heat, r, regime_caution=False) == "crypto_beta"


def test_caution_regime_halves_the_total_cap() -> None:
    r = 750.0
    heat = {
        "total": (CAUTION_TOTAL_CAP_R + 0.1) * r,
        "us_equity_beta": 0.0,
        "crypto_beta": 0.0,
        "rates": 0.0,
        "commodities": 0.0,
    }
    assert first_breach(heat, r, regime_caution=False) is None  # under the normal 6R cap
    assert first_breach(heat, r, regime_caution=True) == "total"  # over the caution 3R cap
