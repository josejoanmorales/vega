import pytest

from vega.backtest.costs import apply_cost, cost_bps


def test_buy_raises_price_sell_lowers_it() -> None:
    assert apply_cost(100.0, "buy", 10.0) > 100.0
    assert apply_cost(100.0, "sell", 10.0) < 100.0


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        apply_cost(100.0, "hold", 10.0)


def test_equity_tier_by_median_dollar_volume() -> None:
    assert cost_bps("equity", "AAPL", median_dollar_volume=60_000_000) == 12.0
    assert cost_bps("equity", "SMALLCO", median_dollar_volume=25_000_000) == 20.0


def test_crypto_majors_cheaper_than_others() -> None:
    assert cost_bps("crypto", "BTC") < cost_bps("crypto", "SHIB")


def test_every_tier_is_at_or_above_the_live_paper_haircut() -> None:
    from vega.execution.pnl import SLIPPAGE_BPS

    assert cost_bps("equity", "AAPL", 60_000_000) >= SLIPPAGE_BPS["equity"]
    assert cost_bps("equity", "SMALLCO", 25_000_000) >= SLIPPAGE_BPS["equity"]
    assert cost_bps("crypto", "BTC") >= SLIPPAGE_BPS["crypto"]
    assert cost_bps("crypto", "SHIB") >= SLIPPAGE_BPS["crypto"]
    # "wider for crypto" (acceptance c2)
    assert cost_bps("crypto", "BTC") > cost_bps("equity", "AAPL", 60_000_000)
