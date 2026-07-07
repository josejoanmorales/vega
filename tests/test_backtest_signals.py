import pandas as pd

from vega.backtest.market_view import MarketView
from vega.backtest.signals import SmaCrossSignal


def _series(prices: list[float]) -> pd.DataFrame:
    dates = [f"2026-04-{d:02d}" for d in range(1, len(prices) + 1)]
    return pd.DataFrame({"symbol": ["AAA"] * len(prices), "date": dates, "adj_close": prices})


def test_fires_only_on_the_exact_crossover_day() -> None:
    # fast(3) below slow(6) for a while, then a sharp rise crosses it up on the last day
    prices = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 130]
    signal = SmaCrossSignal(fast=3, slow=6)
    frame = _series(prices)

    view_before = MarketView(frame, as_of="2026-04-11")
    assert signal.scan(view_before, ["AAA"]) == []

    view_cross = MarketView(frame, as_of="2026-04-12")
    proposals = signal.scan(view_cross, ["AAA"])
    assert len(proposals) == 1 and proposals[0].symbol == "AAA"


def test_is_marked_non_promotable() -> None:
    assert SmaCrossSignal().promotable is False


def test_insufficient_history_yields_no_proposals() -> None:
    signal = SmaCrossSignal(fast=3, slow=6)
    frame = _series([100, 101, 102])  # far fewer than slow+1
    view = MarketView(frame, as_of="2026-04-03")
    assert signal.scan(view, ["AAA"]) == []


def test_crypto_asset_class_uses_wider_stop_multiple() -> None:
    prices = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 130]
    frame = _series(prices)
    view = MarketView(frame, as_of="2026-04-12")
    equity_prop = SmaCrossSignal(fast=3, slow=6, asset_class="equity").scan(view, ["AAA"])[0]
    crypto_prop = SmaCrossSignal(fast=3, slow=6, asset_class="crypto").scan(view, ["AAA"])[0]
    assert crypto_prop.stop_atr_mult > equity_prop.stop_atr_mult
