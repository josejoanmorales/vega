from conftest import make_ohlc_frame as _ohlc_frame
from conftest import steep_uptrend_then_shock as _steep_uptrend_then_shock
from vega.backtest.market_view import MarketView
from vega.common.doctrine import STOP_ATR_MULT
from vega.signals.crypto_oversold_reversion import CryptoOversoldReversionSignal


def test_fires_on_a_large_shock_while_still_above_sma100() -> None:
    closes = _steep_uptrend_then_shock(drop_total=39.0)
    frame = _ohlc_frame(closes, shocked={100, 101, 102}, symbol="BTC")
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = CryptoOversoldReversionSignal(k=2.0)
    proposals = signal.scan(view, ["BTC"])
    assert len(proposals) == 1
    assert proposals[0].signal_family == "crypto_oversold_reversion_v1"
    assert proposals[0].time_stop_days == 7  # exit override applied
    assert proposals[0].profit_take_half_at_r == 1.5


def test_higher_k_is_a_stricter_threshold() -> None:
    closes = _steep_uptrend_then_shock(drop_total=39.0)
    frame = _ohlc_frame(closes, shocked={100, 101, 102}, symbol="BTC")
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    loose = CryptoOversoldReversionSignal(k=2.0).scan(view, ["BTC"])
    strict = CryptoOversoldReversionSignal(k=100.0).scan(view, ["BTC"])
    assert len(loose) == 1
    assert strict == []


def test_does_not_fire_on_a_shallow_move() -> None:
    closes = _steep_uptrend_then_shock(drop_total=1.5)  # far too small a move
    frame = _ohlc_frame(closes, shocked=set(), symbol="BTC")
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = CryptoOversoldReversionSignal(k=2.0)
    assert signal.scan(view, ["BTC"]) == []


def test_does_not_fire_below_sma100() -> None:
    base = [200.0 - i * 1.0 for i in range(100)]  # declining trend
    closes = base + [base[-1] - 13.0, base[-1] - 26.0, base[-1] - 39.0]  # same shock, no uptrend
    frame = _ohlc_frame(closes, shocked={100, 101, 102}, symbol="BTC")
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = CryptoOversoldReversionSignal(k=2.0)
    assert signal.scan(view, ["BTC"]) == []


def test_insufficient_history_yields_no_proposals() -> None:
    frame = _ohlc_frame([100.0 + i for i in range(50)], shocked=set(), symbol="BTC")
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = CryptoOversoldReversionSignal(k=2.0)
    assert signal.scan(view, ["BTC"]) == []


def test_is_marked_promotable() -> None:
    assert CryptoOversoldReversionSignal(k=2.0).promotable is True


# ---- THE regression this family exists to guard against (WI-136's landmine) --


def test_every_proposal_carries_the_crypto_stop_distance_explicitly() -> None:
    """EntryProposal.stop_atr_mult defaults to STOP_ATR_MULT['equity'] (2.0) --
    asset-class-neutral by omission, not correctness. simulate.py reads
    proposal.stop_atr_mult directly with no asset_class correction, so any
    crypto signal that leaves this unset silently trades at the WRONG (too
    tight) stop distance versus doctrine's own crypto value (2.5) -- the exact
    mixed-space bug class WI-064/066 found repeatedly, and the one WI-136's
    crypto_breakout_v1 had to fix explicitly. Not re-earning it here."""
    closes = _steep_uptrend_then_shock(drop_total=39.0)
    frame = _ohlc_frame(closes, shocked={100, 101, 102}, symbol="BTC")
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    proposals = CryptoOversoldReversionSignal(k=2.0).scan(view, ["BTC"])
    assert len(proposals) == 1
    assert proposals[0].stop_atr_mult == STOP_ATR_MULT["crypto"]
    assert proposals[0].stop_atr_mult != STOP_ATR_MULT["equity"]  # would be silently wrong
