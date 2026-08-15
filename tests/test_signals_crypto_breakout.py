import pandas as pd

from vega.backtest.market_view import MarketView
from vega.common.doctrine import STOP_ATR_MULT
from vega.signals.crypto_breakout import CryptoBreakoutSignal


def _frame(closes: list[float], volumes: list[float], symbol: str = "BTC") -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "adj_close": closes,
            "close": closes,  # crypto: adj_close == close (verified empirically, no dividends)
            "volume": volumes,
        }
    )


def test_fires_on_new_high_with_volume_spike() -> None:
    closes = [100.0] * 64 + [110.0]  # clean breakout on the last session
    volumes = [1_000_000.0] * 64 + [2_000_000.0]  # 2x the flat median
    frame = _frame(closes, volumes)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = CryptoBreakoutSignal(n_sessions=20)
    proposals = signal.scan(view, ["BTC"])
    assert len(proposals) == 1
    assert proposals[0].signal_family == "crypto_breakout_v1"


def test_does_not_fire_on_new_high_without_volume_confirmation() -> None:
    closes = [100.0] * 64 + [110.0]
    volumes = [1_000_000.0] * 65  # no spike
    frame = _frame(closes, volumes)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = CryptoBreakoutSignal(n_sessions=20)
    assert signal.scan(view, ["BTC"]) == []


def test_does_not_fire_on_volume_spike_without_a_new_high() -> None:
    closes = [100.0] * 64 + [99.0]  # NOT a new high (below the flat prior range)
    volumes = [1_000_000.0] * 64 + [2_000_000.0]
    frame = _frame(closes, volumes)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = CryptoBreakoutSignal(n_sessions=20)
    assert signal.scan(view, ["BTC"]) == []


def test_insufficient_history_yields_no_proposals() -> None:
    frame = _frame([100.0] * 20, [1_000_000.0] * 20)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    signal = CryptoBreakoutSignal(n_sessions=20)
    assert signal.scan(view, ["BTC"]) == []


def test_lookback_covers_both_the_breakout_window_and_the_volume_window() -> None:
    s20 = CryptoBreakoutSignal(n_sessions=20)
    s30 = CryptoBreakoutSignal(n_sessions=30)
    assert s20.n_sessions == 20 and s30.n_sessions == 30
    assert s20.lookback >= 60 and s30.lookback >= 60  # never shorter than the volume window


def test_is_marked_promotable() -> None:
    assert CryptoBreakoutSignal(n_sessions=20).promotable is True


# ---- THE regression this family exists to guard against -------------------


def test_every_proposal_carries_the_crypto_stop_distance_explicitly() -> None:
    """EntryProposal.stop_atr_mult defaults to STOP_ATR_MULT['equity'] (2.0) —
    asset-class-neutral by omission, not correctness. simulate.py reads
    proposal.stop_atr_mult directly with no asset_class correction, so any
    crypto signal that leaves this unset silently trades at the WRONG (too
    tight) stop distance versus doctrine's own crypto value (2.5), understating
    simulated risk in exactly the mixed-space bug class WI-064/066 found
    repeatedly. Every emitted proposal must set it explicitly."""
    closes = [100.0] * 64 + [110.0]
    volumes = [1_000_000.0] * 64 + [2_000_000.0]
    frame = _frame(closes, volumes)
    view = MarketView(frame, as_of=frame["date"].iloc[-1])
    proposals = CryptoBreakoutSignal(n_sessions=20).scan(view, ["BTC"])
    assert len(proposals) == 1
    assert proposals[0].stop_atr_mult == STOP_ATR_MULT["crypto"]
    assert proposals[0].stop_atr_mult != STOP_ATR_MULT["equity"]  # would be silently wrong
