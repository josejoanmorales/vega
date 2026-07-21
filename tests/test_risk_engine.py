import pandas as pd
import pytest

from conftest import flat_history_n
from vega.regime.regime import RegimeState
from vega.risk.engine import open_position_heat, propose, to_recommendation
from vega.risk.gates import EarningsFact
from vega.risk.heat import CAPS_R, OpenPositionHeat
from vega.risk.types import Rejection, SizedProposal

NO_EARNINGS = EarningsFact("none")


def _frame(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(flat_history_n("AAPL", n, month="2026-06"))


def _regime(composite: str = "risk_on") -> RegimeState:
    return RegimeState(
        as_of="2026-06-20",
        trend="risk_on",
        vix=15.0,
        vix_band="calm",
        breadth_pct=60.0,
        crypto_fg=50,
        composite=composite,
    )


def _propose(**overrides: object) -> SizedProposal | Rejection:
    kwargs: dict[str, object] = {
        "symbol": "AAPL",
        "asset_class": "equity",
        "entry_ref_price": 100.0,
        "frame": _frame(),
        "as_of": "2026-06-20",
        "equity": 100_000.0,
        "regime": _regime(),
        "open_positions": [],
        "earnings": NO_EARNINGS,
        "invalidation": "fixture",
    }
    kwargs.update(overrides)
    return propose(**kwargs)  # type: ignore[arg-type]


def test_propose_returns_a_sized_proposal_for_a_clean_case() -> None:
    result = _propose()
    assert isinstance(result, SizedProposal)
    assert result.stop_price < result.entry_ref_price
    assert result.qty > 0
    assert result.cluster == "us_equity_beta"
    # heat is reported in R MULTIPLES, directly comparable to CAPS_R
    r_unit = 0.0075 * 100_000.0
    assert result.heat_after_r["total"] == pytest.approx(result.initial_r_dollars / r_unit)
    assert result.heat_after_r["total"] < CAPS_R["total"]


def test_propose_rejects_when_regime_is_risk_off() -> None:
    result = _propose(regime=_regime("risk_off"))
    assert isinstance(result, Rejection) and result.reason == "regime_risk_off"


def test_propose_fails_closed_on_unavailable_earnings() -> None:
    result = _propose(earnings=EarningsFact("unavailable"))
    assert isinstance(result, Rejection) and result.reason == "earnings_unknown"


def test_propose_rejects_on_insufficient_history() -> None:
    result = _propose(frame=_frame(n=5), as_of="2026-06-05")
    assert isinstance(result, Rejection) and result.reason == "insufficient_history"


def test_propose_rejects_when_heat_cap_would_be_breached() -> None:
    equity = 100_000.0
    r_dollars = 0.0075 * equity
    saturating_position = OpenPositionHeat(
        symbol="OTHER",
        asset_class="equity",
        qty=1.0,
        entry_price=CAPS_R["us_equity_beta"] * r_dollars + 1.0,
        current_stop_price=0.0,
    )
    result = _propose(open_positions=[saturating_position])
    assert isinstance(result, Rejection) and result.reason.startswith("heat_cap:")


def test_crypto_with_spyless_frame_raises_loudly() -> None:
    # a source-filtered frame without SPY is a caller bug, not an unmeasurable fact
    crypto_frame = pd.DataFrame(flat_history_n("BTC", 20, month="2026-06"))
    with pytest.raises(ValueError, match="no SPY rows"):
        _propose(symbol="BTC", asset_class="crypto", frame=crypto_frame)


def test_crypto_correlated_to_spy_contaminates_and_heats_equity_beta() -> None:
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    rows = []
    for i, d in enumerate(dates):
        spy_price = 100.0 + i * 0.5 + (5.0 if i % 7 == 0 else 0.0)  # co-moving w/ wiggle
        btc_price = spy_price * 500.0
        rows.append(
            {
                "symbol": "SPY",
                "date": d,
                "high": spy_price + 1,
                "low": spy_price - 1,
                "close": spy_price,
                "adj_close": spy_price,
            }
        )
        rows.append(
            {
                "symbol": "BTC",
                "date": d,
                "high": btc_price + 500,
                "low": btc_price - 500,
                "close": btc_price,
                "adj_close": btc_price,
            }
        )
    frame = pd.DataFrame(rows)
    result = _propose(
        symbol="BTC",
        asset_class="crypto",
        entry_ref_price=float(frame["close"].iloc[-1]),
        frame=frame,
        as_of=str(dates[-1]),
    )
    assert isinstance(result, SizedProposal)
    assert result.contaminates_equity_beta is True
    # 50% of the crypto R shows up in the equity-beta bucket
    assert result.heat_after_r["us_equity_beta"] == pytest.approx(
        result.heat_after_r["crypto_beta"] / 2, rel=0.01
    )


def test_batch_heat_accumulates_via_open_position_heat_helper() -> None:
    first = _propose()
    assert isinstance(first, SizedProposal)
    second = _propose(
        symbol="MSFT",
        frame=pd.DataFrame(flat_history_n("MSFT", 20, month="2026-06")),
        open_positions=[open_position_heat(first)],
    )
    assert isinstance(second, SizedProposal)
    assert second.heat_after_r["total"] == pytest.approx(first.heat_after_r["total"] * 2, rel=0.01)


def test_proposal_round_trips_into_a_valid_ledger_recommendation() -> None:
    result = _propose(invalidation="close below the 20-session low")
    assert isinstance(result, SizedProposal)
    rec = to_recommendation(
        result,
        thesis="fixture proposal",
        confidence=0.6,
        signal_attribution=("test_signal",),
        as_of="2026-06-20",
    )
    assert rec.qty == result.qty
    assert rec.horizon_days == result.time_stop_sessions
    assert rec.as_of == "2026-06-20"  # decision session — expiry semantics downstream
    assert rec.exit_params is not None
    assert rec.exit_params["time_stop_sessions"] == result.time_stop_sessions  # canonical
    assert rec.exit_params["stop_atr_mult"] == 2.0
    # the ledger date string is a DERIVED display value: sessions * 7/5 calendar days
    assert rec.time_stop_date == "2026-07-11"  # 2026-06-20 + ceil(15 * 1.4) = 21 days


def test_all_four_exit_overrides_flow_into_the_spec() -> None:
    # WI-067 review: propose() must honor the SAME four per-family exit params
    # the backtester honors — a subset silently re-created live/backtest drift.
    result = _propose(
        time_stop_sessions=7,
        profit_take_half_at_r=1.5,
        stop_atr_mult=2.5,
        profit_trail_atr_mult=3.0,
    )
    assert isinstance(result, SizedProposal)
    atr = result.exit_params["atr_at_proposal"]
    assert result.stop_price == pytest.approx(100.0 - 2.5 * atr)
    assert result.exit_params["stop_atr_mult"] == 2.5
    assert result.exit_params["trail_atr_mult"] == 3.0
    assert result.exit_params["take_half_at_r"] == 1.5
    assert result.exit_params["time_stop_sessions"] == 7
    assert "+1.5R" in result.profit_rule_text and "3xATR" in result.profit_rule_text


def test_out_of_doctrine_exit_overrides_are_rejected() -> None:
    # A typo'd exit spec must never become a binding ledger contract.
    for override in (
        {"time_stop_sessions": 45},
        {"profit_take_half_at_r": 0.15},
        {"stop_atr_mult": 0.5},
        {"profit_trail_atr_mult": 9.0},
    ):
        result = _propose(**override)
        assert isinstance(result, Rejection)
        assert result.reason == "exit_spec_out_of_doctrine"
