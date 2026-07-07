import pandas as pd
import pytest

from vega.regime.regime import RegimeState
from vega.risk import gates
from vega.risk.engine import propose, to_recommendation
from vega.risk.heat import CAPS_R, OpenPositionHeat
from vega.risk.types import Rejection, SizedProposal


def _flat_history(symbol: str, n: int, h=101.0, low=99.0, c=100.0) -> list[dict]:
    dates = [f"2026-06-{d:02d}" for d in range(1, n + 1)]
    return [
        {"symbol": symbol, "date": d, "high": h, "low": low, "close": c, "adj_close": c}
        for d in dates
    ]


def _frame(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(_flat_history("AAPL", n))


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


@pytest.fixture(autouse=True)
def _no_network_earnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gates, "next_earnings", lambda symbol: None)


def test_propose_returns_a_sized_proposal_for_a_clean_case() -> None:
    result = propose(
        symbol="AAPL",
        asset_class="equity",
        entry_ref_price=100.0,
        frame=_frame(),
        as_of="2026-06-20",
        equity=100_000.0,
        regime=_regime(),
        open_positions=[],
        invalidation="fixture",
    )
    assert isinstance(result, SizedProposal)
    assert result.stop_price < result.entry_ref_price
    assert result.qty > 0
    assert result.cluster == "us_equity_beta"
    assert result.heat_after["total"] == pytest.approx(result.initial_r_dollars)


def test_propose_rejects_when_regime_is_risk_off() -> None:
    result = propose(
        symbol="AAPL",
        asset_class="equity",
        entry_ref_price=100.0,
        frame=_frame(),
        as_of="2026-06-20",
        equity=100_000.0,
        regime=_regime("risk_off"),
        open_positions=[],
        invalidation="fixture",
    )
    assert isinstance(result, Rejection) and result.reason == "regime_risk_off"


def test_propose_rejects_on_insufficient_history() -> None:
    result = propose(
        symbol="AAPL",
        asset_class="equity",
        entry_ref_price=100.0,
        frame=_frame(n=5),
        as_of="2026-06-05",
        equity=100_000.0,
        regime=_regime(),
        open_positions=[],
        invalidation="fixture",
    )
    assert isinstance(result, Rejection) and result.reason == "insufficient_history"


def test_propose_rejects_when_heat_cap_would_be_breached() -> None:
    # crowd the us_equity_beta cluster right up to its 4R cap
    equity = 100_000.0
    r_dollars = 0.0075 * equity
    saturating_position = OpenPositionHeat(
        symbol="OTHER",
        asset_class="equity",
        qty=1.0,
        entry_price=CAPS_R["us_equity_beta"] * r_dollars + 1.0,
        current_stop_price=0.0,
    )
    result = propose(
        symbol="AAPL",
        asset_class="equity",
        entry_ref_price=100.0,
        frame=_frame(),
        as_of="2026-06-20",
        equity=equity,
        regime=_regime(),
        open_positions=[saturating_position],
        invalidation="fixture",
    )
    assert isinstance(result, Rejection) and result.reason.startswith("heat_cap:")


def test_proposal_round_trips_into_a_valid_ledger_recommendation() -> None:
    result = propose(
        symbol="AAPL",
        asset_class="equity",
        entry_ref_price=100.0,
        frame=_frame(),
        as_of="2026-06-20",
        equity=100_000.0,
        regime=_regime(),
        open_positions=[],
        invalidation="close below the 20-session low",
    )
    assert isinstance(result, SizedProposal)
    rec = to_recommendation(
        result,
        thesis="fixture proposal",
        confidence=0.6,
        signal_attribution=("test_signal",),
        as_of="2026-06-20",
    )
    assert rec.qty == result.qty
    assert rec.horizon_days == result.time_stop_days
    assert rec.exit_params is not None and rec.exit_params["stop_atr_mult"] == 2.0
